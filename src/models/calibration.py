"""Probability calibration, and how to tell whether it worked.

A gradient-boosted classifier is good at *ranking* subscribers and bad at
saying how likely any of them is to churn. It will happily assign 0.9 to a
group that churns 60% of the time. For ranking that is harmless - the ordering
is still right - and the whole project has so far only relied on the ordering:
ROC-AUC, PR-AUC and a tuned threshold all care about order, not magnitude.

It stops being harmless the moment a probability is multiplied by money. "Send
a £20 retention offer when expected loss exceeds £20" requires 0.9 to mean 90%,
and :mod:`src.models.costs` does exactly that arithmetic. So calibration is not
a nice-to-have here - it is the precondition for the cost-based threshold being
anything other than a plausible-looking number.

Measuring it
------------

Two numbers, because they say different things:

**Brier score** is the mean squared error of the probabilities. It is a
*proper* scoring rule - it rewards being both well-ordered and well-scaled - so
it can get worse even when AUC is unchanged.

**Expected Calibration Error** bins predictions and compares each bin's mean
predicted probability against its observed frequency. It answers the specific
question "when this model says 0.7, does 70% of that group actually churn?",
which Brier bundles together with sharpness.

Both are needed. A model that always predicts the base rate is perfectly
calibrated and completely useless; its ECE is near zero and its AUC is 0.5.
Calibration is only meaningful alongside a discrimination metric.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Isotonic by default: it is non-parametric, so it can correct the S-shaped
# distortion boosting produces without assuming the shape in advance. It needs
# a few hundred calibration rows to avoid overfitting, which the validation
# split comfortably has; below that, sigmoid is the safer choice.
DEFAULT_METHOD = "isotonic"


def calibrate_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    method: str = DEFAULT_METHOD,
) -> CalibratedClassifierCV:
    """Wrap an already-fitted pipeline in a calibration layer.

    The pipeline is *not* refitted: it keeps the model that was selected and
    gated, and learns only the mapping from its scores to probabilities.
    Calibrating on the training split would be circular - the model already
    fits that data well by construction - so this must be given a split the
    pipeline has not seen.

    scikit-learn 1.6 removed ``cv="prefit"`` in favour of wrapping the fitted
    estimator in ``FrozenEstimator``. Both spellings are supported here so the
    project keeps working across the version range ``requirements.txt`` allows.

    Args:
        pipeline: A fitted pipeline.
        features: Calibration features, unseen during training.
        target: Their labels.
        method: ``"isotonic"`` or ``"sigmoid"``.

    Returns:
        A fitted calibrated classifier exposing the same predict/predict_proba
        interface, so everything downstream keeps working unchanged.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"Unknown calibration method {method!r}; use isotonic or sigmoid.")

    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
    except ImportError:  # pragma: no cover - scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(pipeline, method=method, cv="prefit")

    calibrated.fit(features, target)
    logger.info("Calibrated the pipeline with %s on %d rows", method, len(features))
    return calibrated


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Bins are equal-width across [0, 1] rather than equal-count, because the
    question is about the probability scale itself: "of everything scored near
    0.7, how much actually churned?" Quantile bins would let a densely
    populated region dominate and hide a badly calibrated tail.

    Empty bins contribute nothing, and each bin is weighted by how much of the
    population it holds - so a wildly wrong bin containing three rows cannot
    swamp the score.
    """
    truth = np.asarray(y_true, dtype=float)
    proba = np.asarray(y_proba, dtype=float)
    if truth.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed so a prediction of exactly 1.0 lands in the last bin rather
    # than falling outside every one of them.
    indices = np.clip(np.digitize(proba, edges[1:-1], right=True), 0, n_bins - 1)

    error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / truth.size) * abs(proba[mask].mean() - truth[mask].mean())
    return float(error)


def reliability_table(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Per-bin predicted vs observed rates - the reliability diagram as data.

    Returned as a frame rather than a plot so it can be asserted on in tests
    and logged as an artifact, instead of only being eyeballed.
    """
    truth = np.asarray(y_true, dtype=float)
    proba = np.asarray(y_proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(proba, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": round(float(edges[bin_index]), 3),
                "bin_upper": round(float(edges[bin_index + 1]), 3),
                "n": count,
                "mean_predicted": round(float(proba[mask].mean()), 4) if count else None,
                "observed_rate": round(float(truth[mask].mean()), 4) if count else None,
                "gap": (
                    round(float(proba[mask].mean() - truth[mask].mean()), 4) if count else None
                ),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Summarise how trustworthy these probabilities are as probabilities."""
    truth = np.asarray(y_true, dtype=float)
    proba = np.asarray(y_proba, dtype=float)

    return {
        "brier_score": round(float(brier_score_loss(truth, proba)), 5),
        "expected_calibration_error": round(
            expected_calibration_error(truth, proba, n_bins), 5
        ),
        "mean_predicted": round(float(proba.mean()), 4),
        "observed_rate": round(float(truth.mean()), 4),
        # If these two disagree the model is systematically over- or
        # under-predicting, which no amount of good AUC will fix.
        "mean_bias": round(float(proba.mean() - truth.mean()), 4),
        "n_samples": int(truth.size),
    }


def compare_calibration(
    y_true: np.ndarray | pd.Series,
    raw_proba: np.ndarray,
    calibrated_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Did calibration actually help? Report both, and say so explicitly.

    Calibration is not guaranteed to improve anything - isotonic regression can
    overfit a small calibration split and make things worse. Applying it
    without checking is how a project ends up with worse probabilities and a
    line in the README claiming otherwise.
    """
    before = calibration_metrics(y_true, raw_proba, n_bins)
    after = calibration_metrics(y_true, calibrated_proba, n_bins)

    return {
        "before": before,
        "after": after,
        "brier_improvement": round(before["brier_score"] - after["brier_score"], 5),
        "ece_improvement": round(
            before["expected_calibration_error"] - after["expected_calibration_error"], 5
        ),
        "improved": after["expected_calibration_error"] < before["expected_calibration_error"],
    }
