"""Detect distribution drift between live traffic and the training reference.

Drift is scored with the **Population Stability Index**:

.. math::

    PSI = \\sum_i (live_i - ref_i) \\cdot \\ln(live_i / ref_i)

summed over the bins of a feature.  It is symmetric, bounded below at zero when
the two distributions match, and grows as mass moves between bins.  PSI is used
here in preference to a KS test because it needs no p-value interpretation and
its thresholds are directly actionable - a number a on-call engineer can read
at 3am without deciding on a significance level.

Nothing in this module retrains or blocks anything; it reports.  Acting on a
drift signal is a judgement call about the business, not about statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.monitoring.profile import (
    DISCRETE_FEATURES,
    category_proportions,
    proportions_in_bins,
)

# Added to every proportion before taking a logarithm.  Without it a category
# absent from either side sends PSI to infinity, which would let one unseen
# plan name swamp the whole report.
EPSILON: float = 1e-6

STABLE = "stable"
MODERATE = "moderate"
SIGNIFICANT = "significant"


def population_stability_index(
    reference: list[float] | np.ndarray, live: list[float] | np.ndarray
) -> float:
    """Compute PSI between two binned distributions.

    Both inputs are proportions over the *same* bins and are expected to sum to
    roughly one.  Returns 0.0 for identical distributions.
    """
    ref = np.clip(np.asarray(reference, dtype=float), EPSILON, None)
    obs = np.clip(np.asarray(live, dtype=float), EPSILON, None)
    if ref.size == 0 or ref.size != obs.size:
        return 0.0
    return float(np.sum((obs - ref) * np.log(obs / ref)))


def classify_drift(psi: float) -> str:
    """Map a PSI value onto ``stable`` / ``moderate`` / ``significant``."""
    if psi >= settings.PSI_SIGNIFICANT:
        return SIGNIFICANT
    if psi >= settings.PSI_MODERATE:
        return MODERATE
    return STABLE


def _aligned_discrete_proportions(
    reference: dict[str, float], live: dict[str, float]
) -> tuple[list[float], list[float]]:
    """Put two category->share maps onto a shared, ordered category list.

    Categories present on only one side count as zero on the other, which is
    what makes an entirely new plan tier register as drift rather than being
    silently skipped.
    """
    categories = sorted(set(reference) | set(live))
    return (
        [reference.get(category, 0.0) for category in categories],
        [live.get(category, 0.0) for category in categories],
    )


def _feature_report(name: str, psi: float, detail: dict[str, Any]) -> dict[str, Any]:
    """Assemble one feature's entry in the drift report."""
    return {"feature": name, "psi": round(psi, 4), "verdict": classify_drift(psi), **detail}


def detect_drift(
    frame: pd.DataFrame,
    profile: dict[str, Any],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score a batch of live subscribers against the training reference.

    Args:
        frame: Raw live rows, same columns as the training input.
        profile: A reference profile from :mod:`src.monitoring.profile`.
        probabilities: Optional model scores for those rows, enabling
            prediction drift on top of input drift.

    Returns:
        A report with per-feature PSI, an overall verdict, and the list of
        features that moved.
    """
    features: list[dict[str, Any]] = []

    for column, reference in profile.get("numeric", {}).items():
        if column not in frame.columns:
            continue
        edges = reference["edges"]
        live_proportions = proportions_in_bins(frame[column], edges)
        psi = population_stability_index(reference["proportions"], live_proportions)
        live_values = pd.to_numeric(frame[column], errors="coerce")
        features.append(
            _feature_report(
                column,
                psi,
                {
                    "kind": "numeric",
                    "reference_mean": reference.get("mean"),
                    "live_mean": round(float(live_values.mean()), 4),
                },
            )
        )

    for column in DISCRETE_FEATURES:
        reference = profile.get("discrete", {}).get(column)
        if reference is None or column not in frame.columns:
            continue
        live_shares = category_proportions(frame[column])
        ref_aligned, live_aligned = _aligned_discrete_proportions(
            reference["proportions"], live_shares
        )
        psi = population_stability_index(ref_aligned, live_aligned)
        unseen = sorted(set(live_shares) - set(reference["proportions"]))
        features.append(
            _feature_report(
                column,
                psi,
                {
                    "kind": "discrete",
                    "unseen_categories": unseen,
                },
            )
        )

    features.sort(key=lambda item: item["psi"], reverse=True)

    prediction: dict[str, Any] | None = None
    reference_prediction = profile.get("prediction")
    if probabilities is not None and len(probabilities) > 0 and reference_prediction:
        scores = pd.Series(np.asarray(probabilities, dtype=float))
        live_proportions = proportions_in_bins(scores, reference_prediction["edges"])
        psi = population_stability_index(reference_prediction["proportions"], live_proportions)
        prediction = _feature_report(
            "dropout_probability",
            psi,
            {
                "kind": "prediction",
                "reference_mean": reference_prediction.get("mean"),
                "live_mean": round(float(scores.mean()), 4),
            },
        )

    scored = features + ([prediction] if prediction else [])
    verdicts = [item["verdict"] for item in scored]
    if SIGNIFICANT in verdicts:
        overall = SIGNIFICANT
    elif MODERATE in verdicts:
        overall = MODERATE
    else:
        overall = STABLE

    n_samples = int(len(frame))
    return {
        "n_samples": n_samples,
        # Small batches produce noisy PSI, so the caller is told not to act on
        # this report rather than being left to discover it the hard way.
        "sufficient_sample": n_samples >= settings.DRIFT_MIN_SAMPLES,
        "min_samples": settings.DRIFT_MIN_SAMPLES,
        "overall_verdict": overall,
        "drifted_features": [
            item["feature"] for item in scored if item["verdict"] != STABLE
        ],
        "features": features,
        "prediction": prediction,
        "reference_created_at": profile.get("created_at"),
        "reference_rows": profile.get("n_reference_rows"),
        "thresholds": {
            "moderate": settings.PSI_MODERATE,
            "significant": settings.PSI_SIGNIFICANT,
        },
    }
