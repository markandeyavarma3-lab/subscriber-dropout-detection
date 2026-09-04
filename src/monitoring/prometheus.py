"""Prometheus exposition for serving and pipeline metrics.

The in-process tracker in :mod:`src.monitoring.tracker` answers "what is this
replica doing right now" and forgets everything when the process restarts.
That is fine for a health check and useless for the question that actually
matters after a deploy: *has this model's behaviour changed over the last three
weeks?*  Prometheus scrapes this endpoint and keeps the history, so the same
numbers become durable, queryable, and alertable across replicas and restarts.

Two families of metric are exposed together, from one scrape target:

**Serving metrics** are incremented live, in
:func:`src.api.service.predict_batch` - the single funnel every prediction
passes through, so a new endpoint cannot quietly escape instrumentation.

**Pipeline metrics** come from the last scheduled run's report on disk.  A
batch job has no HTTP endpoint of its own to scrape, and the usual answer is a
Pushgateway - a whole extra service to run, and one that famously keeps serving
stale values after a job stops existing.  Reading the report file at scrape
time avoids both problems: the numbers are exactly as fresh as the last run,
and there is one fewer moving part.

.. note::
   This lives at ``/metrics/prometheus`` rather than ``/metrics``, which is the
   usual convention.  ``/metrics`` already served a documented JSON payload
   consumed by tests and humans before Prometheus arrived, and silently
   changing its content type would break that contract for the sake of a
   default that Prometheus lets you configure in one line
   (``metrics_path: /metrics/prometheus``).
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.config import settings

logger = logging.getLogger(__name__)

# A dedicated registry rather than the global default: it keeps this project's
# metrics isolated from anything a library might register on import, and lets
# tests build a clean registry without unregistering global collectors.
REGISTRY = CollectorRegistry()

# Numeric encoding of the PSI verdict, so it can be alerted on with a simple
# comparison. Prometheus has no string values - a label would let you match a
# verdict but not express "worse than moderate" in a threshold.
VERDICT_CODES: dict[str, int] = {"stable": 0, "moderate": 1, "significant": 2}


# --------------------------------------------------------------------------- #
# Serving metrics - incremented live
# --------------------------------------------------------------------------- #

PREDICTIONS = Counter(
    "subscriber_predictions_total",
    "Predictions served, by risk band and predicted label.",
    ["risk_level", "predicted_label"],
    registry=REGISTRY,
)

PREDICTION_PROBABILITY = Histogram(
    "subscriber_prediction_probability",
    "Distribution of predicted dropout probabilities.",
    # Buckets are denser at the low end because that is where most of an
    # imbalanced population sits; uniform 0.1 buckets would put ~80% of traffic
    # in the first two and show nothing useful.
    buckets=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0),
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "subscriber_model_loaded",
    "1 when a model artifact is loaded and servable, 0 otherwise.",
    registry=REGISTRY,
)

DECISION_THRESHOLD = Gauge(
    "subscriber_decision_threshold",
    "Probability at or above which a subscriber is flagged.",
    registry=REGISTRY,
)

SERVED_FROM = Gauge(
    "subscriber_model_served_from",
    "1 for the source the live model was loaded from.",
    ["source"],
    registry=REGISTRY,
)

FLAGGED_RATE = Gauge(
    "subscriber_flagged_rate",
    "Share of recent predictions at or above the decision threshold.",
    registry=REGISTRY,
)

PROBABILITY_MEAN_SHIFT = Gauge(
    "subscriber_probability_mean_shift",
    "Live mean predicted probability minus the training-time mean.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
# Pipeline metrics - refreshed from the last run's report at scrape time
# --------------------------------------------------------------------------- #

PIPELINE_LAST_RUN = Gauge(
    "subscriber_pipeline_last_run_timestamp_seconds",
    "Unix timestamp of the last completed retraining pipeline run.",
    registry=REGISTRY,
)

PIPELINE_NEEDS_ATTENTION = Gauge(
    "subscriber_pipeline_needs_attention",
    "1 when the last pipeline run flagged something for a human.",
    registry=REGISTRY,
)

PIPELINE_PROMOTED = Gauge(
    "subscriber_pipeline_promoted",
    "1 when the last run's challenger was promoted, 0 when it was rejected.",
    registry=REGISTRY,
)

DRIFT_VERDICT = Gauge(
    "subscriber_drift_verdict",
    "Overall drift verdict: 0=stable, 1=moderate, 2=significant.",
    registry=REGISTRY,
)

DRIFT_PSI = Gauge(
    "subscriber_drift_psi",
    "Population Stability Index per feature, from the last drift check.",
    ["feature"],
    registry=REGISTRY,
)

DRIFT_SAMPLE_SIZE = Gauge(
    "subscriber_drift_sample_size",
    "Rows compared in the last drift check.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
# Shadow scoring - the challenger running on live traffic, never served
# --------------------------------------------------------------------------- #

SHADOW_COMPARISONS = Counter(
    "subscriber_shadow_comparisons_total",
    "Requests scored by both champion and challenger, by whether they agreed.",
    ["agreed"],
    registry=REGISTRY,
)

SHADOW_DIVERGENCE = Histogram(
    "subscriber_shadow_divergence",
    "Absolute gap between champion and challenger probabilities.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
    registry=REGISTRY,
)

SHADOW_LATENCY = Histogram(
    "subscriber_shadow_latency_seconds",
    "Time spent scoring a batch with the challenger. This is added to request "
    "latency, since shadow scoring runs inline.",
    registry=REGISTRY,
)

SHADOW_ERRORS = Counter(
    "subscriber_shadow_errors_total",
    "Batches the challenger failed to score. Never affects the response, but a "
    "rising count is exactly what should block a promotion.",
    registry=REGISTRY,
)

SHADOW_ACTIVE = Gauge(
    "subscriber_shadow_active",
    "1 when a distinct challenger is loaded and shadow-scoring live traffic.",
    registry=REGISTRY,
)

SHADOW_AGREEMENT_RATE = Gauge(
    "subscriber_shadow_agreement_rate",
    "Share of shadowed requests where both models produced the same label.",
    registry=REGISTRY,
)

SHADOW_FLAGGED_RATE_DELTA = Gauge(
    "subscriber_shadow_flagged_rate_delta",
    "Challenger flagged rate minus champion flagged rate: how much the outreach "
    "list would change in size on promotion.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
# Decision quality - calibration, cost and fairness from the last training run
# --------------------------------------------------------------------------- #
#
# These are computed at training time and written to metrics.json. Without
# exposing them here they would be invisible to the dashboards and alerts that
# everything else in this project is monitored by - a fairness audit nobody
# ever looks at is not a fairness audit.

MODEL_CALIBRATION_ECE = Gauge(
    "subscriber_model_calibration_ece",
    "Expected Calibration Error of the trained model on its test split.",
    registry=REGISTRY,
)

MODEL_FAIRNESS_PASSES = Gauge(
    "subscriber_model_fairness_passes",
    "1 when the last training run's fairness audit found no disparity.",
    registry=REGISTRY,
)

MODEL_FAIRNESS_RATIO = Gauge(
    "subscriber_model_fairness_ratio",
    "Worst-to-best group ratio from the fairness audit, by metric.",
    ["metric"],
    registry=REGISTRY,
)

MODEL_COST_SAVINGS = Gauge(
    "subscriber_model_cost_savings_available",
    "Money left on the table by the threshold in use, versus the cost-optimal "
    "one, on the test split.",
    registry=REGISTRY,
)

# The reality check on the number above. Savings computed by contacting 40% of
# the base are not available to a team that can contact 5%, and the gap between
# what the model wants to send and what anyone can actually send is the number
# that decides whether the cost analysis is a plan or a wish.

MODEL_OFFERS_REQUIRED = Gauge(
    "subscriber_model_offers_required",
    "Retention offers the cost-optimal threshold would send on the test split.",
    registry=REGISTRY,
)

MODEL_CAPACITY_BINDING = Gauge(
    "subscriber_model_capacity_binding",
    "1 when the outreach budget, not the economics, is what limits the "
    "retention campaign.",
    registry=REGISTRY,
)

MODEL_CAPACITY_SHORTFALL = Gauge(
    "subscriber_model_capacity_shortfall",
    "Subscribers the model would contact but the budget cannot cover.",
    registry=REGISTRY,
)

MODEL_CAPACITY_COST = Gauge(
    "subscriber_model_capacity_cost",
    "What the outreach budget ceiling costs on the test split, versus "
    "unconstrained cost-optimal outreach.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def record_prediction(probability: float, predicted_label: int, risk_level: str) -> None:
    """Record one served prediction. Called from the single prediction funnel."""
    PREDICTIONS.labels(risk_level=risk_level, predicted_label=str(predicted_label)).inc()
    PREDICTION_PROBABILITY.observe(float(probability))


def record_predictions(responses: list[dict[str, Any]]) -> None:
    """Record a batch of prediction response payloads."""
    for response in responses:
        record_prediction(
            response["dropout_probability"],
            response["predicted_label"],
            response["risk_level"],
        )


def record_shadow_comparison(comparison: Any) -> None:
    """Record one champion/challenger pair."""
    SHADOW_COMPARISONS.labels(agreed=str(comparison.agreed).lower()).inc()
    SHADOW_DIVERGENCE.observe(comparison.divergence)


def refresh_shadow_gauges(report: dict[str, Any]) -> None:
    """Publish the current shadow comparison summary as gauges."""
    SHADOW_ACTIVE.set(1 if report.get("active") else 0)

    agreement = report.get("agreement_rate")
    if agreement is not None:
        SHADOW_AGREEMENT_RATE.set(float(agreement))

    delta = report.get("flagged_rate_delta")
    if delta is not None:
        SHADOW_FLAGGED_RATE_DELTA.set(float(delta))


def refresh_serving_gauges(live: dict[str, Any]) -> None:
    """Mirror the in-process tracker snapshot onto the gauges.

    Counters and histograms accumulate on their own; gauges describe a current
    state, so they are set at scrape time from the same snapshot ``/metrics``
    returns. That keeps the two endpoints from ever disagreeing.
    """
    MODEL_LOADED.set(1 if live.get("model_loaded") else 0)
    FLAGGED_RATE.set(float(live.get("flagged_rate") or 0.0))

    threshold = live.get("threshold")
    DECISION_THRESHOLD.set(float(threshold) if threshold is not None else 0.0)

    shift = live.get("probability_mean_shift")
    PROBABILITY_MEAN_SHIFT.set(float(shift) if shift is not None else 0.0)


def refresh_model_source(served_from: str | None) -> None:
    """Expose which source the live model came from, as a one-hot gauge."""
    for source in ("registry", "local"):
        SERVED_FROM.labels(source=source).set(1 if served_from == source else 0)


def refresh_pipeline_gauges(report_path: Path | None = None) -> bool:
    """Load the last pipeline run's report and publish it as gauges.

    Returns:
        ``True`` when a report was found and read, ``False`` otherwise. A
        missing report is normal before the first scheduled run, so it is not
        an error - the gauges simply stay unset rather than reporting a
        misleading zero.
    """
    path = report_path or (settings.ARTIFACTS_DIR / "last_pipeline_run.json")
    if not path.exists():
        return False

    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:  # pragma: no cover - partially written file
        logger.warning("Could not parse pipeline report at %s", path)
        return False

    PIPELINE_NEEDS_ATTENTION.set(1 if report.get("needs_attention") else 0)

    promoted = (report.get("training") or {}).get("promoted")
    if promoted is not None:
        PIPELINE_PROMOTED.set(1 if promoted else 0)

    finished = report.get("finished_at")
    if finished:
        with contextlib.suppress(ValueError):  # pragma: no cover - defensive
            PIPELINE_LAST_RUN.set(datetime.fromisoformat(finished).timestamp())

    drift = report.get("drift") or {}
    if drift.get("available"):
        DRIFT_VERDICT.set(VERDICT_CODES.get(drift.get("overall_verdict", "stable"), 0))
        DRIFT_SAMPLE_SIZE.set(int(drift.get("n_samples") or 0))
        for item in drift.get("top_features", []):
            DRIFT_PSI.labels(feature=item["feature"]).set(float(item["psi"]))

    return True


def refresh_decision_quality(metrics_path: Path | None = None) -> bool:
    """Publish the last training run's calibration, cost and fairness results.

    Read from ``metrics.json`` at scrape time, the same way pipeline gauges are
    read from the run report: training is a batch job with no endpoint of its
    own, and adding a Pushgateway to carry four numbers is not worth the
    operational surface.
    """
    path = metrics_path or (settings.ARTIFACTS_DIR / "metrics.json")
    if not path.exists():
        return False

    try:
        quality = (json.loads(path.read_text()) or {}).get("decision_quality") or {}
    except json.JSONDecodeError:  # pragma: no cover - partially written file
        return False

    if not quality or "error" in quality:
        return False

    calibration = quality.get("calibration") or {}
    if "expected_calibration_error" in calibration:
        MODEL_CALIBRATION_ECE.set(float(calibration["expected_calibration_error"]))

    costs = quality.get("costs") or {}
    if costs.get("savings") is not None:
        MODEL_COST_SAVINGS.set(float(costs["savings"]))

    capacity = costs.get("capacity") or {}
    if capacity:
        MODEL_OFFERS_REQUIRED.set(float(capacity["unconstrained"]["offers_required"]))
        MODEL_CAPACITY_BINDING.set(1 if capacity.get("binding") else 0)
        MODEL_CAPACITY_SHORTFALL.set(float(capacity.get("shortfall") or 0))
        MODEL_CAPACITY_COST.set(float(capacity.get("cost_of_constraint") or 0.0))

    fairness = quality.get("fairness") or {}
    if "passes" in fairness:
        MODEL_FAIRNESS_PASSES.set(1 if fairness["passes"] else 0)
    for metric in ("selection_rate_ratio", "recall_ratio", "roc_auc_ratio"):
        value = fairness.get(metric)
        if value is not None:
            MODEL_FAIRNESS_RATIO.labels(metric=metric).set(float(value))

    return True


def render() -> tuple[bytes, str]:
    """Refresh point-in-time gauges and render the exposition payload.

    Returns:
        ``(body, content_type)`` ready to be returned from the endpoint.
    """
    from src.api import service

    try:
        live = service.live_metrics()
        refresh_serving_gauges(live)
        loaded = service._loaded_model  # noqa: SLF001 - same package, avoids a load attempt
        refresh_model_source(
            loaded.metadata.get("served_from", "local") if loaded else None
        )
        refresh_shadow_gauges(service.shadow_report())
    except Exception:  # noqa: BLE001 - scraping must never fail on a degraded service
        # A monitoring endpoint that 500s when the thing it monitors is
        # unhealthy is worse than useless: that is exactly when it is read.
        logger.exception("Could not refresh serving gauges; exposing what we have")

    refresh_pipeline_gauges()
    refresh_decision_quality()
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
