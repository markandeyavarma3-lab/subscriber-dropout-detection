"""Model loading and prediction logic for the inference API.

The artifact is loaded once and cached in a module-level singleton, so the cost
is paid at application startup rather than on the first request.  Tests can
bypass disk entirely with :func:`set_model`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import settings
from src.features.build_features import frame_from_records

logger = logging.getLogger(__name__)

# Risk-band cut-offs applied to the predicted probability.
LOW_RISK_CEILING: float = 0.35
HIGH_RISK_FLOOR: float = 0.65


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before an artifact is available."""


@dataclass
class LoadedModel:
    """A fitted pipeline together with the metadata saved alongside it."""

    pipeline: Pipeline
    threshold: float
    metadata: dict[str, Any]


_loaded_model: LoadedModel | None = None


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    """Read ``metadata.json`` if present; return an empty dict otherwise."""
    if not metadata_path.exists():
        logger.warning("No metadata found at %s; using configured defaults.", metadata_path)
        return {}
    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:  # pragma: no cover - corrupt artifact directory
        logger.warning("Could not parse %s; using configured defaults.", metadata_path)
        return {}


def load_model(
    model_path: Path | None = None, metadata_path: Path | None = None
) -> LoadedModel:
    """Load the pipeline and its metadata from disk and cache them.

    Args:
        model_path: Location of ``model.joblib``. Defaults to settings.
        metadata_path: Location of ``metadata.json``. Defaults to settings.

    Raises:
        ModelNotLoadedError: If the artifact file does not exist.
    """
    global _loaded_model

    path = model_path or settings.MODEL_PATH
    meta_path = metadata_path or settings.METADATA_PATH
    if not path.exists():
        raise ModelNotLoadedError(
            f"No model artifact at {path}. Train one with `python -m src.models.train`."
        )

    pipeline = joblib.load(path)
    metadata = _read_metadata(meta_path)
    threshold = float(metadata.get("decision_threshold", settings.DECISION_THRESHOLD))

    _loaded_model = LoadedModel(pipeline=pipeline, threshold=threshold, metadata=metadata)
    logger.info("Loaded model from %s (threshold=%.2f)", path, threshold)
    return _loaded_model


def get_model() -> LoadedModel:
    """Return the cached model, loading it on first use."""
    if _loaded_model is None:
        return load_model()
    return _loaded_model


def set_model(
    pipeline: Pipeline, threshold: float | None = None, metadata: dict[str, Any] | None = None
) -> LoadedModel:
    """Inject an already-fitted pipeline into the cache (used by tests)."""
    global _loaded_model
    _loaded_model = LoadedModel(
        pipeline=pipeline,
        threshold=float(threshold if threshold is not None else settings.DECISION_THRESHOLD),
        metadata=metadata or {},
    )
    return _loaded_model


def reset_model() -> None:
    """Clear the cached model so the next call reloads from disk."""
    global _loaded_model
    _loaded_model = None


def is_model_loaded() -> bool:
    """Whether an artifact is currently held in memory."""
    return _loaded_model is not None


def classify_risk_level(probability: float) -> str:
    """Map a probability onto a coarse ``low`` / ``medium`` / ``high`` band."""
    if probability >= HIGH_RISK_FLOOR:
        return "high"
    if probability <= LOW_RISK_CEILING:
        return "low"
    return "medium"


def collect_risk_factors(features: dict[str, Any]) -> list[str]:
    """List the rule-based risk signals that fired for one subscriber.

    Deliberately simple and threshold-driven (see
    :class:`src.config.settings.ExplanationRules`).  It explains the *inputs*,
    not the model internals - swap in SHAP later if attribution is needed.
    """
    rules = settings.EXPLANATION_RULES
    factors: list[str] = []

    recency = features.get("last_activity_days_ago", 0)
    sessions = features.get("avg_session_count_last_30d", 0.0)
    tickets = features.get("support_tickets_last_90d", 0)
    failures = features.get("payment_failures_last_6m", 0)
    discounts = features.get("discounts_used_last_6m", 0)
    tenure = features.get("tenure_days", 0)

    if recency > rules.dormant_days:
        factors.append(f"inactive for {recency} days")
    if sessions < rules.low_session_count:
        factors.append(f"low recent activity ({sessions:.1f} sessions/30d)")
    if failures >= rules.high_payment_failures:
        factors.append(f"{failures} payment failures in the last 6 months")
    if tickets >= rules.high_support_tickets:
        factors.append(f"{tickets} support tickets in the last 90 days")
    if not features.get("is_auto_renew_enabled", False):
        factors.append("auto-renew disabled")
    if discounts >= rules.heavy_discount_use:
        factors.append(f"relies on discounts ({discounts} in the last 6 months)")
    if tenure < rules.new_subscriber_days:
        factors.append(f"still new ({tenure} days since signup)")

    return factors


def collect_retention_signals(features: dict[str, Any]) -> list[str]:
    """List the positive signals that argue *against* dropout."""
    rules = settings.EXPLANATION_RULES
    signals: list[str] = []

    if features.get("is_auto_renew_enabled", False):
        signals.append("auto-renew enabled")
    if features.get("avg_session_count_last_30d", 0.0) >= rules.healthy_session_count:
        signals.append("strong recent engagement")
    if features.get("last_activity_days_ago", 999) <= 3:
        signals.append("active in the last few days")
    if features.get("tenure_days", 0) >= rules.loyal_subscriber_days:
        signals.append("long-tenured subscriber")

    return signals


def build_explanation(
    features: dict[str, Any], probability: float, risk_level: str
) -> tuple[str, list[str]]:
    """Compose the human-readable explanation and its supporting factors.

    Returns:
        ``(explanation, top_risk_factors)`` where the factor list holds at most
        the three strongest signals.
    """
    risk_factors = collect_risk_factors(features)
    top_factors = risk_factors[:3]

    if risk_level == "high":
        opening = "High dropout risk"
    elif risk_level == "medium":
        opening = "Moderate dropout risk"
    else:
        opening = "Low dropout risk"

    if top_factors:
        body = ", ".join(top_factors)
        explanation = f"{opening} ({probability:.0%}): {body}."
    else:
        signals = collect_retention_signals(features)
        body = ", ".join(signals) if signals else "no notable risk signals detected"
        explanation = f"{opening} ({probability:.0%}): {body}."

    return explanation, top_factors


def predict_batch(
    records: list[dict[str, Any]], model: LoadedModel | None = None
) -> list[dict[str, Any]]:
    """Score a batch of subscribers.

    Args:
        records: Raw feature dicts, one per subscriber.
        model: Optional preloaded model; the cached one is used by default.

    Returns:
        One response dict per input record, in the same order.

    Raises:
        ModelNotLoadedError: If no artifact is available.
    """
    if not records:
        return []

    loaded = model or get_model()
    frame: pd.DataFrame = frame_from_records(records)
    probabilities = loaded.pipeline.predict_proba(frame)[:, 1]

    responses: list[dict[str, Any]] = []
    for record, probability in zip(records, probabilities, strict=True):
        probability = float(probability)
        risk_level = classify_risk_level(probability)
        explanation, top_factors = build_explanation(record, probability, risk_level)
        responses.append(
            {
                "dropout_probability": round(probability, 4),
                "predicted_label": int(probability >= loaded.threshold),
                "risk_level": risk_level,
                "threshold": round(loaded.threshold, 4),
                "explanation": explanation,
                "top_risk_factors": top_factors,
            }
        )
    return responses


def predict_one(
    features: dict[str, Any], model: LoadedModel | None = None
) -> dict[str, Any]:
    """Score a single subscriber and return the response payload."""
    return predict_batch([features], model=model)[0]


def model_info() -> dict[str, Any]:
    """Describe the artifact currently being served."""
    from src.features.build_features import REQUIRED_INPUT_COLUMNS

    loaded = get_model()
    metadata = loaded.metadata
    return {
        "model_name": metadata.get("model_name", settings.MODEL_NAME),
        "trained_at": metadata.get("trained_at"),
        "decision_threshold": loaded.threshold,
        "required_input_columns": metadata.get(
            "required_input_columns", REQUIRED_INPUT_COLUMNS
        ),
        "library_versions": metadata.get("library_versions", {}),
    }
