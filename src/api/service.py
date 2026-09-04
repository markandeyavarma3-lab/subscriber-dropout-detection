"""Model loading and prediction logic for the inference API.

The artifact is loaded once and cached in a module-level singleton, so the cost
is paid at application startup rather than on the first request.  Tests can
bypass disk entirely with :func:`set_model`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.api.shadow import ShadowComparison, get_shadow_tracker
from src.config import settings
from src.features.build_features import frame_from_records
from src.monitoring import prometheus
from src.monitoring.drift import detect_drift
from src.monitoring.profile import load_reference_profile
from src.monitoring.tracker import get_tracker

logger = logging.getLogger(__name__)

# Risk bands are anchored to the model's decision threshold rather than to
# fixed cut-offs, so a band can never contradict the label: "low" is exactly
# the not-flagged region, and "high" starts halfway from the threshold to
# certainty.  Because training tunes the threshold, the bands move with it.
HIGH_RISK_SPAN: float = 0.5


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before an artifact is available."""


class DriftBaselineUnavailableError(RuntimeError):
    """Raised when drift is requested but the model shipped no reference profile."""


@dataclass
class LoadedModel:
    """A fitted pipeline together with the metadata saved alongside it."""

    pipeline: Pipeline
    threshold: float
    metadata: dict[str, Any]
    # Absent when the artifact predates monitoring or was trained elsewhere;
    # prediction still works, drift reporting is simply unavailable.
    reference_profile: dict[str, Any] | None = None
    # The shadow model. Always optional: there may be no challenger
    # registered, or it may be the same version as the champion (which is the
    # normal state right after a promotion), and neither is a problem.
    challenger: Pipeline | None = None
    challenger_threshold: float | None = None
    challenger_version: str | None = None

    # Built on first use rather than at load time, and cached here so it dies
    # with the model it explains. Constructing a TreeExplainer walks the whole
    # ensemble; doing it per request would dominate prediction latency.
    _attributor: Any | None = field(default=None, repr=False)
    _attributor_built: bool = field(default=False, repr=False)

    @property
    def shadow_active(self) -> bool:
        """Whether a distinct challenger is available to shadow-score with."""
        return self.challenger is not None

    def attributor(self) -> Any | None:
        """The SHAP attributor for this model, or ``None`` if unavailable.

        ``None`` is an ordinary outcome, not an error: ``shap`` is an optional
        dependency and not every model is a tree ensemble. Callers fall back to
        the rule-based explanation.
        """
        if not self._attributor_built:
            from src.evaluation.explain import build_attributor

            self._attributor = build_attributor(self.pipeline)
            self._attributor_built = True
        return self._attributor


_loaded_model: LoadedModel | None = None

# Used only to decide whether a request is sampled for shadow scoring, so it
# needs no reproducibility guarantee - and must not disturb the global seed.
_rng = np.random.default_rng()


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


def _load_from_registry(model_name: str | None = None) -> LoadedModel:
    """Load the ``@champion`` model directly from the MLflow registry.

    This is what makes gated promotion (:mod:`src.registry.promote`) actually
    affect what the API serves.  Without it, promoting a challenger only
    changed a database row - the running service kept serving whatever
    ``model.joblib`` happened to be on disk.

    Raises:
        ModelNotLoadedError: If MLflow is unreachable, no ``@champion`` alias
            is set for this model name, or the aliased version fails to load.
    """
    from src.registry import tracking

    name = model_name or settings.REGISTERED_MODEL_NAME
    try:
        version = tracking.get_alias_version(settings.CHAMPION_ALIAS, name)
    except Exception as exc:  # noqa: BLE001 - MLflow raises several transport/DB types
        raise ModelNotLoadedError(
            f"MLflow registry unreachable at {settings.MLFLOW_TRACKING_URI!r}: {exc}"
        ) from exc

    if version is None:
        raise ModelNotLoadedError(
            f"No @{settings.CHAMPION_ALIAS} alias is set for {name!r} in the MLflow "
            "registry. Train and promote one with "
            "`python -m src.models.train --source warehouse --promote`."
        )

    pipeline = tracking.load_aliased_model(settings.CHAMPION_ALIAS, name)
    if pipeline is None:  # pragma: no cover - alias resolved but artifact unreadable
        raise ModelNotLoadedError(
            f"@{settings.CHAMPION_ALIAS} version {version.version} of {name!r} could "
            "not be deserialised from the MLflow artifact store."
        )

    # The threshold travels as a logged run param (see
    # `_track_and_maybe_promote` in train.py), since the registry has nowhere
    # else to carry it - the pipeline itself is just an sklearn estimator.
    client = tracking.configure()
    run = client.get_run(version.run_id) if version.run_id else None
    params = run.data.params if run else {}
    threshold = float(params.get("decision_threshold", settings.DECISION_THRESHOLD))

    from src.features.build_features import REQUIRED_INPUT_COLUMNS

    metadata: dict[str, Any] = {
        "model_name": settings.MODEL_NAME,
        "decision_threshold": threshold,
        "required_input_columns": REQUIRED_INPUT_COLUMNS,
        "served_from": "registry",
        "registry_version": str(version.version),
        "registry_run_id": version.run_id,
    }

    # Best-effort: the drift baseline is a local file, not a registry
    # artifact, so it still applies even when the model itself came from
    # MLflow. Its absence only disables /monitoring/drift, never /predict.
    profile = load_reference_profile()

    challenger, challenger_threshold, challenger_version = _load_challenger(
        name, champion_version=str(version.version)
    )

    logger.info(
        "Loaded %r @%s -> registry version %s (threshold=%.2f)",
        name,
        settings.CHAMPION_ALIAS,
        version.version,
        threshold,
    )
    return LoadedModel(
        pipeline=pipeline,
        threshold=threshold,
        metadata=metadata,
        reference_profile=profile,
        challenger=challenger,
        challenger_threshold=challenger_threshold,
        challenger_version=challenger_version,
    )


def _load_challenger(
    model_name: str, champion_version: str
) -> tuple[Pipeline | None, float | None, str | None]:
    """Load ``@challenger`` for shadow scoring, if there is a distinct one.

    Never raises. A missing, broken or unreachable challenger disables shadow
    scoring and nothing else - the whole point of a shadow model is that
    production does not depend on it.

    Returns ``(None, None, None)`` when the challenger alias is unset, points
    at the same version as the champion (the normal state straight after a
    promotion, where shadowing would just compare a model to itself), or
    cannot be loaded.
    """
    if not settings.SHADOW_ENABLED:
        return None, None, None

    from src.registry import tracking

    try:
        version = tracking.get_alias_version(settings.CHALLENGER_ALIAS, model_name)
        if version is None:
            return None, None, None
        if str(version.version) == champion_version:
            logger.info(
                "Shadow scoring idle: @%s and @%s both point at version %s",
                settings.CHALLENGER_ALIAS,
                settings.CHAMPION_ALIAS,
                champion_version,
            )
            return None, None, None

        pipeline = tracking.load_aliased_model(settings.CHALLENGER_ALIAS, model_name)
        if pipeline is None:
            return None, None, None

        client = tracking.configure()
        run = client.get_run(version.run_id) if version.run_id else None
        params = run.data.params if run else {}
        threshold = float(params.get("decision_threshold", settings.DECISION_THRESHOLD))

        logger.info(
            "Shadow scoring enabled against @%s version %s (threshold=%.2f)",
            settings.CHALLENGER_ALIAS,
            version.version,
            threshold,
        )
        return pipeline, threshold, str(version.version)
    except Exception:  # noqa: BLE001 - a shadow model must never break startup
        logger.warning("Could not load a challenger for shadow scoring", exc_info=True)
        return None, None, None


def _load_from_disk(
    model_path: Path | None, metadata_path: Path | None, profile_path: Path | None
) -> LoadedModel:
    """Load the pipeline, its metadata and its drift baseline from the filesystem.

    This is the original loading path, unchanged: it is what ``load_model``
    falls back to when the registry is unavailable, and what it uses outright
    when ``SDD_MODEL_SOURCE=local``.

    Raises:
        ModelNotLoadedError: If the artifact file does not exist.
    """
    path = model_path or settings.MODEL_PATH
    meta_path = metadata_path or settings.METADATA_PATH
    if not path.exists():
        raise ModelNotLoadedError(
            f"No model artifact at {path}. Train one with `python -m src.models.train`."
        )

    pipeline = joblib.load(path)
    metadata = _read_metadata(meta_path)
    threshold = float(metadata.get("decision_threshold", settings.DECISION_THRESHOLD))
    profile = load_reference_profile(profile_path)

    logger.info(
        "Loaded model from %s (threshold=%.2f, drift baseline=%s)",
        path,
        threshold,
        "yes" if profile else "no",
    )
    return LoadedModel(
        pipeline=pipeline, threshold=threshold, metadata=metadata, reference_profile=profile
    )


def load_model(
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    profile_path: Path | None = None,
    source: str | None = None,
) -> LoadedModel:
    """Load the model the API will serve, and cache it.

    ``SDD_MODEL_SOURCE`` (default ``"auto"``) decides where from:

    - ``"registry"`` - only ever load ``@champion`` from MLflow; raise if it
      is not there. Use this once you want promotion to be the only way a new
      model reaches production.
    - ``"local"`` - only ever load ``model.joblib`` from disk, ignoring the
      registry entirely. The original behaviour.
    - ``"auto"`` - try the registry first, and fall back to the local
      artifact if MLflow is unreachable or has no ``@champion`` set. This is
      the default so a fresh clone with no MLflow server still serves.

    Raises:
        ModelNotLoadedError: If neither source (per the mode above) produced
            a usable model.
    """
    global _loaded_model

    resolved = source or settings.MODEL_SOURCE
    if resolved not in ("auto", "registry", "local"):
        raise ValueError(
            f"Unknown model source {resolved!r}; expected 'auto', 'registry' or 'local'."
        )

    if resolved in ("auto", "registry"):
        try:
            _loaded_model = _load_from_registry()
            return _loaded_model
        except ModelNotLoadedError as exc:
            if resolved == "registry":
                raise
            logger.info("Registry model unavailable (%s); falling back to local artifact.", exc)

    _loaded_model = _load_from_disk(model_path, metadata_path, profile_path)
    return _loaded_model


def get_model() -> LoadedModel:
    """Return the cached model, loading it on first use."""
    if _loaded_model is None:
        return load_model()
    return _loaded_model


def set_model(
    pipeline: Pipeline,
    threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
    reference_profile: dict[str, Any] | None = None,
) -> LoadedModel:
    """Inject an already-fitted pipeline into the cache (used by tests)."""
    global _loaded_model
    _loaded_model = LoadedModel(
        pipeline=pipeline,
        threshold=float(threshold if threshold is not None else settings.DECISION_THRESHOLD),
        metadata=metadata or {},
        reference_profile=reference_profile,
    )
    return _loaded_model


def reset_model() -> None:
    """Clear the cached model so the next call reloads from disk."""
    global _loaded_model
    _loaded_model = None


def is_model_loaded() -> bool:
    """Whether an artifact is currently held in memory."""
    return _loaded_model is not None


def classify_risk_level(probability: float, threshold: float | None = None) -> str:
    """Map a probability onto a ``low`` / ``medium`` / ``high`` band.

    The bands hang off the decision threshold, which guarantees the invariant
    ``(level == "low") == (predicted_label == 0)``.  Fixed cut-offs used to
    break that: with a tuned threshold of 0.26, a subscriber scoring 0.30 was
    flagged for outreach while being described as low risk.

    Args:
        probability: Predicted dropout probability.
        threshold: Decision threshold in use; defaults to the configured one.
    """
    cutoff = settings.DECISION_THRESHOLD if threshold is None else threshold
    if probability < cutoff:
        return "low"
    if probability >= cutoff + (1.0 - cutoff) * HIGH_RISK_SPAN:
        return "high"
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
    features: dict[str, Any],
    probability: float,
    risk_level: str,
    attributions: list[Any] | None = None,
) -> tuple[str, list[str], str]:
    """Compose the human-readable explanation and its supporting factors.

    Args:
        features: The raw record, used to quote the subscriber's own numbers.
        probability: The predicted dropout probability.
        risk_level: The band the probability falls in.
        attributions: SHAP attributions for this row, when available. These
            explain the *decision*; the rules below only ever explained the
            *input*, firing whether or not the model weighed that feature.

    Returns:
        ``(explanation, top_factors, method)``. ``method`` is ``"shap"`` or
        ``"rules"`` - reported rather than inferred, so nobody has to guess
        which produced the sentence they are reading.
    """
    if risk_level == "high":
        opening = "High dropout risk"
    elif risk_level == "medium":
        opening = "Moderate dropout risk"
    else:
        opening = "Low dropout risk"

    if attributions:
        # The sentence uses both directions. SHAP ranks by magnitude either
        # way, so a strong reason the subscriber is staying can outrank a weak
        # reason they might leave - which is the improvement over the rules,
        # since those could only ever list what was wrong and so made every
        # explanation read like a warning.
        body = ", ".join(item.description for item in attributions)
        # `top_risk_factors` keeps its name's meaning, though: only the
        # contributions that actually push risk *up*. Quietly widening a public
        # field to mean "drivers, any direction" would leave every existing
        # consumer reading reasons-to-stay as reasons-to-worry.
        top_factors = [item.description for item in attributions if item.contribution > 0]
        return f"{opening} ({probability:.0%}): {body}.", top_factors, "shap"

    risk_factors = collect_risk_factors(features)
    top_factors = risk_factors[:3]

    if top_factors:
        body = ", ".join(top_factors)
    else:
        signals = collect_retention_signals(features)
        body = ", ".join(signals) if signals else "no notable risk signals detected"

    return f"{opening} ({probability:.0%}): {body}.", top_factors, "rules"


def _attribute(
    loaded: LoadedModel, frame: pd.DataFrame, records: list[dict[str, Any]]
) -> list[list[Any]]:
    """SHAP attributions for a whole batch, or empty lists if unavailable.

    Computed once per batch rather than per row: TreeSHAP is vectorised, and
    the per-call overhead is most of the cost on small batches.

    Wrapped, like shadow scoring, so that explanation can never take down
    prediction. A model that scores fine but cannot be explained should return
    a probability with a rule-based sentence attached, not a 500.
    """
    attributor = loaded.attributor()
    if attributor is None:
        return [[] for _ in records]

    try:
        return attributor.top_attributions(frame, records)
    except Exception:  # noqa: BLE001 - an explanation must not fail a prediction
        logger.exception("SHAP attribution failed; falling back to rule-based explanations")
        return [[] for _ in records]


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
    attributions = _attribute(loaded, frame, records)

    responses: list[dict[str, Any]] = []
    for record, probability, row in zip(records, probabilities, attributions, strict=True):
        probability = float(probability)
        risk_level = classify_risk_level(probability, loaded.threshold)
        explanation, top_factors, method = build_explanation(
            record, probability, risk_level, row
        )
        responses.append(
            {
                "dropout_probability": round(probability, 4),
                "predicted_label": int(probability >= loaded.threshold),
                "risk_level": risk_level,
                "threshold": round(loaded.threshold, 4),
                "explanation": explanation,
                "top_risk_factors": top_factors,
                "explanation_method": method,
                "attributions": [item.as_dict() for item in row] if row else None,
            }
        )

    # Recorded here rather than in the routes so that every path to a
    # prediction - single, batch, or a future one - is counted exactly once.
    get_tracker().record_many(responses)
    prometheus.record_predictions(responses)

    _shadow_score(loaded, frame, probabilities)
    return responses


def _shadow_score(
    loaded: LoadedModel, frame: pd.DataFrame, champion_probabilities: np.ndarray
) -> None:
    """Score the same batch with the challenger, recording but never serving it.

    Wrapped whole in a bare ``except``, deliberately. The single rule of shadow
    scoring is that the shadow can never affect the response: a challenger that
    raises, hangs on a weird input, or returns the wrong shape must cost a
    logged error and nothing else. Letting it propagate would mean a model
    nobody is serving could take down the service - the exact failure shadow
    deployment exists to prevent.

    Runs inline rather than in a background thread. That costs real latency
    (roughly a second forward pass), which is why ``SDD_SHADOW_SAMPLE_RATE``
    exists; a thread pool would hide the cost but add a queue that can silently
    fall behind under load, which is a worse failure to debug.
    """
    if not loaded.shadow_active or not settings.SHADOW_ENABLED:
        return

    tracker = get_shadow_tracker()
    try:
        if settings.SHADOW_SAMPLE_RATE < 1.0 and _rng.random() >= settings.SHADOW_SAMPLE_RATE:
            return

        threshold = (
            loaded.challenger_threshold
            if loaded.challenger_threshold is not None
            else settings.DECISION_THRESHOLD
        )
        with prometheus.SHADOW_LATENCY.time():
            challenger_probabilities = loaded.challenger.predict_proba(frame)[:, 1]

        for champion_probability, challenger_probability in zip(
            champion_probabilities, challenger_probabilities, strict=True
        ):
            comparison = ShadowComparison(
                champion_probability=float(champion_probability),
                champion_label=int(champion_probability >= loaded.threshold),
                challenger_probability=float(challenger_probability),
                challenger_label=int(challenger_probability >= threshold),
            )
            tracker.record(comparison)
            prometheus.record_shadow_comparison(comparison)
    except Exception:  # noqa: BLE001 - the shadow must never break serving
        tracker.record_error()
        prometheus.SHADOW_ERRORS.inc()
        logger.warning("Challenger failed to shadow-score this batch", exc_info=True)


def shadow_report() -> dict[str, Any]:
    """Summarise what shadow traffic says about promoting the challenger.

    Deliberately reports no accuracy verdict. Shadow traffic has no labels, so
    it cannot say which model is *better* - only how differently they behave,
    and whether the challenger survives production input at all.
    """
    loaded = _loaded_model
    tracker = get_shadow_tracker()

    report: dict[str, Any] = {
        "enabled": settings.SHADOW_ENABLED,
        "active": bool(loaded and loaded.shadow_active),
        "sample_rate": settings.SHADOW_SAMPLE_RATE,
        "champion_version": (loaded.metadata.get("registry_version") if loaded else None),
        "challenger_version": (loaded.challenger_version if loaded else None),
        **tracker.snapshot(),
        **tracker.readiness(),
    }

    if not report["active"]:
        report["detail"] = (
            "No distinct challenger is registered. Shadow scoring stays idle when "
            "@challenger is unset or points at the same version as @champion."
        )
    return report


def predict_one(
    features: dict[str, Any], model: LoadedModel | None = None
) -> dict[str, Any]:
    """Score a single subscriber and return the response payload."""
    return predict_batch([features], model=model)[0]


def live_metrics() -> dict[str, Any]:
    """Summarise what this process has served, plus its drift baseline status."""
    snapshot = get_tracker().snapshot()
    loaded = _loaded_model
    snapshot["model_loaded"] = loaded is not None
    snapshot["threshold"] = round(loaded.threshold, 4) if loaded else None

    profile = loaded.reference_profile if loaded else None
    reference = profile.get("prediction") if profile else None
    snapshot["reference"] = (
        {
            "created_at": profile.get("created_at"),
            "rows": profile.get("n_reference_rows"),
            "probability_mean": reference.get("mean") if reference else None,
        }
        if profile
        else None
    )

    # The single number most worth alerting on: how far the live score mean has
    # moved from what the model produced on its own training data.
    if reference and snapshot["window_size"]:
        snapshot["probability_mean_shift"] = round(
            snapshot["probability"]["mean"] - float(reference.get("mean", 0.0)), 4
        )
    else:
        snapshot["probability_mean_shift"] = None

    return snapshot


def drift_report(
    records: list[dict[str, Any]], model: LoadedModel | None = None
) -> dict[str, Any]:
    """Compare a batch of live subscribers against the training reference.

    Scores the batch as a side effect so prediction drift can be reported
    alongside input drift.

    Raises:
        ModelNotLoadedError: If no artifact is available.
        DriftBaselineUnavailableError: If the artifact shipped no profile.
    """
    loaded = model or get_model()
    if loaded.reference_profile is None:
        raise DriftBaselineUnavailableError(
            "This model has no reference_profile.json. Retrain with "
            "`python -m src.models.train` to generate a drift baseline."
        )

    frame = frame_from_records(records)
    probabilities = loaded.pipeline.predict_proba(frame)[:, 1]
    return detect_drift(frame, loaded.reference_profile, probabilities=probabilities)


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
        "served_from": metadata.get("served_from", "local"),
        "registry_version": metadata.get("registry_version"),
    }
