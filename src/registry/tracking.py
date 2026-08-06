"""MLflow experiment tracking and model registration.

Two problems this solves that a ``model.joblib`` on disk does not:

**Provenance.** Every run records its parameters, metrics, the training window
it used and the library versions it ran under.  "Which model is in production
and what was it trained on?" becomes a query rather than an archaeology
exercise through commit history.

**Non-destructive iteration.** Training previously overwrote the artifact in
place, so a worse model silently replaced a better one and there was no way
back.  Registered versions are immutable and additive; promotion is a separate,
gated decision (see :mod:`src.registry.promote`).

MLflow 3 removed model *stages*, so production status is expressed with
**aliases** - ``@champion`` is whatever is serving, ``@challenger`` is whatever
is being evaluated against it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.config import settings

logger = logging.getLogger(__name__)


def configure(tracking_uri: str | None = None, experiment: str | None = None) -> MlflowClient:
    """Point MLflow at the configured backend and ensure the experiment exists."""
    uri = tracking_uri or settings.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or settings.MLFLOW_EXPERIMENT)
    return MlflowClient(tracking_uri=uri)


@contextmanager
def start_run(run_name: str | None = None, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Open an MLflow run against the configured experiment."""
    configure()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def _flatten(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten a nested metrics dict into MLflow's flat float-only namespace."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flat.update(_flatten(f"{prefix}{key}_", value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def log_training_run(
    model: Pipeline,
    params: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    training_window: dict[str, Any] | None = None,
    input_example: pd.DataFrame | None = None,
    register: bool = True,
    model_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> tuple[str, ModelVersion | None]:
    """Log one training run and optionally register the resulting model.

    Args:
        model: The fitted pipeline.
        params: Hyperparameters and configuration to record.
        validation_metrics: Metrics on the validation split.
        test_metrics: Metrics on the held-out split.
        training_window: The point-in-time window the data came from.
        input_example: A few raw rows, so the registry stores a signature.
        register: Whether to create a registered model version.
        model_name: Registry name. Defaults to settings.
        tags: Extra run tags.

    Returns:
        ``(run_id, model_version)``; the version is ``None`` when not registering.
    """
    client = configure()
    name = model_name or settings.REGISTERED_MODEL_NAME

    with mlflow.start_run(tags=tags) as run:
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics(_flatten("val_", validation_metrics))
        mlflow.log_metrics(_flatten("test_", test_metrics))

        if training_window:
            mlflow.log_params({f"window_{k}": v for k, v in training_window.items()})

        signature = None
        if input_example is not None and not input_example.empty:
            from mlflow.models import infer_signature

            signature = infer_signature(input_example, model.predict_proba(input_example)[:, 1])

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example.head(3) if input_example is not None else None,
            registered_model_name=name if register else None,
            # MLflow 3 defaults to skops, which refuses to serialise custom
            # callables.  Our pipeline embeds `add_derived_features` as a
            # FunctionTransformer precisely so feature engineering travels
            # with the artifact, so skops rejects it by design. cloudpickle
            # carries it, at the same trust level as the joblib file the API
            # already loads: only ever deserialise models this project built.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    version: ModelVersion | None = None
    if register:
        # log_model registers asynchronously in some backends, so resolve the
        # version by run id rather than assuming it is the highest number.
        versions = client.search_model_versions(f"name='{name}'")
        matching = [v for v in versions if v.run_id == run_id]
        version = max(matching, key=lambda v: int(v.version)) if matching else None
        if version is not None:
            client.set_registered_model_alias(name, settings.CHALLENGER_ALIAS, version.version)
            logger.info("Registered %s version %s as @challenger", name, version.version)

    return run_id, version


def get_alias_version(alias: str, model_name: str | None = None) -> ModelVersion | None:
    """Return the model version behind an alias, or ``None`` if unset."""
    client = configure()
    name = model_name or settings.REGISTERED_MODEL_NAME
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:  # noqa: BLE001 - MLflow raises several types for "absent"
        return None


def load_aliased_model(alias: str, model_name: str | None = None) -> Pipeline | None:
    """Load the pipeline behind an alias, or ``None`` if there is not one."""
    name = model_name or settings.REGISTERED_MODEL_NAME
    if get_alias_version(alias, name) is None:
        return None
    configure()
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def set_alias(alias: str, version: str | int, model_name: str | None = None) -> None:
    """Point an alias at a specific registered version."""
    client = configure()
    client.set_registered_model_alias(
        model_name or settings.REGISTERED_MODEL_NAME, alias, str(version)
    )


def run_history(model_name: str | None = None, limit: int = 20) -> pd.DataFrame:
    """Recent registered versions with their key metrics, newest first."""
    client = configure()
    name = model_name or settings.REGISTERED_MODEL_NAME

    # Version objects from a search do not carry their aliases, so the mapping
    # is read from the registered model and inverted.
    try:
        alias_of: dict[str, list[str]] = {}
        for alias, version_number in (client.get_registered_model(name).aliases or {}).items():
            alias_of.setdefault(str(version_number), []).append(alias)
    except Exception:  # noqa: BLE001 - model not registered yet
        alias_of = {}

    rows: list[dict[str, Any]] = []
    for version in client.search_model_versions(f"name='{name}'"):
        run = client.get_run(version.run_id) if version.run_id else None
        metrics = run.data.metrics if run else {}
        rows.append(
            {
                "version": int(version.version),
                "run_id": version.run_id,
                "aliases": alias_of.get(str(version.version), []),
                "created": pd.to_datetime(version.creation_timestamp, unit="ms"),
                **{key: value for key, value in metrics.items() if key.startswith("test_")},
            }
        )

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("version", ascending=False)
    return frame.head(limit).reset_index(drop=True)


def score_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for a fitted pipeline."""
    return model.predict_proba(features)[:, 1]
