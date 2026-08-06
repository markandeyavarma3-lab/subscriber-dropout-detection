"""Shared pytest fixtures.

Training a full model would make the suite slow, so every test that needs a
fitted artifact reuses one small session-scoped model trained into a temporary
directory.  Nothing here writes into the repository.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import service
from src.api.main import app
from src.data.generate import generate_subscribers, write_dataset
from src.models.train import TrainingResult, run_training
from src.monitoring.tracker import get_tracker

TINY_MODEL_PARAMS: dict[str, Any] = {"n_estimators": 25, "max_depth": 2, "learning_rate": 0.2}

# A subscriber with several dropout signals: dormant, disengaged, failing
# payments, auto-renew off.
HIGH_RISK_SUBSCRIBER: dict[str, Any] = {
    "tenure_days": 95,
    "plan_type": "standard",
    "monthly_fee": 19.99,
    "avg_session_count_last_30d": 1.0,
    "last_activity_days_ago": 45,
    "support_tickets_last_90d": 4,
    "payment_failures_last_6m": 3,
    "discounts_used_last_6m": 3,
    "is_auto_renew_enabled": False,
}

# A healthy subscriber: long tenure, engaged, active yesterday, auto-renew on.
LOW_RISK_SUBSCRIBER: dict[str, Any] = {
    "tenure_days": 1200,
    "plan_type": "premium",
    "monthly_fee": 34.99,
    "avg_session_count_last_30d": 26.0,
    "last_activity_days_ago": 1,
    "support_tickets_last_90d": 0,
    "payment_failures_last_6m": 0,
    "discounts_used_last_6m": 0,
    "is_auto_renew_enabled": True,
}


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    """A small synthetic dataset used by the feature tests."""
    return generate_subscribers(n_subscribers=400, seed=7)


@pytest.fixture(scope="session")
def sample_features(sample_frame: pd.DataFrame) -> pd.DataFrame:
    """The sample dataset without the ID and target columns."""
    return sample_frame.drop(columns=["subscriber_id", "dropout"])


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> TrainingResult:
    """Train one small model per test session into a temporary directory."""
    workdir: Path = tmp_path_factory.mktemp("artifacts")
    data_path = workdir / "subscribers.csv"
    write_dataset(output_path=data_path, n_subscribers=600, seed=13)

    return run_training(
        data_path=data_path,
        artifacts_dir=workdir,
        model_params=TINY_MODEL_PARAMS,
        persist_test_split=False,
        verbose=False,
    )


@pytest.fixture(scope="session")
def trained_metadata(trained_model: TrainingResult) -> dict[str, Any]:
    """Metadata written alongside the session's model."""
    return json.loads(trained_model.metadata_path.read_text())


@pytest.fixture(scope="session")
def reference_profile(trained_model: TrainingResult) -> dict[str, Any]:
    """The drift baseline written alongside the session's model."""
    return json.loads(trained_model.reference_profile_path.read_text())


@pytest.fixture()
def client(
    trained_model: TrainingResult,
    trained_metadata: dict[str, Any],
    reference_profile: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """A TestClient whose app serves the session's small model.

    ``service.load_model`` is patched so application startup uses the in-memory
    artifact instead of whatever happens to be on disk, keeping the tests
    independent of the developer's working tree.
    """

    def _load_tiny_model(*_args: Any, **_kwargs: Any) -> service.LoadedModel:
        return service.set_model(
            trained_model.model, trained_model.threshold, trained_metadata, reference_profile
        )

    monkeypatch.setattr(service, "load_model", _load_tiny_model)
    # The tracker is process-wide, so a previous test's predictions would
    # otherwise leak into this one's /metrics assertions.
    get_tracker().reset()
    with TestClient(app) as test_client:
        yield test_client
    service.reset_model()
    get_tracker().reset()


@pytest.fixture()
def client_without_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient for an app that failed to find a model artifact."""

    def _raise(*_args: Any, **_kwargs: Any) -> service.LoadedModel:
        raise service.ModelNotLoadedError("No model artifact at /nonexistent/model.joblib.")

    service.reset_model()
    monkeypatch.setattr(service, "load_model", _raise)
    monkeypatch.setattr(service, "get_model", _raise)
    with TestClient(app) as test_client:
        yield test_client
    service.reset_model()
