"""Tests for MLflow tracking, the model registry and the promotion gate.

The promotion tests are the ones that matter: a gate that promotes everything
is the same as having no gate, and a gate that promotes nothing is a broken
pipeline that will quietly serve a stale model forever.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.config import settings
from src.registry.promote import PromotionDecision, evaluate_promotion, promote, rollback

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """An isolated SQLite-backed MLflow store for this module.

    A file store cannot host a model registry, so SQLite is the minimum that
    exercises the real registry code paths without running a server.
    """
    workdir: Path = tmp_path_factory.mktemp("mlflow")
    return f"sqlite:///{workdir / 'mlflow.db'}"


@pytest.fixture(autouse=True)
def _isolate_mlflow(tracking_uri: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every test in this module at the throwaway store."""
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)


@pytest.fixture(scope="module")
def holdout() -> tuple[pd.DataFrame, pd.Series]:
    """A small labelled holdout whose signal a real model can find."""
    rng = np.random.default_rng(0)
    n = 400
    dormant = rng.integers(0, 60, size=n)
    frame = pd.DataFrame(
        {
            "tenure_days": rng.integers(30, 900, size=n),
            "plan_type": rng.choice(["basic", "standard", "premium"], size=n),
            "monthly_fee": rng.choice([9.99, 19.99, 29.99], size=n),
            "avg_session_count_last_30d": np.clip(30 - dormant / 2, 0, None).astype(float),
            "last_activity_days_ago": dormant,
            "support_tickets_last_90d": rng.integers(0, 4, size=n),
            "payment_failures_last_6m": rng.integers(0, 3, size=n),
            "discounts_used_last_6m": rng.integers(0, 4, size=n),
            "is_auto_renew_enabled": rng.random(size=n) > 0.4,
        }
    )
    # Dormancy drives the label, so a fitted model beats a constant predictor.
    target = pd.Series((dormant > 25).astype(int), name="dropout")
    frame["last_activity_days_ago"] = frame[["last_activity_days_ago", "tenure_days"]].min(axis=1)
    return frame, target


def _fit(model_params: dict[str, Any], features: pd.DataFrame, target: pd.Series) -> Pipeline:
    from src.models.train import build_model_pipeline

    model = build_model_pipeline(model_params)
    model.fit(features, target)
    return model


def _constant_pipeline(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """A pipeline that ignores its inputs, as a deliberately weak challenger."""
    from src.features.build_features import build_feature_pipeline

    model = Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("classifier", DummyClassifier(strategy="prior")),
        ]
    )
    model.fit(features, target)
    return model


# --------------------------------------------------------------------------- #
# Tracking and registration
# --------------------------------------------------------------------------- #


def test_training_run_is_logged_and_registered(holdout) -> None:
    """A run produces a run id and an immutable registered version."""
    from src.registry import tracking

    features, target = holdout
    model = _fit({"n_estimators": 20, "max_depth": 2}, features, target)

    run_id, version = tracking.log_training_run(
        model=model,
        params={"n_estimators": 20},
        validation_metrics={"f1": 0.5, "confusion_matrix": {"true_positives": 3}},
        test_metrics={"f1": 0.51, "pr_auc": 0.6},
        input_example=features.head(5),
        model_name="test-registration",
    )

    assert run_id
    assert version is not None
    assert int(version.version) >= 1


def test_new_version_is_tagged_challenger(holdout) -> None:
    """Every fresh model arrives as @challenger, never as champion."""
    from src.registry import tracking

    features, target = holdout
    model = _fit({"n_estimators": 15, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.4},
        test_metrics={"f1": 0.4},
        model_name="test-challenger-alias",
    )

    alias = tracking.get_alias_version(settings.CHALLENGER_ALIAS, "test-challenger-alias")
    assert alias is not None
    assert alias.version == version.version
    # Nothing has been promoted, so there must be no champion yet.
    assert tracking.get_alias_version(settings.CHAMPION_ALIAS, "test-challenger-alias") is None


def test_registered_model_round_trips(holdout) -> None:
    """A model loaded back from the registry predicts identically.

    This is the guard on the cloudpickle serialisation choice: the pipeline
    carries a custom FunctionTransformer, which the default format refuses.
    """
    from src.registry import tracking

    features, target = holdout
    model = _fit({"n_estimators": 20, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.5},
        test_metrics={"f1": 0.5},
        model_name="test-roundtrip",
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, version.version, "test-roundtrip")

    reloaded = tracking.load_aliased_model(settings.CHAMPION_ALIAS, "test-roundtrip")
    assert reloaded is not None
    np.testing.assert_allclose(
        model.predict_proba(features)[:, 1], reloaded.predict_proba(features)[:, 1]
    )


def test_missing_alias_returns_none() -> None:
    """Asking for an alias that was never set is not an error."""
    from src.registry import tracking

    assert tracking.get_alias_version("champion", "model-that-does-not-exist") is None
    assert tracking.load_aliased_model("champion", "model-that-does-not-exist") is None


def test_run_history_reports_aliases(holdout) -> None:
    """History shows which version currently holds which alias."""
    from src.registry import tracking

    features, target = holdout
    model = _fit({"n_estimators": 10, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.3},
        test_metrics={"f1": 0.3, "pr_auc": 0.4},
        model_name="test-history",
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, version.version, "test-history")

    history = tracking.run_history("test-history")
    row = history[history["version"] == int(version.version)].iloc[0]
    assert settings.CHAMPION_ALIAS in row["aliases"]
    assert "test_pr_auc" in history.columns


# --------------------------------------------------------------------------- #
# The promotion gate
# --------------------------------------------------------------------------- #


def test_first_model_is_promoted_unopposed(holdout) -> None:
    """With no incumbent there is nothing to beat."""
    features, target = holdout
    model = _fit({"n_estimators": 20, "max_depth": 2}, features, target)

    decision = evaluate_promotion(
        model, features, target, challenger_version="1", model_name="test-gate-first"
    )
    assert decision.promoted
    assert "no incumbent" in decision.reason
    assert decision.champion_score is None
    assert decision.improvement is None


def test_worse_challenger_is_rejected(holdout) -> None:
    """A weaker model must not take production."""
    from src.registry import tracking

    features, target = holdout
    name = "test-gate-worse"

    champion = _fit({"n_estimators": 40, "max_depth": 3}, features, target)
    _, champion_version = tracking.log_training_run(
        model=champion,
        params={},
        validation_metrics={"f1": 0.8},
        test_metrics={"f1": 0.8},
        model_name=name,
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, champion_version.version, name)

    weak = _constant_pipeline(features, target)
    decision = evaluate_promotion(
        weak, features, target, challenger_version="99", model_name=name
    )

    assert not decision.promoted
    assert decision.challenger_score < decision.champion_score
    assert "below required" in decision.reason


def test_identical_challenger_is_rejected_by_the_margin(holdout) -> None:
    """A tie must not churn the registry.

    Without a margin, two equivalent models differ only by noise and roughly
    half of all retrains would promote for no reason.
    """
    from src.registry import tracking

    features, target = holdout
    name = "test-gate-tie"

    model = _fit({"n_estimators": 30, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.7},
        test_metrics={"f1": 0.7},
        model_name=name,
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, version.version, name)

    decision = evaluate_promotion(
        model, features, target, challenger_version="2", model_name=name
    )

    assert not decision.promoted
    assert decision.improvement == pytest.approx(0.0, abs=1e-9)


def test_a_clearly_better_challenger_is_promoted(holdout) -> None:
    """The gate is not simply refusing everything."""
    from src.registry import tracking

    features, target = holdout
    name = "test-gate-better"

    weak = _constant_pipeline(features, target)
    _, weak_version = tracking.log_training_run(
        model=weak,
        params={},
        validation_metrics={"f1": 0.1},
        test_metrics={"f1": 0.1},
        model_name=name,
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, weak_version.version, name)

    strong = _fit({"n_estimators": 60, "max_depth": 3}, features, target)
    decision = evaluate_promotion(
        strong, features, target, challenger_version="42", model_name=name
    )

    assert decision.promoted
    assert decision.improvement > settings.PROMOTION_MIN_IMPROVEMENT


def test_promote_moves_the_alias_only_when_it_should(holdout) -> None:
    """``promote`` acts on the decision rather than second-guessing it."""
    from src.registry import tracking

    features, target = holdout
    name = "test-gate-apply"

    model = _fit({"n_estimators": 20, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.5},
        test_metrics={"f1": 0.5},
        model_name=name,
    )

    promote(
        PromotionDecision(
            promoted=True,
            reason="test",
            metric="pr_auc",
            challenger_score=0.9,
            challenger_version=version.version,
        ),
        model_name=name,
    )
    assert tracking.get_alias_version(settings.CHAMPION_ALIAS, name).version == version.version

    # A rejected decision must leave the incumbent untouched.
    promote(
        PromotionDecision(
            promoted=False,
            reason="worse",
            metric="pr_auc",
            challenger_score=0.1,
            challenger_version="999",
        ),
        model_name=name,
    )
    assert tracking.get_alias_version(settings.CHAMPION_ALIAS, name).version == version.version


def test_rollback_restores_a_previous_version(holdout) -> None:
    """Rollback is cheap because promotion never deleted anything."""
    from src.registry import tracking

    features, target = holdout
    name = "test-rollback"

    first = _fit({"n_estimators": 10, "max_depth": 2}, features, target)
    _, v1 = tracking.log_training_run(
        model=first, params={}, validation_metrics={}, test_metrics={}, model_name=name
    )
    second = _fit({"n_estimators": 30, "max_depth": 3}, features, target)
    _, v2 = tracking.log_training_run(
        model=second, params={}, validation_metrics={}, test_metrics={}, model_name=name
    )

    tracking.set_alias(settings.CHAMPION_ALIAS, v2.version, name)
    assert tracking.get_alias_version(settings.CHAMPION_ALIAS, name).version == v2.version

    rollback(v1.version, model_name=name)
    assert tracking.get_alias_version(settings.CHAMPION_ALIAS, name).version == v1.version


def test_single_class_holdout_raises_rather_than_rejecting(holdout) -> None:
    """A broken evaluation set must be loud, not silently reject everything."""
    features, _ = holdout
    all_zero = pd.Series(np.zeros(len(features), dtype=int))
    model = _fit({"n_estimators": 10, "max_depth": 2}, features, pd.Series([0, 1] * (len(features) // 2)))

    with pytest.raises(ValueError, match="both classes"):
        evaluate_promotion(model, features, all_zero, model_name="test-single-class")


def test_decision_serialises_for_logging() -> None:
    """The verdict is recorded in pipelines, so it must be JSON-friendly."""
    decision = PromotionDecision(
        promoted=False,
        reason="too small",
        metric="pr_auc",
        challenger_score=0.51,
        champion_score=0.509,
        champion_version="3",
        challenger_version="4",
    )
    payload = decision.to_dict()

    assert payload["promoted"] is False
    assert payload["improvement"] == pytest.approx(0.001)
    assert "REJECTED" in decision.summary()


def test_promotion_metric_is_pr_auc_by_default() -> None:
    """PR-AUC, not ROC-AUC: the label is imbalanced and outreach cares about precision."""
    assert settings.PROMOTION_METRIC == "pr_auc"
    assert settings.PROMOTION_MIN_IMPROVEMENT > 0


# --------------------------------------------------------------------------- #
# API service: loading the champion model from the registry
# --------------------------------------------------------------------------- #


def test_load_model_falls_back_to_local_when_no_champion_is_set(
    trained_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'auto' mode must not fail startup just because nothing is promoted yet.

    Points settings.MODEL_PATH at a real, self-contained temp artifact rather
    than whatever happens to be on disk in the working tree - this must pass
    on a fresh clone with no prior training run, not just in this repo.
    """
    from src.api import service

    monkeypatch.setattr(settings, "REGISTERED_MODEL_NAME", "test-service-empty-registry")
    monkeypatch.setattr(settings, "MODEL_SOURCE", "auto")
    monkeypatch.setattr(settings, "MODEL_PATH", trained_model.model_path)
    monkeypatch.setattr(settings, "METADATA_PATH", trained_model.metadata_path)
    monkeypatch.setattr(settings, "REFERENCE_PROFILE_PATH", trained_model.reference_profile_path)
    service.reset_model()

    loaded = service.load_model()
    assert loaded.metadata.get("served_from", "local") == "local"
    assert loaded.threshold == pytest.approx(trained_model.threshold)
    service.reset_model()


def test_load_model_registry_mode_raises_without_a_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'registry' mode is strict: no fallback, no silent local load."""
    from src.api import service

    monkeypatch.setattr(settings, "REGISTERED_MODEL_NAME", "test-service-strict-registry")
    monkeypatch.setattr(settings, "MODEL_SOURCE", "registry")
    service.reset_model()

    with pytest.raises(service.ModelNotLoadedError, match="No @champion alias"):
        service.load_model()
    service.reset_model()


def test_load_model_rejects_an_unknown_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd SDD_MODEL_SOURCE must fail loudly, not silently default somewhere."""
    from src.api import service

    monkeypatch.setattr(settings, "MODEL_SOURCE", "s3")
    service.reset_model()

    with pytest.raises(ValueError, match="Unknown model source"):
        service.load_model()
    service.reset_model()


def test_load_model_prefers_the_registry_when_a_champion_exists(
    holdout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of this wiring: promotion must change what the API serves.

    Trains a real model, promotes it, then asserts the API's own load_model()
    - not a mocked stand-in - resolves it from MLflow rather than from disk.
    """
    from src.api import service
    from src.registry import tracking

    features, target = holdout
    name = "test-service-registry-preferred"

    model = _fit({"n_estimators": 25, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={"decision_threshold": 0.33},
        validation_metrics={"f1": 0.5},
        test_metrics={"f1": 0.5, "pr_auc": 0.4},
        model_name=name,
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, version.version, name)

    monkeypatch.setattr(settings, "REGISTERED_MODEL_NAME", name)
    monkeypatch.setattr(settings, "MODEL_SOURCE", "auto")
    service.reset_model()

    loaded = service.load_model()
    assert loaded.metadata["served_from"] == "registry"
    assert loaded.metadata["registry_version"] == str(version.version)
    # The threshold travelled as a logged run param, not the local default.
    assert loaded.threshold == pytest.approx(0.33)

    predictions = service.predict_one(features.iloc[0].to_dict())
    assert 0.0 <= predictions["dropout_probability"] <= 1.0
    service.reset_model()


def test_model_info_reports_registry_version_over_the_api(
    holdout, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/model-info must let a caller prove which source served the prediction."""
    from fastapi.testclient import TestClient

    from src.api import service
    from src.api.main import app
    from src.registry import tracking

    features, target = holdout
    name = "test-service-model-info-registry"

    model = _fit({"n_estimators": 20, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.5},
        test_metrics={"f1": 0.5, "pr_auc": 0.4},
        model_name=name,
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, version.version, name)

    monkeypatch.setattr(settings, "REGISTERED_MODEL_NAME", name)
    monkeypatch.setattr(settings, "MODEL_SOURCE", "auto")
    service.reset_model()

    with TestClient(app) as client:
        info = client.get("/model-info").json()

    assert info["served_from"] == "registry"
    assert info["registry_version"] == str(version.version)
    service.reset_model()


def test_load_model_local_mode_ignores_a_registered_champion(
    holdout, trained_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'local' mode is an explicit escape hatch back to the original behaviour."""
    from src.api import service
    from src.registry import tracking

    features, target = holdout
    name = "test-service-local-mode-ignores-registry"

    model = _fit({"n_estimators": 15, "max_depth": 2}, features, target)
    _, version = tracking.log_training_run(
        model=model,
        params={},
        validation_metrics={"f1": 0.4},
        test_metrics={"f1": 0.4},
        model_name=name,
    )
    tracking.set_alias(settings.CHAMPION_ALIAS, version.version, name)

    monkeypatch.setattr(settings, "REGISTERED_MODEL_NAME", name)
    monkeypatch.setattr(settings, "MODEL_SOURCE", "local")
    monkeypatch.setattr(settings, "MODEL_PATH", trained_model.model_path)
    monkeypatch.setattr(settings, "METADATA_PATH", trained_model.metadata_path)
    monkeypatch.setattr(settings, "REFERENCE_PROFILE_PATH", trained_model.reference_profile_path)
    service.reset_model()

    loaded = service.load_model()
    assert loaded.metadata.get("served_from", "local") == "local"
    assert "registry_version" not in loaded.metadata
    service.reset_model()
