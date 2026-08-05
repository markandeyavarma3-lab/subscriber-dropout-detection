"""Smoke tests for the training pipeline and its artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.data.generate import generate_subscribers, write_dataset
from src.data.loader import load_raw_data, split_data, split_features_target
from src.models.evaluate import compute_metrics
from src.models.train import (
    TrainingResult,
    build_model_pipeline,
    parse_args,
    run_training,
    top_feature_importances,
    tune_decision_threshold,
)

PROBABILITY_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_generated_dataset_has_expected_schema() -> None:
    """The generator produces every documented column and no nulls."""
    frame = generate_subscribers(n_subscribers=200, seed=1)
    expected = {
        "subscriber_id",
        "tenure_days",
        "plan_type",
        "monthly_fee",
        "avg_session_count_last_30d",
        "last_activity_days_ago",
        "support_tickets_last_90d",
        "payment_failures_last_6m",
        "discounts_used_last_6m",
        "is_auto_renew_enabled",
        "dropout",
    }
    assert set(frame.columns) == expected
    assert len(frame) == 200
    assert not frame.isna().any().any()


def test_generated_dataset_is_reproducible() -> None:
    """The same seed yields the same table."""
    first = generate_subscribers(n_subscribers=100, seed=99)
    second = generate_subscribers(n_subscribers=100, seed=99)
    pd.testing.assert_frame_equal(first, second)


def test_generated_dataset_is_plausible() -> None:
    """Both classes appear, and no impossible values are emitted."""
    frame = generate_subscribers(n_subscribers=2000, seed=3)
    assert set(frame["dropout"].unique()) == {0, 1}
    assert 0.05 < frame["dropout"].mean() < 0.60
    assert (frame["last_activity_days_ago"] <= frame["tenure_days"]).all()
    assert (frame["monthly_fee"] > 0).all()
    assert (frame["avg_session_count_last_30d"] >= 0).all()


def test_split_data_is_stratified_and_complete(tmp_path: Path) -> None:
    """Splits partition the data and preserve the class balance."""
    path = write_dataset(output_path=tmp_path / "subs.csv", n_subscribers=1000, seed=5)
    frame = load_raw_data(path)
    splits = split_data(frame, test_size=0.2, validation_size=0.2, seed=5)

    total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
    assert total == len(frame)
    assert len(splits.X_test) == pytest.approx(200, abs=2)
    assert len(splits.X_val) == pytest.approx(200, abs=2)

    overall_rate = frame["dropout"].mean()
    for target in (splits.y_train, splits.y_val, splits.y_test):
        assert target.mean() == pytest.approx(overall_rate, abs=0.05)


def test_split_features_target_drops_identifier() -> None:
    """The subscriber ID must never reach the model."""
    frame = generate_subscribers(n_subscribers=50, seed=2)
    features, target = split_features_target(frame)
    assert settings.ID_COLUMN not in features.columns
    assert settings.TARGET_COLUMN not in features.columns
    assert len(target) == 50


# --------------------------------------------------------------------------- #
# Pipeline construction
# --------------------------------------------------------------------------- #


def test_build_model_pipeline_structure() -> None:
    """The artifact bundles preprocessing and the classifier in one object."""
    pipeline = build_model_pipeline({"n_estimators": 5})
    assert list(pipeline.named_steps) == ["features", "classifier"]
    assert pipeline.named_steps["classifier"].n_estimators == 5


def test_tune_decision_threshold_finds_separating_cutoff() -> None:
    """On a cleanly separable split the tuner finds a perfect cut-off."""
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.08, 0.12, 0.80, 0.90, 0.95])
    threshold, best_f1 = tune_decision_threshold(y_true, y_proba)
    assert best_f1 == pytest.approx(1.0)
    # The returned threshold must actually deliver the reported score.
    assert list((y_proba >= threshold).astype(int)) == list(y_true)


def test_tuned_threshold_reproduces_reported_f1() -> None:
    """The persisted threshold and its reported F1 must never disagree."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y_true = pd.Series(rng.integers(0, 2, size=200))
    y_proba = rng.random(200)
    threshold, best_f1 = tune_decision_threshold(y_true, y_proba)
    recomputed = f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
    assert recomputed == pytest.approx(best_f1, abs=1e-4)


# --------------------------------------------------------------------------- #
# End-to-end training
# --------------------------------------------------------------------------- #


def test_training_writes_all_artifacts(trained_model: TrainingResult) -> None:
    """A training run leaves the model and both JSON reports on disk."""
    assert trained_model.model_path.exists()
    assert trained_model.model_path.name == "model.joblib"
    assert trained_model.model_path.stat().st_size > 0
    assert trained_model.metrics_path.exists()
    assert trained_model.metadata_path.exists()


def test_training_metrics_are_valid(trained_model: TrainingResult) -> None:
    """All reported metrics are probabilities in the unit interval."""
    for metrics in (trained_model.validation_metrics, trained_model.test_metrics):
        for name in PROBABILITY_METRICS:
            value = metrics[name]
            assert value is not None, f"{name} was not computed"
            assert 0.0 <= value <= 1.0, f"{name}={value} out of range"


def test_model_beats_random_guessing(trained_model: TrainingResult) -> None:
    """Even the tiny test model must learn a real signal."""
    assert trained_model.validation_metrics["roc_auc"] > 0.65
    assert trained_model.test_metrics["roc_auc"] > 0.65


def test_metrics_json_is_readable(trained_model: TrainingResult) -> None:
    """``metrics.json`` is valid JSON with the documented sections."""
    report = json.loads(trained_model.metrics_path.read_text())
    assert {"validation", "test", "dataset", "decision_threshold"} <= set(report)
    assert report["dataset"]["rows"] == 600
    assert report["validation"]["confusion_matrix"]["true_positives"] >= 0


def test_metadata_json_records_serving_contract(trained_model: TrainingResult) -> None:
    """Metadata carries what the API needs to serve the artifact correctly."""
    metadata = json.loads(trained_model.metadata_path.read_text())
    assert 0.0 < metadata["decision_threshold"] < 1.0
    assert metadata["required_input_columns"]
    assert "scikit_learn" in metadata["library_versions"]


def test_threshold_is_a_valid_probability(trained_model: TrainingResult) -> None:
    """The tuned threshold stays inside the searched grid."""
    assert 0.05 <= trained_model.threshold <= 0.95


def test_saved_model_round_trips_and_predicts(trained_model: TrainingResult) -> None:
    """The persisted artifact loads and scores raw, un-engineered columns."""
    reloaded = joblib.load(trained_model.model_path)
    raw = generate_subscribers(n_subscribers=10, seed=77).drop(
        columns=["subscriber_id", "dropout"]
    )
    probabilities = reloaded.predict_proba(raw)[:, 1]

    assert probabilities.shape == (10,)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_training_respects_custom_artifacts_dir(tmp_path: Path) -> None:
    """Artifacts are written where the caller asks, not to the default path."""
    data_path = write_dataset(output_path=tmp_path / "subs.csv", n_subscribers=300, seed=11)
    artifacts_dir = tmp_path / "custom_artifacts"

    result = run_training(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        model_params={"n_estimators": 10, "max_depth": 2},
        persist_test_split=False,
        verbose=False,
    )

    assert result.model_path == artifacts_dir / "model.joblib"
    assert result.model_path.exists()
    assert result.model_path != settings.ARTIFACTS_DIR / "model.joblib"


def test_test_split_follows_custom_artifacts_dir(tmp_path: Path) -> None:
    """A redirected run keeps its test split with its artifacts.

    Regression guard: writing to the default processed path would let an
    isolated experiment silently overwrite the project's own test split.
    """
    data_path = write_dataset(output_path=tmp_path / "subs.csv", n_subscribers=300, seed=17)
    artifacts_dir = tmp_path / "run_artifacts"

    run_training(
        data_path=data_path,
        artifacts_dir=artifacts_dir,
        model_params={"n_estimators": 5, "max_depth": 2},
        persist_test_split=True,
        verbose=False,
    )

    assert (artifacts_dir / "test.csv").exists()
    written = pd.read_csv(artifacts_dir / "test.csv")
    assert settings.TARGET_COLUMN in written.columns
    assert settings.ID_COLUMN not in written.columns


def test_training_generates_missing_dataset(tmp_path: Path) -> None:
    """A missing CSV is generated rather than raising, so a clone just works."""
    data_path = tmp_path / "nested" / "subscribers.csv"
    assert not data_path.exists()

    run_training(
        data_path=data_path,
        artifacts_dir=tmp_path / "artifacts",
        model_params={"n_estimators": 5, "max_depth": 2},
        n_subscribers=250,
        persist_test_split=False,
        verbose=False,
    )

    assert data_path.exists()
    assert len(pd.read_csv(data_path)) == 250


def test_feature_importances_are_reported(trained_model: TrainingResult) -> None:
    """Importances map onto named features and are ordered descending."""
    importances = top_feature_importances(trained_model.model, limit=5)
    assert importances
    assert all(isinstance(item["feature"], str) for item in importances)
    values = [item["importance"] for item in importances]
    assert values == sorted(values, reverse=True)


def test_compute_metrics_handles_single_class_split() -> None:
    """ROC-AUC is undefined with one class present and reports as ``None``."""
    metrics = compute_metrics(np.zeros(5, dtype=int), np.array([0.1] * 5), threshold=0.5)
    assert metrics["roc_auc"] is None
    assert metrics["accuracy"] == 1.0


def test_parse_args_defaults_and_overrides() -> None:
    """The CLI exposes the knobs documented in the README."""
    defaults = parse_args([])
    assert defaults.n_estimators is None
    assert defaults.seed == settings.RANDOM_SEED

    overridden = parse_args(["--n-estimators", "42", "--max-depth", "4", "--no-tune-threshold"])
    assert overridden.n_estimators == 42
    assert overridden.max_depth == 4
    assert overridden.no_tune_threshold is True
