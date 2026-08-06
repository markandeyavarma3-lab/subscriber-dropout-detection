"""Tests for reference profiling, drift detection and live metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.monitoring import drift as drift_module
from src.monitoring.profile import (
    build_reference_profile,
    category_proportions,
    load_reference_profile,
    numeric_bin_edges,
    proportions_in_bins,
    save_reference_profile,
)
from src.monitoring.tracker import PredictionTracker
from tests.conftest import HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER

# --------------------------------------------------------------------------- #
# Population Stability Index
# --------------------------------------------------------------------------- #


def test_psi_is_zero_for_identical_distributions() -> None:
    """The whole metric hangs off this: no movement, no signal."""
    reference = [0.2, 0.3, 0.5]
    assert drift_module.population_stability_index(reference, reference) == pytest.approx(0.0)


def test_psi_grows_as_distributions_diverge() -> None:
    """A bigger shift must score higher than a smaller one."""
    reference = [0.25, 0.25, 0.25, 0.25]
    small = drift_module.population_stability_index(reference, [0.3, 0.25, 0.25, 0.2])
    large = drift_module.population_stability_index(reference, [0.7, 0.1, 0.1, 0.1])
    assert 0 < small < large


def test_psi_is_symmetric() -> None:
    """PSI does not care which distribution is called the reference."""
    a, b = [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]
    assert drift_module.population_stability_index(a, b) == pytest.approx(
        drift_module.population_stability_index(b, a)
    )


def test_psi_stays_finite_when_a_bin_empties() -> None:
    """An empty bin must not send the score to infinity."""
    psi = drift_module.population_stability_index([0.5, 0.5], [1.0, 0.0])
    assert np.isfinite(psi)
    assert psi > 0


def test_psi_returns_zero_for_mismatched_bin_counts() -> None:
    """Comparing incompatible binnings is meaningless, not an exception."""
    assert drift_module.population_stability_index([0.5, 0.5], [0.3, 0.3, 0.4]) == 0.0


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.0, "stable"), (0.09, "stable"), (0.1, "moderate"), (0.24, "moderate"),
     (0.25, "significant"), (3.0, "significant")],
)
def test_classify_drift_follows_the_configured_cutoffs(psi: float, expected: str) -> None:
    """Verdicts follow the documented PSI bands."""
    assert drift_module.classify_drift(psi) == expected


# --------------------------------------------------------------------------- #
# Reference profile
# --------------------------------------------------------------------------- #


def test_quantile_bins_split_the_reference_evenly(sample_features: pd.DataFrame) -> None:
    """Quantile binning should give each bin a similar share."""
    edges = numeric_bin_edges(sample_features["tenure_days"])
    proportions = proportions_in_bins(sample_features["tenure_days"], edges)

    assert sum(proportions) == pytest.approx(1.0)
    assert all(0.03 < share < 0.25 for share in proportions)


def test_bin_edges_are_strictly_increasing(sample_features: pd.DataFrame) -> None:
    """Zero-width bins would make the histogram meaningless."""
    edges = numeric_bin_edges(sample_features["monthly_fee"])
    # Deliberately ragged: pairing each edge with its successor.
    assert all(later > earlier for earlier, later in zip(edges, edges[1:], strict=False))


def test_constant_column_still_produces_a_usable_bin() -> None:
    """A column with no variance must not crash the profiler."""
    edges = numeric_bin_edges(pd.Series([5.0] * 50))
    assert len(edges) == 2
    assert proportions_in_bins(pd.Series([5.0] * 10), edges) == pytest.approx([1.0])


def test_empty_column_is_handled() -> None:
    """No rows means no distribution, not an exception."""
    assert numeric_bin_edges(pd.Series([], dtype=float)) == [0.0, 1.0]


def test_values_beyond_the_reference_range_are_clipped_in() -> None:
    """Out-of-range live values are evidence, not noise to discard."""
    edges = [0.0, 1.0, 2.0]
    # Everything sits far above the top edge, so it all lands in the last bin.
    assert proportions_in_bins(pd.Series([99.0, 100.0]), edges) == pytest.approx([0.0, 1.0])


def test_category_proportions_sum_to_one(sample_features: pd.DataFrame) -> None:
    """Discrete shares are relative frequencies."""
    shares = category_proportions(sample_features["plan_type"])
    assert sum(shares.values()) == pytest.approx(1.0)


def test_profile_covers_every_raw_input_column(sample_features: pd.DataFrame) -> None:
    """The baseline must describe everything a caller can send."""
    profile = build_reference_profile(sample_features)
    covered = set(profile["numeric"]) | set(profile["discrete"])
    assert covered == set(settings_required_columns())


def settings_required_columns() -> list[str]:
    """The raw serving contract, imported lazily to keep the import list short."""
    from src.features.build_features import REQUIRED_INPUT_COLUMNS

    return REQUIRED_INPUT_COLUMNS


def test_profile_records_prediction_distribution(sample_features: pd.DataFrame) -> None:
    """Output drift needs the model's own score distribution stored too."""
    probabilities = np.random.default_rng(0).uniform(size=len(sample_features))
    profile = build_reference_profile(sample_features, probabilities=probabilities)

    assert "prediction" in profile
    assert sum(profile["prediction"]["proportions"]) == pytest.approx(1.0)
    assert 0.0 <= profile["prediction"]["mean"] <= 1.0


def test_profile_round_trips_through_disk(sample_features: pd.DataFrame, tmp_path) -> None:
    """The profile is JSON-serialisable and survives a save/load cycle."""
    profile = build_reference_profile(sample_features)
    path = save_reference_profile(profile, tmp_path / "reference_profile.json")
    assert load_reference_profile(path) == profile


def test_missing_profile_loads_as_none(tmp_path) -> None:
    """A model without a baseline degrades, it does not explode."""
    assert load_reference_profile(tmp_path / "absent.json") is None


def test_training_writes_a_reference_profile(reference_profile: dict[str, Any]) -> None:
    """Every training run ships its own drift baseline."""
    assert reference_profile["n_reference_rows"] > 0
    assert reference_profile["numeric"]
    assert "prediction" in reference_profile


# --------------------------------------------------------------------------- #
# Drift detection
# --------------------------------------------------------------------------- #


def test_reference_data_shows_no_drift_against_itself(sample_features: pd.DataFrame) -> None:
    """The critical false-positive check: identical data must read stable."""
    profile = build_reference_profile(sample_features)
    report = drift_module.detect_drift(sample_features, profile)

    assert report["overall_verdict"] == "stable"
    assert report["drifted_features"] == []


def test_shifted_feature_is_detected_and_named(sample_features: pd.DataFrame) -> None:
    """A real shift is caught, and attributed to the column that moved."""
    profile = build_reference_profile(sample_features)
    drifted = sample_features.copy()
    drifted["last_activity_days_ago"] = drifted["last_activity_days_ago"] * 5 + 30

    report = drift_module.detect_drift(drifted, profile)

    assert report["overall_verdict"] == "significant"
    assert "last_activity_days_ago" in report["drifted_features"]
    # Untouched columns must not be dragged along with it.
    assert "monthly_fee" not in report["drifted_features"]


def test_unseen_category_is_reported(sample_features: pd.DataFrame) -> None:
    """A brand-new plan tier is drift, and is named explicitly."""
    profile = build_reference_profile(sample_features)
    drifted = sample_features.copy()
    drifted.loc[drifted.index[:100], "plan_type"] = "enterprise"

    report = drift_module.detect_drift(drifted, profile)
    plan = next(item for item in report["features"] if item["feature"] == "plan_type")

    assert plan["verdict"] != "stable"
    assert plan["unseen_categories"] == ["enterprise"]


def test_features_are_ranked_worst_first(sample_features: pd.DataFrame) -> None:
    """The report leads with whatever moved most."""
    profile = build_reference_profile(sample_features)
    drifted = sample_features.copy()
    drifted["tenure_days"] = drifted["tenure_days"] * 10

    report = drift_module.detect_drift(drifted, profile)
    scores = [item["psi"] for item in report["features"]]

    assert scores == sorted(scores, reverse=True)
    assert report["features"][0]["feature"] == "tenure_days"


def test_prediction_drift_is_reported_when_scores_are_supplied(
    sample_features: pd.DataFrame,
) -> None:
    """Output drift is scored separately from input drift."""
    rng = np.random.default_rng(3)
    profile = build_reference_profile(
        sample_features, probabilities=rng.uniform(0.0, 0.3, size=len(sample_features))
    )
    # Scores now concentrated at the top of the range instead of the bottom.
    report = drift_module.detect_drift(
        sample_features, profile, probabilities=rng.uniform(0.7, 1.0, size=len(sample_features))
    )

    assert report["prediction"]["verdict"] == "significant"
    assert "dropout_probability" in report["drifted_features"]


def test_small_batches_are_flagged_as_insufficient(sample_features: pd.DataFrame) -> None:
    """PSI on a handful of rows is noise; the caller is told so."""
    profile = build_reference_profile(sample_features)

    tiny = drift_module.detect_drift(sample_features.head(5), profile)
    assert tiny["sufficient_sample"] is False
    assert tiny["n_samples"] == 5

    plenty = drift_module.detect_drift(sample_features, profile)
    assert plenty["sufficient_sample"] is True


# --------------------------------------------------------------------------- #
# Prediction tracker
# --------------------------------------------------------------------------- #


def test_tracker_starts_empty() -> None:
    """An idle service reports zeros rather than nulls."""
    snapshot = PredictionTracker().snapshot()
    assert snapshot["served_total"] == 0
    assert snapshot["window_size"] == 0
    assert snapshot["flagged_rate"] == 0.0


def test_tracker_summarises_recorded_predictions() -> None:
    """Means, percentiles and band counts reflect what was recorded."""
    tracker = PredictionTracker(window=100)
    for probability, label, level in [(0.1, 0, "low"), (0.9, 1, "high"), (0.5, 1, "medium")]:
        tracker.record(probability, label, level)

    snapshot = tracker.snapshot()
    assert snapshot["served_total"] == 3
    assert snapshot["flagged_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert snapshot["probability"]["mean"] == pytest.approx(0.5, abs=1e-4)
    assert snapshot["risk_levels"] == {"low": 1, "medium": 1, "high": 1}


def test_tracker_window_is_bounded_but_total_is_not() -> None:
    """Memory stays flat under load while the running total stays honest."""
    tracker = PredictionTracker(window=10)
    for _ in range(50):
        tracker.record(0.5, 1, "medium")

    snapshot = tracker.snapshot()
    assert snapshot["window_size"] == 10
    assert snapshot["served_total"] == 50


def test_tracker_reset_clears_everything() -> None:
    """Reset returns the tracker to its initial state."""
    tracker = PredictionTracker()
    tracker.record(0.5, 1, "medium")
    tracker.reset()
    assert tracker.snapshot()["served_total"] == 0


# --------------------------------------------------------------------------- #
# Monitoring endpoints
# --------------------------------------------------------------------------- #


def test_metrics_is_served_without_a_model(client_without_model: TestClient) -> None:
    """A monitoring endpoint must not fail when the thing it watches is down."""
    response = client_without_model.get("/metrics")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_metrics_counts_served_predictions(client: TestClient) -> None:
    """Predictions made through the API show up in /metrics."""
    assert client.get("/metrics").json()["served_total"] == 0

    client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    client.post("/predict", json=LOW_RISK_SUBSCRIBER)

    payload = client.get("/metrics").json()
    assert payload["served_total"] == 2
    assert payload["window_size"] == 2
    assert 0.0 <= payload["flagged_rate"] <= 1.0
    assert payload["reference"] is not None


def test_metrics_counts_batch_predictions_individually(client: TestClient) -> None:
    """A batch of three counts as three, not one."""
    client.post("/predict/batch", json={"subscribers": [HIGH_RISK_SUBSCRIBER] * 3})
    assert client.get("/metrics").json()["served_total"] == 3


def test_metrics_reports_shift_against_the_training_mean(client: TestClient) -> None:
    """The headline alerting number is present once traffic has been seen."""
    client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    payload = client.get("/metrics").json()
    assert payload["probability_mean_shift"] is not None


def test_drift_endpoint_reports_stable_for_training_like_traffic(
    client: TestClient, sample_frame: pd.DataFrame
) -> None:
    """Traffic resembling training data must not raise a false alarm."""
    records = sample_frame.drop(columns=["subscriber_id", "dropout"]).head(300)
    response = client.post("/monitoring/drift", json={"subscribers": records.to_dict("records")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_samples"] == 300
    assert payload["sufficient_sample"] is True
    assert payload["overall_verdict"] in {"stable", "moderate"}


def test_drift_endpoint_detects_a_shifted_population(
    client: TestClient, sample_frame: pd.DataFrame
) -> None:
    """A dormant population is reported as drifted, with the cause named."""
    records = sample_frame.drop(columns=["subscriber_id", "dropout"]).head(300).copy()
    records["avg_session_count_last_30d"] = 0.0

    response = client.post("/monitoring/drift", json={"subscribers": records.to_dict("records")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_verdict"] == "significant"
    assert "avg_session_count_last_30d" in payload["drifted_features"]


def test_drift_endpoint_rejects_an_empty_batch(client: TestClient) -> None:
    """There is nothing to compare against an empty sample."""
    assert client.post("/monitoring/drift", json={"subscribers": []}).status_code == 422


def test_drift_endpoint_returns_503_without_a_model(client_without_model: TestClient) -> None:
    """Drift needs a model to score the batch with."""
    response = client_without_model.post(
        "/monitoring/drift", json={"subscribers": [HIGH_RISK_SUBSCRIBER]}
    )
    assert response.status_code == 503


def test_drift_endpoint_returns_503_without_a_baseline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older artifact with no profile fails clearly, not with a 500."""
    from src.api import service

    monkeypatch.setattr(service.get_model(), "reference_profile", None)
    response = client.post("/monitoring/drift", json={"subscribers": [HIGH_RISK_SUBSCRIBER]})

    assert response.status_code == 503
    assert "reference_profile" in response.json()["detail"]


def test_monitoring_endpoints_are_documented(client: TestClient) -> None:
    """Both endpoints belong in the public OpenAPI contract."""
    paths = client.get("/openapi.json").json()["paths"]
    assert "/metrics" in paths
    assert "/monitoring/drift" in paths


def test_psi_thresholds_come_from_settings() -> None:
    """The cut-offs are configurable, not buried in the drift module."""
    assert settings.PSI_MODERATE < settings.PSI_SIGNIFICANT
