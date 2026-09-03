"""Tests for champion/challenger shadow scoring.

The safety tests are the ones that matter most here. Shadow scoring runs an
unproven model against real production traffic; if it can affect a served
response in any way, the feature is worse than not having it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import service
from src.api.shadow import ShadowComparison, ShadowTracker, get_shadow_tracker
from src.config import settings
from tests.conftest import HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER


class _ExplodingModel:
    """A challenger that fails on every request."""

    def predict_proba(self, frame):  # noqa: ANN001, ANN201
        raise RuntimeError("challenger is catastrophically broken")


class _ConstantModel:
    """A challenger that always returns the same probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame):  # noqa: ANN001, ANN201
        import numpy as np

        column = np.full(len(frame), self.probability)
        return np.column_stack([1 - column, column])


@pytest.fixture()
def shadowing_client(client: TestClient) -> TestClient:
    """A client whose loaded model has a distinct, working challenger."""
    loaded = service.get_model()
    loaded.challenger = _ConstantModel(0.99)
    loaded.challenger_threshold = 0.5
    loaded.challenger_version = "test-challenger"
    get_shadow_tracker().reset()
    yield client
    loaded.challenger = None
    get_shadow_tracker().reset()


# --------------------------------------------------------------------------- #
# The safety contract
# --------------------------------------------------------------------------- #


def test_a_broken_challenger_cannot_break_serving(client: TestClient) -> None:
    """The single rule of shadow scoring.

    An unproven model runs against real traffic. If it can fail a request, the
    feature has introduced exactly the risk it exists to remove.
    """
    loaded = service.get_model()
    loaded.challenger = _ExplodingModel()
    loaded.challenger_threshold = 0.5
    loaded.challenger_version = "broken"
    get_shadow_tracker().reset()

    response = client.post("/predict", json=HIGH_RISK_SUBSCRIBER)

    assert response.status_code == 200
    assert 0.0 <= response.json()["dropout_probability"] <= 1.0

    loaded.challenger = None


def test_a_broken_challenger_is_counted_not_swallowed(client: TestClient) -> None:
    """A silent failure would hide the finding that should block promotion."""
    loaded = service.get_model()
    loaded.challenger = _ExplodingModel()
    loaded.challenger_version = "broken"
    get_shadow_tracker().reset()

    client.post("/predict", json=HIGH_RISK_SUBSCRIBER)

    report = client.get("/monitoring/shadow").json()
    assert report["errors_total"] == 1
    assert report["compared_total"] == 0

    loaded.challenger = None


def test_the_served_answer_is_always_the_champions(shadowing_client: TestClient) -> None:
    """The challenger scores everything and decides nothing.

    The stand-in challenger returns 0.99 for every subscriber, so if any of its
    output leaked into a response this would fail on the healthy subscriber.
    """
    response = shadowing_client.post("/predict", json=LOW_RISK_SUBSCRIBER).json()

    assert response["dropout_probability"] < 0.5
    assert response["predicted_label"] == 0
    assert response["threshold"] == pytest.approx(service.get_model().threshold, abs=1e-6)


# --------------------------------------------------------------------------- #
# Comparison mechanics
# --------------------------------------------------------------------------- #


def test_predictions_are_shadow_scored(shadowing_client: TestClient) -> None:
    """Every served request should produce a paired comparison."""
    shadowing_client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    shadowing_client.post("/predict", json=LOW_RISK_SUBSCRIBER)

    report = shadowing_client.get("/monitoring/shadow").json()
    assert report["active"] is True
    assert report["compared_total"] == 2


def test_batches_are_compared_row_by_row(shadowing_client: TestClient) -> None:
    """A batch of five is five comparisons, not one."""
    shadowing_client.post("/predict/batch", json={"subscribers": [LOW_RISK_SUBSCRIBER] * 5})

    assert shadowing_client.get("/monitoring/shadow").json()["compared_total"] == 5


def test_disagreement_is_measured(shadowing_client: TestClient) -> None:
    """The challenger flags everyone; the champion does not flag a healthy one."""
    shadowing_client.post("/predict/batch", json={"subscribers": [LOW_RISK_SUBSCRIBER] * 4})

    report = shadowing_client.get("/monitoring/shadow").json()
    assert report["agreement_rate"] == 0.0
    assert report["challenger_flagged_rate"] == 1.0
    assert report["champion_flagged_rate"] == 0.0
    # The operational headline: promoting would flag everyone instead of no one.
    assert report["flagged_rate_delta"] == 1.0
    assert report["challenger_flags_more"] == 4
    assert report["challenger_flags_fewer"] == 0


def test_shadow_is_idle_without_a_challenger(client: TestClient) -> None:
    """No challenger registered is a normal state, not a degraded one."""
    service.get_model().challenger = None
    report = client.get("/monitoring/shadow").json()

    assert report["active"] is False
    assert "No distinct challenger" in report["detail"]


def test_shadow_reports_no_accuracy_verdict(shadowing_client: TestClient) -> None:
    """The most important thing this endpoint does NOT claim.

    Shadow traffic is unlabelled - nobody has churned yet - so any field
    implying the challenger is more accurate would be a lie. Accuracy comes
    from the labelled holdout the promotion gate uses.
    """
    shadowing_client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    report = shadowing_client.get("/monitoring/shadow").json()

    for field in ("accuracy", "auc", "roc_auc", "pr_auc", "better", "winner", "recommendation"):
        assert field not in report


# --------------------------------------------------------------------------- #
# Evidence sufficiency
# --------------------------------------------------------------------------- #


def test_evidence_is_insufficient_until_enough_traffic(
    shadowing_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """100% agreement over four requests is noise dressed as a result."""
    monkeypatch.setattr(settings, "SHADOW_MIN_COMPARISONS", 100)

    shadowing_client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    report = shadowing_client.get("/monitoring/shadow").json()

    assert report["sufficient_evidence"] is False
    assert report["required"] == 100


def test_evidence_becomes_sufficient_with_enough_traffic(
    shadowing_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the threshold, the comparison is worth acting on."""
    monkeypatch.setattr(settings, "SHADOW_MIN_COMPARISONS", 5)

    shadowing_client.post("/predict/batch", json={"subscribers": [HIGH_RISK_SUBSCRIBER] * 6})
    report = shadowing_client.get("/monitoring/shadow").json()

    assert report["sufficient_evidence"] is True


# --------------------------------------------------------------------------- #
# The tracker in isolation
# --------------------------------------------------------------------------- #


def test_tracker_starts_empty() -> None:
    """An idle tracker reports nulls, not misleading zeros."""
    snapshot = ShadowTracker().snapshot()

    assert snapshot["compared_total"] == 0
    assert snapshot["agreement_rate"] is None
    assert snapshot["flagged_rate_delta"] is None


def test_tracker_computes_agreement_and_divergence() -> None:
    """Hand-checked arithmetic on a known set of pairs."""
    tracker = ShadowTracker(window=100)
    tracker.record(ShadowComparison(0.10, 0, 0.12, 0))  # agree, gap 0.02
    tracker.record(ShadowComparison(0.80, 1, 0.90, 1))  # agree, gap 0.10
    tracker.record(ShadowComparison(0.40, 0, 0.70, 1))  # disagree, gap 0.30

    snapshot = tracker.snapshot()
    assert snapshot["agreement_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert snapshot["mean_absolute_divergence"] == pytest.approx(0.14, abs=1e-4)
    assert snapshot["max_absolute_divergence"] == pytest.approx(0.30, abs=1e-4)
    assert snapshot["challenger_flags_more"] == 1
    assert snapshot["challenger_flags_fewer"] == 0


def test_tracker_window_is_bounded_but_total_is_not() -> None:
    """Memory stays flat under load while the running total stays honest."""
    tracker = ShadowTracker(window=10)
    for _ in range(50):
        tracker.record(ShadowComparison(0.5, 1, 0.5, 1))

    snapshot = tracker.snapshot()
    assert snapshot["window_size"] == 10
    assert snapshot["compared_total"] == 50


def test_comparison_helpers() -> None:
    """`agreed` is about the action taken, not the probability."""
    close_but_disagreeing = ShadowComparison(0.49, 0, 0.51, 1)
    assert close_but_disagreeing.agreed is False
    assert close_but_disagreeing.divergence == pytest.approx(0.02)

    far_but_agreeing = ShadowComparison(0.60, 1, 0.99, 1)
    assert far_but_agreeing.agreed is True


# --------------------------------------------------------------------------- #
# Prometheus wiring
# --------------------------------------------------------------------------- #


def test_shadow_metrics_reach_prometheus(shadowing_client: TestClient) -> None:
    """The comparison has to be visible to the same dashboards as everything else."""
    shadowing_client.post("/predict/batch", json={"subscribers": [LOW_RISK_SUBSCRIBER] * 3})

    body = shadowing_client.get("/metrics/prometheus").text

    assert "subscriber_shadow_comparisons_total" in body
    assert "subscriber_shadow_active 1.0" in body
    assert "subscriber_shadow_divergence_count" in body
    assert "subscriber_shadow_agreement_rate" in body


def test_shadow_errors_are_exposed_as_a_metric(client: TestClient) -> None:
    """A challenger failing on live traffic must be alertable."""
    loaded = service.get_model()
    loaded.challenger = _ExplodingModel()
    loaded.challenger_version = "broken"
    get_shadow_tracker().reset()

    client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    body = client.get("/metrics/prometheus").text

    assert "subscriber_shadow_errors_total" in body
    loaded.challenger = None
