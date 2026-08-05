"""Tests for the FastAPI inference service."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import service
from src.api.schemas import SubscriberFeaturesRequest
from src.config import settings
from tests.conftest import HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER

VALID_RISK_LEVELS = {"low", "medium", "high"}


# --------------------------------------------------------------------------- #
# Operational endpoints
# --------------------------------------------------------------------------- #


def test_health_returns_ok(client: TestClient) -> None:
    """``/health`` is the liveness probe and must stay trivial."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_loaded_model(client: TestClient) -> None:
    """``/ready`` confirms an artifact is in memory."""
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_model_info_describes_artifact(client: TestClient) -> None:
    """``/model-info`` surfaces the serving contract and threshold."""
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"]
    assert 0.0 < payload["decision_threshold"] < 1.0
    assert "tenure_days" in payload["required_input_columns"]


def test_openapi_docs_are_served(client: TestClient) -> None:
    """Swagger UI is available for manual exploration."""
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


def test_predict_returns_expected_payload(client: TestClient) -> None:
    """A valid request returns a well-formed prediction."""
    response = client.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    assert response.status_code == 200

    payload = response.json()
    assert {"dropout_probability", "predicted_label", "explanation"} <= set(payload)

    probability = payload["dropout_probability"]
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
    assert payload["predicted_label"] in (0, 1)
    assert payload["risk_level"] in VALID_RISK_LEVELS
    assert payload["explanation"].strip()


def test_predict_label_agrees_with_threshold(client: TestClient) -> None:
    """The hard label is exactly the probability compared to the threshold."""
    payload = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    expected = int(payload["dropout_probability"] >= payload["threshold"])
    assert payload["predicted_label"] == expected


def test_high_risk_scores_above_low_risk(client: TestClient) -> None:
    """A distressed subscriber must score higher than a thriving one."""
    high = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    low = client.post("/predict", json=LOW_RISK_SUBSCRIBER).json()
    assert high["dropout_probability"] > low["dropout_probability"]


def test_explanation_mentions_risk_drivers(client: TestClient) -> None:
    """The rule-based explanation names the signals that fired."""
    payload = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    explanation = payload["explanation"].lower()
    assert "inactive" in explanation
    assert payload["top_risk_factors"]
    assert len(payload["top_risk_factors"]) <= 3


def test_explanation_for_healthy_subscriber(client: TestClient) -> None:
    """With no risk signals the explanation falls back to retention signals."""
    payload = client.post("/predict", json=LOW_RISK_SUBSCRIBER).json()
    assert payload["top_risk_factors"] == []
    assert payload["explanation"].strip()


def test_predict_accepts_mixed_case_plan_type(client: TestClient) -> None:
    """Plan names are normalised before validation."""
    request = copy.deepcopy(LOW_RISK_SUBSCRIBER)
    request["plan_type"] = "  Premium  "
    assert client.post("/predict", json=request).status_code == 200


def test_predict_is_deterministic(client: TestClient) -> None:
    """The same payload always yields the same probability."""
    first = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    second = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    assert first["dropout_probability"] == second["dropout_probability"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenure_days", -1),
        ("monthly_fee", -5.0),
        ("avg_session_count_last_30d", -0.5),
        ("support_tickets_last_90d", -2),
        ("plan_type", "enterprise"),
        ("is_auto_renew_enabled", "maybe"),
    ],
)
def test_predict_rejects_invalid_values(client: TestClient, field: str, value: Any) -> None:
    """Out-of-domain values are refused with 422, not scored."""
    request = copy.deepcopy(HIGH_RISK_SUBSCRIBER)
    request[field] = value
    assert client.post("/predict", json=request).status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    """Every feature is mandatory."""
    request = copy.deepcopy(HIGH_RISK_SUBSCRIBER)
    del request["monthly_fee"]
    response = client.post("/predict", json=request)
    assert response.status_code == 422
    assert "monthly_fee" in response.text


def test_predict_rejects_activity_older_than_tenure(client: TestClient) -> None:
    """A subscriber cannot be inactive longer than they have existed."""
    request = copy.deepcopy(HIGH_RISK_SUBSCRIBER)
    request["tenure_days"] = 10
    request["last_activity_days_ago"] = 400
    response = client.post("/predict", json=request)
    assert response.status_code == 422
    assert "last_activity_days_ago" in response.text


def test_predict_rejects_empty_body(client: TestClient) -> None:
    """An empty payload is a validation error."""
    assert client.post("/predict", json={}).status_code == 422


# --------------------------------------------------------------------------- #
# /predict/batch
# --------------------------------------------------------------------------- #


def test_batch_prediction_preserves_order(client: TestClient) -> None:
    """Batch results line up with the submitted subscribers."""
    body = {"subscribers": [HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER]}
    response = client.post("/predict/batch", json=body)
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
    assert (
        payload["predictions"][0]["dropout_probability"]
        > payload["predictions"][1]["dropout_probability"]
    )


def test_batch_matches_single_prediction(client: TestClient) -> None:
    """Batch and single scoring paths agree."""
    single = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    batch = client.post("/predict/batch", json={"subscribers": [HIGH_RISK_SUBSCRIBER]}).json()
    assert batch["predictions"][0]["dropout_probability"] == single["dropout_probability"]


def test_batch_rejects_empty_list(client: TestClient) -> None:
    """At least one subscriber is required."""
    assert client.post("/predict/batch", json={"subscribers": []}).status_code == 422


# --------------------------------------------------------------------------- #
# Degraded mode (no artifact on disk)
# --------------------------------------------------------------------------- #


def test_health_ok_without_model(client_without_model: TestClient) -> None:
    """Liveness stays green so the container is not restart-looped."""
    assert client_without_model.get("/health").json() == {"status": "ok"}


def test_ready_reports_degraded_without_model(client_without_model: TestClient) -> None:
    """Readiness makes the missing artifact explicit."""
    payload = client_without_model.get("/ready").json()
    assert payload["status"] == "degraded"
    assert payload["model_loaded"] is False
    assert payload["detail"]


def test_predict_returns_503_without_model(client_without_model: TestClient) -> None:
    """Prediction fails with a clear 503 rather than a 500."""
    response = client_without_model.post("/predict", json=HIGH_RISK_SUBSCRIBER)
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()


def test_model_info_returns_503_without_model(client_without_model: TestClient) -> None:
    """Metadata is unavailable without an artifact."""
    assert client_without_model.get("/model-info").status_code == 503


# --------------------------------------------------------------------------- #
# Service-level helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.02, "low"), (0.35, "low"), (0.5, "medium"), (0.64, "medium"), (0.65, "high"), (0.99, "high")],
)
def test_classify_risk_level(probability: float, expected: str) -> None:
    """Risk bands follow the documented cut-offs."""
    assert service.classify_risk_level(probability) == expected


def test_collect_risk_factors_flags_known_problems() -> None:
    """Each rule contributes a readable phrase."""
    factors = service.collect_risk_factors(HIGH_RISK_SUBSCRIBER)
    joined = " ".join(factors).lower()
    assert "inactive" in joined
    assert "payment failures" in joined
    assert "auto-renew disabled" in joined


def test_collect_risk_factors_quiet_for_healthy_subscriber() -> None:
    """A thriving subscriber trips no rules."""
    assert service.collect_risk_factors(LOW_RISK_SUBSCRIBER) == []


def test_predict_batch_with_empty_input_short_circuits() -> None:
    """An empty batch returns immediately without touching the model."""
    assert service.predict_batch([]) == []


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


def _dashboard_html() -> str:
    """The dashboard source, read straight from the configured location."""
    return settings.DASHBOARD_PATH.read_text(encoding="utf-8")


def _extract_presets() -> dict[str, dict[str, Any]]:
    """Pull the ``PRESETS`` object out of the page's inline script.

    The presets are a JavaScript object literal, so unquoted keys and trailing
    commas are normalised before parsing them as JSON.
    """
    match = re.search(r"const PRESETS = (\{.*?\n  \});", _dashboard_html(), re.DOTALL)
    assert match, "PRESETS object not found in the dashboard script"

    literal = re.sub(r"(\w+):", r'"\1":', match.group(1))  # quote the keys
    literal = re.sub(r",(\s*[}\]])", r"\1", literal)  # drop trailing commas
    return json.loads(literal)


def test_dashboard_is_served_at_root(client: TestClient) -> None:
    """``GET /`` returns the HTML page rather than a JSON payload."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Subscriber Dropout Detection" in response.text


def test_dashboard_is_served_without_a_model(client_without_model: TestClient) -> None:
    """The page is static, so it loads even when the artifact is missing.

    That is what lets the UI *show* the degraded state instead of the user
    meeting a blank error page.
    """
    assert client_without_model.get("/").status_code == 200


def test_static_assets_are_mounted(client: TestClient) -> None:
    """The static mount serves the same file for direct asset requests."""
    assert client.get("/static/index.html").status_code == 200


def test_dashboard_is_excluded_from_the_openapi_schema(client: TestClient) -> None:
    """The UI route must not pollute the machine-readable API contract."""
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_dashboard_form_fields_match_the_request_schema() -> None:
    """Every schema field has an input, and no input is invented.

    This is the drift guard: renaming a field in ``schemas.py`` without
    updating the form fails here instead of silently breaking the page.
    """
    field_ids = set(re.findall(r'<(?:input|select) type="[^"]*" id="(\w+)"', _dashboard_html()))
    field_ids |= set(re.findall(r'<select id="(\w+)"', _dashboard_html()))

    assert field_ids == set(SubscriberFeaturesRequest.model_fields)


def test_dashboard_calls_only_real_endpoints(client: TestClient) -> None:
    """Every path the page fetches is actually routed by this app."""
    fetched = set(re.findall(r'fetch\("(/[^"]*)"', _dashboard_html()))
    assert fetched, "the dashboard should call the API"

    routed = {route.path for route in client.app.routes}
    assert fetched <= routed


@pytest.mark.parametrize("preset_name", ["at_risk", "healthy", "borderline"])
def test_dashboard_presets_score_successfully(client: TestClient, preset_name: str) -> None:
    """Each one-click preset is a payload the API accepts and can score.

    Covers the primary user journey end to end: load a preset, hit score.
    """
    response = client.post("/predict", json=_extract_presets()[preset_name])
    assert response.status_code == 200, response.text

    payload = response.json()
    assert 0.0 <= payload["dropout_probability"] <= 1.0
    assert payload["risk_level"] in VALID_RISK_LEVELS


def test_dashboard_at_risk_preset_outranks_the_healthy_one(client: TestClient) -> None:
    """The demo presets must actually demonstrate a difference in risk."""
    presets = _extract_presets()
    scored = {
        name: client.post("/predict", json=presets[name]).json()["dropout_probability"]
        for name in ("at_risk", "healthy")
    }
    assert scored["at_risk"] > scored["healthy"]
