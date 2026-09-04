"""Tests for SHAP attribution and its fallback.

The important test here is additivity. An attribution that does not reconstruct
the prediction is decoration - it will still look plausible on a dashboard, and
still be wrong. Everything else in this file is about the two things that
surround it: that the numbers are grouped back into concepts without being
distorted, and that a missing `shap` never costs anyone a prediction.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.special import expit

from src.api import service
from src.evaluation import explain
from tests.conftest import HIGH_RISK_SUBSCRIBER, LOW_RISK_SUBSCRIBER

shap = pytest.importorskip("shap", reason="attribution is an optional dependency")


@pytest.fixture(scope="module")
def trained_pipeline(trained_model):
    """The session's fitted pipeline, under a name that says what it is."""
    return trained_model.model


@pytest.fixture(scope="module")
def feature_frame(sample_features):
    """A handful of rows to attribute. Small on purpose: TreeSHAP is exact but
    not free, and every assertion here holds row-wise."""
    return sample_features.head(25).reset_index(drop=True)


@pytest.fixture(scope="module")
def attributor(trained_pipeline):
    built = explain.build_attributor(trained_pipeline)
    assert built is not None, "shap is installed, so an attributor must be buildable"
    return built


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #


def test_contributions_reconstruct_the_prediction(attributor, trained_pipeline, feature_frame):
    """The property that makes the whole thing trustworthy.

    Contributions plus the base value, pushed through the logistic link, must
    equal what the model actually predicted. If this fails, every explanation
    the service has ever returned was a plausible-looking fiction.
    """
    contributions = attributor.contributions(feature_frame)
    base = np.asarray(attributor._explainer.expected_value).ravel()[0]  # noqa: SLF001

    reconstructed = expit(contributions.sum(axis=1) + base)
    predicted = trained_pipeline.predict_proba(feature_frame)[:, 1]

    np.testing.assert_allclose(reconstructed, predicted, rtol=1e-6, atol=1e-6)


def test_grouping_conserves_the_total(attributor, feature_frame):
    """Summing into concepts must not lose or invent contribution.

    Valid only because Shapley values are additive - which is exactly why the
    grouping is done by summing rather than by averaging or by taking a max.
    """
    contributions = attributor.contributions(feature_frame)

    for row, totals in zip(contributions, attributor.grouped(feature_frame), strict=True):
        assert sum(totals.values()) == pytest.approx(float(row.sum()), abs=1e-9)


def test_every_model_column_is_assigned_to_a_concept(attributor):
    """An unmapped column silently lands in "other", which explains nothing.

    This catches a feature added to build_features.py without a matching entry
    in FEATURE_GROUPS - the failure would otherwise be a vague attribution
    nobody traces back to its cause.
    """
    unmapped = [
        column
        for column in attributor._columns  # noqa: SLF001
        if column not in explain._COLUMN_TO_GROUP  # noqa: SLF001
    ]
    assert not unmapped, f"columns with no concept group: {unmapped}"


def test_concept_groups_do_not_overlap():
    """A column in two groups would be counted twice and break conservation."""
    seen: set[str] = set()
    for columns in explain.FEATURE_GROUPS.values():
        for column in columns:
            assert column not in seen, f"{column} appears in two groups"
            seen.add(column)


def test_attributions_are_ranked_by_magnitude_not_by_sign(attributor, feature_frame):
    """A strong reason to stay outranks a weak reason to leave.

    The rule engine could only ever list what was wrong, so every explanation
    read like a warning even for a subscriber who was obviously staying.
    """
    records = feature_frame.to_dict("records")
    for row in attributor.top_attributions(feature_frame, records):
        magnitudes = [abs(item.contribution) for item in row]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_wording_follows_the_dominant_column_not_the_group():
    """The bug this phrasing exists to prevent.

    A four-day gap is a long one for a 32-day-old account, and SHAP attributes
    that to `recency_ratio`. Phrasing it from the group name produced
    "inactive for 4 days", which described the wrong thing entirely.
    """
    record = {"last_activity_days_ago": 4, "tenure_days": 32}

    ratio_driven = explain._phrase("recency_ratio", record, raises_risk=True)  # noqa: SLF001
    gap_driven = explain._phrase("last_activity_days_ago", record, raises_risk=True)  # noqa: SLF001

    assert "32" in ratio_driven and "4" in ratio_driven
    assert "inactive" not in ratio_driven
    assert "inactive" in gap_driven


def test_phrases_flip_with_the_direction():
    """The same concept means opposite things depending on which way it pushed."""
    record = {"is_auto_renew_enabled": False}

    assert explain._phrase("is_auto_renew_enabled", record, True) == "auto-renew disabled"  # noqa: SLF001
    assert explain._phrase("is_auto_renew_enabled", record, False) == "auto-renew enabled"  # noqa: SLF001


def test_an_unknown_column_still_produces_readable_text():
    """A feature added tomorrow degrades to its own name, not to a crash."""
    phrase = explain._phrase("some_new_signal", {}, raises_risk=True)  # noqa: SLF001
    assert phrase == "some new signal"


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


def test_a_non_tree_model_falls_back_instead_of_raising():
    """Explanation is a feature of a prediction, not a precondition for one."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = np.random.default_rng(0).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    pipeline = Pipeline([("classifier", LogisticRegression().fit(X, y))])

    assert explain.build_attributor(pipeline) is None


def test_predictions_survive_shap_being_unavailable(client: TestClient, monkeypatch) -> None:
    """The whole point of the optional dependency being optional."""
    loaded = service.get_model()
    monkeypatch.setattr(type(loaded), "attributor", lambda self: None)

    payload = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()

    assert payload["explanation_method"] == "rules"
    assert payload["attributions"] is None
    assert payload["explanation"].strip()
    assert payload["top_risk_factors"]


def test_predictions_survive_an_attributor_that_raises(client: TestClient, monkeypatch) -> None:
    """A model that scores fine but cannot be explained returns a probability.

    Not a 500. Same rule as shadow scoring: the optional path never reaches the
    response except as content.
    """

    class Broken:
        def top_attributions(self, *_args, **_kwargs):
            raise RuntimeError("explainer exploded")

    loaded = service.get_model()
    monkeypatch.setattr(type(loaded), "attributor", lambda self: Broken())

    response = client.post("/predict", json=LOW_RISK_SUBSCRIBER)

    assert response.status_code == 200
    assert response.json()["explanation_method"] == "rules"


# --------------------------------------------------------------------------- #
# The serving contract
# --------------------------------------------------------------------------- #


def test_the_response_says_which_explanation_it_gave(client: TestClient) -> None:
    """The two read almost identically and mean quite different things.

    One attributes the model's decision; the other describes the input against
    fixed thresholds and fires whether or not the model weighed that feature.
    Nobody should have to guess which they are looking at.
    """
    payload = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()

    assert payload["explanation_method"] == "shap"
    assert payload["attributions"]
    for item in payload["attributions"]:
        assert item["direction"] in {"increases_risk", "decreases_risk"}
        assert (item["contribution"] > 0) == (item["direction"] == "increases_risk")


def test_risk_factors_never_include_reasons_to_stay(client: TestClient) -> None:
    """`top_risk_factors` keeps the meaning its name promises.

    Widening it to "drivers, any direction" would leave every existing consumer
    rendering reasons-to-stay as reasons-to-worry.
    """
    payload = client.post("/predict", json=LOW_RISK_SUBSCRIBER).json()

    raising = {
        item["description"]
        for item in payload["attributions"]
        if item["direction"] == "increases_risk"
    }
    assert set(payload["top_risk_factors"]) <= raising


def test_a_high_risk_and_a_low_risk_subscriber_get_opposite_stories(
    client: TestClient,
) -> None:
    """Attribution has to discriminate, or it is just a constant with decimals."""
    high = client.post("/predict", json=HIGH_RISK_SUBSCRIBER).json()
    low = client.post("/predict", json=LOW_RISK_SUBSCRIBER).json()

    assert sum(item["contribution"] for item in high["attributions"]) > 0
    assert sum(item["contribution"] for item in low["attributions"]) < 0
