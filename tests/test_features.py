"""Unit tests for the feature engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DERIVED_BINARY_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    REQUIRED_INPUT_COLUMNS,
    add_derived_features,
    build_feature_pipeline,
    build_preprocessor,
    frame_from_records,
)


def test_required_input_columns_are_the_raw_schema() -> None:
    """The API contract must match what the pipeline expects as input."""
    assert set(REQUIRED_INPUT_COLUMNS) == {
        "tenure_days",
        "plan_type",
        "monthly_fee",
        "avg_session_count_last_30d",
        "last_activity_days_ago",
        "support_tickets_last_90d",
        "payment_failures_last_6m",
        "discounts_used_last_6m",
        "is_auto_renew_enabled",
    }


def test_add_derived_features_adds_expected_columns(sample_features: pd.DataFrame) -> None:
    """Every declared derived column is actually produced."""
    result = add_derived_features(sample_features)

    for column in DERIVED_NUMERIC_FEATURES + DERIVED_BINARY_FEATURES:
        assert column in result.columns, f"missing derived column {column}"
    # Raw columns survive the transformation.
    for column in REQUIRED_INPUT_COLUMNS:
        assert column in result.columns


def test_add_derived_features_preserves_row_count(sample_features: pd.DataFrame) -> None:
    """Feature derivation is row-wise: it must not add or drop rows."""
    result = add_derived_features(sample_features)
    assert len(result) == len(sample_features)


def test_add_derived_features_does_not_mutate_input(sample_features: pd.DataFrame) -> None:
    """The caller's frame must be left untouched."""
    before = sample_features.copy()
    add_derived_features(sample_features)
    pd.testing.assert_frame_equal(sample_features, before)


def test_derived_features_are_finite(sample_features: pd.DataFrame) -> None:
    """No NaN or infinity may leak out of the derivation step."""
    result = add_derived_features(sample_features)
    numeric = result[DERIVED_NUMERIC_FEATURES].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_derived_features_handle_zero_denominators() -> None:
    """A brand-new, never-active subscriber must not divide by zero."""
    frame = pd.DataFrame(
        [
            {
                "tenure_days": 0,
                "plan_type": "basic",
                "monthly_fee": 9.99,
                "avg_session_count_last_30d": 0.0,
                "last_activity_days_ago": 0,
                "support_tickets_last_90d": 0,
                "payment_failures_last_6m": 0,
                "discounts_used_last_6m": 0,
                "is_auto_renew_enabled": True,
            }
        ]
    )
    result = add_derived_features(frame)
    values = result[DERIVED_NUMERIC_FEATURES].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert result.loc[0, "recency_ratio"] == 0.0
    assert result.loc[0, "fee_per_session"] == pytest.approx(9.99)


@pytest.mark.parametrize(
    ("last_activity_days_ago", "expected"),
    [(0, 0), (5, 0), (21, 0), (22, 1), (90, 1)],
)
def test_is_dormant_flag(last_activity_days_ago: int, expected: int) -> None:
    """``is_dormant`` fires strictly above the configured dormancy window."""
    frame = pd.DataFrame(
        [
            {
                "tenure_days": 500,
                "plan_type": "basic",
                "monthly_fee": 9.99,
                "avg_session_count_last_30d": 3.0,
                "last_activity_days_ago": last_activity_days_ago,
                "support_tickets_last_90d": 0,
                "payment_failures_last_6m": 0,
                "discounts_used_last_6m": 0,
                "is_auto_renew_enabled": True,
            }
        ]
    )
    assert int(add_derived_features(frame).loc[0, "is_dormant"]) == expected


def test_friction_score_sums_support_and_billing_issues() -> None:
    """Spot-check one derived formula against a hand-computed value."""
    frame = pd.DataFrame(
        [
            {
                "tenure_days": 200,
                "plan_type": "premium",
                "monthly_fee": 34.99,
                "avg_session_count_last_30d": 9.0,
                "last_activity_days_ago": 4,
                "support_tickets_last_90d": 2,
                "payment_failures_last_6m": 3,
                "discounts_used_last_6m": 1,
                "is_auto_renew_enabled": False,
            }
        ]
    )
    result = add_derived_features(frame)
    assert result.loc[0, "friction_score"] == pytest.approx(5.0)
    assert result.loc[0, "support_tickets_per_month"] == pytest.approx(2 / 3)
    assert result.loc[0, "payment_failures_per_month"] == pytest.approx(0.5)


def test_add_derived_features_rejects_missing_column(sample_features: pd.DataFrame) -> None:
    """A missing required column fails loudly instead of producing NaNs."""
    with pytest.raises(KeyError, match="monthly_fee"):
        add_derived_features(sample_features.drop(columns=["monthly_fee"]))


def test_preprocessor_output_shape(sample_features: pd.DataFrame) -> None:
    """One column per numeric feature, per plan level, and per binary flag."""
    preprocessor = build_preprocessor()
    derived = add_derived_features(sample_features)
    matrix = preprocessor.fit_transform(derived)

    n_plan_levels = sample_features["plan_type"].nunique()
    expected_columns = len(NUMERIC_FEATURES) + n_plan_levels + len(BINARY_FEATURES)
    assert matrix.shape == (len(sample_features), expected_columns)


def test_preprocessor_scales_numeric_features(sample_features: pd.DataFrame) -> None:
    """Scaled numeric columns come out roughly zero-mean and unit-variance."""
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(add_derived_features(sample_features))
    numeric_block = matrix[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(numeric_block.std(axis=0), 1.0, atol=1e-8)


def test_full_pipeline_transform(sample_features: pd.DataFrame) -> None:
    """The end-to-end feature pipeline yields a finite numeric matrix."""
    pipeline = build_feature_pipeline()
    matrix = pipeline.fit_transform(sample_features)
    assert matrix.shape[0] == len(sample_features)
    assert np.isfinite(np.asarray(matrix, dtype=float)).all()


def test_pipeline_handles_unseen_plan_type(sample_features: pd.DataFrame) -> None:
    """An unknown plan encodes to an all-zero block instead of raising."""
    pipeline = build_feature_pipeline()
    pipeline.fit(sample_features)

    unseen = sample_features.head(1).copy()
    unseen.loc[unseen.index[0], "plan_type"] = "enterprise"
    matrix = pipeline.transform(unseen)

    n_numeric = len(NUMERIC_FEATURES)
    n_plans = sample_features["plan_type"].nunique()
    plan_block = np.asarray(matrix)[0, n_numeric : n_numeric + n_plans]
    assert plan_block.sum() == 0.0
    assert len(CATEGORICAL_FEATURES) == 1


def test_frame_from_records_orders_columns() -> None:
    """API dicts become a frame whose columns match the training order."""
    record = {
        "is_auto_renew_enabled": True,
        "plan_type": "basic",
        "monthly_fee": 9.99,
        "tenure_days": 10,
        "avg_session_count_last_30d": 1.0,
        "last_activity_days_ago": 2,
        "support_tickets_last_90d": 0,
        "payment_failures_last_6m": 0,
        "discounts_used_last_6m": 0,
    }
    frame = frame_from_records([record])
    assert list(frame.columns) == REQUIRED_INPUT_COLUMNS
    assert len(frame) == 1


def test_frame_from_records_rejects_incomplete_payload() -> None:
    """Missing features are reported by name."""
    with pytest.raises(KeyError, match="monthly_fee"):
        frame_from_records([{"tenure_days": 10}])
