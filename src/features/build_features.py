"""Feature engineering for the subscriber dropout model.

The derived features and the encoders live inside a single scikit-learn
pipeline.  That matters for serving: the API sends the *raw* subscriber columns
and the saved artifact performs every transformation itself, so there is no way
for training-time and serving-time feature code to drift apart.

Pipeline shape::

    raw columns
      -> FunctionTransformer(add_derived_features)   # ratios, rates, flags
      -> ColumnTransformer
           numeric      : median impute -> standard scale
           categorical  : most-frequent impute -> one-hot (unknown-safe)
           binary flags : passthrough
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import settings

# Columns as they arrive from the CSV / API request.
RAW_NUMERIC_FEATURES: list[str] = [
    "tenure_days",
    "monthly_fee",
    "avg_session_count_last_30d",
    "last_activity_days_ago",
    "support_tickets_last_90d",
    "payment_failures_last_6m",
    "discounts_used_last_6m",
]
CATEGORICAL_FEATURES: list[str] = ["plan_type"]
RAW_BINARY_FEATURES: list[str] = ["is_auto_renew_enabled"]

# Columns created by :func:`add_derived_features`.
DERIVED_NUMERIC_FEATURES: list[str] = [
    "tenure_months",
    "sessions_per_day_last_30d",
    "recency_ratio",
    "engagement_recency_score",
    "fee_per_session",
    "support_tickets_per_month",
    "payment_failures_per_month",
    "discount_dependency",
    "friction_score",
]
DERIVED_BINARY_FEATURES: list[str] = ["is_dormant"]

NUMERIC_FEATURES: list[str] = RAW_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
BINARY_FEATURES: list[str] = RAW_BINARY_FEATURES + DERIVED_BINARY_FEATURES

# Every column a caller must supply.
REQUIRED_INPUT_COLUMNS: list[str] = (
    RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES + RAW_BINARY_FEATURES
)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural ratio and rate features to a raw subscriber frame.

    Raw counts are hard for a model to compare across subscribers with very
    different tenures and plans, so we add normalised versions: activity per
    day, friction per month, and how recent the last session is relative to the
    subscriber's lifetime.  Denominators carry a ``+ 1`` guard so a brand-new or
    completely inactive subscriber cannot produce a division by zero.

    Args:
        frame: Raw features, one row per subscriber.

    Returns:
        A copy of ``frame`` with the derived columns appended.
    """
    df = frame.copy()

    # Missing raw inputs would silently propagate as NaN through the ratios, so
    # fill them here; the imputers downstream handle anything left over.
    for column in RAW_NUMERIC_FEATURES:
        if column not in df.columns:
            raise KeyError(f"Missing required feature column: {column!r}")

    tenure = df["tenure_days"].astype(float)
    sessions = df["avg_session_count_last_30d"].astype(float)
    recency = df["last_activity_days_ago"].astype(float)
    tickets = df["support_tickets_last_90d"].astype(float)
    failures = df["payment_failures_last_6m"].astype(float)
    discounts = df["discounts_used_last_6m"].astype(float)
    fee = df["monthly_fee"].astype(float)

    df["tenure_months"] = tenure / 30.44
    df["sessions_per_day_last_30d"] = sessions / 30.0
    df["recency_ratio"] = recency / (tenure + 1.0)
    # High when the subscriber is both active and recently seen.
    df["engagement_recency_score"] = sessions / (recency + 1.0)
    # What each session effectively costs: a proxy for perceived value.
    df["fee_per_session"] = fee / (sessions + 1.0)
    df["support_tickets_per_month"] = tickets / 3.0
    df["payment_failures_per_month"] = failures / 6.0
    df["discount_dependency"] = discounts / 6.0
    df["friction_score"] = tickets + failures

    df["is_dormant"] = (recency > settings.EXPLANATION_RULES.dormant_days).astype(int)

    if "is_auto_renew_enabled" in df.columns:
        df["is_auto_renew_enabled"] = df["is_auto_renew_enabled"].astype(bool).astype(int)

    # Guard against infinities introduced by extreme inputs.
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the column-wise preprocessing step.

    Numeric columns are median-imputed then standard-scaled; the categorical
    plan is one-hot encoded with ``handle_unknown="ignore"`` so an unseen plan
    name at inference time yields an all-zero block instead of an exception.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """Return the full feature pipeline (derivation + preprocessing)."""
    return Pipeline(
        steps=[
            ("derive", FunctionTransformer(add_derived_features, validate=False)),
            ("preprocess", build_preprocessor()),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return output column names of a *fitted* preprocessor.

    Falls back to positional names if the transformer cannot report them.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - defensive, sklearn version dependent
        return [f"feature_{i}" for i in range(preprocessor.transform_output_shape_[1])]


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """Build a model-ready DataFrame from API request dictionaries.

    Columns are ordered to match :data:`REQUIRED_INPUT_COLUMNS` so the frame
    looks exactly like the training input regardless of JSON key order.
    """
    frame = pd.DataFrame.from_records(records)
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return frame[REQUIRED_INPUT_COLUMNS]
