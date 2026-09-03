"""Pydantic request and response models for the inference API.

These schemas are the public contract of the service.  They mirror the raw
training columns exactly (minus ``subscriber_id`` and the target), because the
saved pipeline performs its own feature engineering.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PlanType(str, Enum):
    """Subscription tiers known to the model."""

    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"


class RiskLevel(str, Enum):
    """Coarse risk band derived from the predicted probability."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_EXAMPLE_SUBSCRIBER: dict[str, Any] = {
    "tenure_days": 95,
    "plan_type": "standard",
    "monthly_fee": 19.99,
    "avg_session_count_last_30d": 2.0,
    "last_activity_days_ago": 34,
    "support_tickets_last_90d": 3,
    "payment_failures_last_6m": 2,
    "discounts_used_last_6m": 1,
    "is_auto_renew_enabled": False,
}


class SubscriberFeaturesRequest(BaseModel):
    """Behavioural and account features for a single subscriber."""

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_SUBSCRIBER})

    tenure_days: int = Field(
        ..., ge=0, le=20_000, description="Days since the subscription started."
    )
    plan_type: PlanType = Field(..., description="Subscription tier.")
    monthly_fee: float = Field(..., ge=0, le=10_000, description="Current monthly fee.")
    avg_session_count_last_30d: float = Field(
        ..., ge=0, le=10_000, description="Average number of sessions in the last 30 days."
    )
    last_activity_days_ago: int = Field(
        ..., ge=0, le=20_000, description="Days since the subscriber was last active."
    )
    support_tickets_last_90d: int = Field(
        ..., ge=0, le=1_000, description="Support tickets raised in the last 90 days."
    )
    payment_failures_last_6m: int = Field(
        ..., ge=0, le=1_000, description="Failed payment attempts in the last 6 months."
    )
    discounts_used_last_6m: int = Field(
        ..., ge=0, le=1_000, description="Discounts or coupons redeemed in the last 6 months."
    )
    is_auto_renew_enabled: bool = Field(..., description="Whether auto-renew is switched on.")

    @field_validator("plan_type", mode="before")
    @classmethod
    def _normalise_plan_type(cls, value: Any) -> Any:
        """Accept ``"Premium"`` / ``" premium "`` as well as ``"premium"``."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def _check_activity_within_tenure(self) -> SubscriberFeaturesRequest:
        """A subscriber cannot have been inactive for longer than they existed."""
        if self.last_activity_days_ago > self.tenure_days:
            raise ValueError(
                "last_activity_days_ago cannot exceed tenure_days "
                f"({self.last_activity_days_ago} > {self.tenure_days})."
            )
        return self

    def to_features(self) -> dict[str, Any]:
        """Return a plain dict with model-ready (JSON-safe) values."""
        payload = self.model_dump()
        payload["plan_type"] = self.plan_type.value
        return payload


class PredictionResponse(BaseModel):
    """Model output for one subscriber."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dropout_probability": 0.8421,
                "predicted_label": 1,
                "risk_level": "high",
                "threshold": 0.42,
                "explanation": (
                    "High dropout risk: inactive for 34 days, low recent activity "
                    "(2.0 sessions/30d), 2 payment failures in the last 6 months, "
                    "auto-renew disabled."
                ),
                "top_risk_factors": [
                    "inactive for 34 days",
                    "low recent activity (2.0 sessions/30d)",
                    "2 payment failures in the last 6 months",
                ],
            }
        }
    )

    dropout_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability the subscriber drops out."
    )
    predicted_label: int = Field(
        ..., ge=0, le=1, description="1 if the probability meets the decision threshold, else 0."
    )
    risk_level: RiskLevel = Field(..., description="Coarse risk band.")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision threshold applied.")
    explanation: str = Field(..., description="Human-readable, rule-based rationale.")
    top_risk_factors: list[str] = Field(
        default_factory=list, description="Individual risk signals that fired."
    )


class BatchPredictionRequest(BaseModel):
    """A batch of subscribers to score in one call."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"subscribers": [_EXAMPLE_SUBSCRIBER]}}
    )

    subscribers: list[SubscriberFeaturesRequest] = Field(
        ..., min_length=1, max_length=1_000, description="Between 1 and 1000 subscribers."
    )


class BatchPredictionResponse(BaseModel):
    """Predictions for a batch, in the order they were submitted."""

    predictions: list[PredictionResponse]
    count: int = Field(..., description="Number of subscribers scored.")


class HealthResponse(BaseModel):
    """Liveness payload."""

    model_config = ConfigDict(json_schema_extra={"example": {"status": "ok"}})

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """Readiness payload: liveness plus whether the model artifact is loaded."""

    # ``model_loaded`` collides with pydantic's reserved ``model_`` namespace.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    detail: str | None = None


class DriftRequest(BaseModel):
    """A sample of live subscribers to compare against the training data."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"subscribers": [_EXAMPLE_SUBSCRIBER]}}
    )

    subscribers: list[SubscriberFeaturesRequest] = Field(
        ...,
        min_length=1,
        max_length=50_000,
        description=(
            "Recent live subscribers. PSI is noisy below a few hundred rows; the "
            "response reports whether the sample was large enough to trust."
        ),
    )


class FeatureDrift(BaseModel):
    """Drift measured for a single feature."""

    feature: str
    psi: float = Field(..., description="Population Stability Index against the training data.")
    verdict: str = Field(..., description="stable | moderate | significant")
    kind: str = Field(..., description="numeric | discrete | prediction")
    reference_mean: float | None = None
    live_mean: float | None = None
    unseen_categories: list[str] = Field(default_factory=list)


class DriftResponse(BaseModel):
    """Per-feature and overall drift between live traffic and training."""

    n_samples: int
    sufficient_sample: bool = Field(
        ..., description="False when the batch is too small for PSI to be meaningful."
    )
    min_samples: int
    overall_verdict: str
    drifted_features: list[str]
    features: list[FeatureDrift]
    prediction: FeatureDrift | None = None
    reference_created_at: str | None = None
    reference_rows: int | None = None
    thresholds: dict[str, float]


class MetricsResponse(BaseModel):
    """Live statistics for the predictions this process has served."""

    model_config = ConfigDict(protected_namespaces=())

    served_total: int
    window_size: int
    window_capacity: int
    flagged_rate: float
    probability: dict[str, float]
    risk_levels: dict[str, int]
    model_loaded: bool
    threshold: float | None = None
    reference: dict[str, Any] | None = None
    probability_mean_shift: float | None = Field(
        None, description="Live mean score minus the training-time mean score."
    )


class ModelInfoResponse(BaseModel):
    """Metadata about the currently served artifact."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    trained_at: str | None = None
    decision_threshold: float
    required_input_columns: list[str]
    library_versions: dict[str, str] = Field(default_factory=dict)
    served_from: str = Field(
        "local", description="Where this model was loaded from: 'registry' or 'local'."
    )
    registry_version: str | None = Field(
        None, description="MLflow registered model version, when served_from='registry'."
    )
