"""SHAP attributions for individual predictions.

Until now ``/predict`` explained its answers with a set of hand-written rules:
"inactive for 45 days", "2 payment failures". Those are true statements about
the *input*, and they were never claimed to be more than that - but they are
not an explanation of the *decision*. A rule fires whether or not the model
paid any attention to that feature, and it stays silent about anything the
model weighed that nobody thought to write a rule for.

SHAP closes that gap. For a tree ensemble the Shapley values are exact rather
than sampled, and they are additive: for any one subscriber the contributions
sum to the model's output minus its base rate. That is a property this module
tests directly, because an attribution that does not reconstruct the prediction
is decoration.

Three decisions worth stating
-----------------------------

**Log-odds, not probability points.** Additivity holds in the space the trees
actually score in. Converting each contribution to "percentage points of risk"
would break the sum and produce numbers that look precise and are not, so the
raw contributions stay in log-odds and the API phrases them qualitatively.

**Grouped back to business concepts.** The model sees 21 columns, most of them
derived - ``recency_ratio``, ``engagement_recency_score``, ``friction_score``.
Telling a retention manager that ``recency_ratio`` contributed +0.31 is not an
explanation either. Contributions are summed back into the handful of concepts
a person can act on, which is valid precisely because SHAP is additive.

**Optional at runtime.** ``shap`` pulls in numba and llvmlite - a heavy
dependency for a service whose job is to return a probability in a few
milliseconds. If it is not installed, or the model is not a tree ensemble,
attribution degrades to the rule-based explanation rather than failing the
request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Grouping transformed columns back into concepts
# --------------------------------------------------------------------------- #

# Every transformed column belongs to exactly one group. Some derived columns
# genuinely mix two concepts, and each of those is assigned deliberately rather
# than split, because splitting a joint contribution between parents means
# inventing a ratio:
#
#   fee_per_session            fee / (sessions + 1). Grouped under price: it is
#                              a value-for-money signal, and the engagement
#                              part is already carried by three other columns.
#   engagement_recency_score   sessions / (recency + 1). Grouped under
#                              engagement for the same reason in reverse.
#   friction_score             tickets + failures. Kept as its own group rather
#                              than halved between support and payments, since
#                              half of a Shapley value is not a Shapley value.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "recency": ("last_activity_days_ago", "recency_ratio", "is_dormant"),
    "engagement": (
        "avg_session_count_last_30d",
        "sessions_per_day_last_30d",
        "engagement_recency_score",
    ),
    "tenure": ("tenure_days", "tenure_months"),
    "payments": ("payment_failures_last_6m", "payment_failures_per_month"),
    "support": ("support_tickets_last_90d", "support_tickets_per_month"),
    "friction": ("friction_score",),
    "discounts": ("discounts_used_last_6m", "discount_dependency"),
    "price": ("monthly_fee", "fee_per_session"),
    "auto_renew": ("is_auto_renew_enabled",),
    "plan": ("plan_type_basic", "plan_type_standard", "plan_type_premium"),
}

_COLUMN_TO_GROUP: dict[str, str] = {
    column: group for group, columns in FEATURE_GROUPS.items() for column in columns
}


def _phrase(column: str, record: dict[str, Any], raises_risk: bool) -> str:
    """Describe one concept in the subscriber's own numbers.

    Keyed on the *dominant column* within the group rather than on the group
    name, which matters more than it sounds. A first draft phrased every
    ``recency`` attribution as "inactive for N days" and produced this, on a
    real subscriber: "+1.02 inactive for 4 days". Four days is not inactive.
    The contribution had come from ``recency_ratio`` - a four-day gap is a long
    one for an account that is only 32 days old - and quoting the raw day count
    described the wrong thing entirely.

    SHAP still decides which concepts to mention and in which direction; this
    only decides which of the subscriber's numbers to quote back.
    """
    days = record.get("last_activity_days_ago", 0)
    sessions = float(record.get("avg_session_count_last_30d", 0.0) or 0.0)
    tenure = record.get("tenure_days", 0)
    failures = record.get("payment_failures_last_6m", 0)
    tickets = record.get("support_tickets_last_90d", 0)
    discounts = record.get("discounts_used_last_6m", 0)
    fee = float(record.get("monthly_fee", 0.0) or 0.0)
    plan = record.get("plan_type", "unknown")

    stated = _boolean_phrase(column, record)
    if stated is not None:
        return stated

    up, down = {
        "last_activity_days_ago": (
            f"inactive for {days} days",
            f"last seen {days} days ago",
        ),
        # The ratio, not the raw gap: a short gap can still be a long one for a
        # brand-new account, and vice versa.
        "recency_ratio": (
            f"a {days}-day gap is long for a {tenure}-day-old account",
            f"a {days}-day gap is short against {tenure} days of tenure",
        ),
        "avg_session_count_last_30d": (
            f"low recent activity ({sessions:.1f} sessions/30d)",
            f"steady recent activity ({sessions:.1f} sessions/30d)",
        ),
        "sessions_per_day_last_30d": (
            f"low recent activity ({sessions:.1f} sessions/30d)",
            f"steady recent activity ({sessions:.1f} sessions/30d)",
        ),
        "engagement_recency_score": (
            "little activity for how long they have been away",
            "active and recently seen",
        ),
        "tenure_days": (f"only {tenure} days since signup", f"{tenure} days of tenure"),
        "tenure_months": (f"only {tenure} days since signup", f"{tenure} days of tenure"),
        "payment_failures_last_6m": (
            f"{failures} payment failures in the last 6 months",
            "no recent payment trouble",
        ),
        "payment_failures_per_month": (
            f"{failures} payment failures in the last 6 months",
            "no recent payment trouble",
        ),
        "support_tickets_last_90d": (
            f"{tickets} support tickets in the last 90 days",
            "little contact with support",
        ),
        "support_tickets_per_month": (
            f"{tickets} support tickets in the last 90 days",
            "little contact with support",
        ),
        "friction_score": (
            f"combined support and billing friction ({tickets + failures} events)",
            "few support or billing problems",
        ),
        "discounts_used_last_6m": (
            f"relies on discounts ({discounts} in the last 6 months)",
            f"{discounts} discounts used in the last 6 months",
        ),
        "discount_dependency": (
            f"relies on discounts ({discounts} in the last 6 months)",
            f"{discounts} discounts used in the last 6 months",
        ),
        "monthly_fee": (f"high monthly fee ({fee:.2f})", f"monthly fee of {fee:.2f}"),
        "fee_per_session": (
            f"poor value per session ({fee / (sessions + 1.0):.2f} per session)",
            f"good value per session ({fee / (sessions + 1.0):.2f} per session)",
        ),
    }.get(column, _fallback_phrase(column, plan))

    return up if raises_risk else down


# Columns whose wording states a *fact about the input*, not a direction.
#
# Everything else in the table above describes a quantity and quotes the
# subscriber's real number, so keying the two variants on SHAP's direction is
# fine - "inactive for 45 days" and "last seen 45 days ago" are both true.
#
# Booleans are different, and getting this wrong produced a genuinely wrong
# sentence: a request with is_auto_renew_enabled=False came back explained as
# "auto-renew enabled", because on real KKBox data the model reads a missing
# auto-renew as *lowering* risk, and the phrase followed the direction instead
# of the value. An explanation that contradicts its own input is worse than no
# explanation. These read the record instead.
def _boolean_phrase(column: str, record: dict[str, Any]) -> str | None:
    """Wording for columns that must state the input's value, or None."""
    if column == "is_auto_renew_enabled":
        return (
            "auto-renew enabled"
            if record.get("is_auto_renew_enabled")
            else "auto-renew disabled"
        )
    if column == "is_dormant":
        days = record.get("last_activity_days_ago", 0)
        dormant_after = settings.EXPLANATION_RULES.dormant_days
        return (
            f"dormant - no session in {days} days"
            if days > dormant_after
            else f"active within the last {days} days"
        )
    return None


def _fallback_phrase(column: str, plan: str) -> tuple[str, str]:
    """Wording for columns with no bespoke phrase.

    The one-hot plan columns read the same in both directions - "on the
    premium plan" is a fact, not a complaint - and anything else falls back to
    its own name so a newly added feature degrades to something readable
    rather than disappearing from explanations.
    """
    if column.startswith("plan_type"):
        return (f"on the {plan} plan", f"on the {plan} plan")
    readable = column.replace("_", " ")
    return (readable, readable)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def _flatten_steps(estimator: Any) -> list[Any]:
    """Every transformer in a possibly-nested pipeline, in execution order."""
    if isinstance(estimator, Pipeline):
        return [inner for _, step in estimator.steps for inner in _flatten_steps(step)]
    return [estimator]


@lru_cache(maxsize=1)
def shap_available() -> bool:
    """Whether the optional ``shap`` dependency can be imported.

    Cached: this is asked once per prediction batch, and a failing import is
    not cheap to repeat.
    """
    try:
        import shap  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "fall back"
        return False
    return True


@dataclass(frozen=True)
class _GroupTotal:
    """A group's summed contribution, plus the column that drove it."""

    total: float
    dominant: str
    dominant_weight: float


@dataclass(frozen=True)
class Attribution:
    """One concept's contribution to one subscriber's score."""

    feature: str
    contribution: float
    description: str

    @property
    def direction(self) -> str:
        return "increases_risk" if self.contribution > 0 else "decreases_risk"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "direction": self.direction,
            # Log-odds. Rounded for the wire, not for the arithmetic.
            "contribution": round(float(self.contribution), 4),
            "description": self.description,
        }


class TreeAttributor:
    """Exact SHAP attributions for a fitted pipeline ending in a tree ensemble.

    Holds the explainer rather than rebuilding it per request: constructing a
    ``TreeExplainer`` walks the whole ensemble, which costs far more than
    scoring a batch with it.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        import shap

        self._pipeline = pipeline
        self._preprocess = pipeline[:-1]
        estimator = pipeline[-1]

        # ``check_additivity`` in shap_values() re-runs the model to verify the
        # sum; leave it on. It is the only thing standing between a correct
        # attribution and a confident wrong one.
        self._explainer = shap.TreeExplainer(estimator)
        self._columns = list(self._feature_names())

    def _feature_names(self) -> list[str]:
        """Column names as they leave preprocessing.

        Asking the preprocessing pipeline directly does not work here: it opens
        with a ``FunctionTransformer`` that adds the derived columns, and a
        ``FunctionTransformer`` cannot say what it produced. scikit-learn
        refuses to name the whole chain because of it.

        So walk backwards to the last step that *can* name its output. That is
        the ``ColumnTransformer``, which is also the step that decides the
        final column order - exactly the names SHAP's columns line up with.

        Falls back to positional names rather than raising: an unnamed column
        groups under "other" and is still attributed, which beats losing
        explanations entirely because a transformer changed.
        """
        for step in reversed(list(_flatten_steps(self._preprocess))):
            try:
                return list(step.get_feature_names_out())
            except Exception:  # noqa: BLE001 - try the next step up
                continue

        logger.warning("Preprocessing exposes no feature names; using positions")
        return []

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """Raw per-column SHAP values for a batch, in log-odds.

        Returns:
            Array of shape ``(n_rows, n_transformed_columns)``.
        """
        transformed = self._preprocess.transform(frame)
        values = self._explainer.shap_values(transformed, check_additivity=True)
        values = np.asarray(values)

        # Binary classifiers report either (n, k) for the positive class or
        # (n, k, 2) for both, depending on the estimator and the shap version.
        # Take the positive class in either case.
        if values.ndim == 3:
            values = values[:, :, 1]
        return values

    def _group_rows(self, frame: pd.DataFrame) -> list[dict[str, _GroupTotal]]:
        """Per-row group totals, each remembering which column dominated it.

        Summing within a group is valid because Shapley values are additive:
        the total is the group's joint contribution, with no approximation. The
        dominant column is carried alongside because it decides the wording -
        see :func:`_phrase`.
        """
        values = self.contributions(frame)
        names = self._columns or [f"column_{i}" for i in range(values.shape[1])]

        rows: list[dict[str, _GroupTotal]] = []
        for row in values:
            totals: dict[str, _GroupTotal] = {}
            for name, raw in zip(names, row, strict=True):
                value = float(raw)
                group = _COLUMN_TO_GROUP.get(name, "other")
                current = totals.get(group)
                if current is None:
                    totals[group] = _GroupTotal(value, name, abs(value))
                    continue
                dominant, weight = (
                    (name, abs(value))
                    if abs(value) > current.dominant_weight
                    else (current.dominant, current.dominant_weight)
                )
                totals[group] = _GroupTotal(current.total + value, dominant, weight)
            rows.append(totals)
        return rows

    def grouped(self, frame: pd.DataFrame) -> list[dict[str, float]]:
        """Contributions summed into business concepts, one dict per row."""
        return [
            {group: total.total for group, total in row.items()}
            for row in self._group_rows(frame)
        ]

    def top_attributions(
        self, frame: pd.DataFrame, records: list[dict[str, Any]], limit: int = 3
    ) -> list[list[Attribution]]:
        """The ``limit`` strongest concepts per row, largest magnitude first.

        Ranked by absolute contribution, so a strong reason the subscriber is
        *staying* can outrank a weak reason they might leave. That asymmetry is
        the point: the rule-based explanation could only ever list what was
        wrong, which made every explanation read like a warning.
        """
        out: list[list[Attribution]] = []
        for totals, record in zip(self._group_rows(frame), records, strict=True):
            ranked = sorted(totals.items(), key=lambda item: abs(item[1].total), reverse=True)
            out.append(
                [
                    Attribution(
                        feature=group,
                        contribution=entry.total,
                        description=_phrase(
                            entry.dominant, record, raises_risk=entry.total > 0
                        ),
                    )
                    for group, entry in ranked[:limit]
                    # A contribution of exactly zero is not a reason for
                    # anything and should not take a slot.
                    if entry.total != 0.0
                ]
            )
        return out


def build_attributor(pipeline: Pipeline) -> TreeAttributor | None:
    """Return an attributor for ``pipeline``, or ``None`` if it cannot explain it.

    Never raises. Explanation is a feature of a prediction, not a precondition
    for one: a missing dependency, a non-tree model or an unexpected pipeline
    shape all mean "fall back to the rules", not "fail the request".
    """
    if not shap_available():
        logger.info("shap is not installed; predictions will use rule-based explanations")
        return None
    try:
        return TreeAttributor(pipeline)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Could not build a SHAP attributor (%s); falling back to rules", exc)
        return None
