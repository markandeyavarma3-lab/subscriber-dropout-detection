"""Gated promotion: a challenger only takes over on evidence.

The default behaviour of most training scripts is to overwrite the artifact
every run.  That is a deployment with no gate: a model trained on a bad day, a
corrupted backfill, or an unlucky seed replaces a good one silently, and
rolling back means retraining and hoping.

Here promotion is a decision with three properties:

**Comparable.** Champion and challenger are scored on the *same* held-out data,
in the same process.  Comparing a challenger's fresh test score against a
number recorded from the champion's own training run months ago is the classic
way to promote a worse model - the two numbers were never measured on the same
population.

**Marginal.** The challenger must win by ``PROMOTION_MIN_IMPROVEMENT``, not
merely tie.  Two equivalent models differ by noise, and a zero-margin gate
promotes on that noise roughly half the time, churning the registry forever.

**Reversible.** Losing does not delete anything.  Every version stays in the
registry, so promotion is a moved alias and rollback is moving it back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import settings
from src.models.evaluate import compute_metrics
from src.registry import tracking

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotionDecision:
    """The outcome of comparing a challenger against the reigning champion."""

    promoted: bool
    reason: str
    metric: str
    challenger_score: float
    champion_score: float | None = None
    champion_version: str | None = None
    challenger_version: str | None = None
    required_improvement: float = settings.PROMOTION_MIN_IMPROVEMENT
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float | None:
        """How much the challenger beat the champion by, if there was one."""
        if self.champion_score is None:
            return None
        return round(self.challenger_score - self.champion_score, 6)

    def summary(self) -> str:
        """One-line human-readable verdict."""
        verdict = "PROMOTED" if self.promoted else "REJECTED"
        if self.champion_score is None:
            return f"{verdict}: {self.reason} ({self.metric}={self.challenger_score:.4f})"
        return (
            f"{verdict}: {self.reason} | {self.metric} "
            f"challenger={self.challenger_score:.4f} champion={self.champion_score:.4f} "
            f"(delta={self.improvement:+.4f}, required=+{self.required_improvement:.4f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form, for logging into a run or a pipeline artifact."""
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "metric": self.metric,
            "challenger_score": self.challenger_score,
            "champion_score": self.champion_score,
            "improvement": self.improvement,
            "required_improvement": self.required_improvement,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            **self.details,
        }


def _score(model: Pipeline, features: pd.DataFrame, target: pd.Series, metric: str) -> float:
    """Evaluate one model on one dataset and pull out the gating metric."""
    probabilities = tracking.score_probabilities(model, features)
    metrics = compute_metrics(target, probabilities, threshold=settings.DECISION_THRESHOLD)
    if metric not in metrics:
        raise KeyError(f"Promotion metric {metric!r} not in computed metrics: {sorted(metrics)}")

    value = metrics[metric]
    if value is None:
        # Ranking metrics are undefined on a single-class split. Treating that
        # as a score of zero would silently reject every challenger, so it is a
        # broken evaluation set and must be raised, not swallowed.
        raise ValueError(
            f"Metric {metric!r} is undefined on this evaluation set "
            f"({len(target)} rows, {int(target.sum())} positives). "
            "Promotion needs both classes present."
        )
    return float(value)


def evaluate_promotion(
    challenger: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    challenger_version: str | None = None,
    metric: str | None = None,
    min_improvement: float | None = None,
    model_name: str | None = None,
) -> PromotionDecision:
    """Decide whether a challenger should replace the current champion.

    Both models are scored on the supplied held-out data, which must be data
    neither of them was trained on.

    Args:
        challenger: The newly trained pipeline.
        features: Held-out feature matrix.
        target: Held-out labels.
        challenger_version: Registry version of the challenger, for the record.
        metric: Gating metric. Defaults to ``settings.PROMOTION_METRIC``.
        min_improvement: Required margin. Defaults to settings.
        model_name: Registry name. Defaults to settings.

    Returns:
        A :class:`PromotionDecision`. It does **not** move any alias; call
        :func:`promote` to act on it.
    """
    gate_metric = metric or settings.PROMOTION_METRIC
    margin = (
        settings.PROMOTION_MIN_IMPROVEMENT if min_improvement is None else float(min_improvement)
    )

    challenger_score = _score(challenger, features, target, gate_metric)

    champion_version = tracking.get_alias_version(settings.CHAMPION_ALIAS, model_name)
    if champion_version is None:
        return PromotionDecision(
            promoted=True,
            reason="no incumbent champion",
            metric=gate_metric,
            challenger_score=challenger_score,
            challenger_version=challenger_version,
            required_improvement=margin,
            details={"n_eval_rows": int(len(features))},
        )

    champion = tracking.load_aliased_model(settings.CHAMPION_ALIAS, model_name)
    if champion is None:  # pragma: no cover - alias set but artifact unreadable
        return PromotionDecision(
            promoted=True,
            reason="champion artifact could not be loaded",
            metric=gate_metric,
            challenger_score=challenger_score,
            challenger_version=challenger_version,
            required_improvement=margin,
        )

    champion_score = _score(champion, features, target, gate_metric)
    beat_by = challenger_score - champion_score
    promoted = beat_by >= margin

    return PromotionDecision(
        promoted=promoted,
        reason=(
            f"beat champion by {beat_by:+.4f}"
            if promoted
            else f"improvement {beat_by:+.4f} below required +{margin:.4f}"
        ),
        metric=gate_metric,
        challenger_score=challenger_score,
        champion_score=champion_score,
        champion_version=champion_version.version,
        challenger_version=challenger_version,
        required_improvement=margin,
        details={"n_eval_rows": int(len(features))},
    )


def promote(
    decision: PromotionDecision, model_name: str | None = None
) -> PromotionDecision:
    """Move the champion alias if the decision says to.

    Returns the decision unchanged, so it can be used inline in a pipeline.
    """
    if decision.promoted and decision.challenger_version is not None:
        tracking.set_alias(settings.CHAMPION_ALIAS, decision.challenger_version, model_name)
        logger.info(
            "Promoted version %s to @%s", decision.challenger_version, settings.CHAMPION_ALIAS
        )
    else:
        logger.info("No promotion: %s", decision.reason)
    return decision


def rollback(version: str | int, model_name: str | None = None) -> None:
    """Point the champion alias back at an earlier version.

    Rollback is cheap precisely because promotion never deleted anything.
    """
    tracking.set_alias(settings.CHAMPION_ALIAS, version, model_name)
    logger.warning("Rolled @%s back to version %s", settings.CHAMPION_ALIAS, version)
