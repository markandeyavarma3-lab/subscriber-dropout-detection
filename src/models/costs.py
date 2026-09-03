"""Choose the decision threshold from business cost, not from F1.

Training currently tunes the threshold to maximise F1. F1 is a reasonable
default when nothing is known about the consequences, but it encodes a specific
and usually wrong assumption: that a false positive and a false negative cost
the same.

For subscriber retention they do not, and the gap is large. A false negative is
a churned subscriber - their remaining lifetime value, gone. A false positive is
a retention offer sent to somebody who was staying anyway - the cost of the
offer, and some irritation. If a subscriber is worth £240 and an offer costs
£20, missing one churner costs as much as twelve wasted offers, and a threshold
that treats them as equal will be far too conservative.

Two ways to pick the threshold
------------------------------

**Analytically.** With *calibrated* probabilities, expected cost is minimised by
acting whenever ``p × cost_fn > (1 - p) × cost_fp``, which rearranges to a
threshold of ``cost_fp / (cost_fp + cost_fn)``. No search, no data.

**Empirically.** Sweep the threshold and pick the cheapest on a holdout.

Both are computed here, and :func:`threshold_report` shows them side by side on
purpose. When they agree, the probabilities are well calibrated and the
analytic answer can be trusted going forward. When they diverge, that gap *is*
the calibration error made visible in currency - which is a far more legible
argument for calibrating than an ECE of 0.04.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostModel:
    """What each kind of mistake costs, in whatever currency you like.

    Attributes:
        false_negative: Cost of missing a churner - their remaining value.
        false_positive: Cost of a retention offer to somebody who was staying.
        offer_cost: What one retention offer costs to make.
        offer_efficacy: Probability the offer actually saves a subscriber who
            would otherwise have churned. **This is the parameter that decides
            whether the whole exercise is honest.** An earlier version of this
            model implicitly set it to 1.0 - assume every correctly-targeted
            offer works - and the "cost-optimal" threshold it produced was to
            contact 506 of 508 subscribers. That is arithmetically correct and
            operationally absurd: if intervening always works and costs a
            twelfth of what churn costs, blanket outreach wins. Real retention
            offers convert perhaps a fifth to a third of the time, and at that
            rate acting is only worth it on people who are genuinely likely to
            leave.
    """

    false_negative: float = settings.COST_FALSE_NEGATIVE
    false_positive: float = settings.COST_FALSE_POSITIVE
    offer_cost: float = settings.COST_TRUE_POSITIVE
    offer_efficacy: float = settings.OFFER_EFFICACY

    @property
    def true_positive(self) -> float:
        """Cost of correctly flagging a churner and making them an offer.

        You pay for the offer *and*, most of the time, still lose them: only
        ``offer_efficacy`` of these are actually saved. Modelling this as just
        the offer cost is what makes blanket outreach look free.
        """
        return self.offer_cost + (1.0 - self.offer_efficacy) * self.false_negative

    @property
    def ratio(self) -> float:
        """How many wasted offers one missed churner is worth."""
        if self.false_positive == 0:
            return float("inf")
        return self.false_negative / self.false_positive

    @property
    def analytic_threshold(self) -> float:
        """The cost-minimising cut-off for perfectly calibrated probabilities.

        Acting is worth it when the expected saving beats the expected waste:
        ``p × (cost_fn − cost_tp) > (1 − p) × cost_fp``. The benefit term is
        the *net* gain from intervening, which shrinks as efficacy falls - so a
        cheap offer that rarely works has a high threshold, not a low one.
        """
        benefit = self.false_negative - self.true_positive
        if benefit <= 0:
            # Acting costs at least as much as the churn it prevents.
            return 1.0
        return float(self.false_positive / (self.false_positive + benefit))


def expected_cost(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    costs: CostModel | None = None,
) -> dict[str, Any]:
    """Total cost of operating at one threshold, plus the confusion counts."""
    model = costs or CostModel()
    truth = np.asarray(y_true).astype(int)
    predicted = (np.asarray(y_proba) >= threshold).astype(int)

    true_positives = int(((predicted == 1) & (truth == 1)).sum())
    false_positives = int(((predicted == 1) & (truth == 0)).sum())
    false_negatives = int(((predicted == 0) & (truth == 1)).sum())
    true_negatives = int(((predicted == 0) & (truth == 0)).sum())

    total = (
        false_negatives * model.false_negative
        + false_positives * model.false_positive
        + true_positives * model.true_positive
    )

    return {
        "threshold": round(float(threshold), 4),
        "total_cost": round(float(total), 2),
        "cost_per_subscriber": round(float(total / max(len(truth), 1)), 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        # The number somebody has to staff: how many offers get sent.
        "flagged": true_positives + false_positives,
        "flagged_rate": round(float((true_positives + false_positives) / max(len(truth), 1)), 4),
    }


def cost_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    costs: CostModel | None = None,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Sweep thresholds and return the cheapest, with its full cost breakdown.

    Ties go to the *higher* threshold. Two thresholds costing the same money
    are not equally good operationally: the higher one flags fewer people, so
    it needs less human capacity for the same result.
    """
    model = costs or CostModel()
    candidates = np.round(grid if grid is not None else np.arange(0.01, 1.0, 0.01), 4)

    scored = [
        (float(threshold), expected_cost(y_true, y_proba, float(threshold), model))
        for threshold in candidates
    ]

    # Sort by cost ascending, then by threshold descending, and take the first.
    # The second key is the tie-break: equal-cost thresholds are not equally
    # good operationally, and the higher one flags fewer people for the same
    # money - less human capacity needed to act on the same result.
    threshold, detail = min(scored, key=lambda item: (item[1]["total_cost"], -item[0]))
    return threshold, detail


def threshold_report(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    current_threshold: float,
    costs: CostModel | None = None,
) -> dict[str, Any]:
    """Compare the threshold in use against the cost-optimal one.

    The headline is ``savings``: what switching would be worth on this holdout,
    stated in currency rather than in a metric nobody outside the team can
    interpret.
    """
    model = costs or CostModel()
    empirical, empirical_detail = cost_optimal_threshold(y_true, y_proba, model)
    current_detail = expected_cost(y_true, y_proba, current_threshold, model)
    analytic = model.analytic_threshold
    analytic_detail = expected_cost(y_true, y_proba, analytic, model)

    savings = current_detail["total_cost"] - empirical_detail["total_cost"]

    return {
        "costs": {
            "false_negative": model.false_negative,
            "false_positive": model.false_positive,
            "offer_cost": model.offer_cost,
            "offer_efficacy": model.offer_efficacy,
            "true_positive": round(model.true_positive, 2),
            "ratio": round(model.ratio, 2),
        },
        "current": current_detail,
        "cost_optimal": empirical_detail,
        # Equal to the empirical optimum only when the probabilities are well
        # calibrated. The gap between them is calibration error, priced.
        "analytic": {**analytic_detail, "threshold": round(analytic, 4)},
        "analytic_vs_empirical_gap": round(abs(analytic - empirical), 4),
        "savings": round(float(savings), 2),
        "savings_per_subscriber": round(
            float(savings / max(len(np.asarray(y_true)), 1)), 4
        ),
        "extra_offers_required": (
            empirical_detail["flagged"] - current_detail["flagged"]
        ),
    }
