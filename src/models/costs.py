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

The capacity constraint
-----------------------

Both methods above share an assumption that no retention team has ever been
able to make: that you can act on everyone the threshold flags. In practice
there is a budget, a contact-frequency policy, and a finite number of people to
run the campaign. Under a hard cap of ``k`` offers the optimal policy is no
longer "everyone above t" - it is **the top k by score**, which is still a
threshold, just one the score distribution picks rather than the arithmetic.

The two combine cleanly: the constrained threshold is
``max(unconstrained_optimum, kth_highest_score)``. The cap raises the bar when
it binds; below the unconstrained optimum a slot is not worth using even when
it is free, because contacting that person loses money whether or not you had
the capacity.

:func:`capacity_report` computes this whether or not a cap is configured,
because the interesting output needs no budget number to exist: the marginal
value of one more offer slot. That is the argument somebody takes to the
person who owns the budget.
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


@dataclass(frozen=True)
class CapacityConstraint:
    """How many retention offers can actually be made per scoring cycle.

    Expressed either as an absolute count or as a fraction of the scored
    population, never both - they are two ways of saying the same thing and
    accepting both at once invites a silent contradiction.

    The rate is the portable one. An absolute cap tuned against a 200,000
    subscriber base is meaningless applied to a 1,200-row holdout, and the
    holdout is where every number in this module is computed.
    """

    max_offers: int | None = None
    max_offer_rate: float | None = None

    def __post_init__(self) -> None:
        if self.max_offers is not None and self.max_offer_rate is not None:
            raise ValueError(
                "Set max_offers or max_offer_rate, not both: they express the "
                "same constraint and would disagree on any population size."
            )
        if self.max_offers is not None and self.max_offers < 0:
            raise ValueError("max_offers cannot be negative.")
        if self.max_offer_rate is not None and not 0.0 <= self.max_offer_rate <= 1.0:
            raise ValueError("max_offer_rate is a fraction of the population.")

    @classmethod
    def from_settings(cls) -> CapacityConstraint:
        """Build from configuration, which leaves both unset by default."""
        return cls(
            max_offers=settings.RETENTION_CAPACITY,
            max_offer_rate=settings.RETENTION_CAPACITY_RATE,
        )

    @property
    def configured(self) -> bool:
        """Whether any cap is in force.

        Note that ``max_offers=0`` is configured: "we can contact nobody this
        cycle" is a real and occasionally true statement, which is why the
        unset case is ``None`` rather than zero.
        """
        return self.max_offers is not None or self.max_offer_rate is not None

    def offers_for(self, population: int) -> int | None:
        """Resolve the cap against a population size, or ``None`` if unset.

        Rounds down. Rounding a budget up is how a campaign goes over.
        """
        if self.max_offers is not None:
            return min(int(self.max_offers), int(population))
        if self.max_offer_rate is not None:
            return int(np.floor(self.max_offer_rate * population))
        return None


def capacity_threshold(y_proba: np.ndarray, max_offers: int) -> float:
    """The lowest score that still fits inside ``max_offers`` offers.

    This is what "top k by score" looks like expressed as a threshold, which is
    what the serving path actually applies.

    Ties are broken *conservatively*: if several subscribers share the cut-off
    score, the threshold rises to the next distinct value up, so the campaign
    under-fills rather than overspending its budget. Handing an operations team
    530 names against a cap of 500 is not a rounding detail to them.
    """
    scores = np.sort(np.asarray(y_proba, dtype=float))[::-1]
    population = scores.size

    if population == 0:
        return 0.0
    if max_offers <= 0:
        # No threshold in [0, 1] flags nobody, so step just past the top score.
        return float(np.nextafter(scores[0], np.inf))
    if max_offers >= population:
        return 0.0

    cut = scores[max_offers - 1]
    if scores[max_offers] == cut:
        higher = scores[scores > cut]
        cut = higher[-1] if higher.size else np.nextafter(cut, np.inf)
    return float(cut)


def constrained_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    max_offers: int,
    costs: CostModel | None = None,
) -> float:
    """The cheapest threshold that also fits the budget.

    ``max(unconstrained_optimum, kth_highest_score)``. The cap only ever raises
    the bar: below the unconstrained optimum an offer loses money whether or
    not a slot was free, so a spare slot is not a reason to use it.
    """
    unconstrained, _ = cost_optimal_threshold(y_true, y_proba, costs)
    return max(unconstrained, capacity_threshold(y_proba, max_offers))


def _budget_ladder(unconstrained_flagged: int, population: int) -> list[int]:
    """Budget levels to price, spanning either side of the unconstrained answer.

    Anchored on what the model *wants* to send rather than on round numbers:
    the useful question is "what does 25% of the recommended list cost me", not
    "what does 100 offers cost me".

    The resulting curve trends downward but is not monotone step by step. That
    is honest, not a defect: the ranking is imperfect, so on a finite holdout
    one block of slots will occasionally contain more churners than the block
    above it. Read the shape, not the individual steps.
    """
    steps = {
        int(round(unconstrained_flagged * fraction))
        for fraction in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    }
    return sorted({step for step in steps if 0 < step <= population})


def capacity_report(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    costs: CostModel | None = None,
    capacity: CapacityConstraint | None = None,
) -> dict[str, Any]:
    """What a finite outreach budget costs, and what one more slot would buy.

    Computed whether or not a cap is configured. The headline output needs no
    budget number to exist: ``marginal_value`` prices each extra offer slot, so
    the case for more budget - or the case that the current budget is already
    past the point of diminishing returns - can be made in currency rather than
    in a shrug.
    """
    model = costs or CostModel()
    limit = capacity or CapacityConstraint()
    scores = np.asarray(y_proba, dtype=float)
    population = int(scores.size)

    unconstrained_threshold, unconstrained = cost_optimal_threshold(y_true, scores, model)
    wanted = int(unconstrained["flagged"])

    # Price a ladder of budgets. Each entry is the *total* cost of operating
    # with that many slots, so the differences between them are the marginal
    # value of the slots in between.
    ladder: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for offers in _budget_ladder(wanted, population):
        threshold = constrained_threshold(y_true, scores, offers, model)
        detail = expected_cost(y_true, scores, threshold, model)
        entry = {
            "max_offers": offers,
            "threshold": detail["threshold"],
            "offers_used": detail["flagged"],
            "total_cost": detail["total_cost"],
            "value_per_extra_slot": None,
        }
        if previous is not None:
            gained = offers - previous["max_offers"]
            saved = previous["total_cost"] - detail["total_cost"]
            # Negative means the extra slots were spent below the break-even
            # score and actively lost money - which the constrained threshold
            # prevents, so in practice this floors at zero.
            entry["value_per_extra_slot"] = round(float(saved / gained), 2) if gained else None
        ladder.append(entry)
        previous = entry

    report: dict[str, Any] = {
        "population": population,
        "configured": limit.configured,
        "max_offers": limit.offers_for(population),
        "unconstrained": {
            "threshold": unconstrained["threshold"],
            "offers_required": wanted,
            "total_cost": unconstrained["total_cost"],
        },
        "marginal_value": ladder,
    }

    allowed = report["max_offers"]
    if allowed is None:
        # No cap in force. Say so explicitly rather than reporting a
        # constrained result that silently equals the unconstrained one.
        report["binding"] = False
        report["constrained"] = None
        report["shortfall"] = 0
        report["cost_of_constraint"] = 0.0
        return report

    threshold = constrained_threshold(y_true, scores, allowed, model)
    detail = expected_cost(y_true, scores, threshold, model)
    report["constrained"] = detail
    # Binding means the budget, not the economics, is what stopped you.
    report["binding"] = bool(detail["flagged"] < wanted)
    report["shortfall"] = max(wanted - allowed, 0)
    # What the constraint costs: the gap between the budget you have and the
    # budget the model would spend if you let it.
    report["cost_of_constraint"] = round(
        float(detail["total_cost"] - unconstrained["total_cost"]), 2
    )
    return report


def threshold_report(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    current_threshold: float,
    costs: CostModel | None = None,
    capacity: CapacityConstraint | None = None,
) -> dict[str, Any]:
    """Compare the threshold in use against the cost-optimal one.

    The headline is ``savings``: what switching would be worth on this holdout,
    stated in currency rather than in a metric nobody outside the team can
    interpret.

    ``capacity`` carries the reality check on that number: savings computed by
    flagging 40% of the base are not available to a team that can contact 5%.
    """
    model = costs or CostModel()
    limit = capacity if capacity is not None else CapacityConstraint.from_settings()
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
        # The savings above assume every flagged subscriber is actually
        # contacted. This says whether that is affordable, and prices each
        # extra offer slot if it is not.
        "capacity": capacity_report(y_true, y_proba, model, limit),
    }
