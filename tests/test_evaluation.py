"""Tests for calibration, cost-based thresholds and fairness.

These three answer one question the accuracy metrics cannot: is this model fit
for the decision it actually drives? A model can have an excellent AUC and
still produce probabilities that mean nothing, a threshold that loses money,
and predictions that work far better for some people than others.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.fairness import audit, disparity_report, group_metrics
from src.models.calibration import (
    calibrate_pipeline,
    calibration_metrics,
    compare_calibration,
    expected_calibration_error,
    reliability_table,
)
from src.models.costs import (
    CapacityConstraint,
    CostModel,
    capacity_report,
    capacity_threshold,
    constrained_threshold,
    cost_optimal_threshold,
    expected_cost,
    threshold_report,
)

# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_perfect_calibration_scores_zero_error() -> None:
    """A model whose stated rate matches the observed rate exactly."""
    # 1000 rows at p=0.3, of which exactly 30% are positive.
    proba = np.full(1000, 0.3)
    truth = np.array([1] * 300 + [0] * 700)

    assert expected_calibration_error(truth, proba, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_systematic_overconfidence_is_detected() -> None:
    """Claiming 90% for a group that churns 30% of the time."""
    proba = np.full(1000, 0.9)
    truth = np.array([1] * 300 + [0] * 700)

    assert expected_calibration_error(truth, proba) == pytest.approx(0.6, abs=1e-6)


def test_calibration_metrics_expose_directional_bias() -> None:
    """`mean_bias` says which way the model is wrong, not just how much."""
    proba = np.full(100, 0.8)
    truth = np.array([1] * 20 + [0] * 80)

    metrics = calibration_metrics(truth, proba)
    assert metrics["mean_bias"] == pytest.approx(0.6, abs=1e-6)
    assert metrics["observed_rate"] == pytest.approx(0.2)


def test_ece_weights_bins_by_population() -> None:
    """A wildly wrong bin holding three rows must not swamp the score."""
    # 997 well-calibrated rows, 3 catastrophically wrong ones.
    proba = np.concatenate([np.full(997, 0.0), np.full(3, 1.0)])
    truth = np.concatenate([np.zeros(997), np.zeros(3)])

    # The bad bin is 0.3% of the population, so it contributes ~0.003.
    assert expected_calibration_error(truth, proba) < 0.01


def test_reliability_table_covers_the_whole_range() -> None:
    """Empty bins are reported as empty, not silently omitted."""
    table = reliability_table(np.array([0, 1]), np.array([0.05, 0.95]), n_bins=10)

    assert len(table) == 10
    assert table["n"].sum() == 2
    assert table[table["n"] == 0]["mean_predicted"].isna().all()


def test_a_prediction_of_exactly_one_lands_in_the_last_bin() -> None:
    """Off-by-one at the boundary would drop rows from the report entirely."""
    table = reliability_table(np.array([1]), np.array([1.0]), n_bins=10)

    assert table.iloc[-1]["n"] == 1
    assert table["n"].sum() == 1


def test_compare_calibration_reports_failure_honestly() -> None:
    """The guard that matters: calibration is not guaranteed to help.

    Isotonic regression can overfit a small calibration split and make the
    probabilities worse. Applying it blindly is how a project ends up with
    worse probabilities and a README claiming otherwise.
    """
    truth = np.array([1] * 300 + [0] * 700)
    good = np.full(1000, 0.3)  # perfectly calibrated
    bad = np.full(1000, 0.9)  # wildly overconfident

    # "After" is worse than "before", and the report must say so.
    result = compare_calibration(truth, good, bad)

    assert result["improved"] is False
    assert result["ece_improvement"] < 0


def test_calibrated_pipeline_keeps_the_predict_proba_interface(
    trained_model, sample_features, sample_frame
) -> None:
    """Everything downstream must keep working after wrapping."""
    target = sample_frame["dropout"]
    calibrated = calibrate_pipeline(trained_model.model, sample_features, target)

    proba = calibrated.predict_proba(sample_features)[:, 1]
    assert proba.shape == (len(sample_features),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_calibration_method_is_rejected(trained_model, sample_features, sample_frame) -> None:
    """A typo must fail loudly rather than silently pick a default."""
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrate_pipeline(
            trained_model.model, sample_features, sample_frame["dropout"], method="platt"
        )


# --------------------------------------------------------------------------- #
# Cost-based thresholds
# --------------------------------------------------------------------------- #


def test_offer_efficacy_drives_the_threshold() -> None:
    """The parameter that decides whether the exercise is honest.

    Assuming every correctly-aimed offer works makes blanket outreach optimal -
    which is arithmetically true and operationally absurd. A realistic efficacy
    pushes the threshold up sharply.
    """
    always_works = CostModel(false_negative=240, false_positive=20, offer_cost=20, offer_efficacy=1.0)
    realistic = CostModel(false_negative=240, false_positive=20, offer_cost=20, offer_efficacy=0.3)

    assert realistic.analytic_threshold > always_works.analytic_threshold * 2


def test_a_useless_offer_means_never_act() -> None:
    """If intervening cannot save anyone, no threshold is worth crossing."""
    hopeless = CostModel(false_negative=240, false_positive=20, offer_cost=20, offer_efficacy=0.0)

    assert hopeless.analytic_threshold == 1.0


def test_expected_cost_counts_every_error_type() -> None:
    """Hand-checked arithmetic on a known confusion matrix."""
    truth = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.1, 0.9, 0.1])
    costs = CostModel(false_negative=100, false_positive=10, offer_cost=5, offer_efficacy=1.0)

    outcome = expected_cost(truth, proba, threshold=0.5, costs=costs)

    assert outcome["true_positives"] == 1
    assert outcome["false_negatives"] == 1
    assert outcome["false_positives"] == 1
    assert outcome["true_negatives"] == 1
    # 1 FN x 100 + 1 FP x 10 + 1 TP x 5 (efficacy 1.0 so TP cost is just the offer)
    assert outcome["total_cost"] == pytest.approx(115.0)


def test_the_cheapest_threshold_is_chosen() -> None:
    """A cost sweep must actually find the minimum."""
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=500)
    truth = (proba + rng.normal(0, 0.2, 500) > 0.6).astype(int)

    threshold, detail = cost_optimal_threshold(truth, proba, CostModel())
    sweep = [
        expected_cost(truth, proba, t, CostModel())["total_cost"] for t in np.arange(0.05, 1.0, 0.05)
    ]

    assert detail["total_cost"] <= min(sweep)
    assert 0.0 < threshold < 1.0


def test_ties_prefer_the_higher_threshold() -> None:
    """Equal cost is not equal value: the higher threshold flags fewer people.

    Fewer flags means less human capacity needed for the same money.
    """
    # Nobody churns, so every threshold above the max score costs nothing.
    truth = np.zeros(50, dtype=int)
    proba = np.full(50, 0.1)

    threshold, detail = cost_optimal_threshold(truth, proba, CostModel())

    assert detail["total_cost"] == 0.0
    assert threshold > 0.1  # flags nobody, rather than an equally free lower cut-off


def test_threshold_report_states_savings_in_currency() -> None:
    """The output has to be legible to somebody outside the team."""
    rng = np.random.default_rng(1)
    proba = rng.uniform(size=400)
    truth = (proba + rng.normal(0, 0.25, 400) > 0.65).astype(int)

    report = threshold_report(truth, proba, current_threshold=0.9, costs=CostModel())

    assert report["savings"] > 0  # 0.9 is far too conservative at a 12:1 ratio
    assert "cost_per_subscriber" in report["current"]
    assert report["extra_offers_required"] > 0
    assert report["costs"]["offer_efficacy"] == pytest.approx(0.30)


def test_the_analytic_and_empirical_thresholds_are_reported_together() -> None:
    """Their gap is calibration error, priced - a more legible argument than ECE."""
    rng = np.random.default_rng(2)
    proba = rng.uniform(size=300)
    truth = (proba > 0.5).astype(int)

    report = threshold_report(truth, proba, current_threshold=0.5)

    assert "analytic" in report
    assert "cost_optimal" in report
    assert report["analytic_vs_empirical_gap"] >= 0


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #


@pytest.fixture()
def biased_predictions():
    """Two groups where the model ranks one well and the other barely at all."""
    rng = np.random.default_rng(7)
    n = 400

    # Group A: the score genuinely separates churners.
    a_truth = rng.binomial(1, 0.3, n)
    a_proba = np.clip(a_truth * 0.5 + rng.uniform(0, 0.4, n), 0, 1)

    # Group B: the score is noise - same base rate, no signal.
    b_truth = rng.binomial(1, 0.3, n)
    b_proba = rng.uniform(0, 1, n)

    truth = np.concatenate([a_truth, b_truth])
    proba = np.concatenate([a_proba, b_proba])
    groups = np.array(["works"] * n + ["broken"] * n)
    return truth, proba, groups


def test_group_metrics_reports_one_row_per_group(biased_predictions) -> None:
    truth, proba, groups = biased_predictions
    table = group_metrics(truth, proba, groups, threshold=0.5)

    assert set(table["group"]) == {"works", "broken"}
    assert table["n"].sum() == len(truth)


def test_a_group_the_model_cannot_rank_is_identified(biased_predictions) -> None:
    """The finding an aggregate AUC hides completely."""
    truth, proba, groups = biased_predictions
    report = disparity_report(truth, proba, groups, threshold=0.5, attribute="cohort")

    assert report["weakest_group"]["group"] == "broken"
    # Around 0.5 - the model is guessing for this group.
    assert report["weakest_group"]["roc_auc"] < 0.6
    assert not report["passes"]
    assert any("discrimination" in c for c in report["concerns"])


def test_small_groups_are_flagged_but_never_dropped() -> None:
    """Silently excluding a small group is how a minority's problem disappears."""
    truth = np.array([1, 0] * 60 + [1, 0])
    proba = np.array([0.9, 0.1] * 60 + [0.9, 0.1])
    groups = np.array(["big"] * 120 + ["tiny"] * 2)

    report = disparity_report(truth, proba, groups, threshold=0.5, min_size=30)
    names = {g["group"] for g in report["groups"]}

    assert "tiny" in names  # present in the report
    assert report["n_groups"] == 2
    assert report["n_groups_evaluated"] == 1  # but excluded from the ratios
    tiny = next(g for g in report["groups"] if g["group"] == "tiny")
    assert tiny["sufficient_sample"] is False


def test_equal_groups_pass() -> None:
    """No false alarms when the model treats everyone the same.

    Base rates are equal *by construction*, not by random draw. Three binomial
    samples of 200 rows routinely differ by ten percentage points, and the
    resulting selection-rate gap is a real difference in churn rate rather than
    a fairness problem - which is precisely the distinction this module makes.
    """
    rng = np.random.default_rng(3)
    # Identical label pattern in every group: 30% positive, exactly.
    per_group = np.array([1] * 60 + [0] * 140)
    truth = np.tile(per_group, 3)
    proba = np.clip(truth * 0.5 + rng.uniform(0, 0.4, 600), 0, 1)
    groups = np.repeat(["a", "b", "c"], 200)

    report = disparity_report(truth, proba, groups, threshold=0.5)

    assert report["passes"] is True
    assert report["concerns"] == []


def test_a_group_with_one_class_reports_no_auc() -> None:
    """AUC is undefined without both classes - missing, not zero.

    It must also be ``None`` rather than ``NaN``: these records are serialised
    into reports and artifacts, and NaN is not valid JSON.
    """
    truth = np.concatenate([np.array([1, 0] * 20), np.ones(40, dtype=int)])
    proba = np.linspace(0.1, 0.9, 80)
    groups = np.array(["mixed"] * 40 + ["all_churn"] * 40)

    table = group_metrics(truth, proba, groups, threshold=0.5, min_size=1)
    all_churn = table[table["group"] == "all_churn"].iloc[0]

    assert all_churn["roc_auc"] is None
    assert all_churn["recall"] is not None  # recall is still defined


def test_undefined_metrics_survive_json_serialisation() -> None:
    """NaN would make a fairness report unserialisable the moment it is stored."""
    import json

    truth = np.concatenate([np.array([1, 0] * 20), np.ones(40, dtype=int)])
    proba = np.linspace(0.1, 0.9, 80)
    groups = np.array(["mixed"] * 40 + ["all_churn"] * 40)

    report = disparity_report(truth, proba, groups, threshold=0.5, min_size=1)
    encoded = json.dumps(report)

    assert "NaN" not in encoded
    assert json.loads(encoded)["attribute"] == "group"


def test_selection_and_performance_disparity_are_distinguished() -> None:
    """Different meanings: one may be correct, the other is hard to defend."""
    rng = np.random.default_rng(5)
    n = 300
    # Group B genuinely churns more, so being flagged more is correct.
    a_truth = rng.binomial(1, 0.1, n)
    b_truth = rng.binomial(1, 0.5, n)
    a_proba = np.clip(a_truth * 0.6 + rng.uniform(0, 0.3, n), 0, 1)
    b_proba = np.clip(b_truth * 0.6 + rng.uniform(0, 0.3, n), 0, 1)

    report = disparity_report(
        np.concatenate([a_truth, b_truth]),
        np.concatenate([a_proba, b_proba]),
        np.array(["low_risk"] * n + ["high_risk"] * n),
        threshold=0.5,
    )

    # Selection rates differ a lot, but both groups are ranked well.
    assert report["selection_rate_ratio"] < 0.8
    assert report["roc_auc_ratio"] > 0.8
    assert any("selection rates" in c for c in report["concerns"])
    assert not any("discrimination" in c for c in report["concerns"])


def test_audit_covers_several_attributes_at_once(biased_predictions) -> None:
    """One call, one overall verdict, detail preserved per attribute."""
    truth, proba, groups = biased_predictions
    balanced = np.array(["x", "y"] * (len(truth) // 2))

    report = audit(truth, proba, {"cohort": groups, "balanced": balanced}, threshold=0.5)

    assert set(report["reports"]) == {"cohort", "balanced"}
    assert report["passes"] is False
    assert "cohort" in report["attributes_with_concerns"]


# --------------------------------------------------------------------------- #
# Outreach capacity
# --------------------------------------------------------------------------- #


def _scored_population(n: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """A holdout where score and outcome are related but not identical."""
    rng = np.random.default_rng(seed)
    truth = rng.binomial(1, 0.2, n)
    proba = np.clip(rng.beta(2, 6, n) + 0.3 * truth, 0.0, 1.0)
    return truth, proba


def test_capacity_cannot_be_stated_two_ways_at_once() -> None:
    """An absolute cap and a rate disagree on every population but one.

    Accepting both would make which one wins an implementation detail, and the
    loser would be silently ignored.
    """
    with pytest.raises(ValueError, match="not both"):
        CapacityConstraint(max_offers=100, max_offer_rate=0.05)


def test_no_capacity_is_different_from_zero_capacity() -> None:
    """"We can contact nobody this cycle" is a real state, not an absent one."""
    assert CapacityConstraint().configured is False
    assert CapacityConstraint(max_offers=0).configured is True
    assert CapacityConstraint(max_offers=0).offers_for(500) == 0


def test_a_rate_resolves_against_the_population_and_rounds_down() -> None:
    """Rounding a budget up is how a campaign goes over."""
    assert CapacityConstraint(max_offer_rate=0.05).offers_for(1_199) == 59
    # An absolute cap larger than the population is capped by the population.
    assert CapacityConstraint(max_offers=10_000).offers_for(300) == 300


def test_capacity_threshold_selects_exactly_the_top_k() -> None:
    """Top-k by score, expressed as the threshold the serving path applies."""
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    for k in range(1, 7):
        threshold = capacity_threshold(proba, k)
        assert int((proba >= threshold).sum()) == k


def test_ties_at_the_cut_off_under_fill_rather_than_overspend() -> None:
    """Handing operations 530 names against a cap of 500 is not a rounding detail.

    Four subscribers share the boundary score, so no threshold flags exactly
    three. The conservative choice is two.
    """
    proba = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1])

    flagged = int((proba >= capacity_threshold(proba, 3)).sum())
    assert flagged <= 3


def test_a_capacity_of_zero_flags_nobody() -> None:
    proba = np.array([0.99, 0.5, 0.1])
    assert int((proba >= capacity_threshold(proba, 0)).sum()) == 0


def test_capacity_larger_than_the_population_does_not_constrain() -> None:
    proba = np.array([0.9, 0.5, 0.1])
    assert capacity_threshold(proba, 99) == 0.0


def test_a_spare_slot_is_not_a_reason_to_use_it() -> None:
    """The constraint only ever raises the threshold, never lowers it.

    Below the unconstrained optimum an offer loses money whether or not the
    slot was free - a budget is permission to spend, not an obligation.
    """
    truth, proba = _scored_population()
    unconstrained, _ = cost_optimal_threshold(truth, proba, CostModel())

    generous = constrained_threshold(truth, proba, len(proba), CostModel())
    assert generous == pytest.approx(unconstrained)

    tight = constrained_threshold(truth, proba, 20, CostModel())
    assert tight > unconstrained


def test_the_capacity_curve_is_computed_without_a_budget_being_set() -> None:
    """The useful output needs no budget number to exist.

    Pricing each extra offer slot is the argument you take to whoever owns the
    budget - it cannot require the budget to already be decided.
    """
    truth, proba = _scored_population()
    report = capacity_report(truth, proba, CostModel(), CapacityConstraint())

    assert report["configured"] is False
    assert report["max_offers"] is None
    assert report["constrained"] is None
    assert report["binding"] is False
    assert len(report["marginal_value"]) > 1


def test_more_budget_never_costs_more() -> None:
    """The one hard invariant of the ladder.

    Whatever a budget of k can do, a budget of k+1 can also do by leaving the
    extra slot unused - so total cost must be non-increasing in budget. A
    violation would mean the constrained threshold is spending slots it should
    not.
    """
    truth, proba = _scored_population()
    report = capacity_report(truth, proba, CostModel(), CapacityConstraint())

    costs = [step["total_cost"] for step in report["marginal_value"]]
    assert costs == sorted(costs, reverse=True)


def test_early_offer_slots_are_worth_more_than_late_ones() -> None:
    """Diminishing returns, priced - on average, not step by step.

    The model ranks, so the first slots go to the most likely churners. What it
    does *not* guarantee is that every consecutive block is worth less than the
    one before: on a finite holdout an imperfect ranking will sometimes put an
    unusually churn-heavy block below a lighter one. Asserting strict
    monotonicity here would be asserting that the ranking is perfect.
    """
    truth, proba = _scored_population()
    report = capacity_report(truth, proba, CostModel(), CapacityConstraint())

    priced = [
        step["value_per_extra_slot"]
        for step in report["marginal_value"]
        if step["value_per_extra_slot"] is not None
    ]
    half = len(priced) // 2
    assert sum(priced[:half]) / half > sum(priced[half:]) / len(priced[half:])


def test_slots_beyond_the_optimum_are_worth_nothing() -> None:
    """Budget past the break-even point buys exactly zero, not a small loss.

    That is the threshold floor doing its job: the extra slots go unused rather
    than being spent on people it does not pay to contact.
    """
    truth, proba = _scored_population()
    report = capacity_report(truth, proba, CostModel(), CapacityConstraint())

    wanted = report["unconstrained"]["offers_required"]
    beyond = [step for step in report["marginal_value"] if step["max_offers"] > wanted]
    assert beyond, "the ladder should price budgets above the optimum"
    for step in beyond:
        assert step["value_per_extra_slot"] == pytest.approx(0.0)
        assert step["offers_used"] <= wanted


def test_a_binding_budget_is_reported_with_what_it_costs() -> None:
    """The gap between what the model wants and what anyone can send."""
    truth, proba = _scored_population()
    report = capacity_report(
        truth, proba, CostModel(), CapacityConstraint(max_offer_rate=0.02)
    )

    assert report["binding"] is True
    assert report["shortfall"] > 0
    assert report["constrained"]["flagged"] <= report["max_offers"]
    # A tighter budget cannot be cheaper than the unconstrained optimum, which
    # is the cheapest point by construction.
    assert report["cost_of_constraint"] > 0


def test_threshold_report_carries_the_capacity_reality_check() -> None:
    """Savings from contacting 40% of the base are not available to a team
    that can contact 5%, and the report must say so in the same place."""
    truth, proba = _scored_population()
    report = threshold_report(truth, proba, 0.5, CostModel())

    assert "capacity" in report
    assert (
        report["capacity"]["unconstrained"]["offers_required"]
        == report["cost_optimal"]["flagged"]
    )
