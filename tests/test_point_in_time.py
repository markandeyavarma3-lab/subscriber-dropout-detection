"""Tests for the temporal warehouse and point-in-time feature construction.

The leakage tests here are the most important in the suite.  Everything else
checks that the code does what it says; these check that the *training data
itself* is honest, and a failure means every metric the project reports is
worthless.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine

from src.features.build_features import REQUIRED_INPUT_COLUMNS
from src.features.point_in_time import (
    TrainingWindow,
    build_backfill,
    build_training_snapshot,
    monthly_cutoffs,
)
from src.warehouse import schema
from src.warehouse.database import create_schema, insert_rows, table_counts
from src.warehouse.simulate import DriftScenario, simulate_events

CUTOFF = date(2024, 9, 1)

# Feature columns only: the label is expected to change when future events are
# added, so it is compared separately where relevant.
FEATURE_COLUMNS = list(REQUIRED_INPUT_COLUMNS)


@pytest.fixture(scope="module")
def warehouse() -> Engine:
    """A populated in-memory warehouse shared by this module's tests."""
    engine = create_engine("sqlite:///:memory:", future=True)
    simulate_events(
        n_subscribers=800, start="2024-01-01", end="2024-12-31", seed=11, engine=engine
    )
    return engine


@pytest.fixture()
def empty_warehouse() -> Engine:
    """An empty warehouse with the schema created but no rows."""
    engine = create_engine("sqlite:///:memory:", future=True)
    create_schema(engine)
    return engine


def _snapshot(engine: Engine, cutoff: date = CUTOFF) -> pd.DataFrame:
    frame, _ = build_training_snapshot(cutoff, engine=engine)
    return frame


# --------------------------------------------------------------------------- #
# Leakage: the tests that matter most
# --------------------------------------------------------------------------- #


def test_events_after_the_cutoff_cannot_change_features(warehouse: Engine) -> None:
    """The core guarantee: features look strictly backwards.

    A snapshot is taken, a burst of activity is then recorded *after* the
    cutoff, and the snapshot is rebuilt.  If a single feature moves, the
    observation window is leaking the future and every offline metric the
    project reports is inflated.
    """
    before = _snapshot(warehouse).set_index("subscriber_id").sort_index()
    subject = before.index[0]

    after_cutoff = datetime.combine(CUTOFF, datetime.min.time()) + timedelta(days=3)
    insert_rows(
        schema.sessions,
        [
            {"subscriber_id": subject, "occurred_at": after_cutoff, "duration_minutes": 45.0}
            for _ in range(200)
        ],
        engine=warehouse,
    )
    insert_rows(
        schema.support_tickets,
        [{"subscriber_id": subject, "occurred_at": after_cutoff, "category": "billing"}] * 25,
        engine=warehouse,
    )
    insert_rows(
        schema.payments,
        [
            {
                "subscriber_id": subject,
                "occurred_at": after_cutoff,
                "amount": 19.99,
                "status": "failed",
                "discount_applied": True,
            }
        ]
        * 15,
        engine=warehouse,
    )

    after = _snapshot(warehouse).set_index("subscriber_id").sort_index()

    pd.testing.assert_frame_equal(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS])


def test_events_before_the_observation_window_are_excluded(empty_warehouse: Engine) -> None:
    """Only the observation window feeds the behavioural counts."""
    engine = empty_warehouse
    cutoff = datetime(2024, 6, 1)
    insert_rows(
        schema.subscribers,
        [{"subscriber_id": "S1", "signup_date": date(2023, 1, 1), "acquisition_channel": "organic"}],
        engine=engine,
    )
    insert_rows(
        schema.subscription_events,
        [
            {
                "subscriber_id": "S1",
                "event_type": schema.SIGNUP,
                "plan_type": "basic",
                "monthly_fee": 9.99,
                "is_auto_renew_enabled": True,
                "occurred_at": datetime(2023, 1, 1),
            }
        ],
        engine=engine,
    )
    insert_rows(
        schema.sessions,
        # Three inside the 30-day window, five long before it.
        [{"subscriber_id": "S1", "occurred_at": cutoff - timedelta(days=5), "duration_minutes": 10.0}] * 3
        + [{"subscriber_id": "S1", "occurred_at": cutoff - timedelta(days=200), "duration_minutes": 10.0}] * 5,
        engine=engine,
    )

    frame = _snapshot(engine, date(2024, 6, 1))
    assert frame.loc[0, "avg_session_count_last_30d"] == 3


def test_label_comes_only_from_the_prediction_horizon(empty_warehouse: Engine) -> None:
    """A cancellation beyond the horizon is not this snapshot's label."""
    engine = empty_warehouse
    cutoff = date(2024, 6, 1)
    insert_rows(
        schema.subscribers,
        [{"subscriber_id": "S1", "signup_date": date(2024, 1, 1), "acquisition_channel": "organic"}],
        engine=engine,
    )
    insert_rows(
        schema.subscription_events,
        [
            {
                "subscriber_id": "S1",
                "event_type": schema.SIGNUP,
                "plan_type": "basic",
                "monthly_fee": 9.99,
                "is_auto_renew_enabled": True,
                "occurred_at": datetime(2024, 1, 1),
            },
            {
                # 100 days after the cutoff: well outside a 30-day horizon.
                "subscriber_id": "S1",
                "event_type": schema.CANCELLATION,
                "plan_type": "basic",
                "monthly_fee": 9.99,
                "is_auto_renew_enabled": True,
                "occurred_at": datetime.combine(cutoff, datetime.min.time()) + timedelta(days=100),
            },
        ],
        engine=engine,
    )

    assert _snapshot(engine, cutoff).loc[0, "dropout"] == 0

    # Widening the horizon to cover it flips the label, proving the window is
    # what decides - not the mere existence of a cancellation.
    wide, _ = build_training_snapshot(cutoff, horizon_days=200, engine=engine)
    assert wide.loc[0, "dropout"] == 1


def test_already_cancelled_subscribers_are_excluded(warehouse: Engine) -> None:
    """Someone who left before the cutoff is not at risk of leaving again."""
    frame = _snapshot(warehouse, date(2024, 12, 1))
    cancelled = pd.read_sql_query(
        "SELECT DISTINCT subscriber_id FROM subscription_events "
        "WHERE event_type = 'cancellation' AND occurred_at < '2024-12-01'",
        warehouse,
    )
    assert not set(frame["subscriber_id"]) & set(cancelled["subscriber_id"])


def test_subscribers_who_signed_up_after_the_cutoff_are_absent(warehouse: Engine) -> None:
    """A subscriber who does not yet exist cannot be scored."""
    frame = _snapshot(warehouse, date(2024, 3, 1))
    signups = pd.read_sql_query("SELECT subscriber_id, signup_date FROM subscribers", warehouse)
    late = set(signups[pd.to_datetime(signups["signup_date"]) >= "2024-03-01"]["subscriber_id"])
    assert not set(frame["subscriber_id"]) & late


# --------------------------------------------------------------------------- #
# Window arithmetic
# --------------------------------------------------------------------------- #


def test_training_window_boundaries_do_not_overlap() -> None:
    """Feature and label windows must be disjoint by construction."""
    window = TrainingWindow(cutoff=date(2024, 6, 1), observation_days=30, horizon_days=30)
    assert window.window_start == date(2024, 5, 2)
    assert window.horizon_end == date(2024, 7, 1)
    assert window.window_start < window.cutoff < window.horizon_end


def test_training_window_summary_is_json_friendly() -> None:
    """The window is recorded next to trained models, so it must serialise."""
    summary = TrainingWindow(date(2024, 6, 1), 30, 30).summary()
    assert summary["cutoff"] == "2024-06-01"
    assert summary["observation_days"] == 30


def test_tenure_is_measured_at_the_cutoff_not_today(warehouse: Engine) -> None:
    """Tenure is an as-of quantity; using the wall clock would leak."""
    early = _snapshot(warehouse, date(2024, 6, 1)).set_index("subscriber_id")
    later = _snapshot(warehouse, date(2024, 9, 1)).set_index("subscriber_id")
    shared = early.index.intersection(later.index)

    delta = later.loc[shared, "tenure_days"] - early.loc[shared, "tenure_days"]
    assert (delta == 92).all()


def test_last_activity_never_exceeds_tenure(warehouse: Engine) -> None:
    """The API rejects this combination, so training data must not contain it."""
    frame = _snapshot(warehouse)
    assert (frame["last_activity_days_ago"] <= frame["tenure_days"]).all()


# --------------------------------------------------------------------------- #
# Serving contract
# --------------------------------------------------------------------------- #


def test_snapshot_carries_every_serving_column(warehouse: Engine) -> None:
    """Warehouse output must satisfy the same contract the API accepts."""
    frame = _snapshot(warehouse)
    assert set(REQUIRED_INPUT_COLUMNS).issubset(frame.columns)
    assert "dropout" in frame.columns


def test_snapshot_has_no_missing_values(warehouse: Engine) -> None:
    """Left joins must be filled, not left as NaN for the imputer to hide."""
    frame = _snapshot(warehouse)
    assert not frame[REQUIRED_INPUT_COLUMNS].isna().any().any()


def test_snapshot_is_plausible(warehouse: Engine) -> None:
    """Sanity bounds on a generated population."""
    frame = _snapshot(warehouse)
    assert len(frame) > 50
    assert 0.02 < frame["dropout"].mean() < 0.6
    assert (frame["tenure_days"] >= 0).all()
    assert frame["plan_type"].isin({"basic", "standard", "premium"}).all()


def test_empty_warehouse_returns_an_empty_frame(empty_warehouse: Engine) -> None:
    """No data is an empty result, not an exception."""
    frame, _ = build_training_snapshot("2024-06-01", engine=empty_warehouse)
    assert frame.empty


# --------------------------------------------------------------------------- #
# Backfill
# --------------------------------------------------------------------------- #


def test_backfill_stacks_multiple_cutoffs(warehouse: Engine) -> None:
    """Backfilling is what makes scheduled retraining do real work."""
    cutoffs = [date(2024, 6, 1), date(2024, 7, 1), date(2024, 8, 1)]
    panel = build_backfill(cutoffs, engine=warehouse)

    assert set(panel["cutoff"]) == {c.isoformat() for c in cutoffs}
    assert len(panel) > len(_snapshot(warehouse, cutoffs[0]))


def test_backfill_rows_are_unique_per_subscriber_and_cutoff(warehouse: Engine) -> None:
    """A subscriber appears once per cutoff, not once overall."""
    panel = build_backfill([date(2024, 6, 1), date(2024, 7, 1)], engine=warehouse)
    assert not panel.duplicated(subset=["subscriber_id", "cutoff"]).any()
    assert panel.duplicated(subset=["subscriber_id"]).any()


def test_monthly_cutoffs_are_evenly_spaced() -> None:
    """Cutoff generation is inclusive of the start and bounded by the end."""
    cutoffs = monthly_cutoffs("2024-01-01", "2024-04-01", step_days=30)
    assert cutoffs[0] == date(2024, 1, 1)
    assert all(
        (later - earlier).days == 30
        for earlier, later in zip(cutoffs, cutoffs[1:], strict=False)
    )
    assert cutoffs[-1] <= date(2024, 4, 1)


def test_empty_cutoff_list_returns_empty_panel(warehouse: Engine) -> None:
    """Backfilling nothing yields nothing."""
    assert build_backfill([], engine=warehouse).empty


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #


def test_simulation_populates_every_table(warehouse: Engine) -> None:
    """All five event tables receive rows."""
    counts = table_counts(warehouse)
    assert all(count > 0 for count in counts.values())
    # Sessions are by far the highest-volume table, as in a real service.
    assert counts["sessions"] > counts["payments"]


def test_simulation_is_reproducible() -> None:
    """The same seed produces the same warehouse."""
    def run(seed: int) -> dict[str, int]:
        engine = create_engine("sqlite:///:memory:", future=True)
        simulate_events(
            n_subscribers=200, start="2024-01-01", end="2024-04-01", seed=seed, engine=engine
        )
        return table_counts(engine)

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_simulation_rejects_a_backwards_date_range() -> None:
    """An end before the start is a caller error, caught early."""
    with pytest.raises(ValueError, match="must be after"):
        simulate_events(n_subscribers=10, start="2024-06-01", end="2024-01-01")


def test_drift_scenario_suppresses_engagement() -> None:
    """An injected engagement collapse is visible in the warehouse."""
    def sessions_for(scenario: DriftScenario | None) -> int:
        engine = create_engine("sqlite:///:memory:", future=True)
        simulate_events(
            n_subscribers=300,
            start="2024-01-01",
            end="2024-06-30",
            seed=3,
            scenario=scenario,
            engine=engine,
        )
        return table_counts(engine)["sessions"]

    collapsed = DriftScenario(starts_on=date(2024, 4, 1), engagement_multiplier=0.2)
    assert sessions_for(collapsed) < sessions_for(None)


def test_drift_scenario_introduces_an_unseen_plan() -> None:
    """A new tier appears only after the scenario's start date."""
    engine = create_engine("sqlite:///:memory:", future=True)
    simulate_events(
        n_subscribers=400,
        start="2024-01-01",
        end="2024-12-31",
        seed=7,
        scenario=DriftScenario(starts_on=date(2024, 7, 1), new_plan_share=0.8),
        engine=engine,
    )

    plans = pd.read_sql_query(
        "SELECT plan_type, MIN(occurred_at) AS first_seen FROM subscription_events "
        "GROUP BY plan_type",
        engine,
    ).set_index("plan_type")

    assert "enterprise" in plans.index
    assert pd.to_datetime(plans.loc["enterprise", "first_seen"]) >= pd.Timestamp("2024-07-01")


def test_scenario_is_inactive_before_its_start_date() -> None:
    """The scenario is a step change, not a global setting."""
    scenario = DriftScenario(starts_on=date(2024, 6, 1))
    assert not scenario.active_on(date(2024, 5, 31))
    assert scenario.active_on(date(2024, 6, 1))
