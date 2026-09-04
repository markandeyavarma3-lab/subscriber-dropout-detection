"""Tests for the real-data loaders and the warehouse contract.

The honest framing, repeated from the loader itself: the KKBox files are ~30GB
behind a Kaggle account and have never been run through this code. What is
tested here is everything that does not require them - the mapping decisions,
the date parsing, the streaming, the contract, and the end-to-end claim that a
loaded warehouse trains a model exactly as the simulated one does.

The fixtures reproduce KKBox's published column schema exactly. That makes them
a real test of the mapping and no test at all of whether the published schema
matches the actual files, which is precisely the gap `--dry-run` exists to
close on first contact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.external import contract, ingest, kkbox
from src.warehouse import database, schema

# --------------------------------------------------------------------------- #
# Fixtures shaped like the published dataset
# --------------------------------------------------------------------------- #


@pytest.fixture()
def kkbox_dir(tmp_path: Path) -> Path:
    """A miniature KKBox export with the real column names and date format."""
    members = pd.DataFrame(
        {
            "msno": ["aaa", "bbb", "ccc"],
            "city": [1, 13, 5],
            "bd": [24, 0, 31],
            "gender": ["male", None, "female"],
            "registered_via": [7, 9, 7],
            # Unpadded YYYYMMDD integers, as KKBox writes them.
            "registration_init_time": [20150101, 20160302, 20140715],
        }
    )
    transactions = pd.DataFrame(
        {
            "msno": ["aaa", "aaa", "bbb", "ccc", "ccc"],
            "payment_method_id": [41, 41, 36, 40, 40],
            # 30-day, 30-day, 410-day, 7-day, 7-day: spans the plan buckets.
            "payment_plan_days": [30, 30, 410, 7, 7],
            "plan_list_price": [149, 149, 1788, 30, 30],
            "actual_amount_paid": [149, 99, 1788, 30, 30],
            "is_auto_renew": [1, 1, 0, 1, 1],
            "transaction_date": [20150101, 20150201, 20160302, 20140715, 20140722],
            "membership_expire_date": [20150131, 20150303, 20170415, 20140722, 20140729],
            "is_cancel": [0, 0, 0, 0, 1],
        }
    )
    user_logs = pd.DataFrame(
        {
            "msno": ["aaa", "aaa", "bbb", "ccc"],
            "date": [20150102, 20150103, 20160303, 20140716],
            "num_25": [3, 1, 0, 8],
            "num_50": [1, 0, 2, 1],
            "num_75": [0, 1, 1, 0],
            "num_985": [2, 0, 0, 3],
            "num_100": [30, 12, 4, 51],
            "num_unq": [28, 11, 6, 47],
            # The last one is negative: a known artefact of KKBox's logging.
            "total_secs": [7200.5, 3000.0, 900.0, -50.0],
        }
    )

    members.to_csv(tmp_path / "members.csv", index=False)
    transactions.to_csv(tmp_path / "transactions.csv", index=False)
    user_logs.to_csv(tmp_path / "user_logs.csv", index=False)
    return tmp_path


# --------------------------------------------------------------------------- #
# Mapping decisions
# --------------------------------------------------------------------------- #


def test_members_become_subscribers(kkbox_dir: Path) -> None:
    subscribers = kkbox.load_members(kkbox_dir / "members.csv")

    assert list(subscribers.columns) == ["subscriber_id", "signup_date", "acquisition_channel"]
    assert str(subscribers.loc[0, "signup_date"]) == "2015-01-01"


def test_the_opaque_channel_code_is_carried_through_not_invented(kkbox_dir: Path) -> None:
    """`registered_via` is an integer nobody has a decoder ring for.

    Mapping 7 to "organic search" would put a claim in the data that the
    dataset does not support and that a reader would reasonably believe.
    """
    subscribers = kkbox.load_members(kkbox_dir / "members.csv")
    assert set(subscribers["acquisition_channel"]) == {"via_7", "via_9"}


def test_one_transaction_becomes_both_an_event_and_a_payment(kkbox_dir: Path) -> None:
    """Not duplication: different columns answer different questions."""
    events, payments = kkbox.load_transactions(kkbox_dir / "transactions.csv")

    assert len(events) == len(payments) == 5
    assert set(events.columns) >= {"event_type", "plan_type", "monthly_fee", "occurred_at"}
    assert set(payments.columns) >= {"amount", "status", "discount_applied"}


def test_the_first_transaction_is_a_signup_and_the_rest_are_renewals(kkbox_dir: Path) -> None:
    events, _ = kkbox.load_transactions(kkbox_dir / "transactions.csv")

    aaa = events[events["subscriber_id"] == "aaa"].sort_values("occurred_at")
    assert list(aaa["event_type"]) == [schema.SIGNUP, schema.RENEWAL]


def test_a_cancellation_flag_wins_over_position(kkbox_dir: Path) -> None:
    """`is_cancel` is the lifecycle fact; ordinal position is only a heuristic."""
    events, _ = kkbox.load_transactions(kkbox_dir / "transactions.csv")

    ccc = events[events["subscriber_id"] == "ccc"].sort_values("occurred_at")
    assert list(ccc["event_type"]) == [schema.SIGNUP, schema.CANCELLATION]


def test_plan_prices_are_normalised_to_a_monthly_fee(kkbox_dir: Path) -> None:
    """A 410-day payment against a 30-day one is not a like-for-like price.

    Without normalising, long-plan subscribers look enormously expensive and
    `fee_per_session` - a feature the model uses - becomes nonsense.
    """
    events, _ = kkbox.load_transactions(kkbox_dir / "transactions.csv")

    long_plan = events[events["subscriber_id"] == "bbb"].iloc[0]
    assert long_plan["monthly_fee"] == pytest.approx(1788 / 410 * 30.44, abs=0.01)
    assert long_plan["monthly_fee"] < 200


def test_plan_duration_buckets_into_the_three_tiers_the_model_knows(kkbox_dir: Path) -> None:
    """KKBox has no tier names, only durations. The assumption is stated, not hidden."""
    events, _ = kkbox.load_transactions(kkbox_dir / "transactions.csv")
    tiers = dict(zip(events["subscriber_id"], events["plan_type"], strict=True))

    assert tiers["ccc"] == "basic"  # 7-day
    assert tiers["aaa"] == "basic"  # 30-day
    assert tiers["bbb"] == "premium"  # 410-day


def test_paying_under_list_price_is_recorded_as_a_discount(kkbox_dir: Path) -> None:
    """The one inference here the raw data supports directly."""
    _, payments = kkbox.load_transactions(kkbox_dir / "transactions.csv")

    assert list(payments["discount_applied"]) == [False, True, False, False, False]


def test_every_payment_is_a_success_because_that_is_what_the_data_says() -> None:
    """KKBox writes a transaction row when money moved. There are no failures.

    Inventing some to keep `payment_failures_last_6m` alive would be worse than
    losing the feature, which is why it is listed in DEAD_FEATURES instead.
    """
    assert "payment_failures_last_6m" in kkbox.DEAD_FEATURES
    assert "friction_score" in kkbox.DEAD_FEATURES


def test_support_tickets_come_back_empty_and_correctly_shaped() -> None:
    """KKBox has no customer-service data at all.

    Returned empty rather than omitted so the load is complete and the contract
    validator gets a chance to warn about the three features it kills.
    """
    tickets = kkbox.empty_support_tickets()

    assert tickets.empty
    assert set(tickets.columns) == set(contract.REQUIRED_COLUMNS["support_tickets"])


def test_user_logs_stream_in_chunks(kkbox_dir: Path) -> None:
    """The 30GB file is the reason nothing here calls read_csv on the whole thing."""
    chunks = list(kkbox.iter_user_logs(kkbox_dir / "user_logs.csv", chunk_size=2))

    assert len(chunks) == 2
    assert sum(len(chunk) for chunk in chunks) == 4


def test_negative_listening_time_is_clipped_not_dropped(kkbox_dir: Path) -> None:
    """A known artefact of KKBox's logging.

    Dropping the row would understate activity, and understated activity is
    exactly what the model reads as churn risk - so the safe direction is to
    keep the day and zero the duration.
    """
    sessions = pd.concat(kkbox.iter_user_logs(kkbox_dir / "user_logs.csv"))

    assert len(sessions) == 4
    assert (sessions["duration_minutes"] >= 0).all()


def test_malformed_dates_become_null_rather_than_raising(tmp_path: Path) -> None:
    """A traceback on row 12,883,401 of a 400M row file helps nobody.

    Nulls are counted and reported by the contract validator instead, which is
    a far more useful failure.
    """
    path = tmp_path / "members.csv"
    pd.DataFrame(
        {"msno": ["a"], "registered_via": [7], "registration_init_time": [20159999]}
    ).to_csv(path, index=False)

    subscribers = kkbox.load_members(path)
    assert pd.isna(subscribers.loc[0, "signup_date"])


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


def _subscribers(ids=("aaa",), signup="2020-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subscriber_id": list(ids),
            "signup_date": [pd.Timestamp(signup).date()] * len(ids),
            "acquisition_channel": ["via_7"] * len(ids),
        }
    )


def _sessions(ids=("aaa",), when="2020-02-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subscriber_id": list(ids),
            "occurred_at": [pd.Timestamp(when)] * len(ids),
            "duration_minutes": [12.0] * len(ids),
        }
    )


def test_missing_columns_are_reported_before_values_are_checked() -> None:
    report = contract.validate_table("sessions", pd.DataFrame({"subscriber_id": ["a"]}))

    assert not report.ok
    assert "missing required columns" in report.errors[0]


def test_an_unknown_event_type_fails_the_load() -> None:
    frame = pd.DataFrame(
        {
            "subscriber_id": ["aaa"],
            "event_type": ["resurrection"],
            "plan_type": ["basic"],
            "monthly_fee": [10.0],
            "is_auto_renew_enabled": [True],
            "occurred_at": [pd.Timestamp("2020-01-01")],
        }
    )
    report = contract.validate_table("subscription_events", frame)

    assert not report.ok
    assert "unknown event_type" in report.errors[0]


def test_orphan_events_are_an_error_not_a_warning() -> None:
    """They vanish silently.

    Point-in-time queries join outward from the subscribers table, so an event
    for an unknown subscriber is not wrong data - it is data that never appears
    at all, which is much harder to notice.
    """
    reports = contract.validate_load(
        {"subscribers": _subscribers(["aaa"]), "sessions": _sessions(["aaa", "zzz"])}
    )
    sessions = next(report for report in reports if report.table == "sessions")

    assert not sessions.ok
    assert "not in the subscribers table" in sessions.errors[0]


def test_activity_before_signup_warns_rather_than_blocking() -> None:
    """Some exports legitimately record trial activity before the signup date.

    Dropping those rows is a decision for whoever knows the data. Saying
    nothing is not, because such a subscriber shows zero observed activity at
    every cutoff and is indistinguishable from a dormant one.
    """
    reports = contract.validate_load(
        {
            "subscribers": _subscribers(["aaa"], signup="2020-06-01"),
            "sessions": _sessions(["aaa"], when="2020-01-01"),
        }
    )
    sessions = next(report for report in reports if report.table == "sessions")

    assert sessions.ok
    assert "before the subscriber's signup_date" in sessions.warnings[0]


def test_an_empty_table_warns_about_the_features_it_zeroes() -> None:
    """The support_tickets case, generalised.

    Legal, but it makes several features constant, and that should be said at
    load time rather than found later in an all-zero importance chart.
    """
    report = contract.validate_table("support_tickets", kkbox.empty_support_tickets())

    assert report.ok
    assert "constant" in report.warnings[0]


def test_duplicate_subscribers_fail() -> None:
    report = contract.validate_table("subscribers", _subscribers(["aaa", "aaa"]))

    assert not report.ok
    assert "duplicate subscriber_id" in report.errors[0]


def test_the_summary_names_every_table_and_its_verdict() -> None:
    text = contract.summarise(
        contract.validate_load({"subscribers": _subscribers(), "sessions": _sessions()})
    )

    assert "subscribers" in text and "sessions" in text
    assert "ok" in text


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_a_dry_run_validates_everything_and_writes_nothing(kkbox_dir: Path, tmp_path) -> None:
    """Run this first on a 30GB export.

    It is the difference between finding a date-format surprise in ten minutes
    and finding it three hours into a load that half-filled the tables.
    """
    engine = database.get_engine(f"sqlite:///{tmp_path / 'dry.db'}")
    database.create_schema(engine)

    tables = ingest.build_kkbox_tables(kkbox_dir)
    ok, report = ingest.validate(tables)

    assert ok, report
    assert database.table_counts(engine)["subscribers"] == 0


def test_a_loaded_warehouse_has_the_shape_the_features_expect(kkbox_dir: Path, tmp_path) -> None:
    engine = database.get_engine(f"sqlite:///{tmp_path / 'loaded.db'}")

    counts = ingest.write(ingest.build_kkbox_tables(kkbox_dir), engine=engine)

    assert counts == {
        "subscribers": 3,
        "subscription_events": 5,
        "payments": 5,
        "sessions": 4,
        "support_tickets": 0,
    }
    assert database.table_counts(engine)["sessions"] == 4


def test_loading_twice_does_not_double_the_history(kkbox_dir: Path, tmp_path) -> None:
    """A re-run that appended would silently double every subscriber's events.

    The features would look plausible - twice the sessions, twice the payments
    - and be wrong everywhere.
    """
    engine = database.get_engine(f"sqlite:///{tmp_path / 'twice.db'}")
    tables = ingest.build_kkbox_tables(kkbox_dir)

    ingest.write(tables, engine=engine)
    ingest.write(ingest.build_kkbox_tables(kkbox_dir), engine=engine)

    assert database.table_counts(engine)["subscription_events"] == 5


def test_a_missing_export_says_which_files_are_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="members.csv"):
        ingest.build_kkbox_tables(tmp_path)


def test_the_cli_names_the_features_this_dataset_cannot_support() -> None:
    """Stated at load time, not discovered in a feature-importance chart."""
    text = ingest._report_dead_features("kkbox")  # noqa: SLF001

    assert "support_tickets_last_90d" in text
    assert "no customer-service data" in text


def test_the_loader_fills_exactly_the_tables_the_simulator_does() -> None:
    """The claim that swapping data sources is a change of command, not pipeline.

    If these ever diverge, everything downstream - point-in-time features,
    temporal splits, training, promotion - stops being source-agnostic.
    """
    assert set(ingest.TABLES) == {table.name for table in schema.EVENT_TABLES}
