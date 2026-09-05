"""Map the KKBox WSDM Cup churn dataset into the warehouse.

KKBox is a music streaming service; the dataset was released for the WSDM Cup
2018 churn prediction challenge and is the closest public analogue to what this
project simulates - a subscription business with transactions, usage logs, and
a churn label defined by a subscription failing to renew.

Status of this loader
---------------------

**Run against the real dataset.** It was written against the published schema
first and tested against fixtures matching it; first contact with the actual
8.3GB download then found five things the documentation did not mention, all of
which are fixed here:

1. The archive ships ``.7z`` files, not plain CSV.
2. Members arrive as ``members_v3.csv``; transactions and user logs each have a
   ``_v2`` second-stage variant alongside the original.
3. ``msno`` is a 44-character base64 SHA256. The warehouse declared
   ``String(32)``, which SQLite silently ignores and Postgres rejects on every
   row - see ``SUBSCRIBER_ID_LENGTH`` in :mod:`src.warehouse.schema`.
4. 2,656,043 transaction rows reference 432,623 subscribers that
   ``members_v3.csv`` never describes. The contract caught them before the
   write; ``--drop-orphans`` handles them.
5. The full user log is 392 million rows - impractical to write to SQLite and
   far more history than a 30-day observation window uses. ``--sessions-since``
   bounds it.

The fixtures could not have found any of them: they were referentially perfect,
correctly named, plain CSV, four rows long, with three-character ids. That gap
is exactly what `--dry-run` and the contract validation exist to close, and
they did.

What KKBox does not have
------------------------

``support_tickets`` has no counterpart in this dataset. KKBox published
members, transactions and user logs; there is no customer-service data at all.
The loader therefore writes an empty support_tickets table, and the contract
validator warns about it, because the consequence is easy to miss and
important: ``support_tickets_last_90d`` and ``support_tickets_per_month``
become constant zero, and ``friction_score`` collapses to payment failures
alone. Three of the model's twenty-one features stop carrying information.

That is not a defect in the loader. It is what using real data looks like -
the schema you designed and the data you can get do not line up, and the
honest response is to say which features went dark rather than to synthesise
plausible-looking tickets to fill the gap.

The mapping
-----------

===================  ==========================================================
members.csv          subscribers. ``registered_via`` becomes
                     ``acquisition_channel`` - it is the only provenance field
                     in the dataset, and it is an opaque integer code, so it is
                     carried through as ``via_<n>`` rather than invented names.
transactions.csv     subscription_events *and* payments. One transaction is
                     both: a lifecycle event (the subscription continued, or
                     was cancelled) and a payment (money changed hands). Fees
                     are normalised to a monthly rate from
                     ``payment_plan_days``, since the warehouse stores a
                     monthly fee and KKBox sells plans of 7, 30, 90, 180, 410
                     days and more.
user_logs.csv        sessions. KKBox aggregates to one row per user per day,
                     not per session, so one row becomes one session with
                     ``total_secs`` as its duration. Session *counts* are
                     therefore active-day counts. That changes what
                     ``avg_session_count_last_30d`` means - it is bounded above
                     by 30 - without changing whether it is predictive.
(none)               support_tickets. See above.
===================  ==========================================================
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.warehouse.schema import CANCELLATION, RENEWAL, SIGNUP

logger = logging.getLogger(__name__)

# The files this loader reads, and the columns it needs from each. Anything
# else in them is ignored rather than rejected: Kaggle competition data grows
# extra columns between releases, and failing on one would be gratuitous.
MEMBER_COLUMNS = ("msno", "city", "bd", "gender", "registered_via", "registration_init_time")
TRANSACTION_COLUMNS = (
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
)
USER_LOG_COLUMNS = ("msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq",
                    "total_secs")

# KKBox writes every date as an unpadded YYYYMMDD integer.
DATE_FORMAT = "%Y%m%d"

# Plan-length buckets mapped onto the three plan names the model knows. The
# boundaries are arbitrary in the sense that KKBox has no notion of "basic" or
# "premium" - what it has is plan duration and price. Longer commitments are
# treated as higher tiers, which is how subscription pricing generally works
# and is at least a stated assumption rather than a hidden one.
PLAN_BUCKETS = ((0, 31, "basic"), (32, 180, "standard"), (181, 10_000, "premium"))

# Days per month used to normalise plan prices to a monthly fee.
DAYS_PER_MONTH = 30.44


@dataclass(frozen=True)
class KKBoxPaths:
    """Where the extracted CSVs live."""

    members: Path
    transactions: Path
    user_logs: Path

    # The real archive does not use the names the competition docs imply.
    # Found on first contact with the actual download: members ships as
    # `members_v3.csv`, and transactions and user_logs each ship twice - the
    # original and a `_v2` covering the later second-stage period. Candidates
    # are tried in order, so the plain name still wins where it exists and the
    # fixtures in the test suite keep working unchanged.
    MEMBER_NAMES = ("members.csv", "members_v3.csv", "members_v2.csv")
    TRANSACTION_NAMES = ("transactions.csv", "transactions_v2.csv")
    USER_LOG_NAMES = ("user_logs.csv", "user_logs_v2.csv")

    @classmethod
    def under(cls, directory: Path) -> KKBoxPaths:
        """Locate the three files this loader needs inside ``directory``.

        Falls back to the first candidate name when none exist, so the error
        message from :meth:`missing` names something a person can act on rather
        than an empty path.
        """

        def pick(names: tuple[str, ...]) -> Path:
            for name in names:
                candidate = directory / name
                if candidate.exists():
                    return candidate
            return directory / names[0]

        return cls(
            members=pick(cls.MEMBER_NAMES),
            transactions=pick(cls.TRANSACTION_NAMES),
            user_logs=pick(cls.USER_LOG_NAMES),
        )

    def missing(self) -> list[Path]:
        return [path for path in (self.members, self.transactions, self.user_logs)
                if not path.exists()]


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse KKBox's YYYYMMDD integers, turning junk into NaT rather than raising.

    ``errors="coerce"`` on purpose. A handful of impossible dates in a 400
    million row export should not stop the load; the contract validator counts
    the resulting nulls and reports them, which is a far more useful failure
    than a traceback on row 12,883,401.
    """
    return pd.to_datetime(series.astype("string"), format=DATE_FORMAT, errors="coerce")


def _plan_type(plan_days: pd.Series) -> pd.Series:
    """Bucket plan durations into the three tiers the model was built around."""
    result = pd.Series("standard", index=plan_days.index, dtype="object")
    for low, high, name in PLAN_BUCKETS:
        result = result.mask(plan_days.between(low, high), name)
    return result


def _monthly_fee(amount_paid: pd.Series, plan_days: pd.Series) -> pd.Series:
    """Normalise a plan price to a monthly rate.

    KKBox sells 7, 30, 90, 180 and 410 day plans; the warehouse stores a
    monthly fee. Comparing a 410-day payment against a 30-day one without
    normalising would make long-plan subscribers look enormously expensive,
    and ``fee_per_session`` - a feature the model uses - would be nonsense.
    """
    days = plan_days.replace(0, np.nan)
    return (amount_paid / days * DAYS_PER_MONTH).fillna(0.0).round(2)


def load_members(path: Path) -> pd.DataFrame:
    """members.csv -> the ``subscribers`` table."""
    frame = pd.read_csv(path, usecols=lambda column: column in MEMBER_COLUMNS)

    return pd.DataFrame(
        {
            "subscriber_id": frame["msno"].astype(str),
            "signup_date": _to_datetime(frame["registration_init_time"]).dt.date,
            # The only provenance field in the dataset, and an opaque integer
            # code. Carried through as-is rather than mapped onto invented
            # channel names that would imply knowledge nobody has.
            "acquisition_channel": "via_" + frame["registered_via"].astype("string"),
        }
    )


def load_transactions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """transactions.csv -> ``subscription_events`` and ``payments``.

    One transaction is genuinely both. Splitting it is not duplication: the
    lifecycle question ("did this subscription continue?") and the money
    question ("did this payment go through?") are answered by different
    columns and feed different features.

    Returns:
        ``(subscription_events, payments)``.
    """
    frame = pd.read_csv(path, usecols=lambda column: column in TRANSACTION_COLUMNS)

    occurred_at = _to_datetime(frame["transaction_date"])
    subscriber_id = frame["msno"].astype(str)
    is_cancel = frame["is_cancel"].astype(bool)

    # First transaction per subscriber is the signup; the rest are renewals,
    # except the ones flagged as cancellations. KKBox has no plan-change flag,
    # so PLAN_CHANGE is never emitted - a plan switch appears as a renewal at a
    # different price, which is what the raw data actually says.
    order = occurred_at.groupby(subscriber_id).rank(method="first")
    event_type = np.where(is_cancel, CANCELLATION, np.where(order == 1, SIGNUP, RENEWAL))

    events = pd.DataFrame(
        {
            "subscriber_id": subscriber_id,
            "event_type": event_type,
            "plan_type": _plan_type(frame["payment_plan_days"]),
            "monthly_fee": _monthly_fee(frame["actual_amount_paid"], frame["payment_plan_days"]),
            "is_auto_renew_enabled": frame["is_auto_renew"].astype(bool),
            "occurred_at": occurred_at,
        }
    )

    payments = pd.DataFrame(
        {
            "subscriber_id": subscriber_id,
            "occurred_at": occurred_at,
            "amount": frame["actual_amount_paid"].astype(float),
            # KKBox records no failed payments - a transaction row exists
            # because money moved. So payment_failures_last_6m is another
            # feature that goes constant on this dataset, and pretending
            # otherwise by inventing failures would be worse than losing it.
            "status": "succeeded",
            # Paying less than list price is a discount. This is the one
            # inference here that the raw data supports directly.
            "discount_applied": frame["actual_amount_paid"] < frame["plan_list_price"],
        }
    )

    return events, payments


def iter_user_logs(
    path: Path, chunk_size: int = 2_000_000, since: str | None = None
) -> Iterator[pd.DataFrame]:
    """user_logs.csv -> ``sessions``, in chunks.

    This file is the reason the loader streams rather than reading frames. It
    is roughly 30GB and around 400 million rows; ``pd.read_csv`` on the whole
    thing needs more memory than most machines have, and the load would die
    two hours in having written nothing.

    KKBox aggregates to one row per user per day, so one row becomes one
    session. ``avg_session_count_last_30d`` therefore counts *active days* and
    is bounded above by 30 - a different quantity from the simulator's session
    count, and one worth remembering before comparing the two.

    ``since`` bounds the window. The full log is 392 million rows, which is
    hours of index maintenance to write and far more history than a model with
    a 30-day observation window and a handful of cutoffs can use. Filtering
    happens per chunk, before anything reaches the database.

    One consequence to be aware of: session *recency* scans all loaded history,
    so a subscriber whose last activity predates ``since`` looks like they have
    never had a session. That reads as maximally dormant, which is directionally
    right but not identical to the truth.
    """
    cutoff = pd.Timestamp(since) if since else None

    for chunk in pd.read_csv(
        path, usecols=lambda column: column in USER_LOG_COLUMNS, chunksize=chunk_size
    ):
        occurred_at = _to_datetime(chunk["date"])
        if cutoff is not None:
            # Filtered here rather than after loading, so the rows never reach
            # the database at all. Reading past 392 million rows is I/O; writing
            # them is hours of index maintenance.
            keep = occurred_at >= cutoff
            chunk = chunk[keep]
            occurred_at = occurred_at[keep]
            if chunk.empty:
                continue

        yield pd.DataFrame(
            {
                "subscriber_id": chunk["msno"].astype(str),
                "occurred_at": occurred_at,
                # total_secs occasionally goes negative in this dataset - a
                # known artefact of the logging. Clipped rather than dropped:
                # the day still happened, and dropping it would understate
                # activity, which is the direction that fakes churn risk.
                "duration_minutes": (chunk["total_secs"].clip(lower=0) / 60.0).round(2),
            }
        )


def empty_support_tickets() -> pd.DataFrame:
    """The table KKBox cannot fill.

    Returned empty and correctly shaped rather than omitted, so the load is
    complete and the contract validator gets a chance to warn about it. See
    the module docstring: three model features go constant because of this.
    """
    return pd.DataFrame(
        {
            "subscriber_id": pd.Series(dtype="object"),
            "occurred_at": pd.Series(dtype="datetime64[ns]"),
            "category": pd.Series(dtype="object"),
        }
    )


# Features that lose all variance on this dataset, and why. Surfaced by the
# ingest CLI so the loss is stated at load time rather than discovered later in
# a feature-importance chart full of zeros.
DEAD_FEATURES: dict[str, str] = {
    "support_tickets_last_90d": "KKBox publishes no customer-service data",
    "support_tickets_per_month": "KKBox publishes no customer-service data",
    "payment_failures_last_6m": "a KKBox transaction row exists only when money moved",
    "payment_failures_per_month": "a KKBox transaction row exists only when money moved",
    "friction_score": "sums the two above, so it collapses to zero",
}
