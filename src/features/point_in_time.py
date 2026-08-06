"""Point-in-time-correct feature and label construction.

This module is the reason the warehouse exists.

Churn is a **time-series** problem.  The previous approach - one row per
subscriber, a random train/test split - quietly lets the model learn from
behaviour that happened *after* the outcome it is predicting.  A model
validated that way scores beautifully offline and fails in production, because
at serving time the future is not available.

The fix is a strict separation around a cutoff ``T``:

.. code-block:: text

    [ T - observation_window , T )      features   - only ever looks backwards
    [ T , T + prediction_horizon ]      label      - never seen by features

Nothing computed for a row may touch data at or after ``T``, and the label is
drawn strictly after it.  The two windows are disjoint by construction, and
:mod:`tests.test_point_in_time` asserts it rather than trusting the comment.

Producing one training set per cutoff also gives backfills for free: the same
function over a series of cutoffs yields a panel that grows as the calendar
does, which is what makes scheduled retraining meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import Engine

from src.config import settings
from src.warehouse.database import read_sql

logger = logging.getLogger(__name__)

# Behavioural aggregates over the observation window.  Written as one query so
# the database does the work: pulling every session into pandas would not
# survive the volumes this schema is built for.
_FEATURE_SQL = """
WITH base AS (
    SELECT
        s.subscriber_id,
        s.signup_date,
        s.acquisition_channel
    FROM subscribers s
    WHERE s.signup_date < :cutoff
),
-- The subscription state as of the cutoff: the most recent lifecycle event
-- strictly before T. Ordering by occurred_at then event_id keeps the pick
-- deterministic when two events share a timestamp.
latest_state AS (
    SELECT subscriber_id, plan_type, monthly_fee, is_auto_renew_enabled, event_type
    FROM (
        SELECT
            e.subscriber_id,
            e.plan_type,
            e.monthly_fee,
            e.is_auto_renew_enabled,
            e.event_type,
            ROW_NUMBER() OVER (
                PARTITION BY e.subscriber_id
                ORDER BY e.occurred_at DESC, e.event_id DESC
            ) AS rn
        FROM subscription_events e
        WHERE e.occurred_at < :cutoff
    ) ranked
    WHERE rn = 1
),
session_window AS (
    SELECT
        subscriber_id,
        COUNT(*)              AS session_count,
        AVG(duration_minutes) AS avg_duration,
        MAX(occurred_at)      AS last_session_at
    FROM sessions
    WHERE occurred_at >= :window_start AND occurred_at < :cutoff
    GROUP BY subscriber_id
),
-- Recency must look across all history, not just the window: a subscriber
-- dormant for 90 days has no rows in a 30-day window at all, and treating that
-- as "no data" instead of "long absent" would erase the strongest signal here.
session_recency AS (
    SELECT subscriber_id, MAX(occurred_at) AS last_session_ever
    FROM sessions
    WHERE occurred_at < :cutoff
    GROUP BY subscriber_id
),
payment_window AS (
    SELECT
        subscriber_id,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)      AS payment_failures,
        SUM(CASE WHEN discount_applied THEN 1 ELSE 0 END)       AS discounts_used
    FROM payments
    WHERE occurred_at >= :billing_start AND occurred_at < :cutoff
    GROUP BY subscriber_id
),
ticket_window AS (
    SELECT subscriber_id, COUNT(*) AS support_tickets
    FROM support_tickets
    WHERE occurred_at >= :ticket_start AND occurred_at < :cutoff
    GROUP BY subscriber_id
)
SELECT
    b.subscriber_id,
    b.signup_date,
    b.acquisition_channel,
    ls.plan_type,
    ls.monthly_fee,
    ls.is_auto_renew_enabled,
    ls.event_type                                   AS last_event_type,
    COALESCE(sw.session_count, 0)                   AS avg_session_count_last_30d,
    COALESCE(sw.avg_duration, 0.0)                  AS avg_session_duration,
    sr.last_session_ever,
    COALESCE(pw.payment_failures, 0)                AS payment_failures_last_6m,
    COALESCE(pw.discounts_used, 0)                  AS discounts_used_last_6m,
    COALESCE(tw.support_tickets, 0)                 AS support_tickets_last_90d
FROM base b
JOIN latest_state ls   ON ls.subscriber_id = b.subscriber_id
LEFT JOIN session_window sw  ON sw.subscriber_id = b.subscriber_id
LEFT JOIN session_recency sr ON sr.subscriber_id = b.subscriber_id
LEFT JOIN payment_window pw  ON pw.subscriber_id = b.subscriber_id
LEFT JOIN ticket_window tw   ON tw.subscriber_id = b.subscriber_id
"""

# The label: did this subscriber cancel inside the prediction horizon?
_LABEL_SQL = """
SELECT DISTINCT subscriber_id
FROM subscription_events
WHERE event_type = 'cancellation'
  AND occurred_at >= :cutoff
  AND occurred_at < :horizon_end
"""

# Subscribers who had already cancelled before the cutoff.  They are not at
# risk of churning again and must be excluded, or the training set fills with
# rows whose label is trivially zero.
_ALREADY_GONE_SQL = """
SELECT DISTINCT subscriber_id
FROM subscription_events
WHERE event_type = 'cancellation'
  AND occurred_at < :cutoff
"""


@dataclass(frozen=True)
class TrainingWindow:
    """The time boundaries one training snapshot was built from."""

    cutoff: date
    observation_days: int
    horizon_days: int

    @property
    def window_start(self) -> date:
        """First day whose behaviour is visible to the features."""
        return self.cutoff - timedelta(days=self.observation_days)

    @property
    def horizon_end(self) -> date:
        """Day the label window closes."""
        return self.cutoff + timedelta(days=self.horizon_days)

    def summary(self) -> dict[str, str | int]:
        """JSON-friendly description, recorded alongside trained models."""
        return {
            "cutoff": self.cutoff.isoformat(),
            "observation_start": self.window_start.isoformat(),
            "horizon_end": self.horizon_end.isoformat(),
            "observation_days": self.observation_days,
            "horizon_days": self.horizon_days,
        }


def _as_datetime(value: date) -> datetime:
    """Midnight on a given day, as the databases expect."""
    return datetime.combine(value, datetime.min.time())


def build_training_snapshot(
    cutoff: str | date,
    observation_days: int | None = None,
    horizon_days: int | None = None,
    engine: Engine | None = None,
) -> tuple[pd.DataFrame, TrainingWindow]:
    """Build one labelled training set as of ``cutoff``.

    Args:
        cutoff: The "as of" date ``T``. Features use only data before it.
        observation_days: Behavioural window length. Defaults to settings.
        horizon_days: Label window length. Defaults to settings.
        engine: Optional SQLAlchemy engine.

    Returns:
        ``(frame, window)`` where ``frame`` carries the raw serving columns plus
        ``dropout``, and ``window`` records the boundaries used.
    """
    as_of = cutoff if isinstance(cutoff, date) else date.fromisoformat(cutoff)
    window = TrainingWindow(
        cutoff=as_of,
        observation_days=observation_days or settings.OBSERVATION_WINDOW_DAYS,
        horizon_days=horizon_days or settings.PREDICTION_HORIZON_DAYS,
    )

    cutoff_dt = _as_datetime(window.cutoff)
    params = {
        "cutoff": cutoff_dt,
        "window_start": _as_datetime(window.window_start),
        # Billing and support signals are slower-moving than usage, so they get
        # longer lookbacks; the column names carry the period they cover.
        "billing_start": cutoff_dt - timedelta(days=180),
        "ticket_start": cutoff_dt - timedelta(days=90),
    }

    frame = read_sql(_FEATURE_SQL, params, engine=engine)
    if frame.empty:
        return frame.assign(dropout=pd.Series(dtype=int)), window

    gone = read_sql(_ALREADY_GONE_SQL, {"cutoff": cutoff_dt}, engine=engine)
    frame = frame[~frame["subscriber_id"].isin(set(gone["subscriber_id"]))].copy()

    churned = read_sql(
        _LABEL_SQL,
        {"cutoff": cutoff_dt, "horizon_end": _as_datetime(window.horizon_end)},
        engine=engine,
    )
    frame["dropout"] = frame["subscriber_id"].isin(set(churned["subscriber_id"])).astype(int)

    return _finalise_columns(frame, window), window


def _finalise_columns(frame: pd.DataFrame, window: TrainingWindow) -> pd.DataFrame:
    """Derive the remaining serving columns and drop warehouse-only ones.

    Tenure and recency are computed here rather than in SQL because the two
    databases spell date arithmetic differently; doing it in pandas keeps the
    query portable between SQLite and Postgres.
    """
    cutoff_dt = _as_datetime(window.cutoff)

    signup = pd.to_datetime(frame["signup_date"])
    frame["tenure_days"] = (cutoff_dt - signup).dt.days.clip(lower=0).astype(int)

    last_seen = pd.to_datetime(frame["last_session_ever"])
    # Never-active subscribers are "absent since signup", which is the honest
    # reading - not a missing value to be imputed away.
    days_since = (cutoff_dt - last_seen).dt.days
    frame["last_activity_days_ago"] = (
        days_since.fillna(frame["tenure_days"]).clip(lower=0).astype(int)
    )
    frame["last_activity_days_ago"] = frame[["last_activity_days_ago", "tenure_days"]].min(axis=1)

    frame["is_auto_renew_enabled"] = frame["is_auto_renew_enabled"].astype(bool)
    frame["avg_session_count_last_30d"] = frame["avg_session_count_last_30d"].astype(float)
    for column in (
        "support_tickets_last_90d",
        "payment_failures_last_6m",
        "discounts_used_last_6m",
    ):
        frame[column] = frame[column].astype(int)

    frame["cutoff"] = window.cutoff.isoformat()

    keep = [
        "subscriber_id",
        "cutoff",
        "tenure_days",
        "plan_type",
        "monthly_fee",
        "avg_session_count_last_30d",
        "last_activity_days_ago",
        "support_tickets_last_90d",
        "payment_failures_last_6m",
        "discounts_used_last_6m",
        "is_auto_renew_enabled",
        "dropout",
    ]
    return frame[keep].reset_index(drop=True)


def build_backfill(
    cutoffs: list[str | date],
    observation_days: int | None = None,
    horizon_days: int | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Stack snapshots from several cutoffs into one panel.

    This is what a scheduled pipeline accumulates over time, and what makes a
    retraining job do real work instead of refitting identical data.
    """
    frames = []
    for cutoff in cutoffs:
        frame, window = build_training_snapshot(
            cutoff, observation_days, horizon_days, engine=engine
        )
        logger.info(
            "cutoff %s -> %d rows, dropout rate %.3f",
            window.cutoff,
            len(frame),
            float(frame["dropout"].mean()) if len(frame) else 0.0,
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def monthly_cutoffs(start: str | date, end: str | date, step_days: int = 30) -> list[date]:
    """Evenly spaced cutoffs between two dates, for backfilling."""
    first = start if isinstance(start, date) else date.fromisoformat(start)
    last = end if isinstance(end, date) else date.fromisoformat(end)

    cutoffs: list[date] = []
    current = first
    while current <= last:
        cutoffs.append(current)
        current += timedelta(days=step_days)
    return cutoffs
