"""Event-log schema for the subscriber warehouse.

This replaces the single wide CSV with the shape production data actually
arrives in: **immutable events with timestamps**, one row per thing that
happened, rather than one row per subscriber with everything pre-aggregated.

That change is what makes the rest of the roadmap possible.  A pre-aggregated
table has no "as of" - you cannot ask what a subscriber looked like last March
without having stored a snapshot of last March.  An event log can answer it for
any cutoff, which is what point-in-time-correct training requires.

The same schema runs on SQLite locally and Postgres in the compose stack, so
tests need no database server while the deployed stack gets a real one.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
)

metadata = MetaData()

# Subscriber identifiers are 64 characters, not 32.
#
# The simulator emits short synthetic ids like "SUB-00042", so 32 was ample and
# nothing ever complained. Real exports do not look like that: KKBox identifies
# subscribers by a base64-encoded SHA256 hash, which is exactly 44 characters,
# and other providers hash differently again.
#
# This is the failure mode worth understanding. SQLite ignores VARCHAR lengths
# entirely, so a 44-character id loads locally without a murmur and every test
# passes. Postgres enforces them, so the same load against the compose stack's
# warehouse fails on every row with "value too long for type character
# varying(32)". A constraint that only bites in production is worse than no
# constraint at all, and only real data could have surfaced it.
#
# 64 leaves room for a hex-encoded SHA256 (the other common shape) without
# being unbounded.
SUBSCRIBER_ID_LENGTH = 64

# One row per subscriber: the facts that do not change over time.  Anything
# that *does* change (plan, price, status) lives in the event tables instead.
subscribers = Table(
    "subscribers",
    metadata,
    Column("subscriber_id", String(SUBSCRIBER_ID_LENGTH), primary_key=True),
    Column("signup_date", Date, nullable=False, index=True),
    Column("acquisition_channel", String(32), nullable=False),
)

# The subscription lifecycle: signup, renewal, plan change, cancellation.
subscription_events = Table(
    "subscription_events",
    metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("subscriber_id", String(SUBSCRIBER_ID_LENGTH), nullable=False),
    Column("event_type", String(24), nullable=False),
    Column("plan_type", String(16), nullable=False),
    Column("monthly_fee", Float, nullable=False),
    Column("is_auto_renew_enabled", Boolean, nullable=False),
    Column("occurred_at", DateTime, nullable=False),
    Index("ix_subscription_events_sub_time", "subscriber_id", "occurred_at"),
)

# The highest-volume table: one row per usage session.
sessions = Table(
    "sessions",
    metadata,
    Column("session_id", Integer, primary_key=True, autoincrement=True),
    Column("subscriber_id", String(SUBSCRIBER_ID_LENGTH), nullable=False),
    Column("occurred_at", DateTime, nullable=False),
    Column("duration_minutes", Float, nullable=False),
    Index("ix_sessions_sub_time", "subscriber_id", "occurred_at"),
)

payments = Table(
    "payments",
    metadata,
    Column("payment_id", Integer, primary_key=True, autoincrement=True),
    Column("subscriber_id", String(SUBSCRIBER_ID_LENGTH), nullable=False),
    Column("occurred_at", DateTime, nullable=False),
    Column("amount", Float, nullable=False),
    # "succeeded" | "failed"
    Column("status", String(16), nullable=False),
    Column("discount_applied", Boolean, nullable=False, default=False),
    Index("ix_payments_sub_time", "subscriber_id", "occurred_at"),
)

support_tickets = Table(
    "support_tickets",
    metadata,
    Column("ticket_id", Integer, primary_key=True, autoincrement=True),
    Column("subscriber_id", String(SUBSCRIBER_ID_LENGTH), nullable=False),
    Column("occurred_at", DateTime, nullable=False),
    Column("category", String(32), nullable=False),
    Index("ix_support_tickets_sub_time", "subscriber_id", "occurred_at"),
)

# Every table that carries a timestamp, in load order (parents first).
EVENT_TABLES = (subscribers, subscription_events, sessions, payments, support_tickets)

# Lifecycle event types.
SIGNUP = "signup"
RENEWAL = "renewal"
PLAN_CHANGE = "plan_change"
CANCELLATION = "cancellation"
