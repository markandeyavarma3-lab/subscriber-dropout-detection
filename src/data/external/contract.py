"""What a dataset must satisfy to become a warehouse.

The warehouse schema says what columns exist. It does not say that a session
cannot precede its subscriber's signup, that a payment status has to be one of
two strings, or that a subscriber cannot appear in the event tables without
appearing in ``subscribers``. Those are the invariants every downstream query
assumes and none of them declares.

Against the simulator that never mattered - it produced correct data by
construction. Against a real export it matters enormously: bad rows in a
warehouse do not fail loudly, they produce a model that trains fine and is
quietly wrong. A subscriber whose sessions all predate their signup date has
zero observed activity at every cutoff, and looks exactly like a dormant one.

So validation runs before the load, not after, and returns every problem it
found rather than raising on the first. Someone cleaning a 30GB export needs
the whole list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.warehouse.schema import CANCELLATION, PLAN_CHANGE, RENEWAL, SIGNUP

# Columns each warehouse table needs from a loader. Autoincrement primary keys
# are deliberately absent: the database assigns them, and a loader that
# supplies its own would collide the second time it ran.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "subscribers": ("subscriber_id", "signup_date", "acquisition_channel"),
    "subscription_events": (
        "subscriber_id",
        "event_type",
        "plan_type",
        "monthly_fee",
        "is_auto_renew_enabled",
        "occurred_at",
    ),
    "sessions": ("subscriber_id", "occurred_at", "duration_minutes"),
    "payments": ("subscriber_id", "occurred_at", "amount", "status", "discount_applied"),
    "support_tickets": ("subscriber_id", "occurred_at", "category"),
}

VALID_EVENT_TYPES = frozenset({SIGNUP, RENEWAL, PLAN_CHANGE, CANCELLATION})
VALID_PAYMENT_STATUSES = frozenset({"succeeded", "failed"})


@dataclass
class ValidationReport:
    """Everything wrong with a candidate load, gathered in one pass."""

    table: str
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the table can be loaded. Warnings do not block a load."""
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows": self.rows,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_table(name: str, frame: pd.DataFrame) -> ValidationReport:
    """Check one table against the contract.

    Args:
        name: Warehouse table name.
        frame: Rows a loader proposes to insert.

    Returns:
        A report listing every problem found. Nothing raises: the caller
        decides whether to abort, and needs the whole list either way.
    """
    report = ValidationReport(table=name, rows=len(frame))

    required = REQUIRED_COLUMNS.get(name)
    if required is None:
        report.errors.append(f"{name!r} is not a warehouse table")
        return report

    missing = [column for column in required if column not in frame.columns]
    if missing:
        # No point checking values in columns that are not there.
        report.errors.append(f"missing required columns: {missing}")
        return report

    if frame.empty:
        # Empty is legal - not every dataset carries every event type - but it
        # silently zeroes whichever features read that table, so it is said out
        # loud rather than passed over.
        report.warnings.append("no rows; every feature derived from this table will be constant")
        return report

    for column in required:
        nulls = int(frame[column].isna().sum())
        if nulls:
            report.errors.append(f"{column} has {nulls} null values")

    if name == "subscribers":
        duplicates = int(frame["subscriber_id"].duplicated().sum())
        if duplicates:
            report.errors.append(f"{duplicates} duplicate subscriber_id values")

    if name == "subscription_events":
        unknown = sorted(set(frame["event_type"].unique()) - VALID_EVENT_TYPES)
        if unknown:
            report.errors.append(f"unknown event_type values: {unknown}")
        if (frame["monthly_fee"] < 0).any():
            report.errors.append("negative monthly_fee")

    if name == "payments":
        unknown = sorted(set(frame["status"].unique()) - VALID_PAYMENT_STATUSES)
        if unknown:
            report.errors.append(f"unknown payment status values: {unknown}")

    if name == "sessions" and (frame["duration_minutes"] < 0).any():
        report.errors.append("negative duration_minutes")

    return report


def validate_load(tables: dict[str, pd.DataFrame]) -> list[ValidationReport]:
    """Check a whole candidate warehouse, including the cross-table invariants.

    The per-table checks above cannot see the two failures that actually cause
    silently wrong models:

    **Orphan events.** Events for a subscriber who is not in ``subscribers``
    are invisible to every point-in-time query, which joins from the subscriber
    table outward. They do not error; they simply never appear.

    **Events before signup.** A subscriber whose activity all predates their
    recorded signup date has zero observed activity at every cutoff and is
    indistinguishable from a dormant one - which is the single most predictive
    feature in the model, so this quietly poisons the label relationship.
    """
    reports = [validate_table(name, frame) for name, frame in tables.items()]

    subscribers = tables.get("subscribers")
    if subscribers is None or subscribers.empty:
        return reports

    known = set(subscribers["subscriber_id"])
    signups = dict(
        zip(
            subscribers["subscriber_id"],
            pd.to_datetime(subscribers["signup_date"]),
            strict=True,
        )
    )

    by_name = {report.table: report for report in reports}
    for name, frame in tables.items():
        if name == "subscribers" or frame.empty or "subscriber_id" not in frame:
            continue
        report = by_name[name]

        orphans = int((~frame["subscriber_id"].isin(known)).sum())
        if orphans:
            report.errors.append(
                f"{orphans} rows reference subscribers that are not in the subscribers table; "
                "point-in-time queries join outward from there, so these rows would vanish"
            )

        if "occurred_at" not in frame.columns:
            continue
        expected = frame["subscriber_id"].map(signups)
        early = int((pd.to_datetime(frame["occurred_at"]) < expected).sum())
        if early:
            # A warning, not an error: some exports legitimately record
            # pre-signup trial activity, and dropping those rows is a decision
            # for whoever knows the data, not for the loader.
            report.warnings.append(
                f"{early} rows occur before the subscriber's signup_date; "
                "these contribute no observed activity at any cutoff"
            )

    return reports


def summarise(reports: list[ValidationReport]) -> str:
    """A human-readable digest of a validation run."""
    lines = []
    for report in reports:
        status = "ok" if report.ok else "FAILED"
        lines.append(f"{report.table:<20} {report.rows:>10,} rows  {status}")
        for error in report.errors:
            lines.append(f"    error:   {error}")
        for warning in report.warnings:
            lines.append(f"    warning: {warning}")
    return "\n".join(lines)
