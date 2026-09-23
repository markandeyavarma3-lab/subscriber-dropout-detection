"""Tests for the SQLite-to-Postgres copy.

The copy itself needs Docker and both databases, so it is verified by running
it (``make warehouse-postgres`` ends by comparing every table's row count).
What is tested here is the part that decides *which* rows move and where: the
slicing, the column order, and the sequence reset. Those are the places a
mistake produces a load that finishes cleanly and is still wrong.
"""

from __future__ import annotations

import pytest

from src.warehouse import schema, to_postgres


@pytest.mark.parametrize(
    ("first", "last", "size"),
    [(1, 10, 3), (1, 38_216_556, 5_000_000), (1, 1, 5), (7, 20, 20), (1, 12, 4)],
)
def test_slices_cover_every_rowid_exactly_once(first, last, size):
    """A gap loses rows silently; an overlap duplicates them silently."""
    covered: list[int] = []
    for low, high in to_postgres.slices(first, last, size):
        assert high - low <= size
        covered.extend(range(low + 1, high + 1))
    assert covered == list(range(first, last + 1))


def test_an_empty_table_produces_no_slices():
    assert to_postgres.slices(1, 0, 10) == []


def test_both_sides_use_the_schema_column_order():
    """The SELECT and the COPY must name the same columns in the same order.

    CSV carries no headers, so a mismatch loads values into the wrong columns -
    and a float landing in another float column raises no error at all.
    """
    for table in schema.EVENT_TABLES:
        names = to_postgres.column_names(table)
        select = to_postgres.select_sql(table, 0, 10)
        copy = to_postgres.copy_sql(table)
        assert select.startswith(f"SELECT {', '.join(names)} FROM {table.name} ")
        assert f"({', '.join(names)})" in copy


def test_slices_are_selected_by_rowid_range_not_offset():
    """OFFSET would re-scan every earlier row, making the load quadratic."""
    sql = to_postgres.select_sql(schema.sessions, 5_000_000, 10_000_000)
    assert "rowid > 5000000 AND rowid <= 10000000" in sql
    assert "OFFSET" not in sql.upper()


def test_serial_keys_get_their_sequence_reset():
    """Rows arrive with their ids, so the sequence has to be moved past them."""
    for table in (schema.sessions, schema.payments, schema.subscription_events):
        reset = to_postgres.sequence_reset_sql(table)
        assert reset is not None
        assert f"pg_get_serial_sequence('{table.name}'" in reset


def test_a_natural_key_has_no_sequence_to_reset():
    """subscribers is keyed by the 44-character msno, not an integer."""
    assert to_postgres.serial_column(schema.subscribers) is None
    assert to_postgres.sequence_reset_sql(schema.subscribers) is None
