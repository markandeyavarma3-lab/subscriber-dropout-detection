"""Database connections for the event warehouse.

One URL controls everything.  Locally it defaults to a SQLite file so the
project runs with no server; the compose stack sets ``SDD_DATABASE_URL`` to
Postgres.  Keeping both behind SQLAlchemy means the schema and the
point-in-time queries are written once and exercised by the test suite on every
run, rather than only being tried in the deployed environment.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pandas as pd
from sqlalchemy import Engine, Table, create_engine, delete, text
from sqlalchemy.engine import Connection

from src.config import settings
from src.warehouse.schema import EVENT_TABLES, metadata

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Return a cached engine for the configured database URL."""
    global _engine
    if url is not None:
        return create_engine(url, echo=echo, future=True)
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, echo=echo, future=True)
        logger.info("Warehouse engine: %s", _engine.url.render_as_string(hide_password=True))
    return _engine


def reset_engine() -> None:
    """Drop the cached engine so the next call rebuilds it."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def connect(engine: Engine | None = None) -> Iterator[Connection]:
    """Yield a transactional connection, committing on clean exit."""
    target = engine or get_engine()
    with target.begin() as connection:
        yield connection


def create_schema(engine: Engine | None = None) -> None:
    """Create every warehouse table if it does not already exist."""
    metadata.create_all(engine or get_engine())


def drop_schema(engine: Engine | None = None) -> None:
    """Drop every warehouse table.  Used by tests and by full reloads."""
    metadata.drop_all(engine or get_engine())


def truncate_all(engine: Engine | None = None) -> None:
    """Delete every row while leaving the tables in place."""
    with connect(engine) as connection:
        # Reversed so child tables go before the parents they reference.
        for table in reversed(EVENT_TABLES):
            connection.execute(delete(table))


def insert_rows(
    table: Table, rows: Sequence[dict[str, Any]], engine: Engine | None = None, chunk: int = 5_000
) -> int:
    """Bulk-insert rows into a table, in chunks.

    Chunking keeps a multi-million-row session load from building one enormous
    parameter list, which is where naive bulk inserts fall over.
    """
    if not rows:
        return 0
    with connect(engine) as connection:
        for start in range(0, len(rows), chunk):
            connection.execute(table.insert(), rows[start : start + chunk])
    return len(rows)


def read_sql(query: str, params: dict[str, Any] | None = None, engine: Engine | None = None):
    """Run a SQL query and return the result as a DataFrame."""
    with connect(engine) as connection:
        return pd.read_sql_query(text(query), connection, params=params or {})


def table_counts(engine: Engine | None = None) -> dict[str, int]:
    """Row count per warehouse table, for sanity checks and CLI output."""
    counts: dict[str, int] = {}
    with connect(engine) as connection:
        for table in EVENT_TABLES:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table.name}"))  # noqa: S608
            counts[table.name] = int(result.scalar_one())
    return counts
