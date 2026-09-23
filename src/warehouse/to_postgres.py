"""Copy the SQLite warehouse into the compose stack's Postgres.

    python -m src.warehouse.to_postgres                 # full copy
    python -m src.warehouse.to_postgres --check         # compare counts only

The KKBox load was run into SQLite, because a 30GB ingest into a local file
needs no server. This moves the finished warehouse into Postgres without
re-running that ingest: the cleaning decisions (orphans dropped, ids widened,
fees normalised) are already applied, and repeating them is hours of work that
would produce the same rows.

Why COPY through a pipe rather than ``database.insert_rows``
------------------------------------------------------------

``insert_rows`` is right for the simulator's few hundred thousand rows. For 82.8
million it means 82.8 million bound parameter sets through SQLAlchemy, which is
many hours. Here the ``sqlite3`` CLI writes CSV straight into ``psql``'s
``COPY ... FROM STDIN`` inside the container: no temp files on a disk that has
to hold both copies, no ``psql`` needed on the host, and Postgres's bulk path
does the parsing.

The schema itself still comes from ``schema.metadata`` via
``database.create_schema``, so the Postgres tables are exactly the ones the
pipeline's SQLAlchemy code expects - this module only moves rows.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, Table, inspect, text

from src.config import settings
from src.warehouse import database
from src.warehouse.schema import EVENT_TABLES

logger = logging.getLogger(__name__)

DEFAULT_URL = "postgresql+psycopg://subscriber:subscriber@localhost:5432/warehouse"
DEFAULT_CONTAINER = "subscriber-warehouse"
DEFAULT_SLICE = 5_000_000

# Tables MLflow creates. The compose MLflow was once pointed at this same
# database, so they may be sitting beside the warehouse; they are reported,
# never touched, because they are someone else's data.
MLFLOW_TABLES = {"experiments", "runs", "registered_models", "model_versions", "alembic_version"}

# The demo's first query reads this. It lives outside `schema.metadata`
# because it describes the warehouse rather than being part of it.
SUMMARY_TABLE = "warehouse_summary"


# --------------------------------------------------------------------------- #
# SQL builders - pure, so they can be tested without either database
# --------------------------------------------------------------------------- #


def column_names(table: Table) -> list[str]:
    """Column order for both sides of the copy, taken from the schema."""
    return [column.name for column in table.columns]


def select_sql(table: Table, low: int, high: int) -> str:
    """One slice of the SQLite table, by rowid, in schema column order.

    Slicing by rowid rather than LIMIT/OFFSET keeps every slice an index range
    scan; OFFSET would re-read everything before it, making the load quadratic.
    """
    columns = ", ".join(column_names(table))
    return f"SELECT {columns} FROM {table.name} WHERE rowid > {low} AND rowid <= {high};"  # noqa: S608


def copy_sql(table: Table) -> str:
    """The Postgres side: read CSV from psql's stdin into the named columns."""
    return f"COPY {table.name} ({', '.join(column_names(table))}) FROM STDIN WITH (FORMAT csv)"


def slices(first: int, last: int, size: int) -> list[tuple[int, int]]:
    """Half-open ``(low, high]`` rowid ranges that cover ``first..last`` exactly once."""
    if last < first:
        return []
    bounds = list(range(first - 1, last, size)) + [last]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def serial_column(table: Table) -> str | None:
    """The auto-incrementing integer key, if the table has one.

    Rows are copied with their ids, so Postgres's sequence never advanced; left
    alone, the next ordinary insert would collide with row 1.
    """
    for column in table.primary_key.columns:
        if column.autoincrement is True or (
            column.autoincrement == "auto" and column.type.python_type is int
        ):
            return column.name
    return None


def sequence_reset_sql(table: Table) -> str | None:
    column = serial_column(table)
    if column is None:
        return None
    return (
        f"SELECT setval(pg_get_serial_sequence('{table.name}', '{column}'), "
        f"COALESCE(MAX({column}), 1), MAX({column}) IS NOT NULL) FROM {table.name}"  # noqa: S608
    )


# --------------------------------------------------------------------------- #
# The copy
# --------------------------------------------------------------------------- #


def _psql(container: str, *args: str, stdin=None) -> subprocess.Popen:
    return subprocess.Popen(
        ["docker", "exec", "-i", container, "psql", "-U", "subscriber", "-d", "warehouse",
         "-v", "ON_ERROR_STOP=1", "-q", *args],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def copy_table(table: Table, sqlite_path: Path, container: str, slice_size: int) -> int:
    """Stream one table across, slice by slice, and return the rows copied."""
    with sqlite3.connect(sqlite_path) as source:
        first, last = source.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {table.name}").fetchone()  # noqa: S608
    if first is None:
        logger.info("%-20s empty, nothing to copy", table.name)
        return 0

    copied = 0
    started = time.monotonic()
    for low, high in slices(first, last, slice_size):
        producer = subprocess.Popen(
            ["sqlite3", "-csv", "-noheader", str(sqlite_path), select_sql(table, low, high)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # synchronous_commit off for the load only: a crash mid-load means
        # re-running the load, which truncates first anyway.
        consumer = _psql(
            container, "-c", "SET synchronous_commit = off", "-c", copy_sql(table),
            stdin=producer.stdout,
        )
        producer.stdout.close()  # so a failing consumer sends SIGPIPE upstream
        out, err = consumer.communicate()
        producer.wait()

        if consumer.returncode != 0 or producer.returncode != 0:
            detail = err.strip() or producer.stderr.read().decode().strip()
            raise RuntimeError(f"copy of {table.name} rows {low}-{high} failed: {detail}")

        copied += high - low
        rate = copied / max(time.monotonic() - started, 1e-9)
        logger.info("%-20s %12s / %s rows  (%s rows/s)", table.name,
                    f"{copied:,}", f"{last - first + 1:,}", f"{rate:,.0f}")
    return copied


def foreign_tables(engine: Engine) -> set[str]:
    return MLFLOW_TABLES & set(inspect(engine).get_table_names())


def copy_summary(sqlite_path: Path, engine: Engine) -> None:
    """Carry the demo's row-count table across, if the SQLite side has one."""
    with sqlite3.connect(sqlite_path) as source:
        exists = source.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (SUMMARY_TABLE,)
        ).fetchone()
        if not exists:
            return
        frame = pd.read_sql_query(f"SELECT * FROM {SUMMARY_TABLE}", source)  # noqa: S608
    frame.to_sql(SUMMARY_TABLE, engine, if_exists="replace", index=False)


def sqlite_counts(sqlite_path: Path) -> dict[str, int]:
    with sqlite3.connect(sqlite_path) as source:
        return {
            table.name: source.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()[0]  # noqa: S608
            for table in EVENT_TABLES
        }


def compare_counts(sqlite_path: Path, engine: Engine) -> bool:
    expected = sqlite_counts(sqlite_path)
    actual = database.table_counts(engine)
    ok = True
    for name, want in expected.items():
        got = actual.get(name)
        mark = "ok" if got == want else "MISMATCH"
        ok &= got == want
        shown = "missing" if got is None else f"{got:>12,}"
        print(f"  {name:22s} sqlite {want:>12,}   postgres {shown}   {mark}")
    return ok


def load(sqlite_path: Path, url: str, container: str, slice_size: int) -> bool:
    engine = database.get_engine(url)

    stray = foreign_tables(engine)
    if stray:
        logger.warning(
            "MLflow tables found in this database (%s). They are left alone; "
            "the compose MLflow now uses its own database.", ", ".join(sorted(stray)))

    database.create_schema(engine)

    with engine.begin() as connection:
        # Secondary indexes are dropped for the load and rebuilt after: building
        # a btree once over 38M sorted rows is far cheaper than maintaining it
        # through 38M individual inserts.
        for table in EVENT_TABLES:
            for index in table.indexes:
                index.drop(connection, checkfirst=True)
        connection.execute(text(
            "TRUNCATE " + ", ".join(t.name for t in EVENT_TABLES) + " RESTART IDENTITY"))

    for table in EVENT_TABLES:
        copy_table(table, sqlite_path, container, slice_size)

    with engine.begin() as connection:
        connection.execute(text("SET maintenance_work_mem = '1GB'"))
        for table in EVENT_TABLES:
            for index in table.indexes:
                started = time.monotonic()
                index.create(connection, checkfirst=True)
                logger.info("index %-34s built in %.0fs", index.name, time.monotonic() - started)
            reset = sequence_reset_sql(table)
            if reset:
                connection.execute(text(reset))

    copy_summary(sqlite_path, engine)

    # ANALYZE outside a transaction block is not required, but VACUUM would
    # be; plain ANALYZE is what the planner needs after a bulk load.
    with engine.begin() as connection:
        connection.execute(text("ANALYZE"))

    print("\nRow counts, SQLite against Postgres:")
    return compare_counts(sqlite_path, engine)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sqlite", type=Path, default=settings.DATA_DIR / "warehouse.db")
    parser.add_argument("--url", default=os.getenv("SDD_POSTGRES_URL", DEFAULT_URL))
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--slice-size", type=int, default=DEFAULT_SLICE)
    parser.add_argument("--check", action="store_true", help="compare row counts and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if not args.sqlite.exists():
        print(f"No SQLite warehouse at {args.sqlite}", file=sys.stderr)
        return 1

    if args.check:
        print("Row counts, SQLite against Postgres:")
        return 0 if compare_counts(args.sqlite, database.get_engine(args.url)) else 1

    return 0 if load(args.sqlite, args.url, args.container, args.slice_size) else 1


if __name__ == "__main__":
    raise SystemExit(main())
