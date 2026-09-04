"""Load an external dataset into the warehouse.

    python -m src.data.external.ingest --dataset kkbox --source-dir data/kkbox --dry-run
    python -m src.data.external.ingest --dataset kkbox --source-dir data/kkbox

``--dry-run`` first, always. It reads and maps everything, runs the full
contract validation, and writes nothing - which on a 30GB export is the
difference between finding out about a date-format surprise in ten minutes and
finding out three hours into a load that has already half-filled the tables.

Why this is a separate entry point from ``src.warehouse.simulate``
------------------------------------------------------------------

Because they are the same interface. Both fill the same five tables, and
everything downstream - point-in-time features, temporal splits, training,
promotion, drift - is written against those tables and cannot tell which one
ran. Swapping the simulator for real data is meant to be a change of command,
not a change of pipeline, and keeping the loaders behind one CLI is what makes
that claim testable rather than aspirational.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.data.external import contract, kkbox
from src.warehouse import database, schema

logger = logging.getLogger(__name__)

# Load order matters: subscribers first, so the cross-table validation has
# something to check the event tables against.
TABLES = ("subscribers", "subscription_events", "payments", "sessions", "support_tickets")


def build_kkbox_tables(source_dir: Path, session_chunk_size: int = 2_000_000) -> dict:
    """Map the KKBox export into warehouse-shaped frames.

    Sessions are returned as an iterator rather than a frame - the user log is
    around 400 million rows and will not fit in memory. Every other table is
    small enough to hold whole.
    """
    paths = kkbox.KKBoxPaths.under(source_dir)
    missing = paths.missing()
    if missing:
        raise FileNotFoundError(
            "Missing KKBox files: "
            + ", ".join(str(path) for path in missing)
            + ". Download and extract the WSDM Cup 2018 dataset into this directory."
        )

    events, payments = kkbox.load_transactions(paths.transactions)
    return {
        "subscribers": kkbox.load_members(paths.members),
        "subscription_events": events,
        "payments": payments,
        "sessions": lambda: kkbox.iter_user_logs(paths.user_logs, session_chunk_size),
        "support_tickets": kkbox.empty_support_tickets(),
    }


LOADERS: dict[str, Callable[[Path, int], dict]] = {"kkbox": build_kkbox_tables}


def drop_orphan_events(tables: dict) -> tuple[dict, dict[str, int]]:
    """Remove event rows whose subscriber is absent from ``subscribers``.

    Real exports are not referentially complete. KKBox's members file covers
    6.77M subscribers but its transaction log references 432,623 more that it
    never describes - 18.3% of everyone who actually transacts.

    Dropping is the conservative choice, and it is a modelling decision rather
    than a plumbing one. The alternative is to backfill a subscriber row per
    orphan with ``signup_date`` set to their first transaction. That is
    tempting and wrong here: the earliest inferable signup lands exactly on the
    first day of the transaction log, which means those subscribers almost
    certainly predate the data. Backfilling would assign them a signup that is
    too late, understate ``tenure_days``, and mark long-tenured stable
    subscribers as "still new" - which the model reads as a churn risk signal.

    Losing rows is visible and countable. Corrupting tenure for 432,623
    subscribers would be neither.
    """
    subscribers = tables.get("subscribers")
    if subscribers is None or subscribers.empty:
        return tables, {}

    known = set(subscribers["subscriber_id"])
    cleaned = dict(tables)
    dropped: dict[str, int] = {}

    for name, value in tables.items():
        if name == "subscribers" or callable(value) or value.empty:
            continue
        if "subscriber_id" not in value.columns:
            continue
        keep = value["subscriber_id"].isin(known)
        removed = int((~keep).sum())
        if removed:
            cleaned[name] = value[keep].reset_index(drop=True)
            dropped[name] = removed

    # Streamed tables are filtered lazily, chunk by chunk, so the 400M-row
    # user log never has to be held in memory just to drop a few thousand rows.
    for name, value in tables.items():
        if callable(value):
            cleaned[name] = _filtered_stream(value, known)

    return cleaned, dropped


def _filtered_stream(factory, known: set):
    """Wrap a chunk iterator so each chunk is filtered as it arrives."""

    def wrapped():
        for chunk in factory():
            yield chunk[chunk["subscriber_id"].isin(known)]

    return wrapped


def _validation_sample(tables: dict, session_rows: int = 500_000) -> dict[str, pd.DataFrame]:
    """Materialise enough of each table to validate against the contract.

    Sessions are sampled - the first chunk only. Validating 400 million rows to
    discover a malformed date costs an hour and finds the same answer the first
    two million rows would have. The sample is honest about being one: the
    report says how many rows it covered.
    """
    sample: dict[str, pd.DataFrame] = {}
    for name, value in tables.items():
        if callable(value):
            first = next(iter(value()), None)
            sample[name] = pd.DataFrame() if first is None else first.head(session_rows)
        else:
            sample[name] = value
    return sample


def validate(tables: dict) -> tuple[bool, str]:
    """Run the contract over a candidate load.

    Returns:
        ``(ok, report_text)``. ``ok`` is False if any table has errors;
        warnings never block, because "this dataset has no support tickets" is
        a fact to know rather than a reason to refuse.
    """
    reports = contract.validate_load(_validation_sample(tables))
    return all(report.ok for report in reports), contract.summarise(reports)


def write(tables: dict, engine=None, chunk_size: int = 50_000) -> dict[str, int]:
    """Insert the mapped tables into the warehouse, replacing what is there.

    Truncates first. A partial re-run that appended would silently double every
    subscriber's event history, and the resulting features would look plausible
    - twice the sessions, twice the payments - while being wrong everywhere.
    """
    database.create_schema(engine)
    database.truncate_all(engine)

    counts: dict[str, int] = {}
    for name in TABLES:
        value = tables[name]
        table = getattr(schema, name)

        if callable(value):
            # Streamed. Inserted chunk by chunk so peak memory stays flat
            # regardless of how large the source file is.
            total = 0
            for chunk in value():
                rows = chunk.to_dict("records")
                database.insert_rows(table, rows, chunk=chunk_size, engine=engine)
                total += len(rows)
                logger.info("%s: %s rows written", name, f"{total:,}")
            counts[name] = total
            continue

        rows = value.to_dict("records")
        if rows:
            database.insert_rows(table, rows, chunk=chunk_size, engine=engine)
        counts[name] = len(rows)
        logger.info("%s: %s rows written", name, f"{len(rows):,}")

    return counts


def _report_dead_features(dataset: str) -> str:
    """Name the features this dataset cannot support.

    Stated at load time, on purpose. Discovering that a third of the feature
    set is constant by noticing an all-zero column in a feature-importance
    chart is how people end up trusting a model built on less than they think.
    """
    dead = kkbox.DEAD_FEATURES if dataset == "kkbox" else {}
    if not dead:
        return ""

    lines = [
        "",
        "Features that lose all variance on this dataset:",
        *(f"  {name:<32} {reason}" for name, reason in dead.items()),
        "",
        "This is not a defect in the loader. The schema this project was built",
        "around and the data actually available do not line up, and the honest",
        "response is to say which features went dark rather than to synthesise",
        "plausible-looking values to fill the gap.",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load an external dataset into the warehouse.")
    parser.add_argument("--dataset", choices=sorted(LOADERS), default="kkbox")
    parser.add_argument(
        "--source-dir", type=Path, required=True, help="Directory holding the extracted CSVs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Map and validate everything, write nothing. Run this first.",
    )
    parser.add_argument(
        "--session-chunk-size",
        type=int,
        default=2_000_000,
        help="Rows per chunk when streaming the usage log.",
    )
    parser.add_argument(
        "--drop-orphans",
        action="store_true",
        help=(
            "Discard event rows whose subscriber is missing from the "
            "subscribers table. Required for KKBox, whose members file does "
            "not cover every subscriber in its transaction log."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even if validation found errors. Rarely the right answer.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    tables = LOADERS[args.dataset](args.source_dir, args.session_chunk_size)

    if args.drop_orphans:
        tables, dropped = drop_orphan_events(tables)
        if dropped:
            print("Dropped event rows with no matching subscriber:")
            for name, count in dropped.items():
                print(f"  {name:<22} {count:>12,}")
            print()

    ok, report = validate(tables)
    print(report)
    print(_report_dead_features(args.dataset))

    if not ok and not args.force:
        print("\nValidation failed. Fix the source data, or re-run with --force.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    counts = write(tables)
    print("\nLoaded:")
    for name, count in counts.items():
        print(f"  {name:<22} {count:>12,}")
    print("\nNext: python -m src.models.train --source warehouse")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
