"""Prefect flows that schedule the retraining pipeline.

This module is deliberately thin.  Every step's real work lives in
:mod:`src.orchestration.pipeline`; here they are wrapped in ``@task`` to gain
retries, structured logging and a run history, and composed into flows that can
be scheduled.

Run once, right now::

    python -m src.orchestration.flows run
    python -m src.orchestration.flows run --drift        # inject drift first
    python -m src.orchestration.flows backfill           # walk historical cutoffs

Schedule it (blocks, serving the flow on a cron)::

    python -m src.orchestration.flows serve --cron "0 3 * * *"

Prefect 3 starts a temporary API server automatically when no external one is
configured, so none of the above needs a running Prefect deployment.  Point
``PREFECT_API_URL`` at a real server to get the hosted UI and history instead.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Any

from prefect import flow, get_run_logger, task

from src.config import settings
from src.orchestration import pipeline

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
#
# Retries are set where a failure is plausibly transient - a database still
# starting up, a registry briefly unreachable. They are *not* set on the drift
# check, because a drift failure is a data problem: retrying it three times
# just produces the same answer three times, more slowly.


@task(name="ingest-events", retries=2, retry_delay_seconds=5)
def ingest_task(
    n_subscribers: int | None = None,
    seed: int = settings.RANDOM_SEED,
    force: bool = False,
    scenario: Any | None = None,
) -> dict[str, Any]:
    """Ensure the warehouse holds events (idempotent unless forced)."""
    result = pipeline.ingest_events(
        n_subscribers=n_subscribers, seed=seed, force=force, scenario=scenario
    )
    get_run_logger().info(
        "Warehouse holds %s events across %s tables (regenerated=%s)",
        f"{result['total_events']:,}",
        len(result["tables"]),
        result["regenerated"],
    )
    return result


@task(name="train-and-gate", retries=1, retry_delay_seconds=10)
def train_task(
    cutoffs: list[str] | None = None,
    promote: bool = True,
    seed: int = settings.RANDOM_SEED,
) -> dict[str, Any]:
    """Train on a temporal split, register the run, and run the promotion gate."""
    result = pipeline.train_and_gate(cutoffs=cutoffs, promote=promote, seed=seed)
    run_logger = get_run_logger()
    run_logger.info(
        "Trained: test ROC-AUC=%s PR-AUC=%s (threshold=%s)",
        result["test"].get("roc_auc"),
        result["test"].get("pr_auc"),
        result["threshold"],
    )
    if promote:
        run_logger.info(
            "Promotion %s - %s",
            "ACCEPTED" if result["promoted"] else "REJECTED",
            result["promotion_reason"],
        )
    return result


@task(name="assess-drift")
def drift_task(cutoff: str | None = None) -> dict[str, Any]:
    """Compare recent warehouse data against the drift baseline."""
    result = pipeline.assess_drift(cutoff=cutoff)
    run_logger = get_run_logger()
    if result.get("available"):
        run_logger.info(
            "Drift verdict: %s (%s drifted features)",
            result["overall_verdict"],
            len(result["drifted_features"]),
        )
    else:
        run_logger.info("Drift check skipped: %s", result.get("reason"))
    return result


@task(name="publish-report")
def report_task(
    ingest: dict[str, Any], training: dict[str, Any], drift: dict[str, Any]
) -> dict[str, Any]:
    """Assemble and persist the run report."""
    report = pipeline.build_run_report(ingest, training, drift)
    destination = pipeline.save_run_report(report)

    run_logger = get_run_logger()
    if report["needs_attention"]:
        # Loud, but not a failure: a rejected challenger means the gate did its
        # job, and drift means the world moved. Neither is a broken pipeline,
        # and failing the run would train whoever is on call to ignore it.
        run_logger.warning("Run needs attention: %s", "; ".join(report["attention_reasons"]))
    else:
        run_logger.info("Run clean: model promoted and no significant drift.")
    run_logger.info("Report written to %s", destination)
    return report


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #


@flow(name="subscriber-dropout-retraining", log_prints=True)
def retraining_flow(
    n_subscribers: int | None = None,
    cutoffs: list[str] | None = None,
    promote: bool = True,
    seed: int = settings.RANDOM_SEED,
    force_ingest: bool = False,
    scenario: Any | None = None,
) -> dict[str, Any]:
    """The scheduled pipeline: ingest -> drift -> train + gate -> report.

    The ordering matters and is not arbitrary. Drift is checked **after**
    ingest (it needs the fresh data) but **before** training (which overwrites
    ``reference_profile.json``). Checking it after training would compare a
    baseline against the very data it was just built from, which reports
    ``stable`` every time and would detect nothing, ever.

    Returns:
        The run report, which is also written to the artifacts directory.
    """
    ingest = ingest_task(
        n_subscribers=n_subscribers, seed=seed, force=force_ingest, scenario=scenario
    )
    drift = drift_task(wait_for=[ingest])
    training = train_task(cutoffs=cutoffs, promote=promote, seed=seed, wait_for=[drift])
    return report_task(ingest, training, drift)


@flow(name="subscriber-dropout-backfill", log_prints=True)
def backfill_flow(
    start: str | None = None,
    end: str | None = None,
    step_days: int = 30,
    promote: bool = False,
) -> list[dict[str, Any]]:
    """Train once per historical cutoff window, oldest first.

    This is what a scheduled pipeline would have produced had it been running
    since ``start`` - useful for seeding a registry with history, and for
    checking that model quality is stable over time rather than only good on
    the one window that happened to be tested.

    ``promote`` defaults to False: replaying history should not repeatedly
    move the production alias around based on old data.
    """
    from src.features.point_in_time import monthly_cutoffs

    first = start or settings.SIMULATION_START
    last = end or (
        date.fromisoformat(settings.SIMULATION_END)
        - timedelta(days=settings.PREDICTION_HORIZON_DAYS)
    ).isoformat()

    run_logger = get_run_logger()
    requested = [c.isoformat() for c in monthly_cutoffs(first, last, step_days=step_days)]
    all_cutoffs = pipeline.usable_cutoffs(requested)

    if len(all_cutoffs) < 3:
        run_logger.warning(
            "Only %s of %s cutoffs have enough history behind them; a temporal "
            "split needs at least 3. Widen the range or reduce --step-days.",
            len(all_cutoffs),
            len(requested),
        )
        return []
    if len(all_cutoffs) < len(requested):
        run_logger.info(
            "Skipped %s early cutoff(s) with too little history behind them",
            len(requested) - len(all_cutoffs),
        )

    # A temporal split needs at least three cutoffs (train/validation/test), so
    # each backfill step trains on everything up to that point - an expanding
    # window, which is how a real scheduled retrain accumulates history.
    results: list[dict[str, Any]] = []
    for index in range(3, len(all_cutoffs) + 1):
        window = all_cutoffs[:index]
        run_logger.info(
            "Backfill %s/%s -> test cutoff %s",
            index - 2,
            len(all_cutoffs) - 2,
            window[-1],
        )
        results.append(
            {
                "test_cutoff": window[-1],
                "n_cutoffs": len(window),
                **pipeline.train_and_gate(cutoffs=window, promote=promote),
            }
        )

    run_logger.info("Backfilled %s windows", len(results))
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _drift_scenario() -> Any:
    """A deliberate behavioural shift, for demonstrating the drift path."""
    from src.warehouse.simulate import DriftScenario

    midpoint = date.fromisoformat(settings.SIMULATION_START) + (
        date.fromisoformat(settings.SIMULATION_END)
        - date.fromisoformat(settings.SIMULATION_START)
    ) / 2
    return DriftScenario(
        starts_on=midpoint,
        engagement_multiplier=0.35,
        payment_failure_multiplier=2.5,
        cancellation_multiplier=1.4,
    )


def main(argv: list[str] | None = None) -> Any:  # pragma: no cover - CLI wiring
    """Entry point for running or serving the flows."""
    parser = argparse.ArgumentParser(description="Run the retraining pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the retraining flow once, now.")
    run.add_argument("--no-promote", action="store_true", help="Train without the gate.")
    run.add_argument("--force-ingest", action="store_true", help="Regenerate the warehouse.")
    run.add_argument(
        "--drift",
        action="store_true",
        help="Regenerate the warehouse with an injected behavioural shift, to "
        "demonstrate the drift path end to end.",
    )

    back = sub.add_parser("backfill", help="Train once per historical cutoff.")
    back.add_argument("--start", default=None)
    back.add_argument("--end", default=None)
    back.add_argument("--step-days", type=int, default=30)
    back.add_argument("--promote", action="store_true")

    serve = sub.add_parser("serve", help="Serve the flow on a schedule (blocks).")
    serve.add_argument("--cron", default="0 3 * * *", help="Cron schedule. Default: 03:00 daily.")

    args = parser.parse_args(argv)

    if args.command == "run":
        return retraining_flow(
            promote=not args.no_promote,
            force_ingest=args.force_ingest or args.drift,
            scenario=_drift_scenario() if args.drift else None,
        )
    if args.command == "backfill":
        return backfill_flow(
            start=args.start, end=args.end, step_days=args.step_days, promote=args.promote
        )
    return retraining_flow.serve(name="nightly-retraining", cron=args.cron)


if __name__ == "__main__":  # pragma: no cover
    main()
