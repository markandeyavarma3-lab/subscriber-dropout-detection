"""The retraining pipeline, as plain Python functions.

Deliberately **free of any Prefect import**.  :mod:`src.orchestration.flows`
wraps each function below in a ``@task`` and composes them into a flow, but the
logic itself does not know an orchestrator exists.

Two things follow from that separation:

*Testable.* Prefect 3 spins up a temporary API server for every flow run, which
costs several seconds.  Testing the logic here instead of through the flow keeps
the suite fast, and still covers the part that can actually be wrong.

*Portable.* Swapping Prefect for Airflow, Dagster, or a plain cron script means
rewriting :mod:`~src.orchestration.flows` only.  The pipeline is not held
hostage by the scheduler, which is exactly the coupling that makes orchestration
migrations painful in real projects.

Each step returns a JSON-serialisable dict so a flow run leaves behind a report
that can be diffed, alerted on, or stored - not just log lines to scrape.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Step 1: ingest
# --------------------------------------------------------------------------- #


def ingest_events(
    n_subscribers: int | None = None,
    start: str | None = None,
    end: str | None = None,
    seed: int = settings.RANDOM_SEED,
    force: bool = False,
    scenario: Any | None = None,
) -> dict[str, Any]:
    """Make sure the warehouse holds events, and report what is in it.

    **Idempotent by default**, which is the property that matters for a
    scheduled job: re-running the flow must not silently regenerate the world
    underneath a model that was trained on the previous version of it.  Pass
    ``force=True`` (or a ``scenario``) to deliberately rewrite the warehouse.

    Args:
        n_subscribers: Population size when generating from scratch.
        start: First simulated day. Defaults to settings.
        end: Last simulated day. Defaults to settings.
        seed: Seed for reproducibility.
        force: Regenerate even when the warehouse already has rows.
        scenario: A :class:`~src.warehouse.simulate.DriftScenario` to inject.
            Implies ``force``, since behaviour has to be rewritten to change.

    Returns:
        ``{"regenerated": bool, "tables": {...}, "total_events": int}``
    """
    from src.warehouse.database import table_counts
    from src.warehouse.simulate import ensure_warehouse, simulate_events

    regenerate = force or scenario is not None
    if regenerate:
        result = simulate_events(
            n_subscribers=n_subscribers or 4_000,
            start=start,
            end=end,
            seed=seed,
            scenario=scenario,
        )
        logger.info("Regenerated warehouse: %s", result.summary())
        regenerated = True
    else:
        populated = ensure_warehouse(
            n_subscribers=n_subscribers, start=start, end=end, seed=seed
        )
        regenerated = populated is not None
        if regenerated:
            logger.info("Warehouse was empty; populated it: %s", populated.summary())

    counts = table_counts()
    return {
        "regenerated": regenerated,
        "scenario_applied": scenario is not None,
        "tables": counts,
        # subscribers is a dimension, not an event; excluding it makes this a
        # meaningful "how much behaviour do we have" number.
        "total_events": int(sum(v for k, v in counts.items() if k != "subscribers")),
    }


# --------------------------------------------------------------------------- #
# Step 2: train, register, gate
# --------------------------------------------------------------------------- #


def train_and_gate(
    cutoffs: list[str] | None = None,
    model_params: dict[str, Any] | None = None,
    promote: bool = True,
    seed: int = settings.RANDOM_SEED,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Train on point-in-time data, register the run, and run the promotion gate.

    Calls :func:`src.models.train.run_training` rather than reimplementing it:
    the orchestration layer schedules and reports on work, it does not own a
    second copy of the training logic that could drift from the first.

    Returns:
        A summary including the gate's verdict, so a caller can alert on a
        rejected challenger without parsing logs.
    """
    from src.models.train import run_training

    result = run_training(
        source="warehouse",
        cutoffs=cutoffs,
        model_params=model_params,
        seed=seed,
        artifacts_dir=artifacts_dir,
        track=True,
        promote_model=promote,
        verbose=False,
    )

    promotion = result.promotion or {}
    quality = result.decision_quality or {}
    fairness = quality.get("fairness") or {}
    costs = quality.get("costs") or {}

    return {
        "threshold": result.threshold,
        "validation": _headline(result.validation_metrics),
        "test": _headline(result.test_metrics),
        "mlflow_run_id": result.mlflow_run_id,
        "promoted": promotion.get("promoted"),
        "promotion_reason": promotion.get("reason"),
        "challenger_score": promotion.get("challenger_score"),
        "champion_score": promotion.get("champion_score"),
        "model_path": str(result.model_path),
        # Surfaced so the run report can escalate on them. A model that works
        # measurably worse for one group is not something to leave sitting in
        # an artifact nobody opens.
        "fairness_passes": fairness.get("passes"),
        "fairness_concerns": fairness.get("concerns", []),
        "calibration_ece": (quality.get("calibration") or {}).get(
            "expected_calibration_error"
        ),
        "cost_savings_available": costs.get("savings"),
    }


def usable_cutoffs(cutoffs: list[str], min_subscribers: int = 50) -> list[str]:
    """Drop cutoffs with too little history behind them to train on.

    The earliest cutoffs in any simulation window are genuinely empty: on day
    one nobody has signed up yet, so a snapshot taken then has no rows and a
    temporal split built from it has nothing to fit. That is not a bug to
    swallow inside the splitter - it is a question the caller should not have
    asked - so backfills filter the list first rather than crashing partway
    through a long replay.

    Uses a cheap COUNT rather than building each snapshot, so filtering a long
    cutoff list costs one small query per cutoff instead of a full feature
    build.
    """
    from src.warehouse.database import read_sql

    keep: list[str] = []
    for cutoff in cutoffs:
        found = read_sql(
            "SELECT COUNT(*) AS n FROM subscribers WHERE signup_date < :cutoff",
            {"cutoff": cutoff},
        )
        if int(found.iloc[0]["n"]) >= min_subscribers:
            keep.append(cutoff)
        else:
            logger.info("Skipping cutoff %s: too little history behind it", cutoff)
    return keep


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    """Pull the few metrics worth carrying into a run report."""
    return {
        key: metrics.get(key)
        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall", "n_samples")
    }


# --------------------------------------------------------------------------- #
# Step 3: drift
# --------------------------------------------------------------------------- #


def assess_drift(cutoff: str | date | None = None) -> dict[str, Any]:
    """Score the most recent slice of warehouse data against the baseline.

    Runs *after* training on purpose.  The reference profile written by this
    run is the one the next run will be compared against, so drift here answers
    "has the world moved since the model that is serving was fitted?" rather
    than comparing a profile to the data it was just built from - which would
    trivially report ``stable`` every time and detect nothing, ever.

    Returns a report with ``"available": False`` rather than raising when there
    is no baseline yet: a first-ever run has nothing to compare against, and
    that is not a pipeline failure.
    """
    from src.features.build_features import REQUIRED_INPUT_COLUMNS
    from src.features.point_in_time import build_training_snapshot
    from src.monitoring.drift import detect_drift
    from src.monitoring.profile import load_reference_profile

    profile = load_reference_profile()
    if profile is None:
        return {"available": False, "reason": "no reference profile has been written yet"}

    as_of = cutoff or _latest_scorable_cutoff()
    frame, window = build_training_snapshot(as_of)
    if frame.empty:
        return {
            "available": False,
            "reason": f"no subscribers found as of {window.cutoff.isoformat()}",
        }

    report = detect_drift(frame[REQUIRED_INPUT_COLUMNS], profile)
    return {
        "available": True,
        "cutoff": window.cutoff.isoformat(),
        "n_samples": report["n_samples"],
        "sufficient_sample": report["sufficient_sample"],
        "overall_verdict": report["overall_verdict"],
        "drifted_features": report["drifted_features"],
        "top_features": [
            {"feature": item["feature"], "psi": item["psi"], "verdict": item["verdict"]}
            for item in report["features"][:5]
        ],
    }


def _latest_scorable_cutoff() -> date:
    """The most recent date whose label window fits inside the simulation."""
    return date.fromisoformat(settings.SIMULATION_END) - timedelta(
        days=settings.PREDICTION_HORIZON_DAYS
    )


# --------------------------------------------------------------------------- #
# Step 4: report
# --------------------------------------------------------------------------- #


def build_run_report(
    ingest: dict[str, Any],
    training: dict[str, Any],
    drift: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the single artifact a pipeline run leaves behind.

    ``needs_attention`` is the field worth alerting on. It fires on three
    things, all of which a human should look at and none of which should stop
    the pipeline:

    - a challenger was rejected, so the model did not improve;
    - live data has drifted significantly, so the model may be going stale;
    - the fairness audit found a disparity, so the model may be working
      measurably worse for some group.

    The third was missing initially, which meant a model that discriminated
    could ship with an entirely green pipeline: the audit ran, wrote its
    verdict to an artifact, and nothing ever read it.
    """
    rejected = training.get("promoted") is False
    drifted = drift.get("available") and drift.get("overall_verdict") == "significant"
    unfair = training.get("fairness_passes") is False

    reasons: list[str] = []
    if rejected:
        reasons.append(f"challenger rejected: {training.get('promotion_reason')}")
    if drifted:
        reasons.append(f"significant drift in {', '.join(drift.get('drifted_features', []))}")
    if unfair:
        concerns = training.get("fairness_concerns") or ["disparity detected"]
        reasons.append(f"fairness: {concerns[0]}")

    return {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ingest": ingest,
        "training": training,
        "drift": drift,
        "needs_attention": bool(reasons),
        "attention_reasons": reasons,
    }


def save_run_report(report: dict[str, Any], path: Path | None = None) -> Path:
    """Write a run report to disk, newest run overwriting the previous one."""
    import json

    destination = path or (settings.ARTIFACTS_DIR / "last_pipeline_run.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str))
    return destination
