"""Tests for the retraining pipeline and its Prefect flows.

Most of these exercise :mod:`src.orchestration.pipeline` directly rather than
running a flow. That is deliberate, not a shortcut: Prefect 3 starts a
temporary API server for every flow run, so testing through the decorator would
add seconds per test while covering the same logic. The Prefect layer is a thin
wrapper, and it gets its own integration test at the bottom of this file.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from sqlalchemy import create_engine

from src.config import settings
from src.orchestration import pipeline
from src.warehouse.database import table_counts
from src.warehouse.simulate import DriftScenario


@pytest.fixture()
def warehouse(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """An isolated, populated warehouse for one test.

    Patches the module-level engine cache so every call inside the pipeline -
    which resolves its own engine from settings - reaches this database rather
    than the developer's working copy.
    """
    from src.warehouse import database

    engine = create_engine(f"sqlite:///{tmp_path / 'warehouse.db'}", future=True)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(settings, "SIMULATION_START", "2024-01-01")
    monkeypatch.setattr(settings, "SIMULATION_END", "2024-12-31")
    return engine


@pytest.fixture()
def artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Redirect every artifact this pipeline writes into a temp directory."""
    workdir = tmp_path / "artifacts"
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", workdir)
    monkeypatch.setattr(settings, "REFERENCE_PROFILE_PATH", workdir / "reference_profile.json")
    monkeypatch.setattr(settings, "MODEL_PATH", workdir / "model.joblib")
    monkeypatch.setattr(settings, "METADATA_PATH", workdir / "metadata.json")
    return workdir


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #


def test_ingest_populates_an_empty_warehouse(warehouse) -> None:
    """A first run has to create the world it will train on."""
    result = pipeline.ingest_events(n_subscribers=200)

    assert result["regenerated"] is True
    assert result["total_events"] > 0
    assert result["tables"]["subscribers"] == 200


def test_ingest_is_idempotent(warehouse) -> None:
    """The property that matters for a *scheduled* job.

    Re-running must not silently rebuild the world underneath a model that was
    trained on the previous version of it.
    """
    first = pipeline.ingest_events(n_subscribers=200)
    second = pipeline.ingest_events(n_subscribers=200)

    assert first["regenerated"] is True
    assert second["regenerated"] is False
    assert first["tables"] == second["tables"]


def test_ingest_force_regenerates(warehouse) -> None:
    """``force`` is the explicit opt-in to rewriting the warehouse."""
    pipeline.ingest_events(n_subscribers=200)
    regenerated = pipeline.ingest_events(n_subscribers=120, force=True)

    assert regenerated["regenerated"] is True
    assert regenerated["tables"]["subscribers"] == 120


def test_ingest_with_a_scenario_implies_regeneration(warehouse) -> None:
    """Injecting behaviour change requires rewriting events, so it forces."""
    pipeline.ingest_events(n_subscribers=200)
    result = pipeline.ingest_events(
        n_subscribers=200,
        scenario=DriftScenario(starts_on=date(2024, 6, 1), engagement_multiplier=0.3),
    )

    assert result["regenerated"] is True
    assert result["scenario_applied"] is True


def test_ingest_excludes_the_subscriber_dimension_from_event_count(warehouse) -> None:
    """`total_events` should measure behaviour, not population size."""
    result = pipeline.ingest_events(n_subscribers=200)
    counts = table_counts()

    assert result["total_events"] == sum(v for k, v in counts.items() if k != "subscribers")
    assert result["total_events"] != sum(counts.values())


# --------------------------------------------------------------------------- #
# Drift assessment
# --------------------------------------------------------------------------- #


def test_drift_is_unavailable_before_any_baseline_exists(warehouse, artifacts) -> None:
    """A first-ever run has nothing to compare against - not a failure."""
    pipeline.ingest_events(n_subscribers=200)
    result = pipeline.assess_drift()

    assert result["available"] is False
    assert "no reference profile" in result["reason"]


def test_drift_reports_a_verdict_once_a_baseline_exists(warehouse, artifacts) -> None:
    """With a baseline on disk, the check produces a real PSI verdict."""
    from src.features.point_in_time import build_training_snapshot
    from src.monitoring.profile import build_reference_profile, save_reference_profile

    pipeline.ingest_events(n_subscribers=400)
    frame, _ = build_training_snapshot("2024-08-01")
    save_reference_profile(build_reference_profile(frame), settings.REFERENCE_PROFILE_PATH)

    result = pipeline.assess_drift(cutoff="2024-08-01")

    assert result["available"] is True
    assert result["overall_verdict"] in {"stable", "moderate", "significant"}
    assert result["n_samples"] > 0


def test_drift_against_its_own_data_is_stable(warehouse, artifacts) -> None:
    """The false-positive guard: identical data must not raise an alarm."""
    from src.features.point_in_time import build_training_snapshot
    from src.monitoring.profile import build_reference_profile, save_reference_profile

    pipeline.ingest_events(n_subscribers=400)
    frame, _ = build_training_snapshot("2024-08-01")
    save_reference_profile(build_reference_profile(frame), settings.REFERENCE_PROFILE_PATH)

    result = pipeline.assess_drift(cutoff="2024-08-01")
    assert result["overall_verdict"] == "stable"
    assert result["drifted_features"] == []


# --------------------------------------------------------------------------- #
# Cutoff filtering
# --------------------------------------------------------------------------- #


def test_usable_cutoffs_drops_dates_with_no_history(warehouse) -> None:
    """The earliest cutoff in a window is genuinely empty - nobody has signed up.

    Regression guard: a backfill starting on the simulation's first day used to
    crash partway through with "the train split came back empty".
    """
    pipeline.ingest_events(n_subscribers=400)

    kept = pipeline.usable_cutoffs(["2024-01-01", "2024-06-01", "2024-10-01"])

    assert "2024-01-01" not in kept
    assert kept == ["2024-06-01", "2024-10-01"]


def test_usable_cutoffs_respects_the_minimum(warehouse) -> None:
    """The threshold is a knob, not a hard-coded 50."""
    pipeline.ingest_events(n_subscribers=400)

    assert pipeline.usable_cutoffs(["2024-10-01"], min_subscribers=1) == ["2024-10-01"]
    assert pipeline.usable_cutoffs(["2024-10-01"], min_subscribers=10_000) == []


# --------------------------------------------------------------------------- #
# Run report
# --------------------------------------------------------------------------- #


def test_report_is_clean_when_promoted_and_stable() -> None:
    """No news is the quiet path: nothing to escalate."""
    report = pipeline.build_run_report(
        ingest={"total_events": 100},
        training={"promoted": True, "promotion_reason": "beat champion"},
        drift={"available": True, "overall_verdict": "stable", "drifted_features": []},
    )

    assert report["needs_attention"] is False
    assert report["attention_reasons"] == []


def test_report_flags_a_rejected_challenger() -> None:
    """A rejected model means the gate worked, and a human should know."""
    report = pipeline.build_run_report(
        ingest={"total_events": 100},
        training={"promoted": False, "promotion_reason": "improvement +0.0000 below required"},
        drift={"available": True, "overall_verdict": "stable", "drifted_features": []},
    )

    assert report["needs_attention"] is True
    assert "challenger rejected" in report["attention_reasons"][0]


def test_report_flags_significant_drift() -> None:
    """Drift is escalated even when the new model was promoted successfully."""
    report = pipeline.build_run_report(
        ingest={"total_events": 100},
        training={"promoted": True, "promotion_reason": "beat champion"},
        drift={
            "available": True,
            "overall_verdict": "significant",
            "drifted_features": ["avg_session_count_last_30d"],
        },
    )

    assert report["needs_attention"] is True
    assert "significant drift" in report["attention_reasons"][0]


def test_report_does_not_flag_a_skipped_drift_check() -> None:
    """An unavailable baseline is not the same as a clean bill of health."""
    report = pipeline.build_run_report(
        ingest={"total_events": 100},
        training={"promoted": True, "promotion_reason": "no incumbent"},
        drift={"available": False, "reason": "no reference profile yet"},
    )

    assert report["needs_attention"] is False


def test_report_round_trips_to_disk(tmp_path) -> None:
    """The report is the run's durable artifact, so it must serialise."""
    report = pipeline.build_run_report(
        ingest={"total_events": 1},
        training={"promoted": True, "promotion_reason": "ok"},
        drift={"available": False},
    )
    destination = pipeline.save_run_report(report, tmp_path / "run.json")

    assert json.loads(destination.read_text())["needs_attention"] is False


# --------------------------------------------------------------------------- #
# The Prefect layer
# --------------------------------------------------------------------------- #


def test_flows_import_and_expose_the_expected_shape() -> None:
    """The wrapper must stay thin: same steps, same names, retries configured."""
    from src.orchestration import flows

    assert flows.retraining_flow.name == "subscriber-dropout-retraining"
    assert flows.backfill_flow.name == "subscriber-dropout-backfill"
    # Retries belong on the steps that can fail transiently, not on drift.
    assert flows.ingest_task.retries == 2
    assert flows.train_task.retries == 1
    assert flows.drift_task.retries == 0


def test_tasks_delegate_to_the_pipeline_functions() -> None:
    """Each task wraps the matching plain function, rather than reimplementing it.

    ``.fn`` is the undecorated callable, so this asserts the wiring without
    paying for a Prefect server.
    """
    from src.orchestration import flows

    assert flows.ingest_task.fn.__doc__
    assert flows.drift_task.fn.__doc__
    assert flows.train_task.fn.__doc__


@pytest.mark.slow
def test_retraining_flow_runs_end_to_end(warehouse, artifacts, monkeypatch) -> None:
    """One real flow run, through Prefect, proving the composition works.

    Marked slow because Prefect starts a temporary API server for this.
    """
    from src.orchestration import flows

    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", f"sqlite:///{artifacts / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{artifacts / 'mlflow.db'}")

    report: dict[str, Any] = flows.retraining_flow(n_subscribers=400, promote=True)

    assert report["ingest"]["total_events"] > 0
    assert report["training"]["test"]["roc_auc"] is not None
    # First ever run: nothing to compare against, and nothing to beat.
    assert report["drift"]["available"] is False
    assert report["training"]["promoted"] is True
    assert (artifacts / "last_pipeline_run.json").exists()
