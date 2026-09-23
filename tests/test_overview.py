"""Tests for the Overview endpoint behind the dashboard's first page.

Its job is to tell the project's story from the running system, so the tests
check two things: that the numbers come from the real sources (the model's
report, the warehouse), and that a missing source degrades to
``available: false`` instead of breaking the page.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from src.api import overview
from src.config import settings
from src.warehouse import schema
from src.warehouse.database import create_schema, insert_rows


@pytest.fixture(autouse=True)
def _fresh_cache():
    overview.reset_cache()
    yield
    overview.reset_cache()


def test_the_model_section_comes_from_the_models_own_report(trained_model) -> None:
    report = overview.model_report(trained_model.metrics_path.parent)
    written = json.loads(trained_model.metrics_path.read_text())

    assert report["available"] is True
    assert report["test"]["pr_auc"] == written["test"]["pr_auc"]
    assert report["test"]["roc_auc"] == written["test"]["roc_auc"]
    assert len(report["top_features"]) <= 6


def test_a_missing_report_degrades_instead_of_failing(tmp_path: Path) -> None:
    assert overview.model_report(tmp_path)["available"] is False
    assert overview.drift_report(tmp_path)["available"] is False


def test_the_drift_section_reads_the_last_pipeline_run(tmp_path: Path) -> None:
    (tmp_path / "last_pipeline_run.json").write_text(json.dumps({
        "finished_at": "2026-09-23T10:15:33+00:00",
        "drift": {"available": True, "cutoff": "2017-02-28", "overall_verdict": "stable",
                  "n_samples": 7614, "top_features": [{"feature": "tenure_days", "psi": 0.03}]},
    }))
    drift = overview.drift_report(tmp_path)
    assert drift["verdict"] == "stable" and drift["n_samples"] == 7614


def test_warehouse_counts_every_event_table(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'wh.db'}"
    engine = create_engine(url)
    create_schema(engine)
    insert_rows(schema.subscribers, [
        {"subscriber_id": f"S{i}", "signup_date": __import__("datetime").date(2024, 1, 1),
         "acquisition_channel": "organic"} for i in range(3)
    ], engine=engine)
    engine.dispose()

    summary = overview.warehouse_summary(url)
    assert summary["available"] is True
    assert {t["name"] for t in summary["tables"]} == {t.name for t in schema.EVENT_TABLES}
    assert summary["total_rows"] == 3


def test_a_missing_sqlite_file_is_not_created(tmp_path: Path) -> None:
    """Connecting would silently create an empty database file; it must not."""
    path = tmp_path / "absent.db"
    assert overview.warehouse_summary(f"sqlite:///{path}")["available"] is False
    assert not path.exists()


def test_an_unreachable_database_degrades(monkeypatch) -> None:
    url = "postgresql+psycopg://nobody:nothing@127.0.0.1:1/none"
    assert overview.warehouse_summary(url)["available"] is False


def test_the_endpoint_always_answers(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'absent.db'}")
    body = client.get("/overview").json()
    assert set(body) == {"warehouse", "model", "drift"}
    assert all(section["available"] is False for section in body.values())
