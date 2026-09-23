"""Everything the dashboard's Overview page shows, in one response.

The Overview tells the whole project's story - the data, the model, how good
it is, whether the world has drifted - and every number on it should come
from the running system rather than be typed into the HTML, where it would
quietly go stale the first time the model was retrained. So this gathers them
from where they already live:

- the warehouse's row counts, from the database (``SDD_DATABASE_URL``);
- the served model's own evaluation report (``metrics.json``) and metadata;
- the last pipeline run's drift check (``last_pipeline_run.json``).

Each section reports ``available: false`` with a reason instead of failing,
because the page must still render when, say, the API runs without a database
- which is exactly how the Docker smoke test in CI runs it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, inspect, select, text

from src.config import settings
from src.warehouse.schema import EVENT_TABLES

logger = logging.getLogger(__name__)

# Row counts change only when data is loaded, never per request. Without this,
# every page view would ask the database to count 82.8M rows.
_CACHE_SECONDS = 60.0
_warehouse_cache: tuple[float, str, dict[str, Any]] | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _warehouse(url: str) -> dict[str, Any]:
    """Row counts per table, preferring the precomputed summary table."""
    # Connecting to a SQLite path that doesn't exist silently creates an empty
    # database file there. Refuse instead of leaving litter.
    if url.startswith("sqlite:///") and not Path(url.removeprefix("sqlite:///")).exists():
        return {"available": False, "reason": "no warehouse database at the configured path"}

    connect_args = {"connect_timeout": 3} if url.startswith("postgresql") else {}
    engine = create_engine(url, connect_args=connect_args)
    try:
        with engine.connect() as connection:
            if inspect(connection).has_table("warehouse_summary"):
                rows = connection.execute(
                    text("SELECT table_name, row_count, what_it_holds FROM warehouse_summary")
                ).all()
                counts = {name: (int(count), holds) for name, count, holds in rows}
            else:
                counts = {
                    table.name: (
                        int(connection.execute(select(func.count()).select_from(table)).scalar_one()),
                        None,
                    )
                    for table in EVENT_TABLES
                }
            backend = engine.dialect.name
    except Exception as exc:  # noqa: BLE001 - an overview must never take the API down
        logger.info("Warehouse unavailable for the overview: %s", exc)
        return {"available": False, "reason": "the warehouse database is not reachable"}
    finally:
        engine.dispose()

    tables = [
        {"name": table.name, "rows": counts[table.name][0], "holds": counts[table.name][1]}
        for table in EVENT_TABLES
        if table.name in counts
    ]
    return {
        "available": True,
        "backend": backend,
        "tables": tables,
        "total_rows": sum(t["rows"] for t in tables),
    }


def warehouse_summary(url: str | None = None) -> dict[str, Any]:
    """Cached row counts for the configured warehouse."""
    global _warehouse_cache
    target = url or settings.DATABASE_URL
    now = time.monotonic()
    if _warehouse_cache and _warehouse_cache[1] == target:
        cached_at, _, cached = _warehouse_cache
        if now - cached_at < _CACHE_SECONDS:
            return cached
    result = _warehouse(target)
    _warehouse_cache = (now, target, result)
    return result


def reset_cache() -> None:
    global _warehouse_cache
    _warehouse_cache = None


def model_report(artifacts_dir: Path | None = None) -> dict[str, Any]:
    """The served model's own evaluation, as written at training time."""
    directory = artifacts_dir or settings.ARTIFACTS_DIR
    metrics = _read_json(directory / "metrics.json")
    if not metrics:
        return {"available": False, "reason": "no evaluation report next to the model"}
    metadata = _read_json(directory / "metadata.json") or {}

    def pick(section: str, *keys: str) -> dict[str, Any]:
        values = metrics.get(section) or {}
        return {key: values.get(key) for key in keys}

    quality = metrics.get("decision_quality") or {}
    costs = quality.get("costs") or {}
    fairness = quality.get("fairness") or {}
    return {
        "available": True,
        "model_name": metrics.get("model_name") or metadata.get("model_name"),
        "trained_at": metrics.get("trained_at") or metadata.get("trained_at"),
        "model_params": metadata.get("model_params") or {},
        "test": pick("test", "n_samples", "positive_rate_actual", "pr_auc", "roc_auc",
                     "precision", "recall", "accuracy", "threshold"),
        "validation": pick("validation", "n_samples", "positive_rate_actual", "pr_auc", "roc_auc"),
        "dataset": metrics.get("dataset") or {},
        "top_features": (metrics.get("top_feature_importances") or [])[:6],
        "calibration_error": (quality.get("calibration") or {}).get("expected_calibration_error"),
        "cost_optimal_threshold": (costs.get("cost_optimal") or {}).get("threshold"),
        "cost_savings": costs.get("savings"),
        "fairness": {
            "passes": fairness.get("passes"),
            "attribute": fairness.get("attribute"),
            "weakest_group": fairness.get("weakest_group"),
        },
    }


def drift_report(artifacts_dir: Path | None = None) -> dict[str, Any]:
    """The last pipeline run's drift check, if there has been one."""
    report = _read_json((artifacts_dir or settings.ARTIFACTS_DIR) / "last_pipeline_run.json")
    drift = (report or {}).get("drift") or {}
    if not drift.get("available"):
        return {"available": False, "reason": "no drift check has run for this model yet"}
    return {
        "available": True,
        "finished_at": report.get("finished_at"),
        "cutoff": drift.get("cutoff"),
        "verdict": drift.get("overall_verdict"),
        "n_samples": drift.get("n_samples"),
        "top_features": drift.get("top_features") or [],
    }


def build_overview() -> dict[str, Any]:
    return {
        "warehouse": warehouse_summary(),
        "model": model_report(),
        "drift": drift_report(),
    }
