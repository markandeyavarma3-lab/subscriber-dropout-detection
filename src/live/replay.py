"""Replay the real KKBox winter month by month, retraining as each month lands.

    python -m src.live.replay run      # run the next month (what the button does)
    python -m src.live.replay reset    # forget the replay

The dashboard's "Run next month" button starts this module as a separate
process, so training never blocks the API, and polls the state file it writes
after every stage. Each run pretends it is a specific day in early 2017 and
does exactly what the nightly pipeline would have done that day:

    new data arrives -> build features -> validate -> check drift -> train
                     -> evaluate -> gate -> go live (or keep the champion)

Everything is the project's own code - the point-in-time features, the
training pipeline, the drift detector, and the promotion gate - reading the
real warehouse in Postgres.

Honest about time
-----------------

On each replayed day the run may only use data that existed by then. Every
training label (the 30 days after a cutoff) must be complete before "today",
and the model is tested on the most recent cutoff whose label is complete -
data its training never touched. `SCHEDULE` encodes that, and a test checks it.

The replay is its own lineage: its first model is promoted because nothing
came before it, and each later month is gated against the replay's own
champion. It never compares itself with the production model, which was
trained on data from these very months and would win unfairly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)

LIVE_EXPERIMENT = "subscriber-dropout-live-replay"
LIVE_DIR = Path(os.getenv("SDD_LIVE_DIR", "/tmp/subscriber-dropout-live"))
STATE_FILE = "state.json"
SAMPLE_SIZE = int(os.getenv("SDD_LIVE_SAMPLE", "50000"))


@dataclass(frozen=True)
class ReplayMonth:
    """One replayed day: what is known by then, and what the pipeline does with it."""

    label: str                  # shown on the dashboard
    today: date                 # data before this date exists; nothing after
    arrived_from: date          # the month of data that just landed
    train: tuple[str, ...]      # cutoffs to train on
    test: str                   # cutoff to evaluate and gate on


# Sessions were loaded from 1 Oct 2016 and the export ends on 28 Feb 2017, so
# these are the only cutoffs with a full 30-day history before them and a
# full 30-day label after them.
SCHEDULE: tuple[ReplayMonth, ...] = (
    ReplayMonth("1 Jan 2017", date(2017, 1, 1), date(2016, 12, 1),
                ("2016-11-01",), "2016-12-01"),
    ReplayMonth("31 Jan 2017", date(2017, 1, 31), date(2017, 1, 1),
                ("2016-11-01", "2016-12-01"), "2016-12-31"),
    ReplayMonth("28 Feb 2017", date(2017, 3, 1), date(2017, 1, 31),
                ("2016-11-01", "2016-12-01", "2016-12-31"), "2017-01-29"),
)

STAGES: tuple[tuple[str, str], ...] = (
    ("arrive", "New data arrives"),
    ("features", "Build features"),
    ("validate", "Validate"),
    ("drift", "Check drift"),
    ("train", "Train"),
    ("evaluate", "Evaluate"),
    ("gate", "Gate"),
    ("deploy", "Go live"),
)


# --------------------------------------------------------------------------- #
# State: one JSON file, rewritten atomically after every stage
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_state(schedule: tuple[ReplayMonth, ...] | None = None) -> dict[str, Any]:
    schedule = schedule or SCHEDULE
    return {
        "status": "idle",
        "next_step": 0,
        "months": [m.label for m in schedule],
        "current": None,
        "stages": [{"key": k, "title": t, "status": "pending", "detail": "", "seconds": None}
                   for k, t in STAGES],
        "versions": [],
        "serving": None,
        "log": [],
        "error": None,
        "sample_size": SAMPLE_SIZE,
    }


# Folder and schedule are looked up at call time, not bound as default
# arguments: a default is fixed at import, so pointing LIVE_DIR elsewhere
# afterwards would be silently ignored.
def read_state(live_dir: Path | None = None,
               schedule: tuple[ReplayMonth, ...] | None = None) -> dict:
    try:
        return json.loads(((live_dir or LIVE_DIR) / STATE_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return empty_state(schedule)


def write_state(state: dict[str, Any], live_dir: Path | None = None) -> None:
    live_dir = live_dir or LIVE_DIR
    live_dir.mkdir(parents=True, exist_ok=True)
    tmp = live_dir / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str))
    tmp.replace(live_dir / STATE_FILE)  # atomic: the API never reads half a file


@dataclass
class Recorder:
    """Moves one stage at a time through pending -> running -> done/skipped/failed."""

    state: dict[str, Any]
    live_dir: Path
    started: dict[str, float] = field(default_factory=dict)

    def _stage(self, key: str) -> dict[str, Any]:
        return next(s for s in self.state["stages"] if s["key"] == key)

    def log(self, line: str) -> None:
        self.state["log"] = (self.state["log"] + [f"{datetime.now():%H:%M:%S}  {line}"])[-40:]
        write_state(self.state, self.live_dir)

    def begin(self, key: str, detail: str = "") -> None:
        stage = self._stage(key)
        stage.update(status="running", detail=detail)
        self.started[key] = time.monotonic()
        self.log(f"{stage['title']}…")

    def finish(self, key: str, detail: str, status: str = "done") -> None:
        stage = self._stage(key)
        stage.update(status=status, detail=detail,
                     seconds=round(time.monotonic() - self.started.get(key, time.monotonic()), 1))
        self.log(f"{stage['title']}: {detail}")


# --------------------------------------------------------------------------- #
# The stages
# --------------------------------------------------------------------------- #


def _engine():
    from src.warehouse import database
    return database.get_engine()


def count_arrivals(month: ReplayMonth, engine=None) -> dict[str, int]:
    """How much happened in the month that just landed, per event table."""
    from sqlalchemy import text

    engine = engine or _engine()
    counts = {}
    with engine.connect() as connection:
        for table in ("subscription_events", "payments", "sessions"):
            counts[table] = int(connection.execute(
                text(f"SELECT COUNT(*) FROM {table} "  # noqa: S608 - fixed table names
                     "WHERE occurred_at >= :a AND occurred_at < :b"),
                {"a": datetime.combine(month.arrived_from, datetime.min.time()),
                 "b": datetime.combine(month.today, datetime.min.time())},
            ).scalar_one())
    return counts


def snapshot(cutoff: str, live_dir: Path, sample: int, engine=None) -> tuple[pd.DataFrame, bool]:
    """One point-in-time snapshot, cached: the warehouse's past doesn't change."""
    from src.features.point_in_time import build_training_snapshot

    cache = live_dir / "cache" / f"{cutoff}-{sample}.pkl"
    if cache.exists():
        return pd.read_pickle(cache), True  # noqa: S301 - written by this module
    frame, _ = build_training_snapshot(cutoff, max_subscribers=sample, engine=engine or _engine())
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache)
    return frame, False


def validate(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Checks a run must pass before anything is trained on it. Returns failures."""
    from src.features.build_features import REQUIRED_INPUT_COLUMNS

    problems = []
    for name, frame in (("training", train), ("test", test)):
        if frame.empty:
            problems.append(f"the {name} set is empty")
            continue
        if frame[REQUIRED_INPUT_COLUMNS].isna().any().any():
            problems.append(f"the {name} set has missing values")
        if (frame["last_activity_days_ago"] > frame["tenure_days"]).any():
            problems.append(f"the {name} set has activity before signup")
        rate = frame[settings.TARGET_COLUMN].mean()
        if not 0.001 <= rate <= 0.5:
            problems.append(f"the {name} churn rate {rate:.2%} is implausible")
    if not test.empty and test[settings.TARGET_COLUMN].nunique() < 2:
        problems.append("the test set has only one class, so PR-AUC is undefined")
    return problems


def _load_version(live_dir: Path, version: int):
    return joblib.load(live_dir / f"model-v{version}.joblib")


def _log_to_mlflow(model, params, metrics, window, sample, promoted) -> str | None:
    """Record the run in MLflow under the replay's own model name. Best effort."""
    uri = os.getenv("SDD_LIVE_MLFLOW_URI")
    if not uri:
        return None
    os.environ.update(MLFLOW_TRACKING_URI=uri, MLFLOW_HTTP_REQUEST_MAX_RETRIES="0",
                      MLFLOW_HTTP_REQUEST_TIMEOUT="15")
    from src.registry import tracking

    name = "subscriber-dropout-live"
    _, version = tracking.log_training_run(
        model, params, {}, metrics, training_window=window,
        input_example=sample, register=True, model_name=name,
        tags={"source": "dashboard live replay"},
        tracking_uri=uri, experiment=LIVE_EXPERIMENT,
    )
    if version is not None and promoted:
        from mlflow import MlflowClient

        MlflowClient(tracking_uri=uri).set_registered_model_alias(
            name, settings.CHAMPION_ALIAS, str(version.version))
    # MLflow reports versions as integers; everything that shows them
    # (/model-info, the dashboard) treats them as text, like the registry UI.
    return str(version.version) if version is not None else None


def run_next(live_dir: Path | None = None, schedule: tuple[ReplayMonth, ...] | None = None,
             sample: int | None = None, engine=None, model_params: dict | None = None,
             threshold: float | None = None) -> dict[str, Any]:
    """Run the next replayed month end to end. Returns the final state."""
    live_dir, schedule = live_dir or LIVE_DIR, schedule or SCHEDULE
    sample = sample or SAMPLE_SIZE
    from src.features.build_features import REQUIRED_INPUT_COLUMNS
    from src.models.evaluate import compute_metrics
    from src.models.train import build_model_pipeline, top_feature_importances
    from src.monitoring.drift import detect_drift
    from src.monitoring.profile import build_reference_profile
    from src.registry.promote import evaluate_promotion

    state = read_state(live_dir, schedule)
    step = state["next_step"]
    if step >= len(schedule):
        state["error"] = ("The replay has reached the end of the data (28 Feb 2017). "
                          "Restore to start again.")
        write_state(state, live_dir)
        return state

    month = schedule[step]
    threshold = threshold if threshold is not None else _served_threshold()
    for stage in state["stages"]:
        stage.update(status="pending", detail="", seconds=None)
    state.update(status="running", current=month.label, error=None, started_at=_now(),
                 pid=os.getpid())
    rec = Recorder(state, live_dir)
    rec.log(f"It is {month.label}. Running the pipeline on everything known by then.")

    try:
        # 1. New data arrives
        last_day = month.today - timedelta(days=1)
        rec.begin("arrive", f"{month.arrived_from:%-d %b} – {last_day:%-d %b %Y}")
        counts = count_arrivals(month, engine)
        total = sum(counts.values())
        rec.finish("arrive", f"{total:,} new rows: {counts['sessions']:,} listening days, "
                             f"{counts['subscription_events']:,} subscription events")

        # 2. Features, point in time, for every cutoff this month needs
        rec.begin("features",
                  f"{len(month.train) + 1} monthly snapshots, {sample:,}-subscriber sample")
        frames, fresh = {}, 0
        for cutoff in (*month.train, month.test):
            frames[cutoff], cached = snapshot(cutoff, live_dir, sample, engine)
            fresh += not cached
            rec.log(f"  {cutoff}: {len(frames[cutoff]):,} subscribers"
                    + (" (reused)" if cached else " (built from Postgres)"))
        train = pd.concat([frames[c] for c in month.train], ignore_index=True)
        test = frames[month.test]
        rec.finish("features", f"{len(train):,} training rows, {len(test):,} test rows"
                               + (f" · {fresh} built now" if fresh else " · all reused"))

        # 3. Validate
        rec.begin("validate")
        problems = validate(train, test)
        if problems:
            rec.finish("validate", "; ".join(problems), status="failed")
            raise RuntimeError("validation failed: " + "; ".join(problems))
        churn = test[settings.TARGET_COLUMN].mean()
        rec.finish("validate", f"all checks passed · churn {churn:.2%} in the test month")

        X_train, y_train = train[REQUIRED_INPUT_COLUMNS], train[settings.TARGET_COLUMN]
        X_test, y_test = test[REQUIRED_INPUT_COLUMNS], test[settings.TARGET_COLUMN]

        champion_entry = next((v for v in reversed(state["versions"]) if v.get("serving")), None)
        champion = _load_version(live_dir, champion_entry["version"]) if champion_entry else None

        # 4. Drift: has the world moved since the serving model was fitted?
        rec.begin("drift")
        if champion_entry is None:
            rec.finish("drift", "first model: nothing to compare against yet", status="skipped")
            drift_summary = None
        else:
            profile_path = live_dir / f"profile-v{champion_entry['version']}.json"
            profile = json.loads(profile_path.read_text())
            report = detect_drift(X_test, profile)
            top = report["features"][0] if report["features"] else None
            drift_summary = {"verdict": report["overall_verdict"],
                             "top": top and {"feature": top["feature"], "psi": top["psi"]}}
            rec.finish("drift", f"{report['overall_verdict']} · largest shift "
                                f"{top['feature'].replace('_', ' ')} (PSI {top['psi']:.3f})" if top
                                else report["overall_verdict"])

        # 5. Train
        rec.begin("train", f"gradient boosting on {len(X_train):,} rows")
        model = build_model_pipeline(model_params).fit(X_train, y_train)
        rec.finish("train", f"{len(month.train)} month(s) of history, {len(X_train):,} rows")

        # 6. Evaluate on the month it never saw
        rec.begin("evaluate")
        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, proba, threshold=threshold)
        rec.finish("evaluate", f"PR-AUC {metrics['pr_auc']:.3f} · ROC-AUC {metrics['roc_auc']:.3f} "
                               f"· {int(y_test.sum())} churners in {len(y_test):,}")

        # 7. Gate - the same function the nightly pipeline uses
        rec.begin("gate")
        version = len(state["versions"]) + 1
        decision = evaluate_promotion(
            model, X_test, y_test, challenger_version=f"v{version}",
            champion=champion, champion_label=champion_entry and f"v{champion_entry['version']}",
            consult_registry=False,
        )
        if decision.champion_score is None:
            gate_text = "PROMOTED · first model, no champion yet"
        else:
            gate_text = (f"{'PROMOTED' if decision.promoted else 'REJECTED'} · "
                         f"{decision.challenger_score:.4f} vs {decision.champion_score:.4f} "
                         f"({decision.challenger_score - decision.champion_score:+.4f}, "
                         f"needs +{decision.required_improvement:.4f})")
        rec.finish("gate", gate_text, status="done" if decision.promoted else "rejected")

        # 8. Go live
        rec.begin("deploy")
        live_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, live_dir / f"model-v{version}.joblib")
        profile = build_reference_profile(X_train, model.predict_proba(X_train)[:, 1])
        (live_dir / f"profile-v{version}.json").write_text(json.dumps(profile, default=str))
        try:
            registry_version = _log_to_mlflow(
                model, {"replay_day": month.label, "train_cutoffs": ",".join(month.train),
                        "test_cutoff": month.test, "sample": sample},
                metrics, {"cutoffs": ",".join(month.train)}, X_train.head(5), decision.promoted)
        except Exception as exc:  # noqa: BLE001 - the registry is a record, not a dependency
            registry_version = None
            rec.log(f"  MLflow not reachable, run not recorded there ({type(exc).__name__})")

        entry = {
            "version": version,
            "month": month.label,
            "train_cutoffs": list(month.train),
            "test_cutoff": month.test,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "recall": metrics.get("recall"),
            "churn_rate": float(y_test.mean()),
            "champion_score": decision.champion_score,
            "promoted": decision.promoted,
            "reason": decision.reason,
            "registry_version": registry_version,
            "drift": drift_summary,
            "top_features": top_feature_importances(model, limit=5),
            "trained_at": _now(),
            "threshold": threshold,
            "serving": False,
        }
        if decision.promoted:
            for v in state["versions"]:
                v["serving"] = False
            entry["serving"] = True
            state["serving"] = version
            where = f" · MLflow v{registry_version} @champion" if registry_version else ""
            rec.finish("deploy", f"v{version} is now serving the website{where}")
        else:
            rec.finish("deploy",
                       f"v{champion_entry['version']} keeps serving; v{version} kept on record",
                       status="skipped")
        state["versions"].append(entry)
        state.update(status="done", next_step=step + 1, finished_at=_now())
        rec.log(f"Done. {'New model live.' if decision.promoted else 'The champion stays.'}")
    except Exception as exc:  # noqa: BLE001 - surfaced on the dashboard, not swallowed
        for stage in state["stages"]:
            if stage["status"] == "running":
                stage["status"] = "failed"
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rec.log(f"Failed: {exc}")
        logger.error("Live replay failed:\n%s", traceback.format_exc())
    write_state(state, live_dir)
    return state


def _served_threshold() -> float:
    """Replayed models keep the production model's threshold; only the model changes."""
    try:
        meta = json.loads((settings.ARTIFACTS_DIR / "metadata.json").read_text())
        return float(meta["decision_threshold"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return settings.DECISION_THRESHOLD


def reset(live_dir: Path | None = None) -> None:
    """Forget the replay's models and state. The feature cache is kept: it's just the past."""
    live_dir = live_dir or LIVE_DIR
    for path in live_dir.glob("model-v*.joblib"):
        path.unlink()
    for path in live_dir.glob("profile-v*.json"):
        path.unlink()
    (live_dir / STATE_FILE).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the real months through the pipeline.")
    parser.add_argument("command", choices=["run", "reset"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "reset":
        reset()
        return 0
    return 0 if run_next()["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
