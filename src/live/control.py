"""The API's side of the live replay: start a run, follow it, swap models, restore.

The replay itself runs in a separate process (``python -m src.live.replay run``)
so a minute of feature building and training never blocks a prediction. This
module owns that process, reads the state file it writes, and does the one
thing only the API process can do: when the replay promotes a model, load it
into the running service so the very next ``/predict`` uses it.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any

import joblib

from src.config import settings
from src.live import replay

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_process: subprocess.Popen | None = None
_loaded_version: int | None = None


class ReplayBusy(RuntimeError):
    """A run is already in progress."""


class ReplayFinished(RuntimeError):
    """Every replayed month has run; restore to start again."""


def _running() -> bool:
    return _process is not None and _process.poll() is None


def _launch() -> subprocess.Popen:
    """Start one replayed month in its own process. Tests replace this."""
    log = open(replay.LIVE_DIR / "run.log", "a")  # noqa: SIM115 - handed to the child
    return subprocess.Popen(
        [sys.executable, "-m", "src.live.replay", "run"],
        cwd=settings.PROJECT_ROOT, env=os.environ.copy(),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def _restore_original() -> None:
    from src.api import service

    service.load_model()


def _load_live(state: dict[str, Any], version: int) -> None:
    """Put a promoted replay model behind /predict, with honest metadata."""
    from src.api import service

    entry = next(v for v in state["versions"] if v["version"] == version)
    pipeline = joblib.load(replay.LIVE_DIR / f"model-v{version}.joblib")
    profile = json.loads((replay.LIVE_DIR / f"profile-v{version}.json").read_text())
    metadata = {
        **service.get_model().metadata,
        "trained_at": entry["trained_at"],
        "decision_threshold": entry["threshold"],
        "served_from": "live",
        # Text, whatever MLflow handed back: /model-info's schema says str,
        # and an int here turned every /model-info into a 500.
        "registry_version": (None if entry.get("registry_version") is None
                             else str(entry["registry_version"])),
        "live_version": version,
        "replay_month": entry["month"],
    }
    service.set_model(pipeline, entry["threshold"], metadata, profile)


def sync() -> dict[str, Any]:
    """Current replay state, with the served model brought in line with it."""
    global _loaded_version
    with _lock:
        state = replay.read_state()
        # A run that died without writing "failed" (killed, container restart)
        # must not look busy forever.
        if state.get("status") == "running" and not _running():
            state.update(status="failed",
                         error=state.get("error") or "the run stopped unexpectedly")
            for stage in state["stages"]:
                if stage["status"] == "running":
                    stage["status"] = "failed"
            replay.write_state(state)

        serving = state.get("serving")
        if serving and serving != _loaded_version:
            try:
                _load_live(state, serving)
                _loaded_version = serving
                logger.info("Live replay model v%s is now serving", serving)
            except Exception as exc:  # noqa: BLE001 - keep serving the old model
                logger.error("Could not load live model v%s: %s", serving, exc)
        elif not serving and _loaded_version is not None:
            _restore_original()
            _loaded_version = None

        state["running"] = _running()
        state["total_steps"] = len(replay.SCHEDULE)
        return state


def start() -> dict[str, Any]:
    """Run the next replayed month."""
    global _process
    with _lock:
        if _running():
            raise ReplayBusy("a replay run is already in progress")
        state = replay.read_state()
        if state["next_step"] >= len(replay.SCHEDULE):
            raise ReplayFinished("every replayed month has run; restore to start again")
        # Mark it running before the child starts, so a poll a moment later
        # never sees the previous run's "done" and thinks nothing happened.
        for stage in state["stages"]:
            stage.update(status="pending", detail="", seconds=None)
        state.update(status="running", current=replay.SCHEDULE[state["next_step"]].label,
                     error=None)
        replay.write_state(state)
        _process = _launch()
    return sync()


def restore() -> dict[str, Any]:
    """Stop any run, forget the replay, and put the tested model back."""
    global _process, _loaded_version
    with _lock:
        if _running():
            _process.terminate()
            try:
                _process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                _process.kill()
        _process = None
        replay.reset()
        _restore_original()
        _loaded_version = None
    return sync()
