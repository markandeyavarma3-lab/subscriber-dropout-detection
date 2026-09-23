"""Tests for the live replay behind the dashboard's "Run next month" button.

The replay claims three things, and each is tested here: it never uses data
from after its replayed "today"; it runs the project's real pipeline end to
end (features, validation, drift, training, the gate); and a promoted model
really does start serving, while "restore" really does put the tested one back.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from src.api import service
from src.config import settings
from src.live import control, replay
from src.warehouse.simulate import simulate_events

HORIZON = timedelta(days=settings.PREDICTION_HORIZON_DAYS)
TINY = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}

# Two replayed days inside the simulator's 2024 range, built the same way as
# the real schedule: each new day adds the previous test month to training.
SIM_SCHEDULE = (
    replay.ReplayMonth("1 Jul 2024", date(2024, 7, 1), date(2024, 6, 1),
                       ("2024-05-01",), "2024-06-01"),
    replay.ReplayMonth("1 Aug 2024", date(2024, 8, 1), date(2024, 7, 1),
                       ("2024-05-01", "2024-06-01"), "2024-07-01"),
)


@pytest.fixture(scope="module")
def sim_engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("live") / "warehouse.db"
    engine = create_engine(f"sqlite:///{path}")
    simulate_events(n_subscribers=900, start="2024-01-01", end="2024-09-30", seed=5, engine=engine)
    return engine


# --------------------------------------------------------------------------- #
# Honest about time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("month", replay.SCHEDULE, ids=lambda m: m.label)
def test_every_label_the_replay_uses_is_complete_by_its_today(month) -> None:
    """The whole point of a replay: nothing from after the replayed day."""
    for cutoff in (*month.train, month.test):
        label_ends = date.fromisoformat(cutoff) + HORIZON
        assert label_ends <= month.today, f"{cutoff}'s label isn't known on {month.label}"


@pytest.mark.parametrize("month", replay.SCHEDULE, ids=lambda m: m.label)
def test_the_model_is_tested_on_a_month_it_never_trained_on(month) -> None:
    assert month.test not in month.train
    assert all(date.fromisoformat(c) < date.fromisoformat(month.test) for c in month.train)


def test_each_replayed_month_learns_from_everything_before_it() -> None:
    for earlier, later in zip(replay.SCHEDULE, replay.SCHEDULE[1:], strict=False):
        assert set(earlier.train) < set(later.train)
        assert earlier.test in later.train
        assert later.today > earlier.today


def test_validation_catches_bad_training_data() -> None:
    good = pd.DataFrame({
        "tenure_days": [100, 200], "plan_type": ["basic", "premium"], "monthly_fee": [9.9, 20.0],
        "avg_session_count_last_30d": [3.0, 8.0], "last_activity_days_ago": [2, 5],
        "support_tickets_last_90d": [0, 0], "payment_failures_last_6m": [0, 0],
        "discounts_used_last_6m": [0, 1], "is_auto_renew_enabled": [True, False],
        "dropout": [0, 1],
    })
    assert replay.validate(good, good) == []

    impossible = good.assign(last_activity_days_ago=[500, 5])
    assert any("before signup" in p for p in replay.validate(impossible, good))
    one_class = good.assign(dropout=[0, 0])
    assert any("one class" in p for p in replay.validate(good, one_class))


# --------------------------------------------------------------------------- #
# The real pipeline, end to end
# --------------------------------------------------------------------------- #


def test_a_replay_runs_every_stage_and_promotes_its_first_model(tmp_path, sim_engine) -> None:
    state = replay.run_next(tmp_path, SIM_SCHEDULE, sample=900, engine=sim_engine,
                            model_params=TINY, threshold=0.5)

    assert state["status"] == "done", state.get("error")
    assert [s["status"] for s in state["stages"]] == [
        "done", "done", "done", "skipped", "done", "done", "done", "done"]
    first = state["versions"][0]
    assert first["promoted"] and first["serving"] and state["serving"] == 1
    assert (tmp_path / "model-v1.joblib").exists()

    # The second month is gated against the first: promoted or rejected, the
    # decision must be recorded with both scores, and drift must have run.
    state = replay.run_next(tmp_path, SIM_SCHEDULE, sample=900, engine=sim_engine,
                            model_params=TINY, threshold=0.5)
    assert state["status"] == "done", state.get("error")
    second = state["versions"][1]
    assert second["champion_score"] is not None
    assert next(s for s in state["stages"] if s["key"] == "drift")["status"] == "done"
    assert state["serving"] == (2 if second["promoted"] else 1)

    # And the replay stops at the end of the data instead of inventing more.
    state = replay.run_next(tmp_path, SIM_SCHEDULE, sample=900, engine=sim_engine,
                            model_params=TINY, threshold=0.5)
    assert "end of the data" in state["error"]


def test_the_feature_cache_is_reused(tmp_path, sim_engine) -> None:
    replay.snapshot("2024-05-01", tmp_path, 900, sim_engine)
    _, cached = replay.snapshot("2024-05-01", tmp_path, 900, sim_engine)
    assert cached


# --------------------------------------------------------------------------- #
# The API: promotion really swaps the served model, restore really restores
# --------------------------------------------------------------------------- #


@pytest.fixture()
def live_client(client: TestClient, tmp_path, sim_engine, monkeypatch):
    monkeypatch.setattr(replay, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(replay, "SCHEDULE", SIM_SCHEDULE)
    monkeypatch.setattr(control, "_loaded_version", None)

    class Done:  # stands in for the child process, which has already finished
        def poll(self):
            return 0

    def run_inline():
        replay.run_next(tmp_path, SIM_SCHEDULE, sample=900, engine=sim_engine,
                        model_params=TINY, threshold=0.5)
        return Done()

    monkeypatch.setattr(control, "_launch", run_inline)
    yield client
    control.restore()


def test_status_starts_idle(live_client: TestClient) -> None:
    body = live_client.get("/live/status").json()
    assert body["status"] == "idle" and body["versions"] == [] and body["total_steps"] == 2


def test_a_promoted_model_starts_serving_and_restore_puts_the_original_back(
    live_client: TestClient,
) -> None:
    original = service.get_model()

    body = live_client.post("/live/run").json()
    assert body["serving"] == 1
    info = live_client.get("/model-info").json()
    assert info["served_from"] == "live"
    assert service.get_model().pipeline is not original.pipeline

    # /predict now answers with the replay's model.
    assert live_client.post("/predict", json={
        "tenure_days": 120, "plan_type": "basic", "monthly_fee": 9.99,
        "avg_session_count_last_30d": 8.0, "last_activity_days_ago": 18,
        "support_tickets_last_90d": 1, "payment_failures_last_6m": 1,
        "discounts_used_last_6m": 2, "is_auto_renew_enabled": True,
    }).status_code == 200

    body = live_client.post("/live/reset").json()
    assert body["status"] == "idle" and body["serving"] is None
    assert live_client.get("/model-info").json()["served_from"] != "live"


def test_model_info_survives_a_registry_version_from_mlflow(
    live_client: TestClient, monkeypatch,
) -> None:
    """Found live: with MLflow connected, the replay recorded the registry
    version as an int, /model-info's schema wants text, and every /model-info
    call failed with a 500 for as long as a replayed model was serving."""
    real_run_next = replay.run_next

    def run_and_register(*args, **kwargs):
        state = real_run_next(*args, **kwargs)
        state["versions"][-1]["registry_version"] = 7  # what MLflow returns
        replay.write_state(state, args[0] if args else kwargs.get("live_dir"))
        return state

    monkeypatch.setattr(replay, "run_next", run_and_register)
    live_client.post("/live/run")
    response = live_client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["registry_version"] == "7"


def test_running_past_the_end_is_refused(live_client: TestClient) -> None:
    for _ in SIM_SCHEDULE:
        assert live_client.post("/live/run").status_code == 200
    assert live_client.post("/live/run").status_code == 409


def test_replay_runs_land_in_their_own_experiment_and_lineage(
    tmp_path, sim_engine, monkeypatch,
) -> None:
    """Found live: log_training_run re-configured MLflow with the defaults, so
    replay runs were filed under the main experiment instead of their own."""
    from mlflow import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("SDD_LIVE_MLFLOW_URI", uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "mlruns"))
    state = replay.run_next(tmp_path / "live", SIM_SCHEDULE, sample=900, engine=sim_engine,
                            model_params=TINY, threshold=0.5)
    assert state["status"] == "done", state.get("error")
    assert state["versions"][0]["registry_version"] == "1"

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(replay.LIVE_EXPERIMENT)
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id])) == 1
    champion = client.get_model_version_by_alias("subscriber-dropout-live", "champion")
    assert str(champion.version) == "1"
