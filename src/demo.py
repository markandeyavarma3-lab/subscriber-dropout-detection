"""Bring the whole stack up for a live demo, prove it is healthy, take it down.

    python -m src.demo up        # start everything, warm it up, print the URLs
    python -m src.demo check     # the pre-demo checklist, as a green/red table
    python -m src.demo down      # stop everything (keeps all data)

Why this exists
---------------

Every piece of the stack already had its own make target. What was missing is
the thing a live demo actually needs: one command that starts the pieces in the
right order, waits until each one is genuinely serving rather than merely
"Up", and then puts some traffic through it - because a freshly started
Prometheus and Grafana show "No data" everywhere, which reads as broken to
anyone who does not already know why.

``check`` is the other half. It is the checklist you would otherwise run by
hand ten minutes before presenting, and it checks what matters rather than
what is easy: that the model being served is the one the registry promoted,
that the three example subscribers still score where the dashboard expects,
and that the warehouse holds every row it should.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from src.config import settings

ROOT = settings.PROJECT_ROOT
STATE = ROOT / ".demo"

SERVICES = [
    "postgres", "subscriber-api", "prometheus", "alertmanager",
    "grafana", "redpanda", "stream-scorer",
]
MLFLOW_PORT = 5050  # 5000 belongs to macOS AirPlay Receiver

URLS = {
    "Dashboard": "http://127.0.0.1:8000/",
    "API docs": "http://127.0.0.1:8000/docs",
    "MLflow": f"http://127.0.0.1:{MLFLOW_PORT}/",
    "Grafana": "http://127.0.0.1:3000/",
    "Prometheus": "http://127.0.0.1:9090/",
    "Alertmanager": "http://127.0.0.1:9093/",
}

# The dashboard's three example subscribers and where they must land. If the
# served model changes, these move - and a demo that opens on "At-risk: 2%
# low" is worse than no demo, so `check` fails loudly instead.
PRESETS: dict[str, tuple[dict[str, Any], float, str]] = {
    "at_risk": ({
        "tenure_days": 120, "plan_type": "basic", "monthly_fee": 9.99,
        "avg_session_count_last_30d": 8.0, "last_activity_days_ago": 18,
        "support_tickets_last_90d": 1, "payment_failures_last_6m": 1,
        "discounts_used_last_6m": 2, "is_auto_renew_enabled": True,
    }, 0.6037, "high"),
    "healthy": ({
        "tenure_days": 1800, "plan_type": "premium", "monthly_fee": 149.0,
        "avg_session_count_last_30d": 22.0, "last_activity_days_ago": 2,
        "support_tickets_last_90d": 0, "payment_failures_last_6m": 0,
        "discounts_used_last_6m": 0, "is_auto_renew_enabled": False,
    }, 0.0005, "low"),
    "borderline": ({
        "tenure_days": 200, "plan_type": "basic", "monthly_fee": 180.0,
        "avg_session_count_last_30d": 4.0, "last_activity_days_ago": 1,
        "support_tickets_last_90d": 0, "payment_failures_last_6m": 0,
        "discounts_used_last_6m": 0, "is_auto_renew_enabled": True,
    }, 0.0501, "medium"),
}

# Row counts after cleaning. `check` compares Postgres against these exactly.
EXPECTED_ROWS = {
    "subscribers": 6_769_473,
    "subscription_events": 18_891_703,
    "payments": 18_891_703,
    "sessions": 38_216_556,
    "support_tickets": 0,
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _run(*cmd: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=check, capture_output=capture, text=True)


def _http(url: str, payload: Any = None, timeout: float = 5.0) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode()
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as err:
        return err.code, None
    except (urllib.error.URLError, OSError):
        return 0, None


def _wait(label: str, probe, timeout: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe():
            print(f"  ready   {label}")
            return True
        time.sleep(2)
    print(f"  TIMEOUT {label}")
    return False


def _psql(sql: str, database: str = "warehouse") -> str:
    result = _run("docker", "exec", "subscriber-warehouse", "psql", "-U", "subscriber",
                  "-d", database, "-Atc", sql, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def jittered_subscribers(n: int, seed: int = 7) -> list[dict[str, Any]]:
    """Plausible traffic around the three presets, always valid for the API.

    Deterministic, so two warm-ups put the same shape of traffic through and
    the Grafana panels look the same on the day as they did in rehearsal.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        base = dict(list(PRESETS.values())[i % len(PRESETS)][0])

        def wobble(value: float, spread: float = 0.35) -> float:
            return max(0.0, value * (1 + rng.uniform(-spread, spread)))

        base["tenure_days"] = max(1, int(wobble(base["tenure_days"])))
        base["monthly_fee"] = round(wobble(base["monthly_fee"]), 2)
        base["avg_session_count_last_30d"] = round(wobble(base["avg_session_count_last_30d"]), 1)
        base["last_activity_days_ago"] = min(
            base["tenure_days"], int(wobble(base["last_activity_days_ago"] + 1, 0.8))
        )
        base["discounts_used_last_6m"] = rng.choice([0, 0, 1, 2, 3]) if i % 3 == 0 else 0
        base["is_auto_renew_enabled"] = rng.random() < 0.55
        rows.append(base)
    return rows


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def ensure_docker() -> bool:
    if _run("docker", "info", check=False).returncode == 0:
        return True
    print("Docker is not running - starting Docker Desktop…")
    _run("open", "-a", "Docker", check=False)
    return _wait("Docker", lambda: _run("docker", "info", check=False).returncode == 0, 180)


def ensure_mlflow_database() -> None:
    """The compose MLflow's own database, for volumes created before it existed."""
    if _psql("SELECT 1 FROM pg_database WHERE datname = 'mlflow'", "postgres") != "1":
        _psql("CREATE DATABASE mlflow OWNER subscriber", "postgres")


def start_mlflow_ui() -> None:
    """Serve the real training runs and registry from the local mlflow.db."""
    STATE.mkdir(exist_ok=True)
    pid_file = STATE / "mlflow.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            return  # already running
        except (OSError, ValueError):
            pid_file.unlink()

    log = open(STATE / "mlflow.log", "w")  # noqa: SIM115 - handed to the child
    process = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui", "--host", "127.0.0.1",
         "--port", str(MLFLOW_PORT), "--backend-store-uri", f"sqlite:///{ROOT / 'mlflow.db'}"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    pid_file.write_text(str(process.pid))


def warm_up() -> None:
    """Put traffic through so no panel opens on "No data"."""
    rows = jittered_subscribers(300)
    status, _ = _http("http://127.0.0.1:8000/predict/batch", {"subscribers": rows}, timeout=60)
    print(f"  {'sent   ' if status == 200 else 'FAILED '} 300 predictions through /predict/batch")

    produced = _run(sys.executable, "-m", "src.streaming.produce", "--count", "60",
                    "--rate", "0", "--brokers", "localhost:19092", check=False)
    ok = produced.returncode == 0
    print(f"  {'sent   ' if ok else 'FAILED '} 60 events onto the Kafka input topic")


def up() -> int:
    if not ensure_docker():
        return 1

    missing = _run("docker", "image", "inspect", "subscriber-dropout-api:latest", check=False)
    if missing.returncode != 0:
        print("The API image is not built. Run `make demo-prepare` first (once, the day before).")
        return 1

    print("Starting services…")
    _run("docker", "compose", "up", "-d", "--no-build", *SERVICES, capture=False)

    print("Waiting for each one to actually serve…")
    ok = _wait("Postgres", lambda: _psql("SELECT 1") == "1")
    ensure_mlflow_database()
    start_mlflow_ui()
    ok &= _wait("API + model", lambda: (_http("http://127.0.0.1:8000/ready")[1] or {})
                .get("model_loaded") is True)
    ok &= _wait("Prometheus", lambda: _http("http://127.0.0.1:9090/-/ready")[0] == 200)
    ok &= _wait("Alertmanager", lambda: _http("http://127.0.0.1:9093/-/ready")[0] == 200)
    ok &= _wait("Grafana", lambda: _http("http://127.0.0.1:3000/api/health")[0] == 200)
    ok &= _wait("MLflow", lambda: _http(f"http://127.0.0.1:{MLFLOW_PORT}/health")[0] == 200, 90)
    # Straight after a restart Prometheus still holds the old containers' last
    # scrape, and reports them down until its next pass. Wait it out, so the
    # checklist straight after never shows a failure that is only a race.
    ok &= _wait("Prometheus scraping every target", _all_targets_up, 90)

    print("Warming up…")
    warm_up()

    print("\nOpen these:")
    for name, url in URLS.items():
        print(f"  {name:13s} {url}")
    print("  TablePlus     postgres://subscriber:subscriber@localhost:5432/warehouse")
    print("\nThen run `make demo-check`.")
    return 0 if ok else 1


def _all_targets_up() -> bool:
    _, targets = _http("http://127.0.0.1:9090/api/v1/targets")
    active = ((targets or {}).get("data") or {}).get("activeTargets", [])
    return bool(active) and all(t.get("health") == "up" for t in active)


def check() -> int:
    results: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))

    for name, url in URLS.items():
        status, _ = _http(url)
        record(f"{name} responds", status == 200, f"HTTP {status or 'no answer'}")

    status, info = _http("http://127.0.0.1:8000/model-info")
    info = info or {}
    local = json.loads((settings.ARTIFACTS_DIR / "metadata.json").read_text())
    same = info.get("trained_at") == local.get("trained_at")
    record("Serving the promoted KKBox model", same,
           f"served trained_at={info.get('trained_at')}, artifact={local.get('trained_at')}")

    # A model unpickled by a different scikit-learn than the one that trained
    # it can load cleanly and still score differently. requirements.txt pins
    # ranges, so every image rebuild is a chance for exactly that.
    trained = local.get("library_versions") or {}
    probe = _run("docker", "exec", "subscriber-dropout-api", "python", "-c",
                 "import json, sklearn, pandas, numpy; print(json.dumps({"
                 "'scikit_learn': sklearn.__version__, 'pandas': pandas.__version__, "
                 "'numpy': numpy.__version__}))", check=False)
    running = json.loads(probe.stdout) if probe.returncode == 0 else {}
    skew = {k: (v, running.get(k)) for k, v in trained.items() if running.get(k) != v}
    record("Container libraries match training", bool(running) and not skew,
           "identical" if running and not skew
           else ", ".join(f"{k} trained {a} / running {b}" for k, (a, b) in skew.items())
           or "could not inspect the container")

    for name, (payload, expected, band) in PRESETS.items():
        status, body = _http("http://127.0.0.1:8000/predict", payload)
        got = (body or {}).get("dropout_probability")
        level = (body or {}).get("risk_level")
        passed = got is not None and abs(got - expected) < 0.002 and level == band
        record(f"Preset {name}", passed, f"{got} {level} (expect {expected} {band})")

    for table, want in EXPECTED_ROWS.items():
        got = _psql(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        record(f"Postgres {table}", got == str(want), f"{got or 'missing'} rows (expect {want:,})")

    report = settings.ARTIFACTS_DIR / "last_pipeline_run.json"
    record("Drift report for Grafana", report.exists(),
           "present" if report.exists() else "run `make demo-prepare`")

    status, targets = _http("http://127.0.0.1:9090/api/v1/targets")
    active = ((targets or {}).get("data") or {}).get("activeTargets", [])
    down = [t["labels"].get("job") for t in active if t.get("health") != "up"]
    record("Prometheus scraping every target", bool(active) and not down,
           "all up" if active and not down else f"down: {', '.join(down) or 'no targets'}")

    width = max(len(name) for name, _, _ in results)
    for name, passed, detail in results:
        mark = "\033[32m PASS \033[0m" if passed else "\033[31m FAIL \033[0m"
        print(f"{mark} {name:<{width}}  {detail}")
    failed = sum(not passed for _, passed, _ in results)
    print(f"\n{len(results) - failed}/{len(results)} checks passed.")
    return 1 if failed else 0


def down() -> int:
    pid_file = STATE / "mlflow.pid"
    if pid_file.exists():
        with contextlib.suppress(OSError, ValueError):
            os.killpg(int(pid_file.read_text()), signal.SIGTERM)
        pid_file.unlink()
    # stop, never `down -v`: the volumes hold the 82.8M-row warehouse.
    _run("docker", "compose", "stop", capture=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the stack for a live demo.")
    parser.add_argument("command", choices=["up", "check", "down"])
    args = parser.parse_args(argv)
    return {"up": up, "check": check, "down": down}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
