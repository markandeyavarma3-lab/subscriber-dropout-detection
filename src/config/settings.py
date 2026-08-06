"""Central configuration for the Subscriber Dropout Detection System.

Every tunable value (paths, seeds, split sizes, hyperparameters, decision
thresholds) lives here so that no other module has to hard-code them.  Values
can be overridden at runtime with ``SDD_``-prefixed environment variables,
which is how the Docker image and the CI pipeline redirect artifacts without
touching the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# settings.py -> config/ -> src/ -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Return a path from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = _env_path("SDD_DATA_DIR", SRC_DIR / "data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = _env_path("SDD_ARTIFACTS_DIR", SRC_DIR / "models" / "artifacts")

RAW_DATA_PATH: Path = _env_path("SDD_RAW_DATA_PATH", RAW_DATA_DIR / "subscribers.csv")
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

MODEL_PATH: Path = _env_path("SDD_MODEL_PATH", ARTIFACTS_DIR / "model.joblib")
METADATA_PATH: Path = ARTIFACTS_DIR / "metadata.json"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"

# Static assets for the browser dashboard served at ``/``.
API_STATIC_DIR: Path = SRC_DIR / "api" / "static"
DASHBOARD_PATH: Path = API_STATIC_DIR / "index.html"

# Distribution snapshot of the training data, written at training time and used
# as the baseline that live traffic is compared against.
REFERENCE_PROFILE_PATH: Path = ARTIFACTS_DIR / "reference_profile.json"

# --------------------------------------------------------------------------- #
# Monitoring & drift detection
# --------------------------------------------------------------------------- #

# Number of quantile bins used to summarise each numeric feature.
DRIFT_BIN_COUNT: int = _env_int("SDD_DRIFT_BIN_COUNT", 10)

# Population Stability Index cut-offs.  These are the long-standing convention
# in credit-risk monitoring, where PSI originated: below 0.10 a shift is noise,
# 0.10-0.25 is worth watching, and above 0.25 the population has moved enough
# that the model's training data no longer describes it.
PSI_MODERATE: float = _env_float("SDD_PSI_MODERATE", 0.10)
PSI_SIGNIFICANT: float = _env_float("SDD_PSI_SIGNIFICANT", 0.25)

# A PSI computed on a handful of rows is dominated by sampling noise rather
# than by any real shift, so report it but mark it untrustworthy.
DRIFT_MIN_SAMPLES: int = _env_int("SDD_DRIFT_MIN_SAMPLES", 100)

# How many recent predictions ``/metrics`` keeps in memory.
METRICS_WINDOW: int = _env_int("SDD_METRICS_WINDOW", 5_000)

# --------------------------------------------------------------------------- #
# Event warehouse
# --------------------------------------------------------------------------- #

# SQLite by default so the project runs with no database server; the compose
# stack overrides this with a Postgres URL.
DATABASE_URL: str = os.getenv("SDD_DATABASE_URL", f"sqlite:///{DATA_DIR / 'warehouse.db'}")

# The window the simulator generates events across.
SIMULATION_START: str = os.getenv("SDD_SIMULATION_START", "2024-01-01")
SIMULATION_END: str = os.getenv("SDD_SIMULATION_END", "2025-06-30")

# --------------------------------------------------------------------------- #
# Point-in-time feature computation
# --------------------------------------------------------------------------- #

# How far back from a cutoff behavioural features look.
OBSERVATION_WINDOW_DAYS: int = _env_int("SDD_OBSERVATION_WINDOW_DAYS", 30)

# How far forward the label looks.  A subscriber is a dropout if their
# subscription lapses within this many days *after* the cutoff, so the feature
# window and the label window never overlap.
PREDICTION_HORIZON_DAYS: int = _env_int("SDD_PREDICTION_HORIZON_DAYS", 30)

# --------------------------------------------------------------------------- #
# Data & splitting
# --------------------------------------------------------------------------- #

RANDOM_SEED: int = _env_int("SDD_RANDOM_SEED", 42)
N_SUBSCRIBERS: int = _env_int("SDD_N_SUBSCRIBERS", 8000)

TARGET_COLUMN: str = "dropout"
ID_COLUMN: str = "subscriber_id"

# Fractions of the *full* dataset. Train gets the remainder (0.70 by default).
TEST_SIZE: float = _env_float("SDD_TEST_SIZE", 0.15)
VALIDATION_SIZE: float = _env_float("SDD_VALIDATION_SIZE", 0.15)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

MODEL_NAME: str = "gradient_boosting_classifier"

MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": _env_int("SDD_N_ESTIMATORS", 300),
    "learning_rate": _env_float("SDD_LEARNING_RATE", 0.05),
    "max_depth": _env_int("SDD_MAX_DEPTH", 3),
    "subsample": _env_float("SDD_SUBSAMPLE", 0.9),
    "min_samples_leaf": _env_int("SDD_MIN_SAMPLES_LEAF", 20),
    "random_state": RANDOM_SEED,
}

# Probability at or above which a subscriber is labelled a dropout risk.
# ``train.py`` can tune this on the validation set and persist the result to
# ``metadata.json``; this value is the fallback.
DECISION_THRESHOLD: float = _env_float("SDD_DECISION_THRESHOLD", 0.5)

# When true, training searches for the threshold maximising validation F1.
TUNE_THRESHOLD: bool = os.getenv("SDD_TUNE_THRESHOLD", "1") not in {"0", "false", "False"}

# --------------------------------------------------------------------------- #
# Rule-based explanation thresholds (used by the API, not by the model)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExplanationRules:
    """Cut-offs that turn raw feature values into human-readable risk reasons."""

    dormant_days: int = 21
    low_session_count: float = 4.0
    high_support_tickets: int = 3
    high_payment_failures: int = 2
    heavy_discount_use: int = 3
    new_subscriber_days: int = 60
    loyal_subscriber_days: int = 365
    healthy_session_count: float = 12.0


EXPLANATION_RULES: ExplanationRules = ExplanationRules()

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class APISettings:
    """Metadata and runtime options for the FastAPI service."""

    title: str = "Subscriber Dropout Detection API"
    description: str = (
        "Predicts the probability that a subscriber will drop out "
        "(cancel or stop using the service) in the near future."
    )
    version: str = "1.0.0"
    host: str = field(default_factory=lambda: os.getenv("SDD_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("SDD_API_PORT", 8000))


API_SETTINGS: APISettings = APISettings()


def ensure_directories() -> None:
    """Create every directory the pipeline writes to (idempotent)."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
