"""Central configuration for the Subscriber Dropout Detection System.

Every tunable value (paths, seeds, split sizes, hyperparameters, decision
thresholds) lives here so that no other module has to hard-code them.  Values
can be overridden at runtime with ``SDD_``-prefixed environment variables,
which is how the Docker image and the CI pipeline redirect artifacts without
touching the code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# settings.py -> config/ -> src/ -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_params_path() -> Path:
    raw = os.getenv("SDD_PARAMS_PATH")
    return Path(raw).expanduser().resolve() if raw else PROJECT_ROOT / "params.yaml"


def _load_params(path: Path) -> dict[str, Any]:
    """Read ``params.yaml``, or return an empty mapping if it is unusable.

    Never raises. A missing or malformed parameter file must not stop the API
    from starting: every setting below carries its own default, so the worst
    case is a service running on those rather than a service that is down.
    The warning is loud because a silently ignored params.yaml would mean an
    experiment quietly training on something other than what it says.
    """
    if not path.exists():
        return {}
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text())
    except Exception as exc:  # noqa: BLE001 - see docstring
        logging.getLogger(__name__).warning("Could not read %s (%s); using defaults", path, exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


# --------------------------------------------------------------------------- #
# Parameter file
# --------------------------------------------------------------------------- #

PARAMS_PATH: Path = _env_params_path()
_PARAMS: dict[str, Any] = _load_params(PARAMS_PATH)


def _param(key: str) -> Any:
    """Look up a dotted key in ``params.yaml``, or ``None`` if absent.

    Absent is an ordinary outcome: the file may not exist (a wheel installed
    somewhere else entirely), and every caller carries its own default.
    """
    node: Any = _PARAMS
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _env_path(name: str, default: Path) -> Path:
    """Return a path from the environment, falling back to ``default``.

    Paths are deliberately not parameterised: where a run writes its files is a
    deployment fact, not an experiment parameter.
    """
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _resolve(name: str, param: str | None, default: Any) -> Any:
    """Environment variable, then ``params.yaml``, then the code default.

    The environment wins because containers and CI need to shrink models and
    redirect artifacts without editing a tracked file - and because a committed
    params.yaml should describe the pipeline's intent rather than whatever the
    last debugging session needed.
    """
    raw = os.getenv(name)
    if raw not in (None, ""):
        return raw
    if param is not None:
        value = _param(param)
        if value is not None:
            return value
    return default


def _env_int(name: str, default: int, param: str | None = None) -> int:
    return int(_resolve(name, param, default))


def _env_float(name: str, default: float, param: str | None = None) -> float:
    return float(_resolve(name, param, default))


def _env_str(name: str, default: str, param: str | None = None) -> str:
    return str(_resolve(name, param, default))


def _env_bool(name: str, default: bool, param: str | None = None) -> bool:
    value = _resolve(name, param, default)
    if isinstance(value, bool):
        return value
    return str(value) not in {"0", "false", "False"}


def _env_optional_int(name: str, param: str | None = None) -> int | None:
    """Read an int that is meaningfully absent rather than zero.

    A capacity of 0 ("we can contact nobody") is a different statement from no
    capacity constraint at all, so the unset case cannot be folded into a
    numeric default - and ``null`` in params.yaml has to survive as ``None``.
    """
    value = _resolve(name, param, None)
    return int(value) if value not in (None, "") else None


def _env_optional_float(name: str, param: str | None = None) -> float | None:
    value = _resolve(name, param, None)
    return float(value) if value not in (None, "") else None


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
DRIFT_BIN_COUNT: int = _env_int("SDD_DRIFT_BIN_COUNT", 10, "drift.bin_count")

# Population Stability Index cut-offs.  These are the long-standing convention
# in credit-risk monitoring, where PSI originated: below 0.10 a shift is noise,
# 0.10-0.25 is worth watching, and above 0.25 the population has moved enough
# that the model's training data no longer describes it.
PSI_MODERATE: float = _env_float("SDD_PSI_MODERATE", 0.10, "drift.psi_moderate")
PSI_SIGNIFICANT: float = _env_float("SDD_PSI_SIGNIFICANT", 0.25, "drift.psi_significant")

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
SIMULATION_START: str = _env_str("SDD_SIMULATION_START", "2024-01-01", "simulation.start")
SIMULATION_END: str = _env_str("SDD_SIMULATION_END", "2025-06-30", "simulation.end")

# --------------------------------------------------------------------------- #
# Point-in-time feature computation
# --------------------------------------------------------------------------- #

# How far back from a cutoff behavioural features look.
OBSERVATION_WINDOW_DAYS: int = _env_int(
    "SDD_OBSERVATION_WINDOW_DAYS", 30, "features.observation_window_days"
)

# How far forward the label looks.  A subscriber is a dropout if their
# subscription lapses within this many days *after* the cutoff, so the feature
# window and the label window never overlap.
PREDICTION_HORIZON_DAYS: int = _env_int(
    "SDD_PREDICTION_HORIZON_DAYS", 30, "features.prediction_horizon_days"
)

# --------------------------------------------------------------------------- #
# Data & splitting
# --------------------------------------------------------------------------- #

RANDOM_SEED: int = _env_int("SDD_RANDOM_SEED", 42, "split.random_seed")
N_SUBSCRIBERS: int = _env_int("SDD_N_SUBSCRIBERS", 8000, "simulation.n_subscribers")

TARGET_COLUMN: str = "dropout"
ID_COLUMN: str = "subscriber_id"

# Fractions of the *full* dataset. Train gets the remainder (0.70 by default).
TEST_SIZE: float = _env_float("SDD_TEST_SIZE", 0.15, "split.test_size")
VALIDATION_SIZE: float = _env_float("SDD_VALIDATION_SIZE", 0.15, "split.validation_size")

# --------------------------------------------------------------------------- #
# Experiment tracking & model registry (MLflow)
# --------------------------------------------------------------------------- #

# The registry needs a database-backed store, so a bare file:// path will not
# do.  SQLite keeps it serverless locally; the compose stack points this at the
# MLflow server instead.
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_EXPERIMENT: str = os.getenv("SDD_MLFLOW_EXPERIMENT", "subscriber-dropout")
REGISTERED_MODEL_NAME: str = os.getenv("SDD_REGISTERED_MODEL", "subscriber-dropout-classifier")

# Where the API loads its model from at startup.
#   "registry" - only ever load @champion from MLflow; fail loudly if absent.
#   "local"    - only ever load model.joblib from disk (the original behaviour).
#   "auto"     - try the registry first, fall back to the local artifact if the
#                registry is unreachable or has no @champion set. This is the
#                default so a fresh clone with no MLflow server still serves.
MODEL_SOURCE: str = os.getenv("SDD_MODEL_SOURCE", "auto")

# MLflow 3 removed model stages, so promotion is expressed with aliases.
CHAMPION_ALIAS: str = "champion"
CHALLENGER_ALIAS: str = "challenger"

# --------------------------------------------------------------------------- #
# Shadow scoring
# --------------------------------------------------------------------------- #

# Score every request with @challenger as well as @champion, returning only the
# champion's answer. Off by default in the image build, on wherever you want
# deployment evidence before promoting.
SHADOW_ENABLED: bool = os.getenv("SDD_SHADOW_ENABLED", "1") not in {"0", "false", "False"}

# Fraction of requests also scored by the challenger. Shadow scoring runs a
# second model inline, so it costs real latency; sampling trades evidence
# accumulation speed against that cost.
SHADOW_SAMPLE_RATE: float = _env_float("SDD_SHADOW_SAMPLE_RATE", 1.0)

# How many paired comparisons the rolling window keeps.
SHADOW_WINDOW: int = _env_int("SDD_SHADOW_WINDOW", 5_000)

# Below this many comparisons, agreement rates are noise dressed as evidence:
# 100% agreement over four requests says nothing at all.
SHADOW_MIN_COMPARISONS: int = _env_int("SDD_SHADOW_MIN_COMPARISONS", 200)

# --------------------------------------------------------------------------- #
# Decision costs & calibration
# --------------------------------------------------------------------------- #

# What each mistake costs. F1 - the current tuning objective - implicitly
# assumes these are equal, which for retention they are emphatically not.
#
# Defaults model a ~£20/month subscriber worth roughly a year of remaining
# revenue against a £20 offer: missing one churner costs as much as twelve
# wasted offers. Replace with real figures before quoting any of the money
# numbers this produces.
COST_FALSE_NEGATIVE: float = _env_float(
    "SDD_COST_FALSE_NEGATIVE", 240.0, "costs.false_negative"
)
COST_FALSE_POSITIVE: float = _env_float(
    "SDD_COST_FALSE_POSITIVE", 20.0, "costs.false_positive"
)
# An offer aimed correctly still costs money. Zero would mean "the offer is
# free", which is a claim worth making explicitly rather than by omission.
COST_TRUE_POSITIVE: float = _env_float("SDD_COST_TRUE_POSITIVE", 20.0, "costs.offer")

# How often a retention offer actually saves a subscriber who would otherwise
# have left. Setting this to 1.0 - "every correctly-aimed offer works" - makes
# blanket outreach look optimal, because it is, under that assumption. Real
# campaigns convert perhaps a fifth to a third.
OFFER_EFFICACY: float = _env_float("SDD_OFFER_EFFICACY", 0.30, "costs.offer_efficacy")

# How many retention offers the team can actually make per scoring cycle.
#
# The cost-optimal threshold assumes you can act on everyone it flags. Real
# retention teams cannot: there is a budget, a contact-frequency policy, and a
# finite number of people to run the campaign. Under a hard cap the optimal
# policy stops being "everyone above t" and becomes "the top k by score" -
# which is still a threshold, just one set by the data rather than by the
# arithmetic.
#
# Left unset by default, deliberately. Inventing a capacity number would repeat
# the mistake OFFER_EFFICACY was added to fix: a default that quietly asserts
# something about a business nobody asked. Unset means the report still
# computes the full capacity curve - what each extra slot would be worth - it
# just does not constrain the threshold.
#
#   SDD_RETENTION_CAPACITY       absolute offers per cycle (e.g. 500)
#   SDD_RETENTION_CAPACITY_RATE  fraction of the scored population (e.g. 0.05)
#
# Set at most one. The rate is the portable one: an absolute count tuned
# against a 200k subscriber base means nothing on a 1,200-row holdout.
RETENTION_CAPACITY: int | None = _env_optional_int("SDD_RETENTION_CAPACITY")
RETENTION_CAPACITY_RATE: float | None = _env_optional_float(
    "SDD_RETENTION_CAPACITY_RATE", "capacity.max_offer_rate"
)

# Isotonic is non-parametric and fixes boosting's S-shaped distortion without
# assuming its shape; it needs a few hundred rows, which validation has.
CALIBRATION_METHOD: str = _env_str(
    "SDD_CALIBRATION_METHOD", "isotonic", "costs.calibration_method"
)

# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #

# A group whose metrics differ from the best group by more than this is
# reported as a disparity. 0.8 is the long-standing "four-fifths rule" from US
# employment law - not a law of nature, and not a legal standard here, but a
# widely understood starting point that beats inventing a number.
FAIRNESS_DISPARITY_THRESHOLD: float = _env_float(
    "SDD_FAIRNESS_DISPARITY", 0.8, "fairness.disparity_threshold"
)

# Below this many rows a group's metrics are noise; reported, but flagged.
FAIRNESS_MIN_GROUP_SIZE: int = _env_int(
    "SDD_FAIRNESS_MIN_GROUP", 30, "fairness.min_group_size"
)

# --------------------------------------------------------------------------- #
# Streaming inference
# --------------------------------------------------------------------------- #

# Redpanda speaks the Kafka protocol, so any Kafka client works against it.
STREAM_BROKERS: str = os.getenv("SDD_STREAM_BROKERS", "localhost:19092")
STREAM_INPUT_TOPIC: str = os.getenv("SDD_STREAM_INPUT_TOPIC", "subscriber-events")
STREAM_OUTPUT_TOPIC: str = os.getenv("SDD_STREAM_OUTPUT_TOPIC", "subscriber-scores")

# Messages that cannot be scored land here instead of stopping the consumer.
# Without a dead-letter topic, one malformed record poisons the partition and
# the consumer crash-loops on it forever.
STREAM_DEAD_LETTER_TOPIC: str = os.getenv("SDD_STREAM_DLQ_TOPIC", "subscriber-scores-dlq")

STREAM_CONSUMER_GROUP: str = os.getenv("SDD_STREAM_GROUP", "subscriber-scorer")

# Scoring is vectorised, so batching is most of the throughput story: 200 rows
# in one predict_proba call is far cheaper than 200 separate ones.
STREAM_BATCH_SIZE: int = _env_int("SDD_STREAM_BATCH_SIZE", 200)
STREAM_POLL_TIMEOUT: float = _env_float("SDD_STREAM_POLL_TIMEOUT", 1.0)

# The scorer runs as its own process with no API, so it serves its Prometheus
# metrics itself. Different port from the API so both can run on one host.
STREAM_METRICS_PORT: int = _env_int("SDD_STREAM_METRICS_PORT", 8001)

# The gate a challenger must clear to take over.  PR-AUC rather than ROC-AUC:
# with a ~20% positive rate, average precision reflects retention-outreach
# performance far more honestly than ROC-AUC, which flatters imbalanced data.
PROMOTION_METRIC: str = _env_str("SDD_PROMOTION_METRIC", "pr_auc", "promotion.metric")

# A challenger must beat the champion by this margin, not merely tie it.
# Without a margin, noise alone would promote a new model roughly half the time
# and the registry would churn forever.
PROMOTION_MIN_IMPROVEMENT: float = _env_float(
    "SDD_PROMOTION_MIN_IMPROVEMENT", 0.005, "promotion.min_improvement"
)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

MODEL_NAME: str = "gradient_boosting_classifier"

MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": _env_int("SDD_N_ESTIMATORS", 300, "model.n_estimators"),
    "learning_rate": _env_float("SDD_LEARNING_RATE", 0.05, "model.learning_rate"),
    "max_depth": _env_int("SDD_MAX_DEPTH", 3, "model.max_depth"),
    "subsample": _env_float("SDD_SUBSAMPLE", 0.9, "model.subsample"),
    "min_samples_leaf": _env_int("SDD_MIN_SAMPLES_LEAF", 20, "model.min_samples_leaf"),
    "random_state": RANDOM_SEED,
}

# Probability at or above which a subscriber is labelled a dropout risk.
# ``train.py`` can tune this on the validation set and persist the result to
# ``metadata.json``; this value is the fallback.
DECISION_THRESHOLD: float = _env_float(
    "SDD_DECISION_THRESHOLD", 0.5, "threshold.decision_threshold"
)

# When true, training searches for the threshold maximising validation F1.
TUNE_THRESHOLD: bool = _env_bool("SDD_TUNE_THRESHOLD", True, "threshold.tune")

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
