# Subscriber Dropout Detection System

[![CI](https://github.com/markandeyavarma3-lab/subscriber-dropout-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/markandeyavarma3-lab/subscriber-dropout-detection/actions/workflows/ci.yml)

An end-to-end MLOps project that predicts whether a subscriber is likely to **drop out**
(cancel or stop using the service) in the near future, and serves those predictions over a
FastAPI HTTP API. It covers the full path from data generation and feature engineering
through training, evaluation, containerisation and CI.

Built to be understood and extended by a single developer in one to two weeks.

---

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ (developed and tested on 3.11) |
| ML | scikit-learn (`GradientBoostingClassifier`), pandas, NumPy, joblib |
| API | FastAPI, Pydantic v2, uvicorn |
| UI | Dashboard in plain HTML/CSS/JS, served by the same app (no build step) |
| Testing | pytest, `fastapi.testclient` |
| Quality | ruff |
| Packaging | Docker (multi-stage), docker compose |
| Data | SQLAlchemy, Postgres (SQLite locally) |
| MLOps | MLflow tracking + model registry with gated promotion |
| Orchestration | Prefect (scheduled retraining, backfills, retries) |
| Observability | Prometheus + Grafana (provisioned dashboard, 10 alert rules) |
| Safe deployment | Champion/challenger shadow scoring on live traffic |
| Streaming | Redpanda (Kafka protocol), at-least-once with dead-lettering |
| Deployment | Kubernetes manifests (probes, HPA, PDB) |
| CI | GitHub Actions |

---

## Quick start

```bash
# 1. Clone and enter the project
git clone https://github.com/markandeyavarma3-lab/subscriber-dropout-detection.git
cd subscriber-dropout-detection

# 2. Create a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt  # or requirements.txt for runtime only

# 4. Train on the event warehouse with a temporal split, register it in
#    MLflow, and promote it to @champion. The warehouse auto-populates on
#    first use, so there is no separate "load data" step to remember.
python -m src.models.train --source warehouse --promote

# 5. Serve the API - it loads @champion straight from the MLflow registry
uvicorn src.api.main:app --reload
```

Then open **http://127.0.0.1:8000/** for the dashboard, or
**http://127.0.0.1:8000/docs** for interactive Swagger UI. Check
**http://127.0.0.1:8000/model-info** and look for `"served_from": "registry"` -
that is the proof the API is serving the model the promotion gate approved,
not a stale file on disk.

**No Docker and no external database are required for any of this.** Both the
event warehouse and the MLflow tracking store default to local SQLite files
(`SDD_DATABASE_URL`, `MLFLOW_TRACKING_URI`) - Postgres and a standalone MLflow
server only come into play if you choose to run the full `docker compose`
stack (see [Running with Docker](#running-with-docker)).

A `Makefile` wraps the same workflow:

```bash
make simulate          # populate the event warehouse (optional - training does this automatically)
make train-warehouse   # train on it with a temporal split
make train-promote     # + register in MLflow and run the gate
make serve              # uvicorn --reload
```

### The fastest path, if you just want something running

The original flat-CSV path still works, needs no warehouse or MLflow store at
all, and is what CI's primary test job uses because it has zero moving parts:

```bash
python -m src.models.train   # or: make train
uvicorn src.api.main:app --reload
```

This trains on a random split of a single generated CSV rather than a
point-in-time temporal split - see
[Temporal validation, and what it cost](#temporal-validation-and-what-it-cost)
for why that distinction matters and what it costs the reported metrics.
`/model-info` will show `"served_from": "local"` in this mode, since nothing
was registered or promoted.

### Prerequisites

- Python 3.10 or newer (3.11 recommended)
- `pip`
- Docker (optional - only for the containerised `docker compose` stack with
  Postgres and a standalone MLflow server; everything above runs without it)

---

## Usage

### Dashboard

A browser UI is served at the root of the same app. Pick one of the three
one-click presets (at-risk / healthy / borderline) or type in your own subscriber, and it
renders the probability against the model's decision threshold, the risk band, and the
signals that fired:

```
┌──────────────────────────────────────────┐
│  Subscriber Dropout Detection            │
├──────────────────────────────────────────┤
│ Tenure (days)   [ 95      ]              │
│ Plan            [ standard ▾]            │
│ Last active     [ 34      ] days ago     │
│ ☐ Auto-renew enabled                     │
│           [  Score subscriber  ]         │
├──────────────────────────────────────────┤
│   99%   HIGH RISK                        │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░          │
│        ╵ threshold 0.26                  │
│   • inactive for 34 days                 │
│   • low recent activity (2.0/30d)        │
│   • 2 payment failures in 6 months       │
└──────────────────────────────────────────┘
```

It is a single dependency-free HTML file (`src/api/static/index.html`) with inline CSS and
JavaScript — no build step, no bundler, no CDN — so it ships inside the same container and
needs nothing added to `requirements.txt`. It calls this service's own `/predict`, `/ready`
and `/model-info`, which means the UI cannot drift from the API contract without a test
failing. The header badge reflects readiness, so a container running without an artifact
shows *"no model loaded — run training"* rather than a dead page. Light and dark themes
follow the OS setting.

### Predict for one subscriber

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "tenure_days": 95,
        "plan_type": "standard",
        "monthly_fee": 19.99,
        "avg_session_count_last_30d": 2.0,
        "last_activity_days_ago": 34,
        "support_tickets_last_90d": 3,
        "payment_failures_last_6m": 2,
        "discounts_used_last_6m": 1,
        "is_auto_renew_enabled": false
      }'
```

Response:

```json
{
  "dropout_probability": 0.9888,
  "predicted_label": 1,
  "risk_level": "high",
  "threshold": 0.26,
  "explanation": "High dropout risk (99%): inactive for 34 days, low recent activity (2.0 sessions/30d), 2 payment failures in the last 6 months.",
  "top_risk_factors": [
    "inactive for 34 days",
    "low recent activity (2.0 sessions/30d)",
    "2 payment failures in the last 6 months"
  ]
}
```

A healthy subscriber (long tenure, active yesterday, auto-renew on) returns:

```json
{
  "dropout_probability": 0.0084,
  "predicted_label": 0,
  "risk_level": "low",
  "threshold": 0.26,
  "explanation": "Low dropout risk (1%): auto-renew enabled, strong recent engagement, active in the last few days, long-tenured subscriber.",
  "top_risk_factors": []
}
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Browser dashboard. |
| `GET` | `/health` | Liveness probe. Always `{"status": "ok"}` while the process is up. |
| `GET` | `/ready` | Readiness probe. Reports whether the model artifact is loaded. |
| `GET` | `/model-info` | Metadata: model name, threshold, expected columns, and `served_from` (`"registry"` or `"local"`). |
| `POST` | `/predict` | Score one subscriber. |
| `POST` | `/predict/batch` | Score up to 1000 subscribers in one call. |
| `GET` | `/metrics` | Live prediction statistics for this process. |
| `POST` | `/monitoring/drift` | Compare a sample of live traffic against the training data. |
| `GET` | `/monitoring/shadow` | Champion vs challenger comparison from live traffic. |
| `GET` | `/docs` | Swagger UI. |

`/health` and `/ready` are deliberately separate: a missing model artifact leaves the
service **live but not ready**, so an orchestrator reports a clear degraded state instead
of crash-looping the container. In that state `/predict` returns `503`, not `500`.

### Input schema

| Field | Type | Constraint |
| --- | --- | --- |
| `tenure_days` | int | `>= 0` — days since subscription start |
| `plan_type` | enum | `basic` \| `standard` \| `premium` (case-insensitive) |
| `monthly_fee` | float | `>= 0` |
| `avg_session_count_last_30d` | float | `>= 0` |
| `last_activity_days_ago` | int | `>= 0`, and **must not exceed `tenure_days`** |
| `support_tickets_last_90d` | int | `>= 0` |
| `payment_failures_last_6m` | int | `>= 0` |
| `discounts_used_last_6m` | int | `>= 0` |
| `is_auto_renew_enabled` | bool | — |

Invalid payloads return `422` with a field-level explanation. Note that `subscriber_id` is
**not** accepted: it carries no signal and is dropped before training.

---

## Running with Docker

The image trains the model during the build, so the container ships ready to serve — no
volume mount or extra step required.

```bash
docker build -t subscriber-dropout-api .
docker run -p 8000:8000 subscriber-dropout-api
```

Or with compose:

```bash
docker compose up --build
```

Either way the dashboard is then at **http://localhost:8000/**.

The runtime stage is a slim Python 3.11 base running as a non-root user, with a
`HEALTHCHECK` wired to `/health`. To serve a model trained on your host instead of the
one baked into the image, uncomment the `volumes:` block in `docker-compose.yml`.

---

## Project architecture

```
subscriber-dropout-detection/
├── README.md
├── pyproject.toml            # metadata + pytest/ruff config
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # + pytest, ruff, httpx
├── Makefile                  # shortcuts for the common commands
├── Dockerfile                # multi-stage: build & train -> slim runtime
├── docker-compose.yml
├── src/
│   ├── config/settings.py    # all paths, seeds, hyperparameters, thresholds
│   ├── data/
│   │   ├── generate.py       # synthetic dataset generator
│   │   ├── loader.py         # loading + stratified train/val/test split
│   │   ├── raw/              # subscribers.csv (generated)
│   │   └── processed/        # test.csv, the persisted held-out split
│   ├── features/
│   │   └── build_features.py # derived features + ColumnTransformer pipeline
│   ├── models/
│   │   ├── train.py          # training entrypoint
│   │   ├── evaluate.py       # metrics + standalone evaluation script
│   │   └── artifacts/        # model.joblib, metrics.json, metadata.json,
│   │                         #   reference_profile.json
│   ├── monitoring/
│   │   ├── profile.py        # builds the training distribution baseline
│   │   ├── drift.py          # PSI scoring of live traffic vs baseline
│   │   └── tracker.py        # rolling window of served predictions
│   └── api/
│       ├── main.py           # FastAPI app and routes
│       ├── schemas.py        # Pydantic request/response models
│       ├── service.py        # model loading, prediction, explanations
│       └── static/
│           └── index.html    # dashboard (inline CSS/JS, no build step)
├── tests/
└── .github/workflows/ci.yml
```

### Temporal event warehouse

> See [ROADMAP.md](ROADMAP.md) for where this is heading. Stage 1 is done.

Churn is a **time-series** problem, and the original design got that wrong: one row per
subscriber with pre-aggregated columns and a random train/test split. That silently lets a
model learn from behaviour that happened *after* the outcome it predicts — a validation
score built that way is a lie, because at serving time the future is not available.

`src/warehouse/` replaces the flat CSV with the shape production data actually arrives in:
immutable timestamped events across five tables (`subscribers`, `subscription_events`,
`sessions`, `payments`, `support_tickets`). Nothing is pre-aggregated. The same schema runs
on **SQLite locally and Postgres in the compose stack**, so the queries are exercised by
every test run rather than only in deployment.

`src/features/point_in_time.py` then builds training data with a strict separation around
a cutoff `T`:

```
[ T − 30d , T )        features  — only ever looks backwards
[ T , T + 30d ]        label     — never visible to the features
```

The windows are disjoint by construction, and the test suite **asserts it rather than
trusting the comment**: a snapshot is taken, 200 sessions and 25 tickets are then written
*after* the cutoff, and the snapshot is rebuilt. If any feature moves, the test fails. That
guard is mutation-checked — deleting the `< :cutoff` bound from the SQL makes it fail, so
it cannot pass vacuously.

Aggregation happens **in SQL**, not pandas, because pulling every session row into memory
would not survive the volumes this schema exists for. One subtlety worth noting: session
*recency* deliberately scans all history rather than the observation window, since a
subscriber dormant for 90 days has no rows in a 30-day window at all — treating that as
"no data" instead of "long absent" would erase the strongest signal in the model.

Because features are a function of the cutoff, the same call over a series of cutoffs
gives **backfills** for free, which is what makes scheduled retraining do real work instead
of refitting identical data.

```bash
make simulate                      # populate the warehouse
python -m src.warehouse.simulate --subscribers 4000 --start 2024-01-01 --end 2025-06-30
```

**Injectable drift.** `DriftScenario` changes subscriber behaviour from a chosen date
onward — engagement collapse, payment failures, a new plan tier appearing. This is the
reason to simulate rather than download: you cannot demonstrate a drift detector firing, or
a challenger overtaking a champion, on a static dataset.

### Temporal validation, and what it cost

With the warehouse in place, training can split by **time** instead of at random — fit on
earlier cutoffs, score a strictly later one, which is the situation the model actually
faces in production:

```bash
python -m src.models.train --source warehouse            # temporal split
python -m src.models.train --source warehouse --promote  # + register and gate
```

```
Temporal split over 15 cutoffs -> train: 2024-01-01 … 2024-12-26 (13 cutoffs)
                                  validation: 2025-01-25
                                  test:       2025-02-24
Split -> train=4,207 validation=481 test=499
```

Honest reporting of the result, because the headline number moved a long way:

| | ROC-AUC | PR-AUC | Recall |
| --- | --- | --- | --- |
| Old flat CSV, random split | 0.907 | 0.766 | 0.724 |
| Warehouse, temporal split | **0.709** | **0.442** | **0.653** |

**That drop is not a regression — it is the removal of a fiction.** The old generator drew
the `dropout` label from a logistic function of the very columns it then handed the model,
so 0.907 was the model rediscovering a rule we wrote. The warehouse draws cancellations
from a daily hazard over latent traits that are only partially visible in behaviour. ~0.71
is ordinary territory for a real churn model; 0.907 never was.

One measurement worth recording, since it contradicts the usual claim: switching from a
random to a temporal split changed ROC-AUC by only **−0.065** on this data. Random
splitting was *not* what inflated the old number. The temporal split is still the correct
default — it is the only scheme that stays honest once `DriftScenario` introduces a
time-varying regime — but on stationary data it buys correctness, not points.

### Experiment tracking and gated promotion

Training used to overwrite `model.joblib` in place: a deployment with no gate, where an
unlucky seed silently replaced a good model and rollback meant retraining and hoping.

`src/registry/` replaces that with MLflow. Every run records its params, metrics and the
window it trained on; every model becomes an immutable registered version. Promotion is a
separate, **gated** decision:

```
[run 1] v1  PROMOTED: no incumbent champion (pr_auc=0.3707)
[run 2] v2  REJECTED: improvement -0.0524 below required +0.0050
            | pr_auc challenger=0.3183 champion=0.3707
[run 3] v3  REJECTED: improvement +0.0000 below required +0.0050
```

Three properties make it a gate rather than a formality:

- **Comparable.** Champion and challenger are scored on the *same* held-out data in the
  same process. Comparing a fresh challenger score against a number the champion recorded
  months ago is the classic way to promote a worse model — those two numbers were never
  measured on the same population.
- **Marginal.** A challenger must win by `PROMOTION_MIN_IMPROVEMENT` (default 0.005), not
  merely tie. Run 3 above is the identical model retrained: a zero-margin gate would
  promote it, and noise alone would churn the registry on roughly half of all retrains.
- **Reversible.** Losing deletes nothing. Promotion moves the `@champion` alias, so
  `rollback()` moves it back.

The gating metric is **PR-AUC, not ROC-AUC** — at a ~20% positive rate, average precision
reflects retention-outreach performance far more honestly.

Browse it with `make mlflow-ui` at http://127.0.0.1:5000.

> **Note on serialisation.** MLflow 3 defaults to `skops`, which refuses to serialise
> custom callables — and this pipeline deliberately embeds `add_derived_features` as a
> `FunctionTransformer` so feature engineering travels with the artifact. The two collide,
> so the registry uses `cloudpickle`, at the same trust level as the `joblib` file the API
> already loads.

#### The API serves from the registry, not from a stale file

A gate only matters if the thing it gates actually changes what runs. `SDD_MODEL_SOURCE`
(default `auto`) controls where `load_model()` gets its pipeline from at startup:

| Value | Behaviour |
| --- | --- |
| `auto` *(default)* | Try `@champion` from the MLflow registry first; fall back to the local `model.joblib` if the registry is unreachable or nothing has been promoted yet. A fresh clone with no MLflow server still serves. |
| `registry` | Only ever load `@champion`; raise loudly at startup if there is none. Use this once promotion is meant to be the *only* way a new model reaches production. |
| `local` | Only ever load `model.joblib` from disk, ignoring the registry entirely — the original behaviour, as an explicit escape hatch. |

`GET /model-info` reports which one actually happened, so a promotion can be verified from
the outside rather than trusted on faith:

```json
{
  "served_from": "registry",
  "registry_version": "1",
  "decision_threshold": 0.2,
  ...
}
```

The decision threshold travels as a logged MLflow run parameter — the registry has nowhere
else to carry it, since the artifact itself is just a fitted `sklearn` pipeline. The drift
baseline (`reference_profile.json`) remains a local file either way: it is not a registry
artifact, so `/monitoring/drift` keeps working whether the model came from MLflow or disk.

### Data (legacy flat path)

The original **synthetic flat dataset** still exists and is what training currently uses,
so the repository stays self-contained and CI stays hermetic.

`src/data/generate.py` does not draw columns independently. Each subscriber gets three
latent traits that are never written to the CSV — `engagement`, `dissatisfaction`,
`price_sensitivity` — and the observable columns are drawn conditionally on those traits
and on the plan. Premium subscribers engage more; unhappy ones file more tickets;
price-sensitive ones redeem more discounts; heavy users were seen more recently. The
`dropout` label is then drawn from a logistic model over the observable columns plus a
noise term.

That noise is deliberate. Together with the Bernoulli draw it caps the achievable ROC-AUC
in a believable 0.85–0.91 band rather than a suspicious 1.0. The generator is seeded, so
`python -m src.data.generate` reproduces the same 8,000 rows every time, with a dropout
rate near **20%** — a realistic class imbalance rather than a convenient 50/50 split.

### Features

Feature engineering lives **inside** the saved pipeline. This is the most important design
decision in the project: the API posts raw columns and the artifact performs every
transformation itself, so training-time and serving-time feature code cannot drift apart.

Raw counts are hard to compare across subscribers with different tenures and plans, so the
pipeline adds normalised versions — `recency_ratio`, `engagement_recency_score`,
`fee_per_session`, `friction_score`, per-month rates, and an `is_dormant` flag. Every
denominator carries a `+ 1` guard so a brand-new or never-active subscriber cannot divide
by zero.

Those derived features carry the model. In the current run they take four of the top six
importance slots, led by `recency_ratio` at **0.52**:

| Feature | Importance |
| --- | --- |
| `recency_ratio` (derived) | 0.515 |
| `is_auto_renew_enabled` | 0.108 |
| `fee_per_session` (derived) | 0.078 |
| `last_activity_days_ago` | 0.069 |
| `friction_score` (derived) | 0.057 |
| `engagement_recency_score` (derived) | 0.049 |

The preprocessing itself is a `ColumnTransformer`: numeric columns are median-imputed then
standard-scaled, `plan_type` is one-hot encoded with `handle_unknown="ignore"` (an unseen
plan encodes to zeros instead of raising), and binary flags pass through.

### Model training

`python -m src.models.train` loads or generates the data, makes a stratified 70/15/15
split, fits a `GradientBoostingClassifier`, and — rather than assuming 0.5 — **tunes the
decision threshold on the validation split** to maximise F1. With a 20% base rate the
tuned threshold lands near 0.26, trading precision for the recall that actually matters
when the output triggers retention outreach.

Four artifacts are written to `src/models/artifacts/`:

- `model.joblib` — the fitted pipeline; the only file the API needs
- `metrics.json` — validation and test metrics plus feature importances
- `metadata.json` — decision threshold, expected input columns, library versions
- `reference_profile.json` — the training distribution baseline used for drift detection

Current results (seed 42, 8,000 subscribers):

| Metric | Validation | Test |
| --- | --- | --- |
| Accuracy | 0.848 | 0.853 |
| Precision | 0.596 | 0.609 |
| Recall | 0.741 | 0.724 |
| F1 | 0.660 | 0.662 |
| **ROC-AUC** | **0.890** | **0.907** |

Validation and test track each other closely, which is the signal that the model is not
overfitting the split.

`python -m src.models.evaluate` reloads the saved artifact and re-scores the persisted test
split, verifying that what was written to disk still generalises.

Everything tunable — paths, seeds, split sizes, hyperparameters, rule thresholds — lives in
`src/config/settings.py` and can be overridden with `SDD_`-prefixed environment variables
(`SDD_N_ESTIMATORS`, `SDD_DECISION_THRESHOLD`, `SDD_ARTIFACTS_DIR`, …). No paths are
hard-coded elsewhere. The training CLI also takes `--n-estimators`, `--max-depth`,
`--learning-rate`, `--seed`, `--artifacts-dir` and `--no-tune-threshold`.

### API serving layer

The artifact is loaded **once** at startup via a FastAPI lifespan handler and cached in a
module-level singleton, so the load cost is not paid per request. `service.py` owns model
loading, prediction and the explanation logic; `schemas.py` owns the public contract;
`main.py` is thin routing.

The `explanation` field is intentionally **rule-based**, not SHAP — it reads the input
values against thresholds in `settings.ExplanationRules` and reports which signals fired.
It explains the inputs, not the model internals, which is honest about what it is and
costs nothing at inference time. Swapping in SHAP later is a contained change to
`build_explanation`.

`risk_level` is derived from the **decision threshold**, not from fixed cut-offs: `low` is
exactly the not-flagged region (`probability < threshold`), and `high` begins halfway
between the threshold and 1.0. This enforces the invariant that a `low` band always means
`predicted_label == 0`. Fixed bands broke it — at a tuned threshold of 0.26 a subscriber
scoring 0.30 was flagged for retention outreach while the API called them *low risk*.
Because the bands hang off the threshold, retraining retunes them automatically.

The dashboard is served from this same app rather than as a separate frontend. That is a
deliberate trade: one deployable, one port, no CORS configuration and no second build
pipeline, at the cost of the UI not being independently scalable — the right call at this
size.

### Monitoring & drift detection

A model silently decays when the world stops looking like its training data. This project
detects that in two directions.

**The baseline.** Every training run writes `reference_profile.json` describing the
population the model actually learned from: quantile bins for each numeric column,
category shares for `plan_type` and `is_auto_renew_enabled`, and the model's own score
distribution. Distributions are stored as **bins, not rows** — the artifact stays small
and fixed-size instead of shipping a copy of the training data next to the model. The
baseline is built from the *training* split specifically, not the full dataset, so it
describes what the model saw rather than what merely existed.

**The metric.** Drift is scored with the Population Stability Index:

```
PSI = Σ (live_i − ref_i) · ln(live_i / ref_i)
```

PSI is used in preference to a KS test because it needs no significance level to
interpret — its conventional bands are directly actionable, which matters when someone is
reading the number at 3am:

| PSI | Verdict | Meaning |
| --- | --- | --- |
| `< 0.10` | `stable` | Ordinary sampling noise |
| `0.10 – 0.25` | `moderate` | Worth watching |
| `> 0.25` | `significant` | The population has genuinely moved |

Bins are **quantile-based**, so each holds a comparable share of the reference; equal-width
bins on a skewed column leave most bins near-empty and make PSI explode on meaningless
movements. Every proportion carries an epsilon before the logarithm, so a category missing
from either side cannot send the score to infinity. Live values beyond the training range
are clipped into the outer bins rather than discarded — a value larger than anything seen
in training is still evidence.

**`POST /monitoring/drift`** scores a sample and attributes the movement:

```json
{
  "overall_verdict": "significant",
  "n_samples": 400,
  "sufficient_sample": true,
  "drifted_features": ["last_activity_days_ago", "dropout_probability"],
  "features": [
    {"feature": "last_activity_days_ago", "psi": 10.6784, "verdict": "significant"},
    {"feature": "avg_session_count_last_30d", "psi": 0.02, "verdict": "stable"}
  ],
  "prediction": {"feature": "dropout_probability", "psi": 3.1911, "verdict": "significant"}
}
```

Small batches produce noisy PSI, so the response reports `sufficient_sample` rather than
leaving the caller to discover that the hard way.

Input drift and output drift are scored **separately**, because they fail differently. A
population can shift while scores hold steady — an unseen `plan_type` registers PSI 5.4 on
the input while prediction drift stays flat, because `handle_unknown="ignore"` quietly
absorbs it into an all-zero block. That silent absorption is precisely the failure input
monitoring exists to catch, and no amount of watching the output would reveal it.

**`GET /metrics`** reports what this process has served — rolling mean, p50/p90, flagged
rate, risk-band counts, and `probability_mean_shift`, the live mean score minus the
training-time mean. That last number is the most alertable value here: it moves when the
model's behaviour changes, whether or not anyone deployed anything.

The tracker is a bounded in-process window, so memory stays flat under load while
`served_total` stays honest. It is deliberately **not** a metrics backend — it answers
"what is this replica doing right now". `/metrics` always returns 200, even with no model
loaded: a monitoring endpoint that fails when the thing it monitors is unhealthy is worse
than useless.

Nothing here retrains or blocks automatically. Drift is reported; acting on it is a
judgement call about the business, not about statistics.

### Scheduled retraining pipeline

`src/orchestration/` turns the pieces above into one job that can run unattended:

```
ingest  ──▶  drift  ──▶  train + gate  ──▶  report
```

```bash
make pipeline         # run it once, now
make pipeline-drift   # inject a behavioural shift first, to watch drift fire
make backfill         # replay training across historical cutoffs
make schedule         # serve it on a nightly cron (blocks)
```

**The ordering is load-bearing.** Drift is checked *after* ingest (it needs the fresh data)
but *before* training — because training overwrites `reference_profile.json`. Checking it
afterwards would compare a baseline against the very data it was just built from, report
`stable` every single time, and detect nothing, ever.

**Ingest is idempotent by default.** A scheduled job that regenerates the world on every
run would retrain on different data each night while claiming to be reproducible. Only
`--force-ingest`, or injecting a drift scenario, rewrites the warehouse.

The whole loop, run three times against the same warehouse:

```
run 1   ingest: regenerated=True  · drift: skipped, no baseline yet
        promotion ACCEPTED - no incumbent champion

run 2   ingest: regenerated=False ← idempotent
        promotion REJECTED - improvement +0.0000 below required +0.0050

run 3   (--drift injected)
        drift: significant - avg_session_count_last_30d (PSI 4.46),
                             last_activity_days_ago (PSI 0.58)
        promotion ACCEPTED - beat champion by +0.0240
```

Run 3 is the whole thesis of this project in four lines: behaviour changed, the detector
named *exactly* the two features that moved and left `tenure_days` alone, retraining on the
new reality produced a genuinely better model, and the gate let it through on evidence.

Each run writes `last_pipeline_run.json` with a **`needs_attention`** flag, set when a
challenger is rejected or drift is significant. Neither *fails* the run — a rejected
challenger means the gate did its job, and drift means the world moved. Failing on either
would train whoever is on call to ignore the alert.

**Prefect is kept at arm's length.** The pipeline itself is plain functions in
`pipeline.py` with no Prefect import; `flows.py` wraps each in a `@task` for retries and
run history. Retries sit on ingest and training, where failures are plausibly transient —
deliberately *not* on the drift check, where a failure is a data problem and retrying it
three times just produces the same answer more slowly.

No Prefect server is required: Prefect 3 starts a temporary API automatically. Point
`PREFECT_API_URL` at a real one for the hosted UI.

### Durable observability

`GET /metrics` is a bounded in-process window that forgets everything on restart. That is
fine for a health check and useless for the question that matters after a deploy: *has this
model's behaviour changed over the last three weeks?*

`GET /metrics/prometheus` answers it, exposing two families from one scrape target:

| Metric | What it tells you |
| --- | --- |
| `subscriber_predictions_total{risk_level,predicted_label}` | Traffic by risk band |
| `subscriber_prediction_probability` (histogram) | Score distribution — p50/p90/p99 |
| `subscriber_flagged_rate` | Share above the threshold (training base rate ≈ 20%) |
| `subscriber_probability_mean_shift` | Live mean minus training mean — the single most alertable number |
| `subscriber_model_served_from{source}` | Whether the registry or a local file is serving |
| `subscriber_drift_verdict` | 0 stable · 1 moderate · 2 significant |
| `subscriber_drift_psi{feature}` | Which inputs actually moved |
| `subscriber_pipeline_last_run_timestamp_seconds` | Catches a silently-dead scheduler |

**Batch metrics without a Pushgateway.** The scheduled pipeline has no HTTP endpoint to
scrape. The usual answer is a Pushgateway — a whole extra service, and one that keeps
serving stale values after a job stops existing. Instead the API reads
`last_pipeline_run.json` at scrape time, so the numbers are exactly as fresh as the last
run and there is one fewer moving part.

**Why not `/metrics`?** Prometheus conventionally owns that path, but it already served a
documented JSON contract here with a published schema. Silently changing its content type
would break existing consumers to satisfy a default that Prometheus lets you override in
one line (`metrics_path: /metrics/prometheus`).

Seven alert rules live in [`deploy/prometheus/alerts.yml`](deploy/prometheus/alerts.yml).
The severities encode a judgement: `ModelNotLoaded` is **critical**, significant drift is
**warning**, and a rejected challenger is only **info** — the gate refusing a worse model
is the system working correctly, and paging on it teaches whoever is on call to ignore the
pager.

```bash
docker compose up -d prometheus grafana   # or: make observability-up
```

Grafana comes up already wired — datasource and dashboard are provisioned from
`deploy/grafana`, so there is nothing to click before anything is visible:

- **Grafana** http://localhost:3000 (anonymous viewer enabled, local demo only)
- **Prometheus** http://localhost:9090

Two test classes guard the config, not just the code: every alert expression and every
dashboard panel query is cross-checked against the metrics the code actually exposes. An
alert on a renamed metric never fires and never complains, which is the worst failure mode
monitoring has.

### Shadow scoring — the challenger on live traffic

Every request is scored twice. `@champion` answers the caller; `@challenger` scores the
same input and its answer is **recorded and thrown away**. The challenger sees exactly the
traffic production sees, and no caller is ever affected by it.

```bash
curl localhost:8000/monitoring/shadow
```

A real run against two genuinely different models, over 300 live requests:

```
compared_total            300      errors_total              0
agreement_rate            0.7833   sufficient_evidence       true
champion_flagged_rate     0.3167   challenger_flagged_rate   0.4067
flagged_rate_delta        0.09     challenger_flags_more     46
mean_abs_divergence       0.0553   challenger_flags_fewer    19
```

That is a finding the offline gate cannot produce. PR-AUC says the challenger is more
accurate; shadow says promoting it **grows the retention outreach list by 28% relative** —
an operational cost somebody should agree to in advance rather than discover from a
surprised marketing team.

#### What shadow scoring does *not* prove

It is tempting to read this as "which model is better". It is not, and that is the most
common mistake made with shadow deployment. **Live traffic has no labels** — nobody has
churned yet — so there is no ground truth to score either model against. Accuracy still
comes from the labelled holdout in
[the promotion gate](#experiment-tracking-and-gated-promotion).

What shadow answers instead:

- **Does the challenger survive production input?** Real traffic contains shapes no
  fixture has. A model raising on 1% of requests would have taken the service down had it
  been promoted — and a clean offline holdout would never have revealed it.
- **What is the blast radius?** 97% agreement means promotion changes almost nothing. 78%
  means a fifth of all decisions flip.
- **What does it cost?** The flagged-rate delta is the size of the list someone has to act
  on, which no accuracy metric measures.

#### The one hard rule

A shadow model must never affect a response. The whole scoring block is wrapped in a bare
`except`: a challenger that raises, hangs on odd input, or returns the wrong shape costs a
counted error and nothing else. Verified directly — with a challenger that raises on
*every* call, `/predict` still returns `200` with a valid probability, and
`errors_total` increments so the failure is visible rather than silent. A rising
`subscriber_shadow_errors_total` is exactly what should block a promotion.

Shadow runs **inline**, so it adds a second forward pass to request latency. That cost is
measured (`subscriber_shadow_latency_seconds`) rather than hidden, and
`SDD_SHADOW_SAMPLE_RATE` trades evidence-gathering speed against it. A background thread
would hide the latency but add a queue that silently falls behind under load — a worse
failure to debug than an honest one.

Shadow stays **idle** when `@challenger` is unset or points at the same version as
`@champion` — the normal state right after a promotion, where shadowing would compare a
model to itself.

### Streaming inference

The warehouse has been event-shaped since Stage 1, so scoring a stream is a natural fit
rather than a bolt-on. `src/streaming/` consumes subscriber events, scores them in batches,
and writes risk scores back to an output topic:

```
subscriber-events  ──▶  score  ──▶  subscriber-scores
                          │
                          └──────▶  subscriber-scores-dlq
```

```bash
docker compose up -d redpanda stream-scorer   # or: make stream-up
```

A real run over 52 messages, two of them deliberately poisoned:

```
polls=10 messages=52 scored=50 dead_lettered=2 retries=0 commits=3 produce_failures=0

  offset 50: message at offset 50 is not valid JSON: Expecting property name…
  offset 51: missing required fields: monthly_fee, avg_session_count_last_30d, …
```

#### A stream is not a request/response API

That difference drives every design choice here. A malformed HTTP body gets a `422` and the
caller's problem stays the caller's. A malformed *message* sits in the topic forever — a
consumer that dies on it dies on it again on every restart. That is the poison-pill loop
that takes a pipeline down for a day over one bad record.

So nothing raises on bad data. Every message ends in exactly one of three places:

- **Scored** — onto the output topic, keyed by subscriber so one person's scores stay in
  order on a single partition.
- **Dead-lettered** — with a reason *and the original payload*, because a dead-letter topic
  you cannot replay from is just an expensive log line.
- **Left uncommitted** — when the *model* is unavailable. Those messages are fine;
  dead-lettering good data because a model was briefly missing would be destroying it.

Valid and invalid messages are separated *before* scoring, so one bad record cannot cost
the whole batch — the good ones still go through in a single vectorised call.

#### At-least-once, deliberately

Offsets are committed **only after** scored records are flushed to the output topic. Crash
in between and those messages are re-read and re-scored — a duplicate, which is harmless
because scoring is deterministic and output is keyed by subscriber.

Committing first would be at-most-once and would silently lose predictions on any crash.
For a churn model that means a subscriber is quietly never scored, never contacted, and
nobody ever finds out. **Duplicates are cheap; silence is not.** A test asserts that a
failed produce does not commit.

#### What is tested, and what is not

The transport is a protocol, and the tests drive an in-memory broker through the real loop
— including malformed payloads, produce failures, and a missing model. `kafka.py` does
nothing but translate between `kafka-python` and that protocol, and it is **the one module
in this project with no test coverage**: there has never been a broker here to run it
against. Keeping it to pure translation is what makes that gap small and visible instead of
hidden.

`kafka-python` is imported lazily, so the project installs and runs without it.

### Kubernetes

`deploy/kubernetes/` carries manifests for both workloads. Two decisions worth reading the
files for:

**The probe split is why `/health` and `/ready` were built separately.** A pod with no model
is *live but not ready*: it leaves the Service's endpoints without being restarted into the
same state forever. A `startupProbe` gives model loading room so a slow boot is not
mistaken for a hang.

**Only the API is autoscaled.** A Kafka consumer group cannot usefully have more consumers
than partitions, so an HPA on the stream scorer would just schedule idle pods — scaling it
means repartitioning the topic first. The manifests say so rather than leaving it to be
discovered.

> **Never applied.** No cluster and no Docker on the development machine, so these are
> structurally validated by tests and nothing more. See [ROADMAP.md](ROADMAP.md).

### Tests

292 tests across eleven files, all runnable with `pytest`:

- `test_features.py` — derived-column presence, row-count preservation, input immutability,
  finiteness, zero-denominator edge cases, hand-computed formula checks, output shape,
  scaling, and the unseen-category path
- `test_train.py` — dataset schema, reproducibility, split stratification, artifact
  creation, metric validity, better-than-chance performance, joblib round-trip, and that
  a tuned threshold reproduces its own reported F1
- `test_api.py` — every endpoint, validation failures (`422`), the degraded no-model path
  (`503`), batch/single agreement, and a behavioural check that a distressed subscriber
  scores strictly higher than a healthy one
- the dashboard tests in `test_api.py` parse the page itself: its form fields must match
  `SubscriberFeaturesRequest` exactly, every path it `fetch`es must be a real route, and
  each preset must round-trip through `/predict` — so the UI cannot drift from the API
- `test_monitoring.py` — PSI algebra (zero for identical inputs, symmetric, finite when a
  bin empties), quantile binning edge cases (constant and empty columns), the
  false-positive check that reference data shows no drift against itself, detection and
  attribution of a real shift, unseen categories, tracker bounds, and both endpoints
- `test_orchestration.py` — ingest idempotency (the property a scheduled job lives or dies
  on), the empty-early-cutoff regression guard, run-report escalation rules, and one real
  Prefect flow run marked `slow` so `pytest -m "not slow"` stays fast
- `test_prometheus.py` — exposition format, counter/histogram wiring through the real
  endpoint, pipeline gauges loaded from a run report, and the dead-alert guards that
  cross-check every alert rule and dashboard panel against the exposed metric names
- `test_shadow.py` — the safety contract above all: a challenger that raises on every
  request must not change a single served response, and the served answer must always be
  the champion's. Plus agreement/divergence arithmetic, evidence-sufficiency gating, and
  a check that the response contains no accuracy verdict it has no basis to make
- `test_streaming.py` — mostly the bad paths, because that is where streaming differs from
  HTTP: poison messages, partial batch failures, a produce failure that must not commit,
  and a missing model that must leave good data uncommitted rather than dead-letter it.
  Plus structural checks on the Kubernetes manifests and compose wiring

The suite trains one small model per session into a temporary directory, so it runs in
under a second and never writes into the repository or depends on your working tree.

### CI pipeline

`.github/workflows/ci.yml` runs on push and pull request to `main`:

**Job 1 — lint and test:** checkout → set up Python 3.11 (with pip caching) → install →
`ruff check` → `pytest -v` → train end-to-end (flat CSV) → evaluate the saved artifact →
upload the artifacts for download. Deliberately the simplest possible path — zero external
services — so it fails fast and fails clearly.

**Job 2 — warehouse:** runs only after job 1 passes. Simulates the event warehouse from
scratch, trains on a temporal split, registers and promotes the result in MLflow, then
starts the API and asserts `/model-info` reports `"served_from":"registry"` before posting
a real `/predict` request. This needs no Postgres and no Docker — both the warehouse and
MLflow default to local SQLite files — so it runs on a bare GitHub-hosted runner. This job
is the one that actually proves promotion changes what gets served, not just that the gate
logic passes in isolation.

**Job 3 — Docker:** runs only after job 1 passes. Builds the image, starts a container,
polls `/health`, and posts a real request to `/predict`. Building an image proves it
compiles; the smoke test proves it *serves*. Any failing step fails the workflow.

---

## Future work

- **Real production data.** Replace the synthetic generator with a warehouse extract; the
  loader and feature pipeline are already isolated behind `src/data/loader.py`, so only
  that boundary changes.
- **Durable, cross-replica metrics.** `/metrics` is a bounded in-process window that resets
  with the process. Export to Prometheus (or write scores to a warehouse) so drift can be
  tracked across replicas and over weeks rather than since the last deploy.
- **Scheduled drift checks.** `/monitoring/drift` is pull-based today. Run it nightly over
  the previous day's traffic and alert on a `significant` verdict, instead of relying on
  someone remembering to ask.
- **Experiment tracking.** MLflow or Weights & Biases in place of the hand-rolled
  `metrics.json`, once more than a handful of runs need comparing.
- **Event-driven scoring.** Consume subscriber activity events from Kafka or SQS and write
  risk scores back to the CRM, instead of synchronous request/response only.
- **True explainability.** Swap the rule-based explanation for SHAP values to attribute
  each prediction to the model's actual decision surface.
- **Threshold by business cost.** Tune the cut-off against the real cost ratio of a missed
  churner versus a wasted retention offer, rather than F1.
- **Calibration.** Add `CalibratedClassifierCV` so the probabilities can be used directly
  in expected-value calculations.

---

## License

MIT.
