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

# 4. Train the model (generates the dataset on first run)
python -m src.models.train

# 5. Serve the API
uvicorn src.api.main:app --reload
```

Then open **http://127.0.0.1:8000/** for the dashboard, or
**http://127.0.0.1:8000/docs** for interactive Swagger UI.

A `Makefile` wraps the same commands: `make install`, `make train`, `make serve`,
`make test`, `make lint`, `make docker-build`, `make docker-run`.

### Prerequisites

- Python 3.10 or newer (3.11 recommended)
- `pip`
- Docker (optional, only for the containerised workflow)

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
| `GET` | `/model-info` | Metadata: model name, training time, threshold, expected columns. |
| `POST` | `/predict` | Score one subscriber. |
| `POST` | `/predict/batch` | Score up to 1000 subscribers in one call. |
| `GET` | `/metrics` | Live prediction statistics for this process. |
| `POST` | `/monitoring/drift` | Compare a sample of live traffic against the training data. |
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

### Tests

189 tests across six files, all runnable with `pytest`:

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

The suite trains one small model per session into a temporary directory, so it runs in
under a second and never writes into the repository or depends on your working tree.

### CI pipeline

`.github/workflows/ci.yml` runs on push and pull request to `main`:

**Job 1 — lint and test:** checkout → set up Python 3.11 (with pip caching) → install →
`ruff check` → `pytest -v` → train end-to-end → evaluate the saved artifact → upload the
artifacts for download.

**Job 2 — Docker:** runs only after job 1 passes. Builds the image, starts a container,
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
