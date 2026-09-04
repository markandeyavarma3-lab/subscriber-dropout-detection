# Subscriber Dropout Detection — Project Status

**A pin-to-pin record of what this project is, what's built, what's verified, and what's
left.** Written to be read by a human or an AI agent picking up this project cold, with no
prior context. Every number here was re-verified against the working tree before writing.

**Last verified:** 2026-08-25 · **Local HEAD:** `82c5c5d` · **Remote HEAD:** `f262bfa`
**Tests:** 203 passing · **Repo:** https://github.com/markandeyavarma3-lab/subscriber-dropout-detection

---

## 1. What this project is

A machine learning system that predicts whether a subscriber is about to cancel
("dropout"/"churn"), serves that prediction over an HTTP API and a browser dashboard, and
is partway through being rebuilt into something that can be *operated* in production —
not just trained once and forgotten.

It started as a scoped 1–2 week portfolio build (fully complete). It is now being extended
into a six-stage MLOps roadmap, of which **two stages are done**.

### The prediction problem

- **Type:** binary classification. Target column: `dropout` (1 = will cancel soon, 0 = will stay).
- **Class balance:** ~20% positive. This single fact drives several design choices below
  (threshold tuning instead of 0.5, PR-AUC instead of ROC-AUC for gating, etc.)
- **Nine input features**, all raw/observable — no derived or leaked columns accepted by
  the API:

  | Field | Type | Constraint |
  |---|---|---|
  | `tenure_days` | int | ≥ 0 |
  | `plan_type` | enum | basic / standard / premium (case-insensitive) |
  | `monthly_fee` | float | ≥ 0 |
  | `avg_session_count_last_30d` | float | ≥ 0 |
  | `last_activity_days_ago` | int | ≥ 0, must not exceed `tenure_days` |
  | `support_tickets_last_90d` | int | ≥ 0 |
  | `payment_failures_last_6m` | int | ≥ 0 |
  | `discounts_used_last_6m` | int | ≥ 0 |
  | `is_auto_renew_enabled` | bool | — |

  `subscriber_id` is deliberately **rejected** by the API — it carries no signal and
  accepting it would invite training on an identifier.

---

## 2. Current state of the git repository

```
82c5c5d  Stage 2: MLflow tracking, model registry and gated promotion   ← local HEAD
388ec83  Stage 1: temporal event warehouse and point-in-time features
0b82ce1  Add monitoring and drift detection
f262bfa  Ignore .env in git and Docker builds                          ← remote HEAD (GitHub)
aad4a9d  Add CI badge and the real clone URL
381ac92  Anchor risk bands to the decision threshold
96ef8ce  Make the borderline preset actually borderline
cbab7d2  Add browser dashboard served by the API
0e047ba  Subscriber Dropout Detection System: end-to-end ML pipeline and API
```

**⚠️ Three commits are local-only and have never been pushed to GitHub:**
`0b82ce1` (monitoring), `388ec83` (Stage 1 warehouse), `82c5c5d` (Stage 2 registry).

The remote is three commits behind. A plain `git push` from the project directory will
push them — no force flag is needed this time (force-with-lease was only required once,
early on, to displace GitHub's auto-generated stub README).

### ⚠️ Outstanding security item

`/Users/satya_03/subscriber-dropout-detection/.env` contains a **live GitHub Personal
Access Token** in plaintext:

```
GITHUB_USER=markandeyavarma3-lab
GITHUB_TOKEN=ghp_************************************   (redacted — see below)
```

This file **is correctly gitignored** and has never been committed to git history —
verified with `git check-ignore` and `git log --all -p -- .env` (zero matches).

**A real near-miss happened while writing an earlier draft of this document.** The token
was quoted verbatim in this file, purely as an example of "here is the problem" — and that
draft was committed locally. GitHub's **push protection** caught it and rejected the push
outright before it ever reached the remote:

```
remote: —— GitHub Personal Access Token ——————————————————————
remote:   (?) To push, remove secret from commit(s) or follow this URL to allow the secret.
To https://github.com/.../subscriber-dropout-detection.git
 ! [remote rejected] main -> main (push declined due to repository rule violations)
```

The offending commit was amended (safe here — it had never been pushed, so nothing public
was rewritten) to redact the value before it reached GitHub. The token itself was **never
exposed on the remote**, but this is exactly why secret values should be redacted even when
the whole point of the sentence is "this secret exists" — the value adds nothing a redacted
placeholder doesn't already convey, and it only takes one slip to leak it for real.

**Action still needed from you:** delete the file (`rm .env`) and revoke the token at
https://github.com/settings/tokens — it has now been visible in a chat transcript, and a
credential with that history should be treated as burned regardless of where else it did or
didn't end up. macOS Keychain already holds the credential from a prior successful push, so
`git push` keeps working without this file once you generate a replacement.

---

## 3. What's DONE — in detail

### 3.1 Original scope (commit `0e047ba`) — ✅ complete, pushed, CI green

Everything originally requested: a full ML project with training, a FastAPI inference
service, Docker packaging, and GitHub Actions CI.

**Repository structure:**
```
subscriber-dropout-detection/
├── README.md                 ~600 lines — full design rationale, usage, architecture
├── ROADMAP.md                 the six-stage plan (see §5 below)
├── PROJECT_STATUS.md          this file
├── Makefile                   18 targets (see §7)
├── Dockerfile                 multi-stage: build+train → slim runtime
├── docker-compose.yml         postgres · mlflow · subscriber-api
├── requirements.txt           10 runtime dependencies
├── requirements-dev.txt       + pytest, ruff, httpx
├── pyproject.toml             package metadata, pytest/ruff config
├── .github/workflows/ci.yml   2-job CI pipeline
└── src/
    ├── config/settings.py         (215 lines) every tunable value, 32 env var overrides
    ├── data/                      legacy flat-CSV generator/loader (still the default path)
    │   ├── generate.py            (198 lines) synthetic dataset generator
    │   └── loader.py              (145 lines) CSV loading + random stratified split
    ├── warehouse/                 ← Stage 1 (see §3.3)
    │   ├── schema.py               (98 lines)
    │   ├── database.py            (103 lines)
    │   └── simulate.py            (380 lines)
    ├── features/
    │   ├── build_features.py      (180 lines) derived features + ColumnTransformer
    │   └── point_in_time.py       (397 lines) ← Stage 1 cutoff-correct SQL
    ├── models/
    │   ├── train.py                (468 lines) training entrypoint
    │   ├── evaluate.py             (195 lines) metric computation
    │   └── artifacts/                   model.joblib + 3 JSON files (generated, gitignored)
    ├── registry/                  ← Stage 2 (see §3.4)
    │   ├── tracking.py            (203 lines)
    │   └── promote.py             (209 lines)
    ├── monitoring/                ← added after original scope (see §3.2)
    │   ├── profile.py             (166 lines)
    │   ├── drift.py               (193 lines)
    │   └── tracker.py             (115 lines)
    └── api/
        ├── main.py                 (138 lines) FastAPI routes
        ├── schemas.py               (182 lines) Pydantic request/response contract
        ├── service.py               (271 lines) model loading, prediction, explanations
        └── static/index.html              browser dashboard, no build step
└── tests/                     14 files, 413 tests total (see §3.6)
```

**The model pipeline** (`src/models/train.py` + `src/features/build_features.py`):

```
raw columns
  → FunctionTransformer(add_derived_features)     ratios, rates, flags
  → ColumnTransformer
       numeric      : median impute → standard scale
       categorical  : most-frequent impute → one-hot (handle_unknown="ignore")
       binary flags : passthrough
  → GradientBoostingClassifier
       n_estimators=300  learning_rate=0.05  max_depth=3
       subsample=0.9     min_samples_leaf=20  random_state=42
```

Nine derived features are computed **inside** the pipeline (not as a separate
preprocessing step) so the API can post raw columns and the saved artifact performs every
transformation itself — training-time and serving-time feature code cannot drift apart.
Derived features: `recency_ratio`, `engagement_recency_score`, `fee_per_session`,
`friction_score`, `tenure_months`, `sessions_per_day_last_30d`,
`support_tickets_per_month`, `payment_failures_per_month`, `discount_dependency`,
`is_dormant`. Every denominator carries a `+1` guard against division by zero.

The **decision threshold is tuned**, not assumed at 0.5 — `tune_decision_threshold()`
searches for the F1-maximizing cutoff on the validation split. At the current ~20% base
rate it lands near **0.26**.

**Current metrics** (flat-CSV path, which is still the default `make train` uses):

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Validation | 0.848 | 0.596 | 0.741 | 0.660 | 0.890 | 0.709 |
| Test | 0.853 | 0.609 | 0.724 | 0.662 | 0.907 | 0.766 |

**⚠️ Important caveat on these numbers — see §6.1.** The 0.907 ROC-AUC is real for this
data path, but this data path has a fundamental honesty problem explained in detail below.
Do not quote 0.907 as "the model's real-world performance" without that context.

**FastAPI service** — 9 routes:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Browser dashboard |
| GET | `/health` | Liveness — always `{"status":"ok"}` while the process is up |
| GET | `/ready` | Readiness — reports whether the model artifact is loaded |
| GET | `/model-info` | Model name, training time, threshold, expected columns |
| POST | `/predict` | Score one subscriber |
| POST | `/predict/batch` | Score up to 1,000 subscribers in one call |
| GET | `/metrics` | Live prediction statistics for this process (added post-scope) |
| POST | `/monitoring/drift` | Compare live traffic to training distribution (added post-scope) |
| GET | `/docs` | Swagger UI |

`/health` and `/ready` are deliberately separate: a missing model artifact leaves the
service live-but-not-ready, so an orchestrator sees a clean degraded state instead of
crash-looping the container. In that state `/predict` returns `503`, not `500`.

The artifact loads **once** at startup via a FastAPI lifespan handler, cached in a
module-level singleton — load cost is not paid per request.

**The `explanation` field is SHAP-attributed**, with the rule engine kept as a fallback.
TreeSHAP is exact for this ensemble; contributions are grouped from the 21 model columns
back into ten business concepts, and the test suite checks that they reconstruct
`predict_proba` — an attribution that does not is decoration. `explanation_method` reports
which engine wrote the sentence, and `attributions` carries the signed per-concept
contributions in log-odds. A missing `shap`, a non-tree model, or an explainer that raises
all degrade to the rules rather than failing the request.

**Dockerization:** multi-stage Dockerfile. The builder stage installs dependencies and
**trains the model during the image build**, so the resulting image ships ready to serve
with no volume mount required. Runtime stage is a slim `python:3.11-slim` base running as
a non-root user, with a `HEALTHCHECK` wired to `/health`.

**CI** (`.github/workflows/ci.yml`), two jobs on push/PR to `main`:
1. **Lint and test** — checkout → Python 3.11 (pip-cached) → install → `ruff check` →
   `pytest` → train end-to-end → evaluate the saved artifact → upload artifacts.
2. **Docker** (runs only if job 1 passes) — builds the image, starts a container, polls
   `/health`, asserts the dashboard serves, posts a real request to `/predict`.

Both jobs have passed on the last verified run. For a long stretch the Docker job was the
one part of this project only CI had ever executed — Docker Desktop was not installed on
the local development machine. It has since been installed and the full eight-service
compose stack brought up locally (see §6.4), which is what surfaced the two bugs recorded
there: neither would have been found by CI's narrower three-service smoke test.

### 3.2 Browser dashboard (commits `cbab7d2`, `96ef8ce`, `381ac92`) — ✅ complete, pushed

A single dependency-free HTML file (`src/api/static/index.html`) with inline CSS/JS — no
build step, no bundler, no CDN dependency. Ships inside the same container; nothing was
added to `requirements.txt`.

- Three one-click presets: at-risk, healthy, borderline.
- Renders probability as a bar **against the model's actual decision threshold** (marker
  at 0.26), not against an assumed 50% cutoff.
- Header badge reflects `/ready` — shows "no model loaded" if the artifact is missing,
  rather than presenting a dead form.
- Light/dark themes follow the OS setting.

**Contract-guard tests (10 of them) parse the HTML file itself** and assert: the form's
input field IDs exactly equal `SubscriberFeaturesRequest.model_fields`; every path the page
`fetch()`es is a real registered route; each of the three presets round-trips through
`/predict` successfully; the at-risk preset scores strictly higher than the healthy one.
Renaming a field in `schemas.py` without updating the HTML form fails these tests.

**A real bug was found and fixed during this work:** the "borderline" demo preset was
originally chosen to score 0.069 against a 0.26 threshold — solidly low-risk, not
borderline at all, so it demonstrated nothing. It was replaced with inputs that score
0.2605, landing just past the threshold.

### 3.3 Monitoring & drift detection (commit `0b82ce1`) — ✅ complete, LOCAL ONLY (not pushed)

Two new modules plus two new endpoints, added because a model decays silently when the
world stops resembling its training data.

**`src/monitoring/profile.py`** — every training run now writes a fourth artifact,
`reference_profile.json`, alongside `model.joblib`. It stores the training data's
distribution as **quantile bins**, not raw rows — keeping the artifact small and
fixed-size rather than shipping a copy of the training set. Covers every numeric input
column, category shares for `plan_type` and `is_auto_renew_enabled`, and the model's own
score distribution.

**`src/monitoring/drift.py`** — scores live traffic against that baseline using the
**Population Stability Index**:

```
PSI = Σ (live_i − ref_i) · ln(live_i / ref_i)
```

Chosen over a KS test because its conventional bands are directly actionable without
picking a significance level:

| PSI | Verdict |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | moderate |
| > 0.25 | significant |

Implementation details that make it behave correctly:
- **Quantile bins, not equal-width** — on a skewed column, equal-width bins leave most
  bins near-empty and PSI explodes on meaningless movement.
- **Epsilon added before the logarithm** — one category absent from either side would
  otherwise send PSI to infinity and swamp the whole report.
- **Out-of-range live values are clipped into the outer bins, not discarded** — a value
  larger than anything in training is still evidence.

**Verified against three real scenarios** (numbers from actual test runs, not estimates):

| Scenario | Result |
|---|---|
| Held-out test data vs. itself | `stable`, all PSI < 0.02 — no false alarms |
| Simulated engagement collapse | `significant` — correctly isolated to `last_activity_days_ago` (PSI 10.68) and prediction drift (PSI 3.19); `monthly_fee` correctly left alone |
| Unseen "enterprise" plan tier | `significant`, isolated entirely to `plan_type` (PSI 5.37); everything else stable |

**The third scenario is the important one.** The unseen plan tier moves the *input*
distribution sharply while *prediction* drift stays essentially flat, because
`OneHotEncoder(handle_unknown="ignore")` silently absorbs the unknown category into an
all-zero block. Watching only the model's output would never reveal this — which is why
input and output drift are computed and reported **separately**.

**`src/monitoring/tracker.py`** — an in-process, bounded rolling window (`deque`,
default 5,000 entries) of every prediction served. Deliberately **not** a metrics backend
(no Prometheus, no persistence) — it answers "what is this replica doing right now," and
resets when the process does. `GET /metrics` always returns `200` even with no model
loaded, since a monitoring endpoint that fails when the thing it watches is unhealthy is
worse than useless.

`GET /metrics` reports: rolling mean/p50/p90 probability, flagged rate, risk-band counts,
and `probability_mean_shift` — live mean score minus the training-time mean. That last
number is the single most alertable value in the whole monitoring layer.

**A real bug was found and fixed while building this:** `classify_risk_level()` originally
used fixed cutoffs (`low < 0.35`, `high ≥ 0.65`) chosen independently of the tunable
decision threshold. With a tuned threshold of 0.26, every subscriber scoring between 0.26
and 0.35 was **flagged for retention outreach by `predicted_label`** while the API
simultaneously described them as `"Low dropout risk"` — a direct self-contradiction, and
not an edge case (a real slice of traffic falls in that band). Fixed by anchoring bands to
the threshold: `low` is now defined as exactly the not-flagged region
(`probability < threshold`), `high` begins halfway between the threshold and 1.0. This
enforces the invariant `risk_level == "low" ⟺ predicted_label == 0`, guarded by a 36-case
test sweeping four thresholds against nine probabilities, plus an end-to-end API check.

**43 tests** in `tests/test_monitoring.py` cover PSI algebra (zero for identical
distributions, symmetric, stays finite when a bin empties), binning edge cases (constant
columns, empty columns), the critical false-positive check (reference data vs. itself must
read `stable`), real-shift detection and attribution, tracker window bounds, and both new
endpoints including their degraded-state (no model / no baseline) paths.

### 3.4 Stage 1 — Temporal event warehouse (commit `388ec83`) — ✅ complete, LOCAL ONLY (not pushed)

**The problem this solves:** the original design stored one wide row per subscriber with
everything pre-aggregated, and split into train/test **randomly**. This is wrong twice
over:
1. Churn is a time-series problem. A random split lets the model learn from behavior that
   happened *after* the outcome it's predicting — a validation score built that way is
   dishonest, because at serving time the future is not available.
2. A pre-aggregated static table has no "as of." You cannot ask what a subscriber looked
   like last March without having stored a snapshot of March. Every later MLOps concept —
   backfills, scheduled retraining, drift over time — depends on being able to ask exactly
   that question, and none of them work on a table with no time dimension.

**`src/warehouse/schema.py`** — five SQLAlchemy tables of immutable, timestamped events.
Nothing is pre-aggregated:

| Table | Grain | Carries |
|---|---|---|
| `subscribers` | one row per person | signup date, acquisition channel — facts that don't change |
| `subscription_events` | one row per lifecycle event | signup / renewal / plan_change / cancellation, plus plan & fee at that moment |
| `sessions` | one row per usage session | by far the highest-volume table |
| `payments` | one row per charge attempt | amount, succeeded/failed, discount applied |
| `support_tickets` | one row per ticket | category |

Runs on **SQLite locally and Postgres in the compose stack** via the same SQLAlchemy
schema, so the SQL is exercised by every local test run and not only in deployment.

**`src/warehouse/simulate.py`** — event generator, replacing the old flat-file generator.
Each subscriber carries three latent traits (`engagement`, `dissatisfaction`,
`price_sensitivity`) that are **never written to any table**; observable behavior
(sessions, tickets, payment failures) is drawn conditionally on those traits, so features
are genuinely predictive without the label being a closed-form function of any single
column.

The daily cancellation hazard combines those latent traits with one **observable** term:
**dormancy** (days since last session). This was added during Stage 2 work after
discovering the hazard originally depended *only* on hidden state, meaning no feature set
could possibly beat random guessing (ROC-AUC was stuck at 0.656). Dormancy is the
strongest real-world churn signal and it's directly visible in the session log — adding it
raised temporal-split ROC-AUC from 0.656 to 0.709 on the 4,000-subscriber warehouse of
the time. The current 8,000-subscriber pipeline reports 0.674; the dormancy finding stands,
the absolute number moved with the population and window.

**Injectable drift** — `DriftScenario` changes subscriber behavior from a chosen date
onward: engagement collapse, payment-failure spikes, ticket surges, or an entirely new plan
tier appearing partway through the simulation. This is the actual reason to simulate data
rather than download a static dataset: you cannot demonstrate a drift detector firing, a
retrain triggering, or a challenger overtaking a champion on a fixed Kaggle CSV.

**`src/features/point_in_time.py`** — the module that makes the warehouse meaningful.
Builds labeled training snapshots with a strict separation around a cutoff `T`:

```
[ T − 30 days , T )        features   — only ever looks backwards
[ T , T + 30 days ]        label      — never visible to the features
```

Windows are disjoint by construction. Aggregation happens **in SQL**, not pandas —
pulling every session row into memory would not survive real volumes (2,000 simulated
subscribers over 15 months already produce ~126,000 session rows).

**One subtlety worth knowing:** session *recency* deliberately scans **all** history, not
just the 30-day observation window. A subscriber dormant for 90 days has *zero* rows in a
30-day window — treating that as "no data" instead of "long absent" would erase the
single strongest signal in the model. Counts use the window; recency uses everything
before the cutoff.

**The leakage guard — the most important test in the entire repository.** The test:
1. Build a snapshot at cutoff `T`.
2. Write 200 extra sessions, 25 support tickets, and 15 failed payments dated *after* `T`.
3. Rebuild the snapshot.
4. Assert not a single feature value changed.

This test was **mutation-tested** to prove it isn't vacuous: the `< :cutoff` bound was
deliberately deleted from the SQL, and the test was confirmed to then fail. Output from
that check, captured during development:
```
CORRECT SQL  -> features unchanged  ✓  (test passes, as expected)
LEAKY SQL    -> features CHANGED    ✓  (test correctly fails)
```

**Backfills and temporal splits:** because a snapshot is a pure function of the cutoff
date, running the same function across a series of cutoffs produces a growing panel —
this is what makes a scheduled retrain do real work instead of refitting identical data
forever. `build_temporal_splits()` assigns the *latest* cutoff to the test set, the one
before it to validation, and everything earlier to training — mirroring exactly what the
model faces in production (fit on the past, score the future). Example real output:

```
Temporal split over 18 cutoffs
  train      2024-01-01 … 2025-03-26   (16 cutoffs)  18,599 rows
  validation 2025-04-25                               1,610 rows
  test       2025-05-25                               1,615 rows
```

**23 tests** in `tests/test_point_in_time.py`: the leakage guard itself, window boundary
arithmetic, label-horizon exclusion, already-cancelled-subscriber exclusion,
not-yet-signed-up exclusion, backfill stacking, and simulator reproducibility/drift
injection.

### 3.5 Stage 2 — MLflow tracking, model registry, gated promotion (commit `82c5c5d`) — ✅ complete, LOCAL ONLY (not pushed)

**The problem this solves:** training previously overwrote `model.joblib` in place — a
deployment with no gate. An unlucky seed, a corrupted backfill, or a bad training run
would silently replace a good model, and rolling back meant retraining and hoping.

**`src/registry/tracking.py`** — wraps MLflow. Every training run logs its
hyperparameters, validation/test metrics, the point-in-time window it trained on, and
registers the fitted pipeline as a new, **immutable** version in the model registry.
MLflow 3 removed model *stages*, so production status is expressed via **aliases**:
`@champion` is whatever is currently serving, `@challenger` is whatever is being
evaluated.

**⚠️ A real compatibility issue found and fixed:** MLflow 3 defaults to `skops`
serialization, which refuses to serialize custom Python callables. This project's pipeline
deliberately embeds `add_derived_features` as a `FunctionTransformer` precisely so feature
engineering travels with the saved artifact — and `skops` rejects exactly that pattern by
design. Fixed by forcing `SERIALIZATION_FORMAT_CLOUDPICKLE` explicitly — the same trust
level as the `joblib` file the API already loads — with a round-trip test asserting
identical predictions after a save/reload cycle.

**`src/registry/promote.py`** — the gate itself. `evaluate_promotion()` produces a
`PromotionDecision`; `promote()` acts on it by moving the `@champion` alias only if
`decision.promoted` is true. Three properties make this a real gate rather than a
formality:

1. **Comparable** — champion and challenger are scored on the *same* held-out data, in the
   same process. (Comparing a fresh challenger score against a number the champion
   recorded months ago is the classic way to accidentally promote a worse model — those
   two numbers were never measured on the same population.)
2. **Marginal** — a challenger must beat the champion by `PROMOTION_MIN_IMPROVEMENT`
   (default `0.005`), not merely tie. Without a margin, two statistically equivalent
   models differ only by noise, and a zero-margin gate would promote on that noise roughly
   half the time, churning the registry forever.
3. **Reversible** — a rejected challenger is not deleted; it stays registered. Promotion
   only moves the `@champion` alias, so `rollback()` is just moving it back to an earlier
   version.

**Gating metric is PR-AUC, not ROC-AUC** — at a ~20% positive class rate, average
precision reflects retention-outreach performance far more honestly; ROC-AUC flatters
imbalanced data.

**Real end-to-end verification output** (captured during development, four sequential
training runs against the same warehouse):

```
[run 1]  v1  PROMOTED: no incumbent champion (pr_auc=0.3707)
[run 2]  v2  REJECTED: improvement -0.0524 below required +0.0050
             | pr_auc challenger=0.3183 champion=0.3707
[run 3]  v3  REJECTED: improvement +0.0000 below required +0.0050
[run 4]  v4  REJECTED: improvement -0.0082 below required +0.0050
```

Run 3 is the critical proof point — it is the *identical model retrained with the same
hyperparameters*. A gate without a margin would have promoted it purely on floating-point
noise. Run 4 shows a plausible-looking "improvement" (more estimators, more depth) still
correctly rejected because it didn't clear the margin on this particular held-out window —
the gate doesn't reward effort, only measured improvement.

**14 tests** in `tests/test_registry.py`, using a throwaway SQLite-backed MLflow store
(a file store cannot host a model registry, so SQLite is the minimum viable backend that
exercises real registry code without running a server): registration, alias resolution,
the cloudpickle round-trip, and all four gate outcomes (first-model-unopposed,
worse-challenger-rejected, tied-challenger-rejected, clearly-better-challenger-promoted),
plus rollback.

**`train.py` gained new CLI flags:**
```bash
python -m src.models.train --source warehouse              # point-in-time features, temporal split
python -m src.models.train --source warehouse --promote    # + register in MLflow, run the gate
python -m src.models.train --cutoffs 2024-06-01 2024-07-01 2024-08-01   # explicit cutoffs
```

### 3.6 Full test suite — 413 tests, all passing

```
tests/test_api.py            83 tests   — every endpoint, 422 validation, 503 degraded
                                          paths, batch/single agreement, risk-band
                                          invariant, 10 dashboard-contract tests
tests/test_monitoring.py     43 tests   — PSI algebra, binning edges, false-positive
                                          check, drift attribution, tracker bounds
tests/test_point_in_time.py  23 tests   — the leakage guard, window arithmetic, label
                                          horizons, backfills, simulator + drift injection
tests/test_train.py          21 tests   — dataset schema, reproducibility, artifact
                                          creation, better-than-chance performance
tests/test_features.py       19 tests   — derived columns, immutability, zero-denominator
                                          edge cases, unseen-category handling
tests/test_registry.py       20 tests   — registration, alias handling, cloudpickle
                                          round-trip, all 4 promotion-gate outcomes
tests/test_evaluation.py     38 tests   — calibration arithmetic, cost model, offer
                                          efficacy, capacity constraint, fairness
tests/test_prometheus.py     35 tests   — exposition format, both coverage guards,
                                          Alertmanager routing and inhibition
tests/test_streaming.py      36 tests   — poison messages, partial batch failures, a
                                          produce failure that must not commit
tests/test_external_data.py  26 tests   — KKBox mapping, warehouse contract, orphan
                                          events, pre-signup activity
tests/test_orchestration.py  22 tests   — ingest idempotency, run-report escalation,
                                          one real Prefect flow run
tests/test_dvc.py            17 tests   — params.yaml coverage in both directions,
                                          pipeline structure, dvc.lock exists
tests/test_shadow.py         16 tests   — the safety contract: a raising challenger
                                          must never change a served response
tests/test_explain.py        14 tests   — SHAP additivity, concept grouping, and the
                                          three ways attribution degrades safely
─────────────────────────────────────────
Total                       413 tests   — runs in ~15-30 seconds
```

The suite trains one small model per session into a temporary directory — it never writes
into the repository and doesn't depend on the working tree's current artifacts.

### 3.7 Configuration — everything tunable in one place

`src/config/settings.py` (215 lines) holds every path, seed, hyperparameter, and threshold.
**32 environment variables** (all `SDD_`-prefixed except MLflow's own) override them:

```
Paths:          SDD_DATA_DIR  SDD_ARTIFACTS_DIR  SDD_MODEL_PATH  SDD_RAW_DATA_PATH
Warehouse:      SDD_DATABASE_URL  SDD_SIMULATION_START  SDD_SIMULATION_END
Point-in-time:  SDD_OBSERVATION_WINDOW_DAYS  SDD_PREDICTION_HORIZON_DAYS
Model:          SDD_N_ESTIMATORS  SDD_LEARNING_RATE  SDD_MAX_DEPTH  SDD_SUBSAMPLE
                SDD_MIN_SAMPLES_LEAF  SDD_RANDOM_SEED
Thresholds:     SDD_DECISION_THRESHOLD  SDD_TUNE_THRESHOLD  SDD_TEST_SIZE
                SDD_VALIDATION_SIZE
Drift:          SDD_PSI_MODERATE  SDD_PSI_SIGNIFICANT  SDD_DRIFT_BIN_COUNT
                SDD_DRIFT_MIN_SAMPLES  SDD_METRICS_WINDOW
Registry:       MLFLOW_TRACKING_URI  SDD_MLFLOW_EXPERIMENT  SDD_REGISTERED_MODEL
                SDD_PROMOTION_METRIC  SDD_PROMOTION_MIN_IMPROVEMENT
API:            SDD_API_HOST  SDD_API_PORT
```

---

## 4. Runtime dependencies

```
# Runtime (requirements.txt) — 10 packages, all version-bounded
fastapi>=0.110,<1.0        uvicorn[standard]>=0.27,<1.0     pydantic>=2.6,<3.0
scikit-learn>=1.4,<2.0     pandas>=2.2,<3.0                 numpy>=1.26,<3.0
joblib>=1.3,<2.0           SQLAlchemy>=2.0,<3.0             psycopg[binary]>=3.1,<4.0
mlflow>=2.14,<4.0

# Dev only (requirements-dev.txt) — adds:
pytest>=8.0,<9.0    pytest-cov>=5.0,<7.0    httpx>=0.27,<1.0    ruff>=0.5,<1.0
```

All carry upper bounds so a surprise major release can't silently break the build.

---

## 5. What's LEFT — the six-stage roadmap

Full detail lives in `ROADMAP.md`; summarized here with current status:

| # | Stage | Status | What it proves |
|---|---|---|---|
| 1 | Temporal data layer (warehouse + point-in-time features) | ✅ **Done** | Point-in-time correctness, leak-free features, backfills |
| 2 | Experiment tracking + model registry + gated promotion (MLflow) | ✅ **Done** | Reproducibility, model governance, no silent overwrites |
| 3 | Orchestration (Prefect) | ⬜ **Not started** | Scheduling, retries, idempotency, recovery |
| 4 | Observability (Prometheus + Grafana) | ⬜ **Not started** | Durable, cross-replica monitoring (replacing the in-process `/metrics` window) |
| 5 | Champion/challenger shadow scoring | ⬜ **Not started** | Safe deployment — the highest-signal stage in the plan |
| 6 | Scale (full compose stack → K8s, streaming inference) | ⬜ **Not started** | Production shape |

### Stage 3 detail (next up): Orchestration with Prefect

One flow: `ingest → build features → train → evaluate → gate → deploy`. Scheduled,
retryable, and backfillable across historical cutoffs. **Prefect was chosen over Airflow**
deliberately — Airflow needs a scheduler, webserver, and metadata DB running permanently,
which is heavy operational overhead for a solo project; Prefect demonstrates the same
core concepts (scheduling, retries, backfills, observability) for a fraction of the setup
cost. Airflow would only be worth the extra week of plumbing if a specific target job
posting names it explicitly.

### Stage 4 detail: Observability with Prometheus + Grafana

Replaces `src/monitoring/tracker.py`'s in-process window (which resets when the process
restarts and can't be compared across replicas) with durable, exportable time-series
metrics. Scheduled drift jobs would write PSI values as a time series; dashboards would
show score distributions and drift trends over weeks; alerts would fire on a `significant`
verdict automatically instead of requiring someone to manually call
`POST /monitoring/drift`.

### Stage 5 detail: Champion/challenger shadow scoring

Both the champion and challenger model score **every** live request. The challenger's
predictions are logged but never actually returned to the caller. After enough shadow
traffic accumulates, the promotion decision is made from real production-traffic evidence
rather than only from a static held-out test split. This is also where the API would
switch from loading `model.joblib` off disk to resolving `@champion` (and `@challenger`)
directly from the MLflow registry — see caveat in §6.3.

### Stage 6 detail: Scale

Full `docker-compose` stack exercised end-to-end (currently blocked — see §6.4), optional
Kubernetes manifests, and — deliberately saved for last — streaming inference over
Redpanda (a lighter, Kafka-API-compatible broker), since the event-shaped warehouse from
Stage 1 makes that addition natural once everything else is proven.

### Deliberately excluded from the roadmap (with reasons)

- **Feature store (Feast).** At this project's scale it would be mostly YAML
  configuration. The hard problem a feature store actually solves — offline/online
  parity — is already directly demonstrated by the Stage 1 point-in-time SQL code.
- **Airflow** (see Stage 3 rationale above).
- **Bolting on many tools shallowly.** Four tools that genuinely work end-to-end beats
  twelve that each half-work — the latter reads as box-checking, not engineering.

---

## 6. Honest caveats — things a careful reviewer would find

Listed plainly so they're discovered here, not by surprise.

### 6.1 The 0.907 ROC-AUC is not a real-world number — and neither, fully, is 0.674

Two data paths currently coexist in this codebase:

| Path | ROC-AUC | PR-AUC | Recall | How it's validated |
|---|---|---|---|---|
| Flat CSV (`make train`, **the CI/current default**) | 0.907 | 0.766 | 0.724 | Random stratified split |
| Warehouse (`--source warehouse`, `dvc repro`) | 0.674 | 0.345 | 0.755 | **Temporal split** (train on past, test on future) |

**The 0.907 number is inflated by circularity, not by leakage.** The original flat-CSV
generator drew the `dropout` label directly from a logistic function of the very columns
it then handed to the model — the model was rediscovering a rule that had been written
into the data generator. That number should never be quoted as evidence of real predictive
skill.

**A finding that contradicts the "random split leaks the future" intuition:** switching
*only* the split methodology (random → temporal) on the *same* warehouse data moved
ROC-AUC by just **−0.065** — and in the direction that made the random split look
slightly *worse*, not artificially better. So while temporal splitting remains the
methodologically correct choice going forward (it's the only scheme that stays honest once
`DriftScenario` introduces genuine time-varying behavior), it was **not** the source of the
old inflated number. That was purely the circular-label problem in the old generator,
described above.

**Bottom line: ~0.71 ROC-AUC on the warehouse/temporal path is the number to trust**, and
it's ordinary, unremarkable territory for a real churn model — which is exactly what an
honest number should look like.

### 6.2 Training still defaults to the legacy flat-CSV path

`make train` (and CI) both currently run the **random-split CSV path**, not the
warehouse/temporal path. The warehouse path exists, is fully tested, and is the
methodologically honest one — but it is opt-in via `--source warehouse`, not the default.

This was a deliberate choice during Stage 1/2 development: switching the default would
have required rewriting existing tests and risked breaking CI mid-migration. The tradeoff
is that `make train` (and the number a casual reader sees first in the metrics file) is
still the less honest 0.907 pathway. **Flipping the default to `--source warehouse` is
unfinished work**, not yet scheduled into a specific stage.

### 6.3 The API serves from disk, not from the registry

`src/api/service.py` loads `model.joblib` directly from the filesystem. Stage 2 built a
full model registry with `@champion`/`@challenger` aliases and a gated promotion
mechanism — but the running API does not yet consult it. Promotion currently governs
*the registry's bookkeeping*, not *what the live service actually serves*. This is
intentionally deferred to Stage 5, where the serving layer has to learn how to resolve two
models (champion + challenger) simultaneously anyway, so wiring registry-loading in twice
would be wasted work.

### 6.4 Docker — installed, and the full eight-service stack has now been run

Docker Desktop was not installed on the machine this project is developed on for most of
its life. It has since been installed, and `docker compose up --build` has been run against
the full stack — `postgres`, `mlflow`, `subscriber-api`, `prometheus`, `alertmanager`,
`grafana`, `redpanda`, `stream-scorer` — with all eight containers reaching a running state
and Prometheus's four scrape targets confirmed **up**, including `stream-scorer` on `:8001`,
which no test running outside a container could ever exercise.

Bringing it up for real, rather than only reading the YAML, found two bugs that a
structural test cannot see and CI's narrower three-service job never touched:

- **`stream-scorer` was permanently reported unhealthy.** It shares an image with
  `subscriber-api`, and that image bakes in a `HEALTHCHECK` that curls `:8000/health`.
  Correct for the API; meaningless for the streaming consumer, which never binds 8000.
  `docker ps` showed the container unhealthy regardless of whether it was actually
  consuming — fixed by disabling the inherited healthcheck for this service, since a
  stalled consumer already surfaces correctly through the `StreamScorerDown` alert on
  `up{job="stream-scorer"}`.
- **MLflow was unreachable from outside its own container.** `mlflow server --host 0.0.0.0`
  reliably bound gunicorn to `127.0.0.1` when run as the container's own boot command —
  reproduced and confirmed by running the identical command manually inside the already-running
  container, where it bound `0.0.0.0` correctly. `--gunicorn-opts` did not override it either.
  `docker compose up` gave no error: the server logged "Listening" and looked healthy while
  being unreachable from the host, from Prometheus, or from any other container. Fixed by
  running gunicorn directly against MLflow's own WSGI app — MLflow's documented pattern for
  production deployment — configured through the same private environment variables
  `mlflow server` sets internally before the step that was going wrong.

A third issue was a plain port collision: macOS's Control Center (AirPlay Receiver) binds
`*:5000` by default on Sonoma and later, which is also MLflow's default port, and
`docker compose up` failed outright with "address already in use" until the host-side
mapping moved to `5001:5000`. All three fixes are covered by regression tests in
`tests/test_streaming.py` that parse `docker-compose.yml` and assert the fix stays in place.

Kubernetes remains unverified — the manifests need a real cluster (`kind` or Docker
Desktop's built-in one), which is a separate exercise from bringing up compose.

### 6.5 Model quality is modest, and stated honestly rather than oversold

On the temporal/warehouse path: ROC-AUC 0.674, recall 0.755 at the tuned threshold. These
come from `dvc repro`, with `dvc.lock` recording the exact inputs. Earlier revisions of
this document reported 0.709 from a 4,000-subscriber warehouse over a shorter window — the
simulate CLI defaulted to 4,000 while the configuration said 8,000. Validation and test are
adjacent cutoffs of the same run and differ by 0.035 ROC-AUC, so the spread between those
two reported figures is inside the run's own cutoff-to-cutoff variation. No
probability calibration has been attempted (e.g. `CalibratedClassifierCV`), so raw
predicted probabilities should not be plugged directly into expected-value/cost
calculations without that step. Threshold tuning currently optimizes F1, not a real
business cost ratio between a missed churner and a wasted retention offer — that's listed
as unstarted future work in the README.

### 6.6 Data is entirely synthetic

Every row in this project — flat CSV and warehouse alike — comes from a generator, never
from a real subscription business. For a portfolio piece meant to demonstrate *modeling*
skill, that's a real limitation and real data (e.g. the KKBox WSDM Cup 2018 churn dataset)
would be a stronger foundation. For a portfolio piece meant to demonstrate
*infrastructure/MLOps* skill, synthetic data with **injectable, on-demand drift** is
arguably the *better* choice, specifically because it lets the drift-detection and
promotion-gate machinery be demonstrated actually firing — something no static downloaded
CSV could ever provide. The README states this tradeoff plainly rather than hiding it.

---

## 7. Every command available right now

```bash
# --- setup ---
make install                 install dev dependencies (this venv is uv-managed, no pip)

# --- data ---
make data                    generate the legacy flat synthetic CSV
make simulate                populate the event warehouse (Stage 1 path)
make warehouse-up            start Postgres only, via docker compose

# --- training ---
make train                   flat CSV, random split                    (current CI default)
make train-warehouse         point-in-time features, temporal split    (the honest path)
make train-promote           + register in MLflow and run the gate
make evaluate                re-score the currently-saved artifact
make mlflow-ui               browse runs/registry at http://127.0.0.1:5000

# --- serving ---
make serve                   uvicorn --reload → http://127.0.0.1:8000
make docker-build            build the image (trains model during build) — untested locally
make docker-up               full compose stack — untested locally

# --- quality ---
make test                    run all 413 tests
make lint                    ruff check
make clean                   remove generated data, artifacts, caches
```

Direct CLI equivalents for the training script:
```bash
python -m src.models.train                                   # flat CSV, default
python -m src.models.train --source warehouse                 # temporal split
python -m src.models.train --source warehouse --promote       # + MLflow gate
python -m src.models.train --cutoffs 2024-06-01 2024-07-01 2024-08-01
python -m src.warehouse.simulate --subscribers 2000 --start 2024-01-01 --end 2025-03-31
```

---

## 8. Immediate next actions, in priority order

1. **Delete `.env` and revoke the exposed GitHub token** (§2, ⚠️ outstanding). Not
   blocking any other work, but it's live and unresolved.
2. **`git push`** — three finished, tested stages of work (monitoring, Stage 1, Stage 2)
   are sitting local-only and not reflected on GitHub or in CI at all yet.
3. **Install Docker Desktop** — hard blocker for Stage 3 onward, and the only way to
   finally exercise `docker-compose.yml` locally instead of trusting CI alone.
4. **Begin Stage 3** (Prefect orchestration) once Docker is available.
5. Lower priority / not yet scheduled: flip `make train`'s default to the warehouse path
   (§6.2), wire the API to load from the MLflow registry (§6.3).
