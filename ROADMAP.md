# Roadmap: from trained model to operated system

The project currently trains a good model and serves it well. What it does not do is
*operate* one. This roadmap closes that gap.

The ordering is not arbitrary. Each stage is load-bearing for the next, and the first is
load-bearing for all of them.

---

## Why Stage 1 comes first

Today the project has **no time dimension**: one static CSV, one training run, a random
train/test split.

Every later concept is meaningless without time:

- *Scheduled retraining* on an unchanging CSV retrains an identical model, forever.
- *Drift detection* cannot fire, because the distribution never moves.
- *Backfills* have nothing to fill.
- *Champion vs challenger* cannot be compared, because there is no new data to compare on.

A random split also **leaks the future**: churn is a time-series problem, and predicting
January's churn from February's behaviour is not a skill that survives contact with
production. Stage 1 replaces the CSV with a temporal event log and point-in-time-correct
feature computation.

## A note on synthetic data

The data remains simulated, and the README says so plainly.

For an *ML* portfolio that would be a serious weakness — a model that recovers a rule the
generator wrote proves nothing. For an *infrastructure* portfolio it is the better
instrument: a simulator whose drift can be **switched on deliberately** is what makes the
monitoring, retraining and promotion machinery demonstrable rather than hypothetical. You
cannot show a drift detector firing on a static Kaggle download.

---

## Stages

### 1. Temporal data layer — ✅ done
Event tables (`subscribers`, `subscription_events`, `sessions`, `payments`,
`support_tickets`) in Postgres, written by an event simulator that runs over a date range
and accepts injectable drift. Features are computed **in SQL** against a cutoff date, from
an observation window that only ever looks backwards. Labels come from a disjoint
prediction window after the cutoff.

*Proves:* point-in-time correctness, leak-free feature engineering, backfills.

### 2. Experiment tracking and model registry — MLflow — ✅ done
Every run logs params, metrics and artifacts. Promotion is **gated**: a challenger only
takes over if it beats the incumbent by a margin, scored on the same held-out window in the
same process. Rollback is an alias move, because nothing is ever deleted.

MLflow 3 removed model *stages*, so production status is an alias — `@champion` is what
serves, `@challenger` is what is being judged.

*Proves:* reproducibility, model governance, no silent overwrites.

The API resolves `@champion` from the registry at startup (`SDD_MODEL_SOURCE`, default
`auto`), falling back to the local artifact when MLflow is unreachable — so promotion
changes what actually gets served, not just a database row.

### 3. Orchestration — Prefect — ✅ done
One flow: `ingest → drift → train + gate → report`. Scheduled, retryable, and backfillable
across historical cutoffs.

The pipeline logic lives in `src/orchestration/pipeline.py` as **plain functions with no
Prefect import**; `flows.py` wraps each in a `@task`. That keeps the test suite fast
(Prefect starts a temporary API server per flow run) and means swapping orchestrator later
touches one thin module rather than the pipeline itself.

*Proves:* automation, idempotency, recovery.

*Still open:* alerting is a `needs_attention` flag in the run report, not a page or an
email. Wiring it to a real notifier belongs with Stage 4, where there is somewhere to send
it.

### 4. Observability — Prometheus + Grafana
Replaces the in-process `/metrics` window, which resets when the process does. Scheduled
drift jobs write PSI as time series; dashboards show score distributions and drift over
weeks; alerts fire on a `significant` verdict.

*Proves:* durable, cross-replica monitoring.

### 5. Champion / challenger with shadow scoring
Both models score every request. The challenger's scores are **logged, never served**.
After enough traffic, promotion is decided on evidence rather than on a hunch.

*Proves:* safe deployment. This is the most senior-signal piece in the roadmap.

### 6. Scale
Full `docker-compose` stack, then Kubernetes if warranted. Streaming inference over
Redpanda last — the event layer from Stage 1 already makes the data the right shape.

---

## Deliberately excluded

- **Feature store (Feast).** At this scale it is mostly YAML. The hard problem it solves —
  offline/online parity — is demonstrated directly by the Stage 1 point-in-time code.
- **Airflow.** Prefect shows the same concepts for a fraction of the operational overhead
  on a solo project.
- **Twelve tools bolted together.** Four that genuinely work beats a dozen that half-do.
