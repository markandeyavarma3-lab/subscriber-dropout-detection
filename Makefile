# Shortcuts for the common workflows. Run `make help` for the list.
.PHONY: help install data simulate ingest ingest-check warehouse-up train train-warehouse train-promote pipeline pipeline-drift backfill schedule stream stream-up stream-events stream-poison audit observability-up metrics mlflow-ui evaluate repro params metrics-diff dag serve test lint format docker-build docker-run docker-up clean warehouse-postgres monitor demo-prepare demo demo-check demo-down

# python3, not python: macOS has shipped python3 via Xcode's Command Line
# Tools for years and dropped the bare `python` symlink long ago, so on any
# machine without Homebrew's own python formula - confirmed live, on a stock
# Sonoma install with neither Homebrew nor a virtualenv active - `python`
# resolves to nothing and every target below fails with "No such file or
# directory". `?=` still lets an activated venv or an explicit `PYTHON=...`
# override this, so nothing is lost for anyone who does have `python` on PATH.
PYTHON ?= python3
IMAGE  ?= subscriber-dropout-api

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install development dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

data:  ## Generate the synthetic dataset
	$(PYTHON) -m src.data.generate

simulate:  ## Populate the event warehouse with simulated subscriber events
	$(PYTHON) -m src.warehouse.simulate

ingest-check:  ## Validate an external dataset without writing (SRC=path/to/kkbox)
	$(PYTHON) -m src.data.external.ingest --dataset kkbox --source-dir $(SRC) --dry-run

ingest:  ## Load an external dataset into the warehouse (SRC=path/to/kkbox)
	$(PYTHON) -m src.data.external.ingest --dataset kkbox --source-dir $(SRC)

warehouse-up:  ## Start the Postgres warehouse only
	docker compose up -d postgres

train:  ## Train the model and write the artifacts
	$(PYTHON) -m src.models.train

train-warehouse:  ## Train on point-in-time warehouse data with a temporal split
	$(PYTHON) -m src.models.train --source warehouse

train-promote:  ## Train, register in MLflow, and run the promotion gate
	$(PYTHON) -m src.models.train --source warehouse --promote

pipeline:  ## Run the full retraining pipeline once (ingest -> drift -> train -> gate)
	$(PYTHON) -m src.orchestration.flows run

pipeline-drift:  ## Same, but inject a behavioural shift first to demo drift detection
	$(PYTHON) -m src.orchestration.flows run --drift

backfill:  ## Replay training across historical cutoffs, oldest first
	$(PYTHON) -m src.orchestration.flows backfill

schedule:  ## Serve the pipeline on a nightly cron (blocks; Ctrl-C to stop)
	$(PYTHON) -m src.orchestration.flows serve --cron "0 3 * * *"

stream:  ## Run the streaming scorer against Redpanda (needs the broker up)
	$(PYTHON) -m src.streaming.kafka

stream-events:  ## Publish sample events onto the input topic (N=50, RATE=0)
	$(PYTHON) -m src.streaming.produce --count $(or $(N),50) --rate $(or $(RATE),0) --brokers localhost:19092

stream-poison:  ## Publish malformed events to exercise the dead-letter topic
	$(PYTHON) -m src.streaming.produce --count 5 --corrupt --brokers localhost:19092

stream-up:  ## Start Redpanda and the streaming scorer
	docker compose up -d redpanda stream-scorer

observability-up:  ## Start Prometheus + Grafana (needs the API running)
	docker compose up -d prometheus grafana

metrics:  ## Show the raw Prometheus exposition from a running API
	@curl -s http://127.0.0.1:8000/metrics/prometheus | grep -E '^subscriber_' || \
		echo "API not running - start it with 'make serve'"

audit:  ## Print the decision-quality report from the last training run
	@$(PYTHON) -c "import json;d=json.load(open('src/models/artifacts/metrics.json'))['decision_quality'];print(json.dumps(d,indent=2))"

# 5050, not MLflow's default 5000: macOS's AirPlay Receiver owns *:5000.
mlflow-ui:  ## Browse runs and the registry at http://127.0.0.1:5050
	$(PYTHON) -m mlflow ui --port 5050 \
		--allowed-hosts localhost,localhost:5050,127.0.0.1,127.0.0.1:5050,host.docker.internal,host.docker.internal:5050 \
		--backend-store-uri $${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}

repro:  ## Rerun the DVC pipeline, skipping stages whose inputs did not change
	dvc repro

params:  ## Show the parameters the last locked run actually used
	dvc params diff --all

metrics-diff:  ## How the working tree's metrics compare against main
	dvc metrics diff main

dag:  ## Print the pipeline graph
	dvc dag

evaluate:  ## Evaluate the saved artifact on the held-out test split
	$(PYTHON) -m src.models.evaluate

serve:  ## Run the API locally with autoreload
	$(PYTHON) -m uvicorn src.api.main:app --reload

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format .

docker-build:  ## Build the Docker image (trains the model during the build)
	docker build -t $(IMAGE) .

docker-run:  ## Run the built image on port 8000
	docker run --rm -p 8000:8000 $(IMAGE)

docker-up:  ## Build and start via docker compose
	docker compose up --build

clean:  ## Remove generated data, artifacts and caches
	rm -f src/data/raw/*.csv src/data/processed/*.csv
	rm -f src/models/artifacts/*.joblib src/models/artifacts/*.json
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# --------------------------------------------------------------------------- #
# Postgres and the live demo
# --------------------------------------------------------------------------- #

PG_URL ?= postgresql+psycopg://subscriber:subscriber@localhost:5432/warehouse

warehouse-postgres:  ## Copy the SQLite warehouse into the compose Postgres (82.8M rows)
	docker compose up -d postgres
	$(PYTHON) -m src.warehouse.to_postgres

# The drift step of the nightly pipeline, run against Postgres, without
# retraining. Feeds Grafana's drift panels. CUTOFF is the last month of data.
monitor:  ## Check drift on the Postgres warehouse and write the pipeline report
	SDD_DATABASE_URL=$(PG_URL) SDD_MAX_SUBSCRIBERS=$(or $(N),35000) \
		$(PYTHON) -m src.orchestration.flows monitor --cutoff $(or $(CUTOFF),2017-02-28)

demo-prepare:  ## Once, the day before: build the image and write the drift report
	docker compose build subscriber-api
	$(MAKE) monitor

demo:  ## Start everything for a live demo, warm it up, print the URLs
	$(PYTHON) -m src.demo up

demo-check:  ## The pre-demo checklist: every service, the model, the presets, the rows
	$(PYTHON) -m src.demo check

demo-down:  ## Stop the demo stack (keeps every volume and all data)
	$(PYTHON) -m src.demo down
