# syntax=docker/dockerfile:1
#
# Multi-stage build.
#
#   builder  installs dependencies and trains the model, so the image ships
#            with a ready-to-serve artifact and `docker run` needs no volume.
#   runtime  carries only the installed packages, the source and the artifact.
#
# Build:  docker build -t subscriber-dropout-api .
# Run:    docker run -p 8000:8000 subscriber-dropout-api

# --------------------------------------------------------------------------- #
# Stage 1: build dependencies and train the model
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency layer first so it is cached across source-only changes.
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

COPY src/ ./src/

# Make the --prefix install importable, then generate the dataset and train.
# This bakes src/models/artifacts/model.joblib into the image.
ENV PATH="/install/bin:${PATH}" \
    PYTHONPATH="/app:/install/lib/python3.11/site-packages"
RUN python -m src.models.train

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SDD_API_HOST=0.0.0.0 \
    SDD_API_PORT=8000

WORKDIR /app

# Installed site-packages and console scripts from the builder.
COPY --from=builder /install /usr/local

# Source plus the artifacts produced during the build.
COPY --from=builder /app/src /app/src
COPY pyproject.toml README.md ./

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
