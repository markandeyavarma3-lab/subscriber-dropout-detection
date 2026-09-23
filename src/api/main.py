"""FastAPI application exposing the subscriber dropout model.

Run locally with::

    uvicorn src.api.main:app --reload

Interactive documentation is served at ``/docs`` (Swagger UI) and ``/redoc``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api import overview, service
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriftRequest,
    DriftResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    ShadowResponse,
    SubscriberFeaturesRequest,
)
from src.config import settings
from src.live import control as live

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_UNAVAILABLE_DETAIL = (
    "Model artifact is not available. Train one with `python -m src.models.train`."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup.

    A missing artifact is logged rather than raised: the container still starts
    and answers ``/health``, while ``/predict`` reports 503 until a model is
    present.  That keeps a liveness probe from crash-looping the deployment.
    """
    try:
        service.load_model()
    except service.ModelNotLoadedError as exc:
        logger.warning("Starting without a model: %s", exc)
    yield
    service.reset_model()


app = FastAPI(
    title=settings.API_SETTINGS.title,
    description=settings.API_SETTINGS.description,
    version=settings.API_SETTINGS.version,
    lifespan=lifespan,
)


@app.exception_handler(service.ModelNotLoadedError)
async def _model_not_loaded_handler(_request, exc: service.ModelNotLoadedError) -> JSONResponse:
    """Translate a missing artifact into a 503 rather than a 500."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
    )


if settings.API_STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=settings.API_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the browser dashboard.

    Plain HTML/CSS/JS with no build step and no external requests: the page
    calls this same service's ``/predict``, ``/ready`` and ``/model-info``, so
    the UI can never drift from the contract the API actually serves.
    """
    if not settings.DASHBOARD_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard assets are not installed.",
        )
    return FileResponse(settings.DASHBOARD_PATH, media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness probe: confirms the process is up."""
    return HealthResponse(status="ok")


@app.get(
    "/ready", response_model=ReadinessResponse, tags=["operations"],
    responses={503: {"model": ReadinessResponse, "description": "No model is loaded"}},
)
def ready(response: Response) -> ReadinessResponse:
    """Readiness probe: confirms the model artifact is loaded and servable.

    503 when it isn't. Probes read only the status code: a 200 that says
    "degraded" in its body sent traffic to a pod that could not predict.
    """
    loaded = service.is_model_loaded()
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        detail=None if loaded else MODEL_UNAVAILABLE_DETAIL,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["operations"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the artifact currently being served."""
    try:
        return ModelInfoResponse(**service.model_info())
    except service.ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(request: SubscriberFeaturesRequest) -> PredictionResponse:
    """Predict the dropout risk for a single subscriber.

    Returns the probability, the label at the model's decision threshold, a
    coarse risk band, and a rule-based explanation of the drivers.
    """
    try:
        result = service.predict_one(request.to_features())
    except service.ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to 1000 subscribers in a single call."""
    try:
        results = service.predict_batch(
            [subscriber.to_features() for subscriber in request.subscribers]
        )
    except service.ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    predictions = [PredictionResponse(**result) for result in results]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
def metrics() -> MetricsResponse:
    """Live statistics for the predictions this process has served.

    Always 200, even with no model loaded: a monitoring endpoint that fails
    when the thing it monitors is unhealthy is worse than useless.
    """
    return MetricsResponse(**service.live_metrics())


@app.get("/overview", tags=["monitoring"])
def project_overview() -> dict:
    """Everything the dashboard's Overview page shows, from the live system.

    Warehouse row counts, the served model's own evaluation report, and the
    last drift check. Always 200: a section that can't be read says so with
    ``available: false`` rather than failing the page.
    """
    return overview.build_overview()


@app.get("/live/status", tags=["live replay"])
def live_status() -> dict:
    """Where the live replay is, stage by stage, and which model is serving."""
    return live.sync()


@app.post("/live/run", tags=["live replay"])
def live_run() -> dict:
    """Replay the next month of real data through the whole pipeline.

    Runs in its own process; poll ``/live/status``. If the new model beats the
    replay's champion at the gate, it starts serving immediately.
    """
    try:
        return live.start()
    except (live.ReplayBusy, live.ReplayFinished) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/live/reset", tags=["live replay"])
def live_reset() -> dict:
    """Stop the replay and put the tested production model back."""
    return live.restore()


@app.get("/monitoring/shadow", response_model=ShadowResponse, tags=["monitoring"])
def shadow() -> ShadowResponse:
    """Report what shadow traffic says about promoting the challenger.

    Deliberately returns no accuracy verdict: shadow traffic has no labels, so
    it can show how differently the two models behave but not which is right.
    """
    return ShadowResponse(**service.shadow_report())


@app.get("/metrics/prometheus", tags=["monitoring"], include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus exposition for serving and last-pipeline-run metrics.

    Deliberately not at ``/metrics``: that path already serves a documented
    JSON contract. Point Prometheus here with ``metrics_path``.
    """
    from src.monitoring import prometheus

    body, content_type = prometheus.render()
    return Response(content=body, media_type=content_type)


@app.post("/monitoring/drift", response_model=DriftResponse, tags=["monitoring"])
def drift(request: DriftRequest) -> DriftResponse:
    """Score a sample of live subscribers against the training distribution.

    Reports a Population Stability Index per feature plus one for the model's
    own output, so a shift can be traced to the inputs that caused it.
    """
    try:
        report = service.drift_report(
            [subscriber.to_features() for subscriber in request.subscribers]
        )
    except service.ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except service.DriftBaselineUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return DriftResponse(**report)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_SETTINGS.host,
        port=settings.API_SETTINGS.port,
        reload=True,
    )
