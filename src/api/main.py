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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api import service
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    SubscriberFeaturesRequest,
)
from src.config import settings

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


@app.get("/ready", response_model=ReadinessResponse, tags=["operations"])
def ready() -> ReadinessResponse:
    """Readiness probe: confirms the model artifact is loaded and servable."""
    loaded = service.is_model_loaded()
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


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_SETTINGS.host,
        port=settings.API_SETTINGS.port,
        reload=True,
    )
