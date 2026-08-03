"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes_batch import router as batch_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_health import router as health_router
from app.api.routes_inference import router as inference_router
from app.api.routes_llm import router as llm_router
from app.api.routes_observability import router as observability_router
from app.api.routes_release import router as release_router
from app.api.routes_slo import router as slo_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.torch_runtime import configure_torch_runtime
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.inference.cache import get_model_cached

logger = get_logger(__name__)
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_torch_runtime(settings)
    init_db()
    if settings.preload_prod_models:
        _preload_prod_models()
    logger.info("app_started", version=settings.app_version)
    yield
    logger.info("app_shutdown")


def _preload_prod_models() -> None:
    """Warm the in-process model cache with current production models."""
    session = get_session()
    try:
        prod_models = [m for m in repo.list_models(session) if m.status == "prod"]
        for mv in prod_models:
            try:
                get_model_cached(
                    mv.model_name,
                    mv.model_version,
                    mv.artifact_path,
                    architecture=mv.architecture,
                )
                logger.info(
                    "prod_model_preloaded",
                    model_name=mv.model_name,
                    model_version=mv.model_version,
                    architecture=mv.architecture,
                )
            except Exception as exc:
                logger.warning(
                    "prod_model_preload_failed",
                    model_name=mv.model_name,
                    model_version=mv.model_version,
                    error=str(exc),
                )
    finally:
        session.close()


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="ML Inference & Evaluation Platform",
        version=get_settings().app_version,
        lifespan=lifespan,
    )

    app.add_middleware(RequestIdMiddleware)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    app.include_router(health_router)
    app.include_router(inference_router)
    app.include_router(batch_router)
    app.include_router(slo_router)
    app.include_router(llm_router)
    app.include_router(observability_router)
    app.include_router(release_router)
    app.include_router(dashboard_router)

    return app


app = create_app()
