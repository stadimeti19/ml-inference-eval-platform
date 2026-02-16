"""FastAPI application factory."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes_batch import router as batch_router
from app.api.routes_dashboard import router as dashboard_router
from app.api.routes_health import router as health_router
from app.api.routes_inference import router as inference_router
from app.api.routes_slo import router as slo_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.session import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db()
    logger.info("app_started", version=settings.app_version)
    yield
    logger.info("app_shutdown")


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

    app.include_router(health_router)
    app.include_router(inference_router)
    app.include_router(batch_router)
    app.include_router(slo_router)
    app.include_router(dashboard_router)

    return app


app = create_app()
