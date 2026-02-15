"""Health and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Liveness / readiness probe."""
    return {"status": "ok", "version": get_settings().app_version}


@router.get("/metrics")
def metrics() -> Response:
    """Prometheus text exposition endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
