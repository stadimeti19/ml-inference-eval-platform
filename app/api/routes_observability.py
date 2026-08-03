"""JSON observability and deployment operations endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import func

from app.core.runtime_metrics import get_inference_metrics
from app.db import repositories as repo
from app.db.models import BatchJob
from app.db.session import get_session
from app.inference.cache import get_cache_stats
from app.registry.manager import rollback_with_summary

router = APIRouter()


@router.get("/metrics/cache")
def cache_metrics() -> dict:
    return get_cache_stats()


@router.get("/metrics/inference")
def inference_metrics() -> dict:
    return get_inference_metrics()


@router.get("/metrics/batch")
def batch_metrics() -> dict:
    session = get_session()
    try:
        rows = (
            session.query(BatchJob.status, func.count(BatchJob.id))
            .group_by(BatchJob.status)
            .all()
        )
        by_status = {status: int(count) for status, count in rows}
        return {
            "total_jobs": sum(by_status.values()),
            "queued": by_status.get("queued", 0),
            "running": by_status.get("running", 0),
            "completed": by_status.get("succeeded", 0),
            "failed": by_status.get("failed", 0),
            "by_status": by_status,
        }
    finally:
        session.close()


@router.get("/metrics/platform")
def platform_metrics(model_name: str | None = Query(None)) -> dict:
    session = get_session()
    try:
        models = repo.list_models(session, model_name=model_name)
        prod_models = [m for m in models if m.status == "prod"]
        current_prod = prod_models[0] if prod_models else None
        return {
            "model_count": len(models),
            "current_prod_model": {
                "model_name": current_prod.model_name,
                "model_version": current_prod.model_version,
                "architecture": current_prod.architecture,
                "metrics": json.loads(current_prod.metrics)
                if current_prod.metrics
                else {},
            }
            if current_prod
            else None,
            "cache": get_cache_stats(),
            "inference": get_inference_metrics(),
            "batch": batch_metrics(),
        }
    finally:
        session.close()


@router.get("/deployments/events")
def deployment_events(
    model_name: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> list[dict]:
    session = get_session()
    try:
        events = repo.list_deployment_events(
            session, model_name=model_name, limit=limit
        )
        return [
            {
                "id": e.id,
                "model_name": e.model_name,
                "version": e.version,
                "previous_status": e.previous_status,
                "new_status": e.new_status,
                "event_type": e.event_type,
                "reason": e.reason,
                "created_at": str(e.created_at),
            }
            for e in events
        ]
    finally:
        session.close()


class RollbackResponse(BaseModel):
    model_name: str
    previous_prod_version: str | None
    new_prod_version: str | None
    rolled_back: bool
    reason: str


@router.post("/models/{model_name}/rollback", response_model=RollbackResponse)
def rollback_model(model_name: str) -> RollbackResponse:
    summary = rollback_with_summary(model_name)
    return RollbackResponse(
        model_name=summary.model_name,
        previous_prod_version=summary.previous_prod_version,
        new_prod_version=summary.new_prod_version,
        rolled_back=summary.rolled_back,
        reason=summary.reason,
    )
