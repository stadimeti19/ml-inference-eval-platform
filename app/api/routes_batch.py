"""Batch inference endpoints."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import get_logger
from app.db import repositories as repo
from app.db.session import get_session

logger = get_logger(__name__)
router = APIRouter(prefix="/batch")


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------

class BatchSubmitRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    dataset_id: str = "mnist_1000"
    config: Optional[dict[str, Any]] = None


class BatchSubmitResponse(BaseModel):
    job_id: str
    status: str


class BatchStatusResponse(BaseModel):
    job_id: str
    status: str
    result_metrics: Optional[dict[str, Any]] = None


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.post("/submit", response_model=BatchSubmitResponse)
def submit_batch(req: BatchSubmitRequest) -> BatchSubmitResponse:
    """Enqueue a batch inference job."""
    session = get_session()
    try:
        # Resolve model version
        if req.model_version:
            mv = repo.get_model(
                session, model_name=req.model_name, model_version=req.model_version
            )
        else:
            mv = repo.get_prod_model(session, model_name=req.model_name)

        if mv is None:
            raise HTTPException(
                status_code=404,
                detail=f"No model found for {req.model_name}",
            )

        job = repo.create_batch_job(
            session,
            model_name=mv.model_name,
            model_version=mv.model_version,
            dataset_id=req.dataset_id,
            config=req.config,
        )

        # Attempt to enqueue via RQ; gracefully degrade if Redis unavailable
        try:
            from app.jobs.queue import get_queue
            q = get_queue()
            q.enqueue("app.jobs.tasks.run_batch_inference", job.id)
            logger.info("batch_job_enqueued", job_id=job.id)
        except Exception:
            logger.warning(
                "redis_unavailable_running_sync",
                job_id=job.id,
            )
            # Fallback: run synchronously
            from app.jobs.tasks import run_batch_inference
            run_batch_inference(job.id)

        return BatchSubmitResponse(job_id=job.id, status=job.status)
    finally:
        session.close()


@router.get("/status/{job_id}", response_model=BatchStatusResponse)
def batch_status(job_id: str) -> BatchStatusResponse:
    """Check the status of a batch job."""
    session = get_session()
    try:
        job = repo.get_batch_job(session, job_id=job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        metrics = json.loads(job.result_metrics) if job.result_metrics else None
        return BatchStatusResponse(
            job_id=job.id,
            status=job.status,
            result_metrics=metrics,
        )
    finally:
        session.close()
