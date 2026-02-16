"""Data-access functions for all ORM models."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import BatchJob, GateResult, ModelVersion


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def register_model(
    session: Session,
    *,
    model_name: str,
    model_version: str,
    artifact_path: str,
    git_sha: str | None = None,
    tags: dict[str, Any] | None = None,
    status: str = "staging",
    metrics: dict[str, Any] | None = None,
    architecture: str = "default",
) -> ModelVersion:
    """Insert a new model version row."""
    mv = ModelVersion(
        model_name=model_name,
        model_version=model_version,
        artifact_path=artifact_path,
        git_sha=git_sha,
        tags=json.dumps(tags) if tags else None,
        status=status,
        metrics=json.dumps(metrics) if metrics else None,
        architecture=architecture,
    )
    session.add(mv)
    session.commit()
    session.refresh(mv)
    return mv


def get_model(
    session: Session, *, model_name: str, model_version: str
) -> ModelVersion | None:
    """Fetch a specific model version."""
    return (
        session.query(ModelVersion)
        .filter_by(model_name=model_name, model_version=model_version)
        .first()
    )


def get_prod_model(session: Session, *, model_name: str) -> ModelVersion | None:
    """Return the current production model for *model_name*."""
    return (
        session.query(ModelVersion)
        .filter_by(model_name=model_name, status="prod")
        .order_by(desc(ModelVersion.created_at))
        .first()
    )


def promote_model(
    session: Session, *, model_name: str, model_version: str
) -> ModelVersion | None:
    """Set *model_version* to prod and demote all other versions to staging."""
    # Demote existing prod
    session.query(ModelVersion).filter_by(
        model_name=model_name, status="prod"
    ).update({"status": "staging"})

    mv = get_model(session, model_name=model_name, model_version=model_version)
    if mv is None:
        session.rollback()
        return None
    mv.status = "prod"
    session.commit()
    session.refresh(mv)
    return mv


def rollback_model(session: Session, *, model_name: str) -> ModelVersion | None:
    """Revert to the previous prod version (most recent staging)."""
    # Demote current prod
    session.query(ModelVersion).filter_by(
        model_name=model_name, status="prod"
    ).update({"status": "staging"})

    # Pick the most recently created staging version
    prev = (
        session.query(ModelVersion)
        .filter_by(model_name=model_name, status="staging")
        .order_by(desc(ModelVersion.created_at))
        .first()
    )
    if prev is None:
        session.commit()
        return None
    prev.status = "prod"
    session.commit()
    session.refresh(prev)
    return prev


def list_models(
    session: Session, *, model_name: str | None = None
) -> list[ModelVersion]:
    """List model versions, optionally filtered by name."""
    q = session.query(ModelVersion).order_by(desc(ModelVersion.created_at))
    if model_name:
        q = q.filter_by(model_name=model_name)
    return list(q.all())


# ---------------------------------------------------------------------------
# Batch Jobs
# ---------------------------------------------------------------------------

def create_batch_job(
    session: Session,
    *,
    model_name: str,
    model_version: str,
    dataset_id: str,
    config: dict[str, Any] | None = None,
) -> BatchJob:
    job = BatchJob(
        model_name=model_name,
        model_version=model_version,
        dataset_id=dataset_id,
        config=json.dumps(config) if config else None,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def update_batch_job(
    session: Session,
    *,
    job_id: str,
    status: str | None = None,
    result_metrics: dict[str, Any] | None = None,
) -> BatchJob | None:
    job = session.query(BatchJob).filter_by(id=job_id).first()
    if job is None:
        return None
    if status is not None:
        job.status = status
    if result_metrics is not None:
        job.result_metrics = json.dumps(result_metrics)
    session.commit()
    session.refresh(job)
    return job


def get_batch_job(session: Session, *, job_id: str) -> BatchJob | None:
    return session.query(BatchJob).filter_by(id=job_id).first()


# ---------------------------------------------------------------------------
# Gate Results
# ---------------------------------------------------------------------------

def save_gate_result(
    session: Session,
    *,
    model_name: str,
    candidate_version: str,
    baseline_version: str,
    passed: bool,
    details: dict[str, Any] | None = None,
) -> GateResult:
    gr = GateResult(
        model_name=model_name,
        candidate_version=candidate_version,
        baseline_version=baseline_version,
        passed=passed,
        details=json.dumps(details) if details else None,
    )
    session.add(gr)
    session.commit()
    session.refresh(gr)
    return gr
