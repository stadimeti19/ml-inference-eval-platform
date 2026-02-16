"""Data-access functions for all ORM models."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import BatchJob, GateResult, ModelVersion, ShadowResult, SloPolicy


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


# ---------------------------------------------------------------------------
# Shadow Results
# ---------------------------------------------------------------------------

def save_shadow_result(
    session: Session,
    *,
    model_name: str,
    prod_version: str,
    shadow_version: str,
    prod_prediction: int,
    shadow_prediction: int,
    prod_latency_ms: float,
    shadow_latency_ms: float,
) -> ShadowResult:
    """Record a single shadow comparison between prod and candidate."""
    sr = ShadowResult(
        model_name=model_name,
        prod_version=prod_version,
        shadow_version=shadow_version,
        prod_prediction=prod_prediction,
        shadow_prediction=shadow_prediction,
        agreed=(prod_prediction == shadow_prediction),
        prod_latency_ms=prod_latency_ms,
        shadow_latency_ms=shadow_latency_ms,
    )
    session.add(sr)
    session.commit()
    session.refresh(sr)
    return sr


def get_shadow_summary(
    session: Session,
    *,
    model_name: str,
    shadow_version: str,
    prod_version: str | None = None,
) -> dict[str, Any]:
    """Aggregate shadow results into a summary report.

    Returns agreement rate, latency comparison, and sample count.
    """
    from sqlalchemy import func as sa_func

    q = session.query(ShadowResult).filter_by(
        model_name=model_name, shadow_version=shadow_version
    )
    if prod_version:
        q = q.filter_by(prod_version=prod_version)

    total = q.count()
    if total == 0:
        return {"total_comparisons": 0}

    agreed = q.filter_by(agreed=True).count()

    stats = q.with_entities(
        sa_func.avg(ShadowResult.prod_latency_ms).label("avg_prod_ms"),
        sa_func.avg(ShadowResult.shadow_latency_ms).label("avg_shadow_ms"),
        sa_func.max(ShadowResult.shadow_latency_ms).label("max_shadow_ms"),
    ).one()

    return {
        "model_name": model_name,
        "shadow_version": shadow_version,
        "prod_version": prod_version,
        "total_comparisons": total,
        "agreements": agreed,
        "disagreements": total - agreed,
        "agreement_rate": round(agreed / total, 4),
        "disagreement_rate": round((total - agreed) / total, 4),
        "avg_prod_latency_ms": round(float(stats.avg_prod_ms), 3),
        "avg_shadow_latency_ms": round(float(stats.avg_shadow_ms), 3),
        "max_shadow_latency_ms": round(float(stats.max_shadow_ms), 3),
    }


# ---------------------------------------------------------------------------
# SLO Policies
# ---------------------------------------------------------------------------

def create_slo_policy(
    session: Session,
    *,
    name: str,
    model_name: str,
    constraints: dict[str, Any],
) -> SloPolicy:
    """Create a named SLO policy with absolute constraints."""
    policy = SloPolicy(
        name=name,
        model_name=model_name,
        constraints=json.dumps(constraints),
    )
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def get_slo_policy(session: Session, *, name: str) -> SloPolicy | None:
    return session.query(SloPolicy).filter_by(name=name).first()


def get_slo_policies_for_model(
    session: Session, *, model_name: str
) -> list[SloPolicy]:
    return list(
        session.query(SloPolicy)
        .filter_by(model_name=model_name)
        .order_by(desc(SloPolicy.created_at))
        .all()
    )


def delete_slo_policy(session: Session, *, name: str) -> bool:
    """Delete a policy by name. Returns True if deleted."""
    policy = session.query(SloPolicy).filter_by(name=name).first()
    if policy is None:
        return False
    session.delete(policy)
    session.commit()
    return True
