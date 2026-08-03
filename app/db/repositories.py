"""Data-access functions for all ORM models."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.db.models import (
    BatchJob,
    DatasetVersion,
    Deployment,
    DeploymentEvent,
    EvaluationReport,
    EvaluationRun,
    GatePolicy,
    GateResult,
    ModelVersion,
    ShadowResult,
    SloPolicy,
)

# ---------------------------------------------------------------------------
# Release control
# ---------------------------------------------------------------------------


def create_dataset_version(
    session: Session,
    *,
    name: str,
    version: str,
    uri: str,
    checksum: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetVersion:
    dataset = DatasetVersion(
        name=name,
        version=version,
        uri=uri,
        checksum=checksum,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset


def get_dataset_version(session: Session, dataset_id: str) -> DatasetVersion | None:
    return session.query(DatasetVersion).filter_by(id=dataset_id).first()


def list_dataset_versions(session: Session) -> list[DatasetVersion]:
    return list(
        session.query(DatasetVersion).order_by(desc(DatasetVersion.created_at)).all()
    )


def save_evaluation_report(
    session: Session,
    *,
    model_name: str,
    model_version: str,
    evaluation_run_id: str,
    dataset_version_id: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    content_hash: str,
) -> EvaluationReport:
    report = EvaluationReport(
        model_name=model_name,
        model_version=model_version,
        evaluation_run_id=evaluation_run_id,
        dataset_version_id=dataset_version_id,
        metrics_json=json.dumps(metrics, sort_keys=True),
        config_json=json.dumps(config, sort_keys=True),
        content_hash=content_hash,
    )
    session.add(report)
    session.commit()
    session.refresh(report)
    return report


def create_evaluation_run(
    session: Session,
    *,
    model_name: str,
    model_version: str,
    dataset_version_id: str,
    config: dict[str, Any],
) -> EvaluationRun:
    run = EvaluationRun(
        model_name=model_name,
        model_version=model_version,
        dataset_version_id=dataset_version_id,
        status="running",
        config_json=json.dumps(config, sort_keys=True),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def list_evaluation_runs(
    session: Session, *, model_name: str | None = None
) -> list[EvaluationRun]:
    query = session.query(EvaluationRun).order_by(desc(EvaluationRun.started_at))
    if model_name:
        query = query.filter_by(model_name=model_name)
    return list(query.all())


def get_evaluation_report(session: Session, report_id: str) -> EvaluationReport | None:
    return session.query(EvaluationReport).filter_by(id=report_id).first()


def list_evaluation_reports(
    session: Session, *, model_name: str | None = None, model_version: str | None = None
) -> list[EvaluationReport]:
    query = session.query(EvaluationReport).order_by(desc(EvaluationReport.created_at))
    if model_name:
        query = query.filter_by(model_name=model_name)
    if model_version:
        query = query.filter_by(model_version=model_version)
    return list(query.all())


def create_gate_policy(
    session: Session, *, name: str, model_name: str, constraints: dict[str, Any]
) -> GatePolicy:
    policy = GatePolicy(
        name=name, model_name=model_name, constraints_json=json.dumps(constraints)
    )
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def get_gate_policy(session: Session, name: str) -> GatePolicy | None:
    return session.query(GatePolicy).filter_by(name=name).first()


def list_gate_policies(
    session: Session, model_name: str | None = None
) -> list[GatePolicy]:
    query = session.query(GatePolicy).order_by(desc(GatePolicy.created_at))
    if model_name:
        query = query.filter_by(model_name=model_name)
    return list(query.all())


def get_deployment(
    session: Session, model_name: str, *, for_update: bool = False
) -> Deployment | None:
    query = session.query(Deployment).filter_by(model_name=model_name)
    if for_update:
        query = query.with_for_update()
    return query.first()


def list_deployments(session: Session) -> list[Deployment]:
    return list(session.query(Deployment).order_by(desc(Deployment.updated_at)).all())


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
    session.query(ModelVersion).filter_by(model_name=model_name, status="prod").update(
        {"status": "staging"}
    )

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
    session.query(ModelVersion).filter_by(model_name=model_name, status="prod").update(
        {"status": "staging"}
    )

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
# Deployment Events
# ---------------------------------------------------------------------------


def create_deployment_event(
    session: Session,
    *,
    model_name: str,
    version: str,
    previous_status: str | None,
    new_status: str | None,
    event_type: str,
    reason: str | None = None,
) -> DeploymentEvent:
    """Append an audit event for a model lifecycle or gate action."""
    event = DeploymentEvent(
        model_name=model_name,
        version=version,
        previous_status=previous_status,
        new_status=new_status,
        event_type=event_type,
        reason=reason,
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return event


def list_deployment_events(
    session: Session,
    *,
    model_name: str | None = None,
    limit: int = 100,
) -> list[DeploymentEvent]:
    q = session.query(DeploymentEvent).order_by(desc(DeploymentEvent.created_at))
    if model_name:
        q = q.filter_by(model_name=model_name)
    return list(q.limit(limit).all())


def get_previous_prod_version_from_events(
    session: Session,
    *,
    model_name: str,
    current_version: str | None = None,
) -> str | None:
    """Return the latest earlier version that was promoted to prod."""
    q = (
        session.query(DeploymentEvent)
        .filter(
            DeploymentEvent.model_name == model_name,
            DeploymentEvent.new_status == "prod",
            DeploymentEvent.event_type.in_(["promote", "rollback"]),
        )
        .order_by(desc(DeploymentEvent.created_at))
    )
    for event in q.all():
        if current_version is None or event.version != current_version:
            return event.version
    return None


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

    Returns agreement rate, latency comparison, sample count, and examples
    of prod/candidate disagreement.
    """
    q = session.query(ShadowResult).filter_by(
        model_name=model_name, shadow_version=shadow_version
    )
    if prod_version:
        q = q.filter_by(prod_version=prod_version)

    rows = q.order_by(desc(ShadowResult.created_at)).all()
    total = len(rows)
    if total == 0:
        return {"total_comparisons": 0}

    agreed = sum(1 for r in rows if r.agreed)
    prod_latencies = [float(r.prod_latency_ms) for r in rows]
    shadow_latencies = [float(r.shadow_latency_ms) for r in rows]
    latency_deltas = [
        s - p for p, s in zip(prod_latencies, shadow_latencies, strict=True)
    ]
    avg_prod_ms = sum(prod_latencies) / total
    avg_shadow_ms = sum(shadow_latencies) / total
    avg_delta_ms = sum(latency_deltas) / total
    p95_delta_ms = _percentile(latency_deltas, 95)
    disagreements = [r for r in rows if not r.agreed][:10]
    faster_count = sum(1 for d in latency_deltas if d < 0)
    slower_count = sum(1 for d in latency_deltas if d > 0)

    return {
        "model_name": model_name,
        "shadow_version": shadow_version,
        "prod_version": prod_version,
        "total_comparisons": total,
        "agreements": agreed,
        "disagreements": total - agreed,
        "agreement_rate": round(agreed / total, 4),
        "disagreement_rate": round((total - agreed) / total, 4),
        "avg_prod_latency_ms": round(avg_prod_ms, 3),
        "avg_shadow_latency_ms": round(avg_shadow_ms, 3),
        "avg_latency_delta_ms": round(avg_delta_ms, 3),
        "p95_latency_delta_ms": round(p95_delta_ms, 3),
        "max_shadow_latency_ms": round(max(shadow_latencies), 3),
        "candidate_latency_summary": {
            "faster_count": faster_count,
            "slower_count": slower_count,
            "same_count": total - faster_count - slower_count,
            "summary": "candidate_faster"
            if faster_count > slower_count
            else "candidate_slower"
            if slower_count > faster_count
            else "mixed",
        },
        "disagreement_examples": [
            {
                "id": r.id,
                "prod_version": r.prod_version,
                "shadow_version": r.shadow_version,
                "prod_prediction": r.prod_prediction,
                "shadow_prediction": r.shadow_prediction,
                "prod_latency_ms": round(float(r.prod_latency_ms), 3),
                "shadow_latency_ms": round(float(r.shadow_latency_ms), 3),
                "created_at": str(r.created_at),
            }
            for r in disagreements
        ],
    }


def list_shadow_summaries(session: Session) -> list[dict[str, Any]]:
    """Return summaries for every observed prod/shadow pair."""
    pairs = (
        session.query(
            ShadowResult.model_name,
            ShadowResult.prod_version,
            ShadowResult.shadow_version,
        )
        .group_by(
            ShadowResult.model_name,
            ShadowResult.prod_version,
            ShadowResult.shadow_version,
        )
        .all()
    )
    return [
        get_shadow_summary(
            session,
            model_name=p.model_name,
            prod_version=p.prod_version,
            shadow_version=p.shadow_version,
        )
        for p in pairs
    ]


def delete_shadow_results(
    session: Session,
    *,
    model_name: str | None = None,
    shadow_version: str | None = None,
    prod_version: str | None = None,
) -> int:
    """Delete shadow comparison rows matching optional filters."""
    q = session.query(ShadowResult)
    if model_name:
        q = q.filter_by(model_name=model_name)
    if shadow_version:
        q = q.filter_by(shadow_version=shadow_version)
    if prod_version:
        q = q.filter_by(prod_version=prod_version)
    count = q.count()
    q.delete(synchronize_session=False)
    session.commit()
    return int(count)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    weight = k - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


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


def get_slo_policies_for_model(session: Session, *, model_name: str) -> list[SloPolicy]:
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
