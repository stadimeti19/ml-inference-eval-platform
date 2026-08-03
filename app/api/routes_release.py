"""Release-control API for datasets, evaluations, policies, and deployments."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db import repositories as repo
from app.db.models import Deployment, EvaluationReport
from app.db.session import get_session
from app.release.control import (
    create_deployment,
    create_evaluation_report,
    evaluate_policy,
    transition_deployment,
)

router = APIRouter(prefix="/release", tags=["release-control"])


class DatasetRequest(BaseModel):
    name: str
    version: str
    uri: str
    checksum: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    model_name: str
    model_version: str
    dataset_version_id: str
    metrics: dict[str, float]
    config: dict[str, Any] = Field(default_factory=dict)


class GatePolicyRequest(BaseModel):
    name: str
    model_name: str
    constraints: dict[str, dict[str, float]]


class DeploymentRequest(BaseModel):
    model_name: str
    candidate_version: str
    min_requests: int = Field(20, ge=1)
    max_error_rate: float = Field(0.05, ge=0, le=1)
    max_avg_latency_ms: float | None = Field(None, gt=0)


class EvaluateDeploymentRequest(BaseModel):
    policy_name: str
    candidate_report_id: str
    baseline_report_id: str


class TransitionRequest(BaseModel):
    target_state: str
    reason: str
    traffic_percentage: float | None = None


def _report_payload(report: EvaluationReport) -> dict[str, Any]:
    return {
        "id": report.id,
        "model_name": report.model_name,
        "model_version": report.model_version,
        "evaluation_run_id": report.evaluation_run_id,
        "dataset_version_id": report.dataset_version_id,
        "metrics": json.loads(report.metrics_json),
        "config": json.loads(report.config_json),
        "content_hash": report.content_hash,
        "created_at": str(report.created_at),
    }


def deployment_payload(deployment: Deployment) -> dict[str, Any]:
    requests = deployment.request_count
    return {
        "id": deployment.id,
        "model_name": deployment.model_name,
        "baseline_version": deployment.baseline_version,
        "candidate_version": deployment.candidate_version,
        "state": deployment.state,
        "traffic_percentage": deployment.traffic_percentage,
        "request_count": requests,
        "error_count": deployment.error_count,
        "error_rate": round(deployment.error_count / requests, 6) if requests else 0.0,
        "average_latency_ms": round(deployment.latency_sum_ms / requests, 3)
        if requests
        else 0.0,
        "min_requests": deployment.min_requests,
        "max_error_rate": deployment.max_error_rate,
        "max_avg_latency_ms": deployment.max_avg_latency_ms,
        "last_reason": deployment.last_reason,
        "updated_at": str(deployment.updated_at),
    }


@router.post("/datasets", status_code=201)
def register_dataset(req: DatasetRequest) -> dict[str, Any]:
    session = get_session()
    try:
        dataset = repo.create_dataset_version(
            session,
            name=req.name,
            version=req.version,
            uri=req.uri,
            checksum=req.checksum,
            metadata=req.metadata,
        )
        return {
            "id": dataset.id,
            "name": dataset.name,
            "version": dataset.version,
            "uri": dataset.uri,
            "checksum": dataset.checksum,
        }
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/datasets")
def datasets() -> list[dict[str, Any]]:
    session = get_session()
    try:
        return [
            {
                "id": d.id,
                "name": d.name,
                "version": d.version,
                "uri": d.uri,
                "checksum": d.checksum,
                "metadata": json.loads(d.metadata_json) if d.metadata_json else {},
                "created_at": str(d.created_at),
            }
            for d in repo.list_dataset_versions(session)
        ]
    finally:
        session.close()


@router.post("/evaluations", status_code=201)
def save_evaluation(req: EvaluationRequest) -> dict[str, Any]:
    session = get_session()
    try:
        report = create_evaluation_report(session, **req.model_dump())
        return _report_payload(report)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/evaluations")
def evaluations(
    model_name: str | None = None, model_version: str | None = None
) -> list[dict[str, Any]]:
    session = get_session()
    try:
        return [
            _report_payload(report)
            for report in repo.list_evaluation_reports(
                session, model_name=model_name, model_version=model_version
            )
        ]
    finally:
        session.close()


@router.get("/evaluation-runs")
def evaluation_runs(model_name: str | None = None) -> list[dict[str, Any]]:
    session = get_session()
    try:
        return [
            {
                "id": run.id,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "dataset_version_id": run.dataset_version_id,
                "status": run.status,
                "config": json.loads(run.config_json),
                "started_at": str(run.started_at),
                "completed_at": str(run.completed_at) if run.completed_at else None,
            }
            for run in repo.list_evaluation_runs(session, model_name=model_name)
        ]
    finally:
        session.close()


@router.post("/gate-policies", status_code=201)
def save_gate_policy(req: GatePolicyRequest) -> dict[str, Any]:
    session = get_session()
    try:
        allowed_rules = {"min", "max", "max_drop", "max_increase_pct"}
        invalid = [
            f"{metric}.{rule}"
            for metric, rules in req.constraints.items()
            for rule in rules
            if rule not in allowed_rules
        ]
        if not req.constraints or invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid or empty constraints: {', '.join(invalid)}",
            )
        policy = repo.create_gate_policy(
            session,
            name=req.name,
            model_name=req.model_name,
            constraints=req.constraints,
        )
        return {
            "id": policy.id,
            "name": policy.name,
            "model_name": policy.model_name,
            "constraints": req.constraints,
        }
    except HTTPException:
        raise
    except Exception as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/gate-policies")
def gate_policies(model_name: str | None = None) -> list[dict[str, Any]]:
    session = get_session()
    try:
        return [
            {
                "id": p.id,
                "name": p.name,
                "model_name": p.model_name,
                "constraints": json.loads(p.constraints_json),
                "created_at": str(p.created_at),
            }
            for p in repo.list_gate_policies(session, model_name)
        ]
    finally:
        session.close()


@router.post("/deployments", status_code=201)
def start_deployment(req: DeploymentRequest) -> dict[str, Any]:
    session = get_session()
    try:
        return deployment_payload(create_deployment(session, **req.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        session.close()


@router.get("/deployments")
def deployments() -> list[dict[str, Any]]:
    session = get_session()
    try:
        return [
            deployment_payload(deployment)
            for deployment in repo.list_deployments(session)
        ]
    finally:
        session.close()


@router.post("/deployments/{model_name}/evaluate")
def evaluate_deployment(
    model_name: str, req: EvaluateDeploymentRequest
) -> dict[str, Any]:
    session = get_session()
    try:
        deployment = repo.get_deployment(session, model_name, for_update=True)
        policy = repo.get_gate_policy(session, req.policy_name)
        candidate = repo.get_evaluation_report(session, req.candidate_report_id)
        baseline = repo.get_evaluation_report(session, req.baseline_report_id)
        if not all((deployment, policy, candidate, baseline)):
            raise HTTPException(
                status_code=404,
                detail="Deployment, policy, or evaluation report not found",
            )
        assert deployment and policy and candidate and baseline
        if (
            policy.model_name != model_name
            or candidate.model_name != model_name
            or baseline.model_name != model_name
            or candidate.model_version != deployment.candidate_version
            or baseline.model_version != deployment.baseline_version
            or candidate.dataset_version_id != baseline.dataset_version_id
        ):
            raise HTTPException(
                status_code=422,
                detail="Policy or reports do not match the deployment and dataset",
            )
        decision = evaluate_policy(
            candidate_metrics=json.loads(candidate.metrics_json),
            baseline_metrics=json.loads(baseline.metrics_json),
            constraints=json.loads(policy.constraints_json),
        )
        transition_deployment(
            session,
            deployment=deployment,
            target_state="evaluated",
            reason=f"Evaluated with policy {policy.name}",
        )
        transition_deployment(
            session,
            deployment=deployment,
            target_state="approved" if decision.passed else "failed",
            reason="All policy checks passed"
            if decision.passed
            else "One or more policy checks failed",
        )
        repo.save_gate_result(
            session,
            model_name=model_name,
            candidate_version=candidate.model_version,
            baseline_version=baseline.model_version,
            passed=decision.passed,
            details={"policy": policy.name, "checks": decision.checks},
        )
        return {
            "passed": decision.passed,
            "checks": decision.checks,
            "deployment": deployment_payload(deployment),
        }
    finally:
        session.close()


@router.post("/deployments/{model_name}/transition")
def transition(model_name: str, req: TransitionRequest) -> dict[str, Any]:
    session = get_session()
    try:
        deployment = repo.get_deployment(session, model_name, for_update=True)
        if deployment is None:
            raise HTTPException(status_code=404, detail="Deployment not found")
        return deployment_payload(
            transition_deployment(
                session,
                deployment=deployment,
                target_state=req.target_state,
                reason=req.reason,
                traffic_percentage=req.traffic_percentage,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        session.close()
