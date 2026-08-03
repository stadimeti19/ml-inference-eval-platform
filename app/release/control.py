"""Dataset, evaluation, policy, deployment, and canary release control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db import repositories as repo
from app.db.models import Deployment, EvaluationReport

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "registered": {"evaluated", "failed"},
    "evaluated": {"approved", "failed"},
    "approved": {"shadow", "canary", "failed"},
    "shadow": {"canary", "failed"},
    "canary": {"production", "rolled_back", "failed"},
    "production": set(),
    "failed": set(),
    "rolled_back": set(),
}


@dataclass(frozen=True)
class PolicyDecision:
    passed: bool
    checks: list[dict[str, Any]]


def create_evaluation_report(
    session: Session,
    *,
    model_name: str,
    model_version: str,
    dataset_version_id: str,
    metrics: dict[str, float],
    config: dict[str, Any] | None = None,
) -> EvaluationReport:
    if (
        repo.get_model(session, model_name=model_name, model_version=model_version)
        is None
    ):
        raise ValueError(f"Model {model_name}@{model_version} is not registered")
    if repo.get_dataset_version(session, dataset_version_id) is None:
        raise ValueError(f"Dataset version {dataset_version_id} does not exist")
    canonical = {
        "model_name": model_name,
        "model_version": model_version,
        "dataset_version_id": dataset_version_id,
        "metrics": metrics,
        "config": config or {},
    }
    content_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    existing = (
        session.query(EvaluationReport).filter_by(content_hash=content_hash).first()
    )
    if existing:
        return existing
    run = repo.create_evaluation_run(
        session,
        model_name=model_name,
        model_version=model_version,
        dataset_version_id=dataset_version_id,
        config=config or {},
    )
    report = repo.save_evaluation_report(
        session,
        model_name=model_name,
        model_version=model_version,
        evaluation_run_id=run.id,
        dataset_version_id=dataset_version_id,
        metrics=metrics,
        config=config or {},
        content_hash=content_hash,
    )
    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.commit()
    model = repo.get_model(session, model_name=model_name, model_version=model_version)
    if model:
        model.metrics = json.dumps(metrics)
        session.commit()
    return report


def evaluate_policy(
    *,
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    constraints: dict[str, Any],
) -> PolicyDecision:
    checks: list[dict[str, Any]] = []
    for metric, rules in constraints.items():
        candidate = candidate_metrics.get(metric)
        baseline = baseline_metrics.get(metric)
        if candidate is None:
            checks.append(
                {
                    "metric": metric,
                    "rule": "present",
                    "passed": False,
                    "reason": "candidate metric missing",
                }
            )
            continue
        candidate_value = float(candidate)
        for rule, threshold in rules.items():
            passed = False
            actual: float | None = candidate_value
            if rule == "min":
                passed = candidate_value >= float(threshold)
            elif rule == "max":
                passed = candidate_value <= float(threshold)
            elif rule == "max_drop":
                actual = (
                    None if baseline is None else float(baseline) - float(candidate)
                )
                passed = actual is not None and actual <= float(threshold)
            elif rule == "max_increase_pct":
                actual = (
                    None
                    if baseline in (None, 0)
                    else ((float(candidate) - float(baseline)) / float(baseline)) * 100
                )
                passed = actual is not None and actual <= float(threshold)
            else:
                raise ValueError(f"Unsupported gate rule: {metric}.{rule}")
            checks.append(
                {
                    "metric": metric,
                    "rule": rule,
                    "threshold": threshold,
                    "candidate": candidate,
                    "baseline": baseline,
                    "actual": round(actual, 6) if actual is not None else None,
                    "passed": passed,
                }
            )
    return PolicyDecision(
        passed=bool(checks) and all(check["passed"] for check in checks), checks=checks
    )


def create_deployment(
    session: Session,
    *,
    model_name: str,
    candidate_version: str,
    min_requests: int = 20,
    max_error_rate: float = 0.05,
    max_avg_latency_ms: float | None = None,
) -> Deployment:
    existing = repo.get_deployment(session, model_name)
    if existing and existing.state not in {"production", "failed", "rolled_back"}:
        raise ValueError(f"An active deployment already exists for {model_name}")
    candidate = repo.get_model(
        session, model_name=model_name, model_version=candidate_version
    )
    if candidate is None:
        raise ValueError(f"Candidate {model_name}@{candidate_version} does not exist")
    baseline = repo.get_prod_model(session, model_name=model_name)
    deployment = existing or Deployment(
        model_name=model_name, candidate_version=candidate_version
    )
    deployment.baseline_version = baseline.model_version if baseline else None
    deployment.candidate_version = candidate_version
    deployment.state = "registered"
    deployment.traffic_percentage = 0.0
    deployment.min_requests = min_requests
    deployment.max_error_rate = max_error_rate
    deployment.max_avg_latency_ms = max_avg_latency_ms
    deployment.request_count = 0
    deployment.error_count = 0
    deployment.latency_sum_ms = 0.0
    deployment.last_reason = None
    if existing is None:
        session.add(deployment)
    session.commit()
    session.refresh(deployment)
    return deployment


def transition_deployment(
    session: Session,
    *,
    deployment: Deployment,
    target_state: str,
    reason: str,
    traffic_percentage: float | None = None,
) -> Deployment:
    if target_state not in ALLOWED_TRANSITIONS.get(deployment.state, set()):
        raise ValueError(
            f"Invalid deployment transition: {deployment.state} -> {target_state}"
        )
    if target_state == "canary":
        percentage = 10.0 if traffic_percentage is None else traffic_percentage
        if not 0 < percentage < 100:
            raise ValueError("Canary traffic percentage must be between 0 and 100")
        deployment.traffic_percentage = percentage
        deployment.request_count = 0
        deployment.error_count = 0
        deployment.latency_sum_ms = 0.0
    elif target_state in {"failed", "rolled_back"}:
        deployment.traffic_percentage = 0.0
    elif target_state == "production":
        if deployment.request_count < deployment.min_requests:
            raise ValueError(
                f"Canary requires {deployment.min_requests} requests before production"
            )
        error_rate = deployment.error_count / deployment.request_count
        average_latency = deployment.latency_sum_ms / deployment.request_count
        if error_rate > deployment.max_error_rate:
            raise ValueError("Canary error rate exceeds the deployment threshold")
        if (
            deployment.max_avg_latency_ms is not None
            and average_latency > deployment.max_avg_latency_ms
        ):
            raise ValueError("Canary average latency exceeds the deployment threshold")
        repo.promote_model(
            session,
            model_name=deployment.model_name,
            model_version=deployment.candidate_version,
        )
        deployment.traffic_percentage = 100.0
    candidate = repo.get_model(
        session,
        model_name=deployment.model_name,
        model_version=deployment.candidate_version,
    )
    if candidate and target_state != "production":
        candidate.status = target_state
    previous = deployment.state
    deployment.state = target_state
    deployment.last_reason = reason
    session.commit()
    session.refresh(deployment)
    repo.create_deployment_event(
        session,
        model_name=deployment.model_name,
        version=deployment.candidate_version,
        previous_status=previous,
        new_status=target_state,
        event_type=f"deployment_{target_state}",
        reason=reason,
    )
    return deployment


def should_route_to_canary(deployment: Deployment | None, routing_key: str) -> bool:
    if deployment is None or deployment.state != "canary":
        return False
    bucket = int(hashlib.sha256(routing_key.encode()).hexdigest()[:8], 16) % 10_000
    return bucket < int(deployment.traffic_percentage * 100)


def record_canary_result(
    session: Session, *, deployment: Deployment, latency_ms: float, error: bool
) -> bool:
    """Persist canary health and return True when an automatic rollback occurs."""
    deployment.request_count += 1
    deployment.error_count += int(error)
    deployment.latency_sum_ms += latency_ms
    session.commit()
    if deployment.request_count < deployment.min_requests:
        return False
    error_rate = deployment.error_count / deployment.request_count
    average_latency = deployment.latency_sum_ms / deployment.request_count
    reasons = []
    if error_rate > deployment.max_error_rate:
        reasons.append(
            f"error rate {error_rate:.3f} exceeded {deployment.max_error_rate:.3f}"
        )
    if (
        deployment.max_avg_latency_ms is not None
        and average_latency > deployment.max_avg_latency_ms
    ):
        reasons.append(
            f"average latency {average_latency:.3f}ms exceeded {deployment.max_avg_latency_ms:.3f}ms"
        )
    if not reasons:
        return False
    transition_deployment(
        session,
        deployment=deployment,
        target_state="rolled_back",
        reason="Automatic rollback: " + "; ".join(reasons),
    )
    return True
