"""Tests for Phase 2 release-control entities and rollout behavior."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.db import repositories as repo
from app.db.models import EvaluationReport
from app.db.session import get_session, init_db
from app.release.control import (
    create_deployment,
    create_evaluation_report,
    evaluate_policy,
    record_canary_result,
    should_route_to_canary,
    transition_deployment,
)


def _register_versions(session, sample_model: str) -> None:
    for version in ("v1", "v2"):
        repo.register_model(
            session,
            model_name="classifier",
            model_version=version,
            artifact_path=sample_model,
        )
    repo.promote_model(session, model_name="classifier", model_version="v1")


def test_evaluation_reports_are_idempotent_and_immutable(db_session, sample_model):
    _register_versions(db_session, sample_model)
    dataset = repo.create_dataset_version(
        db_session,
        name="holdout",
        version="2026-08-03",
        uri="s3://example/holdout.parquet",
        checksum="sha256:abc",
    )
    first = create_evaluation_report(
        db_session,
        model_name="classifier",
        model_version="v2",
        dataset_version_id=dataset.id,
        metrics={"accuracy": 0.98, "p95_ms": 12.0},
    )
    duplicate = create_evaluation_report(
        db_session,
        model_name="classifier",
        model_version="v2",
        dataset_version_id=dataset.id,
        metrics={"accuracy": 0.98, "p95_ms": 12.0},
    )

    assert duplicate.id == first.id
    assert first.evaluation_run_id
    first.metrics_json = json.dumps({"accuracy": 0.1})
    with pytest.raises(ValueError, match="immutable"):
        db_session.commit()


def test_configurable_policy_supports_absolute_and_relative_rules():
    decision = evaluate_policy(
        candidate_metrics={"accuracy": 0.975, "p95_ms": 11.0},
        baseline_metrics={"accuracy": 0.98, "p95_ms": 10.0},
        constraints={
            "accuracy": {"min": 0.97, "max_drop": 0.01},
            "p95_ms": {"max": 15.0, "max_increase_pct": 15.0},
        },
    )
    assert decision.passed is True
    assert len(decision.checks) == 4

    rejected = evaluate_policy(
        candidate_metrics={"accuracy": 0.90},
        baseline_metrics={"accuracy": 0.98},
        constraints={"accuracy": {"max_drop": 0.01}},
    )
    assert rejected.passed is False


def test_deployment_state_machine_rejects_invalid_transition(db_session, sample_model):
    _register_versions(db_session, sample_model)
    deployment = create_deployment(
        db_session,
        model_name="classifier",
        candidate_version="v2",
        min_requests=2,
    )

    with pytest.raises(ValueError, match="Invalid deployment transition"):
        transition_deployment(
            db_session,
            deployment=deployment,
            target_state="production",
            reason="skip all safety stages",
        )

    for state in ("evaluated", "approved", "shadow"):
        transition_deployment(
            db_session, deployment=deployment, target_state=state, reason="test"
        )
    transition_deployment(
        db_session,
        deployment=deployment,
        target_state="canary",
        reason="start rollout",
        traffic_percentage=25,
    )
    assert deployment.state == "canary"
    assert deployment.traffic_percentage == 25
    with pytest.raises(ValueError, match="requires 2 requests"):
        transition_deployment(
            db_session,
            deployment=deployment,
            target_state="production",
            reason="too early",
        )
    record_canary_result(db_session, deployment=deployment, latency_ms=10, error=False)
    record_canary_result(db_session, deployment=deployment, latency_ms=11, error=False)
    transition_deployment(
        db_session,
        deployment=deployment,
        target_state="production",
        reason="healthy canary",
    )
    assert (
        repo.get_prod_model(db_session, model_name="classifier").model_version == "v2"
    )


def test_canary_assignment_is_deterministic(db_session, sample_model):
    _register_versions(db_session, sample_model)
    deployment = create_deployment(
        db_session, model_name="classifier", candidate_version="v2"
    )
    for state in ("evaluated", "approved"):
        transition_deployment(
            db_session, deployment=deployment, target_state=state, reason="test"
        )
    transition_deployment(
        db_session,
        deployment=deployment,
        target_state="canary",
        reason="test",
        traffic_percentage=50,
    )
    assert should_route_to_canary(deployment, "customer-123") == should_route_to_canary(
        deployment, "customer-123"
    )


def test_canary_automatically_rolls_back_on_errors(db_session, sample_model):
    _register_versions(db_session, sample_model)
    deployment = create_deployment(
        db_session,
        model_name="classifier",
        candidate_version="v2",
        min_requests=2,
        max_error_rate=0.25,
    )
    for state in ("evaluated", "approved"):
        transition_deployment(
            db_session, deployment=deployment, target_state=state, reason="test"
        )
    transition_deployment(
        db_session,
        deployment=deployment,
        target_state="canary",
        reason="test",
        traffic_percentage=50,
    )

    assert (
        record_canary_result(
            db_session, deployment=deployment, latency_ms=10, error=True
        )
        is False
    )
    assert (
        record_canary_result(
            db_session, deployment=deployment, latency_ms=10, error=False
        )
        is True
    )
    assert deployment.state == "rolled_back"
    assert deployment.traffic_percentage == 0
    assert "Automatic rollback" in deployment.last_reason


def test_release_api_end_to_end(sample_model):
    init_db()
    session = get_session()
    _register_versions(session, sample_model)
    session.close()

    with TestClient(create_app()) as client:
        dataset = client.post(
            "/release/datasets",
            json={
                "name": "holdout",
                "version": "v1",
                "uri": "file:///data/holdout",
                "checksum": "sha256:abc",
            },
        )
        assert dataset.status_code == 201
        dataset_id = dataset.json()["id"]

        reports = []
        for version, accuracy in (("v1", 0.97), ("v2", 0.98)):
            response = client.post(
                "/release/evaluations",
                json={
                    "model_name": "classifier",
                    "model_version": version,
                    "dataset_version_id": dataset_id,
                    "metrics": {"accuracy": accuracy, "p95_ms": 10.0},
                },
            )
            assert response.status_code == 201
            reports.append(response.json()["id"])

        runs = client.get("/release/evaluation-runs?model_name=classifier")
        assert runs.status_code == 200
        assert len(runs.json()) == 2
        assert {run["status"] for run in runs.json()} == {"completed"}

        assert (
            client.post(
                "/release/gate-policies",
                json={
                    "name": "safe-release",
                    "model_name": "classifier",
                    "constraints": {"accuracy": {"min": 0.95, "max_drop": 0.01}},
                },
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/release/deployments",
                json={"model_name": "classifier", "candidate_version": "v2"},
            ).status_code
            == 201
        )

        decision = client.post(
            "/release/deployments/classifier/evaluate",
            json={
                "policy_name": "safe-release",
                "candidate_report_id": reports[1],
                "baseline_report_id": reports[0],
            },
        )
        assert decision.status_code == 200
        assert decision.json()["passed"] is True
        assert decision.json()["deployment"]["state"] == "approved"


def test_evaluation_report_delete_is_rejected(db_session):
    report = EvaluationReport(
        model_name="m",
        model_version="v1",
        evaluation_run_id="run",
        dataset_version_id="dataset",
        metrics_json="{}",
        config_json="{}",
        content_hash="hash",
    )
    db_session.add(report)
    db_session.commit()
    db_session.delete(report)
    with pytest.raises(ValueError, match="immutable"):
        db_session.commit()
