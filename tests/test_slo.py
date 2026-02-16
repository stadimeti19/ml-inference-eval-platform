"""Tests for SLO-based gating -- pure evaluation logic and API endpoints."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.eval.slo import evaluate_slo


# -------------------------------------------------------------------
# Unit tests for the evaluate_slo engine (no DB needed)
# -------------------------------------------------------------------

class TestEvaluateSlo:
    """Pure-function tests for the constraint checker."""

    def test_all_pass(self):
        constraints = {
            "p95_ms_max": 50.0,
            "accuracy_min": 0.95,
        }
        metrics = {"p95_ms": 30.0, "accuracy": 0.97}
        result = evaluate_slo(constraints, metrics)

        assert result["passed"] is True
        assert all(c["passed"] for c in result["checks"])

    def test_latency_breach(self):
        constraints = {"p95_ms_max": 50.0}
        metrics = {"p95_ms": 75.0}
        result = evaluate_slo(constraints, metrics)

        assert result["passed"] is False
        assert result["checks"][0]["passed"] is False
        assert result["checks"][0]["actual"] == 75.0

    def test_accuracy_breach(self):
        constraints = {"accuracy_min": 0.95}
        metrics = {"accuracy": 0.90}
        result = evaluate_slo(constraints, metrics)

        assert result["passed"] is False
        assert result["checks"][0]["passed"] is False

    def test_throughput_breach(self):
        constraints = {"throughput_qps_min": 100.0}
        metrics = {"throughput_qps": 50.0}
        result = evaluate_slo(constraints, metrics)

        assert result["passed"] is False

    def test_missing_metric(self):
        constraints = {"p95_ms_max": 50.0}
        metrics = {}
        result = evaluate_slo(constraints, metrics)

        assert result["passed"] is False
        assert "not found" in result["checks"][0]["reason"]

    def test_multiple_constraints_mixed(self):
        constraints = {
            "p95_ms_max": 50.0,
            "accuracy_min": 0.95,
            "throughput_qps_min": 100.0,
        }
        metrics = {"p95_ms": 30.0, "accuracy": 0.97, "throughput_qps": 80.0}
        result = evaluate_slo(constraints, metrics)

        assert result["passed"] is False
        passed_checks = [c for c in result["checks"] if c["passed"]]
        failed_checks = [c for c in result["checks"] if not c["passed"]]
        assert len(passed_checks) == 2
        assert len(failed_checks) == 1
        assert failed_checks[0]["constraint"] == "throughput_qps_min"

    def test_boundary_exact_max(self):
        constraints = {"p95_ms_max": 50.0}
        metrics = {"p95_ms": 50.0}
        result = evaluate_slo(constraints, metrics)
        assert result["passed"] is True

    def test_boundary_exact_min(self):
        constraints = {"accuracy_min": 0.95}
        metrics = {"accuracy": 0.95}
        result = evaluate_slo(constraints, metrics)
        assert result["passed"] is True


# -------------------------------------------------------------------
# Integration tests for SLO gate (with DB)
# -------------------------------------------------------------------

def test_slo_gate_pass(db_session):
    """Model meets all SLO constraints -> gate passes."""
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v1",
        artifact_path="/tmp/m.pt",
        metrics={"accuracy": 0.97, "p95_ms": 30.0, "throughput_qps": 200.0},
    )
    repo.create_slo_policy(
        db_session,
        name="prod_slo",
        model_name="m",
        constraints={
            "p95_ms_max": 50.0,
            "accuracy_min": 0.95,
            "throughput_qps_min": 100.0,
        },
    )

    with patch("app.eval.slo.get_session", return_value=db_session):
        from app.eval.slo import run_slo_gate
        result = run_slo_gate("m", "v1", "prod_slo")

    assert result.passed is True
    details = json.loads(result.details)
    assert details["passed"] is True
    assert details["policy_name"] == "prod_slo"


def test_slo_gate_fail(db_session):
    """Model violates latency SLO -> gate fails."""
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v1",
        artifact_path="/tmp/m.pt",
        metrics={"accuracy": 0.97, "p95_ms": 80.0, "throughput_qps": 200.0},
    )
    repo.create_slo_policy(
        db_session,
        name="strict_slo",
        model_name="m",
        constraints={"p95_ms_max": 50.0, "accuracy_min": 0.95},
    )

    with patch("app.eval.slo.get_session", return_value=db_session):
        from app.eval.slo import run_slo_gate
        result = run_slo_gate("m", "v1", "strict_slo")

    assert result.passed is False
    details = json.loads(result.details)
    latency_check = [c for c in details["checks"] if c["constraint"] == "p95_ms_max"][0]
    assert latency_check["passed"] is False
    assert latency_check["actual"] == 80.0


def test_slo_gate_missing_policy(db_session):
    """Non-existent policy -> gate fails with error."""
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v1",
        artifact_path="/tmp/m.pt",
        metrics={"accuracy": 0.97},
    )

    with patch("app.eval.slo.get_session", return_value=db_session):
        from app.eval.slo import run_slo_gate
        result = run_slo_gate("m", "v1", "nonexistent")

    assert result.passed is False
    details = json.loads(result.details)
    assert "not found" in details["error"]


def test_slo_gate_no_metrics(db_session):
    """Model without metrics -> gate fails."""
    repo.register_model(
        db_session,
        model_name="m",
        model_version="v1",
        artifact_path="/tmp/m.pt",
    )
    repo.create_slo_policy(
        db_session,
        name="slo1",
        model_name="m",
        constraints={"p95_ms_max": 50.0},
    )

    with patch("app.eval.slo.get_session", return_value=db_session):
        from app.eval.slo import run_slo_gate
        result = run_slo_gate("m", "v1", "slo1")

    assert result.passed is False


# -------------------------------------------------------------------
# API endpoint tests
# -------------------------------------------------------------------

@pytest.fixture()
def slo_client(tmp_path):
    """TestClient with init_db called."""
    init_db()
    app = create_app()
    with TestClient(app) as tc:
        yield tc


def test_create_and_get_policy(slo_client):
    resp = slo_client.post(
        "/slo/policies",
        json={
            "name": "test_policy",
            "model_name": "mnist_cnn",
            "constraints": {"p95_ms_max": 50.0, "accuracy_min": 0.95},
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "test_policy"
    assert body["constraints"]["p95_ms_max"] == 50.0

    resp2 = slo_client.get("/slo/policies/test_policy")
    assert resp2.status_code == 200
    assert resp2.json()["name"] == "test_policy"


def test_create_duplicate_policy(slo_client):
    payload = {
        "name": "dup",
        "model_name": "m",
        "constraints": {"p95_ms_max": 50.0},
    }
    slo_client.post("/slo/policies", json=payload)
    resp = slo_client.post("/slo/policies", json=payload)
    assert resp.status_code == 409


def test_list_policies(slo_client):
    slo_client.post(
        "/slo/policies",
        json={"name": "a", "model_name": "m", "constraints": {"p95_ms_max": 50}},
    )
    slo_client.post(
        "/slo/policies",
        json={"name": "b", "model_name": "m", "constraints": {"accuracy_min": 0.9}},
    )
    resp = slo_client.get("/slo/policies")
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


def test_delete_policy(slo_client):
    slo_client.post(
        "/slo/policies",
        json={"name": "del_me", "model_name": "m", "constraints": {"p95_ms_max": 50}},
    )
    resp = slo_client.delete("/slo/policies/del_me")
    assert resp.status_code == 204

    resp2 = slo_client.get("/slo/policies/del_me")
    assert resp2.status_code == 404


def test_delete_nonexistent_policy(slo_client):
    resp = slo_client.delete("/slo/policies/ghost")
    assert resp.status_code == 404


def test_slo_check_endpoint(slo_client):
    """End-to-end: create policy, register model with metrics, check SLO."""
    session = get_session()
    repo.register_model(
        session,
        model_name="mnist_cnn",
        model_version="v1.0.0",
        artifact_path="/tmp/m.pt",
        metrics={"accuracy": 0.97, "p95_ms": 30.0, "throughput_qps": 200.0},
    )
    session.close()

    slo_client.post(
        "/slo/policies",
        json={
            "name": "prod",
            "model_name": "mnist_cnn",
            "constraints": {
                "p95_ms_max": 50.0,
                "accuracy_min": 0.95,
                "throughput_qps_min": 100.0,
            },
        },
    )

    resp = slo_client.post(
        "/slo/check",
        json={
            "model_name": "mnist_cnn",
            "model_version": "v1.0.0",
            "policy_name": "prod",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert all(c["passed"] for c in body["checks"])


def test_slo_check_fails(slo_client):
    """Model violates SLO -> check endpoint reports failure."""
    session = get_session()
    repo.register_model(
        session,
        model_name="mnist_cnn",
        model_version="v1.0.0",
        artifact_path="/tmp/m.pt",
        metrics={"accuracy": 0.80, "p95_ms": 30.0},
    )
    session.close()

    slo_client.post(
        "/slo/policies",
        json={
            "name": "strict",
            "model_name": "mnist_cnn",
            "constraints": {"accuracy_min": 0.95},
        },
    )

    resp = slo_client.post(
        "/slo/check",
        json={
            "model_name": "mnist_cnn",
            "model_version": "v1.0.0",
            "policy_name": "strict",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is False
