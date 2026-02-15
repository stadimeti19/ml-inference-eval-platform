"""Tests for regression gate pass/fail logic."""

from __future__ import annotations

import json
from unittest.mock import patch

from app.db import repositories as repo


def _register_with_metrics(db_session, version: str, accuracy: float, p95_ms: float):
    mv = repo.register_model(
        db_session,
        model_name="m",
        model_version=version,
        artifact_path=f"/tmp/{version}.pt",
        metrics={"accuracy": accuracy, "p95_ms": p95_ms},
    )
    return mv


def test_gate_pass(db_session):
    _register_with_metrics(db_session, "v1", accuracy=0.95, p95_ms=10.0)
    _register_with_metrics(db_session, "v2", accuracy=0.95, p95_ms=10.5)

    # Patch get_session where it was imported (inside app.eval.gates)
    with patch("app.eval.gates.get_session", return_value=db_session):
        from app.eval.gates import run_regression_gate
        result = run_regression_gate("m", "v2", "v1")

    assert result.passed is True
    details = json.loads(result.details)
    assert details["passed"] is True
    assert all(c["passed"] for c in details["checks"])


def test_gate_fail_accuracy(db_session):
    _register_with_metrics(db_session, "v1", accuracy=0.95, p95_ms=10.0)
    _register_with_metrics(db_session, "v2", accuracy=0.90, p95_ms=10.0)

    with patch("app.eval.gates.get_session", return_value=db_session):
        from app.eval.gates import run_regression_gate
        result = run_regression_gate("m", "v2", "v1")

    assert result.passed is False
    details = json.loads(result.details)
    acc_check = details["checks"][0]
    assert acc_check["check"] == "accuracy_drop"
    assert acc_check["passed"] is False


def test_gate_fail_latency(db_session):
    _register_with_metrics(db_session, "v1", accuracy=0.95, p95_ms=10.0)
    _register_with_metrics(db_session, "v2", accuracy=0.95, p95_ms=15.0)

    with patch("app.eval.gates.get_session", return_value=db_session):
        from app.eval.gates import run_regression_gate
        result = run_regression_gate("m", "v2", "v1")

    assert result.passed is False
    details = json.loads(result.details)
    p95_check = details["checks"][1]
    assert p95_check["check"] == "p95_latency_increase"
    assert p95_check["passed"] is False
