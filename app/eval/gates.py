"""Regression gate logic comparing candidate vs baseline model metrics."""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.db import repositories as repo
from app.db.models import GateResult
from app.db.session import get_session, init_db

logger = get_logger(__name__)

# Gate thresholds
MAX_ACCURACY_DROP = 0.01  # 1 %
MAX_P95_LATENCY_INCREASE = 0.10  # 10 %


def _decision_summary(
    *,
    candidate_version: str,
    passed: bool,
    accuracy_delta_percent: float,
    p95_delta_percent: float,
    threshold_pct: float,
) -> str:
    accuracy_word = "improved" if accuracy_delta_percent >= 0 else "dropped"
    latency_word = "increased" if p95_delta_percent >= 0 else "decreased"
    action = "Approved" if passed else "Rejected"
    return (
        f"{action} {candidate_version}: accuracy {accuracy_word} by "
        f"{abs(accuracy_delta_percent):.2f}%, but p95 latency {latency_word} by "
        f"{abs(p95_delta_percent):.2f}%"
        + (
            f", exceeding {threshold_pct:.0f}% threshold."
            if not passed and p95_delta_percent > threshold_pct
            else "."
        )
    )


def _save_gate_result_with_event(
    session,
    *,
    model_name: str,
    candidate_version: str,
    baseline_version: str,
    passed: bool,
    details: dict,
) -> GateResult:
    result = repo.save_gate_result(
        session,
        model_name=model_name,
        candidate_version=candidate_version,
        baseline_version=baseline_version,
        passed=passed,
        details=details,
    )
    candidate = repo.get_model(
        session, model_name=model_name, model_version=candidate_version
    )
    status = candidate.status if candidate else None
    repo.create_deployment_event(
        session,
        model_name=model_name,
        version=candidate_version,
        previous_status=status,
        new_status=status,
        event_type="gate_pass" if passed else "gate_fail",
        reason=details.get("decision_summary") or details.get("reason"),
    )
    return result


def run_regression_gate(
    model_name: str,
    candidate_version: str,
    baseline_version: str,
) -> GateResult:
    """Compare candidate metrics against baseline and return pass/fail.

    Rules:
    - accuracy must not drop more than 1 %
    - p95 latency must not worsen more than 10 %
    """
    init_db()
    session = get_session()

    try:
        candidate = repo.get_model(
            session, model_name=model_name, model_version=candidate_version
        )
        baseline = repo.get_model(
            session, model_name=model_name, model_version=baseline_version
        )

        if candidate is None or baseline is None:
            missing = []
            if candidate is None:
                missing.append(f"candidate {candidate_version}")
            if baseline is None:
                missing.append(f"baseline {baseline_version}")
            detail: dict[str, Any] = {
                "error": f"Model version(s) not found: {', '.join(missing)}",
                "reason": "Gate could not run because required model metadata is missing",
                "recommendation": "Register both model versions before gating.",
            }
            return _save_gate_result_with_event(
                session,
                model_name=model_name,
                candidate_version=candidate_version,
                baseline_version=baseline_version,
                passed=False,
                details=detail,
            )

        cand_metrics = json.loads(candidate.metrics) if candidate.metrics else {}
        base_metrics = json.loads(baseline.metrics) if baseline.metrics else {}

        if not cand_metrics or not base_metrics:
            detail = {
                "error": "One or both model versions have no evaluation metrics. "
                "Run batch inference first.",
                "candidate_has_metrics": bool(cand_metrics),
                "baseline_has_metrics": bool(base_metrics),
                "reason": "Gate could not run because evaluation metrics are missing",
                "recommendation": "Run batch evaluation for both versions, then rerun the gate.",
            }
            return _save_gate_result_with_event(
                session,
                model_name=model_name,
                candidate_version=candidate_version,
                baseline_version=baseline_version,
                passed=False,
                details=detail,
            )

        checks: list[dict] = []
        passed = True

        # Accuracy check
        cand_acc = cand_metrics.get("accuracy", 0.0)
        base_acc = base_metrics.get("accuracy", 0.0)
        acc_drop = base_acc - cand_acc
        accuracy_delta = cand_acc - base_acc
        accuracy_delta_percent = accuracy_delta * 100.0
        acc_ok = acc_drop <= MAX_ACCURACY_DROP
        if not acc_ok:
            passed = False
        checks.append(
            {
                "check": "accuracy_drop",
                "baseline": base_acc,
                "candidate": cand_acc,
                "drop": round(acc_drop, 6),
                "threshold": MAX_ACCURACY_DROP,
                "passed": acc_ok,
            }
        )

        # P95 latency check
        cand_p95 = cand_metrics.get("p95_ms", 0.0)
        base_p95 = base_metrics.get("p95_ms", 0.0)
        if base_p95 > 0:
            p95_increase = (cand_p95 - base_p95) / base_p95
        else:
            p95_increase = 0.0
        p95_delta_percent = p95_increase * 100.0
        p95_ok = p95_increase <= MAX_P95_LATENCY_INCREASE
        if not p95_ok:
            passed = False
        checks.append(
            {
                "check": "p95_latency_increase",
                "baseline_ms": base_p95,
                "candidate_ms": cand_p95,
                "increase_pct": round(p95_increase * 100, 2),
                "threshold_pct": MAX_P95_LATENCY_INCREASE * 100,
                "passed": p95_ok,
            }
        )

        failed_checks = [c["check"] for c in checks if not c["passed"]]
        threshold_pct = MAX_P95_LATENCY_INCREASE * 100
        reason = (
            "All regression checks passed"
            if passed
            else f"Failed checks: {', '.join(failed_checks)}"
        )
        recommendation = (
            f"Promote {candidate_version} to production."
            if passed
            else (
                "Do not promote. Optimize model latency, reduce model size, "
                "or validate a larger latency budget with an SLO policy."
            )
        )
        decision_summary = _decision_summary(
            candidate_version=candidate_version,
            passed=passed,
            accuracy_delta_percent=accuracy_delta_percent,
            p95_delta_percent=p95_delta_percent,
            threshold_pct=threshold_pct,
        )

        detail = {
            "passed": passed,
            "baseline_accuracy": base_acc,
            "candidate_accuracy": cand_acc,
            "accuracy_delta": round(accuracy_delta, 6),
            "accuracy_delta_percent": round(accuracy_delta_percent, 4),
            "baseline_p95_ms": base_p95,
            "candidate_p95_ms": cand_p95,
            "p95_delta_percent": round(p95_delta_percent, 4),
            "threshold_used": {
                "max_accuracy_drop": MAX_ACCURACY_DROP,
                "max_p95_latency_increase_percent": threshold_pct,
            },
            "reason": reason,
            "recommendation": recommendation,
            "decision_summary": decision_summary,
            "checks": checks,
        }

        result = _save_gate_result_with_event(
            session,
            model_name=model_name,
            candidate_version=candidate_version,
            baseline_version=baseline_version,
            passed=passed,
            details=detail,
        )
        logger.info(
            "gate_result",
            model_name=model_name,
            candidate=candidate_version,
            baseline=baseline_version,
            passed=passed,
        )
        return result
    finally:
        session.close()
