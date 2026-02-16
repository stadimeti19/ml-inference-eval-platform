"""SLO-based model gating.

Instead of relative regression checks ("not worse than baseline"), SLO
gates enforce absolute production constraints that a model must meet::

    {
        "p95_ms_max": 50.0,       # p95 latency must be under 50 ms
        "p99_ms_max": 100.0,      # p99 latency must be under 100 ms
        "accuracy_min": 0.95,     # accuracy must be at least 95 %
        "error_rate_max": 0.001,  # error rate must be under 0.1 %
        "throughput_qps_min": 100  # must sustain at least 100 QPS
    }

Every key in the constraint dict is matched against the model's
evaluation metrics.  Keys ending in ``_max`` require ``metric <= threshold``;
keys ending in ``_min`` require ``metric >= threshold``.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger
from app.db import repositories as repo
from app.db.models import GateResult
from app.db.session import get_session, init_db

logger = get_logger(__name__)

# Recognised constraint suffixes
_MAX_SUFFIX = "_max"
_MIN_SUFFIX = "_min"

# Map constraint names to the metric key they check
_CONSTRAINT_TO_METRIC: dict[str, str] = {
    "p95_ms_max": "p95_ms",
    "p99_ms_max": "p99_ms",
    "p50_ms_max": "p50_ms",
    "accuracy_min": "accuracy",
    "error_rate_max": "error_rate",
    "throughput_qps_min": "throughput_qps",
}


def _metric_key(constraint_name: str) -> str:
    """Derive the metric key from a constraint name.

    Uses the explicit mapping first, then falls back to stripping the
    ``_max`` / ``_min`` suffix.
    """
    if constraint_name in _CONSTRAINT_TO_METRIC:
        return _CONSTRAINT_TO_METRIC[constraint_name]
    for suffix in (_MAX_SUFFIX, _MIN_SUFFIX):
        if constraint_name.endswith(suffix):
            return constraint_name[: -len(suffix)]
    return constraint_name


def evaluate_slo(
    constraints: dict[str, float],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Check *metrics* against *constraints*.

    Returns a dict with ``passed`` (bool) and ``checks`` (list of per-rule
    dicts with ``constraint``, ``metric_key``, ``threshold``, ``actual``,
    ``passed``).
    """
    checks: list[dict[str, Any]] = []
    all_passed = True

    for constraint_name, threshold in constraints.items():
        metric_name = _metric_key(constraint_name)
        actual = metrics.get(metric_name)

        if actual is None:
            checks.append({
                "constraint": constraint_name,
                "metric_key": metric_name,
                "threshold": threshold,
                "actual": None,
                "passed": False,
                "reason": f"metric '{metric_name}' not found in model metrics",
            })
            all_passed = False
            continue

        actual = float(actual)

        if constraint_name.endswith(_MAX_SUFFIX):
            ok = actual <= threshold
        elif constraint_name.endswith(_MIN_SUFFIX):
            ok = actual >= threshold
        else:
            ok = actual <= threshold

        if not ok:
            all_passed = False

        checks.append({
            "constraint": constraint_name,
            "metric_key": metric_name,
            "threshold": threshold,
            "actual": round(actual, 6),
            "passed": ok,
        })

    return {"passed": all_passed, "checks": checks}


def run_slo_gate(
    model_name: str,
    model_version: str,
    policy_name: str,
) -> GateResult:
    """Evaluate a model version against a named SLO policy.

    The result is stored as a ``GateResult`` with
    ``baseline_version='SLO:<policy_name>'`` so it is distinguishable
    from regression gates.
    """
    init_db()
    session = get_session()

    try:
        policy = repo.get_slo_policy(session, name=policy_name)
        if policy is None:
            detail = {"error": f"SLO policy '{policy_name}' not found"}
            return repo.save_gate_result(
                session,
                model_name=model_name,
                candidate_version=model_version,
                baseline_version=f"SLO:{policy_name}",
                passed=False,
                details=detail,
            )

        mv = repo.get_model(
            session, model_name=model_name, model_version=model_version
        )
        if mv is None:
            detail = {"error": f"Model {model_name}@{model_version} not found"}
            return repo.save_gate_result(
                session,
                model_name=model_name,
                candidate_version=model_version,
                baseline_version=f"SLO:{policy_name}",
                passed=False,
                details=detail,
            )

        model_metrics = json.loads(mv.metrics) if mv.metrics else {}
        if not model_metrics:
            detail = {
                "error": "Model has no evaluation metrics. Run batch inference first."
            }
            return repo.save_gate_result(
                session,
                model_name=model_name,
                candidate_version=model_version,
                baseline_version=f"SLO:{policy_name}",
                passed=False,
                details=detail,
            )

        constraints = json.loads(policy.constraints)
        evaluation = evaluate_slo(constraints, model_metrics)

        result = repo.save_gate_result(
            session,
            model_name=model_name,
            candidate_version=model_version,
            baseline_version=f"SLO:{policy_name}",
            passed=evaluation["passed"],
            details={
                "policy_name": policy_name,
                "constraints": constraints,
                **evaluation,
            },
        )

        logger.info(
            "slo_gate_result",
            model_name=model_name,
            model_version=model_version,
            policy=policy_name,
            passed=evaluation["passed"],
        )
        return result
    finally:
        session.close()
