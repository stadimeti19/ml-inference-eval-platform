#!/usr/bin/env python3
"""Create a reproducible candidate that fails latency gates.

The script reuses the current prod artifact, registers a candidate version,
assigns intentionally slower evaluation metrics, then runs the regression gate.
It is designed for clear demos without retraining or external downloads.
"""

from __future__ import annotations

import argparse
import json

from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.eval.gates import run_regression_gate
from app.registry.manager import register


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a latency-regressing model")
    parser.add_argument("--model_name", default="mnist_cnn")
    parser.add_argument("--candidate_version", default="v_bad_latency")
    parser.add_argument("--accuracy_lift", type=float, default=0.008)
    parser.add_argument("--latency_multiplier", type=float, default=3.0)
    args = parser.parse_args()

    init_db()
    session = get_session()
    try:
        prod = repo.get_prod_model(session, model_name=args.model_name)
        if prod is None:
            raise SystemExit(
                f"No prod model found for {args.model_name}. Run make train first."
            )
        baseline_metrics = json.loads(prod.metrics) if prod.metrics else {
            "accuracy": 0.98,
            "p50_ms": 0.05,
            "p95_ms": 0.06,
            "p99_ms": 0.07,
            "throughput_qps": 18000.0,
            "n_samples": 10000,
        }
    finally:
        session.close()

    candidate = register(
        model_name=args.model_name,
        model_version=args.candidate_version,
        artifact_path=prod.artifact_path,
        tags={"demo": "bad_latency", "source": "simulate_bad_model"},
        architecture=prod.architecture,
    )

    candidate_metrics = {
        **baseline_metrics,
        "accuracy": round(
            min(float(baseline_metrics.get("accuracy", 0.0)) + args.accuracy_lift, 1.0),
            6,
        ),
        "p50_ms": round(float(baseline_metrics.get("p50_ms", 0.0)) * args.latency_multiplier, 3),
        "p95_ms": round(float(baseline_metrics.get("p95_ms", 0.0)) * args.latency_multiplier, 3),
        "p99_ms": round(float(baseline_metrics.get("p99_ms", 0.0)) * args.latency_multiplier, 3),
        "throughput_qps": round(float(baseline_metrics.get("throughput_qps", 1.0)) / args.latency_multiplier, 2),
        "simulation": "accuracy improves, latency violates regression gate",
    }

    session = get_session()
    try:
        mv = repo.get_model(
            session,
            model_name=args.model_name,
            model_version=candidate.model_version,
        )
        if mv is None:
            raise SystemExit("Candidate registration failed")
        mv.metrics = json.dumps(candidate_metrics)
        session.commit()
    finally:
        session.close()

    result = run_regression_gate(
        model_name=args.model_name,
        candidate_version=args.candidate_version,
        baseline_version=prod.model_version,
    )
    details = json.loads(result.details) if result.details else {}

    print("=== Bad Model Simulation ===")
    print(f"Baseline:  {args.model_name}@{prod.model_version}")
    print(f"Candidate: {args.model_name}@{args.candidate_version}")
    print(details.get("decision_summary", f"Gate passed={result.passed}"))
    print(f"Recommendation: {details.get('recommendation', '-')}")

    if result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
