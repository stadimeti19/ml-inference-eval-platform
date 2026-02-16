#!/usr/bin/env python3
"""Full CI/CD-style pipeline: train two model versions, evaluate both,
run regression gate, and promote the winner.

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time

from app.core.logging import get_logger
from app.datasets.mnist import get_mnist_subset
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.eval.gates import run_regression_gate
from app.eval.metrics import compute_eval_metrics
from app.inference.cache import clear_cache, get_model_cached
from app.inference.model import train_mnist_model
from app.inference.predict import predict_batch
from app.registry.manager import list_models, promote, register

logger = get_logger(__name__)

BASELINE_VERSION = "v1.0.0"
CANDIDATE_VERSION = "v2.0.0"
MODEL_NAME = "mnist_cnn"
N_EVAL_SAMPLES = 1000
BATCH_SIZE = 64


def _train_and_register(
    version: str, architecture: str, epochs: int, do_promote: bool = False
) -> None:
    """Train, register, and optionally promote a model version."""
    # Check if already registered
    existing = list_models(model_name=MODEL_NAME)
    for m in existing:
        if m.model_version == version:
            print(f"  {MODEL_NAME}@{version} already registered, skipping training.")
            if do_promote and m.status != "prod":
                promote(model_name=MODEL_NAME, model_version=version)
                print(f"  Promoted {MODEL_NAME}@{version} to prod.")
            return

    saved_path = train_mnist_model(
        epochs=epochs,
        architecture=architecture,
        model_name=MODEL_NAME,
        model_version=version,
    )
    register(
        model_name=MODEL_NAME,
        model_version=version,
        artifact_path=saved_path,
        tags={"framework": "pytorch", "dataset": "mnist"},
        architecture=architecture,
    )
    print(f"  Registered {MODEL_NAME}@{version} (arch={architecture})")

    if do_promote:
        promote(model_name=MODEL_NAME, model_version=version)
        print(f"  Promoted {MODEL_NAME}@{version} to prod.")


def _run_batch_eval(version: str) -> dict:
    """Run batch evaluation and store metrics on the model version."""
    init_db()
    session = get_session()
    try:
        mv = repo.get_model(session, model_name=MODEL_NAME, model_version=version)
        if mv is None:
            print(f"  ERROR: {MODEL_NAME}@{version} not found")
            sys.exit(1)

        # Check if already has metrics
        if mv.metrics:
            existing = json.loads(mv.metrics)
            if existing.get("accuracy"):
                print(f"  {MODEL_NAME}@{version} already has eval metrics, skipping.")
                return existing

        model = get_model_cached(
            mv.model_name, mv.model_version, mv.artifact_path,
            architecture=mv.architecture,
        )

        images, labels = get_mnist_subset(n=N_EVAL_SAMPLES)
        dataset = list(zip(images.split(BATCH_SIZE), labels.split(BATCH_SIZE)))

        start = time.perf_counter()
        result = predict_batch(model, dataset)  # type: ignore[arg-type]
        duration = time.perf_counter() - start

        metrics = compute_eval_metrics(
            predictions=result.predictions,
            labels=result.labels,
            latencies_ms=result.latencies_ms,
        )
        metrics["total_time_s"] = round(duration, 3)
        metrics["n_samples"] = len(result.predictions)

        mv.metrics = json.dumps(metrics)
        session.commit()

        return metrics
    finally:
        session.close()


def main() -> None:
    init_db()

    print("=" * 60)
    print("  ML INFERENCE PIPELINE")
    print("=" * 60)

    # --- Step 1: Train baseline (default arch) ---
    print(f"\n[1/5] Training baseline {MODEL_NAME}@{BASELINE_VERSION} (arch=default)")
    _train_and_register(BASELINE_VERSION, architecture="default", epochs=3, do_promote=True)

    # --- Step 2: Train candidate (large arch) ---
    print(f"\n[2/5] Training candidate {MODEL_NAME}@{CANDIDATE_VERSION} (arch=large)")
    _train_and_register(CANDIDATE_VERSION, architecture="large", epochs=3)

    # --- Step 3: Evaluate baseline ---
    print(f"\n[3/5] Evaluating {MODEL_NAME}@{BASELINE_VERSION} on {N_EVAL_SAMPLES} samples")
    baseline_metrics = _run_batch_eval(BASELINE_VERSION)
    print(f"  accuracy={baseline_metrics['accuracy']:.4f}  "
          f"p95={baseline_metrics['p95_ms']:.3f}ms")

    # --- Step 4: Evaluate candidate ---
    print(f"\n[4/5] Evaluating {MODEL_NAME}@{CANDIDATE_VERSION} on {N_EVAL_SAMPLES} samples")
    candidate_metrics = _run_batch_eval(CANDIDATE_VERSION)
    print(f"  accuracy={candidate_metrics['accuracy']:.4f}  "
          f"p95={candidate_metrics['p95_ms']:.3f}ms")

    # --- Step 5: Regression gate ---
    print(f"\n[5/5] Running regression gate: {CANDIDATE_VERSION} vs {BASELINE_VERSION}")
    gate_result = run_regression_gate(
        model_name=MODEL_NAME,
        candidate_version=CANDIDATE_VERSION,
        baseline_version=BASELINE_VERSION,
    )

    details = json.loads(gate_result.details) if gate_result.details else {}
    status = "PASS" if gate_result.passed else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  GATE RESULT: {status}")
    print(f"{'=' * 60}")
    for check in details.get("checks", []):
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['check']}")
        if "drop" in check:
            print(f"        baseline={check['baseline']:.4f}  "
                  f"candidate={check['candidate']:.4f}  "
                  f"drop={check['drop']:.4f}  threshold={check['threshold']}")
        if "increase_pct" in check:
            print(f"        baseline={check['baseline_ms']:.3f}ms  "
                  f"candidate={check['candidate_ms']:.3f}ms  "
                  f"increase={check['increase_pct']:.1f}%  "
                  f"threshold={check['threshold_pct']:.0f}%")

    # --- Promote or keep baseline ---
    if gate_result.passed:
        print(f"\nGate passed. Promoting {MODEL_NAME}@{CANDIDATE_VERSION} to prod.")
        promote(model_name=MODEL_NAME, model_version=CANDIDATE_VERSION)
        print("Done.")
    else:
        print(f"\nGate FAILED. Keeping {MODEL_NAME}@{BASELINE_VERSION} in prod.")
        print("Candidate was not promoted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
