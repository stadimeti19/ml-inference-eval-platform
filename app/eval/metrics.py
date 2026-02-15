"""Evaluation metric computation."""

from __future__ import annotations

import numpy as np


def compute_eval_metrics(
    predictions: list[int],
    labels: list[int],
    latencies_ms: list[float],
) -> dict:
    """Compute accuracy and latency percentiles.

    Returns:
        Dict with keys: accuracy, p50_ms, p95_ms, p99_ms, throughput_qps.
    """
    preds = np.array(predictions)
    labs = np.array(labels)
    lats = np.array(latencies_ms)

    accuracy = float((preds == labs).mean()) if len(preds) > 0 else 0.0
    p50 = float(np.percentile(lats, 50)) if len(lats) > 0 else 0.0
    p95 = float(np.percentile(lats, 95)) if len(lats) > 0 else 0.0
    p99 = float(np.percentile(lats, 99)) if len(lats) > 0 else 0.0

    total_time_s = sum(latencies_ms) / 1000.0 if latencies_ms else 1.0
    throughput = len(predictions) / total_time_s if total_time_s > 0 else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "throughput_qps": round(throughput, 2),
    }
