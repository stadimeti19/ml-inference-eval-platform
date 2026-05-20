"""In-process runtime counters for JSON observability endpoints."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import DefaultDict

import numpy as np

_lock = threading.Lock()
_inference_request_count: DefaultDict[str, int] = defaultdict(int)
_prediction_error_count: DefaultDict[str, int] = defaultdict(int)
_inference_latencies_ms: DefaultDict[str, list[float]] = defaultdict(list)


def record_inference_request(model_name: str, latency_ms: float) -> None:
    with _lock:
        _inference_request_count[model_name] += 1
        _inference_latencies_ms[model_name].append(latency_ms)


def record_prediction_error(model_name: str) -> None:
    with _lock:
        _prediction_error_count[model_name] += 1


def get_inference_metrics() -> dict:
    with _lock:
        models = sorted(
            set(_inference_request_count)
            | set(_prediction_error_count)
            | set(_inference_latencies_ms)
        )
        per_model = {}
        total_requests = 0
        total_errors = 0
        all_latencies: list[float] = []
        for model_name in models:
            latencies = list(_inference_latencies_ms.get(model_name, []))
            lats = np.array(latencies) if latencies else np.array([])
            requests = int(_inference_request_count.get(model_name, 0))
            errors = int(_prediction_error_count.get(model_name, 0))
            total_requests += requests
            total_errors += errors
            all_latencies.extend(latencies)
            per_model[model_name] = _summarize_latencies(requests, errors, lats)

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": round(total_errors / total_requests, 6)
            if total_requests
            else 0.0,
            "latency_ms": _summarize_latencies(
                total_requests, total_errors, np.array(all_latencies)
            )["latency_ms"],
            "models": per_model,
        }


def _summarize_latencies(requests: int, errors: int, lats: np.ndarray) -> dict:
    if lats.size == 0:
        latency = {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    else:
        latency = {
            "avg": round(float(lats.mean()), 3),
            "p50": round(float(np.percentile(lats, 50)), 3),
            "p95": round(float(np.percentile(lats, 95)), 3),
            "p99": round(float(np.percentile(lats, 99)), 3),
        }
    return {
        "requests": requests,
        "errors": errors,
        "error_rate": round(errors / requests, 6) if requests else 0.0,
        "latency_ms": latency,
    }


def reset_runtime_metrics() -> None:
    with _lock:
        _inference_request_count.clear()
        _prediction_error_count.clear()
        _inference_latencies_ms.clear()
