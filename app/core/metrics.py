"""Prometheus metrics definitions."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Latency of HTTP requests in seconds",
    labelnames=["endpoint", "model_name"],
)

REQUEST_COUNT = Counter(
    "request_count",
    "Total HTTP request count",
    labelnames=["endpoint", "model_name", "status"],
)

BATCH_JOB_DURATION = Histogram(
    "batch_job_duration_seconds",
    "Duration of batch inference jobs in seconds",
    labelnames=["model_name"],
)

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Current number of jobs in the queue",
)

# Shadow / canary deployment metrics
SHADOW_LATENCY = Histogram(
    "shadow_latency_seconds",
    "Latency of shadow model inference in seconds",
    labelnames=["model_name", "shadow_version"],
)

SHADOW_DISAGREEMENT = Counter(
    "shadow_disagreement_total",
    "Number of requests where shadow model disagreed with prod",
    labelnames=["model_name", "shadow_version"],
)

SHADOW_REQUEST_COUNT = Counter(
    "shadow_request_total",
    "Total shadow inference requests",
    labelnames=["model_name", "shadow_version"],
)
