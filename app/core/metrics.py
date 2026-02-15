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
