"""Tests for benchmark result shaping."""

from __future__ import annotations

from scripts.benchmark_inference import (
    _build_payloads,
    compare_reports,
    summarize_latencies,
)


def test_benchmark_summary_shape():
    summary = summarize_latencies(
        latencies_ms=[10.0, 20.0, 30.0, 40.0],
        errors=1,
        total_requests=5,
        wall_time_s=1.0,
        cache_stats={
            "cache_hit_rate": 0.8,
            "cache_hits": 4,
            "cache_misses": 1,
            "model_load_count": 1,
            "avg_model_load_time_ms": 12.3,
        },
    )

    assert summary["requests"] == 5
    assert summary["successful_requests"] == 4
    assert summary["error_rate"] == 0.2
    assert summary["throughput_qps"] == 4.0
    assert summary["p50_latency_ms"] == 25.0
    assert summary["p95_latency_ms"] == 38.5
    assert summary["cache_hit_rate"] == 0.8


def test_benchmark_compare_reports():
    baseline = {
        "model_version": "v1",
        "summary": {
            "p95_latency_ms": 10.0,
            "avg_latency_ms": 8.0,
            "throughput_qps": 100.0,
        },
    }
    candidate = {
        "model_version": "v2",
        "summary": {
            "p95_latency_ms": 15.0,
            "avg_latency_ms": 12.0,
            "throughput_qps": 80.0,
        },
    }

    comparison = compare_reports(baseline, candidate)
    assert comparison["baseline_version"] == "v1"
    assert comparison["candidate_version"] == "v2"
    assert comparison["p95_latency_delta_percent"] == 50.0
    assert comparison["throughput_delta_percent"] == -20.0
    assert comparison["candidate_faster"] is False


def test_random_per_request_payloads_are_varied():
    payloads = _build_payloads(
        count=3,
        model_name="mnist_cnn",
        model_version="v1",
        shadow_version="v2",
        payload_source="random",
        payload_mode="random_per_request",
    )

    assert len(payloads) == 3
    assert all(p["model_version"] == "v1" for p in payloads)
    assert all(p["shadow_version"] == "v2" for p in payloads)
    assert len({p["image_b64"] for p in payloads}) == 3


def test_fixed_payload_reuses_same_payload():
    payloads = _build_payloads(
        count=3,
        model_name="mnist_cnn",
        model_version="v1",
        shadow_version=None,
        payload_source="random",
        payload_mode="fixed",
    )

    assert len(payloads) == 3
    assert len({p["image_b64"] for p in payloads}) == 1
