#!/usr/bin/env python3
"""Reproducible online inference benchmark for /predict.

Examples:
    python scripts/benchmark_inference.py --requests 10000 --concurrency 100 --model_version v1.0.0
    python scripts/benchmark_inference.py --model_version v1.0.0 --candidate_version v2.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from PIL import Image


def make_payload(
    *,
    model_name: str,
    model_version: str | None,
    shadow_version: str | None = None,
    payload_source: str,
    mnist_data_dir: str = "./data",
    mnist_index: int | None = None,
) -> dict[str, Any]:
    """Build a /predict payload from generated, dataset, or file input."""
    if payload_source in {"random", "random_noise"}:
        image_b64 = _make_random_image_b64()
    elif payload_source == "mnist_test":
        image_b64 = _make_mnist_image_b64(
            mnist_data_dir=mnist_data_dir,
            index=mnist_index,
            train=False,
        )
    elif payload_source == "mnist_train":
        image_b64 = _make_mnist_image_b64(
            mnist_data_dir=mnist_data_dir,
            index=mnist_index,
            train=True,
        )
    else:
        with open(payload_source, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

    payload: dict[str, Any] = {"model_name": model_name, "image_b64": image_b64}
    if model_version:
        payload["model_version"] = model_version
    if shadow_version:
        payload["shadow_version"] = shadow_version
    return payload


def summarize_latencies(
    *,
    latencies_ms: list[float],
    errors: int,
    total_requests: int,
    wall_time_s: float,
    cache_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable benchmark summary dict."""
    lats = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    successful = len(latencies_ms)
    cache_stats = cache_stats or {}
    return {
        "requests": total_requests,
        "successful_requests": successful,
        "errors": errors,
        "error_rate": round(errors / total_requests, 6) if total_requests else 0.0,
        "wall_time_s": round(wall_time_s, 3),
        "throughput_qps": round(successful / wall_time_s, 2) if wall_time_s else 0.0,
        "avg_latency_ms": round(float(lats.mean()), 3),
        "p50_latency_ms": round(float(np.percentile(lats, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(lats, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(lats, 99)), 3),
        "cache_hit_rate": cache_stats.get("cache_hit_rate"),
        "cache_hits": cache_stats.get("cache_hits"),
        "cache_misses": cache_stats.get("cache_misses"),
        "model_load_count": cache_stats.get("model_load_count"),
        "avg_model_load_time_ms": cache_stats.get("avg_model_load_time_ms"),
    }


async def run_benchmark(
    *,
    url: str,
    model_name: str,
    model_version: str | None,
    shadow_version: str | None = None,
    requests: int,
    concurrency: int,
    warmup_requests: int,
    payload_source: str,
    payload_mode: str = "fixed",
    mnist_data_dir: str = "./data",
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark one model version and return a report."""
    warmup_payloads = _build_payloads(
        count=warmup_requests,
        model_name=model_name,
        model_version=model_version,
        shadow_version=shadow_version,
        payload_source=payload_source,
        payload_mode=payload_mode,
        mnist_data_dir=mnist_data_dir,
        seed=seed,
    )
    payloads = _build_payloads(
        count=requests,
        model_name=model_name,
        model_version=model_version,
        shadow_version=shadow_version,
        payload_source=payload_source,
        payload_mode=payload_mode,
        mnist_data_dir=mnist_data_dir,
        seed=seed + 1,
    )
    payload_meta = _payload_metadata(
        count=requests,
        payload_source=payload_source,
        payload_mode=payload_mode,
        mnist_data_dir=mnist_data_dir,
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        for payload in warmup_payloads:
            await _send_request(client, url, payload)

        before_cache = await _get_cache_metrics(client, url)
        semaphore = asyncio.Semaphore(concurrency)
        latencies: list[float] = []
        errors = 0

        async def _bounded(payload: dict[str, Any]) -> None:
            nonlocal errors
            async with semaphore:
                latency = await _send_request(client, url, payload)
                if latency is None:
                    errors += 1
                else:
                    latencies.append(latency)

        start = time.perf_counter()
        await asyncio.gather(*(_bounded(payload) for payload in payloads))
        wall_time_s = time.perf_counter() - start
        after_cache = await _get_cache_metrics(client, url)

    cache_delta = _cache_delta(before_cache, after_cache)
    summary = summarize_latencies(
        latencies_ms=latencies,
        errors=errors,
        total_requests=requests,
        wall_time_s=wall_time_s,
        cache_stats=cache_delta or after_cache,
    )
    return {
        "model_name": model_name,
        "model_version": model_version or "prod",
        "shadow_version": shadow_version,
        "url": url,
        "concurrency": concurrency,
        "warmup_requests": warmup_requests,
        "payload_source": payload_source,
        "payload_mode": payload_mode,
        **payload_meta,
        "summary": summary,
    }


def compare_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare two benchmark reports."""
    base = baseline["summary"]
    cand = candidate["summary"]
    p95_delta = _pct_delta(base["p95_latency_ms"], cand["p95_latency_ms"])
    qps_delta = _pct_delta(base["throughput_qps"], cand["throughput_qps"])
    avg_delta = _pct_delta(base["avg_latency_ms"], cand["avg_latency_ms"])
    return {
        "baseline_version": baseline["model_version"],
        "candidate_version": candidate["model_version"],
        "p95_latency_delta_percent": round(p95_delta, 3),
        "avg_latency_delta_percent": round(avg_delta, 3),
        "throughput_delta_percent": round(qps_delta, 3),
        "candidate_faster": cand["p95_latency_ms"] < base["p95_latency_ms"],
    }


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
) -> float | None:
    start = time.perf_counter()
    try:
        resp = await client.post(f"{url}/predict", json=payload)
        elapsed = (time.perf_counter() - start) * 1000.0
        if resp.status_code == 200:
            return elapsed
    except Exception:
        return None
    return None


async def _get_cache_metrics(
    client: httpx.AsyncClient,
    url: str,
) -> dict[str, Any] | None:
    try:
        resp = await client.get(f"{url}/metrics/cache", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def _cache_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not before or not after:
        return None
    hits = int(after.get("cache_hits", 0)) - int(before.get("cache_hits", 0))
    misses = int(after.get("cache_misses", 0)) - int(before.get("cache_misses", 0))
    loads = int(after.get("model_load_count", 0)) - int(
        before.get("model_load_count", 0)
    )
    total = hits + misses
    return {
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate": round(hits / total, 6) if total else 0.0,
        "model_load_count": loads,
        "avg_model_load_time_ms": after.get("avg_model_load_time_ms"),
    }


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return ((candidate - baseline) / baseline) * 100.0


def _build_payloads(
    *,
    count: int,
    model_name: str,
    model_version: str | None,
    shadow_version: str | None,
    payload_source: str,
    payload_mode: str,
    mnist_data_dir: str = "./data",
    seed: int = 42,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if payload_mode == "fixed":
        payload = make_payload(
            model_name=model_name,
            model_version=model_version,
            shadow_version=shadow_version,
            payload_source=payload_source,
            mnist_data_dir=mnist_data_dir,
            mnist_index=0 if payload_source.startswith("mnist_") else None,
        )
        return [payload] * count
    if payload_mode == "random_per_request":
        rng = random.Random(seed)
        dataset_len = (
            _mnist_len(mnist_data_dir, train=payload_source == "mnist_train")
            if payload_source.startswith("mnist_")
            else None
        )
        return [
            make_payload(
                model_name=model_name,
                model_version=model_version,
                shadow_version=shadow_version,
                payload_source=payload_source,
                mnist_data_dir=mnist_data_dir,
                mnist_index=rng.randrange(dataset_len) if dataset_len else None,
            )
            for _ in range(count)
        ]
    if payload_mode == "sequential":
        if not payload_source.startswith("mnist_"):
            raise ValueError("sequential payload mode requires a MNIST payload source")
        dataset_len = _mnist_len(mnist_data_dir, train=payload_source == "mnist_train")
        return [
            make_payload(
                model_name=model_name,
                model_version=model_version,
                shadow_version=shadow_version,
                payload_source=payload_source,
                mnist_data_dir=mnist_data_dir,
                mnist_index=i % dataset_len,
            )
            for i in range(count)
        ]
    raise ValueError(
        "payload_mode must be 'fixed', 'random_per_request', or 'sequential'"
    )


def _payload_metadata(
    *,
    count: int,
    payload_source: str,
    payload_mode: str,
    mnist_data_dir: str,
) -> dict[str, Any]:
    if not payload_source.startswith("mnist_"):
        return {
            "dataset_size": None,
            "unique_payloads": 1 if payload_mode == "fixed" and count else count,
        }
    dataset_len = _mnist_len(mnist_data_dir, train=payload_source == "mnist_train")
    if payload_mode == "fixed":
        unique = 1 if count else 0
    elif payload_mode == "sequential":
        unique = min(count, dataset_len)
    else:
        unique = None
    return {
        "dataset_size": dataset_len,
        "unique_payloads": unique,
    }


def _make_random_image_b64() -> str:
    arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_mnist_image_b64(
    *,
    mnist_data_dir: str,
    index: int | None,
    train: bool,
) -> str:
    ds = _load_mnist_dataset(mnist_data_dir=mnist_data_dir, train=train)
    if index is None:
        index = random.randrange(len(ds))
    img, _label = ds[index % len(ds)]
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mnist_len(mnist_data_dir: str, train: bool) -> int:
    return len(_load_mnist_dataset(mnist_data_dir=mnist_data_dir, train=train))


def _load_mnist_dataset(mnist_data_dir: str, train: bool):
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for --payload_source mnist_test"
        ) from exc
    try:
        return datasets.MNIST(root=mnist_data_dir, train=train, download=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"MNIST data not found at {mnist_data_dir}. Run training once or "
            "mount/copy the MNIST data before using --payload_source mnist_test."
        ) from exc


def _print_summary(report: dict[str, Any]) -> None:
    print("\n=== Inference Benchmark ===")
    if "reports" in report:
        for item in report["reports"]:
            s = item["summary"]
            print(f"\n{item['model_name']}@{item['model_version']}")
            print(f"  requests: {s['requests']}  concurrency: {item['concurrency']}")
            print(f"  qps: {s['throughput_qps']}  error_rate: {s['error_rate']}")
            print(
                "  latency_ms: "
                f"avg={s['avg_latency_ms']} p50={s['p50_latency_ms']} "
                f"p95={s['p95_latency_ms']} p99={s['p99_latency_ms']}"
            )
            print(f"  cache_hit_rate: {s['cache_hit_rate']}")
        if report.get("comparison"):
            c = report["comparison"]
            print("\nComparison")
            print(
                f"  p95_delta: {c['p95_latency_delta_percent']}%  "
                f"qps_delta: {c['throughput_delta_percent']}%"
            )
    else:
        s = report["summary"]
        print(f"\n{report['model_name']}@{report['model_version']}")
        print(f"  requests: {s['requests']}  concurrency: {report['concurrency']}")
        print(f"  qps: {s['throughput_qps']}  error_rate: {s['error_rate']}")
        print(
            "  latency_ms: "
            f"avg={s['avg_latency_ms']} p50={s['p50_latency_ms']} "
            f"p95={s['p95_latency_ms']} p99={s['p99_latency_ms']}"
        )
        print(f"  cache_hit_rate: {s['cache_hit_rate']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /predict inference")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model_name", default="mnist_cnn")
    parser.add_argument("--model_version", default=None)
    parser.add_argument("--candidate_version", default=None)
    parser.add_argument("--shadow_version", default=None)
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--warmup_requests", type=int, default=20)
    parser.add_argument("--payload_source", default="mnist_test")
    parser.add_argument("--mnist_data_dir", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--payload_mode",
        choices=["fixed", "random_per_request", "sequential"],
        default="sequential",
        help="Use one fixed payload, random varied payloads, or sequential MNIST samples.",
    )
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    async def _run() -> dict[str, Any]:
        baseline = await run_benchmark(
            url=args.url,
            model_name=args.model_name,
            model_version=args.model_version,
            shadow_version=args.shadow_version,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup_requests=args.warmup_requests,
            payload_source=args.payload_source,
            payload_mode=args.payload_mode,
            mnist_data_dir=args.mnist_data_dir,
            seed=args.seed,
        )
        if not args.candidate_version:
            return baseline
        candidate = await run_benchmark(
            url=args.url,
            model_name=args.model_name,
            model_version=args.candidate_version,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup_requests=args.warmup_requests,
            payload_source=args.payload_source,
            payload_mode=args.payload_mode,
            mnist_data_dir=args.mnist_data_dir,
            seed=args.seed,
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reports": [baseline, candidate],
            "comparison": compare_reports(baseline, candidate),
        }

    report = asyncio.run(_run())
    if "timestamp" not in report:
        report["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    print(f"\nSaved results to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
