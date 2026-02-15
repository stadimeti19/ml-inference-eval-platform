#!/usr/bin/env python3
"""Load testing harness for the /predict endpoint.

Usage:
    python scripts/loadtest.py --url http://localhost:8000 --concurrency 10 --total 200
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
from datetime import datetime, timezone

import httpx
import numpy as np
from PIL import Image


def _make_dummy_image_b64() -> str:
    """Create a small 28x28 grayscale PNG encoded as base64."""
    arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _send_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> float | None:
    """Send a single POST /predict and return latency in ms, or None on error."""
    start = time.perf_counter()
    try:
        resp = await client.post(f"{url}/predict", json=payload, timeout=30.0)
        elapsed = (time.perf_counter() - start) * 1000.0
        if resp.status_code == 200:
            return elapsed
        return None
    except Exception:
        return None


async def run_loadtest(
    url: str,
    concurrency: int,
    total_requests: int,
    model_name: str,
) -> dict:
    """Execute the load test and return a summary dict."""
    image_b64 = _make_dummy_image_b64()
    payload = {"model_name": model_name, "image_b64": image_b64}

    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def _bounded_request(client: httpx.AsyncClient) -> None:
        nonlocal errors
        async with semaphore:
            result = await _send_request(client, url, payload)
            if result is not None:
                latencies.append(result)
            else:
                errors += 1

    wall_start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [_bounded_request(client) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - wall_start

    lats = np.array(latencies) if latencies else np.array([0.0])
    report = {
        "url": url,
        "model_name": model_name,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": len(latencies),
        "errors": errors,
        "wall_time_s": round(wall_s, 3),
        "qps": round(len(latencies) / wall_s, 2) if wall_s > 0 else 0,
        "p50_ms": round(float(np.percentile(lats, 50)), 3),
        "p95_ms": round(float(np.percentile(lats, 95)), 3),
        "p99_ms": round(float(np.percentile(lats, 99)), 3),
        "mean_ms": round(float(lats.mean()), 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the /predict endpoint")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--total", type=int, default=100, help="Total requests")
    parser.add_argument("--model_name", default="mnist_cnn")
    args = parser.parse_args()

    print(
        f"Running load test: {args.total} requests, "
        f"concurrency={args.concurrency}, url={args.url}"
    )
    report = asyncio.run(
        run_loadtest(args.url, args.concurrency, args.total, args.model_name)
    )

    print("\n=== Load Test Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    os.makedirs("reports", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = f"reports/loadtest_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
