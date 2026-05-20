"""In-memory LRU model cache to avoid reloading on every request."""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

import torch.nn as nn

from app.core.config import get_settings
from app.inference.model import load_model

_MAX_CACHE_SIZE = 5
_REDIS_STATS_KEY = "platform:model_cache:stats"
_lock = threading.Lock()
_cache: collections.OrderedDict[tuple[str, str], nn.Module] = (
    collections.OrderedDict()
)
_redis_lock = threading.RLock()
_redis_client: Any | None = None
_redis_disabled = False
_stats = {
    "cache_hits": 0,
    "cache_misses": 0,
    "model_load_count": 0,
    "total_model_load_time_ms": 0.0,
}


def get_model_cached(
    model_name: str,
    model_version: str,
    artifact_path: str,
    architecture: str = "default",
) -> nn.Module:
    """Return a cached model or load it from disk."""
    key = (model_name, model_version)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            _stats["cache_hits"] += 1
            _increment_shared_counter("cache_hits")
            return _cache[key]

        _stats["cache_misses"] += 1
        _increment_shared_counter("cache_misses")

    load_start = time.perf_counter()
    model = load_model(artifact_path, architecture=architecture)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    with _lock:
        _cache[key] = model
        _stats["model_load_count"] += 1
        _stats["total_model_load_time_ms"] += load_ms
        _increment_shared_counter("model_load_count")
        _increment_shared_float("total_model_load_time_ms", load_ms)
        if len(_cache) > _MAX_CACHE_SIZE:
            _cache.popitem(last=False)

    return model


def get_cache_stats() -> dict:
    """Return a snapshot of model cache performance counters."""
    shared = _get_shared_stats()
    with _lock:
        local_hits = int(_stats["cache_hits"])
        local_misses = int(_stats["cache_misses"])
        local_load_count = int(_stats["model_load_count"])
        local_total_load_ms = float(_stats["total_model_load_time_ms"])
        if shared is not None:
            hits = shared["cache_hits"]
            misses = shared["cache_misses"]
            load_count = shared["model_load_count"]
            total_load_ms = shared["total_model_load_time_ms"]
            scope = "aggregate"
            backend = "redis"
        else:
            hits = local_hits
            misses = local_misses
            load_count = local_load_count
            total_load_ms = local_total_load_ms
            scope = "process"
            backend = "memory"
        total_lookups = hits + misses
        return {
            "scope": scope,
            "backend": backend,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": round(hits / total_lookups, 6) if total_lookups else 0.0,
            "model_load_count": load_count,
            "avg_model_load_time_ms": round(total_load_ms / load_count, 3)
            if load_count
            else 0.0,
            "total_model_load_time_ms": round(total_load_ms, 3),
            "current_cache_size": len(_cache),
            "max_cache_size": _MAX_CACHE_SIZE,
            "local_process": {
                "cache_hits": local_hits,
                "cache_misses": local_misses,
                "model_load_count": local_load_count,
                "current_cache_size": len(_cache),
            },
            "cached_models": [
                {"model_name": name, "model_version": version}
                for name, version in _cache.keys()
            ],
        }


def clear_cache() -> None:
    """Flush the entire model cache."""
    with _lock:
        _cache.clear()


def reset_cache_stats() -> None:
    """Reset cache counters and flush cached models, primarily for tests."""
    with _lock:
        _cache.clear()
        _stats["cache_hits"] = 0
        _stats["cache_misses"] = 0
        _stats["model_load_count"] = 0
        _stats["total_model_load_time_ms"] = 0.0
    _reset_shared_stats()


def _increment_shared_counter(field: str) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.hincrby(_REDIS_STATS_KEY, field, 1)
    except Exception:
        _disable_redis_stats()


def _increment_shared_float(field: str, value: float) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.hincrbyfloat(_REDIS_STATS_KEY, field, value)
    except Exception:
        _disable_redis_stats()


def _get_shared_stats() -> dict[str, float | int] | None:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        raw = client.hgetall(_REDIS_STATS_KEY)
    except Exception:
        _disable_redis_stats()
        return None
    if not raw:
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "model_load_count": 0,
            "total_model_load_time_ms": 0.0,
        }
    decoded = {
        key.decode() if isinstance(key, bytes) else str(key): value.decode()
        if isinstance(value, bytes)
        else str(value)
        for key, value in raw.items()
    }
    return {
        "cache_hits": int(float(decoded.get("cache_hits", 0))),
        "cache_misses": int(float(decoded.get("cache_misses", 0))),
        "model_load_count": int(float(decoded.get("model_load_count", 0))),
        "total_model_load_time_ms": float(
            decoded.get("total_model_load_time_ms", 0.0)
        ),
    }


def _reset_shared_stats() -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        client.delete(_REDIS_STATS_KEY)
    except Exception:
        _disable_redis_stats()


def _get_redis_client():
    global _redis_client
    if _redis_disabled:
        return None
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            from redis import Redis

            _redis_client = Redis.from_url(
                get_settings().redis_url,
                socket_connect_timeout=0.1,
                socket_timeout=0.1,
            )
            _redis_client.ping()
            return _redis_client
        except Exception:
            _disable_redis_stats()
            return None


def _disable_redis_stats() -> None:
    global _redis_client, _redis_disabled
    with _redis_lock:
        _redis_client = None
        _redis_disabled = True
