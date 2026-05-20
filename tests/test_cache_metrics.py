"""Tests for model cache instrumentation."""

from __future__ import annotations

from unittest.mock import patch

from app.inference.cache import get_cache_stats, get_model_cached, reset_cache_stats


class _DummyModel:
    pass


def test_cache_metrics_track_hits_misses_and_loads():
    reset_cache_stats()

    with patch("app.inference.cache.load_model", return_value=_DummyModel()):
        get_model_cached("m", "v1", "/tmp/m.pt")
        get_model_cached("m", "v1", "/tmp/m.pt")

    stats = get_cache_stats()
    assert stats["scope"] in {"process", "aggregate"}
    assert stats["backend"] in {"memory", "redis"}
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["cache_hit_rate"] == 0.5
    assert stats["model_load_count"] == 1
    assert stats["avg_model_load_time_ms"] >= 0
    assert stats["current_cache_size"] == 1
    assert stats["local_process"]["cache_hits"] == 1
