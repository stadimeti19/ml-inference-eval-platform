"""In-memory LRU model cache to avoid reloading on every request."""

from __future__ import annotations

import collections
import threading

from app.inference.model import MNISTClassifier, load_model

_MAX_CACHE_SIZE = 5
_lock = threading.Lock()
_cache: collections.OrderedDict[tuple[str, str], MNISTClassifier] = (
    collections.OrderedDict()
)


def get_model_cached(
    model_name: str, model_version: str, artifact_path: str
) -> MNISTClassifier:
    """Return a cached model or load it from disk."""
    key = (model_name, model_version)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    model = load_model(artifact_path)

    with _lock:
        _cache[key] = model
        if len(_cache) > _MAX_CACHE_SIZE:
            _cache.popitem(last=False)

    return model


def clear_cache() -> None:
    """Flush the entire model cache."""
    with _lock:
        _cache.clear()
