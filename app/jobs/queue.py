"""Redis connection and RQ queue helpers."""

from __future__ import annotations

from redis import Redis
from rq import Queue

from app.core.config import get_settings


def get_redis_connection() -> Redis:
    """Return a Redis connection from config."""
    settings = get_settings()
    return Redis.from_url(settings.redis_url)


def get_queue(name: str | None = None) -> Queue:
    """Return an RQ queue bound to the configured Redis."""
    settings = get_settings()
    return Queue(name or settings.queue_name, connection=get_redis_connection())
