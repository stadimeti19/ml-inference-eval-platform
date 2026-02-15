"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./platform.db")
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    model_artifacts_dir: str = field(
        default_factory=lambda: os.getenv("MODEL_ARTIFACTS_DIR", "./artifacts")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    queue_name: str = field(
        default_factory=lambda: os.getenv("QUEUE_NAME", "default")
    )
    app_version: str = "0.1.0"


def get_settings() -> Settings:
    """Return a Settings instance populated from env vars."""
    return Settings()
