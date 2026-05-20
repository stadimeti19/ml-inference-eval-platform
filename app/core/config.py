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
    torch_num_threads: int = field(
        default_factory=lambda: int(os.getenv("TORCH_NUM_THREADS", "1"))
    )
    torch_interop_threads: int = field(
        default_factory=lambda: int(os.getenv("TORCH_INTEROP_THREADS", "1"))
    )
    preload_prod_models: bool = field(
        default_factory=lambda: os.getenv("PRELOAD_PROD_MODELS", "true").lower()
        in {"1", "true", "yes", "on"}
    )
    llm_enable_live_providers: bool = field(
        default_factory=lambda: os.getenv("LLM_ENABLE_LIVE_PROVIDERS", "false").lower()
        in {"1", "true", "yes", "on"}
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    app_version: str = "0.1.0"


def get_settings() -> Settings:
    """Return a Settings instance populated from env vars."""
    return Settings()
