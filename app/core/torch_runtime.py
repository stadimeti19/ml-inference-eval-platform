"""PyTorch runtime tuning for predictable inference latency."""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_torch_runtime(settings: Settings) -> None:
    """Apply conservative CPU threading settings.

    In web serving, many concurrent requests plus PyTorch's default CPU
    threadpool can oversubscribe cores. Keeping per-request inference to a
    small thread count generally improves p95/p99 latency under load.
    """
    try:
        import torch

        torch.set_num_threads(settings.torch_num_threads)
        try:
            torch.set_num_interop_threads(settings.torch_interop_threads)
        except RuntimeError:
            logger.warning("torch_interop_threads_already_initialized")

        logger.info(
            "torch_runtime_configured",
            torch_num_threads=torch.get_num_threads(),
            torch_interop_threads=settings.torch_interop_threads,
        )
    except Exception as exc:
        logger.warning("torch_runtime_config_failed", error=str(exc))
