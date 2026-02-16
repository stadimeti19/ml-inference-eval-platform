"""Batch inference task executed by the RQ worker."""

from __future__ import annotations

import json
import time

from torch.utils.data import DataLoader

from app.core.logging import get_logger
from app.core.metrics import BATCH_JOB_DURATION
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.eval.metrics import compute_eval_metrics
from app.inference.cache import get_model_cached
from app.inference.predict import predict_batch

logger = get_logger(__name__)


def _parse_dataset_id(dataset_id: str) -> int:
    """Extract sample count from dataset_id like 'mnist_1000'."""
    parts = dataset_id.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 1000


def run_batch_inference(job_id: str) -> None:
    """Load model, run batch inference, compute metrics, update DB."""
    init_db()
    session = get_session()

    try:
        job = repo.get_batch_job(session, job_id=job_id)
        if job is None:
            logger.error("batch_job_not_found", job_id=job_id)
            return

        repo.update_batch_job(session, job_id=job_id, status="running")
        logger.info(
            "batch_job_started",
            job_id=job_id,
            model=f"{job.model_name}@{job.model_version}",
        )

        # Resolve artifact path
        mv = repo.get_model(
            session, model_name=job.model_name, model_version=job.model_version
        )
        if mv is None:
            repo.update_batch_job(session, job_id=job_id, status="failed")
            logger.error("model_not_found", job_id=job_id)
            return

        model = get_model_cached(
            mv.model_name, mv.model_version, mv.artifact_path,
            architecture=mv.architecture,
        )

        # Load dataset subset
        from app.datasets.mnist import get_mnist_subset
        n_samples = _parse_dataset_id(job.dataset_id)
        images, labels = get_mnist_subset(n=n_samples)

        config: dict = json.loads(job.config) if job.config else {}
        batch_size = config.get("batch_size", 64)

        dataset = list(zip(
            images.split(batch_size),
            labels.split(batch_size),
        ))
        loader: DataLoader = dataset  # type: ignore[assignment]

        start = time.perf_counter()
        result = predict_batch(model, loader)  # type: ignore[arg-type]
        duration_s = time.perf_counter() - start

        BATCH_JOB_DURATION.labels(model_name=job.model_name).observe(duration_s)

        metrics = compute_eval_metrics(
            predictions=result.predictions,
            labels=result.labels,
            latencies_ms=result.latencies_ms,
        )
        metrics["total_time_s"] = round(duration_s, 3)
        metrics["n_samples"] = len(result.predictions)

        repo.update_batch_job(
            session, job_id=job_id, status="succeeded", result_metrics=metrics
        )
        logger.info("batch_job_succeeded", job_id=job_id, metrics=metrics)

        # Also store metrics on the model version for gate comparisons
        mv.metrics = json.dumps(metrics)
        session.commit()

    except Exception as exc:
        repo.update_batch_job(session, job_id=job_id, status="failed")
        logger.error("batch_job_failed", job_id=job_id, error=str(exc))
        raise
    finally:
        session.close()
