"""Tests for batch job lifecycle (DB-level, no Redis required)."""

from __future__ import annotations

import json

from app.db import repositories as repo


def test_batch_job_lifecycle(db_session):
    job = repo.create_batch_job(
        db_session,
        model_name="m",
        model_version="v1",
        dataset_id="mnist_100",
        config={"batch_size": 32},
    )
    assert job.status == "queued"

    updated = repo.update_batch_job(db_session, job_id=job.id, status="running")
    assert updated is not None
    assert updated.status == "running"

    metrics = {"accuracy": 0.95, "p95_ms": 1.2}
    completed = repo.update_batch_job(
        db_session, job_id=job.id, status="succeeded", result_metrics=metrics
    )
    assert completed is not None
    assert completed.status == "succeeded"
    assert json.loads(completed.result_metrics)["accuracy"] == 0.95


def test_get_batch_job(db_session):
    job = repo.create_batch_job(
        db_session,
        model_name="m",
        model_version="v1",
        dataset_id="mnist_500",
    )
    fetched = repo.get_batch_job(db_session, job_id=job.id)
    assert fetched is not None
    assert fetched.id == job.id


def test_missing_batch_job(db_session):
    assert repo.get_batch_job(db_session, job_id="nonexistent") is None
