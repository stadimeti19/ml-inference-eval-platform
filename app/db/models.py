"""SQLAlchemy ORM models."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="staging"
    )
    architecture: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", server_default="default"
    )
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string

    def __repr__(self) -> str:
        return (
            f"<ModelVersion {self.model_name}@{self.model_version} "
            f"status={self.status}>"
        )


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(128), nullable=False)
    config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued"
    )
    result_metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<BatchJob {self.id[:8]} status={self.status}>"


class GateResult(Base):
    __tablename__ = "gate_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_version: Mapped[str] = mapped_column(String(64), nullable=False)
    baseline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"<GateResult {self.model_name} {status}>"


class ShadowResult(Base):
    """Per-request log of shadow (canary) inference alongside production."""

    __tablename__ = "shadow_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    prod_version: Mapped[str] = mapped_column(String(64), nullable=False)
    shadow_version: Mapped[str] = mapped_column(String(64), nullable=False)
    prod_prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    shadow_prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    agreed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prod_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    shadow_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        agree = "agree" if self.agreed else "DISAGREE"
        return (
            f"<ShadowResult {self.model_name} "
            f"prod={self.prod_version} shadow={self.shadow_version} {agree}>"
        )


class SloPolicy(Base):
    """Service Level Objective policy with absolute constraints.

    ``constraints`` is a JSON dict mapping metric names to thresholds::

        {
            "p95_ms_max": 50.0,
            "accuracy_min": 0.95,
            "error_rate_max": 0.001,
            "throughput_qps_min": 100.0
        }
    """

    __tablename__ = "slo_policies"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    constraints: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<SloPolicy {self.name} model={self.model_name}>"
