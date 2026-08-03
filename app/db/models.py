"""SQLAlchemy ORM models."""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
)
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
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="staging")
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
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
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


class DeploymentEvent(Base):
    """Immutable audit record for model lifecycle and gate decisions."""

    __tablename__ = "deployment_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    new_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<DeploymentEvent {self.event_type} {self.model_name}@{self.version}>"


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


class DatasetVersion(Base):
    """Immutable pointer to the exact dataset used by an evaluation."""

    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class EvaluationRun(Base):
    """Execution lifecycle for one model/dataset evaluation."""

    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_version_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )


class EvaluationReport(Base):
    """Append-only result of evaluating one model against one dataset version."""

    __tablename__ = "evaluation_reports"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluation_run_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True
    )
    dataset_version_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class GatePolicy(Base):
    """Reusable absolute and baseline-relative release constraints."""

    __tablename__ = "gate_policies"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    constraints_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class Deployment(Base):
    """Current rollout state and persisted canary health for one model."""

    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    baseline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_version: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="registered", index=True
    )
    traffic_percentage: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    min_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    max_error_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    max_avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_sum_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


@event.listens_for(EvaluationReport, "before_update")
@event.listens_for(EvaluationReport, "before_delete")
def _prevent_evaluation_report_mutation(*_: object) -> None:
    raise ValueError("Evaluation reports are immutable")
