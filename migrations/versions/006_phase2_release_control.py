"""Add release-control entities.

Revision ID: 006
Revises: 005
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("uri", sa.String(1024), nullable=False),
        sa.Column("checksum", sa.String(128), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )
    op.create_index("ix_dataset_versions_name", "dataset_versions", ["name"])
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("dataset_version_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column(
            "started_at", sa.DateTime(), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_evaluation_runs_model_name", "evaluation_runs", ["model_name"]
    )
    op.create_index(
        "ix_evaluation_runs_dataset_version_id",
        "evaluation_runs",
        ["dataset_version_id"],
    )
    op.create_table(
        "evaluation_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("evaluation_run_id", sa.String(36), nullable=False, unique=True),
        sa.Column("dataset_version_id", sa.String(36), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_evaluation_reports_model_name", "evaluation_reports", ["model_name"])
    op.create_index("ix_evaluation_reports_dataset_version_id", "evaluation_reports", ["dataset_version_id"])
    op.create_table(
        "gate_policies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("constraints_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_gate_policies_model_name", "gate_policies", ["model_name"])
    op.create_table(
        "deployments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False, unique=True),
        sa.Column("baseline_version", sa.String(64), nullable=True),
        sa.Column("candidate_version", sa.String(64), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("traffic_percentage", sa.Float(), nullable=False, server_default="0"),
        sa.Column("min_requests", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("max_error_rate", sa.Float(), nullable=False, server_default="0.05"),
        sa.Column("max_avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("request_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_sum_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("last_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_deployments_model_name", "deployments", ["model_name"])
    op.create_index("ix_deployments_state", "deployments", ["state"])


def downgrade() -> None:
    op.drop_table("deployments")
    op.drop_table("gate_policies")
    op.drop_table("evaluation_reports")
    op.drop_table("evaluation_runs")
    op.drop_table("dataset_versions")
