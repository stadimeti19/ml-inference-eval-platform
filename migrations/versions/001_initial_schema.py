"""Initial schema: model_versions, batch_jobs, gate_results.

Revision ID: 001
Revises: None
Create Date: 2026-02-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False, index=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("artifact_path", sa.String(512), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("git_sha", sa.String(40), nullable=True),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="staging"),
        sa.Column("metrics", sa.Text(), nullable=True),
    )

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("dataset_id", sa.String(128), nullable=False),
        sa.Column("config", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("result_metrics", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "gate_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("candidate_version", sa.String(64), nullable=False),
        sa.Column("baseline_version", sa.String(64), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("gate_results")
    op.drop_table("batch_jobs")
    op.drop_table("model_versions")
