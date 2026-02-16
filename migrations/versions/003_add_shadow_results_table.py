"""Add shadow_results table for canary/shadow deployments.

Stores per-request comparisons between production and shadow (candidate)
model predictions, enabling safe rollout validation on live traffic.

Revision ID: 003
Revises: 002
Create Date: 2026-02-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shadow_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False, index=True),
        sa.Column("prod_version", sa.String(64), nullable=False),
        sa.Column("shadow_version", sa.String(64), nullable=False),
        sa.Column("prod_prediction", sa.Integer(), nullable=False),
        sa.Column("shadow_prediction", sa.Integer(), nullable=False),
        sa.Column("agreed", sa.Boolean(), nullable=False),
        sa.Column("prod_latency_ms", sa.Float(), nullable=False),
        sa.Column("shadow_latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("shadow_results")
