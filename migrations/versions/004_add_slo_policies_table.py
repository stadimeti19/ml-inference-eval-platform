"""Add slo_policies table for SLO-based gating.

Stores named policies with absolute constraint thresholds that a model
version must meet before promotion (e.g., p95 < 50ms, accuracy > 0.95).

Revision ID: 004
Revises: 003
Create Date: 2026-02-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "slo_policies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("model_name", sa.String(255), nullable=False, index=True),
        sa.Column("constraints", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("slo_policies")
