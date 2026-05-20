"""Add deployment_events audit table.

Revision ID: 005
Revises: 004
Create Date: 2026-05-20
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "deployment_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("previous_status", sa.String(32), nullable=True),
        sa.Column("new_status", sa.String(32), nullable=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_deployment_events_model_name",
        "deployment_events",
        ["model_name"],
    )
    op.create_index(
        "ix_deployment_events_event_type",
        "deployment_events",
        ["event_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_deployment_events_event_type", table_name="deployment_events")
    op.drop_index("ix_deployment_events_model_name", table_name="deployment_events")
    op.drop_table("deployment_events")
