"""Add architecture column to model_versions.

Promotes architecture from a JSON blob inside `tags` to a proper,
queryable column. Default value 'default' matches the existing convention.

Revision ID: 002
Revises: 001
Create Date: 2026-02-15
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_versions",
        sa.Column(
            "architecture",
            sa.String(64),
            nullable=False,
            server_default="default",
        ),
    )


def downgrade() -> None:
    op.drop_column("model_versions", "architecture")
