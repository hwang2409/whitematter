"""Add is_starred column to conversations.

Revision ID: h2i3j4k5l6m7
Revises: 0796184dc0c2
Create Date: 2026-03-13
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'h2i3j4k5l6m7'
down_revision: Union[str, Sequence[str], None] = '0796184dc0c2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversations',
        sa.Column('is_starred', sa.Boolean, server_default=sa.text('false'), nullable=False),
    )


def downgrade() -> None:
    op.drop_column('conversations', 'is_starred')
