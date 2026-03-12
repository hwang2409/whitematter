"""Add conversations and conversation_messages tables.

Revision ID: g1h2i3j4k5l6
Revises: e6a1b2c3d4e5
Create Date: 2026-03-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'g1h2i3j4k5l6'
down_revision: Union[str, Sequence[str], None] = 'e6a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(16), primary_key=True),
        sa.Column('user_id', sa.String(32), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('phase', sa.String(30), server_default='greeting'),
        sa.Column('architecture', sa.JSON, nullable=True),
        sa.Column('dataset_id', sa.String(32), sa.ForeignKey('datasets.id'), nullable=True),
        sa.Column('model_id', sa.String(32), sa.ForeignKey('models.id'), nullable=True),
        sa.Column('training_job_id', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime),
        sa.Column('updated_at', sa.DateTime),
    )

    op.create_table(
        'conversation_messages',
        sa.Column('id', sa.String(16), primary_key=True),
        sa.Column('conversation_id', sa.String(16), sa.ForeignKey('conversations.id'), nullable=False, index=True),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('message_type', sa.String(30), server_default='text'),
        sa.Column('metadata', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime),
    )


def downgrade() -> None:
    op.drop_table('conversation_messages')
    op.drop_table('conversations')
