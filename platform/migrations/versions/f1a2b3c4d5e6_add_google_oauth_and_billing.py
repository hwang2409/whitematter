"""Add Google OAuth and billing columns to users table.

Revision ID: f1a2b3c4d5e6
Revises: e6a1b2c3d4e5
Create Date: 2026-03-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e6a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_id", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("avatar_url", sa.String(512), nullable=True))
    op.add_column("users", sa.Column("stripe_customer_id", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("plan", sa.String(20), server_default="free", nullable=True))

    op.create_index("ix_users_google_id", "users", ["google_id"], unique=True)
    op.create_index("ix_users_stripe_customer_id", "users", ["stripe_customer_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_stripe_customer_id", table_name="users")
    op.drop_index("ix_users_google_id", table_name="users")
    op.drop_column("users", "plan")
    op.drop_column("users", "stripe_customer_id")
    op.drop_column("users", "avatar_url")
    op.drop_column("users", "google_id")
