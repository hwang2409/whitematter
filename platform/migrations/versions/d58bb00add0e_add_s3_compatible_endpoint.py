"""add s3-compatible endpoint_url and provider to aws_credentials

Revision ID: d58bb00add0e
Revises: d54aa7c96d7c
Create Date: 2026-03-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d58bb00add0e"
down_revision: Union[str, Sequence[str], None] = "d54aa7c96d7c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("aws_credentials", sa.Column("endpoint_url", sa.String(length=512), nullable=True))
    op.add_column("aws_credentials", sa.Column("provider", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("aws_credentials", "provider")
    op.drop_column("aws_credentials", "endpoint_url")
