"""merge heads

Revision ID: 0796184dc0c2
Revises: d58bb00add0e, f1a2b3c4d5e6, g1h2i3j4k5l6
Create Date: 2026-03-12 15:46:17.537153

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0796184dc0c2'
down_revision: Union[str, Sequence[str], None] = ('d58bb00add0e', 'f1a2b3c4d5e6', 'g1h2i3j4k5l6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
