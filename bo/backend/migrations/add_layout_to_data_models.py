"""Add layout column to data_models table."""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

def upgrade():
    """Add layout column to data_models table."""
    op.add_column('data_models', sa.Column('layout', sa.String(), nullable=True))

def downgrade():
    """Remove layout column from data_models table."""
    op.drop_column('data_models', 'layout')
