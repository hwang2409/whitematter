"""
Database connection and session management.

All data is stored in a single SQLite database file (or PostgreSQL).
Binary blobs (model weights, datasets) are stored directly in the database
using LargeBinary columns - no separate filesystem storage needed.
"""
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".whitematter"
DATA_DIR = Path(os.environ.get("WHITEMATTER_DATA_DIR", DEFAULT_DATA_DIR))

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Database URL (defaults to SQLite for dev)
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'whitematter.db'}"
)

# Create engine
# For SQLite, we need special settings for thread safety
if DATABASE_URL.startswith("sqlite://"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL or other databases
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database session.
    Usage: with get_db_session() as db: ...
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_data_dir() -> Path:
    """Get the data directory path."""
    return DATA_DIR
