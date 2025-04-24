from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger("agentor-backoffice.db")

# Get database URL from environment variable or use SQLite as default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agentor_backoffice.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db():
    """Get a database session.

    Yields:
        A database session

    Notes:
        This function is used as a dependency in FastAPI endpoints.
        It creates a new session for each request and closes it when the request is done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def get_db_session():
    """Get an async database session.

    Yields:
        A database session

    Notes:
        This function is used as an async context manager for database operations.
        It creates a new session and closes it when the context is exited.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database.

    This function creates all tables in the database.
    """
    try:
        # Import all models to ensure they are registered with the Base metadata
        from .models import User, Team, Agent, Tool, Workflow, LLMConnection
        # Training models
        from .models import TrainingDataset, TrainingSession, TrainingModel
        # Reporting models
        from .reporting import DataSource, Dimension, Metric, CalculatedMetric, Report, ReportPermission
        from .reporting import DataModel, DataModelTable, DataModelRelationship

        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
