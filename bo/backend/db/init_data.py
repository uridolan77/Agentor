"""Initialize the database with default data."""

import logging
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from .database import SessionLocal
from .models import User, TrainingDataset
from .reporting import DataSource
from bo.backend.auth.utils import get_password_hash

# Setup logging
logger = logging.getLogger("agentor-backoffice.db.init_data")

def create_default_admin(db: Session):
    """Create a default admin user if no users exist."""
    # Check if any users exist
    user_count = db.query(User).count()
    logger.info(f"Found {user_count} existing users")
    
    if user_count > 0:
        logger.info("Users already exist, skipping default admin creation")
        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if admin_user:
            logger.info("Admin user exists")
            # Print admin user details for debugging
            logger.info(f"Admin user details: username={admin_user.username}, role={admin_user.role}, active={admin_user.is_active}")
            # Only update password if FORCE_UPDATE_ADMIN_PASSWORD env var is set
            import os
            if os.environ.get("FORCE_UPDATE_ADMIN_PASSWORD"):
                admin_user.hashed_password = get_password_hash("Admin123")
                db.commit()
                logger.info("Admin password updated for testing")
            else:
                logger.info("Skipping admin password update")
        return
    
    try:
        # Create default admin user
        logger.info("Creating default admin user")
        hashed_password = get_password_hash("Admin123")
        logger.info(f"Hashed password created: {hashed_password[:10]}...")
        
        admin_user = User(
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            hashed_password=hashed_password,
            is_active=True,
            role="admin"
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        logger.info(f"Default admin user created successfully with ID: {admin_user.id}")
    except Exception as e:
        logger.warning(f"Error creating admin user: {e}")
        db.rollback()
        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if admin_user:
            logger.info("Admin user already exists, updating password")
            admin_user.hashed_password = get_password_hash("Admin123")
            db.commit()
            logger.info("Admin password updated")
        else:
            logger.error("Failed to create admin user and admin user does not exist")

def create_default_data_source(db: Session):
    """Create a default MySQL data source if none exist."""
    # Check if any data sources exist
    data_source_count = db.query(DataSource).count()
    logger.info(f"Found {data_source_count} existing data sources")
    
    if data_source_count > 0:
        logger.info("Data sources already exist, skipping default data source creation")
        return
    
    try:
        # Create default MySQL data source
        logger.info("Creating default MySQL data source")
        
        # MySQL connection configuration
        connection_config = {
            "host": "localhost",
            "port": "3306",
            "username": "root",
            "password": "EKDE1xTGUkr_mnM3UCKd",
            "database": "DailyActionsDB",
            "ssl_enabled": False
        }
        
        default_data_source = DataSource(
            id=str(uuid.uuid4()),
            name="Default MySQL Database",
            description="Default MySQL database connection for the Data Canvas",
            connection_type="mysql",
            connection_config=connection_config,
            creator_id=1,  # Assuming admin user has ID 1
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_active=True
        )
        
        db.add(default_data_source)
        db.commit()
        db.refresh(default_data_source)
        logger.info(f"Default data source created successfully with ID: {default_data_source.id}")
    except Exception as e:
        logger.warning(f"Error creating default data source: {e}")
        db.rollback()

def create_default_datasets(db: Session):
    """Create default datasets for training if none exist."""
    # Check if any datasets exist
    dataset_count = db.query(TrainingDataset).count()
    logger.info(f"Found {dataset_count} existing datasets")
    
    if dataset_count > 0:
        logger.info("Datasets already exist, skipping default dataset creation")
        return
    
    try:
        # Create default datasets
        logger.info("Creating default datasets")
        
        datasets = [
            {
                "id": str(uuid.uuid4()),
                "name": "Text Classification Dataset",
                "description": "A dataset for text classification tasks",
                "format": "json",
                "size": 1024000,  # 1MB
                "dataset_metadata": '{"classes": ["positive", "negative", "neutral"], "samples": 5000}',
                "created_at": datetime.now()
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Named Entity Recognition Dataset",
                "description": "A dataset for NER tasks",
                "format": "json",
                "size": 2048000,  # 2MB
                "dataset_metadata": '{"entities": ["person", "organization", "location", "date"], "samples": 3000}',
                "created_at": datetime.now()
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Question Answering Dataset",
                "description": "A dataset for QA tasks",
                "format": "json",
                "size": 3072000,  # 3MB
                "dataset_metadata": '{"contexts": 500, "questions": 2000, "avg_context_length": 250}',
                "created_at": datetime.now()
            }
        ]
        
        for dataset_data in datasets:
            dataset = TrainingDataset(**dataset_data)
            db.add(dataset)
        
        db.commit()
        logger.info(f"Created {len(datasets)} default datasets")
    except Exception as e:
        logger.warning(f"Error creating default datasets: {e}")
        db.rollback()


def init_data():
    """Initialize the database with default data."""
    logger.info("Initializing database data")
    
    # Create a database session
    db = SessionLocal()
    try:
        # Create default admin user
        create_default_admin(db)
        
        # Create default data source
        create_default_data_source(db)
        
        # Create default datasets
        create_default_datasets(db)
        
        logger.info("Database data initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database data: {e}")
        raise
    finally:
        db.close()
