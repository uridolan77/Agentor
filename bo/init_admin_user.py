#!/usr/bin/env python

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agentor-backoffice")
logger.setLevel(logging.DEBUG)

# Add the parent directory to Python path to make 'bo' module accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def main():
    """Initialize the database and create the admin user."""
    try:
        logger.info("Initializing database...")
        from bo.backend.db.database import init_db
        init_db()
        logger.info("Database initialized successfully")
        
        logger.info("Creating default admin user...")
        from bo.backend.db.init_data import init_data
        init_data()
        logger.info("Default admin user created successfully")
        
        logger.info("Done! You can now log in with username: admin, password: Admin123")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    main()