"""
Example demonstrating the MySQL adapter.

This example shows how to use the MySQL adapter with resilience patterns,
connection pooling, and other advanced features.
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any, Optional, List

from agentor.interfaces.database import (
    MySqlAdapter,
    create_mysql_adapter
)
from agentor.interfaces.database.config.mysql import MySqlConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_mysql_adapter():
    """Test the MySQL adapter."""
    logger.info("Testing MySQL adapter")
    
    # Create a MySQL adapter using the factory function
    mysql = create_mysql_adapter({
        "name": "mysql_test",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": False
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create a table
        logger.info("Creating a table")
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS mysql_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Insert some data
        logger.info("Inserting data")
        for i in range(10):
            result = await mysql.execute(
                "INSERT INTO mysql_test (name, value) VALUES (:name, :value)",
                {"name": f"test{i}", "value": i}
            )
            if not result.success:
                logger.error(f"Failed to insert data: {result.error}")
                return
        
        # Query the data
        logger.info("Querying data")
        result = await mysql.fetch_all("SELECT * FROM mysql_test")
        if not result.success:
            logger.error(f"Failed to query data: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
        # Test transaction
        logger.info("Testing transaction")
        result = await mysql.begin_transaction()
        if not result.success:
            logger.error(f"Failed to begin transaction: {result.error}")
            return
        
        # Update data in transaction
        result = await mysql.execute(
            "UPDATE mysql_test SET value = value * 2 WHERE value > 5"
        )
        if not result.success:
            logger.error(f"Failed to update data: {result.error}")
            await mysql.rollback_transaction()
            return
        
        # Commit the transaction
        result = await mysql.commit_transaction()
        if not result.success:
            logger.error(f"Failed to commit transaction: {result.error}")
            return
        
        # Query the updated data
        logger.info("Querying updated data")
        result = await mysql.fetch_all("SELECT * FROM mysql_test WHERE value > 5")
        if not result.success:
            logger.error(f"Failed to query updated data: {result.error}")
            return
        
        logger.info(f"Updated data: {result.data}")
        
        # Test prepared statement
        logger.info("Testing prepared statement")
        result = await mysql.execute_prepared_statement(
            "get_by_value",
            "SELECT * FROM mysql_test WHERE value > :value",
            {"value": 10}
        )
        if not result.success:
            logger.error(f"Failed to execute prepared statement: {result.error}")
            return
        
        # Get server info
        logger.info("Getting server info")
        result = await mysql.get_server_info()
        if not result.success:
            logger.error(f"Failed to get server info: {result.error}")
            return
        
        logger.info(f"Server version: {result.data['version']}")
        
        # Get metrics
        metrics = await mysql.get_metrics()
        logger.info(f"MySQL adapter metrics: {metrics}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the table
            logger.info("Dropping the table")
            await mysql.execute("DROP TABLE IF EXISTS mysql_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_mysql_config():
    """Test the MySQL configuration."""
    logger.info("Testing MySQL configuration")
    
    # Create a MySQL configuration
    config = MySqlConfig(
        name="mysql_config_test",
        mysql_user="root",  # Replace with your MySQL user
        mysql_password="",  # Replace with your MySQL password
        mysql_host="localhost",
        mysql_port=3306,
        mysql_database="test",  # Replace with your MySQL database
        mysql_charset="utf8mb4",
        mysql_collation="utf8mb4_unicode_ci",
        mysql_autocommit=False
    )
    
    # Convert to connection parameters
    params = config.to_connection_params()
    logger.info(f"MySQL connection parameters: {params}")
    
    # Create a MySQL adapter using the configuration
    mysql = create_mysql_adapter(config)
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Test a simple query
        logger.info("Testing a simple query")
        result = await mysql.fetch_one("SELECT 1 AS test")
        if not result.success:
            logger.error(f"Failed to execute query: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
    finally:
        # Disconnect
        if mysql.connected:
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_resilience():
    """Test the resilience patterns."""
    logger.info("Testing resilience patterns")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_resilience_test",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": False
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create a table
        logger.info("Creating a table")
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS resilience_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Test retry pattern with a simulated failure
        logger.info("Testing retry pattern with a simulated failure")
        
        # Patch the execute method to simulate failures
        original_execute = mysql.execute
        failure_count = 0
        
        async def mock_execute(query, params=None):
            nonlocal failure_count
            if "INSERT" in query and failure_count < 2:
                failure_count += 1
                logger.info(f"Simulating failure {failure_count}/2")
                return DatabaseResult.error_result(f"Simulated failure {failure_count}")
            return await original_execute(query, params)
        
        # Replace the execute method
        mysql.execute = mock_execute
        
        # Try to insert data (should succeed after retries)
        result = await mysql.execute(
            "INSERT INTO resilience_test (name, value) VALUES (:name, :value)",
            {"name": "retry_test", "value": 42}
        )
        
        # Restore the original execute method
        mysql.execute = original_execute
        
        if not result.success:
            logger.error(f"Failed to insert data after retries: {result.error}")
            return
        
        logger.info("Insert succeeded after retries")
        
        # Query the data to verify
        result = await mysql.fetch_one("SELECT * FROM resilience_test WHERE name = 'retry_test'")
        if not result.success:
            logger.error(f"Failed to query data: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the table
            logger.info("Dropping the table")
            await mysql.execute("DROP TABLE IF EXISTS resilience_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL adapter examples."""
    logger.info("Starting MySQL adapter examples")
    
    # Test the MySQL adapter
    await test_mysql_adapter()
    logger.info("-" * 80)
    
    # Test the MySQL configuration
    await test_mysql_config()
    logger.info("-" * 80)
    
    # Test resilience patterns
    await test_resilience()
    
    logger.info("MySQL adapter examples completed")


if __name__ == "__main__":
    asyncio.run(main())
