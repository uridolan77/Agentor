"""
Example demonstrating the MySQL adapter with connection pooling.

This example shows how to use the MySQL adapter with advanced connection pooling features.
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
from agentor.interfaces.database.pool import (
    ConnectionPoolConfig,
    ConnectionValidationMode,
    MySqlConnectionPool
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_connection_pooling():
    """Test connection pooling."""
    logger.info("Testing connection pooling")
    
    # Create a connection pool configuration
    pool_config = ConnectionPoolConfig(
        min_size=2,
        max_size=10,
        target_size=5,
        max_idle_time=300.0,
        max_lifetime=3600.0,
        connect_timeout=10.0,
        validation_mode=ConnectionValidationMode.PING,
        validation_interval=60.0,
        retry_attempts=3,
        retry_delay=1.0,
        health_check_interval=60.0,
        health_check_timeout=5.0,
        scale_up_threshold=0.8,
        scale_down_threshold=0.2,
        scale_up_step=2,
        scale_down_step=1,
        collect_metrics=True,
        metrics_interval=60.0
    )
    
    # Create a MySQL connection pool
    pool = MySqlConnectionPool(
        name="test_pool",
        config=pool_config,
        host="localhost",
        port=3306,
        user="root",  # Replace with your MySQL user
        password="",  # Replace with your MySQL password
        database="test",  # Replace with your MySQL database
        charset="utf8mb4"
    )
    
    try:
        # Initialize the pool
        logger.info("Initializing the connection pool")
        await pool.initialize()
        
        # Get pool metrics
        metrics = await pool.get_metrics()
        logger.info(f"Initial pool metrics: {metrics}")
        
        # Check pool health
        healthy = await pool.check_health()
        logger.info(f"Pool health: {healthy}")
        
        # Create a test table
        logger.info("Creating a test table")
        conn, acquisition_time = await pool.acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS connection_pool_test (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        value INT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        finally:
            await pool.release(conn)
        
        # Insert data using multiple connections
        logger.info("Inserting data using multiple connections")
        tasks = []
        for i in range(20):
            tasks.append(insert_data(pool, i))
        
        await asyncio.gather(*tasks)
        
        # Query data
        logger.info("Querying data")
        conn, acquisition_time = await pool.acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM connection_pool_test")
                count = await cursor.fetchone()
                logger.info(f"Total records: {count[0]}")
        finally:
            await pool.release(conn)
        
        # Get pool metrics after operations
        metrics = await pool.get_metrics()
        logger.info(f"Pool metrics after operations: {metrics}")
        
        # Validate connections
        logger.info("Validating connections")
        await pool.validate_connections()
        
        # Scale the pool
        logger.info("Scaling the pool")
        await pool.scale_pool()
        
        # Get pool metrics after scaling
        metrics = await pool.get_metrics()
        logger.info(f"Pool metrics after scaling: {metrics}")
        
    finally:
        # Clean up
        logger.info("Cleaning up")
        conn, acquisition_time = await pool.acquire()
        try:
            async with conn.cursor() as cursor:
                await cursor.execute("DROP TABLE IF EXISTS connection_pool_test")
        finally:
            await pool.release(conn)
        
        # Close the pool
        logger.info("Closing the connection pool")
        await pool.close()


async def insert_data(pool: MySqlConnectionPool, index: int):
    """Insert data using a connection from the pool.
    
    Args:
        pool: The connection pool
        index: The index of the data
    """
    # Acquire a connection
    conn, acquisition_time = await pool.acquire()
    try:
        # Insert data
        async with conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO connection_pool_test (name, value) VALUES (%s, %s)",
                [f"test{index}", index]
            )
        
        # Simulate some work
        await asyncio.sleep(random.uniform(0.1, 0.5))
    finally:
        # Release the connection
        await pool.release(conn)


async def test_mysql_adapter_with_pooling():
    """Test the MySQL adapter with connection pooling."""
    logger.info("Testing MySQL adapter with connection pooling")
    
    # Create a MySQL adapter with a custom connection pool configuration
    mysql = create_mysql_adapter({
        "name": "mysql_pooling_test",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": False,
        "mysql_pool_min_size": 2,
        "mysql_pool_max_size": 10,
        "mysql_pool_recycle": 3600
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create a test table
        logger.info("Creating a test table")
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS mysql_pooling_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Insert data using multiple concurrent operations
        logger.info("Inserting data using multiple concurrent operations")
        tasks = []
        for i in range(20):
            tasks.append(
                mysql.execute(
                    "INSERT INTO mysql_pooling_test (name, value) VALUES (:name, :value)",
                    {"name": f"test{i}", "value": i}
                )
            )
        
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            if not result.success:
                logger.error(f"Failed to insert data {i}: {result.error}")
        
        # Query data
        logger.info("Querying data")
        result = await mysql.fetch_all("SELECT COUNT(*) AS count FROM mysql_pooling_test")
        if not result.success:
            logger.error(f"Failed to query data: {result.error}")
            return
        
        logger.info(f"Total records: {result.data[0]['count']}")
        
        # Get metrics
        metrics = await mysql.get_metrics()
        logger.info(f"MySQL adapter metrics: {metrics}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the table
            logger.info("Dropping the table")
            await mysql.execute("DROP TABLE IF EXISTS mysql_pooling_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL connection pooling examples."""
    logger.info("Starting MySQL connection pooling examples")
    
    # Test connection pooling
    await test_connection_pooling()
    logger.info("-" * 80)
    
    # Test MySQL adapter with connection pooling
    await test_mysql_adapter_with_pooling()
    
    logger.info("MySQL connection pooling examples completed")


if __name__ == "__main__":
    asyncio.run(main())
