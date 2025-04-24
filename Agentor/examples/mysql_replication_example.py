"""
Example demonstrating MySQL replication.

This example shows how to use the MySQL adapter with replication features:
- Replication configuration
- Read/write splitting
- Failover handling
- Replication monitoring
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any, Optional, List, Tuple

from agentor.interfaces.database import (
    MySqlAdapter,
    create_mysql_adapter
)
from agentor.interfaces.database.replication import (
    ReplicationConfig,
    ReplicationMode,
    ReplicationRole,
    ServerConfig,
    MySqlReplicationManager
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


async def create_connection_pool(server_config: ServerConfig) -> MySqlConnectionPool:
    """Create a connection pool for a server.
    
    Args:
        server_config: The server configuration
        
    Returns:
        The connection pool
    """
    # Create a connection pool configuration
    pool_config = ConnectionPoolConfig(
        min_size=1,
        max_size=server_config.max_connections,
        max_lifetime=3600.0,
        connect_timeout=server_config.connection_timeout,
        validation_mode=ConnectionValidationMode.PING,
        validation_interval=60.0,
        health_check_interval=60.0,
        collect_metrics=True
    )
    
    # Create a connection pool
    pool = MySqlConnectionPool(
        name=f"{server_config.host}:{server_config.port}",
        config=pool_config,
        host=server_config.host,
        port=server_config.port,
        user=server_config.user,
        password=server_config.password,
        database=server_config.database,
        charset="utf8mb4"
    )
    
    # Initialize the pool
    await pool.initialize()
    
    return pool


async def test_replication_setup():
    """Test a replication setup."""
    logger.info("Testing replication setup")
    
    # Create a replication configuration
    replication_config = ReplicationConfig(
        mode=ReplicationMode.SINGLE_PRIMARY,
        servers=[
            ServerConfig(
                host="localhost",  # Replace with your primary server host
                port=3306,  # Replace with your primary server port
                user="root",  # Replace with your primary server user
                password="",  # Replace with your primary server password
                database="test",  # Replace with your primary server database
                max_connections=10,
                connection_timeout=10.0,
                role=ReplicationRole.PRIMARY,
                weight=1
            ),
            ServerConfig(
                host="localhost",  # Replace with your replica server host
                port=3307,  # Replace with your replica server port
                user="root",  # Replace with your replica server user
                password="",  # Replace with your replica server password
                database="test",  # Replace with your replica server database
                max_connections=10,
                connection_timeout=10.0,
                role=ReplicationRole.REPLICA,
                weight=1
            )
        ],
        read_write_splitting=True,
        read_from_primary=True,
        write_to_replica=False,
        load_balancing_strategy="round_robin",
        failover_enabled=True,
        failover_timeout=30.0,
        failover_retry_interval=5.0,
        max_failover_attempts=3,
        health_check_interval=60.0,
        health_check_timeout=5.0,
        monitoring_enabled=True,
        monitoring_interval=60.0
    )
    
    # Create a replication manager
    replication_manager = MySqlReplicationManager(
        name="test_replication",
        config=replication_config,
        connection_factory=create_connection_pool
    )
    
    try:
        # Initialize the replication manager
        logger.info("Initializing the replication manager")
        await replication_manager.initialize()
        
        # Get the replication status
        replication_status = await replication_manager.get_replication_status()
        logger.info(f"Replication status: {replication_status}")
        
        # Get the server status
        server_status = await replication_manager.get_server_status()
        logger.info(f"Server status: {server_status}")
        
        # Create test tables
        logger.info("Creating test tables")
        
        # Get a write connection
        write_conn, write_server = await replication_manager.get_write_connection()
        if not write_conn or not write_server:
            logger.error("Failed to get a write connection")
            return
        
        try:
            # Create a users table
            async with write_conn.cursor() as cursor:
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create an orders table
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        amount DECIMAL(10, 2) NOT NULL,
                        status VARCHAR(50) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
        finally:
            # Release the write connection
            await replication_manager.release_connection(write_conn, write_server)
        
        # Insert test data
        logger.info("Inserting test data")
        
        # Insert users
        for i in range(10):
            # Get a write connection
            write_conn, write_server = await replication_manager.get_write_connection()
            if not write_conn or not write_server:
                logger.error("Failed to get a write connection")
                continue
            
            try:
                # Insert a user
                async with write_conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT INTO users (name, email) VALUES (%s, %s)",
                        [f"User {i}", f"user{i}@example.com"]
                    )
            finally:
                # Release the write connection
                await replication_manager.release_connection(write_conn, write_server)
        
        # Insert orders
        for i in range(50):
            # Get a write connection
            write_conn, write_server = await replication_manager.get_write_connection()
            if not write_conn or not write_server:
                logger.error("Failed to get a write connection")
                continue
            
            try:
                # Insert an order
                async with write_conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT INTO orders (user_id, amount, status) VALUES (%s, %s, %s)",
                        [
                            random.randint(1, 10),
                            random.uniform(10.0, 1000.0),
                            random.choice(["pending", "completed", "cancelled"])
                        ]
                    )
            finally:
                # Release the write connection
                await replication_manager.release_connection(write_conn, write_server)
        
        # Test read/write splitting
        logger.info("Testing read/write splitting")
        
        # Perform read operations
        for i in range(20):
            # Get a read connection
            read_conn, read_server = await replication_manager.get_read_connection()
            if not read_conn or not read_server:
                logger.error("Failed to get a read connection")
                continue
            
            try:
                # Execute a read query
                async with read_conn.cursor() as cursor:
                    await cursor.execute(
                        "SELECT * FROM users WHERE id = %s",
                        [random.randint(1, 10)]
                    )
                    result = await cursor.fetchone()
                    
                    logger.info(f"Read from {read_server.id} (primary: {read_server.is_primary_server}): {result}")
            finally:
                # Release the read connection
                await replication_manager.release_connection(read_conn, read_server)
        
        # Perform write operations
        for i in range(5):
            # Get a write connection
            write_conn, write_server = await replication_manager.get_write_connection()
            if not write_conn or not write_server:
                logger.error("Failed to get a write connection")
                continue
            
            try:
                # Execute a write query
                async with write_conn.cursor() as cursor:
                    await cursor.execute(
                        "INSERT INTO orders (user_id, amount, status) VALUES (%s, %s, %s)",
                        [
                            random.randint(1, 10),
                            random.uniform(10.0, 1000.0),
                            random.choice(["pending", "completed", "cancelled"])
                        ]
                    )
                    
                    logger.info(f"Write to {write_server.id} (primary: {write_server.is_primary_server})")
            finally:
                # Release the write connection
                await replication_manager.release_connection(write_conn, write_server)
        
        # Test failover
        logger.info("Testing failover")
        
        # Get the current primary server
        current_primary = replication_manager.current_primary
        if current_primary:
            logger.info(f"Current primary server: {current_primary.id}")
            
            # Simulate a primary server failure
            logger.info("Simulating a primary server failure")
            current_primary.is_connected = False
            
            # Perform a failover
            success = await replication_manager.failover()
            
            if success:
                logger.info(f"Failover successful, new primary server: {replication_manager.current_primary.id}")
            else:
                logger.error("Failover failed")
            
            # Restore the primary server
            current_primary.is_connected = True
        
        # Get replication metrics
        metrics = await replication_manager.get_metrics()
        logger.info(f"Replication metrics: {metrics}")
        
    finally:
        # Clean up
        logger.info("Cleaning up")
        
        # Drop the tables
        write_conn, write_server = await replication_manager.get_write_connection()
        if write_conn and write_server:
            try:
                async with write_conn.cursor() as cursor:
                    await cursor.execute("DROP TABLE IF EXISTS orders")
                    await cursor.execute("DROP TABLE IF EXISTS users")
            finally:
                await replication_manager.release_connection(write_conn, write_server)
        
        # Close the replication manager
        await replication_manager.close()


async def test_mysql_adapter_with_replication():
    """Test the MySQL adapter with replication."""
    logger.info("Testing MySQL adapter with replication")
    
    # Create a MySQL adapter with replication
    mysql = create_mysql_adapter({
        "name": "mysql_replication",
        "mysql_replication_mode": "single_primary",
        "mysql_servers": [
            {
                "host": "localhost",  # Replace with your primary server host
                "port": 3306,  # Replace with your primary server port
                "user": "root",  # Replace with your primary server user
                "password": "",  # Replace with your primary server password
                "database": "test",  # Replace with your primary server database
                "role": "primary"
            },
            {
                "host": "localhost",  # Replace with your replica server host
                "port": 3307,  # Replace with your replica server port
                "user": "root",  # Replace with your replica server user
                "password": "",  # Replace with your replica server password
                "database": "test",  # Replace with your replica server database
                "role": "replica"
            }
        ],
        "mysql_read_write_splitting": True,
        "mysql_read_from_primary": True,
        "mysql_write_to_replica": False,
        "mysql_load_balancing_strategy": "round_robin",
        "mysql_failover_enabled": True,
        "mysql_failover_timeout": 30.0,
        "mysql_health_check_interval": 60.0,
        "mysql_monitoring_enabled": True,
        "mysql_monitoring_interval": 60.0
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create test tables
        logger.info("Creating test tables")
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                stock INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        logger.info("Inserting test data")
        
        # Insert products
        for i in range(10):
            await mysql.execute(
                "INSERT INTO products (name, price, stock) VALUES (:name, :price, :stock)",
                {
                    "name": f"Product {i}",
                    "price": random.uniform(10.0, 1000.0),
                    "stock": random.randint(1, 100)
                }
            )
        
        # Test read operations
        logger.info("Testing read operations")
        
        # Perform read operations
        for i in range(10):
            result = await mysql.fetch_one(
                "SELECT * FROM products WHERE id = :id",
                {"id": random.randint(1, 10)}
            )
            
            if result.success:
                logger.info(f"Read result: {result.data}")
            else:
                logger.error(f"Failed to read: {result.error}")
        
        # Test write operations
        logger.info("Testing write operations")
        
        # Perform write operations
        for i in range(5):
            result = await mysql.execute(
                "UPDATE products SET stock = :stock WHERE id = :id",
                {
                    "id": random.randint(1, 10),
                    "stock": random.randint(1, 100)
                }
            )
            
            if result.success:
                logger.info(f"Write result: {result.affected_rows} rows affected")
            else:
                logger.error(f"Failed to write: {result.error}")
        
        # Get replication status
        replication_status = await mysql.get_replication_status()
        logger.info(f"Replication status: {replication_status}")
        
        # Get metrics
        metrics = await mysql.get_metrics()
        logger.info(f"MySQL adapter metrics: {metrics}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS products")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL replication examples."""
    logger.info("Starting MySQL replication examples")
    
    # Test replication setup
    await test_replication_setup()
    logger.info("-" * 80)
    
    # Test MySQL adapter with replication
    await test_mysql_adapter_with_replication()
    
    logger.info("MySQL replication examples completed")


if __name__ == "__main__":
    asyncio.run(main())
