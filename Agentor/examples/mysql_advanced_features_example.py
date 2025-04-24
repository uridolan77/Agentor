"""
Example demonstrating the advanced MySQL features.

This example shows how to use the MySQL adapter with stored procedures,
user-defined functions, triggers, and events.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, Optional, List

from agentor.interfaces.database import (
    MySqlAdapter,
    create_mysql_adapter
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_stored_procedures():
    """Test stored procedures."""
    logger.info("Testing stored procedures")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_stored_procedures_test",
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
        
        # Create a test table
        logger.info("Creating a test table")
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS stored_procedures_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Create a stored procedure
        logger.info("Creating a stored procedure")
        result = await mysql.execute("""
            DROP PROCEDURE IF EXISTS insert_test_data;
        """)
        if not result.success:
            logger.error(f"Failed to drop procedure: {result.error}")
            return
        
        result = await mysql.execute("""
            CREATE PROCEDURE insert_test_data(IN p_name VARCHAR(255), IN p_value INT)
            BEGIN
                INSERT INTO stored_procedures_test (name, value) VALUES (p_name, p_value);
                SELECT LAST_INSERT_ID() AS id;
            END;
        """)
        if not result.success:
            logger.error(f"Failed to create procedure: {result.error}")
            return
        
        # Call the stored procedure
        logger.info("Calling the stored procedure")
        result = await mysql.call_procedure("insert_test_data", ["test_name", 42])
        if not result.success:
            logger.error(f"Failed to call procedure: {result.error}")
            return
        
        logger.info(f"Procedure result: {result.data}")
        
        # Create a stored procedure that returns multiple result sets
        logger.info("Creating a stored procedure with multiple result sets")
        result = await mysql.execute("""
            DROP PROCEDURE IF EXISTS get_test_data;
        """)
        if not result.success:
            logger.error(f"Failed to drop procedure: {result.error}")
            return
        
        result = await mysql.execute("""
            CREATE PROCEDURE get_test_data()
            BEGIN
                SELECT * FROM stored_procedures_test WHERE value > 20;
                SELECT COUNT(*) AS count FROM stored_procedures_test;
            END;
        """)
        if not result.success:
            logger.error(f"Failed to create procedure: {result.error}")
            return
        
        # Call the stored procedure
        logger.info("Calling the stored procedure with multiple result sets")
        result = await mysql.call_procedure("get_test_data")
        if not result.success:
            logger.error(f"Failed to call procedure: {result.error}")
            return
        
        logger.info(f"Procedure result: {result.data}")
        
        # Get stored procedures
        logger.info("Getting stored procedures")
        result = await mysql.get_procedures()
        if not result.success:
            logger.error(f"Failed to get procedures: {result.error}")
            return
        
        logger.info(f"Procedures: {result.data}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the stored procedures
            logger.info("Dropping the stored procedures")
            await mysql.execute("DROP PROCEDURE IF EXISTS insert_test_data")
            await mysql.execute("DROP PROCEDURE IF EXISTS get_test_data")
            
            # Drop the table
            logger.info("Dropping the table")
            await mysql.execute("DROP TABLE IF EXISTS stored_procedures_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_user_defined_functions():
    """Test user-defined functions."""
    logger.info("Testing user-defined functions")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_udf_test",
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
        
        # Create a user-defined function
        logger.info("Creating a user-defined function")
        result = await mysql.execute("""
            DROP FUNCTION IF EXISTS calculate_factorial;
        """)
        if not result.success:
            logger.error(f"Failed to drop function: {result.error}")
            return
        
        result = await mysql.execute("""
            CREATE FUNCTION calculate_factorial(n INT) RETURNS INT
            DETERMINISTIC
            BEGIN
                DECLARE result INT;
                SET result = 1;
                WHILE n > 0 DO
                    SET result = result * n;
                    SET n = n - 1;
                END WHILE;
                RETURN result;
            END;
        """)
        if not result.success:
            logger.error(f"Failed to create function: {result.error}")
            return
        
        # Call the user-defined function
        logger.info("Calling the user-defined function")
        result = await mysql.call_function("calculate_factorial", [5])
        if not result.success:
            logger.error(f"Failed to call function: {result.error}")
            return
        
        logger.info(f"Function result: {result.data}")
        
        # Get user-defined functions
        logger.info("Getting user-defined functions")
        result = await mysql.get_functions()
        if not result.success:
            logger.error(f"Failed to get functions: {result.error}")
            return
        
        logger.info(f"Functions: {result.data}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the user-defined function
            logger.info("Dropping the user-defined function")
            await mysql.execute("DROP FUNCTION IF EXISTS calculate_factorial")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_triggers():
    """Test triggers."""
    logger.info("Testing triggers")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_triggers_test",
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
        
        # Create test tables
        logger.info("Creating test tables")
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS triggers_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS triggers_audit (
                id INT AUTO_INCREMENT PRIMARY KEY,
                action VARCHAR(50) NOT NULL,
                table_name VARCHAR(50) NOT NULL,
                record_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Create a trigger
        logger.info("Creating a trigger")
        result = await mysql.create_trigger(
            trigger_name="after_insert_triggers_test",
            table_name="triggers_test",
            timing="AFTER",
            event="INSERT",
            body="""
            BEGIN
                INSERT INTO triggers_audit (action, table_name, record_id)
                VALUES ('INSERT', 'triggers_test', NEW.id);
            END
            """
        )
        if not result.success:
            logger.error(f"Failed to create trigger: {result.error}")
            return
        
        # Insert data to trigger the trigger
        logger.info("Inserting data to trigger the trigger")
        result = await mysql.execute(
            "INSERT INTO triggers_test (name, value) VALUES (:name, :value)",
            {"name": "trigger_test", "value": 42}
        )
        if not result.success:
            logger.error(f"Failed to insert data: {result.error}")
            return
        
        # Check if the trigger worked
        logger.info("Checking if the trigger worked")
        result = await mysql.fetch_all("SELECT * FROM triggers_audit")
        if not result.success:
            logger.error(f"Failed to query audit table: {result.error}")
            return
        
        logger.info(f"Audit records: {result.data}")
        
        # Get triggers
        logger.info("Getting triggers")
        result = await mysql.get_triggers()
        if not result.success:
            logger.error(f"Failed to get triggers: {result.error}")
            return
        
        logger.info(f"Triggers: {result.data}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the trigger
            logger.info("Dropping the trigger")
            await mysql.drop_trigger("after_insert_triggers_test")
            
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS triggers_audit")
            await mysql.execute("DROP TABLE IF EXISTS triggers_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_events():
    """Test events."""
    logger.info("Testing events")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_events_test",
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
        
        # Create a test table
        logger.info("Creating a test table")
        result = await mysql.execute("""
            CREATE TABLE IF NOT EXISTS events_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                value INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Enable event scheduler
        logger.info("Enabling event scheduler")
        result = await mysql.execute("SET GLOBAL event_scheduler = ON")
        if not result.success:
            logger.error(f"Failed to enable event scheduler: {result.error}")
            return
        
        # Create an event
        logger.info("Creating an event")
        result = await mysql.create_event(
            event_name="insert_test_data_event",
            schedule="EVERY 1 MINUTE",
            body="""
            BEGIN
                INSERT INTO events_test (name, value)
                VALUES (CONCAT('event_', NOW()), FLOOR(RAND() * 100));
            END
            """,
            on_completion="NOT PRESERVE",
            status="ENABLED",
            comment="Test event that inserts data every minute"
        )
        if not result.success:
            logger.error(f"Failed to create event: {result.error}")
            return
        
        # Wait for the event to execute
        logger.info("Waiting for the event to execute (5 seconds)...")
        await asyncio.sleep(5)
        
        # Check if the event worked
        logger.info("Checking if the event worked")
        result = await mysql.fetch_all("SELECT * FROM events_test")
        if not result.success:
            logger.error(f"Failed to query test table: {result.error}")
            return
        
        logger.info(f"Test records: {result.data}")
        
        # Get events
        logger.info("Getting events")
        result = await mysql.get_events()
        if not result.success:
            logger.error(f"Failed to get events: {result.error}")
            return
        
        logger.info(f"Events: {result.data}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the event
            logger.info("Dropping the event")
            await mysql.drop_event("insert_test_data_event")
            
            # Drop the table
            logger.info("Dropping the table")
            await mysql.execute("DROP TABLE IF EXISTS events_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL advanced features examples."""
    logger.info("Starting MySQL advanced features examples")
    
    # Test stored procedures
    await test_stored_procedures()
    logger.info("-" * 80)
    
    # Test user-defined functions
    await test_user_defined_functions()
    logger.info("-" * 80)
    
    # Test triggers
    await test_triggers()
    logger.info("-" * 80)
    
    # Test events
    await test_events()
    
    logger.info("MySQL advanced features examples completed")


if __name__ == "__main__":
    asyncio.run(main())
