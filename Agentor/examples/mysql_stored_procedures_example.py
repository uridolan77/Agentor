"""
Example demonstrating MySQL stored procedures, functions, triggers, and events.

This example shows how to use the MySQL adapter with advanced features:
- Stored procedures
- User-defined functions
- Triggers
- Event scheduling
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
from agentor.interfaces.database.procedure import (
    ProcedureConfig,
    MySqlProcedureManager
)
from agentor.interfaces.database.function import (
    FunctionConfig,
    MySqlFunctionManager
)
from agentor.interfaces.database.trigger import (
    TriggerConfig,
    MySqlTriggerManager
)
from agentor.interfaces.database.event import (
    EventConfig,
    MySqlEventManager
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
        "name": "mysql_procedures",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": True
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create a procedure configuration
        procedure_config = ProcedureConfig(
            cache_metadata=True,
            cache_ttl=3600.0,
            timeout=30.0,
            max_rows=10000,
            log_calls=True,
            log_results=True,
            allow_create=True,
            allow_alter=True,
            allow_drop=True
        )
        
        # Create a procedure manager
        procedure_manager = MySqlProcedureManager(
            name="test_procedures",
            config=procedure_config,
            connection_func=lambda: mysql.pool
        )
        
        # Initialize the procedure manager
        await procedure_manager.initialize()
        
        # Create test tables
        logger.info("Creating test tables")
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                amount DECIMAL(10, 2) NOT NULL,
                status VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Insert test data
        logger.info("Inserting test data")
        
        # Insert users
        for i in range(10):
            await mysql.execute(
                "INSERT INTO users (name, email) VALUES (:name, :email)",
                {"name": f"User {i}", "email": f"user{i}@example.com"}
            )
        
        # Insert orders
        for i in range(50):
            await mysql.execute(
                "INSERT INTO orders (user_id, amount, status) VALUES (:user_id, :amount, :status)",
                {
                    "user_id": random.randint(1, 10),
                    "amount": random.uniform(10.0, 1000.0),
                    "status": random.choice(["pending", "completed", "cancelled"])
                }
            )
        
        # Create a stored procedure
        logger.info("Creating a stored procedure")
        success, error = await procedure_manager.create_procedure(
            procedure_name="get_user_orders",
            params=[
                {"name": "p_user_id", "mode": "IN", "type": "INT"},
                {"name": "p_status", "mode": "IN", "type": "VARCHAR(50)"}
            ],
            body="""
            BEGIN
                SELECT
                    o.id,
                    o.amount,
                    o.status,
                    o.created_at
                FROM
                    orders o
                WHERE
                    o.user_id = p_user_id
                    AND (p_status IS NULL OR o.status = p_status)
                ORDER BY
                    o.created_at DESC;
            END
            """
        )
        
        if not success:
            logger.error(f"Failed to create procedure: {error}")
            return
        
        logger.info("Stored procedure created successfully")
        
        # Get procedure metadata
        metadata = await procedure_manager.get_procedure_metadata("get_user_orders")
        logger.info(f"Procedure metadata: {metadata.to_dict() if metadata else None}")
        
        # Call the stored procedure
        logger.info("Calling the stored procedure")
        success, results, error = await procedure_manager.call_procedure(
            procedure_name="get_user_orders",
            params=[1, "completed"]
        )
        
        if not success:
            logger.error(f"Failed to call procedure: {error}")
        else:
            logger.info(f"Procedure results: {results}")
        
        # Create another stored procedure with output parameters
        logger.info("Creating a stored procedure with output parameters")
        success, error = await procedure_manager.create_procedure(
            procedure_name="get_user_order_stats",
            params=[
                {"name": "p_user_id", "mode": "IN", "type": "INT"},
                {"name": "p_total_orders", "mode": "OUT", "type": "INT"},
                {"name": "p_total_amount", "mode": "OUT", "type": "DECIMAL(10, 2)"},
                {"name": "p_avg_amount", "mode": "OUT", "type": "DECIMAL(10, 2)"}
            ],
            body="""
            BEGIN
                SELECT
                    COUNT(*),
                    IFNULL(SUM(amount), 0),
                    IFNULL(AVG(amount), 0)
                INTO
                    p_total_orders,
                    p_total_amount,
                    p_avg_amount
                FROM
                    orders
                WHERE
                    user_id = p_user_id;
            END
            """
        )
        
        if not success:
            logger.error(f"Failed to create procedure: {error}")
            return
        
        logger.info("Stored procedure with output parameters created successfully")
        
        # List procedures
        procedures = await procedure_manager.list_procedures()
        logger.info(f"Procedures: {procedures}")
        
        # Alter a procedure
        logger.info("Altering a procedure")
        success, error = await procedure_manager.alter_procedure(
            procedure_name="get_user_orders",
            params=[
                {"name": "p_user_id", "mode": "IN", "type": "INT"},
                {"name": "p_status", "mode": "IN", "type": "VARCHAR(50)"},
                {"name": "p_limit", "mode": "IN", "type": "INT"}
            ],
            body="""
            BEGIN
                SELECT
                    o.id,
                    o.amount,
                    o.status,
                    o.created_at
                FROM
                    orders o
                WHERE
                    o.user_id = p_user_id
                    AND (p_status IS NULL OR o.status = p_status)
                ORDER BY
                    o.created_at DESC
                LIMIT
                    p_limit;
            END
            """
        )
        
        if not success:
            logger.error(f"Failed to alter procedure: {error}")
        else:
            logger.info("Procedure altered successfully")
        
        # Call the altered procedure
        logger.info("Calling the altered procedure")
        success, results, error = await procedure_manager.call_procedure(
            procedure_name="get_user_orders",
            params=[1, "completed", 5]
        )
        
        if not success:
            logger.error(f"Failed to call procedure: {error}")
        else:
            logger.info(f"Procedure results: {results}")
        
        # Get procedure metrics
        metrics = await procedure_manager.get_metrics()
        logger.info(f"Procedure metrics: {metrics}")
        
        # Drop procedures
        logger.info("Dropping procedures")
        success, error = await procedure_manager.drop_procedure("get_user_orders")
        if not success:
            logger.error(f"Failed to drop procedure: {error}")
        
        success, error = await procedure_manager.drop_procedure("get_user_order_stats")
        if not success:
            logger.error(f"Failed to drop procedure: {error}")
        
        # Close the procedure manager
        await procedure_manager.close()
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS orders")
            await mysql.execute("DROP TABLE IF EXISTS users")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_user_defined_functions():
    """Test user-defined functions."""
    logger.info("Testing user-defined functions")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_functions",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": True
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create a function configuration
        function_config = FunctionConfig(
            cache_metadata=True,
            cache_ttl=3600.0,
            timeout=30.0,
            log_calls=True,
            log_results=True,
            allow_create=True,
            allow_alter=True,
            allow_drop=True
        )
        
        # Create a function manager
        function_manager = MySqlFunctionManager(
            name="test_functions",
            config=function_config,
            connection_func=lambda: mysql.pool
        )
        
        # Initialize the function manager
        await function_manager.initialize()
        
        # Create a user-defined function
        logger.info("Creating a user-defined function")
        success, error = await function_manager.create_function(
            function_name="calculate_discount",
            params=[
                {"name": "p_amount", "type": "DECIMAL(10, 2)"},
                {"name": "p_discount_percent", "type": "DECIMAL(5, 2)"}
            ],
            returns={"type": "DECIMAL(10, 2)"},
            body="""
            BEGIN
                DECLARE discount DECIMAL(10, 2);
                SET discount = p_amount * (p_discount_percent / 100);
                RETURN p_amount - discount;
            END
            """,
            is_deterministic=True,
            data_access="NO SQL"
        )
        
        if not success:
            logger.error(f"Failed to create function: {error}")
            return
        
        logger.info("User-defined function created successfully")
        
        # Get function metadata
        metadata = await function_manager.get_function_metadata("calculate_discount")
        logger.info(f"Function metadata: {metadata.to_dict() if metadata else None}")
        
        # Call the function
        logger.info("Calling the function")
        success, result, error = await function_manager.call_function(
            function_name="calculate_discount",
            params=[100.00, 20.00]
        )
        
        if not success:
            logger.error(f"Failed to call function: {error}")
        else:
            logger.info(f"Function result: {result}")
        
        # Create another function
        logger.info("Creating another function")
        success, error = await function_manager.create_function(
            function_name="format_currency",
            params=[
                {"name": "p_amount", "type": "DECIMAL(10, 2)"},
                {"name": "p_currency", "type": "VARCHAR(3)"}
            ],
            returns={"type": "VARCHAR(20)"},
            body="""
            BEGIN
                RETURN CONCAT(p_currency, ' ', FORMAT(p_amount, 2));
            END
            """,
            is_deterministic=True,
            data_access="NO SQL"
        )
        
        if not success:
            logger.error(f"Failed to create function: {error}")
            return
        
        logger.info("Second function created successfully")
        
        # List functions
        functions = await function_manager.list_functions()
        logger.info(f"Functions: {functions}")
        
        # Alter a function
        logger.info("Altering a function")
        success, error = await function_manager.alter_function(
            function_name="calculate_discount",
            params=[
                {"name": "p_amount", "type": "DECIMAL(10, 2)"},
                {"name": "p_discount_percent", "type": "DECIMAL(5, 2)"},
                {"name": "p_min_discount", "type": "DECIMAL(10, 2)"}
            ],
            returns={"type": "DECIMAL(10, 2)"},
            body="""
            BEGIN
                DECLARE discount DECIMAL(10, 2);
                SET discount = p_amount * (p_discount_percent / 100);
                IF discount < p_min_discount THEN
                    SET discount = p_min_discount;
                END IF;
                RETURN p_amount - discount;
            END
            """,
            is_deterministic=True,
            data_access="NO SQL"
        )
        
        if not success:
            logger.error(f"Failed to alter function: {error}")
        else:
            logger.info("Function altered successfully")
        
        # Call the altered function
        logger.info("Calling the altered function")
        success, result, error = await function_manager.call_function(
            function_name="calculate_discount",
            params=[100.00, 5.00, 10.00]
        )
        
        if not success:
            logger.error(f"Failed to call function: {error}")
        else:
            logger.info(f"Function result: {result}")
        
        # Get function metrics
        metrics = await function_manager.get_metrics()
        logger.info(f"Function metrics: {metrics}")
        
        # Drop functions
        logger.info("Dropping functions")
        success, error = await function_manager.drop_function("calculate_discount")
        if not success:
            logger.error(f"Failed to drop function: {error}")
        
        success, error = await function_manager.drop_function("format_currency")
        if not success:
            logger.error(f"Failed to drop function: {error}")
        
        # Close the function manager
        await function_manager.close()
        
    finally:
        # Disconnect
        if mysql.connected:
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_triggers():
    """Test triggers."""
    logger.info("Testing triggers")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_triggers",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": True
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create a trigger configuration
        trigger_config = TriggerConfig(
            cache_metadata=True,
            cache_ttl=3600.0,
            log_operations=True,
            allow_create=True,
            allow_drop=True
        )
        
        # Create a trigger manager
        trigger_manager = MySqlTriggerManager(
            name="test_triggers",
            config=trigger_config,
            connection_func=lambda: mysql.pool
        )
        
        # Initialize the trigger manager
        await trigger_manager.initialize()
        
        # Create test tables
        logger.info("Creating test tables")
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                stock INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS product_audit (
                id INT AUTO_INCREMENT PRIMARY KEY,
                product_id INT NOT NULL,
                action VARCHAR(10) NOT NULL,
                old_price DECIMAL(10, 2) NULL,
                new_price DECIMAL(10, 2) NULL,
                old_stock INT NULL,
                new_stock INT NULL,
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
        
        # Create triggers
        logger.info("Creating triggers")
        
        # Create an INSERT trigger
        success, error = await trigger_manager.create_trigger(
            trigger_name="products_after_insert",
            table_name="products",
            event="INSERT",
            timing="AFTER",
            body="""
            BEGIN
                INSERT INTO product_audit (product_id, action, new_price, new_stock)
                VALUES (NEW.id, 'INSERT', NEW.price, NEW.stock);
            END
            """
        )
        
        if not success:
            logger.error(f"Failed to create INSERT trigger: {error}")
            return
        
        logger.info("INSERT trigger created successfully")
        
        # Create an UPDATE trigger
        success, error = await trigger_manager.create_trigger(
            trigger_name="products_after_update",
            table_name="products",
            event="UPDATE",
            timing="AFTER",
            body="""
            BEGIN
                INSERT INTO product_audit (product_id, action, old_price, new_price, old_stock, new_stock)
                VALUES (NEW.id, 'UPDATE', OLD.price, NEW.price, OLD.stock, NEW.stock);
            END
            """
        )
        
        if not success:
            logger.error(f"Failed to create UPDATE trigger: {error}")
            return
        
        logger.info("UPDATE trigger created successfully")
        
        # Create a DELETE trigger
        success, error = await trigger_manager.create_trigger(
            trigger_name="products_after_delete",
            table_name="products",
            event="DELETE",
            timing="AFTER",
            body="""
            BEGIN
                INSERT INTO product_audit (product_id, action, old_price, old_stock)
                VALUES (OLD.id, 'DELETE', OLD.price, OLD.stock);
            END
            """
        )
        
        if not success:
            logger.error(f"Failed to create DELETE trigger: {error}")
            return
        
        logger.info("DELETE trigger created successfully")
        
        # Get trigger metadata
        metadata = await trigger_manager.get_trigger_metadata("products_after_update")
        logger.info(f"Trigger metadata: {metadata.to_dict() if metadata else None}")
        
        # List triggers
        triggers = await trigger_manager.list_triggers()
        logger.info(f"Triggers: {triggers}")
        
        # Test the triggers
        logger.info("Testing the triggers")
        
        # Test INSERT trigger
        logger.info("Testing INSERT trigger")
        await mysql.execute(
            "INSERT INTO products (name, price, stock) VALUES (:name, :price, :stock)",
            {
                "name": "Test Product",
                "price": 99.99,
                "stock": 50
            }
        )
        
        # Test UPDATE trigger
        logger.info("Testing UPDATE trigger")
        await mysql.execute(
            "UPDATE products SET price = :price, stock = :stock WHERE name = :name",
            {
                "name": "Test Product",
                "price": 149.99,
                "stock": 75
            }
        )
        
        # Test DELETE trigger
        logger.info("Testing DELETE trigger")
        await mysql.execute(
            "DELETE FROM products WHERE name = :name",
            {
                "name": "Test Product"
            }
        )
        
        # Check the audit table
        logger.info("Checking the audit table")
        result = await mysql.fetch_all("SELECT * FROM product_audit ORDER BY id DESC LIMIT 3")
        if result.success:
            logger.info(f"Audit records: {result.data}")
        
        # Get trigger metrics
        metrics = await trigger_manager.get_metrics()
        logger.info(f"Trigger metrics: {metrics}")
        
        # Drop triggers
        logger.info("Dropping triggers")
        success, error = await trigger_manager.drop_trigger("products_after_insert")
        if not success:
            logger.error(f"Failed to drop trigger: {error}")
        
        success, error = await trigger_manager.drop_trigger("products_after_update")
        if not success:
            logger.error(f"Failed to drop trigger: {error}")
        
        success, error = await trigger_manager.drop_trigger("products_after_delete")
        if not success:
            logger.error(f"Failed to drop trigger: {error}")
        
        # Close the trigger manager
        await trigger_manager.close()
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS product_audit")
            await mysql.execute("DROP TABLE IF EXISTS products")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_events():
    """Test events."""
    logger.info("Testing events")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_events",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": True
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create an event configuration
        event_config = EventConfig(
            cache_metadata=True,
            cache_ttl=3600.0,
            log_operations=True,
            allow_create=True,
            allow_alter=True,
            allow_drop=True
        )
        
        # Create an event manager
        event_manager = MySqlEventManager(
            name="test_events",
            config=event_config,
            connection_func=lambda: mysql.pool
        )
        
        # Initialize the event manager
        await event_manager.initialize()
        
        # Create test tables
        logger.info("Creating test tables")
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_name VARCHAR(255) NOT NULL,
                message VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create events
        logger.info("Creating events")
        
        # Create a one-time event
        success, error = await event_manager.create_event(
            event_name="one_time_event",
            body="""
            BEGIN
                INSERT INTO event_log (event_name, message)
                VALUES ('one_time_event', 'This is a one-time event');
            END
            """,
            execute_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 60)),  # 1 minute from now
            status="ENABLED",
            on_completion="NOT PRESERVE",
            comment="A one-time event example"
        )
        
        if not success:
            logger.error(f"Failed to create one-time event: {error}")
            return
        
        logger.info("One-time event created successfully")
        
        # Create a recurring event
        success, error = await event_manager.create_event(
            event_name="recurring_event",
            body="""
            BEGIN
                INSERT INTO event_log (event_name, message)
                VALUES ('recurring_event', CONCAT('This is a recurring event at ', NOW()));
            END
            """,
            interval_value=1,
            interval_field="MINUTE",
            starts=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 30)),  # 30 seconds from now
            ends=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 300)),  # 5 minutes from now
            status="ENABLED",
            on_completion="NOT PRESERVE",
            comment="A recurring event example"
        )
        
        if not success:
            logger.error(f"Failed to create recurring event: {error}")
            return
        
        logger.info("Recurring event created successfully")
        
        # Get event metadata
        metadata = await event_manager.get_event_metadata("recurring_event")
        logger.info(f"Event metadata: {metadata.to_dict() if metadata else None}")
        
        # List events
        events = await event_manager.list_events()
        logger.info(f"Events: {events}")
        
        # Alter an event
        logger.info("Altering an event")
        success, error = await event_manager.alter_event(
            event_name="recurring_event",
            interval_value=2,  # Change to every 2 minutes
            comment="An altered recurring event example"
        )
        
        if not success:
            logger.error(f"Failed to alter event: {error}")
        else:
            logger.info("Event altered successfully")
        
        # Wait for events to execute
        logger.info("Waiting for events to execute (60 seconds)...")
        await asyncio.sleep(60)
        
        # Check the event log
        logger.info("Checking the event log")
        result = await mysql.fetch_all("SELECT * FROM event_log ORDER BY id DESC")
        if result.success:
            logger.info(f"Event log records: {result.data}")
        
        # Get event metrics
        metrics = await event_manager.get_metrics()
        logger.info(f"Event metrics: {metrics}")
        
        # Drop events
        logger.info("Dropping events")
        success, error = await event_manager.drop_event("one_time_event")
        if not success:
            logger.error(f"Failed to drop event: {error}")
        
        success, error = await event_manager.drop_event("recurring_event")
        if not success:
            logger.error(f"Failed to drop event: {error}")
        
        # Close the event manager
        await event_manager.close()
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS event_log")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL stored procedures, functions, triggers, and events examples."""
    logger.info("Starting MySQL stored procedures, functions, triggers, and events examples")
    
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
    
    logger.info("MySQL stored procedures, functions, triggers, and events examples completed")


if __name__ == "__main__":
    asyncio.run(main())
