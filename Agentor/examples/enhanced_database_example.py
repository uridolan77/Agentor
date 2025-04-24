"""
Example demonstrating the enhanced database interfaces.

This example shows how to use the enhanced database interfaces with resilience patterns,
connection pooling, and other advanced features.
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any, Optional, List

from agentor.interfaces.database.sql import SqlDialect
from agentor.interfaces.database.nosql import NoSqlType
from agentor.interfaces.database.enhanced import (
    EnhancedSqlConnection,
    EnhancedDocumentStore,
    EnhancedKeyValueStore,
    EnhancedGraphDatabase,
    create_database_connection
)
from agentor.interfaces.database.config import (
    SqlDatabaseConfig,
    DocumentStoreConfig,
    KeyValueStoreConfig,
    GraphDatabaseConfig
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_sql_connection():
    """Test the enhanced SQL connection."""
    logger.info("Testing enhanced SQL connection")
    
    # Create an enhanced SQL connection
    sql = EnhancedSqlConnection(
        name="sqlite",
        dialect=SqlDialect.SQLITE,
        connection_string="sqlite:///test.db"
    )
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await sql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        # Create a table
        logger.info("Creating a table")
        result = await sql.execute("""
            CREATE TABLE IF NOT EXISTS test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Insert some data
        logger.info("Inserting data")
        for i in range(10):
            result = await sql.execute(
                "INSERT INTO test (name, value) VALUES (:name, :value)",
                {"name": f"test{i}", "value": i}
            )
            if not result.success:
                logger.error(f"Failed to insert data: {result.error}")
                return
        
        # Query the data
        logger.info("Querying data")
        result = await sql.fetch_all("SELECT * FROM test")
        if not result.success:
            logger.error(f"Failed to query data: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
        # Test transaction
        logger.info("Testing transaction")
        result = await sql.begin_transaction()
        if not result.success:
            logger.error(f"Failed to begin transaction: {result.error}")
            return
        
        # Update data in transaction
        result = await sql.execute(
            "UPDATE test SET value = value * 2 WHERE value > 5"
        )
        if not result.success:
            logger.error(f"Failed to update data: {result.error}")
            await sql.rollback_transaction()
            return
        
        # Commit the transaction
        result = await sql.commit_transaction()
        if not result.success:
            logger.error(f"Failed to commit transaction: {result.error}")
            return
        
        # Query the updated data
        logger.info("Querying updated data")
        result = await sql.fetch_all("SELECT * FROM test WHERE value > 5")
        if not result.success:
            logger.error(f"Failed to query updated data: {result.error}")
            return
        
        logger.info(f"Updated data: {result.data}")
        
        # Get metrics
        metrics = await sql.get_metrics()
        logger.info(f"SQL connection metrics: {metrics}")
        
    finally:
        # Clean up
        if sql.connected:
            # Drop the table
            logger.info("Dropping the table")
            await sql.execute("DROP TABLE IF EXISTS test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await sql.disconnect()


async def test_document_store():
    """Test the enhanced document store."""
    logger.info("Testing enhanced document store")
    
    # Skip this test if MongoDB is not available
    logger.info("Skipping document store test (requires MongoDB)")
    return
    
    # Create an enhanced document store
    document_store = EnhancedDocumentStore(
        name="mongodb",
        db_type=NoSqlType.MONGODB,
        connection_string="mongodb://localhost:27017",
        database_name="agentor_test"
    )
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await document_store.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        # Insert some documents
        logger.info("Inserting documents")
        for i in range(10):
            result = await document_store.insert_document(
                "test_collection",
                {"name": f"test{i}", "value": i}
            )
            if not result.success:
                logger.error(f"Failed to insert document: {result.error}")
                return
        
        # Query documents
        logger.info("Querying documents")
        result = await document_store.query_documents(
            "test_collection",
            {"value": {"$gt": 5}}
        )
        if not result.success:
            logger.error(f"Failed to query documents: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
        # Update a document
        logger.info("Updating a document")
        result = await document_store.update_document(
            "test_collection",
            result.data[0]["_id"],
            {"$set": {"value": 100}}
        )
        if not result.success:
            logger.error(f"Failed to update document: {result.error}")
            return
        
        # Get the updated document
        logger.info("Getting the updated document")
        result = await document_store.get_document(
            "test_collection",
            result.data[0]["_id"]
        )
        if not result.success:
            logger.error(f"Failed to get document: {result.error}")
            return
        
        logger.info(f"Updated document: {result.data}")
        
        # Count documents
        logger.info("Counting documents")
        result = await document_store.count_documents(
            "test_collection",
            {"value": {"$gt": 5}}
        )
        if not result.success:
            logger.error(f"Failed to count documents: {result.error}")
            return
        
        logger.info(f"Document count: {result.data}")
        
        # Get metrics
        metrics = await document_store.get_metrics()
        logger.info(f"Document store metrics: {metrics}")
        
    finally:
        # Clean up
        if document_store.connected:
            # Delete all documents
            logger.info("Deleting all documents")
            result = await document_store.query_documents("test_collection", {})
            if result.success:
                for doc in result.data:
                    await document_store.delete_document("test_collection", doc["_id"])
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await document_store.disconnect()


async def test_key_value_store():
    """Test the enhanced key-value store."""
    logger.info("Testing enhanced key-value store")
    
    # Skip this test if Redis is not available
    logger.info("Skipping key-value store test (requires Redis)")
    return
    
    # Create an enhanced key-value store
    key_value_store = EnhancedKeyValueStore(
        name="redis",
        db_type=NoSqlType.REDIS,
        connection_string="redis://localhost:6379"
    )
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await key_value_store.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        # Set some values
        logger.info("Setting values")
        for i in range(10):
            result = await key_value_store.set(f"test:{i}", f"value{i}")
            if not result.success:
                logger.error(f"Failed to set value: {result.error}")
                return
        
        # Get a value
        logger.info("Getting a value")
        result = await key_value_store.get("test:5")
        if not result.success:
            logger.error(f"Failed to get value: {result.error}")
            return
        
        logger.info(f"Value: {result.data}")
        
        # Check if a key exists
        logger.info("Checking if a key exists")
        result = await key_value_store.exists("test:5")
        if not result.success:
            logger.error(f"Failed to check if key exists: {result.error}")
            return
        
        logger.info(f"Key exists: {result.data}")
        
        # Increment a value
        logger.info("Incrementing a value")
        result = await key_value_store.set("test:counter", "0")
        if not result.success:
            logger.error(f"Failed to set counter: {result.error}")
            return
        
        result = await key_value_store.increment("test:counter", 5)
        if not result.success:
            logger.error(f"Failed to increment counter: {result.error}")
            return
        
        logger.info(f"Incremented value: {result.data}")
        
        # Set expiration
        logger.info("Setting expiration")
        result = await key_value_store.expire("test:5", 10)
        if not result.success:
            logger.error(f"Failed to set expiration: {result.error}")
            return
        
        # Get metrics
        metrics = await key_value_store.get_metrics()
        logger.info(f"Key-value store metrics: {metrics}")
        
    finally:
        # Clean up
        if key_value_store.connected:
            # Delete all keys
            logger.info("Deleting all keys")
            for i in range(10):
                await key_value_store.delete(f"test:{i}")
            await key_value_store.delete("test:counter")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await key_value_store.disconnect()


async def test_graph_database():
    """Test the enhanced graph database."""
    logger.info("Testing enhanced graph database")
    
    # Skip this test if Neo4j is not available
    logger.info("Skipping graph database test (requires Neo4j)")
    return
    
    # Create an enhanced graph database
    graph_db = EnhancedGraphDatabase(
        name="neo4j",
        db_type=NoSqlType.NEO4J,
        connection_string="neo4j://localhost:7687"
    )
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await graph_db.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        # Create some nodes
        logger.info("Creating nodes")
        person_ids = []
        for i in range(5):
            result = await graph_db.create_node(
                {"name": f"Person{i}", "age": 20 + i},
                ["Person"]
            )
            if not result.success:
                logger.error(f"Failed to create node: {result.error}")
                return
            person_ids.append(result.data)
        
        # Create relationships
        logger.info("Creating relationships")
        for i in range(4):
            result = await graph_db.create_relationship(
                person_ids[i],
                person_ids[i + 1],
                "KNOWS",
                {"since": 2020 + i}
            )
            if not result.success:
                logger.error(f"Failed to create relationship: {result.error}")
                return
        
        # Execute a query
        logger.info("Executing a query")
        result = await graph_db.execute_query(
            "MATCH (p:Person)-[r:KNOWS]->(friend) RETURN p.name, friend.name, r.since"
        )
        if not result.success:
            logger.error(f"Failed to execute query: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
        # Get a node
        logger.info("Getting a node")
        result = await graph_db.get_node(person_ids[0])
        if not result.success:
            logger.error(f"Failed to get node: {result.error}")
            return
        
        logger.info(f"Node: {result.data}")
        
        # Update a node
        logger.info("Updating a node")
        result = await graph_db.update_node(
            person_ids[0],
            {"age": 100}
        )
        if not result.success:
            logger.error(f"Failed to update node: {result.error}")
            return
        
        # Get metrics
        metrics = await graph_db.get_metrics()
        logger.info(f"Graph database metrics: {metrics}")
        
    finally:
        # Clean up
        if graph_db.connected:
            # Delete all nodes
            logger.info("Deleting all nodes")
            for node_id in person_ids:
                await graph_db.delete_node(node_id, detach=True)
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await graph_db.disconnect()


async def test_factory():
    """Test the database connection factory."""
    logger.info("Testing database connection factory")
    
    # Create an SQL connection using the factory
    logger.info("Creating an SQL connection using the factory")
    sql = create_database_connection(
        "sql",
        {
            "name": "sqlite",
            "dialect": SqlDialect.SQLITE,
            "connection_string": "sqlite:///test.db"
        }
    )
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await sql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        # Create a table
        logger.info("Creating a table")
        result = await sql.execute("""
            CREATE TABLE IF NOT EXISTS factory_test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Insert some data
        logger.info("Inserting data")
        for i in range(5):
            result = await sql.execute(
                "INSERT INTO factory_test (name, value) VALUES (:name, :value)",
                {"name": f"factory{i}", "value": i}
            )
            if not result.success:
                logger.error(f"Failed to insert data: {result.error}")
                return
        
        # Query the data
        logger.info("Querying data")
        result = await sql.fetch_all("SELECT * FROM factory_test")
        if not result.success:
            logger.error(f"Failed to query data: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
    finally:
        # Clean up
        if sql.connected:
            # Drop the table
            logger.info("Dropping the table")
            await sql.execute("DROP TABLE IF EXISTS factory_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await sql.disconnect()


async def test_resilience():
    """Test the resilience patterns."""
    logger.info("Testing resilience patterns")
    
    # Create an enhanced SQL connection
    sql = EnhancedSqlConnection(
        name="sqlite",
        dialect=SqlDialect.SQLITE,
        connection_string="sqlite:///test.db"
    )
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await sql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        # Create a table
        logger.info("Creating a table")
        result = await sql.execute("""
            CREATE TABLE IF NOT EXISTS resilience_test (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value INTEGER
            )
        """)
        if not result.success:
            logger.error(f"Failed to create table: {result.error}")
            return
        
        # Test retry pattern with a simulated failure
        logger.info("Testing retry pattern with a simulated failure")
        
        # Patch the execute method to simulate failures
        original_execute = sql.execute
        failure_count = 0
        
        async def mock_execute(query, params=None):
            nonlocal failure_count
            if "INSERT" in query and failure_count < 2:
                failure_count += 1
                logger.info(f"Simulating failure {failure_count}/2")
                return await asyncio.sleep(0.1, result=sql._create_error_result("Simulated failure"))
            return await original_execute(query, params)
        
        # Replace the execute method
        sql.execute = mock_execute
        
        # Try to insert data (should succeed after retries)
        result = await sql.execute(
            "INSERT INTO resilience_test (name, value) VALUES (:name, :value)",
            {"name": "retry_test", "value": 42}
        )
        
        # Restore the original execute method
        sql.execute = original_execute
        
        if not result.success:
            logger.error(f"Failed to insert data after retries: {result.error}")
            return
        
        logger.info("Insert succeeded after retries")
        
        # Query the data to verify
        result = await sql.fetch_one("SELECT * FROM resilience_test WHERE name = 'retry_test'")
        if not result.success:
            logger.error(f"Failed to query data: {result.error}")
            return
        
        logger.info(f"Query result: {result.data}")
        
    finally:
        # Clean up
        if sql.connected:
            # Drop the table
            logger.info("Dropping the table")
            await sql.execute("DROP TABLE IF EXISTS resilience_test")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await sql.disconnect()


async def main():
    """Run the enhanced database examples."""
    logger.info("Starting enhanced database examples")
    
    # Test each database type
    await test_sql_connection()
    logger.info("-" * 80)
    
    await test_document_store()
    logger.info("-" * 80)
    
    await test_key_value_store()
    logger.info("-" * 80)
    
    await test_graph_database()
    logger.info("-" * 80)
    
    # Test the factory
    await test_factory()
    logger.info("-" * 80)
    
    # Test resilience patterns
    await test_resilience()
    
    logger.info("Enhanced database examples completed")


if __name__ == "__main__":
    asyncio.run(main())
