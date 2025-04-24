"""
Tests for the MySQL adapter.

This module contains tests for the MySQL adapter with resilience patterns.
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aiomysql

from agentor.interfaces.database import (
    MySqlAdapter,
    create_mysql_adapter,
    DatabaseResult
)
from agentor.interfaces.database.config.mysql import MySqlConfig


@pytest.fixture
def mock_cursor():
    """Create a mock cursor."""
    cursor = MagicMock()
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.rowcount = 0
    cursor.lastrowid = 0
    return cursor


@pytest.fixture
def mock_connection():
    """Create a mock connection."""
    conn = MagicMock()
    conn.cursor = AsyncMock()
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    conn.close = AsyncMock()
    return conn


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    pool.close = AsyncMock()
    pool.wait_closed = AsyncMock()
    return pool


@pytest.fixture
def mysql_adapter(mock_pool, mock_connection, mock_cursor):
    """Create a MySQL adapter with mocked dependencies."""
    # Set up the mock cursor
    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
    
    # Set up the mock connection
    mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
    
    # Patch the aiomysql.create_pool function
    with patch("aiomysql.create_pool", AsyncMock(return_value=mock_pool)):
        adapter = MySqlAdapter(
            name="test_mysql",
            connection_string="mysql://user:password@localhost:3306/test"
        )
        yield adapter


@pytest.mark.asyncio
async def test_connect(mysql_adapter, mock_pool):
    """Test connecting to the database."""
    # Set up the mock cursor to return a successful test query result
    mock_cursor = mock_pool.acquire.return_value.__aenter__.return_value.cursor.return_value.__aenter__.return_value
    mock_cursor.fetchone.return_value = (1,)
    
    # Connect to the database
    result = await mysql_adapter.connect()
    
    # Check the result
    assert result.success
    assert mysql_adapter.connected
    
    # Verify that the pool was created
    assert mysql_adapter.pool is not None


@pytest.mark.asyncio
async def test_disconnect(mysql_adapter, mock_pool):
    """Test disconnecting from the database."""
    # Connect first
    await mysql_adapter.connect()
    
    # Disconnect
    result = await mysql_adapter.disconnect()
    
    # Check the result
    assert result.success
    assert not mysql_adapter.connected
    
    # Verify that the pool was closed
    mock_pool.close.assert_called_once()
    mock_pool.wait_closed.assert_called_once()


@pytest.mark.asyncio
async def test_execute(mysql_adapter, mock_cursor):
    """Test executing a query."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.rowcount = 1
    mock_cursor.lastrowid = 42
    
    # Execute a query
    result = await mysql_adapter.execute("INSERT INTO test (name) VALUES (:name)", {"name": "test"})
    
    # Check the result
    assert result.success
    assert result.affected_rows == 1
    assert result.last_insert_id == 42
    
    # Verify that the query was executed
    mock_cursor.execute.assert_called_once()
    
    # Verify that the parameters were converted correctly
    args, kwargs = mock_cursor.execute.call_args
    assert args[0] == "INSERT INTO test (name) VALUES (%s)"
    assert args[1] == ["test"]


@pytest.mark.asyncio
async def test_fetch_one(mysql_adapter, mock_cursor):
    """Test fetching a single row."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.fetchone.return_value = {"id": 1, "name": "test"}
    
    # Fetch a row
    result = await mysql_adapter.fetch_one("SELECT * FROM test WHERE id = :id", {"id": 1})
    
    # Check the result
    assert result.success
    assert result.data == {"id": 1, "name": "test"}
    
    # Verify that the query was executed
    mock_cursor.execute.assert_called_once()
    
    # Verify that the parameters were converted correctly
    args, kwargs = mock_cursor.execute.call_args
    assert args[0] == "SELECT * FROM test WHERE id = %s"
    assert args[1] == [1]


@pytest.mark.asyncio
async def test_fetch_all(mysql_adapter, mock_cursor):
    """Test fetching all rows."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.fetchall.return_value = [
        {"id": 1, "name": "test1"},
        {"id": 2, "name": "test2"}
    ]
    
    # Fetch all rows
    result = await mysql_adapter.fetch_all("SELECT * FROM test")
    
    # Check the result
    assert result.success
    assert len(result.data) == 2
    assert result.data[0]["name"] == "test1"
    assert result.data[1]["name"] == "test2"
    
    # Verify that the query was executed
    mock_cursor.execute.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_value(mysql_adapter, mock_cursor):
    """Test fetching a single value."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.fetchone.return_value = (42,)
    
    # Fetch a value
    result = await mysql_adapter.fetch_value("SELECT COUNT(*) FROM test")
    
    # Check the result
    assert result.success
    assert result.data == 42
    
    # Verify that the query was executed
    mock_cursor.execute.assert_called_once()


@pytest.mark.asyncio
async def test_execute_many(mysql_adapter, mock_cursor):
    """Test executing multiple queries."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.rowcount = 2
    
    # Execute multiple queries
    result = await mysql_adapter.execute_many(
        "INSERT INTO test (name, value) VALUES (:name, :value)",
        [
            {"name": "test1", "value": 1},
            {"name": "test2", "value": 2}
        ]
    )
    
    # Check the result
    assert result.success
    assert result.affected_rows == 2
    
    # Verify that the query was executed
    mock_cursor.executemany.assert_called_once()


@pytest.mark.asyncio
async def test_transaction(mysql_adapter, mock_connection, mock_cursor):
    """Test transaction management."""
    # Connect first
    await mysql_adapter.connect()
    
    # Begin a transaction
    result = await mysql_adapter.begin_transaction()
    assert result.success
    assert mysql_adapter.in_transaction
    
    # Execute a query in the transaction
    await mysql_adapter.execute("INSERT INTO test (name) VALUES (:name)", {"name": "test"})
    
    # Commit the transaction
    result = await mysql_adapter.commit_transaction()
    assert result.success
    assert not mysql_adapter.in_transaction
    
    # Verify that the transaction was committed
    mock_connection.commit.assert_called_once()


@pytest.mark.asyncio
async def test_rollback_transaction(mysql_adapter, mock_connection, mock_cursor):
    """Test rolling back a transaction."""
    # Connect first
    await mysql_adapter.connect()
    
    # Begin a transaction
    result = await mysql_adapter.begin_transaction()
    assert result.success
    assert mysql_adapter.in_transaction
    
    # Execute a query in the transaction
    await mysql_adapter.execute("INSERT INTO test (name) VALUES (:name)", {"name": "test"})
    
    # Rollback the transaction
    result = await mysql_adapter.rollback_transaction()
    assert result.success
    assert not mysql_adapter.in_transaction
    
    # Verify that the transaction was rolled back
    mock_connection.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_execute_prepared_statement(mysql_adapter, mock_cursor):
    """Test executing a prepared statement."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.rowcount = 1
    
    # Execute a prepared statement
    result = await mysql_adapter.execute_prepared_statement(
        "test_statement",
        "SELECT * FROM test WHERE id = :id",
        {"id": 1}
    )
    
    # Check the result
    assert result.success
    
    # Verify that the prepared statement was created and executed
    assert mock_cursor.execute.call_count == 3  # PREPARE, EXECUTE, DEALLOCATE


@pytest.mark.asyncio
async def test_get_server_info(mysql_adapter, mock_cursor):
    """Test getting server information."""
    # Connect first
    await mysql_adapter.connect()
    
    # Set up the mock cursor
    mock_cursor.fetchall.side_effect = [
        [{"Variable_name": "version", "Value": "8.0.0"}],  # SHOW VARIABLES
        [{"Variable_name": "status", "Value": "OK"}]       # SHOW STATUS
    ]
    mock_cursor.fetchone.return_value = {"version": "8.0.0"}  # SELECT VERSION()
    
    # Get server info
    result = await mysql_adapter.get_server_info()
    
    # Check the result
    assert result.success
    assert result.data["version"] == "8.0.0"
    assert "variables" in result.data
    assert "status" in result.data


@pytest.mark.asyncio
async def test_get_metrics(mysql_adapter):
    """Test getting metrics."""
    # Connect first
    await mysql_adapter.connect()
    
    # Get metrics
    metrics = await mysql_adapter.get_metrics()
    
    # Check the metrics
    assert metrics["name"] == "test_mysql"
    assert "mysql_metrics" in metrics
    assert "pool_metrics" in metrics


@pytest.mark.asyncio
async def test_create_mysql_adapter():
    """Test creating a MySQL adapter using the factory function."""
    # Create a MySQL configuration
    config = MySqlConfig(
        name="factory_test",
        mysql_user="user",
        mysql_password="password",
        mysql_host="localhost",
        mysql_port=3306,
        mysql_database="test"
    )
    
    # Create a MySQL adapter using the factory function
    with patch("aiomysql.create_pool", AsyncMock()):
        adapter = create_mysql_adapter(config)
        
        # Check the adapter
        assert adapter.name == "factory_test"
        assert adapter.dialect.value == "mysql"
        assert adapter.charset == "utf8mb4"
        assert adapter.collation == "utf8mb4_unicode_ci"
        assert not adapter.autocommit
