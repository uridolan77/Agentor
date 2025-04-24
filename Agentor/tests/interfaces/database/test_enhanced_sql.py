"""
Tests for the enhanced SQL database interface.

This module contains tests for the enhanced SQL database interface with resilience patterns.
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sqlite3

from agentor.interfaces.database.sql import SqlDialect
from agentor.interfaces.database.enhanced.sql import EnhancedSqlConnection
from agentor.interfaces.database.base import DatabaseResult


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=[])
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.rowcount = 0
    return conn


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    engine.connect = AsyncMock()
    engine.dispose = AsyncMock()
    return engine


@pytest.fixture
def enhanced_sql_connection(mock_engine, mock_connection):
    """Create an enhanced SQL connection with mocked dependencies."""
    with patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine):
        with patch.object(mock_engine, "connect", return_value=mock_connection):
            conn = EnhancedSqlConnection(
                name="test_sql",
                dialect=SqlDialect.SQLITE,
                connection_string="sqlite:///test.db"
            )
            yield conn


@pytest.mark.asyncio
async def test_connect(enhanced_sql_connection, mock_engine):
    """Test connecting to the database."""
    result = await enhanced_sql_connection.connect()
    
    assert result.success
    mock_engine.connect.assert_called_once()
    assert enhanced_sql_connection.connected


@pytest.mark.asyncio
async def test_disconnect(enhanced_sql_connection, mock_engine):
    """Test disconnecting from the database."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Disconnect
    result = await enhanced_sql_connection.disconnect()
    
    assert result.success
    mock_engine.dispose.assert_called_once()
    assert not enhanced_sql_connection.connected


@pytest.mark.asyncio
async def test_execute(enhanced_sql_connection, mock_connection):
    """Test executing a query."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Execute a query
    result = await enhanced_sql_connection.execute("SELECT 1")
    
    assert result.success
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("SELECT 1", None)


@pytest.mark.asyncio
async def test_fetch_one(enhanced_sql_connection, mock_connection):
    """Test fetching a single row."""
    # Mock the fetchone method to return a row
    mock_connection.cursor().fetchone.return_value = (1, "test")
    
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Fetch a row
    result = await enhanced_sql_connection.fetch_one("SELECT 1")
    
    assert result.success
    assert result.data == (1, "test")
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("SELECT 1", None)
    mock_connection.cursor().fetchone.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_all(enhanced_sql_connection, mock_connection):
    """Test fetching all rows."""
    # Mock the fetchall method to return rows
    mock_connection.cursor().fetchall.return_value = [(1, "test1"), (2, "test2")]
    
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Fetch all rows
    result = await enhanced_sql_connection.fetch_all("SELECT 1")
    
    assert result.success
    assert result.data == [(1, "test1"), (2, "test2")]
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("SELECT 1", None)
    mock_connection.cursor().fetchall.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_value(enhanced_sql_connection, mock_connection):
    """Test fetching a single value."""
    # Mock the fetchone method to return a row
    mock_connection.cursor().fetchone.return_value = (42,)
    
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Fetch a value
    result = await enhanced_sql_connection.fetch_value("SELECT 1")
    
    assert result.success
    assert result.data == 42
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("SELECT 1", None)
    mock_connection.cursor().fetchone.assert_called_once()


@pytest.mark.asyncio
async def test_begin_transaction(enhanced_sql_connection, mock_connection):
    """Test beginning a transaction."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Begin a transaction
    result = await enhanced_sql_connection.begin_transaction()
    
    assert result.success
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("BEGIN", None)
    assert enhanced_sql_connection.in_transaction


@pytest.mark.asyncio
async def test_commit_transaction(enhanced_sql_connection, mock_connection):
    """Test committing a transaction."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Begin a transaction
    await enhanced_sql_connection.begin_transaction()
    
    # Reset the mock to clear the call history
    mock_connection.cursor.reset_mock()
    mock_connection.cursor().execute.reset_mock()
    
    # Commit the transaction
    result = await enhanced_sql_connection.commit_transaction()
    
    assert result.success
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("COMMIT", None)
    assert not enhanced_sql_connection.in_transaction


@pytest.mark.asyncio
async def test_rollback_transaction(enhanced_sql_connection, mock_connection):
    """Test rolling back a transaction."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Begin a transaction
    await enhanced_sql_connection.begin_transaction()
    
    # Reset the mock to clear the call history
    mock_connection.cursor.reset_mock()
    mock_connection.cursor().execute.reset_mock()
    
    # Rollback the transaction
    result = await enhanced_sql_connection.rollback_transaction()
    
    assert result.success
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("ROLLBACK", None)
    assert not enhanced_sql_connection.in_transaction


@pytest.mark.asyncio
async def test_execute_error(enhanced_sql_connection, mock_connection):
    """Test error handling when executing a query."""
    # Mock the execute method to raise an exception
    mock_connection.cursor().execute.side_effect = sqlite3.Error("Test error")
    
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Execute a query
    result = await enhanced_sql_connection.execute("SELECT 1")
    
    assert not result.success
    assert "Test error" in str(result.error)
    mock_connection.cursor.assert_called_once()
    mock_connection.cursor().execute.assert_called_once_with("SELECT 1", None)


@pytest.mark.asyncio
async def test_resilience_retry(enhanced_sql_connection, mock_connection):
    """Test retry resilience pattern."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Mock the execute method to fail twice and then succeed
    side_effects = [
        sqlite3.Error("Test error 1"),
        sqlite3.Error("Test error 2"),
        None  # Success on the third attempt
    ]
    mock_connection.cursor().execute.side_effect = side_effects
    
    # Execute a query
    result = await enhanced_sql_connection.execute("SELECT 1")
    
    assert result.success
    assert mock_connection.cursor().execute.call_count == 3
    mock_connection.cursor().execute.assert_called_with("SELECT 1", None)


@pytest.mark.asyncio
async def test_get_metrics(enhanced_sql_connection):
    """Test getting connection metrics."""
    # Connect first
    await enhanced_sql_connection.connect()
    
    # Get metrics
    metrics = await enhanced_sql_connection.get_metrics()
    
    assert metrics["name"] == "test_sql"
    assert metrics["dialect"] == "sqlite"
    assert metrics["connected"] is True
    assert metrics["in_transaction"] is False
    assert "pool_metrics" in metrics
