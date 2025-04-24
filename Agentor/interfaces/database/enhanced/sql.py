"""
Enhanced SQL database interface for the Agentor framework.

This module provides an enhanced SQL database interface with resilience patterns,
connection pooling, and other advanced features.
"""

import os
import time
import asyncio
import re
import urllib.parse
from typing import Dict, Any, Optional, List, Union, Tuple, TypeVar, cast
import logging

from ..base import (
    DatabaseConnection, DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from ..sql import SqlConnection, SqlDialect
from ..config import SqlDatabaseConfig
from ..resilience import (
    with_database_resilience, database_resilience_context, wrap_database_result,
    DATABASE_CONNECTION_RETRY_CONFIG, DATABASE_QUERY_RETRY_CONFIG, DATABASE_TRANSACTION_RETRY_CONFIG
)
from agentor.llm_gateway.utils.timeout import TimeoutStrategy
from agentor.core.config import get_typed_config

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class EnhancedSqlConnection(SqlConnection):
    """Enhanced SQL database connection with resilience patterns."""
    
    def __init__(
        self,
        name: str,
        dialect: SqlDialect,
        connection_string: str,
        pool_size: int = 10,
        pool_recycle: int = 3600,
        **kwargs
    ):
        """Initialize the SQL database connection.
        
        Args:
            name: The name of the connection
            dialect: The SQL dialect to use
            connection_string: The connection string for the database
            pool_size: The maximum number of connections in the pool
            pool_recycle: The connection recycle time in seconds
            **kwargs: Additional connection parameters
        """
        super().__init__(
            name=name,
            dialect=dialect,
            connection_string=connection_string,
            pool_size=pool_size,
            pool_recycle=pool_recycle,
            **kwargs
        )
        
        # Load configuration
        try:
            self.config = get_typed_config(SqlDatabaseConfig)
        except Exception as e:
            logger.warning(f"Failed to load SQL database configuration, using defaults: {e}")
            self.config = SqlDatabaseConfig(
                name=name,
                dialect=dialect,
                connection_string=connection_string,
                pool_min_size=1,
                pool_max_size=pool_size,
                pool_recycle=pool_recycle,
                connection_params=kwargs
            )
        
        # Set up connection pool metrics
        self.pool_metrics = {
            "created_connections": 0,
            "closed_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "waiting_requests": 0,
            "max_waiting_requests": 0,
            "total_wait_time": 0.0,
            "total_execution_time": 0.0,
            "total_queries": 0,
            "failed_queries": 0
        }
        
        # Set up connection pool management
        self.pool_lock = asyncio.Lock()
        self.pool_condition = asyncio.Condition(self.pool_lock)
        self.pool_connections: List[Any] = []
        self.pool_active: Dict[Any, float] = {}  # Connection -> acquisition time
        self.pool_idle: List[Any] = []
    
    @with_database_resilience(
        database="sql",
        operation="connect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=False
    )
    async def connect(self) -> DatabaseResult:
        """Connect to the SQL database with resilience patterns."""
        # Call the parent implementation
        result = await super().connect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "connect", result)
    
    @with_database_resilience(
        database="sql",
        operation="disconnect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=False,
        use_bulkhead=False
    )
    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the SQL database with resilience patterns."""
        # Call the parent implementation
        result = await super().disconnect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "disconnect", result)
    
    @with_database_resilience(
        database="sql",
        operation="execute",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the SQL database with resilience patterns."""
        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()
        
        # Call the parent implementation
        result = await super().execute(query, params)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.pool_metrics["total_execution_time"] += execution_time
        if not result.success:
            self.pool_metrics["failed_queries"] += 1
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "execute", result)
    
    @with_database_resilience(
        database="sql",
        operation="fetch_one",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single row from the SQL database with resilience patterns."""
        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()
        
        # Call the parent implementation
        result = await super().fetch_one(query, params)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.pool_metrics["total_execution_time"] += execution_time
        if not result.success:
            self.pool_metrics["failed_queries"] += 1
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "fetch_one", result)
    
    @with_database_resilience(
        database="sql",
        operation="fetch_all",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=60.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all rows from the SQL database with resilience patterns."""
        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()
        
        # Call the parent implementation
        result = await super().fetch_all(query, params)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.pool_metrics["total_execution_time"] += execution_time
        if not result.success:
            self.pool_metrics["failed_queries"] += 1
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "fetch_all", result)
    
    @with_database_resilience(
        database="sql",
        operation="fetch_value",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the SQL database with resilience patterns."""
        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()
        
        # Call the parent implementation
        result = await super().fetch_value(query, params)
        
        # Update metrics
        execution_time = time.time() - start_time
        self.pool_metrics["total_execution_time"] += execution_time
        if not result.success:
            self.pool_metrics["failed_queries"] += 1
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "fetch_value", result)
    
    @with_database_resilience(
        database="sql",
        operation="begin_transaction",
        retry_config=DATABASE_TRANSACTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction with resilience patterns."""
        # Call the parent implementation
        result = await super().begin_transaction()
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "begin_transaction", result)
    
    @with_database_resilience(
        database="sql",
        operation="commit_transaction",
        retry_config=DATABASE_TRANSACTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction with resilience patterns."""
        # Call the parent implementation
        result = await super().commit_transaction()
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "commit_transaction", result)
    
    @with_database_resilience(
        database="sql",
        operation="rollback_transaction",
        retry_config=DATABASE_TRANSACTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=False,
        use_bulkhead=False
    )
    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction with resilience patterns."""
        # Call the parent implementation
        result = await super().rollback_transaction()
        
        # Wrap the result with additional error handling
        return wrap_database_result("sql", "rollback_transaction", result)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "name": self.name,
            "dialect": self.dialect.value,
            "connected": self.connected,
            "in_transaction": self.in_transaction,
            "last_activity": self.last_activity,
            "pool_metrics": self.pool_metrics
        }
