"""
Enhanced graph database interface for the Agentor framework.

This module provides an enhanced graph database interface with resilience patterns,
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
from ..nosql import GraphDatabase, NoSqlType
from ..config import GraphDatabaseConfig
from ..resilience import (
    with_database_resilience, database_resilience_context, wrap_database_result,
    DATABASE_CONNECTION_RETRY_CONFIG, DATABASE_QUERY_RETRY_CONFIG, DATABASE_TRANSACTION_RETRY_CONFIG
)
from agentor.llm_gateway.utils.timeout import TimeoutStrategy
from agentor.core.config import get_typed_config

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class EnhancedGraphDatabase(GraphDatabase):
    """Enhanced graph database with resilience patterns."""
    
    def __init__(
        self,
        name: str,
        db_type: NoSqlType,
        connection_string: str,
        **kwargs
    ):
        """Initialize the graph database.
        
        Args:
            name: The name of the connection
            db_type: The NoSQL database type
            connection_string: The connection string for the database
            **kwargs: Additional connection parameters
        """
        super().__init__(
            name=name,
            db_type=db_type,
            connection_string=connection_string,
            **kwargs
        )
        
        # Load configuration
        try:
            self.config = get_typed_config(GraphDatabaseConfig)
        except Exception as e:
            logger.warning(f"Failed to load graph database configuration, using defaults: {e}")
            self.config = GraphDatabaseConfig(
                name=name,
                db_type=db_type,
                connection_string=connection_string,
                connection_params=kwargs
            )
        
        # Set up connection metrics
        self.metrics = {
            "total_operations": 0,
            "failed_operations": 0,
            "total_execution_time": 0.0,
            "last_activity": 0.0,
            "active_operations": 0,
            "max_active_operations": 0
        }
    
    @with_database_resilience(
        database="graph",
        operation="connect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=False
    )
    async def connect(self) -> DatabaseResult:
        """Connect to the graph database with resilience patterns."""
        # Call the parent implementation
        result = await super().connect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("graph", "connect", result)
    
    @with_database_resilience(
        database="graph",
        operation="disconnect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=False,
        use_bulkhead=False
    )
    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the graph database with resilience patterns."""
        # Call the parent implementation
        result = await super().disconnect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("graph", "disconnect", result)
    
    @with_database_resilience(
        database="graph",
        operation="execute_query",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=60.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> DatabaseResult:
        """Execute a query on the graph database with resilience patterns."""
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["active_operations"] += 1
        self.metrics["max_active_operations"] = max(
            self.metrics["max_active_operations"],
            self.metrics["active_operations"]
        )
        start_time = time.time()
        
        try:
            # Call the parent implementation
            result = await super().execute_query(query, params)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("graph", "execute_query", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="graph",
        operation="get_node",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get_node(
        self,
        node_id: str,
        labels: Optional[List[str]] = None
    ) -> DatabaseResult:
        """Get a node from the graph database with resilience patterns."""
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["active_operations"] += 1
        self.metrics["max_active_operations"] = max(
            self.metrics["max_active_operations"],
            self.metrics["active_operations"]
        )
        start_time = time.time()
        
        try:
            # Call the parent implementation
            result = await super().get_node(node_id, labels)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("graph", "get_node", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="graph",
        operation="create_node",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def create_node(
        self,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None
    ) -> DatabaseResult:
        """Create a node in the graph database with resilience patterns."""
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["active_operations"] += 1
        self.metrics["max_active_operations"] = max(
            self.metrics["max_active_operations"],
            self.metrics["active_operations"]
        )
        start_time = time.time()
        
        try:
            # Call the parent implementation
            result = await super().create_node(properties, labels)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("graph", "create_node", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="graph",
        operation="update_node",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        labels: Optional[List[str]] = None
    ) -> DatabaseResult:
        """Update a node in the graph database with resilience patterns."""
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["active_operations"] += 1
        self.metrics["max_active_operations"] = max(
            self.metrics["max_active_operations"],
            self.metrics["active_operations"]
        )
        start_time = time.time()
        
        try:
            # Call the parent implementation
            result = await super().update_node(node_id, properties, labels)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("graph", "update_node", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="graph",
        operation="delete_node",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def delete_node(
        self,
        node_id: str,
        detach: bool = True
    ) -> DatabaseResult:
        """Delete a node from the graph database with resilience patterns."""
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["active_operations"] += 1
        self.metrics["max_active_operations"] = max(
            self.metrics["max_active_operations"],
            self.metrics["active_operations"]
        )
        start_time = time.time()
        
        try:
            # Call the parent implementation
            result = await super().delete_node(node_id, detach)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("graph", "delete_node", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="graph",
        operation="create_relationship",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> DatabaseResult:
        """Create a relationship in the graph database with resilience patterns."""
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["active_operations"] += 1
        self.metrics["max_active_operations"] = max(
            self.metrics["max_active_operations"],
            self.metrics["active_operations"]
        )
        start_time = time.time()
        
        try:
            # Call the parent implementation
            result = await super().create_relationship(start_node_id, end_node_id, relationship_type, properties)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("graph", "create_relationship", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="graph",
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
        return wrap_database_result("graph", "begin_transaction", result)
    
    @with_database_resilience(
        database="graph",
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
        return wrap_database_result("graph", "commit_transaction", result)
    
    @with_database_resilience(
        database="graph",
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
        return wrap_database_result("graph", "rollback_transaction", result)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "name": self.name,
            "db_type": self.db_type.value,
            "connected": self.connected,
            "in_transaction": self.in_transaction,
            "metrics": self.metrics
        }
