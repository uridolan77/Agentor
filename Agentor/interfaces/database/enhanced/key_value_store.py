"""
Enhanced key-value store interface for the Agentor framework.

This module provides an enhanced key-value store interface with resilience patterns,
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
from ..nosql import KeyValueStore, NoSqlType
from ..config import KeyValueStoreConfig
from ..resilience import (
    with_database_resilience, database_resilience_context, wrap_database_result,
    DATABASE_CONNECTION_RETRY_CONFIG, DATABASE_QUERY_RETRY_CONFIG
)
from agentor.llm_gateway.utils.timeout import TimeoutStrategy
from agentor.core.config import get_typed_config

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class EnhancedKeyValueStore(KeyValueStore):
    """Enhanced key-value store with resilience patterns."""
    
    def __init__(
        self,
        name: str,
        db_type: NoSqlType,
        connection_string: str,
        **kwargs
    ):
        """Initialize the key-value store.
        
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
            self.config = get_typed_config(KeyValueStoreConfig)
        except Exception as e:
            logger.warning(f"Failed to load key-value store configuration, using defaults: {e}")
            self.config = KeyValueStoreConfig(
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
            "max_active_operations": 0,
            "hits": 0,
            "misses": 0
        }
    
    @with_database_resilience(
        database="key_value",
        operation="connect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=False
    )
    async def connect(self) -> DatabaseResult:
        """Connect to the key-value store with resilience patterns."""
        # Call the parent implementation
        result = await super().connect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("key_value", "connect", result)
    
    @with_database_resilience(
        database="key_value",
        operation="disconnect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=False,
        use_bulkhead=False
    )
    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the key-value store with resilience patterns."""
        # Call the parent implementation
        result = await super().disconnect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("key_value", "disconnect", result)
    
    @with_database_resilience(
        database="key_value",
        operation="get",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get(self, key: str) -> DatabaseResult:
        """Get a value from the key-value store with resilience patterns."""
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
            result = await super().get(key)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            else:
                if result.data is None:
                    self.metrics["misses"] += 1
                else:
                    self.metrics["hits"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("key_value", "get", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="key_value",
        operation="set",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> DatabaseResult:
        """Set a value in the key-value store with resilience patterns."""
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
            result = await super().set(key, value, ttl)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("key_value", "set", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="key_value",
        operation="delete",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def delete(self, key: str) -> DatabaseResult:
        """Delete a value from the key-value store with resilience patterns."""
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
            result = await super().delete(key)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("key_value", "delete", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="key_value",
        operation="exists",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def exists(self, key: str) -> DatabaseResult:
        """Check if a key exists in the key-value store with resilience patterns."""
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
            result = await super().exists(key)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("key_value", "exists", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="key_value",
        operation="increment",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def increment(self, key: str, amount: int = 1) -> DatabaseResult:
        """Increment a value in the key-value store with resilience patterns."""
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
            result = await super().increment(key, amount)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("key_value", "increment", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="key_value",
        operation="expire",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def expire(self, key: str, ttl: int) -> DatabaseResult:
        """Set an expiration time for a key with resilience patterns."""
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
            result = await super().expire(key, ttl)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("key_value", "expire", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "name": self.name,
            "db_type": self.db_type.value,
            "connected": self.connected,
            "metrics": self.metrics,
            "hit_ratio": self.metrics["hits"] / (self.metrics["hits"] + self.metrics["misses"]) if (self.metrics["hits"] + self.metrics["misses"]) > 0 else 0
        }
