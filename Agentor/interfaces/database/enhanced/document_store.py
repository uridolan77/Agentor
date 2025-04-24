"""
Enhanced document store interface for the Agentor framework.

This module provides an enhanced document store interface with resilience patterns,
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
from ..nosql import DocumentStore, NoSqlType
from ..config import DocumentStoreConfig
from ..resilience import (
    with_database_resilience, database_resilience_context, wrap_database_result,
    DATABASE_CONNECTION_RETRY_CONFIG, DATABASE_QUERY_RETRY_CONFIG, DATABASE_TRANSACTION_RETRY_CONFIG
)
from agentor.llm_gateway.utils.timeout import TimeoutStrategy
from agentor.core.config import get_typed_config

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class EnhancedDocumentStore(DocumentStore):
    """Enhanced document store with resilience patterns."""
    
    def __init__(
        self,
        name: str,
        db_type: NoSqlType,
        connection_string: str,
        database_name: str,
        **kwargs
    ):
        """Initialize the document store.
        
        Args:
            name: The name of the connection
            db_type: The NoSQL database type
            connection_string: The connection string for the database
            database_name: The name of the database
            **kwargs: Additional connection parameters
        """
        super().__init__(
            name=name,
            db_type=db_type,
            connection_string=connection_string,
            database_name=database_name,
            **kwargs
        )
        
        # Load configuration
        try:
            self.config = get_typed_config(DocumentStoreConfig)
        except Exception as e:
            logger.warning(f"Failed to load document store configuration, using defaults: {e}")
            self.config = DocumentStoreConfig(
                name=name,
                db_type=db_type,
                connection_string=connection_string,
                database_name=database_name,
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
        database="document",
        operation="connect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=False
    )
    async def connect(self) -> DatabaseResult:
        """Connect to the document store with resilience patterns."""
        # Call the parent implementation
        result = await super().connect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("document", "connect", result)
    
    @with_database_resilience(
        database="document",
        operation="disconnect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=False,
        use_bulkhead=False
    )
    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the document store with resilience patterns."""
        # Call the parent implementation
        result = await super().disconnect()
        
        # Wrap the result with additional error handling
        return wrap_database_result("document", "disconnect", result)
    
    @with_database_resilience(
        database="document",
        operation="get_document",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get_document(self, collection: str, document_id: str) -> DatabaseResult:
        """Get a document from the document store with resilience patterns."""
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
            result = await super().get_document(collection, document_id)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("document", "get_document", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="document",
        operation="query_documents",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=60.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def query_documents(
        self,
        collection: str,
        query: Dict[str, Any],
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        sort: Optional[List[Tuple[str, int]]] = None
    ) -> DatabaseResult:
        """Query documents from the document store with resilience patterns."""
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
            result = await super().query_documents(collection, query, limit, skip, sort)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("document", "query_documents", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="document",
        operation="insert_document",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def insert_document(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> DatabaseResult:
        """Insert a document into the document store with resilience patterns."""
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
            result = await super().insert_document(collection, document)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("document", "insert_document", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="document",
        operation="update_document",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def update_document(
        self,
        collection: str,
        document_id: str,
        update: Dict[str, Any],
        upsert: bool = False
    ) -> DatabaseResult:
        """Update a document in the document store with resilience patterns."""
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
            result = await super().update_document(collection, document_id, update, upsert)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("document", "update_document", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="document",
        operation="delete_document",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def delete_document(
        self,
        collection: str,
        document_id: str
    ) -> DatabaseResult:
        """Delete a document from the document store with resilience patterns."""
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
            result = await super().delete_document(collection, document_id)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("document", "delete_document", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="document",
        operation="count_documents",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def count_documents(
        self,
        collection: str,
        query: Dict[str, Any]
    ) -> DatabaseResult:
        """Count documents in the document store with resilience patterns."""
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
            result = await super().count_documents(collection, query)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["last_activity"] = time.time()
            if not result.success:
                self.metrics["failed_operations"] += 1
            
            # Wrap the result with additional error handling
            return wrap_database_result("document", "count_documents", result)
        finally:
            # Update metrics
            self.metrics["active_operations"] -= 1
    
    @with_database_resilience(
        database="document",
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
        return wrap_database_result("document", "begin_transaction", result)
    
    @with_database_resilience(
        database="document",
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
        return wrap_database_result("document", "commit_transaction", result)
    
    @with_database_resilience(
        database="document",
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
        return wrap_database_result("document", "rollback_transaction", result)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "name": self.name,
            "db_type": self.db_type.value,
            "database_name": self.database_name,
            "connected": self.connected,
            "in_transaction": self.in_transaction,
            "metrics": self.metrics
        }
