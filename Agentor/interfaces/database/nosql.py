"""
NoSQL database interfaces for the Agentor framework.

This module provides interfaces for interacting with NoSQL databases,
including document stores, key-value stores, and graph databases.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
import time

from .base import (
    DatabaseConnection, DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class NoSqlType(Enum):
    """Types of NoSQL databases supported."""
    DOCUMENT = "document"
    KEY_VALUE = "key_value"
    GRAPH = "graph"
    COLUMN = "column"
    TIME_SERIES = "time_series"


class NoSqlConnection(DatabaseConnection):
    """Base class for NoSQL database connections."""

    def __init__(
        self,
        name: str,
        connection_string: str,
        db_type: NoSqlType,
        **kwargs
    ):
        super().__init__(name=name, connection_string=connection_string, **kwargs)
        self.db_type = db_type

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the NoSQL database.
        
        This is a generic method that may not be applicable to all NoSQL databases.
        Specific implementations should override this method as needed.
        """
        return DatabaseResult.error_result("Method not implemented for this NoSQL database type")

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> DatabaseResult:
        """Execute a query multiple times with different parameters.
        
        This is a generic method that may not be applicable to all NoSQL databases.
        Specific implementations should override this method as needed.
        """
        return DatabaseResult.error_result("Method not implemented for this NoSQL database type")

    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single document/record from the NoSQL database.
        
        This is a generic method that may not be applicable to all NoSQL databases.
        Specific implementations should override this method as needed.
        """
        return DatabaseResult.error_result("Method not implemented for this NoSQL database type")

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all documents/records from the NoSQL database.
        
        This is a generic method that may not be applicable to all NoSQL databases.
        Specific implementations should override this method as needed.
        """
        return DatabaseResult.error_result("Method not implemented for this NoSQL database type")

    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the NoSQL database.
        
        This is a generic method that may not be applicable to all NoSQL databases.
        Specific implementations should override this method as needed.
        """
        return DatabaseResult.error_result("Method not implemented for this NoSQL database type")


# Import specific NoSQL database implementations
from .document_store import DocumentStore
from .key_value_store import KeyValueStore
from .graph_database import GraphDatabase

__all__ = [
    'NoSqlType', 'NoSqlConnection',
    'DocumentStore', 'KeyValueStore', 'GraphDatabase'
]
