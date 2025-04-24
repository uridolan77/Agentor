"""
Base classes for database interfaces in the Agentor framework.

This module provides the base classes and interfaces for database connections
and operations, including error handling and result types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
import time
from contextlib import asynccontextmanager

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type
M = TypeVar('M')  # Model type


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConnectionError(DatabaseError):
    """Exception raised when a database connection fails."""
    pass


class QueryError(DatabaseError):
    """Exception raised when a database query fails."""
    pass


class TransactionError(DatabaseError):
    """Exception raised when a database transaction fails."""
    pass


class DatabaseResult(Generic[T]):
    """Result of a database operation."""

    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[Exception] = None,
        affected_rows: int = 0,
        last_insert_id: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.affected_rows = affected_rows
        self.last_insert_id = last_insert_id
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        if self.success:
            return f"DatabaseResult(success={self.success}, affected_rows={self.affected_rows})"
        else:
            return f"DatabaseResult(success={self.success}, error={self.error})"

    @classmethod
    def success_result(cls, data: Optional[T] = None, affected_rows: int = 0, last_insert_id: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None) -> 'DatabaseResult[T]':
        """Create a successful database result."""
        return cls(success=True, data=data, affected_rows=affected_rows, last_insert_id=last_insert_id, metadata=metadata)

    @classmethod
    def error_result(cls, error: Union[Exception, str], metadata: Optional[Dict[str, Any]] = None) -> 'DatabaseResult[T]':
        """Create an error database result."""
        if isinstance(error, str):
            error = DatabaseError(error)
        return cls(success=False, error=error, metadata=metadata)


class DatabaseConnection(ABC):
    """Base class for database connections."""

    def __init__(self, name: str, connection_string: str, **kwargs):
        self.name = name
        self.connection_string = connection_string
        self.connected = False
        self.connection_params = kwargs
        self.connection = None
        self.last_activity = time.time()

    @abstractmethod
    async def connect(self) -> DatabaseResult:
        """Connect to the database."""
        pass

    @abstractmethod
    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the database."""
        pass

    @abstractmethod
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the database."""
        pass

    @abstractmethod
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> DatabaseResult:
        """Execute a query multiple times with different parameters."""
        pass

    @abstractmethod
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single row from the database."""
        pass

    @abstractmethod
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all rows from the database."""
        pass

    @abstractmethod
    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the database."""
        pass

    @abstractmethod
    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction."""
        pass

    @abstractmethod
    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction."""
        pass

    @abstractmethod
    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction."""
        pass

    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactions."""
        result = await self.begin_transaction()
        if not result.success:
            raise TransactionError(f"Failed to begin transaction: {result.error}")

        try:
            yield self
            result = await self.commit_transaction()
            if not result.success:
                raise TransactionError(f"Failed to commit transaction: {result.error}")
        except Exception as e:
            await self.rollback_transaction()
            raise TransactionError(f"Transaction failed: {e}")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, connected={self.connected})"
