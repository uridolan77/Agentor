"""
Resilience patterns for database operations.

This module provides resilience patterns for database operations, including
retry, circuit breaker, timeout, and bulkhead patterns.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union, Type, Callable, TypeVar, Generic, Awaitable
from functools import wraps
from contextlib import asynccontextmanager

from agentor.llm_gateway.utils.retry import (
    RetryConfig, RetryStrategy, retry_async, retry_async_decorator
)
from agentor.llm_gateway.utils.circuit_breaker import CircuitBreaker
from agentor.llm_gateway.utils.timeout import with_timeout, TimeoutStrategy
from agentor.llm_gateway.utils.bulkhead import with_bulkhead
from agentor.llm_gateway.utils.error_handler import (
    ErrorContext, ErrorSeverity, ErrorCategory, EnhancedError
)

from .base import DatabaseError, ConnectionError, QueryError, TransactionError, DatabaseResult

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


# Database-specific retry configurations
DATABASE_NETWORK_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=1.0,
    max_delay=30.0,
    jitter=0.2,
    retry_on=[
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError
    ]
)

DATABASE_QUERY_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=0.5,
    max_delay=10.0,
    jitter=0.1,
    retry_on=[
        QueryError
    ]
)

DATABASE_CONNECTION_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=2.0,
    max_delay=30.0,
    jitter=0.2,
    retry_on=[
        ConnectionError
    ]
)

DATABASE_TRANSACTION_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=1.0,
    max_delay=15.0,
    jitter=0.1,
    retry_on=[
        TransactionError
    ]
)


# Database-specific error categories
class DatabaseErrorCategory(ErrorCategory):
    """Error categories for database operations."""
    CONNECTION = "connection"
    QUERY = "query"
    TRANSACTION = "transaction"
    CONSTRAINT = "constraint"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


# Circuit breaker registry for database operations
class DatabaseCircuitBreakerRegistry:
    """Registry for database circuit breakers."""
    
    def __init__(self):
        """Initialize the registry."""
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, database: str, operation: str) -> CircuitBreaker:
        """Get a circuit breaker for a database operation.
        
        Args:
            database: The name of the database
            operation: The operation being performed
            
        Returns:
            A circuit breaker for the database operation
        """
        key = f"{database}:{operation}"
        if key not in self.breakers:
            self.breakers[key] = CircuitBreaker(
                name=key,
                failure_threshold=5,
                recovery_timeout=30,
                half_open_max_calls=1
            )
        
        return self.breakers[key]


# Create a global circuit breaker registry
circuit_breaker_registry = DatabaseCircuitBreakerRegistry()


def with_database_resilience(
    database: str,
    operation: str,
    retry_config: Optional[RetryConfig] = None,
    timeout_seconds: float = 30.0,
    timeout_strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE,
    use_circuit_breaker: bool = True,
    use_bulkhead: bool = True,
    max_concurrent: int = 10,
    max_queue_size: int = 20
):
    """Decorator for adding resilience patterns to database operations.
    
    This decorator combines retry, circuit breaker, timeout, and bulkhead patterns.
    
    Args:
        database: The name of the database
        operation: The operation being performed
        retry_config: The retry configuration, or None to use defaults
        timeout_seconds: Timeout for the operation in seconds
        timeout_strategy: Timeout strategy to use
        use_circuit_breaker: Whether to use the circuit breaker pattern
        use_bulkhead: Whether to use the bulkhead pattern
        max_concurrent: Maximum number of concurrent operations (for bulkhead)
        max_queue_size: Maximum size of the waiting queue (for bulkhead)
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the circuit breaker if needed
            circuit_breaker = None
            if use_circuit_breaker:
                circuit_breaker = circuit_breaker_registry.get_breaker(database, operation)
            
            # Define the operation function
            async def operation_func():
                # Use the circuit breaker if needed
                if circuit_breaker:
                    async with circuit_breaker:
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            
            # Apply the retry pattern
            retry_func = retry_async(
                operation_func,
                "database",
                f"{database}.{operation}",
                retry_config or DATABASE_QUERY_RETRY_CONFIG
            )
            
            # Apply the timeout pattern
            timeout_func = with_timeout(
                "database",
                f"{database}.{operation}",
                strategy=timeout_strategy,
                base_timeout=timeout_seconds,
                max_timeout=timeout_seconds * 2
            )(retry_func)
            
            # Apply the bulkhead pattern if needed
            if use_bulkhead:
                bulkhead_func = with_bulkhead(
                    f"database.{database}",
                    max_concurrent=max_concurrent,
                    max_queue_size=max_queue_size
                )(timeout_func)
                return await bulkhead_func()
            else:
                return await timeout_func()
        
        return wrapper
    
    return decorator


@asynccontextmanager
async def database_resilience_context(
    database: str,
    operation: str,
    retry_config: Optional[RetryConfig] = None,
    timeout_seconds: float = 30.0,
    timeout_strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE,
    use_circuit_breaker: bool = True
):
    """Context manager for adding resilience patterns to database operations.
    
    This context manager combines retry, circuit breaker, and timeout patterns.
    
    Args:
        database: The name of the database
        operation: The operation being performed
        retry_config: The retry configuration, or None to use defaults
        timeout_seconds: Timeout for the operation in seconds
        timeout_strategy: Timeout strategy to use
        use_circuit_breaker: Whether to use the circuit breaker pattern
        
    Yields:
        None
    """
    # Get the circuit breaker if needed
    circuit_breaker = None
    if use_circuit_breaker:
        circuit_breaker = circuit_breaker_registry.get_breaker(database, operation)
    
    # Start the timeout
    start_time = time.time()
    
    try:
        # Use the circuit breaker if needed
        if circuit_breaker:
            async with circuit_breaker:
                yield
        else:
            yield
    except Exception as e:
        # Create an error context
        error_context = ErrorContext(
            component="database",
            operation=f"{database}.{operation}",
            severity=ErrorSeverity.ERROR,
            category=DatabaseErrorCategory.QUERY
        )
        
        # Check if we should retry
        retry_config = retry_config or DATABASE_QUERY_RETRY_CONFIG
        if retry_config.should_retry(e, 0):
            # This is a retryable error, so we'll let the retry decorator handle it
            raise
        
        # This is not a retryable error, so we'll enhance it and re-raise
        raise EnhancedError(str(e), error_context, e) from e
    
    # Check if we've exceeded the timeout
    elapsed_time = time.time() - start_time
    if elapsed_time > timeout_seconds:
        # Create an error context
        error_context = ErrorContext(
            component="database",
            operation=f"{database}.{operation}",
            severity=ErrorSeverity.ERROR,
            category=DatabaseErrorCategory.TIMEOUT
        )
        
        # Raise a timeout error
        raise EnhancedError(
            f"Operation {database}.{operation} timed out after {elapsed_time:.2f}s",
            error_context,
            TimeoutError(f"Operation timed out after {elapsed_time:.2f}s")
        )


def wrap_database_result(
    database: str,
    operation: str,
    result: DatabaseResult
) -> DatabaseResult:
    """Wrap a database result with additional error handling.
    
    Args:
        database: The name of the database
        operation: The operation being performed
        result: The database result to wrap
        
    Returns:
        The wrapped database result
    """
    if not result.success:
        # Create an error context
        error_context = ErrorContext(
            component="database",
            operation=f"{database}.{operation}",
            severity=ErrorSeverity.ERROR,
            category=DatabaseErrorCategory.QUERY,
            metadata=result.metadata
        )
        
        # Enhance the error
        error = result.error
        if error:
            enhanced_error = EnhancedError(str(error), error_context, error)
            result.error = enhanced_error
        
        # Log the error
        logger.error(f"Database operation {database}.{operation} failed: {result.error}")
    
    return result
