"""
Adaptive retry strategies for the Agentor framework.

This module provides mechanisms for retrying operations with various backoff strategies,
adapting to different error types and system conditions.
"""

from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Type, Tuple
import time
import random
import logging
import asyncio
import math
from enum import Enum
from functools import wraps

from agentor.llm_gateway.utils.error_handler import ErrorCategory, ErrorSeverity, ErrorContext, EnhancedError

logger = logging.getLogger(__name__)

# Define type variables for type hints
T = TypeVar('T')
E = TypeVar('E', bound=Exception)


class RetryStrategy(Enum):
    """Strategies for retrying operations."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class RetryableError(Exception):
    """Base class for errors that can be retried."""
    pass


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.1,
        retry_on: Optional[List[Type[Exception]]] = None,
        retry_if: Optional[Callable[[Exception], bool]] = None
    ):
        """Initialize the retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            strategy: The retry strategy to use
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter: Random jitter factor to add to delays (0.0 to 1.0)
            retry_on: List of exception types to retry on
            retry_if: Function that returns True if an exception should be retried
        """
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on or [RetryableError]
        self.retry_if = retry_if
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried.
        
        Args:
            exception: The exception that occurred
            attempt: The current attempt number (0-based)
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        # Check if we've reached the maximum number of retries
        if attempt >= self.max_retries:
            return False
        
        # Check if the exception is of a retryable type
        is_retryable_type = any(isinstance(exception, exc_type) for exc_type in self.retry_on)
        
        # Check if the retry_if function returns True
        custom_check = self.retry_if(exception) if self.retry_if else True
        
        return is_retryable_type and custom_check
    
    def get_delay(self, attempt: int) -> float:
        """Calculate the delay before the next retry.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            The delay in seconds
        """
        # Calculate the base delay based on the strategy
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        
        elif self.strategy == RetryStrategy.FIBONACCI:
            # Calculate the (attempt+2)th Fibonacci number
            a, b = 1, 1
            for _ in range(attempt):
                a, b = b, a + b
            delay = self.base_delay * a
        
        elif self.strategy == RetryStrategy.RANDOM:
            # Random delay between base_delay and max_delay
            delay = random.uniform(self.base_delay, self.max_delay)
        
        elif self.strategy == RetryStrategy.ADAPTIVE:
            # Start with exponential backoff
            delay = self.base_delay * (2 ** attempt)
            
            # Add a random factor based on the attempt number
            random_factor = random.uniform(0.5, 1.5) * (1 + attempt * 0.1)
            delay *= random_factor
        
        else:
            # Default to exponential backoff
            delay = self.base_delay * (2 ** attempt)
        
        # Apply jitter
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        # Ensure the delay is within bounds
        delay = max(0, min(delay, self.max_delay))
        
        return delay


class RetryState:
    """State information for a retry operation."""
    
    def __init__(self, config: RetryConfig):
        """Initialize the retry state.
        
        Args:
            config: The retry configuration
        """
        self.config = config
        self.attempts = 0
        self.start_time = time.time()
        self.last_attempt_time = 0
        self.last_delay = 0
        self.exceptions: List[Exception] = []
    
    def record_attempt(self, exception: Optional[Exception] = None):
        """Record an attempt.
        
        Args:
            exception: The exception that occurred, if any
        """
        self.attempts += 1
        self.last_attempt_time = time.time()
        
        if exception:
            self.exceptions.append(exception)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retry operation.
        
        Returns:
            Dictionary of retry statistics
        """
        return {
            'attempts': self.attempts,
            'elapsed_time': time.time() - self.start_time,
            'last_delay': self.last_delay,
            'exceptions': [str(e) for e in self.exceptions]
        }


class RetryContext:
    """Context information for a retry operation."""
    
    def __init__(
        self,
        component: str,
        operation: str,
        config: RetryConfig,
        state: RetryState
    ):
        """Initialize the retry context.
        
        Args:
            component: The component performing the operation
            operation: The operation being performed
            config: The retry configuration
            state: The current retry state
        """
        self.component = component
        self.operation = operation
        self.config = config
        self.state = state


async def retry_async(
    func: Callable[..., Awaitable[T]],
    component: str,
    operation: str,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """Retry an async function with the specified configuration.
    
    Args:
        func: The async function to retry
        component: The component performing the operation
        operation: The operation being performed
        config: The retry configuration, or None to use defaults
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Use default config if none provided
    if config is None:
        config = RetryConfig()
    
    # Initialize retry state
    state = RetryState(config)
    
    # Create retry context
    context = RetryContext(component, operation, config, state)
    
    # Try the operation with retries
    last_exception = None
    
    while True:
        try:
            # Attempt the operation
            result = await func(*args, **kwargs)
            
            # Record the successful attempt
            state.record_attempt()
            
            # Log the success after retries
            if state.attempts > 1:
                logger.info(
                    f"Operation {component}.{operation} succeeded after {state.attempts} attempts"
                )
            
            return result
        
        except Exception as e:
            # Record the failed attempt
            state.record_attempt(e)
            last_exception = e
            
            # Check if we should retry
            if not config.should_retry(e, state.attempts - 1):
                # Log the failure
                logger.warning(
                    f"Operation {component}.{operation} failed after {state.attempts} attempts: {str(e)}"
                )
                
                # Create an error context
                error_context = ErrorContext(
                    component=component,
                    operation=operation,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.UNKNOWN,
                    metadata={"retry_stats": state.get_stats()}
                )
                
                # Raise an enhanced error
                raise EnhancedError(str(e), error_context, e) from e
            
            # Calculate the delay
            delay = config.get_delay(state.attempts - 1)
            state.last_delay = delay
            
            # Log the retry
            logger.info(
                f"Retrying operation {component}.{operation} after error: {str(e)}. "
                f"Attempt {state.attempts}/{config.max_retries + 1}, "
                f"waiting {delay:.2f}s"
            )
            
            # Wait before retrying
            await asyncio.sleep(delay)


def retry_async_decorator(
    component: str,
    operation: str,
    config: Optional[RetryConfig] = None
):
    """Decorator for retrying async functions.
    
    Args:
        component: The component performing the operation
        operation: The operation being performed
        config: The retry configuration, or None to use defaults
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, component, operation, config, *args, **kwargs)
        return wrapper
    return decorator


def retry(
    func: Callable[..., T],
    component: str,
    operation: str,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """Retry a function with the specified configuration.
    
    Args:
        func: The function to retry
        component: The component performing the operation
        operation: The operation being performed
        config: The retry configuration, or None to use defaults
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Use default config if none provided
    if config is None:
        config = RetryConfig()
    
    # Initialize retry state
    state = RetryState(config)
    
    # Create retry context
    context = RetryContext(component, operation, config, state)
    
    # Try the operation with retries
    last_exception = None
    
    while True:
        try:
            # Attempt the operation
            result = func(*args, **kwargs)
            
            # Record the successful attempt
            state.record_attempt()
            
            # Log the success after retries
            if state.attempts > 1:
                logger.info(
                    f"Operation {component}.{operation} succeeded after {state.attempts} attempts"
                )
            
            return result
        
        except Exception as e:
            # Record the failed attempt
            state.record_attempt(e)
            last_exception = e
            
            # Check if we should retry
            if not config.should_retry(e, state.attempts - 1):
                # Log the failure
                logger.warning(
                    f"Operation {component}.{operation} failed after {state.attempts} attempts: {str(e)}"
                )
                
                # Create an error context
                error_context = ErrorContext(
                    component=component,
                    operation=operation,
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.UNKNOWN,
                    metadata={"retry_stats": state.get_stats()}
                )
                
                # Raise an enhanced error
                raise EnhancedError(str(e), error_context, e) from e
            
            # Calculate the delay
            delay = config.get_delay(state.attempts - 1)
            state.last_delay = delay
            
            # Log the retry
            logger.info(
                f"Retrying operation {component}.{operation} after error: {str(e)}. "
                f"Attempt {state.attempts}/{config.max_retries + 1}, "
                f"waiting {delay:.2f}s"
            )
            
            # Wait before retrying
            time.sleep(delay)


def retry_decorator(
    component: str,
    operation: str,
    config: Optional[RetryConfig] = None
):
    """Decorator for retrying functions.
    
    Args:
        component: The component performing the operation
        operation: The operation being performed
        config: The retry configuration, or None to use defaults
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry(func, component, operation, config, *args, **kwargs)
        return wrapper
    return decorator


# Common retry configurations for different error types
NETWORK_RETRY_CONFIG = RetryConfig(
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

RATE_LIMIT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=2.0,
    max_delay=60.0,
    jitter=0.1
)

TRANSIENT_ERROR_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.LINEAR,
    base_delay=1.0,
    max_delay=10.0,
    jitter=0.1
)

ADAPTIVE_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    strategy=RetryStrategy.ADAPTIVE,
    base_delay=1.0,
    max_delay=60.0,
    jitter=0.2
)
