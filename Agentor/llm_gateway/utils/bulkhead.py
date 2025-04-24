"""
Bulkhead pattern implementation for the Agentor framework.

This module provides a bulkhead pattern implementation for isolating resources
and preventing cascading failures across the system.
"""

from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Type, Tuple, Awaitable
import time
import logging
import asyncio
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field

from agentor.llm_gateway.utils.error_handler import ErrorCategory, ErrorSeverity, ErrorContext, EnhancedError

logger = logging.getLogger(__name__)

# Define type variables for type hints
T = TypeVar('T')


class BulkheadRejectedError(Exception):
    """Error raised when a request is rejected by a bulkhead."""
    
    def __init__(self, message: str, component: str, max_concurrent: int):
        """Initialize the bulkhead rejected error.
        
        Args:
            message: The error message
            component: The component that rejected the request
            max_concurrent: The maximum number of concurrent executions
        """
        self.component = component
        self.max_concurrent = max_concurrent
        super().__init__(f"{message} [component={component}, max_concurrent={max_concurrent}]")


@dataclass
class BulkheadStats:
    """Statistics for a bulkhead."""
    
    component: str
    """The component protected by the bulkhead."""
    
    max_concurrent: int
    """Maximum number of concurrent executions."""
    
    max_queue_size: int
    """Maximum size of the waiting queue."""
    
    current_concurrent: int = 0
    """Current number of concurrent executions."""
    
    current_queue_size: int = 0
    """Current size of the waiting queue."""
    
    total_executions: int = 0
    """Total number of executions."""
    
    successful_executions: int = 0
    """Number of successful executions."""
    
    failed_executions: int = 0
    """Number of failed executions."""
    
    rejected_executions: int = 0
    """Number of rejected executions."""
    
    total_execution_time: float = 0.0
    """Total execution time in seconds."""
    
    peak_concurrent: int = 0
    """Peak number of concurrent executions."""
    
    peak_queue_size: int = 0
    """Peak size of the waiting queue."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stats to a dictionary.
        
        Returns:
            Dictionary representation of the stats
        """
        return {
            'component': self.component,
            'max_concurrent': self.max_concurrent,
            'max_queue_size': self.max_queue_size,
            'current_concurrent': self.current_concurrent,
            'current_queue_size': self.current_queue_size,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'failed_executions': self.failed_executions,
            'rejected_executions': self.rejected_executions,
            'total_execution_time': self.total_execution_time,
            'peak_concurrent': self.peak_concurrent,
            'peak_queue_size': self.peak_queue_size,
            'avg_execution_time': (
                self.total_execution_time / self.successful_executions
                if self.successful_executions > 0 else 0.0
            ),
            'success_rate': (
                self.successful_executions / (self.successful_executions + self.failed_executions)
                if (self.successful_executions + self.failed_executions) > 0 else 0.0
            )
        }


class Bulkhead:
    """Bulkhead pattern implementation for isolating resources."""
    
    def __init__(
        self,
        component: str,
        max_concurrent: int = 10,
        max_queue_size: int = 20,
        execution_timeout: Optional[float] = None
    ):
        """Initialize the bulkhead.
        
        Args:
            component: The component protected by the bulkhead
            max_concurrent: Maximum number of concurrent executions
            max_queue_size: Maximum size of the waiting queue
            execution_timeout: Timeout for executions in seconds, or None for no timeout
        """
        self.component = component
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.execution_timeout = execution_timeout
        
        # Semaphore to limit concurrent executions
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Queue for waiting executions
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Stats
        self.stats = BulkheadStats(
            component=component,
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size
        )
        
        # Lock for updating stats
        self.stats_lock = asyncio.Lock()
    
    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute a function with bulkhead protection.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
            
        Raises:
            BulkheadRejectedError: If the bulkhead rejects the execution
            asyncio.TimeoutError: If the execution times out
            Exception: Any exception raised by the function
        """
        # Check if the queue is full
        if self.queue.full():
            async with self.stats_lock:
                self.stats.rejected_executions += 1
            
            raise BulkheadRejectedError(
                f"Bulkhead queue is full for component {self.component}",
                self.component,
                self.max_concurrent
            )
        
        # Add to queue
        queue_start_time = time.time()
        await self.queue.put(None)
        
        async with self.stats_lock:
            self.stats.current_queue_size = self.queue.qsize()
            self.stats.peak_queue_size = max(self.stats.peak_queue_size, self.stats.current_queue_size)
        
        try:
            # Acquire semaphore
            async with self.semaphore:
                # Remove from queue
                await self.queue.get()
                queue_time = time.time() - queue_start_time
                
                async with self.stats_lock:
                    self.stats.current_queue_size = self.queue.qsize()
                    self.stats.current_concurrent += 1
                    self.stats.peak_concurrent = max(self.stats.peak_concurrent, self.stats.current_concurrent)
                    self.stats.total_executions += 1
                
                # Execute the function with timeout if specified
                start_time = time.time()
                try:
                    if self.execution_timeout:
                        # Execute with timeout
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=self.execution_timeout
                        )
                    else:
                        # Execute without timeout
                        result = await func(*args, **kwargs)
                    
                    # Update stats for successful execution
                    execution_time = time.time() - start_time
                    async with self.stats_lock:
                        self.stats.successful_executions += 1
                        self.stats.total_execution_time += execution_time
                    
                    return result
                
                except Exception as e:
                    # Update stats for failed execution
                    execution_time = time.time() - start_time
                    async with self.stats_lock:
                        self.stats.failed_executions += 1
                        self.stats.total_execution_time += execution_time
                    
                    # Re-raise the exception
                    raise
                
                finally:
                    # Update concurrent count
                    async with self.stats_lock:
                        self.stats.current_concurrent -= 1
        
        finally:
            # Ensure the queue item is removed if we didn't acquire the semaphore
            if self.queue.qsize() > 0:
                try:
                    self.queue.task_done()
                except ValueError:
                    # The task might already be done
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the bulkhead.
        
        Returns:
            Dictionary of bulkhead statistics
        """
        return self.stats.to_dict()


class BulkheadManager:
    """Manager for bulkheads across the system."""
    
    def __init__(self):
        """Initialize the bulkhead manager."""
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.lock = asyncio.Lock()
    
    async def get_or_create_bulkhead(
        self,
        component: str,
        max_concurrent: int = 10,
        max_queue_size: int = 20,
        execution_timeout: Optional[float] = None
    ) -> Bulkhead:
        """Get or create a bulkhead for a component.
        
        Args:
            component: The component to protect
            max_concurrent: Maximum number of concurrent executions
            max_queue_size: Maximum size of the waiting queue
            execution_timeout: Timeout for executions in seconds, or None for no timeout
            
        Returns:
            The bulkhead for the component
        """
        async with self.lock:
            if component not in self.bulkheads:
                self.bulkheads[component] = Bulkhead(
                    component=component,
                    max_concurrent=max_concurrent,
                    max_queue_size=max_queue_size,
                    execution_timeout=execution_timeout
                )
            
            return self.bulkheads[component]
    
    async def execute(
        self,
        component: str,
        func: Callable[..., Awaitable[T]],
        *args,
        max_concurrent: int = 10,
        max_queue_size: int = 20,
        execution_timeout: Optional[float] = None,
        **kwargs
    ) -> T:
        """Execute a function with bulkhead protection.
        
        Args:
            component: The component to protect
            func: The async function to execute
            *args: Positional arguments for the function
            max_concurrent: Maximum number of concurrent executions
            max_queue_size: Maximum size of the waiting queue
            execution_timeout: Timeout for executions in seconds, or None for no timeout
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
            
        Raises:
            BulkheadRejectedError: If the bulkhead rejects the execution
            asyncio.TimeoutError: If the execution times out
            Exception: Any exception raised by the function
        """
        # Get or create the bulkhead
        bulkhead = await self.get_or_create_bulkhead(
            component=component,
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
            execution_timeout=execution_timeout
        )
        
        # Execute with bulkhead protection
        return await bulkhead.execute(func, *args, **kwargs)
    
    def get_stats(self, component: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get statistics for bulkheads.
        
        Args:
            component: The component to get stats for, or None for all components
            
        Returns:
            Dictionary of bulkhead statistics
        """
        if component:
            if component in self.bulkheads:
                return self.bulkheads[component].get_stats()
            else:
                return {}
        else:
            return {
                component: bulkhead.get_stats()
                for component, bulkhead in self.bulkheads.items()
            }


# Create a global bulkhead manager
bulkhead_manager = BulkheadManager()


def with_bulkhead(
    component: str,
    max_concurrent: int = 10,
    max_queue_size: int = 20,
    execution_timeout: Optional[float] = None
):
    """Decorator for executing a function with bulkhead protection.
    
    Args:
        component: The component to protect
        max_concurrent: Maximum number of concurrent executions
        max_queue_size: Maximum size of the waiting queue
        execution_timeout: Timeout for executions in seconds, or None for no timeout
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await bulkhead_manager.execute(
                component=component,
                func=func,
                *args,
                max_concurrent=max_concurrent,
                max_queue_size=max_queue_size,
                execution_timeout=execution_timeout,
                **kwargs
            )
        return wrapper
    return decorator
