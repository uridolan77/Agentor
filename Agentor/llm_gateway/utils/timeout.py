"""
Timeout management for the Agentor framework.

This module provides mechanisms for managing timeouts in asynchronous operations,
with adaptive timeouts based on historical performance.
"""

from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Type, Tuple, Awaitable
import time
import logging
import asyncio
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field
import statistics
from collections import deque

from agentor.llm_gateway.utils.error_handler import ErrorCategory, ErrorSeverity, ErrorContext, EnhancedError

logger = logging.getLogger(__name__)

# Define type variables for type hints
T = TypeVar('T')


class TimeoutStrategy(Enum):
    """Strategies for timeout calculation."""
    FIXED = "fixed"
    PERCENTILE = "percentile"
    ADAPTIVE = "adaptive"
    DYNAMIC = "dynamic"


class TimeoutError(asyncio.TimeoutError):
    """Error raised when an operation times out."""
    
    def __init__(self, message: str, component: str, operation: str, timeout: float):
        """Initialize the timeout error.
        
        Args:
            message: The error message
            component: The component where the timeout occurred
            operation: The operation that timed out
            timeout: The timeout value in seconds
        """
        self.component = component
        self.operation = operation
        self.timeout = timeout
        super().__init__(
            f"{message} [component={component}, operation={operation}, timeout={timeout:.2f}s]"
        )


@dataclass
class TimeoutStats:
    """Statistics for timeout calculations."""
    
    component: str
    """The component being monitored."""
    
    operation: str
    """The operation being monitored."""
    
    strategy: TimeoutStrategy
    """The timeout strategy being used."""
    
    base_timeout: float
    """The base timeout value in seconds."""
    
    max_timeout: float
    """The maximum timeout value in seconds."""
    
    current_timeout: float
    """The current timeout value in seconds."""
    
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    """Recent execution times in seconds."""
    
    timeout_count: int = 0
    """Number of timeouts that have occurred."""
    
    success_count: int = 0
    """Number of successful executions."""
    
    total_execution_time: float = 0.0
    """Total execution time in seconds."""
    
    def add_execution_time(self, execution_time: float, success: bool = True):
        """Add an execution time to the stats.
        
        Args:
            execution_time: The execution time in seconds
            success: Whether the execution was successful
        """
        self.execution_times.append(execution_time)
        self.total_execution_time += execution_time
        
        if success:
            self.success_count += 1
        else:
            self.timeout_count += 1
    
    def calculate_timeout(self) -> float:
        """Calculate the timeout value based on the strategy.
        
        Returns:
            The calculated timeout value in seconds
        """
        if not self.execution_times:
            return self.base_timeout
        
        if self.strategy == TimeoutStrategy.FIXED:
            return self.base_timeout
        
        elif self.strategy == TimeoutStrategy.PERCENTILE:
            # Use the 95th percentile of execution times
            if len(self.execution_times) >= 10:
                p95 = statistics.quantiles(self.execution_times, n=20)[-1]
                # Add a safety margin
                timeout = p95 * 1.5
            else:
                # Not enough data, use a conservative estimate
                timeout = max(self.execution_times) * 2.0
        
        elif self.strategy == TimeoutStrategy.ADAPTIVE:
            # Use the mean plus 2 standard deviations
            if len(self.execution_times) >= 10:
                mean = statistics.mean(self.execution_times)
                stdev = statistics.stdev(self.execution_times)
                timeout = mean + 2 * stdev
            else:
                # Not enough data, use a conservative estimate
                timeout = max(self.execution_times) * 2.0
        
        elif self.strategy == TimeoutStrategy.DYNAMIC:
            # Adjust based on recent timeouts
            if len(self.execution_times) >= 10:
                mean = statistics.mean(self.execution_times)
                stdev = statistics.stdev(self.execution_times)
                
                # Calculate a base timeout
                base = mean + 2 * stdev
                
                # Adjust based on timeout rate
                total = self.success_count + self.timeout_count
                if total > 0:
                    timeout_rate = self.timeout_count / total
                    
                    # Increase timeout if we're seeing a lot of timeouts
                    if timeout_rate > 0.1:
                        # Increase by up to 50% based on timeout rate
                        adjustment = 1.0 + (timeout_rate * 5.0)
                        timeout = base * adjustment
                    else:
                        timeout = base
                else:
                    timeout = base
            else:
                # Not enough data, use a conservative estimate
                timeout = max(self.execution_times) * 2.0
        
        else:
            # Default to fixed timeout
            timeout = self.base_timeout
        
        # Ensure the timeout is within bounds
        timeout = max(self.base_timeout, min(timeout, self.max_timeout))
        
        return timeout
    
    def update_timeout(self):
        """Update the current timeout value."""
        self.current_timeout = self.calculate_timeout()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the stats to a dictionary.
        
        Returns:
            Dictionary representation of the stats
        """
        return {
            'component': self.component,
            'operation': self.operation,
            'strategy': self.strategy.value,
            'base_timeout': self.base_timeout,
            'max_timeout': self.max_timeout,
            'current_timeout': self.current_timeout,
            'timeout_count': self.timeout_count,
            'success_count': self.success_count,
            'total_execution_time': self.total_execution_time,
            'avg_execution_time': (
                self.total_execution_time / (self.success_count + self.timeout_count)
                if (self.success_count + self.timeout_count) > 0 else 0.0
            ),
            'timeout_rate': (
                self.timeout_count / (self.success_count + self.timeout_count)
                if (self.success_count + self.timeout_count) > 0 else 0.0
            ),
            'execution_times_count': len(self.execution_times),
            'min_execution_time': min(self.execution_times) if self.execution_times else 0.0,
            'max_execution_time': max(self.execution_times) if self.execution_times else 0.0,
            'mean_execution_time': (
                statistics.mean(self.execution_times) if self.execution_times else 0.0
            ),
            'median_execution_time': (
                statistics.median(self.execution_times) if self.execution_times else 0.0
            )
        }


class TimeoutManager:
    """Manager for timeouts across the system."""
    
    def __init__(self):
        """Initialize the timeout manager."""
        self.stats: Dict[Tuple[str, str], TimeoutStats] = {}
        self.lock = asyncio.Lock()
    
    async def get_or_create_stats(
        self,
        component: str,
        operation: str,
        strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE,
        base_timeout: float = 10.0,
        max_timeout: float = 60.0
    ) -> TimeoutStats:
        """Get or create timeout stats for a component and operation.
        
        Args:
            component: The component to monitor
            operation: The operation to monitor
            strategy: The timeout strategy to use
            base_timeout: The base timeout value in seconds
            max_timeout: The maximum timeout value in seconds
            
        Returns:
            The timeout stats
        """
        key = (component, operation)
        
        async with self.lock:
            if key not in self.stats:
                self.stats[key] = TimeoutStats(
                    component=component,
                    operation=operation,
                    strategy=strategy,
                    base_timeout=base_timeout,
                    max_timeout=max_timeout,
                    current_timeout=base_timeout
                )
            
            return self.stats[key]
    
    async def execute_with_timeout(
        self,
        func: Callable[..., Awaitable[T]],
        component: str,
        operation: str,
        strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE,
        base_timeout: float = 10.0,
        max_timeout: float = 60.0,
        *args,
        **kwargs
    ) -> T:
        """Execute a function with timeout protection.
        
        Args:
            func: The async function to execute
            component: The component being monitored
            operation: The operation being performed
            strategy: The timeout strategy to use
            base_timeout: The base timeout value in seconds
            max_timeout: The maximum timeout value in seconds
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function
            
        Raises:
            TimeoutError: If the operation times out
            Exception: Any exception raised by the function
        """
        # Get or create the stats
        stats = await self.get_or_create_stats(
            component=component,
            operation=operation,
            strategy=strategy,
            base_timeout=base_timeout,
            max_timeout=max_timeout
        )
        
        # Update the timeout value
        stats.update_timeout()
        timeout = stats.current_timeout
        
        # Execute with timeout
        start_time = time.time()
        try:
            # Create a task for the function
            task = asyncio.create_task(func(*args, **kwargs))
            
            # Wait for the task with timeout
            try:
                result = await asyncio.wait_for(task, timeout=timeout)
                
                # Record successful execution
                execution_time = time.time() - start_time
                stats.add_execution_time(execution_time, success=True)
                
                return result
            
            except asyncio.TimeoutError:
                # Cancel the task
                task.cancel()
                
                # Record timeout
                execution_time = time.time() - start_time
                stats.add_execution_time(execution_time, success=False)
                
                # Raise a custom timeout error
                raise TimeoutError(
                    f"Operation {operation} timed out after {execution_time:.2f}s",
                    component,
                    operation,
                    timeout
                )
        
        except Exception as e:
            # If it's not a timeout, record as a successful execution
            # (from a timeout perspective, even if it failed for other reasons)
            if not isinstance(e, asyncio.TimeoutError):
                execution_time = time.time() - start_time
                stats.add_execution_time(execution_time, success=True)
            
            # Re-raise the exception
            raise
    
    def get_stats(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None
    ) -> Union[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]:
        """Get statistics for timeouts.
        
        Args:
            component: The component to get stats for, or None for all components
            operation: The operation to get stats for, or None for all operations
            
        Returns:
            Dictionary of timeout statistics
        """
        if component and operation:
            key = (component, operation)
            if key in self.stats:
                return self.stats[key].to_dict()
            else:
                return {}
        else:
            return {
                f"{comp}.{op}": stats.to_dict()
                for (comp, op), stats in self.stats.items()
                if (component is None or comp == component) and
                   (operation is None or op == operation)
            }


# Create a global timeout manager
timeout_manager = TimeoutManager()


def with_timeout(
    component: str,
    operation: str,
    strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE,
    base_timeout: float = 10.0,
    max_timeout: float = 60.0
):
    """Decorator for executing a function with timeout protection.
    
    Args:
        component: The component being monitored
        operation: The operation being performed
        strategy: The timeout strategy to use
        base_timeout: The base timeout value in seconds
        max_timeout: The maximum timeout value in seconds
        
    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await timeout_manager.execute_with_timeout(
                func=func,
                component=component,
                operation=operation,
                strategy=strategy,
                base_timeout=base_timeout,
                max_timeout=max_timeout,
                *args,
                **kwargs
            )
        return wrapper
    return decorator
