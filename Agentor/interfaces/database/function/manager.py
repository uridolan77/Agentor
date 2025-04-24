"""
User-defined function manager for the Agentor framework.

This module provides a manager for database user-defined function management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import FunctionConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Function type


class FunctionManager(Generic[T]):
    """Manager for database user-defined functions."""
    
    def __init__(self, function_class: Type[T], config: FunctionConfig):
        """Initialize the function manager.
        
        Args:
            function_class: The function class
            config: The function configuration
        """
        self.function_class = function_class
        self.config = config
        self.functions: Dict[str, T] = {}
        self.function_tasks: Dict[str, asyncio.Task] = {}
        self.function_metrics: Dict[str, Dict[str, Any]] = {}
        self.function_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_function(self, name: str, **kwargs) -> T:
        """Get a user-defined function.
        
        Args:
            name: The name of the function
            **kwargs: Additional arguments for the function
            
        Returns:
            The user-defined function
        """
        # Check if the function exists
        if name in self.functions:
            return self.functions[name]
        
        # Create a lock for the function
        if name not in self.function_locks:
            self.function_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.function_locks[name]:
            # Check again in case another task created the function
            if name in self.functions:
                return self.functions[name]
            
            # Create the function
            function = self.function_class(name=name, config=self.config, **kwargs)
            
            # Initialize the function
            await function.initialize()
            
            # Store the function
            self.functions[name] = function
            
            # Initialize metrics
            self.function_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "max_execution_time": 0.0,
                "min_execution_time": float('inf')
            }
            
            # Create a task for the function
            self.function_tasks[name] = self.loop.create_task(self._function_maintenance_loop(name))
            
            return function
    
    async def close_function(self, name: str) -> None:
        """Close a user-defined function.
        
        Args:
            name: The name of the function
        """
        # Check if the function exists
        if name not in self.functions:
            return
        
        # Acquire the lock
        async with self.function_locks[name]:
            # Get the function
            function = self.functions[name]
            
            # Close the function
            await function.close()
            
            # Cancel the function task
            if name in self.function_tasks:
                self.function_tasks[name].cancel()
                try:
                    await self.function_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.function_tasks[name]
            
            # Remove the function
            del self.functions[name]
            
            # Remove the metrics
            if name in self.function_metrics:
                del self.function_metrics[name]
    
    async def close_all_functions(self) -> None:
        """Close all user-defined functions."""
        # Get all function names
        function_names = list(self.functions.keys())
        
        # Close each function
        for name in function_names:
            await self.close_function(name)
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get user-defined function metrics.
        
        Args:
            name: The name of the function, or None for all functions
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the function exists
            if name not in self.function_metrics:
                return {}
            
            # Get the function metrics
            return self.function_metrics[name]
        else:
            # Get all function metrics
            return self.function_metrics
    
    async def _function_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for a user-defined function.
        
        Args:
            name: The name of the function
        """
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Check if the function exists
                if name not in self.functions:
                    break
                
                # Get the function
                function = self.functions[name]
                
                try:
                    # Update metrics
                    function_metrics = await function.get_metrics()
                    if function_metrics:
                        self.function_metrics[name].update(function_metrics)
                except Exception as e:
                    logger.error(f"Error in function maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in function maintenance loop for {name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all user-defined functions."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all functions
                for name, function in list(self.functions.items()):
                    try:
                        # Get the function metrics
                        function_metrics = await function.get_metrics()
                        
                        # Update the metrics
                        if function_metrics:
                            self.function_metrics[name].update(function_metrics)
                    except Exception as e:
                        logger.error(f"Error collecting metrics for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in metrics collection loop: {e}")
    
    def __del__(self):
        """Clean up resources when the manager is garbage collected."""
        # Cancel all tasks
        for task in self.function_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
