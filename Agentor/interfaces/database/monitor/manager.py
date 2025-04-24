"""
Performance monitor manager for the Agentor framework.

This module provides a manager for database performance monitoring.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import MonitoringConfig, MonitoringLevel

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Monitor type


class PerformanceMonitorManager(Generic[T]):
    """Manager for database performance monitoring."""
    
    def __init__(self, monitor_class: Type[T], config: MonitoringConfig):
        """Initialize the performance monitor manager.
        
        Args:
            monitor_class: The performance monitor class
            config: The monitoring configuration
        """
        self.monitor_class = monitor_class
        self.config = config
        self.monitors: Dict[str, T] = {}
        self.monitor_tasks: Dict[str, asyncio.Task] = {}
        self.monitor_metrics: Dict[str, Dict[str, Any]] = {}
        self.monitor_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_monitor(self, name: str, **kwargs) -> T:
        """Get a performance monitor.
        
        Args:
            name: The name of the monitor
            **kwargs: Additional arguments for the monitor
            
        Returns:
            The performance monitor
        """
        # Check if the monitor exists
        if name in self.monitors:
            return self.monitors[name]
        
        # Create a lock for the monitor
        if name not in self.monitor_locks:
            self.monitor_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.monitor_locks[name]:
            # Check again in case another task created the monitor
            if name in self.monitors:
                return self.monitors[name]
            
            # Create the monitor
            monitor = self.monitor_class(name=name, config=self.config, **kwargs)
            
            # Initialize the monitor
            await monitor.initialize()
            
            # Store the monitor
            self.monitors[name] = monitor
            
            # Initialize metrics
            self.monitor_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_queries": 0,
                "slow_queries": 0,
                "very_slow_queries": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "max_execution_time": 0.0,
                "min_execution_time": float('inf'),
                "total_rows": 0,
                "avg_rows": 0.0,
                "max_rows": 0,
                "min_rows": float('inf'),
                "total_alerts": 0
            }
            
            # Create a task for the monitor
            self.monitor_tasks[name] = self.loop.create_task(self._monitor_maintenance_loop(name))
            
            return monitor
    
    async def close_monitor(self, name: str) -> None:
        """Close a performance monitor.
        
        Args:
            name: The name of the monitor
        """
        # Check if the monitor exists
        if name not in self.monitors:
            return
        
        # Acquire the lock
        async with self.monitor_locks[name]:
            # Get the monitor
            monitor = self.monitors[name]
            
            # Close the monitor
            await monitor.close()
            
            # Cancel the monitor task
            if name in self.monitor_tasks:
                self.monitor_tasks[name].cancel()
                try:
                    await self.monitor_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.monitor_tasks[name]
            
            # Remove the monitor
            del self.monitors[name]
            
            # Remove the metrics
            if name in self.monitor_metrics:
                del self.monitor_metrics[name]
    
    async def close_all_monitors(self) -> None:
        """Close all performance monitors."""
        # Get all monitor names
        monitor_names = list(self.monitors.keys())
        
        # Close each monitor
        for name in monitor_names:
            await self.close_monitor(name)
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance monitor metrics.
        
        Args:
            name: The name of the monitor, or None for all monitors
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the monitor exists
            if name not in self.monitor_metrics:
                return {}
            
            # Get the monitor metrics
            return self.monitor_metrics[name]
        else:
            # Get all monitor metrics
            return self.monitor_metrics
    
    async def _monitor_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for a performance monitor.
        
        Args:
            name: The name of the monitor
        """
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Check if the monitor exists
                if name not in self.monitors:
                    break
                
                # Get the monitor
                monitor = self.monitors[name]
                
                try:
                    # Update metrics
                    monitor_metrics = await monitor.get_metrics()
                    if monitor_metrics:
                        self.monitor_metrics[name].update(monitor_metrics)
                except Exception as e:
                    logger.error(f"Error in monitor maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in monitor maintenance loop for {name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all performance monitors."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all monitors
                for name, monitor in list(self.monitors.items()):
                    try:
                        # Get the monitor metrics
                        monitor_metrics = await monitor.get_metrics()
                        
                        # Update the metrics
                        if monitor_metrics:
                            self.monitor_metrics[name].update(monitor_metrics)
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
        for task in self.monitor_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
