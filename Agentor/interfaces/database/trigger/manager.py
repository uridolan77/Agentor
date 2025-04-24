"""
Trigger manager for the Agentor framework.

This module provides a manager for database trigger management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import TriggerConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Trigger type


class TriggerManager(Generic[T]):
    """Manager for database triggers."""
    
    def __init__(self, trigger_class: Type[T], config: TriggerConfig):
        """Initialize the trigger manager.
        
        Args:
            trigger_class: The trigger class
            config: The trigger configuration
        """
        self.trigger_class = trigger_class
        self.config = config
        self.triggers: Dict[str, T] = {}
        self.trigger_tasks: Dict[str, asyncio.Task] = {}
        self.trigger_metrics: Dict[str, Dict[str, Any]] = {}
        self.trigger_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_trigger(self, name: str, **kwargs) -> T:
        """Get a trigger.
        
        Args:
            name: The name of the trigger
            **kwargs: Additional arguments for the trigger
            
        Returns:
            The trigger
        """
        # Check if the trigger exists
        if name in self.triggers:
            return self.triggers[name]
        
        # Create a lock for the trigger
        if name not in self.trigger_locks:
            self.trigger_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.trigger_locks[name]:
            # Check again in case another task created the trigger
            if name in self.triggers:
                return self.triggers[name]
            
            # Create the trigger
            trigger = self.trigger_class(name=name, config=self.config, **kwargs)
            
            # Initialize the trigger
            await trigger.initialize()
            
            # Store the trigger
            self.triggers[name] = trigger
            
            # Initialize metrics
            self.trigger_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0
            }
            
            # Create a task for the trigger
            self.trigger_tasks[name] = self.loop.create_task(self._trigger_maintenance_loop(name))
            
            return trigger
    
    async def close_trigger(self, name: str) -> None:
        """Close a trigger.
        
        Args:
            name: The name of the trigger
        """
        # Check if the trigger exists
        if name not in self.triggers:
            return
        
        # Acquire the lock
        async with self.trigger_locks[name]:
            # Get the trigger
            trigger = self.triggers[name]
            
            # Close the trigger
            await trigger.close()
            
            # Cancel the trigger task
            if name in self.trigger_tasks:
                self.trigger_tasks[name].cancel()
                try:
                    await self.trigger_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.trigger_tasks[name]
            
            # Remove the trigger
            del self.triggers[name]
            
            # Remove the metrics
            if name in self.trigger_metrics:
                del self.trigger_metrics[name]
    
    async def close_all_triggers(self) -> None:
        """Close all triggers."""
        # Get all trigger names
        trigger_names = list(self.triggers.keys())
        
        # Close each trigger
        for name in trigger_names:
            await self.close_trigger(name)
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get trigger metrics.
        
        Args:
            name: The name of the trigger, or None for all triggers
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the trigger exists
            if name not in self.trigger_metrics:
                return {}
            
            # Get the trigger metrics
            return self.trigger_metrics[name]
        else:
            # Get all trigger metrics
            return self.trigger_metrics
    
    async def _trigger_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for a trigger.
        
        Args:
            name: The name of the trigger
        """
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Check if the trigger exists
                if name not in self.triggers:
                    break
                
                # Get the trigger
                trigger = self.triggers[name]
                
                try:
                    # Update metrics
                    trigger_metrics = await trigger.get_metrics()
                    if trigger_metrics:
                        self.trigger_metrics[name].update(trigger_metrics)
                except Exception as e:
                    logger.error(f"Error in trigger maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in trigger maintenance loop for {name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all triggers."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all triggers
                for name, trigger in list(self.triggers.items()):
                    try:
                        # Get the trigger metrics
                        trigger_metrics = await trigger.get_metrics()
                        
                        # Update the metrics
                        if trigger_metrics:
                            self.trigger_metrics[name].update(trigger_metrics)
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
        for task in self.trigger_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
