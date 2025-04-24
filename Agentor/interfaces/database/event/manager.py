"""
Event scheduling manager for the Agentor framework.

This module provides a manager for database event scheduling.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import EventConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Event type


class EventManager(Generic[T]):
    """Manager for database events."""
    
    def __init__(self, event_class: Type[T], config: EventConfig):
        """Initialize the event manager.
        
        Args:
            event_class: The event class
            config: The event configuration
        """
        self.event_class = event_class
        self.config = config
        self.events: Dict[str, T] = {}
        self.event_tasks: Dict[str, asyncio.Task] = {}
        self.event_metrics: Dict[str, Dict[str, Any]] = {}
        self.event_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_event(self, name: str, **kwargs) -> T:
        """Get an event.
        
        Args:
            name: The name of the event
            **kwargs: Additional arguments for the event
            
        Returns:
            The event
        """
        # Check if the event exists
        if name in self.events:
            return self.events[name]
        
        # Create a lock for the event
        if name not in self.event_locks:
            self.event_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.event_locks[name]:
            # Check again in case another task created the event
            if name in self.events:
                return self.events[name]
            
            # Create the event
            event = self.event_class(name=name, config=self.config, **kwargs)
            
            # Initialize the event
            await event.initialize()
            
            # Store the event
            self.events[name] = event
            
            # Initialize metrics
            self.event_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0
            }
            
            # Create a task for the event
            self.event_tasks[name] = self.loop.create_task(self._event_maintenance_loop(name))
            
            return event
    
    async def close_event(self, name: str) -> None:
        """Close an event.
        
        Args:
            name: The name of the event
        """
        # Check if the event exists
        if name not in self.events:
            return
        
        # Acquire the lock
        async with self.event_locks[name]:
            # Get the event
            event = self.events[name]
            
            # Close the event
            await event.close()
            
            # Cancel the event task
            if name in self.event_tasks:
                self.event_tasks[name].cancel()
                try:
                    await self.event_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.event_tasks[name]
            
            # Remove the event
            del self.events[name]
            
            # Remove the metrics
            if name in self.event_metrics:
                del self.event_metrics[name]
    
    async def close_all_events(self) -> None:
        """Close all events."""
        # Get all event names
        event_names = list(self.events.keys())
        
        # Close each event
        for name in event_names:
            await self.close_event(name)
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get event metrics.
        
        Args:
            name: The name of the event, or None for all events
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the event exists
            if name not in self.event_metrics:
                return {}
            
            # Get the event metrics
            return self.event_metrics[name]
        else:
            # Get all event metrics
            return self.event_metrics
    
    async def _event_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for an event.
        
        Args:
            name: The name of the event
        """
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Check if the event exists
                if name not in self.events:
                    break
                
                # Get the event
                event = self.events[name]
                
                try:
                    # Update metrics
                    event_metrics = await event.get_metrics()
                    if event_metrics:
                        self.event_metrics[name].update(event_metrics)
                except Exception as e:
                    logger.error(f"Error in event maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in event maintenance loop for {name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all events."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all events
                for name, event in list(self.events.items()):
                    try:
                        # Get the event metrics
                        event_metrics = await event.get_metrics()
                        
                        # Update the metrics
                        if event_metrics:
                            self.event_metrics[name].update(event_metrics)
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
        for task in self.event_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
