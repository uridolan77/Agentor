"""
Stored procedure manager for the Agentor framework.

This module provides a manager for database stored procedure management.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import ProcedureConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Procedure type


class ProcedureManager(Generic[T]):
    """Manager for database stored procedures."""
    
    def __init__(self, procedure_class: Type[T], config: ProcedureConfig):
        """Initialize the procedure manager.
        
        Args:
            procedure_class: The procedure class
            config: The procedure configuration
        """
        self.procedure_class = procedure_class
        self.config = config
        self.procedures: Dict[str, T] = {}
        self.procedure_tasks: Dict[str, asyncio.Task] = {}
        self.procedure_metrics: Dict[str, Dict[str, Any]] = {}
        self.procedure_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_procedure(self, name: str, **kwargs) -> T:
        """Get a stored procedure.
        
        Args:
            name: The name of the procedure
            **kwargs: Additional arguments for the procedure
            
        Returns:
            The stored procedure
        """
        # Check if the procedure exists
        if name in self.procedures:
            return self.procedures[name]
        
        # Create a lock for the procedure
        if name not in self.procedure_locks:
            self.procedure_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.procedure_locks[name]:
            # Check again in case another task created the procedure
            if name in self.procedures:
                return self.procedures[name]
            
            # Create the procedure
            procedure = self.procedure_class(name=name, config=self.config, **kwargs)
            
            # Initialize the procedure
            await procedure.initialize()
            
            # Store the procedure
            self.procedures[name] = procedure
            
            # Initialize metrics
            self.procedure_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "max_execution_time": 0.0,
                "min_execution_time": float('inf'),
                "total_rows": 0,
                "avg_rows": 0.0,
                "max_rows": 0,
                "min_rows": float('inf')
            }
            
            # Create a task for the procedure
            self.procedure_tasks[name] = self.loop.create_task(self._procedure_maintenance_loop(name))
            
            return procedure
    
    async def close_procedure(self, name: str) -> None:
        """Close a stored procedure.
        
        Args:
            name: The name of the procedure
        """
        # Check if the procedure exists
        if name not in self.procedures:
            return
        
        # Acquire the lock
        async with self.procedure_locks[name]:
            # Get the procedure
            procedure = self.procedures[name]
            
            # Close the procedure
            await procedure.close()
            
            # Cancel the procedure task
            if name in self.procedure_tasks:
                self.procedure_tasks[name].cancel()
                try:
                    await self.procedure_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.procedure_tasks[name]
            
            # Remove the procedure
            del self.procedures[name]
            
            # Remove the metrics
            if name in self.procedure_metrics:
                del self.procedure_metrics[name]
    
    async def close_all_procedures(self) -> None:
        """Close all stored procedures."""
        # Get all procedure names
        procedure_names = list(self.procedures.keys())
        
        # Close each procedure
        for name in procedure_names:
            await self.close_procedure(name)
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get stored procedure metrics.
        
        Args:
            name: The name of the procedure, or None for all procedures
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the procedure exists
            if name not in self.procedure_metrics:
                return {}
            
            # Get the procedure metrics
            return self.procedure_metrics[name]
        else:
            # Get all procedure metrics
            return self.procedure_metrics
    
    async def _procedure_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for a stored procedure.
        
        Args:
            name: The name of the procedure
        """
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Check if the procedure exists
                if name not in self.procedures:
                    break
                
                # Get the procedure
                procedure = self.procedures[name]
                
                try:
                    # Update metrics
                    procedure_metrics = await procedure.get_metrics()
                    if procedure_metrics:
                        self.procedure_metrics[name].update(procedure_metrics)
                except Exception as e:
                    logger.error(f"Error in procedure maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in procedure maintenance loop for {name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all stored procedures."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all procedures
                for name, procedure in list(self.procedures.items()):
                    try:
                        # Get the procedure metrics
                        procedure_metrics = await procedure.get_metrics()
                        
                        # Update the metrics
                        if procedure_metrics:
                            self.procedure_metrics[name].update(procedure_metrics)
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
        for task in self.procedure_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
