"""
Connection pool manager for the Agentor framework.

This module provides a manager for database connection pools.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import ConnectionPoolConfig, ConnectionValidationMode

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type
P = TypeVar('P')  # Pool type


class ConnectionPoolManager(Generic[T, P]):
    """Manager for database connection pools."""
    
    def __init__(self, pool_class: Type[P], config: ConnectionPoolConfig):
        """Initialize the connection pool manager.
        
        Args:
            pool_class: The connection pool class
            config: The connection pool configuration
        """
        self.pool_class = pool_class
        self.config = config
        self.pools: Dict[str, P] = {}
        self.pool_tasks: Dict[str, asyncio.Task] = {}
        self.pool_metrics: Dict[str, Dict[str, Any]] = {}
        self.pool_health: Dict[str, Dict[str, Any]] = {}
        self.pool_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the health check task
        if self.config.health_check_interval > 0:
            self.health_check_task = self.loop.create_task(self._health_check_loop())
        else:
            self.health_check_task = None
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_pool(self, name: str, **kwargs) -> P:
        """Get a connection pool.
        
        Args:
            name: The name of the pool
            **kwargs: Additional arguments for the pool
            
        Returns:
            The connection pool
        """
        # Check if the pool exists
        if name in self.pools:
            return self.pools[name]
        
        # Create a lock for the pool
        if name not in self.pool_locks:
            self.pool_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.pool_locks[name]:
            # Check again in case another task created the pool
            if name in self.pools:
                return self.pools[name]
            
            # Create the pool
            pool = self.pool_class(name=name, config=self.config, **kwargs)
            
            # Initialize the pool
            await pool.initialize()
            
            # Store the pool
            self.pools[name] = pool
            
            # Initialize metrics
            self.pool_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_connections": 0,
                "active_connections": 0,
                "idle_connections": 0,
                "waiting_requests": 0,
                "max_waiting_requests": 0,
                "total_wait_time": 0.0,
                "total_execution_time": 0.0,
                "total_operations": 0,
                "failed_operations": 0,
                "validation_failures": 0,
                "connection_timeouts": 0,
                "connection_errors": 0
            }
            
            # Initialize health
            self.pool_health[name] = {
                "status": "healthy",
                "last_check": time.time(),
                "last_failure": None,
                "failure_count": 0,
                "consecutive_failures": 0,
                "last_error": None
            }
            
            # Create a task for the pool
            self.pool_tasks[name] = self.loop.create_task(self._pool_maintenance_loop(name))
            
            return pool
    
    async def close_pool(self, name: str) -> None:
        """Close a connection pool.
        
        Args:
            name: The name of the pool
        """
        # Check if the pool exists
        if name not in self.pools:
            return
        
        # Acquire the lock
        async with self.pool_locks[name]:
            # Get the pool
            pool = self.pools[name]
            
            # Close the pool
            await pool.close()
            
            # Cancel the pool task
            if name in self.pool_tasks:
                self.pool_tasks[name].cancel()
                try:
                    await self.pool_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.pool_tasks[name]
            
            # Remove the pool
            del self.pools[name]
            
            # Remove the metrics
            if name in self.pool_metrics:
                del self.pool_metrics[name]
            
            # Remove the health
            if name in self.pool_health:
                del self.pool_health[name]
    
    async def close_all_pools(self) -> None:
        """Close all connection pools."""
        # Get all pool names
        pool_names = list(self.pools.keys())
        
        # Close each pool
        for name in pool_names:
            await self.close_pool(name)
        
        # Cancel the health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get connection pool metrics.
        
        Args:
            name: The name of the pool, or None for all pools
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the pool exists
            if name not in self.pool_metrics:
                return {}
            
            # Get the pool metrics
            return self.pool_metrics[name]
        else:
            # Get all pool metrics
            return self.pool_metrics
    
    async def get_health(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get connection pool health.
        
        Args:
            name: The name of the pool, or None for all pools
            
        Returns:
            Dictionary of health information
        """
        if name:
            # Check if the pool exists
            if name not in self.pool_health:
                return {}
            
            # Get the pool health
            return self.pool_health[name]
        else:
            # Get all pool health
            return self.pool_health
    
    async def _pool_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for a connection pool.
        
        Args:
            name: The name of the pool
        """
        try:
            while True:
                # Sleep for the validation interval
                await asyncio.sleep(self.config.validation_interval)
                
                # Check if the pool exists
                if name not in self.pools:
                    break
                
                # Get the pool
                pool = self.pools[name]
                
                try:
                    # Validate connections
                    await pool.validate_connections()
                    
                    # Scale the pool
                    await pool.scale_pool()
                    
                    # Update metrics
                    pool_metrics = await pool.get_metrics()
                    if pool_metrics:
                        self.pool_metrics[name].update(pool_metrics)
                except Exception as e:
                    logger.error(f"Error in pool maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in pool maintenance loop for {name}: {e}")
    
    async def _health_check_loop(self) -> None:
        """Health check loop for all connection pools."""
        try:
            while True:
                # Sleep for the health check interval
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check all pools
                for name, pool in list(self.pools.items()):
                    try:
                        # Check the pool health
                        healthy = await pool.check_health()
                        
                        # Update the health status
                        self.pool_health[name]["last_check"] = time.time()
                        
                        if healthy:
                            self.pool_health[name]["status"] = "healthy"
                            self.pool_health[name]["consecutive_failures"] = 0
                        else:
                            self.pool_health[name]["status"] = "unhealthy"
                            self.pool_health[name]["last_failure"] = time.time()
                            self.pool_health[name]["failure_count"] += 1
                            self.pool_health[name]["consecutive_failures"] += 1
                            
                            # Log the failure
                            logger.warning(f"Health check failed for pool {name}")
                            
                            # Try to recover the pool
                            try:
                                await pool.recover()
                            except Exception as e:
                                logger.error(f"Failed to recover pool {name}: {e}")
                                self.pool_health[name]["last_error"] = str(e)
                    except Exception as e:
                        logger.error(f"Error in health check for {name}: {e}")
                        self.pool_health[name]["status"] = "error"
                        self.pool_health[name]["last_failure"] = time.time()
                        self.pool_health[name]["failure_count"] += 1
                        self.pool_health[name]["consecutive_failures"] += 1
                        self.pool_health[name]["last_error"] = str(e)
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in health check loop: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all connection pools."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all pools
                for name, pool in list(self.pools.items()):
                    try:
                        # Get the pool metrics
                        pool_metrics = await pool.get_metrics()
                        
                        # Update the metrics
                        if pool_metrics:
                            self.pool_metrics[name].update(pool_metrics)
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
        for task in self.pool_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
