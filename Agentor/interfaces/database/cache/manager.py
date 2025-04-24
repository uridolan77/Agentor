"""
Query cache manager for the Agentor framework.

This module provides a manager for database query caching.
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import QueryCacheConfig, CacheStrategy

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Cache type
R = TypeVar('R')  # Result type


class QueryCacheManager(Generic[T, R]):
    """Manager for database query caching."""
    
    def __init__(self, cache_class: Type[T], config: QueryCacheConfig):
        """Initialize the query cache manager.
        
        Args:
            cache_class: The query cache class
            config: The query cache configuration
        """
        self.cache_class = cache_class
        self.config = config
        self.caches: Dict[str, T] = {}
        self.cache_tasks: Dict[str, asyncio.Task] = {}
        self.cache_metrics: Dict[str, Dict[str, Any]] = {}
        self.cache_locks: Dict[str, asyncio.Lock] = {}
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
        
        # Set up the metrics collection task
        if self.config.collect_metrics and self.config.metrics_interval > 0:
            self.metrics_task = self.loop.create_task(self._metrics_collection_loop())
        else:
            self.metrics_task = None
    
    async def get_cache(self, name: str, **kwargs) -> T:
        """Get a query cache.
        
        Args:
            name: The name of the cache
            **kwargs: Additional arguments for the cache
            
        Returns:
            The query cache
        """
        # Check if the cache exists
        if name in self.caches:
            return self.caches[name]
        
        # Create a lock for the cache
        if name not in self.cache_locks:
            self.cache_locks[name] = asyncio.Lock()
        
        # Acquire the lock
        async with self.cache_locks[name]:
            # Check again in case another task created the cache
            if name in self.caches:
                return self.caches[name]
            
            # Create the cache
            cache = self.cache_class(name=name, config=self.config, **kwargs)
            
            # Initialize the cache
            await cache.initialize()
            
            # Store the cache
            self.caches[name] = cache
            
            # Initialize metrics
            self.cache_metrics[name] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "total_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_invalidations": 0,
                "cache_evictions": 0,
                "cache_size": 0,
                "cache_memory_usage": 0,
                "total_query_time": 0.0,
                "total_cache_time": 0.0
            }
            
            # Create a task for the cache
            self.cache_tasks[name] = self.loop.create_task(self._cache_maintenance_loop(name))
            
            return cache
    
    async def close_cache(self, name: str) -> None:
        """Close a query cache.
        
        Args:
            name: The name of the cache
        """
        # Check if the cache exists
        if name not in self.caches:
            return
        
        # Acquire the lock
        async with self.cache_locks[name]:
            # Get the cache
            cache = self.caches[name]
            
            # Close the cache
            await cache.close()
            
            # Cancel the cache task
            if name in self.cache_tasks:
                self.cache_tasks[name].cancel()
                try:
                    await self.cache_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.cache_tasks[name]
            
            # Remove the cache
            del self.caches[name]
            
            # Remove the metrics
            if name in self.cache_metrics:
                del self.cache_metrics[name]
    
    async def close_all_caches(self) -> None:
        """Close all query caches."""
        # Get all cache names
        cache_names = list(self.caches.keys())
        
        # Close each cache
        for name in cache_names:
            await self.close_cache(name)
        
        # Cancel the metrics task
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
            self.metrics_task = None
    
    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get query cache metrics.
        
        Args:
            name: The name of the cache, or None for all caches
            
        Returns:
            Dictionary of metrics
        """
        if name:
            # Check if the cache exists
            if name not in self.cache_metrics:
                return {}
            
            # Get the cache metrics
            return self.cache_metrics[name]
        else:
            # Get all cache metrics
            return self.cache_metrics
    
    async def _cache_maintenance_loop(self, name: str) -> None:
        """Maintenance loop for a query cache.
        
        Args:
            name: The name of the cache
        """
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Check if the cache exists
                if name not in self.caches:
                    break
                
                # Get the cache
                cache = self.caches[name]
                
                try:
                    # Update metrics
                    cache_metrics = await cache.get_metrics()
                    if cache_metrics:
                        self.cache_metrics[name].update(cache_metrics)
                except Exception as e:
                    logger.error(f"Error in cache maintenance loop for {name}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in cache maintenance loop for {name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Metrics collection loop for all query caches."""
        try:
            while True:
                # Sleep for the metrics interval
                await asyncio.sleep(self.config.metrics_interval)
                
                # Collect metrics for all caches
                for name, cache in list(self.caches.items()):
                    try:
                        # Get the cache metrics
                        cache_metrics = await cache.get_metrics()
                        
                        # Update the metrics
                        if cache_metrics:
                            self.cache_metrics[name].update(cache_metrics)
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
        for task in self.cache_tasks.values():
            if not task.done():
                task.cancel()
        
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
    
    @staticmethod
    def generate_cache_key(query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a cache key for a query.
        
        Args:
            query: The query
            params: The query parameters
            
        Returns:
            The cache key
        """
        # Create a string representation of the query and parameters
        key_str = query
        if params:
            key_str += json.dumps(params, sort_keys=True)
        
        # Generate a hash of the string
        return hashlib.md5(key_str.encode()).hexdigest()
