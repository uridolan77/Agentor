"""
MySQL query cache for the Agentor framework.

This module provides a specialized query cache for MySQL databases.
"""

import asyncio
import logging
import time
import json
import sys
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import QueryCacheConfig, CacheStrategy

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class CacheEntry:
    """Cache entry for a query result."""
    
    def __init__(self, key: str, result: Any, ttl: float):
        """Initialize the cache entry.
        
        Args:
            key: The cache key
            result: The query result
            ttl: The TTL in seconds
        """
        self.key = key
        self.result = result
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl
        self.last_accessed_at = self.created_at
        self.access_count = 0
        
        # Calculate the memory usage
        self.memory_usage = sys.getsizeof(key) + sys.getsizeof(result)
        if isinstance(result, (list, tuple)):
            for item in result:
                self.memory_usage += sys.getsizeof(item)
                if isinstance(item, dict):
                    for k, v in item.items():
                        self.memory_usage += sys.getsizeof(k) + sys.getsizeof(v)
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired.
        
        Returns:
            True if the cache entry is expired, False otherwise
        """
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """Access the cache entry."""
        self.last_accessed_at = time.time()
        self.access_count += 1


class MySqlQueryCache:
    """MySQL query cache with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: QueryCacheConfig
    ):
        """Initialize the MySQL query cache.
        
        Args:
            name: The name of the cache
            config: The query cache configuration
        """
        self.name = name
        self.config = config
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.table_keys: Dict[str, Set[str]] = {}
        
        # Cache lock
        self.cache_lock = asyncio.Lock()
        
        # Cache metrics
        self.metrics = {
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
    
    async def initialize(self) -> None:
        """Initialize the query cache."""
        logger.info(f"Initialized MySQL query cache {self.name}")
    
    async def close(self) -> None:
        """Close the query cache."""
        # Clear the cache
        async with self.cache_lock:
            self.cache.clear()
            self.table_keys.clear()
        
        logger.info(f"Closed MySQL query cache {self.name}")
    
    async def get(self, key: str) -> Tuple[bool, Any]:
        """Get a result from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            Tuple of (hit, result)
        """
        # Update metrics
        self.metrics["total_queries"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Check if the key exists in the cache
        async with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if the entry is expired
                if entry.is_expired():
                    # Remove the expired entry
                    del self.cache[key]
                    
                    # Update metrics
                    self.metrics["cache_evictions"] += 1
                    self.metrics["cache_size"] = len(self.cache)
                    self.metrics["cache_memory_usage"] -= entry.memory_usage
                    
                    # Return cache miss
                    self.metrics["cache_misses"] += 1
                    return False, None
                
                # Update the entry
                entry.access()
                
                # Update metrics
                self.metrics["cache_hits"] += 1
                
                # Return cache hit
                return True, entry.result
        
        # Key not found in cache
        self.metrics["cache_misses"] += 1
        return False, None
    
    async def set(self, key: str, result: Any, query: str, ttl: Optional[float] = None) -> None:
        """Set a result in the cache.
        
        Args:
            key: The cache key
            result: The query result
            query: The original query
            ttl: The TTL in seconds, or None to use the default
        """
        # Check if the query should be cached
        if not self.config.should_cache_query(query):
            return
        
        # Get the TTL
        if ttl is None:
            ttl = self.config.get_cache_ttl(query)
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Create a new cache entry
        entry = CacheEntry(key, result, ttl)
        
        # Add the entry to the cache
        async with self.cache_lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.config.max_size or self.metrics["cache_memory_usage"] + entry.memory_usage > self.config.max_memory:
                # Evict entries
                await self._evict_entries(entry.memory_usage)
            
            # Add the entry to the cache
            self.cache[key] = entry
            
            # Update metrics
            self.metrics["cache_size"] = len(self.cache)
            self.metrics["cache_memory_usage"] += entry.memory_usage
            
            # Extract tables from the query and associate the key with them
            tables = self._extract_tables_from_query(query)
            for table in tables:
                if table not in self.table_keys:
                    self.table_keys[table] = set()
                
                self.table_keys[table].add(key)
    
    async def invalidate(self, key: str) -> None:
        """Invalidate a cache entry.
        
        Args:
            key: The cache key
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Remove the entry from the cache
        async with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Remove the entry
                del self.cache[key]
                
                # Update metrics
                self.metrics["cache_invalidations"] += 1
                self.metrics["cache_size"] = len(self.cache)
                self.metrics["cache_memory_usage"] -= entry.memory_usage
    
    async def invalidate_query(self, query: str) -> None:
        """Invalidate cache entries based on a query.
        
        Args:
            query: The query
        """
        # Check if the cache should be invalidated
        if not self.config.should_invalidate_on_query(query):
            return
        
        # Get the tables to invalidate
        tables = self.config.get_tables_to_invalidate(query)
        
        # Invalidate the tables
        await self.invalidate_tables(tables)
    
    async def invalidate_tables(self, tables: List[str]) -> None:
        """Invalidate cache entries for specific tables.
        
        Args:
            tables: The tables to invalidate
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Get the keys to invalidate
        keys_to_invalidate = set()
        
        async with self.cache_lock:
            for table in tables:
                if table in self.table_keys:
                    keys_to_invalidate.update(self.table_keys[table])
                    
                    # Clear the table keys
                    self.table_keys[table].clear()
            
            # Invalidate the keys
            for key in keys_to_invalidate:
                if key in self.cache:
                    entry = self.cache[key]
                    
                    # Remove the entry
                    del self.cache[key]
                    
                    # Update metrics
                    self.metrics["cache_invalidations"] += 1
                    self.metrics["cache_size"] = len(self.cache)
                    self.metrics["cache_memory_usage"] -= entry.memory_usage
    
    async def invalidate_all(self) -> None:
        """Invalidate all cache entries."""
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Clear the cache
        async with self.cache_lock:
            # Update metrics
            self.metrics["cache_invalidations"] += len(self.cache)
            self.metrics["cache_size"] = 0
            self.metrics["cache_memory_usage"] = 0
            
            # Clear the cache
            self.cache.clear()
            self.table_keys.clear()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _evict_entries(self, required_space: int) -> None:
        """Evict entries from the cache to make space.
        
        Args:
            required_space: The required space in bytes
        """
        # Check if we need to evict entries
        if len(self.cache) == 0:
            return
        
        # Get the entries sorted by last accessed time
        entries = sorted(
            self.cache.values(),
            key=lambda e: e.last_accessed_at
        )
        
        # Evict entries until we have enough space
        evicted_count = 0
        evicted_memory = 0
        
        for entry in entries:
            # Remove the entry from the cache
            del self.cache[entry.key]
            
            # Update metrics
            evicted_count += 1
            evicted_memory += entry.memory_usage
            
            # Check if we have enough space
            if (len(self.cache) < self.config.max_size and
                self.metrics["cache_memory_usage"] - evicted_memory + required_space <= self.config.max_memory):
                break
        
        # Update metrics
        self.metrics["cache_evictions"] += evicted_count
        self.metrics["cache_size"] = len(self.cache)
        self.metrics["cache_memory_usage"] -= evicted_memory
    
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from a query.
        
        Args:
            query: The query
            
        Returns:
            List of table names
        """
        # Get the custom extraction function
        custom_func = self.config.additional_settings.get("extract_tables_func")
        if custom_func:
            return custom_func(query)
        
        # Simple regex to extract table names from SELECT queries
        # This is not a complete SQL parser, but it works for simple cases
        import re
        
        # Convert to uppercase for case-insensitive matching
        query = query.upper()
        
        # Extract table names from FROM and JOIN clauses
        tables = []
        
        # Extract from FROM clause
        from_match = re.search(r"FROM\s+`?(\w+)`?", query)
        if from_match:
            tables.append(from_match.group(1).lower())
        
        # Extract from JOIN clauses
        join_matches = re.finditer(r"JOIN\s+`?(\w+)`?", query)
        for match in join_matches:
            tables.append(match.group(1).lower())
        
        return tables
