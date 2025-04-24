"""
Enhanced caching utilities for the Agentor framework.

This module provides a comprehensive caching system with various backends
and strategies for optimizing performance of frequently used operations.
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Awaitable, Union, Tuple
import time
import logging
import asyncio
import functools
import inspect
import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Cache value type
K = TypeVar('K')  # Cache key type


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with metadata."""
    
    value: T
    expiry: float  # Expiration timestamp
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry is expired.
        
        Returns:
            True if the entry is expired
        """
        return time.time() > self.expiry
    
    def access(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackend(Generic[K, T], ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: K) -> Optional[CacheEntry[T]]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry, or None if not found
        """
        pass
    
    @abstractmethod
    async def set(self, key: K, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
        """
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear the cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        pass


class InMemoryCache(CacheBackend[K, T]):
    """In-memory cache backend with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU
    ):
        """Initialize the in-memory cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            default_ttl: Default time-to-live in seconds
            strategy: Cache eviction strategy
        """
        self.cache: Dict[K, CacheEntry[T]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.lock = asyncio.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.sets = 0
        self.deletes = 0
    
    async def get(self, key: K) -> Optional[CacheEntry[T]]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry, or None if not found
        """
        async with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired:
                del self.cache[key]
                self.evictions += 1
                self.misses += 1
                return None
            
            # Update access metadata
            entry.access()
            self.hits += 1
            
            return entry
    
    async def set(self, key: K, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        async with self.lock:
            # Check if we need to evict an entry
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()
            
            # Add or update the entry
            self.cache[key] = CacheEntry(value=value, expiry=expiry)
            self.sets += 1
    
    async def delete(self, key: K) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.deletes += 1
                return True
            
            return False
    
    async def clear(self) -> bool:
        """Clear the cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        async with self.lock:
            self.cache.clear()
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        async with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "evictions": self.evictions,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }
    
    async def _evict(self) -> None:
        """Evict an entry based on the selected strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict the least recently used entry
            key_to_evict = min(self.cache.items(), key=lambda x: x[1].last_accessed)[0]
        elif self.strategy == CacheStrategy.LFU:
            # Evict the least frequently used entry
            key_to_evict = min(self.cache.items(), key=lambda x: x[1].access_count)[0]
        elif self.strategy == CacheStrategy.FIFO:
            # Evict the oldest entry
            key_to_evict = min(self.cache.items(), key=lambda x: x[1].created_at)[0]
        else:  # TTL
            # Evict the entry closest to expiration
            key_to_evict = min(self.cache.items(), key=lambda x: x[1].expiry)[0]
        
        del self.cache[key_to_evict]
        self.evictions += 1


class RedisCache(CacheBackend[str, T]):
    """Redis cache backend."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: float = 3600,
        prefix: str = "agentor:"
    ):
        """Initialize the Redis cache.
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            prefix: Prefix for cache keys
        """
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
            self.default_ttl = default_ttl
            self.prefix = prefix
            self.lock = asyncio.Lock()
            
            # Statistics
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.deletes = 0
            
            logger.info(f"Initialized Redis cache with URL: {redis_url}")
        except ImportError:
            logger.error("Redis package not installed. Please install with: pip install redis")
            raise
    
    def _get_key(self, key: str) -> str:
        """Get the full Redis key.
        
        Args:
            key: The cache key
            
        Returns:
            The full Redis key
        """
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry, or None if not found
        """
        redis_key = self._get_key(key)
        
        try:
            # Get the value and metadata
            value_bytes = await self.redis.get(redis_key)
            metadata_bytes = await self.redis.get(f"{redis_key}:meta")
            
            if value_bytes is None or metadata_bytes is None:
                async with self.lock:
                    self.misses += 1
                return None
            
            # Deserialize the value and metadata
            value = pickle.loads(value_bytes)
            metadata = json.loads(metadata_bytes)
            
            # Create the cache entry
            entry = CacheEntry(
                value=value,
                expiry=metadata["expiry"],
                created_at=metadata["created_at"],
                last_accessed=metadata["last_accessed"],
                access_count=metadata["access_count"]
            )
            
            # Check if expired
            if entry.is_expired:
                await self.delete(key)
                async with self.lock:
                    self.misses += 1
                return None
            
            # Update access metadata
            entry.access()
            await self.redis.set(
                f"{redis_key}:meta",
                json.dumps({
                    "expiry": entry.expiry,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count
                }),
                ex=int(entry.expiry - time.time())
            )
            
            async with self.lock:
                self.hits += 1
            
            return entry
        except Exception as e:
            logger.error(f"Error getting value from Redis: {str(e)}")
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        redis_key = self._get_key(key)
        
        try:
            # Serialize the value
            value_bytes = pickle.dumps(value)
            
            # Create metadata
            metadata = {
                "expiry": expiry,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 0
            }
            
            # Set the value and metadata
            await self.redis.set(redis_key, value_bytes, ex=int(ttl))
            await self.redis.set(f"{redis_key}:meta", json.dumps(metadata), ex=int(ttl))
            
            async with self.lock:
                self.sets += 1
        except Exception as e:
            logger.error(f"Error setting value in Redis: {str(e)}")
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        redis_key = self._get_key(key)
        
        try:
            # Delete the value and metadata
            result = await self.redis.delete(redis_key, f"{redis_key}:meta")
            
            if result > 0:
                async with self.lock:
                    self.deletes += 1
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting value from Redis: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear the cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        try:
            # Get all keys with the prefix
            keys = await self.redis.keys(f"{self.prefix}*")
            
            if keys:
                # Delete all keys
                await self.redis.delete(*keys)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        try:
            # Get the number of keys
            keys = await self.redis.keys(f"{self.prefix}*")
            size = len(keys) // 2  # Divide by 2 because we store value and metadata separately
            
            async with self.lock:
                return {
                    "size": size,
                    "hits": self.hits,
                    "misses": self.misses,
                    "sets": self.sets,
                    "deletes": self.deletes,
                    "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting Redis cache stats: {str(e)}")
            return {
                "error": str(e)
            }


class Cache(Generic[K, T]):
    """High-level cache interface with multiple backends."""
    
    def __init__(
        self,
        primary_backend: CacheBackend[K, T],
        secondary_backend: Optional[CacheBackend[K, T]] = None,
        default_ttl: float = 3600,
        key_serializer: Optional[Callable[[Any], K]] = None
    ):
        """Initialize the cache.
        
        Args:
            primary_backend: The primary cache backend
            secondary_backend: The secondary cache backend (optional)
            default_ttl: Default time-to-live in seconds
            key_serializer: Function to serialize complex keys
        """
        self.primary = primary_backend
        self.secondary = secondary_backend
        self.default_ttl = default_ttl
        self.key_serializer = key_serializer or self._default_key_serializer
    
    def _default_key_serializer(self, key: Any) -> Union[str, K]:
        """Default key serializer.
        
        Args:
            key: The key to serialize
            
        Returns:
            The serialized key
        """
        if isinstance(key, (str, int, float, bool)):
            return str(key)
        
        try:
            # Try to serialize as JSON
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
        except (TypeError, ValueError):
            # Fall back to string representation
            return hashlib.md5(str(key).encode()).hexdigest()
    
    async def get(self, key: Any) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found
        """
        serialized_key = self.key_serializer(key)
        
        # Try primary backend first
        entry = await self.primary.get(serialized_key)
        
        if entry is not None:
            return entry.value
        
        # Try secondary backend if available
        if self.secondary is not None:
            entry = await self.secondary.get(serialized_key)
            
            if entry is not None:
                # Copy to primary backend
                await self.primary.set(
                    serialized_key,
                    entry.value,
                    ttl=entry.expiry - time.time()
                )
                
                return entry.value
        
        return None
    
    async def set(self, key: Any, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
        """
        ttl = ttl or self.default_ttl
        serialized_key = self.key_serializer(key)
        
        # Set in primary backend
        await self.primary.set(serialized_key, value, ttl)
        
        # Set in secondary backend if available
        if self.secondary is not None:
            await self.secondary.set(serialized_key, value, ttl)
    
    async def delete(self, key: Any) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        serialized_key = self.key_serializer(key)
        
        # Delete from primary backend
        primary_result = await self.primary.delete(serialized_key)
        
        # Delete from secondary backend if available
        if self.secondary is not None:
            secondary_result = await self.secondary.delete(serialized_key)
            return primary_result or secondary_result
        
        return primary_result
    
    async def clear(self) -> bool:
        """Clear the cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        # Clear primary backend
        primary_result = await self.primary.clear()
        
        # Clear secondary backend if available
        if self.secondary is not None:
            secondary_result = await self.secondary.clear()
            return primary_result and secondary_result
        
        return primary_result
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        # Get stats from primary backend
        primary_stats = await self.primary.get_stats()
        
        # Get stats from secondary backend if available
        if self.secondary is not None:
            secondary_stats = await self.secondary.get_stats()
            return {
                "primary": primary_stats,
                "secondary": secondary_stats
            }
        
        return primary_stats


def cached(
    ttl: Optional[float] = None,
    key_builder: Optional[Callable[..., Any]] = None,
    cache_instance: Optional[Cache] = None
):
    """Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_builder: Function to build the cache key
        cache_instance: Cache instance to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        # Create a default cache if none provided
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = Cache(
                primary_backend=InMemoryCache(max_size=1000, default_ttl=3600)
            )
        
        # Create a default key builder if none provided
        nonlocal key_builder
        if key_builder is None:
            def default_key_builder(*args, **kwargs):
                # Get the function's module and name
                func_id = f"{func.__module__}.{func.__qualname__}"
                
                # Convert args and kwargs to a hashable representation
                args_str = str(args)
                kwargs_str = str(sorted(kwargs.items()))
                
                # Combine everything into a key
                return f"{func_id}:{args_str}:{kwargs_str}"
            
            key_builder = default_key_builder
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build the cache key
            key = key_builder(*args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache_instance.get(key)
            
            if cached_value is not None:
                return cached_value
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Handle coroutines
            if inspect.iscoroutine(result):
                result = await result
            
            # Cache the result
            await cache_instance.set(key, result, ttl)
            
            return result
        
        # Add cache control methods to the wrapper
        wrapper.cache = cache_instance
        wrapper.invalidate = lambda *args, **kwargs: cache_instance.delete(key_builder(*args, **kwargs))
        wrapper.invalidate_all = lambda: cache_instance.clear()
        
        return wrapper
    
    return decorator
