import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Cache value type


@dataclass
class CacheEntry(Generic[T]):
    """An entry in the cache."""
    value: T
    expires_at: float  # Timestamp when this entry expires
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry is expired."""
        return time.time() > self.expires_at


class CacheBackend(ABC, Generic[T]):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry, or None if not found
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry[T]) -> bool:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            entry: The cache entry
            
        Returns:
            True if the value was set, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
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


class InMemoryCache(CacheBackend[T]):
    """In-memory cache backend."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the in-memory cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.cache: Dict[str, CacheEntry[T]] = {}
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry, or None if not found
        """
        async with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                return None
            
            if entry.is_expired:
                del self.cache[key]
                return None
            
            return entry
    
    async def set(self, key: str, entry: CacheEntry[T]) -> bool:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            entry: The cache entry
            
        Returns:
            True if the value was set, False otherwise
        """
        async with self.lock:
            # If the cache is full, remove the oldest entry
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].expires_at)
                del self.cache[oldest_key]
            
            self.cache[key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
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


class RedisCache(CacheBackend[T]):
    """Redis cache backend."""
    
    def __init__(self, redis_client, namespace: str = "cache"):
        """Initialize the Redis cache.
        
        Args:
            redis_client: The Redis client
            namespace: The namespace for cache keys
        """
        self.redis = redis_client
        self.namespace = namespace
    
    def _make_key(self, key: str) -> str:
        """Make a Redis key with the namespace.
        
        Args:
            key: The cache key
            
        Returns:
            The Redis key
        """
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cache entry, or None if not found
        """
        redis_key = self._make_key(key)
        value = await self.redis.get(redis_key)
        
        if value is None:
            return None
        
        try:
            data = json.loads(value)
            entry = CacheEntry(
                value=data["value"],
                expires_at=data["expires_at"],
                metadata=data.get("metadata")
            )
            
            if entry.is_expired:
                await self.delete(key)
                return None
            
            return entry
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error decoding cache entry: {e}")
            return None
    
    async def set(self, key: str, entry: CacheEntry[T]) -> bool:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            entry: The cache entry
            
        Returns:
            True if the value was set, False otherwise
        """
        redis_key = self._make_key(key)
        ttl = max(0, int(entry.expires_at - time.time()))
        
        try:
            data = {
                "value": entry.value,
                "expires_at": entry.expires_at,
                "metadata": entry.metadata
            }
            
            await self.redis.setex(redis_key, ttl, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        redis_key = self._make_key(key)
        result = await self.redis.delete(redis_key)
        return result > 0
    
    async def clear(self) -> bool:
        """Clear the cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        try:
            # Get all keys in the namespace
            pattern = f"{self.namespace}:*"
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


class DistributedCache(Generic[T]):
    """Distributed cache with multiple backends."""
    
    def __init__(
        self,
        primary_backend: CacheBackend[T],
        secondary_backend: Optional[CacheBackend[T]] = None,
        key_prefix: str = "",
        default_ttl: int = 3600
    ):
        """Initialize the distributed cache.
        
        Args:
            primary_backend: The primary cache backend
            secondary_backend: The secondary cache backend (optional)
            key_prefix: Prefix for cache keys
            default_ttl: Default TTL for cache entries in seconds
        """
        self.primary = primary_backend
        self.secondary = secondary_backend
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
    
    def _make_key(self, key: Union[str, Dict[str, Any]]) -> str:
        """Make a cache key.
        
        Args:
            key: The cache key or a dictionary to hash
            
        Returns:
            The cache key
        """
        if isinstance(key, dict):
            # Sort the dictionary to ensure consistent hashing
            serialized = json.dumps(key, sort_keys=True)
            hashed = hashlib.md5(serialized.encode()).hexdigest()
            return f"{self.key_prefix}{hashed}"
        else:
            return f"{self.key_prefix}{key}"
    
    async def get(self, key: Union[str, Dict[str, Any]]) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: The cache key or a dictionary to hash
            
        Returns:
            The cached value, or None if not found
        """
        cache_key = self._make_key(key)
        
        # Try the primary backend first
        entry = await self.primary.get(cache_key)
        
        if entry is not None:
            return entry.value
        
        # If not found and we have a secondary backend, try that
        if self.secondary is not None:
            entry = await self.secondary.get(cache_key)
            
            if entry is not None:
                # Copy to the primary backend
                await self.primary.set(cache_key, entry)
                return entry.value
        
        return None
    
    async def set(
        self,
        key: Union[str, Dict[str, Any]],
        value: T,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set a value in the cache.
        
        Args:
            key: The cache key or a dictionary to hash
            value: The value to cache
            ttl: Time-to-live in seconds (optional, uses default if not provided)
            metadata: Additional metadata for the cache entry
            
        Returns:
            True if the value was set in at least one backend, False otherwise
        """
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        entry = CacheEntry(value=value, expires_at=expires_at, metadata=metadata)
        
        # Set in the primary backend
        primary_result = await self.primary.set(cache_key, entry)
        
        # Set in the secondary backend if available
        secondary_result = True
        if self.secondary is not None:
            secondary_result = await self.secondary.set(cache_key, entry)
        
        return primary_result or secondary_result
    
    async def delete(self, key: Union[str, Dict[str, Any]]) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key or a dictionary to hash
            
        Returns:
            True if the value was deleted from at least one backend, False otherwise
        """
        cache_key = self._make_key(key)
        
        # Delete from the primary backend
        primary_result = await self.primary.delete(cache_key)
        
        # Delete from the secondary backend if available
        secondary_result = True
        if self.secondary is not None:
            secondary_result = await self.secondary.delete(cache_key)
        
        return primary_result or secondary_result
    
    async def clear(self) -> bool:
        """Clear the cache.
        
        Returns:
            True if the cache was cleared from at least one backend, False otherwise
        """
        # Clear the primary backend
        primary_result = await self.primary.clear()
        
        # Clear the secondary backend if available
        secondary_result = True
        if self.secondary is not None:
            secondary_result = await self.secondary.clear()
        
        return primary_result or secondary_result
    
    async def get_or_set(
        self,
        key: Union[str, Dict[str, Any]],
        value_func: Callable[[], T],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> T:
        """Get a value from the cache, or set it if not found.
        
        Args:
            key: The cache key or a dictionary to hash
            value_func: Function to call to get the value if not in cache
            ttl: Time-to-live in seconds (optional, uses default if not provided)
            metadata: Additional metadata for the cache entry
            
        Returns:
            The cached value
        """
        # Try to get from cache first
        cached_value = await self.get(key)
        
        if cached_value is not None:
            return cached_value
        
        # Not in cache, call the value function
        value = value_func()
        
        # Set in cache
        await self.set(key, value, ttl, metadata)
        
        return value
    
    async def get_or_set_async(
        self,
        key: Union[str, Dict[str, Any]],
        value_func: Callable[[], Awaitable[T]],
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> T:
        """Get a value from the cache, or set it if not found (async version).
        
        Args:
            key: The cache key or a dictionary to hash
            value_func: Async function to call to get the value if not in cache
            ttl: Time-to-live in seconds (optional, uses default if not provided)
            metadata: Additional metadata for the cache entry
            
        Returns:
            The cached value
        """
        # Try to get from cache first
        cached_value = await self.get(key)
        
        if cached_value is not None:
            return cached_value
        
        # Not in cache, call the value function
        value = await value_func()
        
        # Set in cache
        await self.set(key, value, ttl, metadata)
        
        return value
