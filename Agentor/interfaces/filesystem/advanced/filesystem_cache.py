"""
Filesystem cache implementations for the Agentor framework.

This module provides caching implementations for filesystem operations.
"""

import os
import time
import json
import hashlib
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Tuple, Set
import logging
from pathlib import Path
import asyncio
import functools
from pydantic import BaseModel, Field

from ..base import FilesystemResult

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies for filesystem operations."""
    
    # No caching
    NONE = "none"
    
    # Cache read operations only
    READ_ONLY = "read_only"
    
    # Cache write operations only
    WRITE_ONLY = "write_only"
    
    # Cache both read and write operations
    READ_WRITE = "read_write"
    
    # Cache metadata operations only (exists, is_file, is_dir, etc.)
    METADATA_ONLY = "metadata_only"
    
    # Cache everything
    ALL = "all"


class CacheConfig(BaseModel):
    """Configuration for filesystem cache."""
    
    # Cache strategy
    strategy: CacheStrategy = Field(
        CacheStrategy.READ_ONLY,
        description="Cache strategy to use"
    )
    
    # Cache TTL in seconds (0 = no expiration)
    ttl: int = Field(
        300,
        description="Cache TTL in seconds (0 = no expiration)"
    )
    
    # Maximum cache size in bytes (0 = no limit)
    max_size: int = Field(
        1024 * 1024 * 100,  # 100 MB
        description="Maximum cache size in bytes (0 = no limit)"
    )
    
    # Maximum number of items in cache (0 = no limit)
    max_items: int = Field(
        10000,
        description="Maximum number of items in cache (0 = no limit)"
    )
    
    # Cache directory for disk cache
    cache_dir: Optional[str] = Field(
        None,
        description="Cache directory for disk cache"
    )
    
    # Whether to validate cache entries on read
    validate_on_read: bool = Field(
        True,
        description="Whether to validate cache entries on read"
    )
    
    # Whether to use cache for operations that might fail
    cache_failures: bool = Field(
        False,
        description="Whether to use cache for operations that might fail"
    )
    
    # List of paths to exclude from caching (glob patterns)
    exclude_paths: List[str] = Field(
        [],
        description="List of paths to exclude from caching (glob patterns)"
    )
    
    # List of operations to exclude from caching
    exclude_operations: List[str] = Field(
        [],
        description="List of operations to exclude from caching"
    )


class CacheEntry:
    """Cache entry for filesystem operations."""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: int = 0,
        created_at: float = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize a cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time-to-live in seconds (0 = no expiration)
            created_at: Creation timestamp (defaults to current time)
            metadata: Additional metadata
        """
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}
    
    @property
    def expires_at(self) -> Optional[float]:
        """Get the expiration timestamp.
        
        Returns:
            Expiration timestamp, or None if the entry doesn't expire
        """
        if self.ttl <= 0:
            return None
        return self.created_at + self.ttl
    
    @property
    def is_expired(self) -> bool:
        """Check if the entry is expired.
        
        Returns:
            True if the entry is expired, False otherwise
        """
        expires_at = self.expires_at
        if expires_at is None:
            return False
        return time.time() > expires_at
    
    @property
    def age(self) -> float:
        """Get the age of the entry in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.created_at
    
    @property
    def size(self) -> int:
        """Get the approximate size of the entry in bytes.
        
        Returns:
            Size in bytes
        """
        try:
            return len(pickle.dumps(self.value))
        except Exception:
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a dictionary.
        
        Returns:
            Dictionary representation of the entry
        """
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create a cache entry from a dictionary.
        
        Args:
            data: Dictionary representation of the entry
            
        Returns:
            Cache entry
        """
        return cls(
            key=data["key"],
            value=data["value"],
            ttl=data["ttl"],
            created_at=data["created_at"],
            metadata=data["metadata"]
        )


class FilesystemCache(ABC):
    """Interface for filesystem cache implementations."""
    
    def __init__(self, config: CacheConfig = None):
        """Initialize the cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides config.ttl if provided)
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear the cache."""
        pass
    
    @abstractmethod
    async def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_size(self) -> int:
        """Get the current size of the cache in bytes.
        
        Returns:
            Size in bytes
        """
        pass
    
    @abstractmethod
    async def get_count(self) -> int:
        """Get the current number of items in the cache.
        
        Returns:
            Number of items
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        pass
    
    def generate_key(self, operation: str, path: str, *args, **kwargs) -> str:
        """Generate a cache key for a filesystem operation.
        
        Args:
            operation: Operation name
            path: File path
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Cache key
        """
        # Create a string representation of the operation and arguments
        key_parts = [operation, path]
        
        # Add args and kwargs
        if args:
            key_parts.append(str(args))
        if kwargs:
            # Sort kwargs by key for consistent ordering
            sorted_kwargs = sorted(kwargs.items())
            key_parts.append(str(sorted_kwargs))
        
        # Join parts and hash
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def should_cache(self, operation: str, path: str) -> bool:
        """Check if an operation should be cached.
        
        Args:
            operation: Operation name
            path: File path
            
        Returns:
            True if the operation should be cached, False otherwise
        """
        # Check if caching is disabled
        if self.config.strategy == CacheStrategy.NONE:
            return False
        
        # Check if the operation is excluded
        if operation in self.config.exclude_operations:
            return False
        
        # Check if the path is excluded
        for exclude_pattern in self.config.exclude_paths:
            if Path(path).match(exclude_pattern):
                return False
        
        # Check if the operation matches the cache strategy
        if self.config.strategy == CacheStrategy.READ_ONLY:
            return operation.startswith("read_") or operation in ["exists", "is_file", "is_dir", "get_info", "list_dir", "get_size", "get_modified_time", "walk"]
        elif self.config.strategy == CacheStrategy.WRITE_ONLY:
            return operation.startswith("write_") or operation.startswith("append_") or operation in ["create_dir", "delete_file", "delete_dir", "copy", "move", "set_modified_time"]
        elif self.config.strategy == CacheStrategy.METADATA_ONLY:
            return operation in ["exists", "is_file", "is_dir", "get_info", "list_dir", "get_size", "get_modified_time", "walk"]
        elif self.config.strategy == CacheStrategy.READ_WRITE:
            return operation.startswith("read_") or operation.startswith("write_") or operation.startswith("append_")
        elif self.config.strategy == CacheStrategy.ALL:
            return True
        
        return False
    
    def should_invalidate(self, operation: str, path: str) -> Tuple[bool, Set[str]]:
        """Check if an operation should invalidate cache entries.
        
        Args:
            operation: Operation name
            path: File path
            
        Returns:
            Tuple of (should_invalidate, paths_to_invalidate)
        """
        # Write operations should invalidate cache entries for the same path
        if operation.startswith("write_") or operation.startswith("append_") or operation in ["delete_file", "set_modified_time"]:
            return True, {path}
        
        # Directory operations should invalidate cache entries for the directory and its contents
        if operation in ["create_dir", "delete_dir"]:
            return True, {path, f"{path}/*"}
        
        # Copy and move operations should invalidate cache entries for both source and destination
        if operation in ["copy", "move"]:
            # We don't have the destination path here, so we'll need to handle this separately
            return True, {path}
        
        return False, set()
    
    async def invalidate(self, paths: Set[str]) -> None:
        """Invalidate cache entries for the given paths.
        
        Args:
            paths: Set of paths to invalidate
        """
        # This is a default implementation that iterates through all keys
        # Subclasses should override this with more efficient implementations
        async with self._lock:
            keys_to_delete = []
            
            # Get all keys
            all_keys = await self._get_all_keys()
            
            # Find keys to delete
            for key in all_keys:
                for path in paths:
                    if path in key:
                        keys_to_delete.append(key)
                        break
            
            # Delete keys
            for key in keys_to_delete:
                await self.delete(key)
    
    async def _get_all_keys(self) -> List[str]:
        """Get all keys in the cache.
        
        Returns:
            List of keys
        """
        # This is a default implementation that should be overridden by subclasses
        return []


class MemoryCache(FilesystemCache):
    """In-memory cache implementation."""
    
    def __init__(self, config: CacheConfig = None):
        """Initialize the cache.
        
        Args:
            config: Cache configuration
        """
        super().__init__(config)
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found
        """
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            # Check if the entry is expired
            if entry.is_expired:
                await self.delete(key)
                self._misses += 1
                return None
            
            self._hits += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides config.ttl if provided)
        """
        async with self._lock:
            # Create a new cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl if ttl is not None else self.config.ttl
            )
            
            # Check if we need to evict entries
            await self._evict_if_needed(entry.size)
            
            # Add the entry to the cache
            self._cache[key] = entry
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False otherwise
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    async def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        async with self._lock:
            if key not in self._cache:
                return False
            
            # Check if the entry is expired
            entry = self._cache[key]
            if entry.is_expired:
                await self.delete(key)
                return False
            
            return True
    
    async def get_size(self) -> int:
        """Get the current size of the cache in bytes.
        
        Returns:
            Size in bytes
        """
        async with self._lock:
            return sum(entry.size for entry in self._cache.values())
    
    async def get_count(self) -> int:
        """Get the current number of items in the cache.
        
        Returns:
            Number of items
        """
        async with self._lock:
            return len(self._cache)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        async with self._lock:
            return {
                "type": "memory",
                "size": await self.get_size(),
                "count": await self.get_count(),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                "evictions": self._evictions
            }
    
    async def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict entries if needed to make room for a new entry.
        
        Args:
            new_entry_size: Size of the new entry in bytes
        """
        # Check if we need to evict based on max_items
        if self.config.max_items > 0 and len(self._cache) >= self.config.max_items:
            # Evict the oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            await self.delete(oldest_key)
            self._evictions += 1
        
        # Check if we need to evict based on max_size
        if self.config.max_size > 0:
            current_size = await self.get_size()
            while current_size + new_entry_size > self.config.max_size and self._cache:
                # Evict the oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
                evicted_size = self._cache[oldest_key].size
                await self.delete(oldest_key)
                self._evictions += 1
                current_size -= evicted_size
    
    async def _get_all_keys(self) -> List[str]:
        """Get all keys in the cache.
        
        Returns:
            List of keys
        """
        async with self._lock:
            return list(self._cache.keys())


class DiskCache(FilesystemCache):
    """Disk-based cache implementation."""
    
    def __init__(self, config: CacheConfig = None):
        """Initialize the cache.
        
        Args:
            config: Cache configuration
        """
        super().__init__(config)
        
        # Ensure cache directory exists
        if not self.config.cache_dir:
            self.config.cache_dir = os.path.join(os.path.expanduser("~"), ".agentor", "cache")
        
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Create index file path
        self._index_path = os.path.join(self.config.cache_dir, "index.json")
        
        # Initialize stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Initialize index
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the cache index from disk."""
        try:
            if os.path.exists(self._index_path):
                with open(self._index_path, "r") as f:
                    self._index = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            self._index = {}
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self._index_path, "w") as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _get_entry_path(self, key: str) -> str:
        """Get the path to a cache entry file.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache entry file
        """
        return os.path.join(self.config.cache_dir, f"{key}.pickle")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found
        """
        async with self._lock:
            # Check if the key exists in the index
            if key not in self._index:
                self._misses += 1
                return None
            
            # Get entry metadata from the index
            entry_meta = self._index[key]
            
            # Check if the entry is expired
            if entry_meta.get("ttl", 0) > 0:
                created_at = entry_meta.get("created_at", 0)
                expires_at = created_at + entry_meta.get("ttl", 0)
                if time.time() > expires_at:
                    await self.delete(key)
                    self._misses += 1
                    return None
            
            # Get the entry file path
            entry_path = self._get_entry_path(key)
            
            # Check if the entry file exists
            if not os.path.exists(entry_path):
                # Entry file is missing, remove from index
                del self._index[key]
                self._save_index()
                self._misses += 1
                return None
            
            # Load the entry value
            try:
                with open(entry_path, "rb") as f:
                    value = pickle.load(f)
                
                self._hits += 1
                return value
            except Exception as e:
                logger.warning(f"Failed to load cache entry: {e}")
                await self.delete(key)
                self._misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides config.ttl if provided)
        """
        async with self._lock:
            # Create entry metadata
            entry_meta = {
                "key": key,
                "created_at": time.time(),
                "ttl": ttl if ttl is not None else self.config.ttl
            }
            
            # Save the entry value
            entry_path = self._get_entry_path(key)
            
            try:
                # Serialize the value
                serialized = pickle.dumps(value)
                entry_meta["size"] = len(serialized)
                
                # Check if we need to evict entries
                await self._evict_if_needed(entry_meta["size"])
                
                # Save the value to disk
                with open(entry_path, "wb") as f:
                    f.write(serialized)
                
                # Update the index
                self._index[key] = entry_meta
                self._save_index()
            except Exception as e:
                logger.warning(f"Failed to save cache entry: {e}")
                # Clean up if the file was created
                if os.path.exists(entry_path):
                    os.remove(entry_path)
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False otherwise
        """
        async with self._lock:
            # Check if the key exists in the index
            if key not in self._index:
                return False
            
            # Get the entry file path
            entry_path = self._get_entry_path(key)
            
            # Delete the entry file
            if os.path.exists(entry_path):
                try:
                    os.remove(entry_path)
                except Exception as e:
                    logger.warning(f"Failed to delete cache entry file: {e}")
            
            # Remove from index
            del self._index[key]
            self._save_index()
            
            return True
    
    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            # Delete all entry files
            for key in list(self._index.keys()):
                entry_path = self._get_entry_path(key)
                if os.path.exists(entry_path):
                    try:
                        os.remove(entry_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete cache entry file: {e}")
            
            # Clear the index
            self._index = {}
            self._save_index()
            
            # Reset stats
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    async def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        async with self._lock:
            # Check if the key exists in the index
            if key not in self._index:
                return False
            
            # Get entry metadata from the index
            entry_meta = self._index[key]
            
            # Check if the entry is expired
            if entry_meta.get("ttl", 0) > 0:
                created_at = entry_meta.get("created_at", 0)
                expires_at = created_at + entry_meta.get("ttl", 0)
                if time.time() > expires_at:
                    await self.delete(key)
                    return False
            
            # Get the entry file path
            entry_path = self._get_entry_path(key)
            
            # Check if the entry file exists
            if not os.path.exists(entry_path):
                # Entry file is missing, remove from index
                del self._index[key]
                self._save_index()
                return False
            
            return True
    
    async def get_size(self) -> int:
        """Get the current size of the cache in bytes.
        
        Returns:
            Size in bytes
        """
        async with self._lock:
            return sum(entry.get("size", 0) for entry in self._index.values())
    
    async def get_count(self) -> int:
        """Get the current number of items in the cache.
        
        Returns:
            Number of items
        """
        async with self._lock:
            return len(self._index)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        async with self._lock:
            return {
                "type": "disk",
                "size": await self.get_size(),
                "count": await self.get_count(),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                "evictions": self._evictions,
                "cache_dir": self.config.cache_dir
            }
    
    async def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict entries if needed to make room for a new entry.
        
        Args:
            new_entry_size: Size of the new entry in bytes
        """
        # Check if we need to evict based on max_items
        if self.config.max_items > 0 and len(self._index) >= self.config.max_items:
            # Evict the oldest entry
            oldest_key = min(self._index.keys(), key=lambda k: self._index[k].get("created_at", 0))
            await self.delete(oldest_key)
            self._evictions += 1
        
        # Check if we need to evict based on max_size
        if self.config.max_size > 0:
            current_size = await self.get_size()
            while current_size + new_entry_size > self.config.max_size and self._index:
                # Evict the oldest entry
                oldest_key = min(self._index.keys(), key=lambda k: self._index[k].get("created_at", 0))
                evicted_size = self._index[oldest_key].get("size", 0)
                await self.delete(oldest_key)
                self._evictions += 1
                current_size -= evicted_size
    
    async def _get_all_keys(self) -> List[str]:
        """Get all keys in the cache.
        
        Returns:
            List of keys
        """
        async with self._lock:
            return list(self._index.keys())


def create_cache(config: CacheConfig) -> FilesystemCache:
    """Create a cache instance based on the configuration.
    
    Args:
        config: Cache configuration
        
    Returns:
        Cache instance
    """
    if config.cache_dir:
        return DiskCache(config)
    else:
        return MemoryCache(config)
