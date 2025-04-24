"""
Cached filesystem implementation for the Agentor framework.

This module provides a cached filesystem implementation that wraps any filesystem.
"""

import os
import time
import asyncio
import functools
from typing import Dict, Any, Optional, List, Union, Tuple, Set, TypeVar, Generic, cast
import logging

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from .filesystem_cache import FilesystemCache, CacheConfig, create_cache

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CachedFilesystem(FilesystemInterface):
    """Cached filesystem implementation that wraps any filesystem."""
    
    def __init__(
        self,
        filesystem: FilesystemInterface,
        cache_config: CacheConfig = None
    ):
        """Initialize the cached filesystem.
        
        Args:
            filesystem: The underlying filesystem to wrap
            cache_config: Cache configuration
        """
        self.filesystem = filesystem
        self.cache_config = cache_config or CacheConfig()
        self.cache = create_cache(self.cache_config)
        self.name = f"cached-{filesystem.name}"
    
    async def connect(self) -> FilesystemResult:
        """Connect to the filesystem."""
        return await self.filesystem.connect()
    
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the filesystem."""
        return await self.filesystem.disconnect()
    
    async def _cached_operation(
        self,
        operation: str,
        path: str,
        func,
        *args,
        invalidate: bool = False,
        **kwargs
    ) -> FilesystemResult:
        """Execute a cached filesystem operation.
        
        Args:
            operation: Operation name
            path: File path
            func: Function to execute
            *args: Additional arguments
            invalidate: Whether to invalidate the cache after the operation
            **kwargs: Additional keyword arguments
            
        Returns:
            Operation result
        """
        # Check if we should cache this operation
        should_cache = self.cache.should_cache(operation, path)
        
        # Generate cache key
        cache_key = self.cache.generate_key(operation, path, *args, **kwargs)
        
        # Check if the result is in the cache
        if should_cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute the operation
        result = await func(*args, **kwargs)
        
        # Cache the result if successful or if we're configured to cache failures
        if should_cache and (result.success or self.cache_config.cache_failures):
            await self.cache.set(cache_key, result)
        
        # Check if we should invalidate the cache
        if invalidate and result.success:
            should_invalidate, paths_to_invalidate = self.cache.should_invalidate(operation, path)
            if should_invalidate:
                await self.cache.invalidate(paths_to_invalidate)
        
        return result
    
    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        return await self._cached_operation(
            "read_text",
            path,
            self.filesystem.read_text,
            path,
            encoding=encoding
        )
    
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        return await self._cached_operation(
            "read_bytes",
            path,
            self.filesystem.read_bytes,
            path
        )
    
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        return await self._cached_operation(
            "write_text",
            path,
            self.filesystem.write_text,
            path,
            content,
            encoding=encoding,
            invalidate=True
        )
    
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        return await self._cached_operation(
            "write_bytes",
            path,
            self.filesystem.write_bytes,
            path,
            content,
            invalidate=True
        )
    
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        return await self._cached_operation(
            "append_text",
            path,
            self.filesystem.append_text,
            path,
            content,
            encoding=encoding,
            invalidate=True
        )
    
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        return await self._cached_operation(
            "append_bytes",
            path,
            self.filesystem.append_bytes,
            path,
            content,
            invalidate=True
        )
    
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        return await self._cached_operation(
            "delete_file",
            path,
            self.filesystem.delete_file,
            path,
            invalidate=True
        )
    
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        return await self._cached_operation(
            "create_dir",
            path,
            self.filesystem.create_dir,
            path,
            exist_ok=exist_ok,
            invalidate=True
        )
    
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        return await self._cached_operation(
            "list_dir",
            path,
            self.filesystem.list_dir,
            path
        )
    
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        return await self._cached_operation(
            "delete_dir",
            path,
            self.filesystem.delete_dir,
            path,
            recursive=recursive,
            invalidate=True
        )
    
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        return await self._cached_operation(
            "get_info",
            path,
            self.filesystem.get_info,
            path
        )
    
    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        return await self._cached_operation(
            "exists",
            path,
            self.filesystem.exists,
            path
        )
    
    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        return await self._cached_operation(
            "is_file",
            path,
            self.filesystem.is_file,
            path
        )
    
    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        return await self._cached_operation(
            "is_dir",
            path,
            self.filesystem.is_dir,
            path
        )
    
    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        result = await self._cached_operation(
            "copy",
            src_path,
            self.filesystem.copy,
            src_path,
            dst_path,
            overwrite=overwrite
        )
        
        # Invalidate cache for both source and destination paths
        if result.success:
            await self.cache.invalidate({dst_path, f"{dst_path}/*"})
        
        return result
    
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        result = await self._cached_operation(
            "move",
            src_path,
            self.filesystem.move,
            src_path,
            dst_path,
            overwrite=overwrite
        )
        
        # Invalidate cache for both source and destination paths
        if result.success:
            await self.cache.invalidate({src_path, f"{src_path}/*", dst_path, f"{dst_path}/*"})
        
        return result
    
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        return await self._cached_operation(
            "get_size",
            path,
            self.filesystem.get_size,
            path
        )
    
    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        return await self._cached_operation(
            "get_modified_time",
            path,
            self.filesystem.get_modified_time,
            path
        )
    
    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        return await self._cached_operation(
            "set_modified_time",
            path,
            self.filesystem.set_modified_time,
            path,
            mtime,
            invalidate=True
        )
    
    async def walk(self, path: str) -> FilesystemResult[List[Tuple[str, List[str], List[str]]]]:
        """Walk a directory tree."""
        return await self._cached_operation(
            "walk",
            path,
            self.filesystem.walk,
            path
        )
    
    async def clear_cache(self) -> None:
        """Clear the cache."""
        await self.cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        return await self.cache.get_stats()
