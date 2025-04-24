"""
Advanced filesystem implementation for the Agentor framework.

This module provides an advanced filesystem implementation that combines caching, compression, and encryption.
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
from .cached_filesystem import CachedFilesystem
from .compression import CompressionConfig
from .compressed_filesystem import CompressedFilesystem
from .encryption import EncryptionConfig
from .encrypted_filesystem import EncryptedFilesystem

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AdvancedFilesystem(FilesystemInterface):
    """Advanced filesystem implementation that combines caching, compression, and encryption."""
    
    def __init__(
        self,
        filesystem: FilesystemInterface,
        cache_config: Optional[CacheConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
        encryption_config: Optional[EncryptionConfig] = None
    ):
        """Initialize the advanced filesystem.
        
        Args:
            filesystem: The underlying filesystem to wrap
            cache_config: Cache configuration
            compression_config: Compression configuration
            encryption_config: Encryption configuration
        """
        self.base_filesystem = filesystem
        
        # Apply the wrappers in the correct order:
        # 1. Encryption (innermost)
        # 2. Compression
        # 3. Caching (outermost)
        
        # Start with the base filesystem
        current_fs = filesystem
        
        # Apply encryption if configured
        if encryption_config is not None:
            current_fs = EncryptedFilesystem(current_fs, encryption_config)
        
        # Apply compression if configured
        if compression_config is not None:
            current_fs = CompressedFilesystem(current_fs, compression_config)
        
        # Apply caching if configured
        if cache_config is not None:
            current_fs = CachedFilesystem(current_fs, cache_config)
        
        # Store the final wrapped filesystem
        self.filesystem = current_fs
        
        # Set the name
        self.name = f"advanced-{filesystem.name}"
        
        # Store configurations
        self.cache_config = cache_config
        self.compression_config = compression_config
        self.encryption_config = encryption_config
    
    async def connect(self) -> FilesystemResult:
        """Connect to the filesystem."""
        return await self.filesystem.connect()
    
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the filesystem."""
        return await self.filesystem.disconnect()
    
    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        return await self.filesystem.read_text(path, encoding)
    
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        return await self.filesystem.read_bytes(path)
    
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        return await self.filesystem.write_text(path, content, encoding)
    
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        return await self.filesystem.write_bytes(path, content)
    
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        return await self.filesystem.append_text(path, content, encoding)
    
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        return await self.filesystem.append_bytes(path, content)
    
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        return await self.filesystem.delete_file(path)
    
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        return await self.filesystem.create_dir(path, exist_ok)
    
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        return await self.filesystem.list_dir(path)
    
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        return await self.filesystem.delete_dir(path, recursive)
    
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        return await self.filesystem.get_info(path)
    
    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        return await self.filesystem.exists(path)
    
    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        return await self.filesystem.is_file(path)
    
    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        return await self.filesystem.is_dir(path)
    
    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        return await self.filesystem.copy(src_path, dst_path, overwrite)
    
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        return await self.filesystem.move(src_path, dst_path, overwrite)
    
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        return await self.filesystem.get_size(path)
    
    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        return await self.filesystem.get_modified_time(path)
    
    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        return await self.filesystem.set_modified_time(path, mtime)
    
    async def walk(self, path: str) -> FilesystemResult[List[Tuple[str, List[str], List[str]]]]:
        """Walk a directory tree."""
        return await self.filesystem.walk(path)
    
    async def clear_cache(self) -> None:
        """Clear the cache if caching is enabled."""
        if isinstance(self.filesystem, CachedFilesystem):
            await self.filesystem.clear_cache()
    
    async def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if caching is enabled.
        
        Returns:
            Dictionary of statistics or None if caching is not enabled
        """
        if isinstance(self.filesystem, CachedFilesystem):
            return await self.filesystem.get_cache_stats()
        return None
    
    def get_features(self) -> Dict[str, bool]:
        """Get the enabled features.
        
        Returns:
            Dictionary of feature flags
        """
        return {
            "caching": self.cache_config is not None,
            "compression": self.compression_config is not None,
            "encryption": self.encryption_config is not None
        }
