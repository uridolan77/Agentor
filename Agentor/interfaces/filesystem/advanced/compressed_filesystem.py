"""
Compressed filesystem implementation for the Agentor framework.

This module provides a compressed filesystem implementation that wraps any filesystem.
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
from .compression import (
    CompressionConfig, CompressionAlgorithm,
    should_compress, compress_data, decompress_data,
    compress_text, decompress_text
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CompressedFilesystem(FilesystemInterface):
    """Compressed filesystem implementation that wraps any filesystem."""
    
    def __init__(
        self,
        filesystem: FilesystemInterface,
        compression_config: CompressionConfig = None
    ):
        """Initialize the compressed filesystem.
        
        Args:
            filesystem: The underlying filesystem to wrap
            compression_config: Compression configuration
        """
        self.filesystem = filesystem
        self.compression_config = compression_config or CompressionConfig()
        self.name = f"compressed-{filesystem.name}"
        
        # Keep track of compressed files
        self._compressed_files: Set[str] = set()
    
    async def connect(self) -> FilesystemResult:
        """Connect to the filesystem."""
        return await self.filesystem.connect()
    
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the filesystem."""
        return await self.filesystem.disconnect()
    
    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        # Check if the file is compressed
        is_compressed = await self._is_compressed(path)
        
        if is_compressed:
            # Read the compressed data
            result = await self.filesystem.read_bytes(path)
            if not result.success:
                return FilesystemResult.error_result(result.error)
            
            try:
                # Decompress the data
                text = decompress_text(result.data, encoding, self.compression_config)
                return FilesystemResult.success_result(data=text)
            except Exception as e:
                logger.error(f"Failed to decompress file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to decompress file: {e}"))
        else:
            # Read the file normally
            return await self.filesystem.read_text(path, encoding)
    
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        # Check if the file is compressed
        is_compressed = await self._is_compressed(path)
        
        if is_compressed:
            # Read the compressed data
            result = await self.filesystem.read_bytes(path)
            if not result.success:
                return result
            
            try:
                # Decompress the data
                data = decompress_data(result.data, self.compression_config)
                return FilesystemResult.success_result(data=data)
            except Exception as e:
                logger.error(f"Failed to decompress file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to decompress file: {e}"))
        else:
            # Read the file normally
            return await self.filesystem.read_bytes(path)
    
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        # Check if the file should be compressed
        if should_compress(path, self.compression_config, len(content.encode(encoding))):
            try:
                # Compress the content
                compressed_data = compress_text(content, encoding, self.compression_config)
                
                # Write the compressed data
                result = await self.filesystem.write_bytes(path, compressed_data)
                
                # If successful, add to the list of compressed files
                if result.success:
                    self._compressed_files.add(path)
                
                return result
            except Exception as e:
                logger.error(f"Failed to compress file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to compress file: {e}"))
        else:
            # Write the file normally
            return await self.filesystem.write_text(path, content, encoding)
    
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        # Check if the file should be compressed
        if should_compress(path, self.compression_config, len(content)):
            try:
                # Compress the content
                compressed_data = compress_data(content, self.compression_config)
                
                # Write the compressed data
                result = await self.filesystem.write_bytes(path, compressed_data)
                
                # If successful, add to the list of compressed files
                if result.success:
                    self._compressed_files.add(path)
                
                return result
            except Exception as e:
                logger.error(f"Failed to compress file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to compress file: {e}"))
        else:
            # Write the file normally
            return await self.filesystem.write_bytes(path, content)
    
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        # Check if the file is already compressed
        is_compressed = await self._is_compressed(path)
        
        if is_compressed:
            # Read the existing content
            result = await self.read_text(path, encoding)
            if not result.success:
                return result
            
            # Append the new content
            new_content = result.data + content
            
            # Write the combined content
            return await self.write_text(path, new_content, encoding)
        else:
            # Check if the file should be compressed
            if should_compress(path, self.compression_config):
                # Check if the file exists
                exists_result = await self.filesystem.exists(path)
                if not exists_result.success:
                    return exists_result
                
                if exists_result.data:
                    # Read the existing content
                    result = await self.filesystem.read_text(path, encoding)
                    if not result.success:
                        return result
                    
                    # Append the new content
                    new_content = result.data + content
                    
                    # Write the combined content
                    return await self.write_text(path, new_content, encoding)
                else:
                    # File doesn't exist, write the content
                    return await self.write_text(path, content, encoding)
            else:
                # Append normally
                return await self.filesystem.append_text(path, content, encoding)
    
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        # Check if the file is already compressed
        is_compressed = await self._is_compressed(path)
        
        if is_compressed:
            # Read the existing content
            result = await self.read_bytes(path)
            if not result.success:
                return result
            
            # Append the new content
            new_content = result.data + content
            
            # Write the combined content
            return await self.write_bytes(path, new_content)
        else:
            # Check if the file should be compressed
            if should_compress(path, self.compression_config):
                # Check if the file exists
                exists_result = await self.filesystem.exists(path)
                if not exists_result.success:
                    return exists_result
                
                if exists_result.data:
                    # Read the existing content
                    result = await self.filesystem.read_bytes(path)
                    if not result.success:
                        return result
                    
                    # Append the new content
                    new_content = result.data + content
                    
                    # Write the combined content
                    return await self.write_bytes(path, new_content)
                else:
                    # File doesn't exist, write the content
                    return await self.write_bytes(path, content)
            else:
                # Append normally
                return await self.filesystem.append_bytes(path, content)
    
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        # Remove from the list of compressed files if present
        if path in self._compressed_files:
            self._compressed_files.remove(path)
        
        # Delete the file normally
        return await self.filesystem.delete_file(path)
    
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        return await self.filesystem.create_dir(path, exist_ok)
    
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        return await self.filesystem.list_dir(path)
    
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        # If recursive, remove all files in the directory from the list of compressed files
        if recursive:
            # Get all files in the directory
            walk_result = await self.walk(path)
            if walk_result.success:
                for root, _, files in walk_result.data:
                    for file in files:
                        file_path = os.path.join(path, root, file)
                        if file_path in self._compressed_files:
                            self._compressed_files.remove(file_path)
        
        # Delete the directory normally
        return await self.filesystem.delete_dir(path, recursive)
    
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        # Get the file info normally
        result = await self.filesystem.get_info(path)
        if not result.success:
            return result
        
        # Check if the file is compressed
        if path in self._compressed_files and result.data.is_file:
            # Get the uncompressed size
            size_result = await self._get_uncompressed_size(path)
            if size_result.success:
                # Update the file info with the uncompressed size
                info = result.data
                info.metadata = info.metadata or {}
                info.metadata["compressed"] = True
                info.metadata["compressed_size"] = info.size
                info.metadata["uncompressed_size"] = size_result.data
                
                # Don't modify the actual size, as that would be misleading
                # info.size = size_result.data
                
                return FilesystemResult.success_result(data=info)
        
        return result
    
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
        # Check if the source file is compressed
        is_compressed = await self._is_compressed(src_path)
        
        if is_compressed:
            # Check if the source is a file
            is_file_result = await self.filesystem.is_file(src_path)
            if not is_file_result.success:
                return is_file_result
            
            if is_file_result.data:
                # Read the source file
                read_result = await self.read_bytes(src_path)
                if not read_result.success:
                    return read_result
                
                # Write to the destination
                write_result = await self.write_bytes(dst_path, read_result.data)
                return write_result
        
        # Copy normally
        result = await self.filesystem.copy(src_path, dst_path, overwrite)
        
        # If successful and the source file is compressed, add the destination to the list
        if result.success and is_compressed:
            self._compressed_files.add(dst_path)
        
        return result
    
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        # Check if the source file is compressed
        is_compressed = await self._is_compressed(src_path)
        
        # Move normally
        result = await self.filesystem.move(src_path, dst_path, overwrite)
        
        # If successful and the source file is compressed, update the list
        if result.success and is_compressed:
            if src_path in self._compressed_files:
                self._compressed_files.remove(src_path)
            self._compressed_files.add(dst_path)
        
        return result
    
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        # Check if the path is a file
        is_file_result = await self.filesystem.is_file(path)
        if not is_file_result.success:
            return is_file_result
        
        if is_file_result.data:
            # Check if the file is compressed
            is_compressed = await self._is_compressed(path)
            
            if is_compressed:
                # Get the uncompressed size
                return await self._get_uncompressed_size(path)
        
        # Get the size normally
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
    
    async def _is_compressed(self, path: str) -> bool:
        """Check if a file is compressed.
        
        Args:
            path: File path
            
        Returns:
            True if the file is compressed, False otherwise
        """
        # Check if the file is in the list of compressed files
        if path in self._compressed_files:
            return True
        
        # Check if the file exists
        exists_result = await self.filesystem.exists(path)
        if not exists_result.success or not exists_result.data:
            return False
        
        # Check if the file is a file
        is_file_result = await self.filesystem.is_file(path)
        if not is_file_result.success or not is_file_result.data:
            return False
        
        # Read the first few bytes to check for the compression header
        try:
            # Read the first 10 bytes (header size)
            read_result = await self.filesystem.read_bytes(path)
            if not read_result.success:
                return False
            
            data = read_result.data
            
            # Check for the compression header
            from .compression import COMPRESSION_HEADER
            if len(data) >= 10 and data[:8] == COMPRESSION_HEADER:
                # Add to the list of compressed files
                self._compressed_files.add(path)
                return True
        except Exception:
            pass
        
        return False
    
    async def _get_uncompressed_size(self, path: str) -> FilesystemResult[int]:
        """Get the uncompressed size of a file.
        
        Args:
            path: File path
            
        Returns:
            Uncompressed size in bytes
        """
        # Read the compressed data
        result = await self.filesystem.read_bytes(path)
        if not result.success:
            return result
        
        try:
            # Decompress the data
            data = decompress_data(result.data, self.compression_config)
            
            # Return the size
            return FilesystemResult.success_result(data=len(data))
        except Exception as e:
            logger.error(f"Failed to get uncompressed size: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to get uncompressed size: {e}"))
