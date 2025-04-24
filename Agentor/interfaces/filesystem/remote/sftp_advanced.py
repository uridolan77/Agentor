"""
Advanced SFTP filesystem operations for the Agentor framework.

This module provides advanced operations for the SFTP filesystem interface.
"""

import os
import io
import asyncio
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Tuple
import time
import stat
import asyncssh

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from .base import RemoteFilesystem
from .sftp import SFTPFilesystem
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


async def sftp_copy(self: SFTPFilesystem, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
    """Copy a file or directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        src_path = self._normalize_path(src_path)
        dst_path = self._normalize_path(dst_path)
        
        # Check if source exists
        src_exists_result = await self.exists(src_path)
        if not src_exists_result.success:
            return src_exists_result
        
        if not src_exists_result.data:
            return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))
        
        # Check if destination exists
        dst_exists_result = await self.exists(dst_path)
        if not dst_exists_result.success:
            return dst_exists_result
        
        if dst_exists_result.data and not overwrite:
            return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
        
        # Check if source is a file or directory
        is_file_result = await self.is_file(src_path)
        if not is_file_result.success:
            return is_file_result
        
        if is_file_result.data:
            # Source is a file, copy it
            read_result = await self.read_bytes(src_path)
            if not read_result.success:
                return read_result
            
            write_result = await self.write_bytes(dst_path, read_result.data)
            return write_result
        else:
            # Source is a directory, copy it recursively
            
            # Create destination directory
            create_dir_result = await self.create_dir(dst_path, exist_ok=True)
            if not create_dir_result.success:
                return create_dir_result
            
            # List source directory
            list_result = await self.list_dir(src_path)
            if not list_result.success:
                return list_result
            
            # Copy each item in the directory
            for name in list_result.data:
                src_item_path = f"{src_path}/{name}"
                dst_item_path = f"{dst_path}/{name}"
                
                copy_result = await self.copy(src_item_path, dst_item_path, overwrite)
                if not copy_result.success:
                    return copy_result
            
            return FilesystemResult.success_result()
    except Exception as e:
        logger.error(f"Failed to copy: {src_path} -> {dst_path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to copy: {e}"))


async def sftp_move(self: SFTPFilesystem, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
    """Move a file or directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        src_path = self._normalize_path(src_path)
        dst_path = self._normalize_path(dst_path)
        full_src_path = self._get_full_path(src_path)
        full_dst_path = self._get_full_path(dst_path)
        
        # Check if source exists
        src_exists_result = await self.exists(src_path)
        if not src_exists_result.success:
            return src_exists_result
        
        if not src_exists_result.data:
            return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))
        
        # Check if destination exists
        dst_exists_result = await self.exists(dst_path)
        if not dst_exists_result.success:
            return dst_exists_result
        
        if dst_exists_result.data and not overwrite:
            return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
        
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(full_dst_path)
        if parent_dir:
            try:
                await self.client.mkdir(parent_dir, parents=True)
            except (asyncssh.SFTPFailure, asyncssh.SFTPPermissionDenied):
                # Ignore if the directory already exists or we don't have permission
                pass
        
        # Try to use rename first (most efficient)
        try:
            # If destination exists and overwrite is True, delete it first
            if dst_exists_result.data and overwrite:
                is_file_result = await self.is_file(dst_path)
                if not is_file_result.success:
                    return is_file_result
                
                if is_file_result.data:
                    await self.delete_file(dst_path)
                else:
                    await self.delete_dir(dst_path, recursive=True)
            
            # Rename the file/directory
            await self.client.rename(full_src_path, full_dst_path)
            return FilesystemResult.success_result()
        except (asyncssh.SFTPFailure, asyncssh.SFTPOpUnsupported):
            # If rename fails (e.g., across different filesystems), fall back to copy and delete
            copy_result = await self.copy(src_path, dst_path, overwrite)
            if not copy_result.success:
                return copy_result
            
            # Delete the source
            is_file_result = await self.is_file(src_path)
            if not is_file_result.success:
                return is_file_result
            
            if is_file_result.data:
                delete_result = await self.delete_file(src_path)
            else:
                delete_result = await self.delete_dir(src_path, recursive=True)
            
            return delete_result
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {src_path} -> {dst_path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {src_path} -> {dst_path}"))
    except Exception as e:
        logger.error(f"Failed to move: {src_path} -> {dst_path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to move: {e}"))


async def sftp_get_size(self: SFTPFilesystem, path: str) -> FilesystemResult[int]:
    """Get the size of a file or directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get file/directory info
        attrs = await self.client.stat(full_path)
        
        if stat.S_ISREG(attrs.permissions):
            # Path is a file, return its size
            return FilesystemResult.success_result(data=attrs.size)
        elif stat.S_ISDIR(attrs.permissions):
            # Path is a directory, calculate total size recursively
            total_size = 0
            
            # List directory contents
            files = await self.client.listdir(full_path)
            
            # Calculate size for each item
            for name in files:
                item_path = f"{path}/{name}"
                
                size_result = await self.get_size(item_path)
                if size_result.success:
                    total_size += size_result.data
            
            return FilesystemResult.success_result(data=total_size)
        else:
            # Not a file or directory
            return FilesystemResult.error_result(FilesystemError(f"Not a file or directory: {path}"))
    except asyncssh.SFTPNoSuchFile:
        logger.error(f"Path not found: {path}")
        return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to get size: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to get size: {e}"))


async def sftp_get_modified_time(self: SFTPFilesystem, path: str) -> FilesystemResult[float]:
    """Get the last modified time of a file or directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get file/directory info
        attrs = await self.client.stat(full_path)
        
        return FilesystemResult.success_result(data=attrs.mtime)
    except asyncssh.SFTPNoSuchFile:
        logger.error(f"Path not found: {path}")
        return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to get modified time: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to get modified time: {e}"))


async def sftp_set_modified_time(self: SFTPFilesystem, path: str, mtime: float) -> FilesystemResult:
    """Set the last modified time of a file or directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get current attributes
        attrs = await self.client.stat(full_path)
        
        # Set the modification time
        await self.client.utime(full_path, (attrs.atime, mtime))
        
        return FilesystemResult.success_result()
    except asyncssh.SFTPNoSuchFile:
        logger.error(f"Path not found: {path}")
        return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to set modified time: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to set modified time: {e}"))


async def sftp_walk(self: SFTPFilesystem, path: str) -> FilesystemResult[List[tuple]]:
    """Walk a directory tree."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a directory
        attrs = await self.client.stat(full_path)
        if not stat.S_ISDIR(attrs.permissions):
            return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
        
        # Walk the directory tree
        walk_result = []
        await self._walk_recursive(path, "", walk_result)
        
        return FilesystemResult.success_result(data=walk_result)
    except asyncssh.SFTPNoSuchFile:
        logger.error(f"Path not found: {path}")
        return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to walk directory: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to walk directory: {e}"))


async def sftp_walk_recursive(self: SFTPFilesystem, base_path: str, rel_path: str, result: List[Tuple[str, List[str], List[str]]]):
    """Recursively walk a directory tree."""
    # Get the full path
    full_path = base_path
    if rel_path:
        full_path = f"{base_path}/{rel_path}"
    
    # List directory contents
    list_result = await self.list_dir(full_path)
    if not list_result.success:
        return
    
    # Separate directories and files
    dirs = []
    files = []
    
    for name in list_result.data:
        item_path = f"{full_path}/{name}"
        
        is_dir_result = await self.is_dir(item_path)
        if not is_dir_result.success:
            continue
        
        if is_dir_result.data:
            dirs.append(name)
        else:
            files.append(name)
    
    # Add this directory to the result
    result.append((rel_path, dirs, files))
    
    # Recursively walk subdirectories
    for dir_name in dirs:
        new_rel_path = dir_name
        if rel_path:
            new_rel_path = f"{rel_path}/{dir_name}"
        
        await self._walk_recursive(base_path, new_rel_path, result)


# Add the methods to the SFTPFilesystem class
SFTPFilesystem.copy = sftp_copy
SFTPFilesystem.move = sftp_move
SFTPFilesystem.get_size = sftp_get_size
SFTPFilesystem.get_modified_time = sftp_get_modified_time
SFTPFilesystem.set_modified_time = sftp_set_modified_time
SFTPFilesystem.walk = sftp_walk
SFTPFilesystem._walk_recursive = sftp_walk_recursive
