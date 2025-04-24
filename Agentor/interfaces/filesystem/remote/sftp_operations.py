"""
SFTP filesystem operations for the Agentor framework.

This module provides additional operations for the SFTP filesystem interface.
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


async def sftp_delete_file(self: SFTPFilesystem, path: str) -> FilesystemResult:
    """Delete a file."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a file
        try:
            attrs = await self.client.stat(full_path)
            if not stat.S_ISREG(attrs.permissions):
                return FilesystemResult.error_result(FilesystemError(f"Not a file: {path}"))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
        
        # Delete the file
        await self.client.remove(full_path)
        
        return FilesystemResult.success_result()
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to delete file: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to delete file: {e}"))


async def sftp_create_dir(self: SFTPFilesystem, path: str, exist_ok: bool = False) -> FilesystemResult:
    """Create a directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the directory already exists
        if exist_ok:
            try:
                attrs = await self.client.stat(full_path)
                if stat.S_ISDIR(attrs.permissions):
                    return FilesystemResult.success_result()
            except asyncssh.SFTPNoSuchFile:
                pass
        
        # Create the directory
        await self.client.mkdir(full_path, parents=True)
        
        return FilesystemResult.success_result()
    except asyncssh.SFTPFailure as e:
        if "File exists" in str(e) and exist_ok:
            return FilesystemResult.success_result()
        logger.error(f"Failed to create directory: {path}, error: {e}")
        return FilesystemResult.error_result(FileExistsError(f"Directory already exists: {path}"))
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to create directory: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to create directory: {e}"))


async def sftp_list_dir(self: SFTPFilesystem, path: str) -> FilesystemResult[List[str]]:
    """List directory contents."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a directory
        try:
            attrs = await self.client.stat(full_path)
            if not stat.S_ISDIR(attrs.permissions):
                return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.error_result(FileNotFoundError(f"Directory not found: {path}"))
        
        # List the directory contents
        files = await self.client.listdir(full_path)
        
        return FilesystemResult.success_result(data=files)
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to list directory: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to list directory: {e}"))


async def sftp_delete_dir(self: SFTPFilesystem, path: str, recursive: bool = False) -> FilesystemResult:
    """Delete a directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a directory
        try:
            attrs = await self.client.stat(full_path)
            if not stat.S_ISDIR(attrs.permissions):
                return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.error_result(FileNotFoundError(f"Directory not found: {path}"))
        
        if recursive:
            # List all files and directories
            files = await self.client.listdir(full_path)
            
            # Delete all files and subdirectories
            for name in files:
                item_path = f"{full_path}/{name}"
                
                # Check if it's a file or directory
                try:
                    attrs = await self.client.stat(item_path)
                    if stat.S_ISDIR(attrs.permissions):
                        # Recursively delete subdirectory
                        subdir_path = f"{path}/{name}"
                        await sftp_delete_dir(self, subdir_path, recursive=True)
                    else:
                        # Delete file
                        await self.client.remove(item_path)
                except Exception as e:
                    logger.error(f"Failed to delete item: {item_path}, error: {e}")
                    # Continue with other items
        
        # Delete the directory
        await self.client.rmdir(full_path)
        
        return FilesystemResult.success_result()
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to delete directory: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to delete directory: {e}"))


async def sftp_get_info(self: SFTPFilesystem, path: str) -> FilesystemResult[FileInfo]:
    """Get information about a file or directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get file info
        attrs = await self.client.stat(full_path)
        
        # Determine if it's a file or directory
        is_file = stat.S_ISREG(attrs.permissions)
        is_dir = stat.S_ISDIR(attrs.permissions)
        
        # Create FileInfo object
        info = FileInfo(
            name=os.path.basename(path),
            path=path,
            size=attrs.size,
            is_file=is_file,
            is_dir=is_dir,
            created_time=attrs.atime,  # SFTP doesn't provide creation time, use access time
            modified_time=attrs.mtime,
            accessed_time=attrs.atime,
            metadata={
                "uid": attrs.uid,
                "gid": attrs.gid,
                "permissions": attrs.permissions,
                "mode": stat.filemode(attrs.permissions)
            }
        )
        
        return FilesystemResult.success_result(data=info)
    except asyncssh.SFTPNoSuchFile:
        logger.error(f"Path not found: {path}")
        return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to get info: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to get info: {e}"))


async def sftp_exists(self: SFTPFilesystem, path: str) -> FilesystemResult[bool]:
    """Check if a file or directory exists."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Try to get file info
        try:
            await self.client.stat(full_path)
            return FilesystemResult.success_result(data=True)
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.success_result(data=False)
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to check if path exists: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to check if path exists: {e}"))


async def sftp_is_file(self: SFTPFilesystem, path: str) -> FilesystemResult[bool]:
    """Check if a path is a file."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        try:
            attrs = await self.client.stat(full_path)
            return FilesystemResult.success_result(data=stat.S_ISREG(attrs.permissions))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.success_result(data=False)
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to check if path is a file: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to check if path is a file: {e}"))


async def sftp_is_dir(self: SFTPFilesystem, path: str) -> FilesystemResult[bool]:
    """Check if a path is a directory."""
    result = await self._ensure_connected()
    if not result.success:
        return result
    
    try:
        self.last_activity = time.time()
        full_path = self._get_full_path(self._normalize_path(path))
        
        try:
            attrs = await self.client.stat(full_path)
            return FilesystemResult.success_result(data=stat.S_ISDIR(attrs.permissions))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.success_result(data=False)
    except asyncssh.SFTPPermissionDenied:
        logger.error(f"Permission denied: {path}")
        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
    except Exception as e:
        logger.error(f"Failed to check if path is a directory: {path}, error: {e}")
        return FilesystemResult.error_result(FilesystemError(f"Failed to check if path is a directory: {e}"))


# Add the methods to the SFTPFilesystem class
SFTPFilesystem.delete_file = sftp_delete_file
SFTPFilesystem.create_dir = sftp_create_dir
SFTPFilesystem.list_dir = sftp_list_dir
SFTPFilesystem.delete_dir = sftp_delete_dir
SFTPFilesystem.get_info = sftp_get_info
SFTPFilesystem.exists = sftp_exists
SFTPFilesystem.is_file = sftp_is_file
SFTPFilesystem.is_dir = sftp_is_dir
