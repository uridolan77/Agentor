"""
Local filesystem operations for the Agentor framework.

This module provides interfaces for interacting with the local filesystem.
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
import time
import aiofiles

from .base import (
    FilesystemInterface, FilesystemResult, FileInfo, FileMode,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class LocalFilesystem(FilesystemInterface):
    """Interface for local filesystem operations."""

    def __init__(self, name: str, root_dir: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.root_dir = root_dir
        self.connected = True  # Local filesystem is always connected
        self.last_activity = time.time()

    async def connect(self) -> FilesystemResult:
        """Connect to the local filesystem.
        
        For local filesystem, this is a no-op since we're always connected.
        """
        self.connected = True
        self.last_activity = time.time()
        return FilesystemResult.success_result()

    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the local filesystem.
        
        For local filesystem, this is a no-op since we're always connected.
        """
        self.last_activity = time.time()
        return FilesystemResult.success_result()

    def _get_full_path(self, path: str) -> str:
        """Get the full path by joining with the root directory if specified."""
        if self.root_dir:
            return os.path.join(self.root_dir, path)
        return path

    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            async with aiofiles.open(full_path, mode='r', encoding=encoding) as file:
                content = await file.read()
            
            return FilesystemResult.success_result(data=content)
        except FileNotFoundError as e:
            logger.error(f"File not found: {path}, error: {e}")
            return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to read text from file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to read text from file: {e}"))

    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            async with aiofiles.open(full_path, mode='rb') as file:
                content = await file.read()
            
            return FilesystemResult.success_result(data=content)
        except FileNotFoundError as e:
            logger.error(f"File not found: {path}, error: {e}")
            return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to read bytes from file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to read bytes from file: {e}"))

    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, mode='w', encoding=encoding) as file:
                await file.write(content)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to write text to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to write text to file: {e}"))

    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, mode='wb') as file:
                await file.write(content)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to write bytes to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to write bytes to file: {e}"))

    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, mode='a', encoding=encoding) as file:
                await file.write(content)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to append text to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to append text to file: {e}"))

    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, mode='ab') as file:
                await file.write(content)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to append bytes to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to append bytes to file: {e}"))

    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
            
            if not os.path.isfile(full_path):
                return FilesystemResult.error_result(FilesystemError(f"Not a file: {path}"))
            
            os.remove(full_path)
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to delete file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to delete file: {e}"))

    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            os.makedirs(full_path, exist_ok=exist_ok)
            return FilesystemResult.success_result()
        except FileExistsError as e:
            logger.error(f"Directory already exists: {path}, error: {e}")
            return FilesystemResult.error_result(FileExistsError(f"Directory already exists: {path}"))
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to create directory: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to create directory: {e}"))

    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Directory not found: {path}"))
            
            if not os.path.isdir(full_path):
                return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
            
            contents = os.listdir(full_path)
            return FilesystemResult.success_result(data=contents)
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to list directory: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to list directory: {e}"))

    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Directory not found: {path}"))
            
            if not os.path.isdir(full_path):
                return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
            
            if recursive:
                shutil.rmtree(full_path)
            else:
                os.rmdir(full_path)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except OSError as e:
            if "Directory not empty" in str(e):
                return FilesystemResult.error_result(FilesystemError(f"Directory not empty: {path}. Use recursive=True to delete non-empty directories."))
            logger.error(f"Failed to delete directory: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to delete directory: {e}"))
        except Exception as e:
            logger.error(f"Failed to delete directory: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to delete directory: {e}"))

    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
            
            stat_result = os.stat(full_path)
            
            info = FileInfo(
                name=os.path.basename(full_path),
                path=path,
                size=stat_result.st_size,
                is_file=os.path.isfile(full_path),
                is_dir=os.path.isdir(full_path),
                created_time=stat_result.st_ctime,
                modified_time=stat_result.st_mtime,
                accessed_time=stat_result.st_atime,
                metadata={
                    "mode": stat_result.st_mode,
                    "uid": stat_result.st_uid,
                    "gid": stat_result.st_gid,
                    "dev": stat_result.st_dev,
                    "ino": stat_result.st_ino,
                    "nlink": stat_result.st_nlink
                }
            )
            
            return FilesystemResult.success_result(data=info)
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to get info: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to get info: {e}"))

    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            exists = os.path.exists(full_path)
            return FilesystemResult.success_result(data=exists)
        except Exception as e:
            logger.error(f"Failed to check if path exists: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to check if path exists: {e}"))

    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            is_file = os.path.isfile(full_path)
            return FilesystemResult.success_result(data=is_file)
        except Exception as e:
            logger.error(f"Failed to check if path is a file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to check if path is a file: {e}"))

    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            is_dir = os.path.isdir(full_path)
            return FilesystemResult.success_result(data=is_dir)
        except Exception as e:
            logger.error(f"Failed to check if path is a directory: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to check if path is a directory: {e}"))

    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        try:
            self.last_activity = time.time()
            full_src_path = self._get_full_path(src_path)
            full_dst_path = self._get_full_path(dst_path)
            
            if not os.path.exists(full_src_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))
            
            if os.path.exists(full_dst_path) and not overwrite:
                return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_dst_path), exist_ok=True)
            
            if os.path.isfile(full_src_path):
                shutil.copy2(full_src_path, full_dst_path)
            else:
                shutil.copytree(full_src_path, full_dst_path, dirs_exist_ok=overwrite)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {src_path} -> {dst_path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {src_path} -> {dst_path}"))
        except Exception as e:
            logger.error(f"Failed to copy: {src_path} -> {dst_path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to copy: {e}"))

    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        try:
            self.last_activity = time.time()
            full_src_path = self._get_full_path(src_path)
            full_dst_path = self._get_full_path(dst_path)
            
            if not os.path.exists(full_src_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))
            
            if os.path.exists(full_dst_path) and not overwrite:
                return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(full_dst_path), exist_ok=True)
            
            if os.path.exists(full_dst_path) and overwrite:
                if os.path.isfile(full_dst_path):
                    os.remove(full_dst_path)
                else:
                    shutil.rmtree(full_dst_path)
            
            shutil.move(full_src_path, full_dst_path)
            
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {src_path} -> {dst_path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {src_path} -> {dst_path}"))
        except Exception as e:
            logger.error(f"Failed to move: {src_path} -> {dst_path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to move: {e}"))

    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
            
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
            else:
                size = 0
                for dirpath, dirnames, filenames in os.walk(full_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        size += os.path.getsize(file_path)
            
            return FilesystemResult.success_result(data=size)
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to get size: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to get size: {e}"))

    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
            
            mtime = os.path.getmtime(full_path)
            return FilesystemResult.success_result(data=mtime)
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to get modified time: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to get modified time: {e}"))

    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
            
            os.utime(full_path, (os.path.getatime(full_path), mtime))
            return FilesystemResult.success_result()
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to set modified time: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to set modified time: {e}"))

    async def walk(self, path: str) -> FilesystemResult[List[tuple]]:
        """Walk a directory tree."""
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(path)
            
            if not os.path.exists(full_path):
                return FilesystemResult.error_result(FileNotFoundError(f"Path not found: {path}"))
            
            if not os.path.isdir(full_path):
                return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
            
            # Convert os.walk result to a list of tuples
            walk_result = []
            for dirpath, dirnames, filenames in os.walk(full_path):
                # Convert absolute paths to relative paths
                rel_dirpath = os.path.relpath(dirpath, full_path)
                if rel_dirpath == ".":
                    rel_dirpath = ""
                
                walk_result.append((rel_dirpath, dirnames, filenames))
            
            return FilesystemResult.success_result(data=walk_result)
        except PermissionError as e:
            logger.error(f"Permission denied: {path}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to walk directory: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to walk directory: {e}"))
