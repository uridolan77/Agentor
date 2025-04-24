"""
FTP filesystem operations for the Agentor framework.

This module provides interfaces for interacting with FTP servers.
"""

import os
import io
import asyncio
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Tuple
import time
import ftplib
import aioftp
from datetime import datetime
from functools import wraps

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo, FileMode,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from . import RemoteFilesystem
from agentor.utils.logging import get_logger
from agentor.llm_gateway.utils.retry import RetryConfig, RetryStrategy, retry_async
from agentor.llm_gateway.utils.timeout import with_timeout, TimeoutStrategy
from agentor.llm_gateway.utils.circuit_breaker import CircuitBreaker
from agentor.core.config import get_typed_config
from ..config import FTPFilesystemConfig

logger = get_logger(__name__)


# Use the FTPFilesystemConfig from the config module


class FTPFilesystem(RemoteFilesystem):
    """Interface for FTP filesystem operations."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int = 21,
        username: str = "anonymous",
        password: str = "",
        root_dir: Optional[str] = None,
        timeout: int = 30,
        passive_mode: bool = True,
        secure: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name,
            host=host,
            port=port,
            username=username,
            password=password,
            root_dir=root_dir,
            timeout=timeout,
            **kwargs
        )
        self.passive_mode = passive_mode
        self.secure = secure
        self.client = None
        self.connected = False

        # Load configuration or use defaults
        try:
            self.config = get_typed_config(FTPFilesystemConfig)
        except Exception as e:
            logger.warning(f"Failed to load FTP configuration, using defaults: {e}")
            self.config = FTPFilesystemConfig(host=host)

        # Set up retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.config.retry_max_attempts,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=self.config.retry_base_delay,
            max_delay=self.config.retry_max_delay,
            jitter=0.1,
            retry_on=[ConnectionError, TimeoutError, asyncio.TimeoutError]
        )

        # Set up circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"ftp-{host}-{port}",
            failure_threshold=self.config.circuit_breaker_failures,
            recovery_timeout=self.config.circuit_breaker_recovery
        )

    @with_timeout("ftp", "connect", strategy=TimeoutStrategy.FIXED)
    async def connect(self) -> FilesystemResult:
        """Connect to the FTP server."""
        async with self.circuit_breaker:
            try:
                # Use retry for connection attempts
                async def _connect_attempt():
                    self.client = aioftp.Client(
                        host=self.host,
                        port=self.port,
                        user=self.username,
                        password=self.password,
                        timeout=self.timeout,
                        ssl=self.secure
                    )

                    await self.client.connect()

                    if self.passive_mode:
                        await self.client.command("PASV")

                    if self.root_dir:
                        try:
                            await self.client.change_directory(self.root_dir)
                        except aioftp.errors.PathNotFoundError:
                            # Try to create the root directory if it doesn't exist
                            await self.client.make_directory(self.root_dir)
                            await self.client.change_directory(self.root_dir)

                # Execute with retry
                await retry_async(
                    _connect_attempt,
                    "ftp",
                    "connect",
                    self.retry_config
                )

                self.connected = True
                self.last_activity = time.time()
                logger.info(f"Connected to FTP server: {self.host}:{self.port}")
                return FilesystemResult.success_result()
            except Exception as e:
                if self.client:
                    try:
                        await self.client.quit()
                    except:
                        pass
                    self.client = None
                logger.error(f"Failed to connect to FTP server: {self.host}:{self.port}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to connect to FTP server: {e}"))

    @with_timeout("ftp", "disconnect", strategy=TimeoutStrategy.FIXED)
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the FTP server."""
        if not self.connected or not self.client:
            return FilesystemResult.success_result()

        try:
            # Use retry for disconnection
            async def _disconnect_attempt():
                await self.client.quit()

            # Execute with retry
            await retry_async(
                _disconnect_attempt,
                "ftp",
                "disconnect",
                self.retry_config
            )

            self.connected = False
            self.client = None
            logger.info(f"Disconnected from FTP server: {self.host}:{self.port}")
            return FilesystemResult.success_result()
        except Exception as e:
            # Even if there's an error, mark as disconnected
            self.connected = False
            self.client = None
            logger.error(f"Failed to disconnect from FTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to disconnect from FTP server: {e}"))

    async def _ensure_connected(self) -> FilesystemResult:
        """Ensure that we are connected to the FTP server."""
        if not self.connected or not self.client:
            return await self.connect()
        return FilesystemResult.success_result()

    # Helper decorator for FTP operations with resilience patterns
    def _with_resilience(operation_name):
        """Decorator to add resilience patterns to FTP operations."""
        def decorator(func):
            @wraps(func)
            @with_timeout("ftp", operation_name, strategy=TimeoutStrategy.ADAPTIVE)
            async def wrapper(self, *args, **kwargs):
                # Ensure connected
                result = await self._ensure_connected()
                if not result.success:
                    return result

                # Use circuit breaker
                async with self.circuit_breaker:
                    try:
                        # Update last activity time
                        self.last_activity = time.time()

                        # Execute the operation with retry
                        async def _operation_attempt():
                            return await func(self, *args, **kwargs)

                        return await retry_async(
                            _operation_attempt,
                            "ftp",
                            operation_name,
                            self.retry_config
                        )
                    except aioftp.errors.PathNotFoundError as e:
                        path = args[0] if args else kwargs.get('path', 'unknown')
                        logger.error(f"File not found: {path}")
                        return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
                    except aioftp.errors.PermissionError as e:
                        path = args[0] if args else kwargs.get('path', 'unknown')
                        logger.error(f"Permission denied: {path}")
                        return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
                    except Exception as e:
                        path = args[0] if args else kwargs.get('path', 'unknown')
                        logger.error(f"Failed to perform {operation_name} on {path}: {e}")
                        return FilesystemResult.error_result(FilesystemError(f"Failed to perform {operation_name}: {e}"))
            return wrapper
        return decorator

    @_with_resilience("read_text")
    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        path = self._normalize_path(path)

        # Create a BytesIO object to store the file contents
        buffer = io.BytesIO()

        # Download the file to the buffer
        await self.client.download_stream(path, buffer)

        # Reset the buffer position and read the contents
        buffer.seek(0)
        content = buffer.read().decode(encoding)

        return FilesystemResult.success_result(data=content)

    @_with_resilience("read_bytes")
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        path = self._normalize_path(path)

        # Create a BytesIO object to store the file contents
        buffer = io.BytesIO()

        # Download the file to the buffer
        await self.client.download_stream(path, buffer)

        # Reset the buffer position and read the contents
        buffer.seek(0)
        content = buffer.read()

        return FilesystemResult.success_result(data=content)

    @_with_resilience("write_text")
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        path = self._normalize_path(path)

        # Create a BytesIO object with the content
        buffer = io.BytesIO(content.encode(encoding))

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(path)
        if parent_dir:
            try:
                await self.client.make_directory(parent_dir, recursive=True)
            except (aioftp.errors.PathAlreadyExistsError, aioftp.errors.PermissionError):
                # Ignore if the directory already exists or we don't have permission
                pass

        # Upload the buffer to the file
        await self.client.upload_stream(buffer, path)

        return FilesystemResult.success_result()

    @_with_resilience("write_bytes")
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        path = self._normalize_path(path)

        # Create a BytesIO object with the content
        buffer = io.BytesIO(content)

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(path)
        if parent_dir:
            try:
                await self.client.make_directory(parent_dir, recursive=True)
            except (aioftp.errors.PathAlreadyExistsError, aioftp.errors.PermissionError):
                # Ignore if the directory already exists or we don't have permission
                pass

        # Upload the buffer to the file
        await self.client.upload_stream(buffer, path)

        return FilesystemResult.success_result()

    @_with_resilience("append_text")
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        # FTP doesn't support direct append operations, so we need to download, append, and upload
        path = self._normalize_path(path)

        # Check if file exists
        try:
            await self.client.stat(path)
            # File exists, download it first
            buffer = io.BytesIO()
            await self.client.download_stream(path, buffer)
            buffer.seek(0)
            existing_content = buffer.read().decode(encoding)
            new_content = existing_content + content

            # Write the combined content back
            write_buffer = io.BytesIO(new_content.encode(encoding))
            await self.client.upload_stream(write_buffer, path)
            return FilesystemResult.success_result()
        except aioftp.errors.PathNotFoundError:
            # File doesn't exist, just write the content
            buffer = io.BytesIO(content.encode(encoding))

            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(path)
            if parent_dir:
                try:
                    await self.client.make_directory(parent_dir, recursive=True)
                except (aioftp.errors.PathAlreadyExistsError, aioftp.errors.PermissionError):
                    pass

            await self.client.upload_stream(buffer, path)
            return FilesystemResult.success_result()

    @_with_resilience("append_bytes")
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        # FTP doesn't support direct append operations, so we need to download, append, and upload
        path = self._normalize_path(path)

        # Check if file exists
        try:
            await self.client.stat(path)
            # File exists, download it first
            buffer = io.BytesIO()
            await self.client.download_stream(path, buffer)
            buffer.seek(0)
            existing_content = buffer.read()
            new_content = existing_content + content

            # Write the combined content back
            write_buffer = io.BytesIO(new_content)
            await self.client.upload_stream(write_buffer, path)
            return FilesystemResult.success_result()
        except aioftp.errors.PathNotFoundError:
            # File doesn't exist, just write the content
            buffer = io.BytesIO(content)

            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(path)
            if parent_dir:
                try:
                    await self.client.make_directory(parent_dir, recursive=True)
                except (aioftp.errors.PathAlreadyExistsError, aioftp.errors.PermissionError):
                    pass

            await self.client.upload_stream(buffer, path)
            return FilesystemResult.success_result()

    @_with_resilience("delete_file")
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        path = self._normalize_path(path)

        # Check if the path exists and is a file
        file_info = await self.client.stat(path)
        if file_info.get("type", "") != "file":
            return FilesystemResult.error_result(FilesystemError(f"Not a file: {path}"))

        # Delete the file
        await self.client.remove_file(path)

        return FilesystemResult.success_result()

    @_with_resilience("create_dir")
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        path = self._normalize_path(path)

        # Check if the directory already exists
        if exist_ok:
            try:
                file_info = await self.client.stat(path)
                if file_info.get("type", "") == "dir":
                    return FilesystemResult.success_result()
            except aioftp.errors.PathNotFoundError:
                pass

        # Create the directory
        try:
            await self.client.make_directory(path, recursive=True)
            return FilesystemResult.success_result()
        except aioftp.errors.PathAlreadyExistsError:
            if exist_ok:
                return FilesystemResult.success_result()
            return FilesystemResult.error_result(FileExistsError(f"Directory already exists: {path}"))

    @_with_resilience("list_dir")
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        path = self._normalize_path(path)

        # List the directory contents
        files = await self.client.list(path)

        # Extract just the names
        names = [file.name for file in files]

        return FilesystemResult.success_result(data=names)

    @_with_resilience("delete_dir")
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        path = self._normalize_path(path)

        # Check if the path exists and is a directory
        file_info = await self.client.stat(path)
        if file_info.get("type", "") != "dir":
            return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))

        if recursive:
            # List all files and directories
            files = await self.client.list(path)

            # Delete all files and subdirectories
            for file in files:
                item_path = f"{path}/{file.name}"

                # Check if it's a file or directory
                item_info = await self.client.stat(item_path)
                if item_info.get("type", "") == "dir":
                    # Recursively delete subdirectory
                    await self.delete_dir(item_path, recursive=True)
                else:
                    # Delete file
                    await self.client.remove_file(item_path)

        # Delete the directory
        await self.client.remove_directory(path)

        return FilesystemResult.success_result()

    @_with_resilience("get_info")
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        path = self._normalize_path(path)

        # Get file info
        file_info = await self.client.stat(path)

        # Convert to FileInfo
        info = FileInfo(
            name=os.path.basename(path),
            path=path,
            size=file_info.get("size", 0),
            is_file=file_info.get("type", "") == "file",
            is_dir=file_info.get("type", "") == "dir",
            created_time=file_info.get("create", 0),
            modified_time=file_info.get("modify", 0),
            accessed_time=file_info.get("modify", 0),  # FTP doesn't provide access time
            metadata=file_info
        )

        return FilesystemResult.success_result(data=info)

    @_with_resilience("exists")
    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        path = self._normalize_path(path)

        # Try to get file info
        try:
            await self.client.stat(path)
            return FilesystemResult.success_result(data=True)
        except aioftp.errors.PathNotFoundError:
            return FilesystemResult.success_result(data=False)

    @_with_resilience("is_file")
    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        path = self._normalize_path(path)

        try:
            file_info = await self.client.stat(path)
            return FilesystemResult.success_result(data=file_info.get("type", "") == "file")
        except aioftp.errors.PathNotFoundError:
            return FilesystemResult.success_result(data=False)

    @_with_resilience("is_dir")
    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        path = self._normalize_path(path)

        try:
            file_info = await self.client.stat(path)
            return FilesystemResult.success_result(data=file_info.get("type", "") == "dir")
        except aioftp.errors.PathNotFoundError:
            return FilesystemResult.success_result(data=False)

    @_with_resilience("copy")
    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        # FTP doesn't have a native copy command, so we need to download and upload
        src_path = self._normalize_path(src_path)
        dst_path = self._normalize_path(dst_path)

        # Check if source exists
        try:
            src_info = await self.client.stat(src_path)
        except aioftp.errors.PathNotFoundError:
            return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))

        # Check if destination exists
        try:
            await self.client.stat(dst_path)
            if not overwrite:
                return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
        except aioftp.errors.PathNotFoundError:
            pass

        # Check if source is a file or directory
        if src_info.get("type", "") == "file":
            # Source is a file, copy it
            buffer = io.BytesIO()
            await self.client.download_stream(src_path, buffer)
            buffer.seek(0)

            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(dst_path)
            if parent_dir:
                try:
                    await self.client.make_directory(parent_dir, recursive=True)
                except (aioftp.errors.PathAlreadyExistsError, aioftp.errors.PermissionError):
                    pass

            await self.client.upload_stream(buffer, dst_path)
            return FilesystemResult.success_result()
        else:
            # Source is a directory, copy it recursively

            # Create destination directory
            try:
                await self.client.make_directory(dst_path, recursive=True)
            except aioftp.errors.PathAlreadyExistsError:
                pass

            # List source directory
            files = await self.client.list(src_path)

            # Copy each item in the directory
            for file in files:
                src_item_path = f"{src_path}/{file.name}"
                dst_item_path = f"{dst_path}/{file.name}"

                await self.copy(src_item_path, dst_item_path, overwrite)

            return FilesystemResult.success_result()

    @_with_resilience("move")
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        # FTP doesn't have a native move command, so we need to copy and delete
        src_path = self._normalize_path(src_path)
        dst_path = self._normalize_path(dst_path)

        # Copy the source to the destination
        copy_result = await self.copy(src_path, dst_path, overwrite)
        if not copy_result.success:
            return copy_result

        # Delete the source
        try:
            src_info = await self.client.stat(src_path)
            if src_info.get("type", "") == "file":
                await self.client.remove_file(src_path)
            else:
                # Recursively delete the directory
                await self.delete_dir(src_path, recursive=True)

            return FilesystemResult.success_result()
        except Exception as e:
            logger.error(f"Failed to delete source after move: {src_path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Move partially completed, but failed to delete source: {e}"))

    @_with_resilience("get_size")
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        path = self._normalize_path(path)

        # Get info for the path
        file_info = await self.client.stat(path)

        if file_info.get("type", "") == "file":
            # Path is a file, return its size
            return FilesystemResult.success_result(data=file_info.get("size", 0))
        else:
            # Path is a directory, calculate total size recursively
            total_size = 0

            # List directory contents
            files = await self.client.list(path)

            # Calculate size for each item
            for file in files:
                item_path = f"{path}/{file.name}"
                item_info = await self.client.stat(item_path)

                if item_info.get("type", "") == "file":
                    total_size += item_info.get("size", 0)
                else:
                    # Recursively get size of subdirectory
                    size_result = await self.get_size(item_path)
                    if size_result.success:
                        total_size += size_result.data

            return FilesystemResult.success_result(data=total_size)

    @_with_resilience("get_modified_time")
    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        path = self._normalize_path(path)

        # Get file info
        file_info = await self.client.stat(path)

        # Return the modified time
        return FilesystemResult.success_result(data=file_info.get("modify", 0))

    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        # FTP doesn't support setting modification times directly
        # Suppress unused parameter warnings
        _ = path, mtime
        return FilesystemResult.error_result("Setting modification time is not supported for FTP")

    @_with_resilience("walk")
    async def walk(self, path: str) -> FilesystemResult[List[tuple]]:
        """Walk a directory tree."""
        path = self._normalize_path(path)

        # Check if the path exists and is a directory
        file_info = await self.client.stat(path)
        if file_info.get("type", "") != "dir":
            return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))

        # Walk the directory tree
        walk_result = []
        await self._walk_recursive(path, "", walk_result)

        return FilesystemResult.success_result(data=walk_result)

    async def _walk_recursive(self, base_path: str, rel_path: str, result: List[Tuple[str, List[str], List[str]]]):
        """Recursively walk a directory tree."""
        # Get the full path
        full_path = base_path
        if rel_path:
            full_path = f"{base_path}/{rel_path}"

        try:
            # List directory contents
            files = await self.client.list(full_path)

            # Separate directories and files
            dirs = []
            file_names = []

            for file in files:
                item_path = f"{full_path}/{file.name}"

                try:
                    item_info = await self.client.stat(item_path)
                    if item_info.get("type", "") == "dir":
                        dirs.append(file.name)
                    else:
                        file_names.append(file.name)
                except Exception:
                    # Skip items we can't stat
                    continue

            # Add this directory to the result
            result.append((rel_path, dirs, file_names))

            # Recursively walk subdirectories
            for dir_name in dirs:
                new_rel_path = dir_name
                if rel_path:
                    new_rel_path = f"{rel_path}/{dir_name}"

                await self._walk_recursive(base_path, new_rel_path, result)
        except Exception as e:
            logger.warning(f"Error walking directory {full_path}: {e}")
            # Continue with other directories even if one fails
