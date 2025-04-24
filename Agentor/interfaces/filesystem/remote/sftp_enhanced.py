"""
Enhanced SFTP filesystem operations for the Agentor framework.

This module provides an enhanced SFTP filesystem implementation with resilience patterns.
"""

import os
import io
import asyncio
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Tuple
import time
import stat
import asyncssh
from functools import wraps

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from .base import RemoteFilesystem
from agentor.utils.logging import get_logger
from agentor.llm_gateway.utils.retry import RetryConfig, RetryStrategy, retry_async
from agentor.llm_gateway.utils.timeout import with_timeout, TimeoutStrategy
from agentor.llm_gateway.utils.circuit_breaker import CircuitBreaker
from agentor.core.config import get_typed_config
from ..config import SFTPFilesystemConfig

logger = get_logger(__name__)


class SFTPEnhancedFilesystem(RemoteFilesystem):
    """Enhanced interface for SFTP filesystem operations with resilience patterns."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int = 22,
        username: str = None,
        password: str = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        known_hosts_path: Optional[str] = None,
        root_dir: Optional[str] = None,
        timeout: int = 30,
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
        self.private_key_path = private_key_path
        self.private_key_passphrase = private_key_passphrase
        self.known_hosts_path = known_hosts_path
        self.conn = None
        self.client = None
        
        # Load configuration or use defaults
        try:
            self.config = get_typed_config(SFTPFilesystemConfig)
        except Exception as e:
            logger.warning(f"Failed to load SFTP configuration, using defaults: {e}")
            self.config = SFTPFilesystemConfig(host=host, username=username or "")
            
        # Set up retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.config.retry_max_attempts,
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=self.config.retry_base_delay,
            max_delay=self.config.retry_max_delay,
            jitter=self.config.retry_jitter,
            retry_on=[ConnectionError, TimeoutError, asyncio.TimeoutError]
        )
        
        # Set up circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"sftp-{host}-{port}",
            failure_threshold=self.config.circuit_breaker_failures,
            recovery_timeout=self.config.circuit_breaker_recovery,
            half_open_max_calls=self.config.circuit_breaker_half_open_calls
        )

    @with_timeout("sftp", "connect", strategy=TimeoutStrategy.FIXED)
    async def connect(self) -> FilesystemResult:
        """Connect to the SFTP server."""
        async with self.circuit_breaker:
            try:
                # Use retry for connection attempts
                async def _connect_attempt():
                    # Prepare connection options
                    options = {
                        'host': self.host,
                        'port': self.port,
                        'username': self.username,
                        'password': self.password,
                        'connect_timeout': self.timeout
                    }
                    
                    # Add private key if specified
                    if self.private_key_path:
                        try:
                            options['client_keys'] = [self.private_key_path]
                            if self.private_key_passphrase:
                                options['passphrase'] = self.private_key_passphrase
                        except Exception as e:
                            raise Exception(f"Failed to load private key: {e}")
                    
                    # Add known hosts if specified
                    if self.known_hosts_path:
                        try:
                            options['known_hosts'] = self.known_hosts_path
                        except Exception as e:
                            raise Exception(f"Failed to load known hosts: {e}")
                    else:
                        # Disable host key checking if no known hosts file is specified
                        options['known_hosts'] = None
                    
                    # Connect to the server
                    self.conn = await asyncssh.connect(**options)
                    
                    # Open an SFTP client
                    self.client = await self.conn.start_sftp_client()
                    
                    # Change to root directory if specified
                    if self.root_dir:
                        try:
                            await self.client.chdir(self.root_dir)
                        except asyncssh.SFTPNoSuchFile:
                            # Try to create the root directory if it doesn't exist
                            try:
                                await self.client.mkdir(self.root_dir, parents=True)
                                await self.client.chdir(self.root_dir)
                            except Exception as e:
                                raise Exception(f"Failed to create or change to root directory: {e}")
                
                # Execute with retry
                await retry_async(
                    _connect_attempt,
                    "sftp",
                    "connect",
                    self.retry_config
                )
                
                self.connected = True
                self.last_activity = time.time()
                logger.info(f"Connected to SFTP server: {self.host}:{self.port}")
                return FilesystemResult.success_result()
            except Exception as e:
                if self.client:
                    self.client.close()
                    self.client = None
                if self.conn:
                    self.conn.close()
                    self.conn = None
                logger.error(f"Failed to connect to SFTP server: {self.host}:{self.port}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to connect to SFTP server: {e}"))

    @with_timeout("sftp", "disconnect", strategy=TimeoutStrategy.FIXED)
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the SFTP server."""
        if not self.connected or not self.client:
            return FilesystemResult.success_result()
        
        try:
            # Use retry for disconnection
            async def _disconnect_attempt():
                if self.client:
                    self.client.close()
                if self.conn:
                    self.conn.close()
            
            # Execute with retry
            await retry_async(
                _disconnect_attempt,
                "sftp",
                "disconnect",
                self.retry_config
            )
            
            self.connected = False
            self.client = None
            self.conn = None
            logger.info(f"Disconnected from SFTP server: {self.host}:{self.port}")
            return FilesystemResult.success_result()
        except Exception as e:
            # Even if there's an error, mark as disconnected
            self.connected = False
            self.client = None
            self.conn = None
            logger.error(f"Failed to disconnect from SFTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to disconnect from SFTP server: {e}"))

    async def _ensure_connected(self) -> FilesystemResult:
        """Ensure that we are connected to the SFTP server."""
        if not self.connected or not self.client:
            return await self.connect()
        return FilesystemResult.success_result()

    def _get_full_path(self, path: str) -> str:
        """Get the full path on the server."""
        # The root directory is already handled by the SFTP client
        return path

    # Helper decorator for SFTP operations with resilience patterns
    def _with_resilience(operation_name):
        """Decorator to add resilience patterns to SFTP operations."""
        def decorator(func):
            @wraps(func)
            @with_timeout("sftp", operation_name, strategy=TimeoutStrategy.ADAPTIVE)
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
                            "sftp",
                            operation_name,
                            self.retry_config
                        )
                    except asyncssh.SFTPNoSuchFile as e:
                        path = args[0] if args else kwargs.get('path', 'unknown')
                        logger.error(f"File not found: {path}")
                        return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
                    except asyncssh.SFTPPermissionDenied as e:
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
        full_path = self._get_full_path(self._normalize_path(path))
        
        async with self.client.open(full_path, 'r', encoding=encoding) as file:
            content = await file.read()
        
        return FilesystemResult.success_result(data=content)

    @_with_resilience("read_bytes")
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        async with self.client.open(full_path, 'rb') as file:
            content = await file.read()
        
        return FilesystemResult.success_result(data=content)

    @_with_resilience("write_text")
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(full_path)
        if parent_dir:
            try:
                await self.client.mkdir(parent_dir, parents=True)
            except (asyncssh.SFTPFailure, asyncssh.SFTPPermissionDenied):
                # Ignore if the directory already exists or we don't have permission
                pass
        
        async with self.client.open(full_path, 'w', encoding=encoding) as file:
            await file.write(content)
        
        return FilesystemResult.success_result()

    @_with_resilience("write_bytes")
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(full_path)
        if parent_dir:
            try:
                await self.client.mkdir(parent_dir, parents=True)
            except (asyncssh.SFTPFailure, asyncssh.SFTPPermissionDenied):
                # Ignore if the directory already exists or we don't have permission
                pass
        
        async with self.client.open(full_path, 'wb') as file:
            await file.write(content)
        
        return FilesystemResult.success_result()

    @_with_resilience("append_text")
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if file exists
        try:
            await self.client.stat(full_path)
            # File exists, open in append mode
            async with self.client.open(full_path, 'a', encoding=encoding) as file:
                await file.write(content)
            return FilesystemResult.success_result()
        except asyncssh.SFTPNoSuchFile:
            # File doesn't exist, create it
            return await self.write_text(path, content, encoding)

    @_with_resilience("append_bytes")
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if file exists
        try:
            await self.client.stat(full_path)
            # File exists, read it first
            existing_content = await self.read_bytes(path)
            if not existing_content.success:
                return existing_content
            
            # Write the combined content back
            return await self.write_bytes(path, existing_content.data + content)
        except asyncssh.SFTPNoSuchFile:
            # File doesn't exist, create it
            return await self.write_bytes(path, content)

    @_with_resilience("delete_file")
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a file
        attrs = await self.client.stat(full_path)
        if not stat.S_ISREG(attrs.permissions):
            return FilesystemResult.error_result(FilesystemError(f"Not a file: {path}"))
        
        # Delete the file
        await self.client.remove(full_path)
        
        return FilesystemResult.success_result()

    @_with_resilience("create_dir")
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
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
        try:
            await self.client.mkdir(full_path, parents=True)
            return FilesystemResult.success_result()
        except asyncssh.SFTPFailure as e:
            if exist_ok and "already exists" in str(e).lower():
                return FilesystemResult.success_result()
            return FilesystemResult.error_result(FileExistsError(f"Directory already exists: {path}"))

    @_with_resilience("list_dir")
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a directory
        attrs = await self.client.stat(full_path)
        if not stat.S_ISDIR(attrs.permissions):
            return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
        
        # List the directory contents
        files = await self.client.listdir(full_path)
        
        return FilesystemResult.success_result(data=files)

    @_with_resilience("delete_dir")
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a directory
        attrs = await self.client.stat(full_path)
        if not stat.S_ISDIR(attrs.permissions):
            return FilesystemResult.error_result(FilesystemError(f"Not a directory: {path}"))
        
        if recursive:
            # List all files and directories
            files = await self.client.listdir(full_path)
            
            # Delete all files and subdirectories
            for name in files:
                item_path = f"{path}/{name}"
                full_item_path = f"{full_path}/{name}"
                
                # Check if it's a file or directory
                try:
                    item_attrs = await self.client.stat(full_item_path)
                    if stat.S_ISDIR(item_attrs.permissions):
                        # Recursively delete subdirectory
                        await self.delete_dir(item_path, recursive=True)
                    else:
                        # Delete file
                        await self.client.remove(full_item_path)
                except Exception:
                    # Skip items we can't stat
                    continue
        
        # Delete the directory
        await self.client.rmdir(full_path)
        
        return FilesystemResult.success_result()

    @_with_resilience("get_info")
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get file info
        attrs = await self.client.stat(full_path)
        
        # Convert to FileInfo
        info = FileInfo(
            name=os.path.basename(path),
            path=path,
            size=attrs.size,
            is_file=stat.S_ISREG(attrs.permissions),
            is_dir=stat.S_ISDIR(attrs.permissions),
            created_time=attrs.atime,  # SFTP doesn't provide creation time, use access time
            modified_time=attrs.mtime,
            accessed_time=attrs.atime,
            metadata={
                "permissions": attrs.permissions,
                "uid": attrs.uid,
                "gid": attrs.gid
            }
        )
        
        return FilesystemResult.success_result(data=info)

    @_with_resilience("exists")
    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Try to get file info
        try:
            await self.client.stat(full_path)
            return FilesystemResult.success_result(data=True)
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.success_result(data=False)

    @_with_resilience("is_file")
    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        try:
            attrs = await self.client.stat(full_path)
            return FilesystemResult.success_result(data=stat.S_ISREG(attrs.permissions))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.success_result(data=False)

    @_with_resilience("is_dir")
    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        try:
            attrs = await self.client.stat(full_path)
            return FilesystemResult.success_result(data=stat.S_ISDIR(attrs.permissions))
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.success_result(data=False)

    @_with_resilience("copy")
    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        src_path = self._normalize_path(src_path)
        dst_path = self._normalize_path(dst_path)
        full_src_path = self._get_full_path(src_path)
        full_dst_path = self._get_full_path(dst_path)
        
        # Check if source exists
        try:
            src_attrs = await self.client.stat(full_src_path)
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))
        
        # Check if destination exists
        try:
            await self.client.stat(full_dst_path)
            if not overwrite:
                return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
        except asyncssh.SFTPNoSuchFile:
            pass
        
        # Check if source is a file or directory
        if stat.S_ISREG(src_attrs.permissions):
            # Source is a file, copy it
            content_result = await self.read_bytes(src_path)
            if not content_result.success:
                return content_result
            
            return await self.write_bytes(dst_path, content_result.data)
        else:
            # Source is a directory, copy it recursively
            
            # Create destination directory
            create_result = await self.create_dir(dst_path, exist_ok=True)
            if not create_result.success:
                return create_result
            
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

    @_with_resilience("move")
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        src_path = self._normalize_path(src_path)
        dst_path = self._normalize_path(dst_path)
        full_src_path = self._get_full_path(src_path)
        full_dst_path = self._get_full_path(dst_path)
        
        # Check if source exists
        try:
            await self.client.stat(full_src_path)
        except asyncssh.SFTPNoSuchFile:
            return FilesystemResult.error_result(FileNotFoundError(f"Source path not found: {src_path}"))
        
        # Check if destination exists
        try:
            await self.client.stat(full_dst_path)
            if not overwrite:
                return FilesystemResult.error_result(FileExistsError(f"Destination path already exists: {dst_path}"))
        except asyncssh.SFTPNoSuchFile:
            pass
        
        # Try to use rename first (most efficient)
        try:
            # If destination exists and overwrite is True, delete it first
            try:
                dst_attrs = await self.client.stat(full_dst_path)
                if overwrite:
                    if stat.S_ISREG(dst_attrs.permissions):
                        await self.client.remove(full_dst_path)
                    else:
                        await self.delete_dir(dst_path, recursive=True)
            except asyncssh.SFTPNoSuchFile:
                pass
            
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

    @_with_resilience("get_size")
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get info for the path
        attrs = await self.client.stat(full_path)
        
        if stat.S_ISREG(attrs.permissions):
            # Path is a file, return its size
            return FilesystemResult.success_result(data=attrs.size)
        else:
            # Path is a directory, calculate total size recursively
            total_size = 0
            
            # List directory contents
            list_result = await self.list_dir(path)
            if not list_result.success:
                return list_result
            
            # Calculate size for each item
            for name in list_result.data:
                item_path = f"{path}/{name}"
                
                size_result = await self.get_size(item_path)
                if size_result.success:
                    total_size += size_result.data
            
            return FilesystemResult.success_result(data=total_size)

    @_with_resilience("get_modified_time")
    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get file info
        attrs = await self.client.stat(full_path)
        
        # Return the modified time
        return FilesystemResult.success_result(data=attrs.mtime)

    @_with_resilience("set_modified_time")
    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Get current attributes
        attrs = await self.client.stat(full_path)
        
        # Create new attributes with the modified time
        new_attrs = asyncssh.SFTPAttrs(mtime=mtime, atime=attrs.atime)
        
        # Set the attributes
        await self.client.setstat(full_path, new_attrs)
        
        return FilesystemResult.success_result()

    @_with_resilience("walk")
    async def walk(self, path: str) -> FilesystemResult[List[Tuple[str, List[str], List[str]]]]:
        """Walk a directory tree."""
        full_path = self._get_full_path(self._normalize_path(path))
        
        # Check if the path exists and is a directory
        attrs = await self.client.stat(full_path)
        if not stat.S_ISDIR(attrs.permissions):
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
        except Exception as e:
            logger.warning(f"Error walking directory {full_path}: {e}")
            # Continue with other directories even if one fails
