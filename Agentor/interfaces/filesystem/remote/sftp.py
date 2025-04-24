"""
SFTP filesystem operations for the Agentor framework.

This module provides interfaces for interacting with SFTP servers.
"""

import os
import io
import asyncio
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
import time
import stat
import asyncssh

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from .base import RemoteFilesystem
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class SFTPFilesystem(RemoteFilesystem):
    """Interface for SFTP filesystem operations."""

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

    async def connect(self) -> FilesystemResult:
        """Connect to the SFTP server."""
        try:
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
                    return FilesystemResult.error_result(f"Failed to load private key: {e}")
            
            # Add known hosts if specified
            if self.known_hosts_path:
                try:
                    options['known_hosts'] = self.known_hosts_path
                except Exception as e:
                    return FilesystemResult.error_result(f"Failed to load known hosts: {e}")
            else:
                # Disable host key checking if no known hosts file is specified
                options['known_hosts'] = None
            
            # Connect to the server
            self.conn = await asyncssh.connect(**options)
            self.client = await self.conn.start_sftp_client()
            
            # Change to root directory if specified
            if self.root_dir:
                try:
                    await self.client.chdir(self.root_dir)
                except (asyncssh.SFTPError, OSError) as e:
                    if 'No such file' in str(e):
                        # Try to create the root directory if it doesn't exist
                        try:
                            await self.client.mkdir(self.root_dir)
                            await self.client.chdir(self.root_dir)
                        except Exception as e:
                            await self.disconnect()
                            return FilesystemResult.error_result(f"Failed to create or change to root directory: {e}")
                    else:
                        await self.disconnect()
                        return FilesystemResult.error_result(f"Failed to change to root directory: {e}")
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to SFTP server: {self.host}:{self.port}")
            return FilesystemResult.success_result()
        except asyncssh.DisconnectError as e:
            logger.error(f"Disconnected from SFTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Disconnected from SFTP server: {e}"))
        except asyncssh.PermissionDenied as e:
            logger.error(f"Permission denied for SFTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied for SFTP server: {e}"))
        except asyncssh.HostKeyNotVerifiable as e:
            logger.error(f"Host key not verifiable for SFTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Host key not verifiable for SFTP server: {e}"))
        except Exception as e:
            logger.error(f"Failed to connect to SFTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to connect to SFTP server: {e}"))

    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the SFTP server."""
        if not self.connected:
            return FilesystemResult.success_result()
        
        try:
            if self.client:
                self.client.close()
                self.client = None
            
            if self.conn:
                self.conn.close()
                self.conn = None
            
            self.connected = False
            logger.info(f"Disconnected from SFTP server: {self.host}:{self.port}")
            return FilesystemResult.success_result()
        except Exception as e:
            logger.error(f"Failed to disconnect from SFTP server: {self.host}:{self.port}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to disconnect from SFTP server: {e}"))

    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(self._normalize_path(path))
            
            async with self.client.open(full_path, 'r', encoding=encoding) as file:
                content = await file.read()
            
            return FilesystemResult.success_result(data=content)
        except asyncssh.SFTPNoSuchFile:
            logger.error(f"File not found: {path}")
            return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
        except asyncssh.SFTPPermissionDenied:
            logger.error(f"Permission denied: {path}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to read text from file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to read text from file: {e}"))

    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(self._normalize_path(path))
            
            async with self.client.open(full_path, 'rb') as file:
                content = await file.read()
            
            return FilesystemResult.success_result(data=content)
        except asyncssh.SFTPNoSuchFile:
            logger.error(f"File not found: {path}")
            return FilesystemResult.error_result(FileNotFoundError(f"File not found: {path}"))
        except asyncssh.SFTPPermissionDenied:
            logger.error(f"Permission denied: {path}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to read bytes from file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to read bytes from file: {e}"))

    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
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
        except asyncssh.SFTPPermissionDenied:
            logger.error(f"Permission denied: {path}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to write text to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to write text to file: {e}"))

    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
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
        except asyncssh.SFTPPermissionDenied:
            logger.error(f"Permission denied: {path}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to write bytes to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to write bytes to file: {e}"))

    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(self._normalize_path(path))
            
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(full_path)
            if parent_dir:
                try:
                    await self.client.mkdir(parent_dir, parents=True)
                except (asyncssh.SFTPFailure, asyncssh.SFTPPermissionDenied):
                    # Ignore if the directory already exists or we don't have permission
                    pass
            
            async with self.client.open(full_path, 'a', encoding=encoding) as file:
                await file.write(content)
            
            return FilesystemResult.success_result()
        except asyncssh.SFTPPermissionDenied:
            logger.error(f"Permission denied: {path}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to append text to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to append text to file: {e}"))

    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
            full_path = self._get_full_path(self._normalize_path(path))
            
            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(full_path)
            if parent_dir:
                try:
                    await self.client.mkdir(parent_dir, parents=True)
                except (asyncssh.SFTPFailure, asyncssh.SFTPPermissionDenied):
                    # Ignore if the directory already exists or we don't have permission
                    pass
            
            async with self.client.open(full_path, 'ab') as file:
                await file.write(content)
            
            return FilesystemResult.success_result()
        except asyncssh.SFTPPermissionDenied:
            logger.error(f"Permission denied: {path}")
            return FilesystemResult.error_result(PermissionError(f"Permission denied: {path}"))
        except Exception as e:
            logger.error(f"Failed to append bytes to file: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to append bytes to file: {e}"))
