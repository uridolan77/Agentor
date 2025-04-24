"""
Base class for remote filesystem operations in the Agentor framework.

This module provides the base class for remote filesystem operations,
including FTP, SFTP, and WebDAV.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import os
import time

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class RemoteFilesystem(FilesystemInterface, ABC):
    """Base class for remote filesystem operations."""

    def __init__(
        self,
        name: str,
        host: str,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        root_dir: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.root_dir = root_dir
        self.timeout = timeout
        self.client = None
        self.connected = False
        self.last_activity = time.time()

    def _get_full_path(self, path: str) -> str:
        """Get the full path by joining with the root directory if specified."""
        if self.root_dir:
            # Use posix-style path joining for remote filesystems
            if path.startswith('/'):
                path = path[1:]
            return f"{self.root_dir.rstrip('/')}/{path}"
        return path

    def _normalize_path(self, path: str) -> str:
        """Normalize a path for the remote filesystem."""
        # Convert backslashes to forward slashes
        path = path.replace('\\', '/')
        # Remove leading/trailing whitespace
        path = path.strip()
        # Remove duplicate slashes
        while '//' in path:
            path = path.replace('//', '/')
        return path

    async def _ensure_connected(self) -> FilesystemResult:
        """Ensure that we are connected to the remote filesystem."""
        if not self.connected or not self.client:
            return await self.connect()
        return FilesystemResult.success_result()

    @abstractmethod
    async def connect(self) -> FilesystemResult:
        """Connect to the remote filesystem."""
        pass

    @abstractmethod
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the remote filesystem."""
        pass
