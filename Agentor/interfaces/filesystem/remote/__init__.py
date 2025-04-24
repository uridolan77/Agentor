"""
Remote filesystem operations for the Agentor framework.

This module provides interfaces for interacting with remote filesystems,
such as FTP, SFTP, and WebDAV.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO
import time
import os

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo, FileMode,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Import SFTP implementations
from .sftp import SFTPFilesystem
from .sftp_enhanced import SFTPEnhancedFilesystem

# Import the SFTP operations to add them to the SFTPFilesystem class
# These imports are needed for their side effects (adding methods to SFTPFilesystem)
from . import sftp_operations
from . import sftp_advanced

__all__ = [
    'RemoteFilesystem',
    'SFTPFilesystem',
    'SFTPEnhancedFilesystem'
]


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
