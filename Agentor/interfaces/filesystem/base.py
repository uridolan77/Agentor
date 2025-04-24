"""
Base classes and interfaces for filesystem operations in the Agentor framework.

This module provides the base classes and interfaces for filesystem operations,
including error handling and result types.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, BinaryIO, TextIO, Iterator
import os
import time
from pathlib import Path

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class FilesystemError(Exception):
    """Base exception for filesystem errors."""
    pass


class FileNotFoundError(FilesystemError):
    """Exception raised when a file is not found."""
    pass


class PermissionError(FilesystemError):
    """Exception raised when permission is denied."""
    pass


class FileExistsError(FilesystemError):
    """Exception raised when a file already exists."""
    pass


class FilesystemResult(Generic[T]):
    """Result of a filesystem operation."""

    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        if self.success:
            return f"FilesystemResult(success={self.success})"
        else:
            return f"FilesystemResult(success={self.success}, error={self.error})"

    @classmethod
    def success_result(cls, data: Optional[T] = None, metadata: Optional[Dict[str, Any]] = None) -> 'FilesystemResult[T]':
        """Create a successful filesystem result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_result(cls, error: Union[Exception, str], metadata: Optional[Dict[str, Any]] = None) -> 'FilesystemResult[T]':
        """Create an error filesystem result."""
        if isinstance(error, str):
            error = FilesystemError(error)
        return cls(success=False, error=error, metadata=metadata)


class FileMode(Enum):
    """File open modes."""
    READ = "r"
    WRITE = "w"
    APPEND = "a"
    READ_BINARY = "rb"
    WRITE_BINARY = "wb"
    APPEND_BINARY = "ab"
    READ_PLUS = "r+"
    WRITE_PLUS = "w+"
    APPEND_PLUS = "a+"
    READ_BINARY_PLUS = "rb+"
    WRITE_BINARY_PLUS = "wb+"
    APPEND_BINARY_PLUS = "ab+"


class FileInfo:
    """Information about a file."""

    def __init__(
        self,
        name: str,
        path: str,
        size: int,
        is_file: bool,
        is_dir: bool,
        created_time: float,
        modified_time: float,
        accessed_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.path = path
        self.size = size
        self.is_file = is_file
        self.is_dir = is_dir
        self.created_time = created_time
        self.modified_time = modified_time
        self.accessed_time = accessed_time
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return f"FileInfo(name={self.name}, path={self.path}, size={self.size}, is_file={self.is_file}, is_dir={self.is_dir})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "is_file": self.is_file,
            "is_dir": self.is_dir,
            "created_time": self.created_time,
            "modified_time": self.modified_time,
            "accessed_time": self.accessed_time,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileInfo':
        """Create from a dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            size=data["size"],
            is_file=data["is_file"],
            is_dir=data["is_dir"],
            created_time=data["created_time"],
            modified_time=data["modified_time"],
            accessed_time=data["accessed_time"],
            metadata=data.get("metadata", {})
        )


class FilesystemInterface(ABC):
    """Interface for filesystem operations."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.connected = False
        self.connection_params = kwargs
        self.last_activity = time.time()

    @abstractmethod
    async def connect(self) -> FilesystemResult:
        """Connect to the filesystem."""
        pass

    @abstractmethod
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the filesystem."""
        pass

    @abstractmethod
    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        pass

    @abstractmethod
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        pass

    @abstractmethod
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        pass

    @abstractmethod
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        pass

    @abstractmethod
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        pass

    @abstractmethod
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        pass

    @abstractmethod
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        pass

    @abstractmethod
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        pass

    @abstractmethod
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        pass

    @abstractmethod
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        pass

    @abstractmethod
    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        pass

    @abstractmethod
    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        pass

    @abstractmethod
    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        pass

    @abstractmethod
    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        pass

    @abstractmethod
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        pass

    @abstractmethod
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        pass

    @abstractmethod
    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        pass

    @abstractmethod
    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        pass

    @abstractmethod
    async def walk(self, path: str) -> FilesystemResult[List[tuple]]:
        """Walk a directory tree."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, connected={self.connected})"
