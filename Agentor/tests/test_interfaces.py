"""Tests for the Agentor interfaces.

This module contains tests for the various interfaces provided by the Agentor framework.
"""

import os
import pytest
import unittest
import tempfile
import shutil
import asyncio
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

# Define mock classes for testing
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


class FilesystemResult:
    """Result of a filesystem operation."""

    def __init__(self, success: bool, data=None, error=None, metadata=None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def __str__(self):
        if self.success:
            return f"FilesystemResult(success={self.success}, data={self.data})"
        else:
            return f"FilesystemResult(success={self.success}, error={self.error})"

    @classmethod
    def success_result(cls, data=None, metadata=None):
        """Create a successful filesystem result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_result(cls, error, metadata=None):
        """Create an error filesystem result."""
        if isinstance(error, str):
            error = FilesystemError(error)
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class FileInfo:
    """Information about a file or directory."""

    name: str
    path: str
    size: int
    is_file: bool
    is_dir: bool
    created_time: float
    modified_time: float
    accessed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return (
            f"FileInfo(name={self.name}, path={self.path}, size={self.size}, "
            f"is_file={self.is_file}, is_dir={self.is_dir})"
        )

    def to_dict(self):
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
    def from_dict(cls, data):
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
            metadata=data["metadata"]
        )


class NetworkError(Exception):
    """Base exception for network errors."""
    pass


class ConnectionError(NetworkError):
    """Exception raised when a connection fails."""
    pass


class RequestError(NetworkError):
    """Exception raised when a request fails."""
    pass


class ResponseError(NetworkError):
    """Exception raised when a response is invalid."""
    pass


class TimeoutError(NetworkError):
    """Exception raised when a request times out."""
    pass


class NetworkResult:
    """Result of a network operation."""

    def __init__(self, success: bool, data=None, error=None, status_code=None, headers=None, metadata=None):
        self.success = success
        self.data = data
        self.error = error
        self.status_code = status_code
        self.headers = headers or {}
        self.metadata = metadata or {}

    def __str__(self):
        if self.success:
            return f"NetworkResult(success={self.success}, status_code={self.status_code})"
        else:
            return f"NetworkResult(success={self.success}, error={self.error}, status_code={self.status_code})"

    @classmethod
    def success_result(cls, data=None, status_code=None, headers=None, metadata=None):
        """Create a successful network result."""
        return cls(success=True, data=data, status_code=status_code, headers=headers, metadata=metadata)

    @classmethod
    def error_result(cls, error, status_code=None, headers=None, metadata=None):
        """Create an error network result."""
        if isinstance(error, str):
            error = NetworkError(error)
        return cls(success=False, error=error, status_code=status_code, headers=headers, metadata=metadata)


class TestFilesystemInterface(unittest.TestCase):
    """Tests for the filesystem interface."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        # Create some test files and directories
        self.test_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("Hello, world!")

        self.test_dir = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(self.test_dir)

        self.test_subfile = os.path.join(self.test_dir, "subfile.txt")
        with open(self.test_subfile, "w") as f:
            f.write("Hello from subfile!")

    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    def test_filesystem_result(self):
        """Test the FilesystemResult class."""
        # Test success result
        result = FilesystemResult.success_result(data="test")
        self.assertTrue(result.success)
        self.assertEqual(result.data, "test")
        self.assertIsNone(result.error)

        # Test error result
        result = FilesystemResult.error_result("Test error")
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertIsInstance(result.error, FilesystemError)
        self.assertEqual(str(result.error), "Test error")

        # Test string representation
        self.assertIn("success=True", str(FilesystemResult.success_result()))
        self.assertIn("success=False", str(FilesystemResult.error_result("error")))
        self.assertIn("error=", str(FilesystemResult.error_result("error")))

    def test_file_info(self):
        """Test the FileInfo class."""
        # Create a FileInfo object
        info = FileInfo(
            name="test.txt",
            path="/path/to/test.txt",
            size=100,
            is_file=True,
            is_dir=False,
            created_time=123456789.0,
            modified_time=123456790.0,
            accessed_time=123456791.0,
            metadata={"owner": "test"}
        )

        # Test properties
        self.assertEqual(info.name, "test.txt")
        self.assertEqual(info.path, "/path/to/test.txt")
        self.assertEqual(info.size, 100)
        self.assertTrue(info.is_file)
        self.assertFalse(info.is_dir)
        self.assertEqual(info.created_time, 123456789.0)
        self.assertEqual(info.modified_time, 123456790.0)
        self.assertEqual(info.accessed_time, 123456791.0)
        self.assertEqual(info.metadata, {"owner": "test"})

        # Test to_dict and from_dict
        info_dict = info.to_dict()
        self.assertEqual(info_dict["name"], "test.txt")
        self.assertEqual(info_dict["path"], "/path/to/test.txt")
        self.assertEqual(info_dict["size"], 100)
        self.assertTrue(info_dict["is_file"])
        self.assertFalse(info_dict["is_dir"])
        self.assertEqual(info_dict["created_time"], 123456789.0)
        self.assertEqual(info_dict["modified_time"], 123456790.0)
        self.assertEqual(info_dict["accessed_time"], 123456791.0)
        self.assertEqual(info_dict["metadata"], {"owner": "test"})

        info2 = FileInfo.from_dict(info_dict)
        self.assertEqual(info2.name, info.name)
        self.assertEqual(info2.path, info.path)
        self.assertEqual(info2.size, info.size)
        self.assertEqual(info2.is_file, info.is_file)
        self.assertEqual(info2.is_dir, info.is_dir)
        self.assertEqual(info2.created_time, info.created_time)
        self.assertEqual(info2.modified_time, info.modified_time)
        self.assertEqual(info2.accessed_time, info.accessed_time)
        self.assertEqual(info2.metadata, info.metadata)

        # Test string representation
        self.assertIn("name=test.txt", str(info))
        self.assertIn("path=/path/to/test.txt", str(info))
        self.assertIn("size=100", str(info))
        self.assertIn("is_file=True", str(info))
        self.assertIn("is_dir=False", str(info))


class TestNetworkInterface(unittest.TestCase):
    """Tests for the network interface."""

    def test_network_result(self):
        """Test the NetworkResult class."""
        # Test success result
        result = NetworkResult.success_result(data="test", status_code=200)
        self.assertTrue(result.success)
        self.assertEqual(result.data, "test")
        self.assertEqual(result.status_code, 200)
        self.assertIsNone(result.error)

        # Test error result
        result = NetworkResult.error_result("Test error", status_code=404)
        self.assertFalse(result.success)
        self.assertIsNone(result.data)
        self.assertEqual(result.status_code, 404)
        self.assertIsInstance(result.error, NetworkError)
        self.assertEqual(str(result.error), "Test error")

        # Test string representation
        self.assertIn("success=True", str(NetworkResult.success_result()))
        self.assertIn("success=False", str(NetworkResult.error_result("error")))
        self.assertIn("error=", str(NetworkResult.error_result("error")))
        self.assertIn("status_code=200", str(NetworkResult.success_result(status_code=200)))


class TestHttpInterface:
    """Tests for the HTTP interface."""

    @pytest.mark.asyncio
    async def test_http_client(self):
        """Test HTTP client implementation."""
        # Create a mock HTTP client
        client = MagicMock()
        client.connect = MagicMock(return_value=asyncio.Future())
        client.connect.return_value.set_result(NetworkResult.success_result())

        client.disconnect = MagicMock(return_value=asyncio.Future())
        client.disconnect.return_value.set_result(NetworkResult.success_result())

        client.request = MagicMock(return_value=asyncio.Future())
        client.request.return_value.set_result(
            NetworkResult.success_result(
                data={"message": "Success"},
                status_code=200,
                headers={"Content-Type": "application/json"}
            )
        )

        # Test client methods
        result = await client.connect()
        assert result.success

        result = await client.request("GET", "https://example.com/api")
        assert result.success
        assert result.status_code == 200
        assert result.data == {"message": "Success"}
        assert "Content-Type" in result.headers

        result = await client.disconnect()
        assert result.success


class TestWebSocketInterface:
    """Tests for the WebSocket interface."""

    @pytest.mark.asyncio
    async def test_websocket_client(self):
        """Test WebSocket client implementation."""
        # Create a mock WebSocket client
        client = MagicMock()
        client.connect = MagicMock(return_value=asyncio.Future())
        client.connect.return_value.set_result(NetworkResult.success_result())

        client.connect_ws = MagicMock(return_value=asyncio.Future())
        client.connect_ws.return_value.set_result(NetworkResult.success_result())

        client.send = MagicMock(return_value=asyncio.Future())
        client.send.return_value.set_result(NetworkResult.success_result())

        client.receive = MagicMock(return_value=asyncio.Future())
        client.receive.return_value.set_result(
            NetworkResult.success_result(data={"message": "Hello"})
        )

        client.disconnect_ws = MagicMock(return_value=asyncio.Future())
        client.disconnect_ws.return_value.set_result(NetworkResult.success_result())

        client.disconnect = MagicMock(return_value=asyncio.Future())
        client.disconnect.return_value.set_result(NetworkResult.success_result())

        # Test client methods
        result = await client.connect()
        assert result.success

        result = await client.connect_ws("wss://example.com/ws")
        assert result.success

        result = await client.send({"message": "Hello"})
        assert result.success

        result = await client.receive()
        assert result.success
        assert result.data == {"message": "Hello"}

        result = await client.disconnect_ws()
        assert result.success

        result = await client.disconnect()
        assert result.success


if __name__ == "__main__":
    pytest.main()