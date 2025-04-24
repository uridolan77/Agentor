"""
Tests for the enhanced SFTP filesystem interface.

This module contains tests for the enhanced SFTP filesystem interface with resilience patterns.
"""

import os
import io
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import asyncssh
import stat

from agentor.interfaces.filesystem.remote.sftp_enhanced import SFTPEnhancedFilesystem
from agentor.interfaces.filesystem.base import FilesystemResult, FileInfo


@pytest.fixture
def mock_sftp_client():
    """Create a mock SFTP client."""
    client = AsyncMock()
    client.open = AsyncMock()
    client.close = AsyncMock()
    client.mkdir = AsyncMock()
    client.rmdir = AsyncMock()
    client.remove = AsyncMock()
    client.rename = AsyncMock()
    client.listdir = AsyncMock(return_value=[])
    client.chdir = AsyncMock()
    client.setstat = AsyncMock()
    
    # Mock stat method to return file info
    async def mock_stat(path):
        attrs = MagicMock()
        if path.endswith(".txt"):
            attrs.size = 100
            attrs.mtime = 1234567890
            attrs.atime = 1234567890
            attrs.permissions = stat.S_IFREG | 0o644  # Regular file
        else:
            attrs.size = 0
            attrs.mtime = 1234567890
            attrs.atime = 1234567890
            attrs.permissions = stat.S_IFDIR | 0o755  # Directory
        
        attrs.uid = 1000
        attrs.gid = 1000
        return attrs
    
    client.stat = AsyncMock(side_effect=mock_stat)
    
    return client


@pytest.fixture
def mock_ssh_connection():
    """Create a mock SSH connection."""
    conn = AsyncMock()
    conn.close = AsyncMock()
    return conn


@pytest.fixture
def sftp_filesystem(mock_sftp_client, mock_ssh_connection):
    """Create a SFTP filesystem with a mock client."""
    with patch("asyncssh.connect", return_value=mock_ssh_connection):
        # Mock the start_sftp_client method
        mock_ssh_connection.start_sftp_client = AsyncMock(return_value=mock_sftp_client)
        
        fs = SFTPEnhancedFilesystem(
            name="test-sftp",
            host="sftp.example.com",
            port=22,
            username="test",
            password="password",
            root_dir="/test",
            private_key_path=None,
            private_key_passphrase=None,
            known_hosts_path=None
        )
        yield fs


@pytest.mark.asyncio
async def test_connect(sftp_filesystem, mock_ssh_connection, mock_sftp_client):
    """Test connecting to the SFTP server."""
    result = await sftp_filesystem.connect()
    
    assert result.success
    mock_ssh_connection.start_sftp_client.assert_called_once()
    mock_sftp_client.chdir.assert_called_once_with("/test")


@pytest.mark.asyncio
async def test_connect_error(sftp_filesystem, mock_ssh_connection):
    """Test error handling when connecting to the SFTP server."""
    mock_ssh_connection.start_sftp_client.side_effect = Exception("Connection error")
    
    result = await sftp_filesystem.connect()
    
    assert not result.success
    assert "Connection error" in str(result.error)


@pytest.mark.asyncio
async def test_read_text(sftp_filesystem, mock_sftp_client):
    """Test reading text from a file."""
    # Mock the open method to return a file-like object
    mock_file = AsyncMock()
    mock_file.__aenter__ = AsyncMock(return_value=mock_file)
    mock_file.__aexit__ = AsyncMock(return_value=None)
    mock_file.read = AsyncMock(return_value="Hello, world!")
    mock_sftp_client.open.return_value = mock_file
    
    # Connect first
    await sftp_filesystem.connect()
    
    # Read the file
    result = await sftp_filesystem.read_text("test.txt")
    
    assert result.success
    assert result.data == "Hello, world!"
    mock_sftp_client.open.assert_called_once_with("test.txt", 'r', encoding="utf-8")


@pytest.mark.asyncio
async def test_read_text_file_not_found(sftp_filesystem, mock_sftp_client):
    """Test error handling when reading a non-existent file."""
    # Mock the stat method to raise SFTPNoSuchFile
    mock_sftp_client.stat.side_effect = asyncssh.SFTPNoSuchFile("File not found")
    
    # Connect first
    await sftp_filesystem.connect()
    
    # Read the file
    result = await sftp_filesystem.read_text("nonexistent.txt")
    
    assert not result.success
    assert "File not found" in str(result.error)


@pytest.mark.asyncio
async def test_write_text(sftp_filesystem, mock_sftp_client):
    """Test writing text to a file."""
    # Mock the open method to return a file-like object
    mock_file = AsyncMock()
    mock_file.__aenter__ = AsyncMock(return_value=mock_file)
    mock_file.__aexit__ = AsyncMock(return_value=None)
    mock_file.write = AsyncMock()
    mock_sftp_client.open.return_value = mock_file
    
    # Connect first
    await sftp_filesystem.connect()
    
    # Write the file
    result = await sftp_filesystem.write_text("test.txt", "Hello, world!")
    
    assert result.success
    mock_sftp_client.open.assert_called_once_with("test.txt", 'w', encoding="utf-8")
    mock_file.write.assert_called_once_with("Hello, world!")


@pytest.mark.asyncio
async def test_create_dir(sftp_filesystem, mock_sftp_client):
    """Test creating a directory."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Create the directory
    result = await sftp_filesystem.create_dir("test_dir")
    
    assert result.success
    mock_sftp_client.mkdir.assert_called_once_with("test_dir", parents=True)


@pytest.mark.asyncio
async def test_list_dir(sftp_filesystem, mock_sftp_client):
    """Test listing directory contents."""
    # Mock the listdir method to return files
    mock_sftp_client.listdir.return_value = ["test.txt", "test2.txt"]
    
    # Connect first
    await sftp_filesystem.connect()
    
    # List the directory
    result = await sftp_filesystem.list_dir("test_dir")
    
    assert result.success
    assert result.data == ["test.txt", "test2.txt"]
    mock_sftp_client.listdir.assert_called_once_with("test_dir")


@pytest.mark.asyncio
async def test_delete_file(sftp_filesystem, mock_sftp_client):
    """Test deleting a file."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Delete the file
    result = await sftp_filesystem.delete_file("test.txt")
    
    assert result.success
    mock_sftp_client.remove.assert_called_once_with("test.txt")


@pytest.mark.asyncio
async def test_delete_dir(sftp_filesystem, mock_sftp_client):
    """Test deleting a directory."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Delete the directory
    result = await sftp_filesystem.delete_dir("test_dir")
    
    assert result.success
    mock_sftp_client.rmdir.assert_called_once_with("test_dir")


@pytest.mark.asyncio
async def test_get_info(sftp_filesystem, mock_sftp_client):
    """Test getting file information."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Get file info
    result = await sftp_filesystem.get_info("test.txt")
    
    assert result.success
    assert result.data.name == "test.txt"
    assert result.data.is_file
    assert not result.data.is_dir
    assert result.data.size == 100
    assert result.data.modified_time == 1234567890


@pytest.mark.asyncio
async def test_exists(sftp_filesystem, mock_sftp_client):
    """Test checking if a file exists."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Check if file exists
    result = await sftp_filesystem.exists("test.txt")
    
    assert result.success
    assert result.data is True


@pytest.mark.asyncio
async def test_exists_not_found(sftp_filesystem, mock_sftp_client):
    """Test checking if a non-existent file exists."""
    # Mock the stat method to raise SFTPNoSuchFile
    mock_sftp_client.stat.side_effect = asyncssh.SFTPNoSuchFile("File not found")
    
    # Connect first
    await sftp_filesystem.connect()
    
    # Check if file exists
    result = await sftp_filesystem.exists("nonexistent.txt")
    
    assert result.success
    assert result.data is False


@pytest.mark.asyncio
async def test_is_file(sftp_filesystem, mock_sftp_client):
    """Test checking if a path is a file."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Check if path is a file
    result = await sftp_filesystem.is_file("test.txt")
    
    assert result.success
    assert result.data is True


@pytest.mark.asyncio
async def test_is_dir(sftp_filesystem, mock_sftp_client):
    """Test checking if a path is a directory."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Check if path is a directory
    result = await sftp_filesystem.is_dir("test_dir")
    
    assert result.success
    assert result.data is True


@pytest.mark.asyncio
async def test_copy_file(sftp_filesystem, mock_sftp_client):
    """Test copying a file."""
    # Mock the open method for reading
    mock_read_file = AsyncMock()
    mock_read_file.__aenter__ = AsyncMock(return_value=mock_read_file)
    mock_read_file.__aexit__ = AsyncMock(return_value=None)
    mock_read_file.read = AsyncMock(return_value=b"Hello, world!")
    
    # Mock the open method for writing
    mock_write_file = AsyncMock()
    mock_write_file.__aenter__ = AsyncMock(return_value=mock_write_file)
    mock_write_file.__aexit__ = AsyncMock(return_value=None)
    mock_write_file.write = AsyncMock()
    
    # Set up the mock to return different file objects for different calls
    mock_sftp_client.open.side_effect = [mock_read_file, mock_write_file]
    
    # Connect first
    await sftp_filesystem.connect()
    
    # Copy the file
    result = await sftp_filesystem.copy("source.txt", "destination.txt")
    
    assert result.success
    assert mock_sftp_client.open.call_count == 2
    mock_write_file.write.assert_called_once_with(b"Hello, world!")


@pytest.mark.asyncio
async def test_move_file(sftp_filesystem, mock_sftp_client):
    """Test moving a file."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Move the file
    result = await sftp_filesystem.move("source.txt", "destination.txt")
    
    assert result.success
    mock_sftp_client.rename.assert_called_once_with("source.txt", "destination.txt")


@pytest.mark.asyncio
async def test_get_size(sftp_filesystem, mock_sftp_client):
    """Test getting the size of a file."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Get file size
    result = await sftp_filesystem.get_size("test.txt")
    
    assert result.success
    assert result.data == 100


@pytest.mark.asyncio
async def test_get_modified_time(sftp_filesystem, mock_sftp_client):
    """Test getting the modified time of a file."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Get modified time
    result = await sftp_filesystem.get_modified_time("test.txt")
    
    assert result.success
    assert result.data == 1234567890


@pytest.mark.asyncio
async def test_set_modified_time(sftp_filesystem, mock_sftp_client):
    """Test setting the modified time of a file."""
    # Connect first
    await sftp_filesystem.connect()
    
    # Set modified time
    result = await sftp_filesystem.set_modified_time("test.txt", 1234567890)
    
    assert result.success
    mock_sftp_client.setstat.assert_called_once()


@pytest.mark.asyncio
async def test_walk(sftp_filesystem, mock_sftp_client):
    """Test walking a directory tree."""
    # Mock the listdir method to return files and directories
    # First call returns the root directory contents
    # Second call returns the subdirectory contents
    mock_sftp_client.listdir.side_effect = [
        ["test1.txt", "test2.txt", "subdir"],
        ["test3.txt"]
    ]
    
    # Mock the is_dir method to identify directories
    async def mock_is_dir(path):
        if "subdir" in path:
            return FilesystemResult.success_result(data=True)
        return FilesystemResult.success_result(data=False)
    
    sftp_filesystem.is_dir = mock_is_dir
    
    # Connect first
    await sftp_filesystem.connect()
    
    # Walk the directory tree
    result = await sftp_filesystem.walk("test_dir")
    
    assert result.success
    assert len(result.data) == 2  # Root dir and one subdir
    assert result.data[0][0] == ""  # Root dir relative path
    assert result.data[0][1] == ["subdir"]  # Root dir subdirectories
    assert sorted(result.data[0][2]) == ["test1.txt", "test2.txt"]  # Root dir files
    assert result.data[1][0] == "subdir"  # Subdir relative path
    assert result.data[1][1] == []  # Subdir subdirectories
    assert result.data[1][2] == ["test3.txt"]  # Subdir files
