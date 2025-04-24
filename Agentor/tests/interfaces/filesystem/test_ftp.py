"""
Tests for the FTP filesystem interface.

This module contains tests for the FTP filesystem interface with resilience patterns.
"""

import os
import io
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aioftp

from agentor.interfaces.filesystem.remote.ftp import FTPFilesystem
from agentor.interfaces.filesystem.base import FilesystemResult, FileInfo


@pytest.fixture
def mock_aioftp_client():
    """Create a mock aioftp client."""
    client = AsyncMock()
    client.connect = AsyncMock()
    client.command = AsyncMock()
    client.quit = AsyncMock()
    client.change_directory = AsyncMock()
    client.make_directory = AsyncMock()
    client.download_stream = AsyncMock()
    client.upload_stream = AsyncMock()
    client.remove_file = AsyncMock()
    client.remove_directory = AsyncMock()
    client.list = AsyncMock(return_value=[])
    
    # Mock stat method to return file info
    async def mock_stat(path):
        if path.endswith(".txt"):
            return {"type": "file", "size": 100, "modify": 1234567890}
        else:
            return {"type": "dir", "size": 0, "modify": 1234567890}
    
    client.stat = AsyncMock(side_effect=mock_stat)
    
    return client


@pytest.fixture
def ftp_filesystem(mock_aioftp_client):
    """Create a FTP filesystem with a mock client."""
    with patch("aioftp.Client", return_value=mock_aioftp_client):
        fs = FTPFilesystem(
            name="test-ftp",
            host="ftp.example.com",
            port=21,
            username="test",
            password="password",
            root_dir="/test",
            passive_mode=True,
            secure=False
        )
        yield fs


@pytest.mark.asyncio
async def test_connect(ftp_filesystem, mock_aioftp_client):
    """Test connecting to the FTP server."""
    result = await ftp_filesystem.connect()
    
    assert result.success
    mock_aioftp_client.connect.assert_called_once()
    mock_aioftp_client.command.assert_called_once_with("PASV")
    mock_aioftp_client.change_directory.assert_called_once_with("/test")


@pytest.mark.asyncio
async def test_connect_error(ftp_filesystem, mock_aioftp_client):
    """Test error handling when connecting to the FTP server."""
    mock_aioftp_client.connect.side_effect = Exception("Connection error")
    
    result = await ftp_filesystem.connect()
    
    assert not result.success
    assert "Connection error" in str(result.error)


@pytest.mark.asyncio
async def test_read_text(ftp_filesystem, mock_aioftp_client):
    """Test reading text from a file."""
    # Mock the download_stream method to write data to the buffer
    async def mock_download_stream(path, buffer):
        buffer.write(b"Hello, world!")
    
    mock_aioftp_client.download_stream.side_effect = mock_download_stream
    
    # Connect first
    await ftp_filesystem.connect()
    
    # Read the file
    result = await ftp_filesystem.read_text("test.txt")
    
    assert result.success
    assert result.data == "Hello, world!"
    mock_aioftp_client.download_stream.assert_called_once()


@pytest.mark.asyncio
async def test_read_text_file_not_found(ftp_filesystem, mock_aioftp_client):
    """Test error handling when reading a non-existent file."""
    # Mock the stat method to raise PathNotFoundError
    mock_aioftp_client.stat.side_effect = aioftp.errors.PathNotFoundError("File not found")
    
    # Connect first
    await ftp_filesystem.connect()
    
    # Read the file
    result = await ftp_filesystem.read_text("nonexistent.txt")
    
    assert not result.success
    assert "File not found" in str(result.error)


@pytest.mark.asyncio
async def test_write_text(ftp_filesystem, mock_aioftp_client):
    """Test writing text to a file."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Write the file
    result = await ftp_filesystem.write_text("test.txt", "Hello, world!")
    
    assert result.success
    mock_aioftp_client.upload_stream.assert_called_once()


@pytest.mark.asyncio
async def test_create_dir(ftp_filesystem, mock_aioftp_client):
    """Test creating a directory."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Create the directory
    result = await ftp_filesystem.create_dir("test_dir")
    
    assert result.success
    mock_aioftp_client.make_directory.assert_called_once_with("test_dir", recursive=True)


@pytest.mark.asyncio
async def test_list_dir(ftp_filesystem, mock_aioftp_client):
    """Test listing directory contents."""
    # Mock the list method to return files
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_aioftp_client.list.return_value = [mock_file]
    
    # Connect first
    await ftp_filesystem.connect()
    
    # List the directory
    result = await ftp_filesystem.list_dir("test_dir")
    
    assert result.success
    assert result.data == ["test.txt"]
    mock_aioftp_client.list.assert_called_once_with("test_dir")


@pytest.mark.asyncio
async def test_delete_file(ftp_filesystem, mock_aioftp_client):
    """Test deleting a file."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Delete the file
    result = await ftp_filesystem.delete_file("test.txt")
    
    assert result.success
    mock_aioftp_client.remove_file.assert_called_once_with("test.txt")


@pytest.mark.asyncio
async def test_delete_dir(ftp_filesystem, mock_aioftp_client):
    """Test deleting a directory."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Delete the directory
    result = await ftp_filesystem.delete_dir("test_dir")
    
    assert result.success
    mock_aioftp_client.remove_directory.assert_called_once_with("test_dir")


@pytest.mark.asyncio
async def test_get_info(ftp_filesystem, mock_aioftp_client):
    """Test getting file information."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Get file info
    result = await ftp_filesystem.get_info("test.txt")
    
    assert result.success
    assert result.data.name == "test.txt"
    assert result.data.is_file
    assert not result.data.is_dir
    assert result.data.size == 100
    assert result.data.modified_time == 1234567890


@pytest.mark.asyncio
async def test_exists(ftp_filesystem, mock_aioftp_client):
    """Test checking if a file exists."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Check if file exists
    result = await ftp_filesystem.exists("test.txt")
    
    assert result.success
    assert result.data is True


@pytest.mark.asyncio
async def test_exists_not_found(ftp_filesystem, mock_aioftp_client):
    """Test checking if a non-existent file exists."""
    # Mock the stat method to raise PathNotFoundError
    mock_aioftp_client.stat.side_effect = aioftp.errors.PathNotFoundError("File not found")
    
    # Connect first
    await ftp_filesystem.connect()
    
    # Check if file exists
    result = await ftp_filesystem.exists("nonexistent.txt")
    
    assert result.success
    assert result.data is False


@pytest.mark.asyncio
async def test_is_file(ftp_filesystem, mock_aioftp_client):
    """Test checking if a path is a file."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Check if path is a file
    result = await ftp_filesystem.is_file("test.txt")
    
    assert result.success
    assert result.data is True


@pytest.mark.asyncio
async def test_is_dir(ftp_filesystem, mock_aioftp_client):
    """Test checking if a path is a directory."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Check if path is a directory
    result = await ftp_filesystem.is_dir("test_dir")
    
    assert result.success
    assert result.data is True


@pytest.mark.asyncio
async def test_copy(ftp_filesystem, mock_aioftp_client):
    """Test copying a file."""
    # Mock the download_stream method to write data to the buffer
    async def mock_download_stream(path, buffer):
        buffer.write(b"Hello, world!")
    
    mock_aioftp_client.download_stream.side_effect = mock_download_stream
    
    # Connect first
    await ftp_filesystem.connect()
    
    # Copy the file
    result = await ftp_filesystem.copy("source.txt", "destination.txt")
    
    assert result.success
    mock_aioftp_client.download_stream.assert_called_once()
    mock_aioftp_client.upload_stream.assert_called_once()


@pytest.mark.asyncio
async def test_move(ftp_filesystem, mock_aioftp_client):
    """Test moving a file."""
    # Mock the download_stream method to write data to the buffer
    async def mock_download_stream(path, buffer):
        buffer.write(b"Hello, world!")
    
    mock_aioftp_client.download_stream.side_effect = mock_download_stream
    
    # Connect first
    await ftp_filesystem.connect()
    
    # Move the file
    result = await ftp_filesystem.move("source.txt", "destination.txt")
    
    assert result.success
    mock_aioftp_client.download_stream.assert_called_once()
    mock_aioftp_client.upload_stream.assert_called_once()
    mock_aioftp_client.remove_file.assert_called_once_with("source.txt")


@pytest.mark.asyncio
async def test_get_size(ftp_filesystem, mock_aioftp_client):
    """Test getting the size of a file."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Get file size
    result = await ftp_filesystem.get_size("test.txt")
    
    assert result.success
    assert result.data == 100


@pytest.mark.asyncio
async def test_get_modified_time(ftp_filesystem, mock_aioftp_client):
    """Test getting the modified time of a file."""
    # Connect first
    await ftp_filesystem.connect()
    
    # Get modified time
    result = await ftp_filesystem.get_modified_time("test.txt")
    
    assert result.success
    assert result.data == 1234567890


@pytest.mark.asyncio
async def test_walk(ftp_filesystem, mock_aioftp_client):
    """Test walking a directory tree."""
    # Mock the list method to return files and directories
    file1 = MagicMock()
    file1.name = "test1.txt"
    file2 = MagicMock()
    file2.name = "test2.txt"
    dir1 = MagicMock()
    dir1.name = "subdir"
    
    # First call returns the root directory contents
    # Second call returns the subdirectory contents
    mock_aioftp_client.list.side_effect = [
        [file1, file2, dir1],
        [file1]
    ]
    
    # Connect first
    await ftp_filesystem.connect()
    
    # Walk the directory tree
    result = await ftp_filesystem.walk("test_dir")
    
    assert result.success
    assert len(result.data) == 2  # Root dir and one subdir
    assert result.data[0][0] == ""  # Root dir relative path
    assert sorted(result.data[0][1]) == ["subdir"]  # Root dir subdirectories
    assert sorted(result.data[0][2]) == ["test1.txt", "test2.txt"]  # Root dir files
    assert result.data[1][0] == "subdir"  # Subdir relative path
    assert result.data[1][1] == []  # Subdir subdirectories
    assert result.data[1][2] == ["test1.txt"]  # Subdir files
