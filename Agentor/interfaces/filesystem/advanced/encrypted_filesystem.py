"""
Encrypted filesystem implementation for the Agentor framework.

This module provides an encrypted filesystem implementation that wraps any filesystem.
"""

import os
import time
import asyncio
import functools
from typing import Dict, Any, Optional, List, Union, Tuple, Set, TypeVar, Generic, cast
import logging

from ..base import (
    FilesystemInterface, FilesystemResult, FileInfo,
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError
)
from .encryption import (
    EncryptionConfig, EncryptionAlgorithm,
    should_encrypt, encrypt_data, decrypt_data,
    encrypt_text, decrypt_text
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EncryptedFilesystem(FilesystemInterface):
    """Encrypted filesystem implementation that wraps any filesystem."""
    
    def __init__(
        self,
        filesystem: FilesystemInterface,
        encryption_config: EncryptionConfig = None
    ):
        """Initialize the encrypted filesystem.
        
        Args:
            filesystem: The underlying filesystem to wrap
            encryption_config: Encryption configuration
        """
        self.filesystem = filesystem
        self.encryption_config = encryption_config or EncryptionConfig()
        self.name = f"encrypted-{filesystem.name}"
        
        # Keep track of encrypted files
        self._encrypted_files: Set[str] = set()
    
    async def connect(self) -> FilesystemResult:
        """Connect to the filesystem."""
        return await self.filesystem.connect()
    
    async def disconnect(self) -> FilesystemResult:
        """Disconnect from the filesystem."""
        return await self.filesystem.disconnect()
    
    async def read_text(self, path: str, encoding: str = "utf-8") -> FilesystemResult[str]:
        """Read text from a file."""
        # Check if the file is encrypted
        is_encrypted = await self._is_encrypted(path)
        
        if is_encrypted:
            # Read the encrypted data
            result = await self.filesystem.read_bytes(path)
            if not result.success:
                return FilesystemResult.error_result(result.error)
            
            try:
                # Decrypt the data
                text = decrypt_text(result.data, encoding, self.encryption_config)
                return FilesystemResult.success_result(data=text)
            except Exception as e:
                logger.error(f"Failed to decrypt file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to decrypt file: {e}"))
        else:
            # Read the file normally
            return await self.filesystem.read_text(path, encoding)
    
    async def read_bytes(self, path: str) -> FilesystemResult[bytes]:
        """Read bytes from a file."""
        # Check if the file is encrypted
        is_encrypted = await self._is_encrypted(path)
        
        if is_encrypted:
            # Read the encrypted data
            result = await self.filesystem.read_bytes(path)
            if not result.success:
                return result
            
            try:
                # Decrypt the data
                data = decrypt_data(result.data, self.encryption_config)
                return FilesystemResult.success_result(data=data)
            except Exception as e:
                logger.error(f"Failed to decrypt file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to decrypt file: {e}"))
        else:
            # Read the file normally
            return await self.filesystem.read_bytes(path)
    
    async def write_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Write text to a file."""
        # Check if the file should be encrypted
        if should_encrypt(path, self.encryption_config, len(content.encode(encoding))):
            try:
                # Encrypt the content
                encrypted_data = encrypt_text(content, encoding, self.encryption_config)
                
                # Write the encrypted data
                result = await self.filesystem.write_bytes(path, encrypted_data)
                
                # If successful, add to the list of encrypted files
                if result.success:
                    self._encrypted_files.add(path)
                
                return result
            except Exception as e:
                logger.error(f"Failed to encrypt file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to encrypt file: {e}"))
        else:
            # Write the file normally
            return await self.filesystem.write_text(path, content, encoding)
    
    async def write_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Write bytes to a file."""
        # Check if the file should be encrypted
        if should_encrypt(path, self.encryption_config, len(content)):
            try:
                # Encrypt the content
                encrypted_data = encrypt_data(content, self.encryption_config)
                
                # Write the encrypted data
                result = await self.filesystem.write_bytes(path, encrypted_data)
                
                # If successful, add to the list of encrypted files
                if result.success:
                    self._encrypted_files.add(path)
                
                return result
            except Exception as e:
                logger.error(f"Failed to encrypt file: {path}, error: {e}")
                return FilesystemResult.error_result(FilesystemError(f"Failed to encrypt file: {e}"))
        else:
            # Write the file normally
            return await self.filesystem.write_bytes(path, content)
    
    async def append_text(self, path: str, content: str, encoding: str = "utf-8") -> FilesystemResult:
        """Append text to a file."""
        # Check if the file is already encrypted
        is_encrypted = await self._is_encrypted(path)
        
        if is_encrypted:
            # Read the existing content
            result = await self.read_text(path, encoding)
            if not result.success:
                return result
            
            # Append the new content
            new_content = result.data + content
            
            # Write the combined content
            return await self.write_text(path, new_content, encoding)
        else:
            # Check if the file should be encrypted
            if should_encrypt(path, self.encryption_config):
                # Check if the file exists
                exists_result = await self.filesystem.exists(path)
                if not exists_result.success:
                    return exists_result
                
                if exists_result.data:
                    # Read the existing content
                    result = await self.filesystem.read_text(path, encoding)
                    if not result.success:
                        return result
                    
                    # Append the new content
                    new_content = result.data + content
                    
                    # Write the combined content
                    return await self.write_text(path, new_content, encoding)
                else:
                    # File doesn't exist, write the content
                    return await self.write_text(path, content, encoding)
            else:
                # Append normally
                return await self.filesystem.append_text(path, content, encoding)
    
    async def append_bytes(self, path: str, content: bytes) -> FilesystemResult:
        """Append bytes to a file."""
        # Check if the file is already encrypted
        is_encrypted = await self._is_encrypted(path)
        
        if is_encrypted:
            # Read the existing content
            result = await self.read_bytes(path)
            if not result.success:
                return result
            
            # Append the new content
            new_content = result.data + content
            
            # Write the combined content
            return await self.write_bytes(path, new_content)
        else:
            # Check if the file should be encrypted
            if should_encrypt(path, self.encryption_config):
                # Check if the file exists
                exists_result = await self.filesystem.exists(path)
                if not exists_result.success:
                    return exists_result
                
                if exists_result.data:
                    # Read the existing content
                    result = await self.filesystem.read_bytes(path)
                    if not result.success:
                        return result
                    
                    # Append the new content
                    new_content = result.data + content
                    
                    # Write the combined content
                    return await self.write_bytes(path, new_content)
                else:
                    # File doesn't exist, write the content
                    return await self.write_bytes(path, content)
            else:
                # Append normally
                return await self.filesystem.append_bytes(path, content)
    
    async def delete_file(self, path: str) -> FilesystemResult:
        """Delete a file."""
        # Remove from the list of encrypted files if present
        if path in self._encrypted_files:
            self._encrypted_files.remove(path)
        
        # Delete the file normally
        return await self.filesystem.delete_file(path)
    
    async def create_dir(self, path: str, exist_ok: bool = False) -> FilesystemResult:
        """Create a directory."""
        return await self.filesystem.create_dir(path, exist_ok)
    
    async def list_dir(self, path: str) -> FilesystemResult[List[str]]:
        """List directory contents."""
        return await self.filesystem.list_dir(path)
    
    async def delete_dir(self, path: str, recursive: bool = False) -> FilesystemResult:
        """Delete a directory."""
        # If recursive, remove all files in the directory from the list of encrypted files
        if recursive:
            # Get all files in the directory
            walk_result = await self.walk(path)
            if walk_result.success:
                for root, _, files in walk_result.data:
                    for file in files:
                        file_path = os.path.join(path, root, file)
                        if file_path in self._encrypted_files:
                            self._encrypted_files.remove(file_path)
        
        # Delete the directory normally
        return await self.filesystem.delete_dir(path, recursive)
    
    async def get_info(self, path: str) -> FilesystemResult[FileInfo]:
        """Get information about a file or directory."""
        # Get the file info normally
        result = await self.filesystem.get_info(path)
        if not result.success:
            return result
        
        # Check if the file is encrypted
        if path in self._encrypted_files and result.data.is_file:
            # Get the unencrypted size
            size_result = await self._get_unencrypted_size(path)
            if size_result.success:
                # Update the file info with the unencrypted size
                info = result.data
                info.metadata = info.metadata or {}
                info.metadata["encrypted"] = True
                info.metadata["encrypted_size"] = info.size
                info.metadata["unencrypted_size"] = size_result.data
                
                # Don't modify the actual size, as that would be misleading
                # info.size = size_result.data
                
                return FilesystemResult.success_result(data=info)
        
        return result
    
    async def exists(self, path: str) -> FilesystemResult[bool]:
        """Check if a file or directory exists."""
        return await self.filesystem.exists(path)
    
    async def is_file(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a file."""
        return await self.filesystem.is_file(path)
    
    async def is_dir(self, path: str) -> FilesystemResult[bool]:
        """Check if a path is a directory."""
        return await self.filesystem.is_dir(path)
    
    async def copy(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Copy a file or directory."""
        # Check if the source file is encrypted
        is_encrypted = await self._is_encrypted(src_path)
        
        if is_encrypted:
            # Check if the source is a file
            is_file_result = await self.filesystem.is_file(src_path)
            if not is_file_result.success:
                return is_file_result
            
            if is_file_result.data:
                # Read the source file
                read_result = await self.read_bytes(src_path)
                if not read_result.success:
                    return read_result
                
                # Write to the destination
                write_result = await self.write_bytes(dst_path, read_result.data)
                return write_result
        
        # Copy normally
        result = await self.filesystem.copy(src_path, dst_path, overwrite)
        
        # If successful and the source file is encrypted, add the destination to the list
        if result.success and is_encrypted:
            self._encrypted_files.add(dst_path)
        
        return result
    
    async def move(self, src_path: str, dst_path: str, overwrite: bool = False) -> FilesystemResult:
        """Move a file or directory."""
        # Check if the source file is encrypted
        is_encrypted = await self._is_encrypted(src_path)
        
        # Move normally
        result = await self.filesystem.move(src_path, dst_path, overwrite)
        
        # If successful and the source file is encrypted, update the list
        if result.success and is_encrypted:
            if src_path in self._encrypted_files:
                self._encrypted_files.remove(src_path)
            self._encrypted_files.add(dst_path)
        
        return result
    
    async def get_size(self, path: str) -> FilesystemResult[int]:
        """Get the size of a file or directory."""
        # Check if the path is a file
        is_file_result = await self.filesystem.is_file(path)
        if not is_file_result.success:
            return is_file_result
        
        if is_file_result.data:
            # Check if the file is encrypted
            is_encrypted = await self._is_encrypted(path)
            
            if is_encrypted:
                # Get the unencrypted size
                return await self._get_unencrypted_size(path)
        
        # Get the size normally
        return await self.filesystem.get_size(path)
    
    async def get_modified_time(self, path: str) -> FilesystemResult[float]:
        """Get the last modified time of a file or directory."""
        return await self.filesystem.get_modified_time(path)
    
    async def set_modified_time(self, path: str, mtime: float) -> FilesystemResult:
        """Set the last modified time of a file or directory."""
        return await self.filesystem.set_modified_time(path, mtime)
    
    async def walk(self, path: str) -> FilesystemResult[List[Tuple[str, List[str], List[str]]]]:
        """Walk a directory tree."""
        return await self.filesystem.walk(path)
    
    async def _is_encrypted(self, path: str) -> bool:
        """Check if a file is encrypted.
        
        Args:
            path: File path
            
        Returns:
            True if the file is encrypted, False otherwise
        """
        # Check if the file is in the list of encrypted files
        if path in self._encrypted_files:
            return True
        
        # Check if the file exists
        exists_result = await self.filesystem.exists(path)
        if not exists_result.success or not exists_result.data:
            return False
        
        # Check if the file is a file
        is_file_result = await self.filesystem.is_file(path)
        if not is_file_result.success or not is_file_result.data:
            return False
        
        # Read the first few bytes to check for the encryption header
        try:
            # Read the first 10 bytes (header size)
            read_result = await self.filesystem.read_bytes(path)
            if not read_result.success:
                return False
            
            data = read_result.data
            
            # Check for the encryption header
            from .encryption import ENCRYPTION_HEADER
            if len(data) >= 10 and data[:8] == ENCRYPTION_HEADER:
                # Add to the list of encrypted files
                self._encrypted_files.add(path)
                return True
        except Exception:
            pass
        
        return False
    
    async def _get_unencrypted_size(self, path: str) -> FilesystemResult[int]:
        """Get the unencrypted size of a file.
        
        Args:
            path: File path
            
        Returns:
            Unencrypted size in bytes
        """
        # Read the encrypted data
        result = await self.filesystem.read_bytes(path)
        if not result.success:
            return result
        
        try:
            # Decrypt the data
            data = decrypt_data(result.data, self.encryption_config)
            
            # Return the size
            return FilesystemResult.success_result(data=len(data))
        except Exception as e:
            logger.error(f"Failed to get unencrypted size: {path}, error: {e}")
            return FilesystemResult.error_result(FilesystemError(f"Failed to get unencrypted size: {e}"))
