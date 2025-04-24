"""
Filesystem interfaces for the Agentor framework.

This module provides interfaces for interacting with various filesystem types,
including local filesystems, remote filesystems, cloud storage, and virtual filesystems.
"""

from .base import (
    FilesystemError, FileNotFoundError, PermissionError, FileExistsError,
    FilesystemResult, FileInfo, FileMode, FilesystemInterface
)
from .local import LocalFilesystem
# Remote filesystems
from .remote.ftp import FTPFilesystem
from .remote.sftp import SFTPFilesystem
from .remote.sftp_enhanced import SFTPEnhancedFilesystem
from .remote.webdav import WebDAVFilesystem

# Cloud filesystems
from .cloud.s3 import S3Filesystem
from .cloud.gcs import GCSFilesystem
from .cloud.azure import AzureBlobFilesystem
from .virtual import VirtualFilesystem
from .registry import FilesystemRegistry, default_registry

# Advanced filesystem features
from .advanced import (
    CacheConfig, CacheStrategy, MemoryCache, DiskCache,
    CompressionConfig, CompressionAlgorithm,
    EncryptionConfig, EncryptionAlgorithm,
    CachedFilesystem, CompressedFilesystem, EncryptedFilesystem,
    AdvancedFilesystem, create_advanced_filesystem
)

__all__ = [
    'FilesystemError', 'FileNotFoundError', 'PermissionError', 'FileExistsError',
    'FilesystemResult', 'FileInfo', 'FileMode', 'FilesystemInterface',
    'LocalFilesystem',
    'FTPFilesystem', 'SFTPFilesystem', 'SFTPEnhancedFilesystem', 'WebDAVFilesystem',
    'S3Filesystem', 'GCSFilesystem', 'AzureBlobFilesystem',
    'VirtualFilesystem',
    'FilesystemRegistry', 'default_registry',
    # Advanced filesystem features
    'CacheConfig', 'CacheStrategy', 'MemoryCache', 'DiskCache',
    'CompressionConfig', 'CompressionAlgorithm',
    'EncryptionConfig', 'EncryptionAlgorithm',
    'CachedFilesystem', 'CompressedFilesystem', 'EncryptedFilesystem',
    'AdvancedFilesystem', 'create_advanced_filesystem'
]
