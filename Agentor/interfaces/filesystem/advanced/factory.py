"""
Factory functions for creating advanced filesystems.

This module provides factory functions for creating advanced filesystems with caching, compression, and encryption.
"""

from typing import Dict, Any, Optional, Union, List
import logging

from ..base import FilesystemInterface
from ..local import LocalFilesystem
from ..remote.ftp import FTPFilesystem
from ..remote.sftp_enhanced import SFTPEnhancedFilesystem
from ..cloud.s3 import S3Filesystem
from .filesystem_cache import CacheConfig, CacheStrategy
from .compression import CompressionConfig, CompressionAlgorithm
from .encryption import EncryptionConfig, EncryptionAlgorithm
from .advanced_filesystem import AdvancedFilesystem

logger = logging.getLogger(__name__)


def create_advanced_filesystem(
    filesystem_type: str,
    filesystem_config: Dict[str, Any],
    cache_config: Optional[Dict[str, Any]] = None,
    compression_config: Optional[Dict[str, Any]] = None,
    encryption_config: Optional[Dict[str, Any]] = None
) -> FilesystemInterface:
    """Create an advanced filesystem with the specified features.
    
    Args:
        filesystem_type: Type of filesystem to create (local, ftp, sftp, s3, etc.)
        filesystem_config: Configuration for the base filesystem
        cache_config: Configuration for caching (optional)
        compression_config: Configuration for compression (optional)
        encryption_config: Configuration for encryption (optional)
        
    Returns:
        Advanced filesystem instance
    """
    # Create the base filesystem
    base_fs = _create_base_filesystem(filesystem_type, filesystem_config)
    
    # Create configuration objects
    cache_conf = CacheConfig(**cache_config) if cache_config else None
    compression_conf = CompressionConfig(**compression_config) if compression_config else None
    encryption_conf = EncryptionConfig(**encryption_config) if encryption_config else None
    
    # Create the advanced filesystem
    return AdvancedFilesystem(
        base_fs,
        cache_config=cache_conf,
        compression_config=compression_conf,
        encryption_config=encryption_conf
    )


def _create_base_filesystem(filesystem_type: str, config: Dict[str, Any]) -> FilesystemInterface:
    """Create a base filesystem of the specified type.
    
    Args:
        filesystem_type: Type of filesystem to create (local, ftp, sftp, s3, etc.)
        config: Configuration for the filesystem
        
    Returns:
        Filesystem instance
    """
    if filesystem_type == "local":
        return LocalFilesystem(**config)
    elif filesystem_type == "ftp":
        return FTPFilesystem(**config)
    elif filesystem_type == "sftp":
        return SFTPEnhancedFilesystem(**config)
    elif filesystem_type == "s3":
        return S3Filesystem(**config)
    else:
        raise ValueError(f"Unsupported filesystem type: {filesystem_type}")


def create_cached_filesystem(
    filesystem: FilesystemInterface,
    strategy: CacheStrategy = CacheStrategy.READ_ONLY,
    ttl: int = 300,
    max_size: int = 1024 * 1024 * 100,  # 100 MB
    max_items: int = 10000,
    cache_dir: Optional[str] = None
) -> AdvancedFilesystem:
    """Create a cached filesystem.
    
    Args:
        filesystem: Base filesystem to wrap
        strategy: Cache strategy
        ttl: Cache TTL in seconds
        max_size: Maximum cache size in bytes
        max_items: Maximum number of items in cache
        cache_dir: Cache directory for disk cache
        
    Returns:
        Advanced filesystem with caching
    """
    cache_config = CacheConfig(
        strategy=strategy,
        ttl=ttl,
        max_size=max_size,
        max_items=max_items,
        cache_dir=cache_dir
    )
    
    return AdvancedFilesystem(
        filesystem,
        cache_config=cache_config
    )


def create_compressed_filesystem(
    filesystem: FilesystemInterface,
    algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
    level: int = 6,
    compress_all: bool = False,
    compress_extensions: Optional[List[str]] = None
) -> AdvancedFilesystem:
    """Create a compressed filesystem.
    
    Args:
        filesystem: Base filesystem to wrap
        algorithm: Compression algorithm
        level: Compression level
        compress_all: Whether to compress all files
        compress_extensions: List of file extensions to compress
        
    Returns:
        Advanced filesystem with compression
    """
    compression_config = CompressionConfig(
        algorithm=algorithm,
        level=level,
        compress_all=compress_all
    )
    
    if compress_extensions:
        compression_config.compress_extensions = compress_extensions
    
    return AdvancedFilesystem(
        filesystem,
        compression_config=compression_config
    )


def create_encrypted_filesystem(
    filesystem: FilesystemInterface,
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM,
    key: Optional[str] = None,
    encrypt_all: bool = False,
    encrypt_extensions: Optional[List[str]] = None
) -> AdvancedFilesystem:
    """Create an encrypted filesystem.
    
    Args:
        filesystem: Base filesystem to wrap
        algorithm: Encryption algorithm
        key: Encryption key (base64-encoded)
        encrypt_all: Whether to encrypt all files
        encrypt_extensions: List of file extensions to encrypt
        
    Returns:
        Advanced filesystem with encryption
    """
    encryption_config = EncryptionConfig(
        algorithm=algorithm,
        key=key,
        encrypt_all=encrypt_all
    )
    
    if encrypt_extensions:
        encryption_config.encrypt_extensions = encrypt_extensions
    
    return AdvancedFilesystem(
        filesystem,
        encryption_config=encryption_config
    )
