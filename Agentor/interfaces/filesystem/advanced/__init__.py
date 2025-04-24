"""
Advanced filesystem features for the Agentor framework.

This package provides advanced filesystem features such as caching, compression, and encryption.
"""

from .filesystem_cache import (
    FilesystemCache, MemoryCache, DiskCache, 
    CacheStrategy, CacheConfig
)
from .cached_filesystem import CachedFilesystem
from .compression import (
    CompressionAlgorithm, CompressionConfig,
    compress_data, decompress_data
)
from .compressed_filesystem import CompressedFilesystem
from .encryption import (
    EncryptionAlgorithm, EncryptionConfig,
    encrypt_data, decrypt_data
)
from .encrypted_filesystem import EncryptedFilesystem
from .advanced_filesystem import AdvancedFilesystem
from .factory import create_advanced_filesystem

__all__ = [
    # Caching
    'FilesystemCache', 'MemoryCache', 'DiskCache',
    'CacheStrategy', 'CacheConfig', 'CachedFilesystem',
    
    # Compression
    'CompressionAlgorithm', 'CompressionConfig',
    'compress_data', 'decompress_data', 'CompressedFilesystem',
    
    # Encryption
    'EncryptionAlgorithm', 'EncryptionConfig',
    'encrypt_data', 'decrypt_data', 'EncryptedFilesystem',
    
    # Combined
    'AdvancedFilesystem', 'create_advanced_filesystem'
]
