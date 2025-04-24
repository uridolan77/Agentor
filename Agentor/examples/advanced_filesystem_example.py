"""
Example demonstrating the advanced filesystem features.

This example shows how to use:
- Caching
- Compression
- Encryption
- Combined advanced filesystem
"""

import asyncio
import logging
import sys
import os
import time
import base64
from typing import Dict, Any, Optional, List

from agentor.interfaces.filesystem.local import LocalFilesystem
from agentor.interfaces.filesystem.advanced import (
    CacheConfig, CacheStrategy, MemoryCache, DiskCache,
    CompressionConfig, CompressionAlgorithm,
    EncryptionConfig, EncryptionAlgorithm,
    CachedFilesystem, CompressedFilesystem, EncryptedFilesystem,
    AdvancedFilesystem, create_advanced_filesystem
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_cached_filesystem():
    """Test the cached filesystem."""
    logger.info("Testing cached filesystem")
    
    # Create a temporary directory
    temp_dir = os.path.join(os.path.expanduser("~"), ".agentor_test")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a local filesystem
    local_fs = LocalFilesystem(name="local")
    
    # Create a cache configuration
    cache_config = CacheConfig(
        strategy=CacheStrategy.ALL,
        ttl=60,
        max_size=1024 * 1024,  # 1 MB
        max_items=100
    )
    
    # Create a cached filesystem
    cached_fs = CachedFilesystem(local_fs, cache_config)
    
    # Test file path
    test_file = os.path.join(temp_dir, "cached_test.txt")
    
    # Write a file
    logger.info("Writing file")
    content = "Hello, world! " * 1000  # Create some substantial content
    start_time = time.time()
    result = await cached_fs.write_text(test_file, content)
    write_time = time.time() - start_time
    
    if not result.success:
        logger.error(f"Failed to write file: {result.error}")
        return
    
    logger.info(f"Write time: {write_time:.6f} seconds")
    
    # Read the file (first time, not cached)
    logger.info("Reading file (first time)")
    start_time = time.time()
    result = await cached_fs.read_text(test_file)
    first_read_time = time.time() - start_time
    
    if not result.success:
        logger.error(f"Failed to read file: {result.error}")
        return
    
    logger.info(f"First read time: {first_read_time:.6f} seconds")
    
    # Read the file again (should be cached)
    logger.info("Reading file (second time, cached)")
    start_time = time.time()
    result = await cached_fs.read_text(test_file)
    second_read_time = time.time() - start_time
    
    if not result.success:
        logger.error(f"Failed to read file: {result.error}")
        return
    
    logger.info(f"Second read time: {second_read_time:.6f} seconds")
    logger.info(f"Cache speedup: {first_read_time / second_read_time:.2f}x")
    
    # Get cache statistics
    stats = await cached_fs.get_cache_stats()
    logger.info(f"Cache stats: {stats}")
    
    # Clean up
    await cached_fs.delete_file(test_file)


async def test_compressed_filesystem():
    """Test the compressed filesystem."""
    logger.info("Testing compressed filesystem")
    
    # Create a temporary directory
    temp_dir = os.path.join(os.path.expanduser("~"), ".agentor_test")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a local filesystem
    local_fs = LocalFilesystem(name="local")
    
    # Create a compression configuration
    compression_config = CompressionConfig(
        algorithm=CompressionAlgorithm.GZIP,
        level=6,
        compress_all=True
    )
    
    # Create a compressed filesystem
    compressed_fs = CompressedFilesystem(local_fs, compression_config)
    
    # Test file paths
    uncompressed_file = os.path.join(temp_dir, "uncompressed_test.txt")
    compressed_file = os.path.join(temp_dir, "compressed_test.txt")
    
    # Create test content (highly compressible)
    content = "A" * 10000
    
    # Write uncompressed file directly
    logger.info("Writing uncompressed file")
    result = await local_fs.write_text(uncompressed_file, content)
    if not result.success:
        logger.error(f"Failed to write uncompressed file: {result.error}")
        return
    
    # Write compressed file
    logger.info("Writing compressed file")
    result = await compressed_fs.write_text(compressed_file, content)
    if not result.success:
        logger.error(f"Failed to write compressed file: {result.error}")
        return
    
    # Get file sizes
    uncompressed_size_result = await local_fs.get_size(uncompressed_file)
    compressed_size_result = await local_fs.get_size(compressed_file)
    
    if uncompressed_size_result.success and compressed_size_result.success:
        uncompressed_size = uncompressed_size_result.data
        compressed_size = compressed_size_result.data
        
        logger.info(f"Uncompressed size: {uncompressed_size} bytes")
        logger.info(f"Compressed size: {compressed_size} bytes")
        logger.info(f"Compression ratio: {uncompressed_size / compressed_size:.2f}x")
    
    # Read the compressed file
    logger.info("Reading compressed file")
    result = await compressed_fs.read_text(compressed_file)
    if not result.success:
        logger.error(f"Failed to read compressed file: {result.error}")
        return
    
    # Verify the content
    if result.data == content:
        logger.info("Content verified successfully")
    else:
        logger.error("Content verification failed")
    
    # Clean up
    await local_fs.delete_file(uncompressed_file)
    await compressed_fs.delete_file(compressed_file)


async def test_encrypted_filesystem():
    """Test the encrypted filesystem."""
    logger.info("Testing encrypted filesystem")
    
    # Create a temporary directory
    temp_dir = os.path.join(os.path.expanduser("~"), ".agentor_test")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a local filesystem
    local_fs = LocalFilesystem(name="local")
    
    # Generate a random encryption key
    key = base64.b64encode(os.urandom(32)).decode()
    
    # Create an encryption configuration
    encryption_config = EncryptionConfig(
        algorithm=EncryptionAlgorithm.AES_GCM,
        key=key,
        encrypt_all=True
    )
    
    # Create an encrypted filesystem
    encrypted_fs = EncryptedFilesystem(local_fs, encryption_config)
    
    # Test file paths
    unencrypted_file = os.path.join(temp_dir, "unencrypted_test.txt")
    encrypted_file = os.path.join(temp_dir, "encrypted_test.txt")
    
    # Create test content
    content = "This is sensitive information that should be encrypted."
    
    # Write unencrypted file directly
    logger.info("Writing unencrypted file")
    result = await local_fs.write_text(unencrypted_file, content)
    if not result.success:
        logger.error(f"Failed to write unencrypted file: {result.error}")
        return
    
    # Write encrypted file
    logger.info("Writing encrypted file")
    result = await encrypted_fs.write_text(encrypted_file, content)
    if not result.success:
        logger.error(f"Failed to write encrypted file: {result.error}")
        return
    
    # Read the unencrypted file directly
    logger.info("Reading unencrypted file directly")
    result = await local_fs.read_text(unencrypted_file)
    if not result.success:
        logger.error(f"Failed to read unencrypted file: {result.error}")
        return
    
    logger.info(f"Unencrypted content: {result.data}")
    
    # Try to read the encrypted file directly (should be encrypted)
    logger.info("Reading encrypted file directly (should be encrypted)")
    result = await local_fs.read_text(encrypted_file)
    if not result.success:
        logger.error(f"Failed to read encrypted file: {result.error}")
        return
    
    logger.info(f"Encrypted content (sample): {result.data[:50]}...")
    
    # Read the encrypted file through the encrypted filesystem
    logger.info("Reading encrypted file through the encrypted filesystem")
    result = await encrypted_fs.read_text(encrypted_file)
    if not result.success:
        logger.error(f"Failed to read encrypted file: {result.error}")
        return
    
    logger.info(f"Decrypted content: {result.data}")
    
    # Verify the content
    if result.data == content:
        logger.info("Content verified successfully")
    else:
        logger.error("Content verification failed")
    
    # Clean up
    await local_fs.delete_file(unencrypted_file)
    await encrypted_fs.delete_file(encrypted_file)


async def test_advanced_filesystem():
    """Test the advanced filesystem with all features."""
    logger.info("Testing advanced filesystem with all features")
    
    # Create a temporary directory
    temp_dir = os.path.join(os.path.expanduser("~"), ".agentor_test")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a local filesystem
    local_fs = LocalFilesystem(name="local")
    
    # Generate a random encryption key
    key = base64.b64encode(os.urandom(32)).decode()
    
    # Create configurations
    cache_config = CacheConfig(
        strategy=CacheStrategy.ALL,
        ttl=60,
        max_size=1024 * 1024,  # 1 MB
        max_items=100
    )
    
    compression_config = CompressionConfig(
        algorithm=CompressionAlgorithm.GZIP,
        level=6,
        compress_all=True
    )
    
    encryption_config = EncryptionConfig(
        algorithm=EncryptionAlgorithm.AES_GCM,
        key=key,
        encrypt_all=True
    )
    
    # Create an advanced filesystem
    advanced_fs = AdvancedFilesystem(
        local_fs,
        cache_config=cache_config,
        compression_config=compression_config,
        encryption_config=encryption_config
    )
    
    # Test file path
    test_file = os.path.join(temp_dir, "advanced_test.txt")
    
    # Create test content (highly compressible)
    content = "A" * 10000
    
    # Write the file
    logger.info("Writing file")
    result = await advanced_fs.write_text(test_file, content)
    if not result.success:
        logger.error(f"Failed to write file: {result.error}")
        return
    
    # Get the file size (should be smaller due to compression)
    size_result = await local_fs.get_size(test_file)
    if size_result.success:
        logger.info(f"File size: {size_result.data} bytes")
        logger.info(f"Compression ratio: {len(content) / size_result.data:.2f}x")
    
    # Try to read the file directly (should be encrypted and compressed)
    logger.info("Reading file directly (should be encrypted and compressed)")
    result = await local_fs.read_text(test_file)
    if not result.success:
        logger.error(f"Failed to read file directly: {result.error}")
        return
    
    logger.info(f"Raw content (sample): {result.data[:50]}...")
    
    # Read the file through the advanced filesystem (first time)
    logger.info("Reading file through the advanced filesystem (first time)")
    start_time = time.time()
    result = await advanced_fs.read_text(test_file)
    first_read_time = time.time() - start_time
    
    if not result.success:
        logger.error(f"Failed to read file: {result.error}")
        return
    
    logger.info(f"First read time: {first_read_time:.6f} seconds")
    
    # Verify the content
    if result.data == content:
        logger.info("Content verified successfully")
    else:
        logger.error("Content verification failed")
    
    # Read the file again (should be cached)
    logger.info("Reading file again (should be cached)")
    start_time = time.time()
    result = await advanced_fs.read_text(test_file)
    second_read_time = time.time() - start_time
    
    if not result.success:
        logger.error(f"Failed to read file again: {result.error}")
        return
    
    logger.info(f"Second read time: {second_read_time:.6f} seconds")
    logger.info(f"Cache speedup: {first_read_time / second_read_time:.2f}x")
    
    # Get cache statistics
    if hasattr(advanced_fs, "get_cache_stats"):
        stats = await advanced_fs.get_cache_stats()
        logger.info(f"Cache stats: {stats}")
    
    # Clean up
    await advanced_fs.delete_file(test_file)


async def test_factory():
    """Test the factory function for creating advanced filesystems."""
    logger.info("Testing factory function")
    
    # Create a temporary directory
    temp_dir = os.path.join(os.path.expanduser("~"), ".agentor_test")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a random encryption key
    key = base64.b64encode(os.urandom(32)).decode()
    
    # Create an advanced filesystem using the factory
    fs = create_advanced_filesystem(
        filesystem_type="local",
        filesystem_config={"name": "local"},
        cache_config={
            "strategy": CacheStrategy.ALL,
            "ttl": 60,
            "max_size": 1024 * 1024,  # 1 MB
            "max_items": 100
        },
        compression_config={
            "algorithm": CompressionAlgorithm.GZIP,
            "level": 6,
            "compress_all": True
        },
        encryption_config={
            "algorithm": EncryptionAlgorithm.AES_GCM,
            "key": key,
            "encrypt_all": True
        }
    )
    
    # Test file path
    test_file = os.path.join(temp_dir, "factory_test.txt")
    
    # Create test content
    content = "This file was created using the factory function."
    
    # Write the file
    logger.info("Writing file")
    result = await fs.write_text(test_file, content)
    if not result.success:
        logger.error(f"Failed to write file: {result.error}")
        return
    
    # Read the file
    logger.info("Reading file")
    result = await fs.read_text(test_file)
    if not result.success:
        logger.error(f"Failed to read file: {result.error}")
        return
    
    # Verify the content
    if result.data == content:
        logger.info("Content verified successfully")
    else:
        logger.error("Content verification failed")
    
    # Clean up
    await fs.delete_file(test_file)


async def main():
    """Run the advanced filesystem examples."""
    logger.info("Starting advanced filesystem examples")
    
    # Test each feature individually
    await test_cached_filesystem()
    logger.info("-" * 80)
    
    await test_compressed_filesystem()
    logger.info("-" * 80)
    
    await test_encrypted_filesystem()
    logger.info("-" * 80)
    
    # Test all features combined
    await test_advanced_filesystem()
    logger.info("-" * 80)
    
    # Test the factory function
    await test_factory()
    
    logger.info("Advanced filesystem examples completed")


if __name__ == "__main__":
    asyncio.run(main())
