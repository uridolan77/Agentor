"""
Compression utilities for the Agentor framework.

This module provides compression utilities for filesystem operations.
"""

import gzip
import zlib
import bz2
import lzma
import io
from enum import Enum
from typing import Dict, Any, Optional, Union, Callable, Tuple
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CompressionAlgorithm(str, Enum):
    """Compression algorithms."""
    
    # No compression
    NONE = "none"
    
    # gzip compression
    GZIP = "gzip"
    
    # zlib compression
    ZLIB = "zlib"
    
    # bzip2 compression
    BZ2 = "bz2"
    
    # lzma compression
    LZMA = "lzma"


class CompressionConfig(BaseModel):
    """Configuration for compression."""
    
    # Compression algorithm
    algorithm: CompressionAlgorithm = Field(
        CompressionAlgorithm.GZIP,
        description="Compression algorithm to use"
    )
    
    # Compression level (1-9, where 9 is highest compression)
    level: int = Field(
        6,
        description="Compression level (1-9, where 9 is highest compression)"
    )
    
    # Whether to compress all files or only specific extensions
    compress_all: bool = Field(
        False,
        description="Whether to compress all files or only specific extensions"
    )
    
    # List of file extensions to compress (if compress_all is False)
    compress_extensions: list[str] = Field(
        [".txt", ".csv", ".json", ".xml", ".html", ".md", ".log"],
        description="List of file extensions to compress (if compress_all is False)"
    )
    
    # List of file extensions to exclude from compression
    exclude_extensions: list[str] = Field(
        [".gz", ".zip", ".bz2", ".xz", ".7z", ".rar", ".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4", ".avi", ".mov"],
        description="List of file extensions to exclude from compression"
    )
    
    # Minimum file size to compress (in bytes)
    min_size: int = Field(
        1024,  # 1 KB
        description="Minimum file size to compress (in bytes)"
    )
    
    # Maximum file size to compress (in bytes, 0 = no limit)
    max_size: int = Field(
        1024 * 1024 * 10,  # 10 MB
        description="Maximum file size to compress (in bytes, 0 = no limit)"
    )
    
    # Whether to add a compression header to identify compressed files
    add_header: bool = Field(
        True,
        description="Whether to add a compression header to identify compressed files"
    )


# Compression header (8 bytes)
COMPRESSION_HEADER = b"AGNTRCMP"


def should_compress(path: str, config: CompressionConfig, size: int = 0) -> bool:
    """Check if a file should be compressed.
    
    Args:
        path: File path
        config: Compression configuration
        size: File size in bytes (if known)
        
    Returns:
        True if the file should be compressed, False otherwise
    """
    # Check if compression is disabled
    if config.algorithm == CompressionAlgorithm.NONE:
        return False
    
    # Check file size
    if size > 0:
        if config.min_size > 0 and size < config.min_size:
            return False
        if config.max_size > 0 and size > config.max_size:
            return False
    
    # Check file extension
    ext = "." + path.split(".")[-1].lower() if "." in path else ""
    
    # Check if the extension is excluded
    if ext in config.exclude_extensions:
        return False
    
    # Check if we should compress all files or only specific extensions
    if config.compress_all:
        return True
    else:
        return ext in config.compress_extensions


def compress_data(data: bytes, config: CompressionConfig) -> bytes:
    """Compress data using the specified algorithm.
    
    Args:
        data: Data to compress
        config: Compression configuration
        
    Returns:
        Compressed data
    """
    if config.algorithm == CompressionAlgorithm.NONE:
        return data
    
    try:
        # Compress the data
        if config.algorithm == CompressionAlgorithm.GZIP:
            compressed = gzip.compress(data, compresslevel=config.level)
        elif config.algorithm == CompressionAlgorithm.ZLIB:
            compressed = zlib.compress(data, level=config.level)
        elif config.algorithm == CompressionAlgorithm.BZ2:
            compressed = bz2.compress(data, compresslevel=config.level)
        elif config.algorithm == CompressionAlgorithm.LZMA:
            compressed = lzma.compress(data, preset=config.level)
        else:
            return data
        
        # Add compression header if configured
        if config.add_header:
            # Header format: AGNTRCMP + algorithm (1 byte) + level (1 byte)
            algorithm_byte = {
                CompressionAlgorithm.GZIP: b"\x01",
                CompressionAlgorithm.ZLIB: b"\x02",
                CompressionAlgorithm.BZ2: b"\x03",
                CompressionAlgorithm.LZMA: b"\x04"
            }.get(config.algorithm, b"\x00")
            
            level_byte = bytes([config.level])
            header = COMPRESSION_HEADER + algorithm_byte + level_byte
            
            return header + compressed
        else:
            return compressed
    except Exception as e:
        logger.warning(f"Compression failed: {e}")
        return data


def decompress_data(data: bytes, config: Optional[CompressionConfig] = None) -> bytes:
    """Decompress data.
    
    Args:
        data: Data to decompress
        config: Compression configuration (optional, used if no header is present)
        
    Returns:
        Decompressed data
    """
    # Check if the data has a compression header
    if len(data) >= 10 and data[:8] == COMPRESSION_HEADER:
        # Extract algorithm and level from the header
        algorithm_byte = data[8:9]
        level_byte = data[9:10]
        
        # Determine the algorithm
        algorithm = {
            b"\x01": CompressionAlgorithm.GZIP,
            b"\x02": CompressionAlgorithm.ZLIB,
            b"\x03": CompressionAlgorithm.BZ2,
            b"\x04": CompressionAlgorithm.LZMA
        }.get(algorithm_byte, CompressionAlgorithm.NONE)
        
        # Skip the header
        data = data[10:]
    elif config is not None:
        # Use the provided configuration
        algorithm = config.algorithm
    else:
        # No header and no configuration, return the data as is
        return data
    
    try:
        # Decompress the data
        if algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(data)
        elif algorithm == CompressionAlgorithm.ZLIB:
            return zlib.decompress(data)
        elif algorithm == CompressionAlgorithm.BZ2:
            return bz2.decompress(data)
        elif algorithm == CompressionAlgorithm.LZMA:
            return lzma.decompress(data)
        else:
            return data
    except Exception as e:
        logger.warning(f"Decompression failed: {e}")
        return data


def compress_text(text: str, encoding: str, config: CompressionConfig) -> bytes:
    """Compress text using the specified algorithm.
    
    Args:
        text: Text to compress
        encoding: Text encoding
        config: Compression configuration
        
    Returns:
        Compressed data
    """
    # Convert text to bytes
    data = text.encode(encoding)
    
    # Compress the data
    return compress_data(data, config)


def decompress_text(data: bytes, encoding: str, config: Optional[CompressionConfig] = None) -> str:
    """Decompress data to text.
    
    Args:
        data: Data to decompress
        encoding: Text encoding
        config: Compression configuration (optional, used if no header is present)
        
    Returns:
        Decompressed text
    """
    # Decompress the data
    decompressed = decompress_data(data, config)
    
    # Convert bytes to text
    return decompressed.decode(encoding)
