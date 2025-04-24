"""
Encryption utilities for the Agentor framework.

This module provides encryption utilities for filesystem operations.
"""

import os
import base64
import hashlib
import hmac
import secrets
from enum import Enum
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import cryptography
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding, hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logger.warning("cryptography package not available, using fallback encryption")
    CRYPTOGRAPHY_AVAILABLE = False


class EncryptionAlgorithm(str, Enum):
    """Encryption algorithms."""
    
    # No encryption
    NONE = "none"
    
    # AES-256-GCM (requires cryptography)
    AES_GCM = "aes-gcm"
    
    # AES-256-CBC (requires cryptography)
    AES_CBC = "aes-cbc"
    
    # ChaCha20-Poly1305 (requires cryptography)
    CHACHA20 = "chacha20"
    
    # XChaCha20-Poly1305 (requires cryptography)
    XCHACHA20 = "xchacha20"
    
    # Fernet (requires cryptography)
    FERNET = "fernet"


class EncryptionConfig(BaseModel):
    """Configuration for encryption."""
    
    # Encryption algorithm
    algorithm: EncryptionAlgorithm = Field(
        EncryptionAlgorithm.AES_GCM if CRYPTOGRAPHY_AVAILABLE else EncryptionAlgorithm.NONE,
        description="Encryption algorithm to use"
    )
    
    # Encryption key (base64-encoded)
    key: Optional[str] = Field(
        None,
        description="Encryption key (base64-encoded)"
    )
    
    # Key derivation iterations
    key_iterations: int = Field(
        100000,
        description="Key derivation iterations"
    )
    
    # Whether to encrypt all files or only specific extensions
    encrypt_all: bool = Field(
        False,
        description="Whether to encrypt all files or only specific extensions"
    )
    
    # List of file extensions to encrypt (if encrypt_all is False)
    encrypt_extensions: List[str] = Field(
        [".txt", ".csv", ".json", ".xml", ".html", ".md", ".log", ".key", ".pem", ".env"],
        description="List of file extensions to encrypt (if encrypt_all is False)"
    )
    
    # List of file extensions to exclude from encryption
    exclude_extensions: List[str] = Field(
        [".enc", ".encrypted", ".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4", ".avi", ".mov"],
        description="List of file extensions to exclude from encryption"
    )
    
    # Minimum file size to encrypt (in bytes)
    min_size: int = Field(
        0,
        description="Minimum file size to encrypt (in bytes)"
    )
    
    # Maximum file size to encrypt (in bytes, 0 = no limit)
    max_size: int = Field(
        1024 * 1024 * 100,  # 100 MB
        description="Maximum file size to encrypt (in bytes, 0 = no limit)"
    )
    
    # Whether to add an encryption header to identify encrypted files
    add_header: bool = Field(
        True,
        description="Whether to add an encryption header to identify encrypted files"
    )
    
    # Whether to verify data integrity
    verify_integrity: bool = Field(
        True,
        description="Whether to verify data integrity"
    )


# Encryption header (8 bytes)
ENCRYPTION_HEADER = b"AGNTRENC"


def should_encrypt(path: str, config: EncryptionConfig, size: int = 0) -> bool:
    """Check if a file should be encrypted.
    
    Args:
        path: File path
        config: Encryption configuration
        size: File size in bytes (if known)
        
    Returns:
        True if the file should be encrypted, False otherwise
    """
    # Check if encryption is disabled
    if config.algorithm == EncryptionAlgorithm.NONE:
        return False
    
    # Check if cryptography is available
    if not CRYPTOGRAPHY_AVAILABLE:
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
    
    # Check if we should encrypt all files or only specific extensions
    if config.encrypt_all:
        return True
    else:
        return ext in config.encrypt_extensions


def derive_key(password: str, salt: bytes = None, iterations: int = 100000) -> Tuple[bytes, bytes]:
    """Derive an encryption key from a password.
    
    Args:
        password: Password to derive the key from
        salt: Salt for key derivation (generated if not provided)
        iterations: Number of iterations for key derivation
        
    Returns:
        Tuple of (key, salt)
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        # Fallback implementation
        if salt is None:
            salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, 32)
        return key, salt
    
    # Use cryptography for key derivation
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 256 bits
        salt=salt,
        iterations=iterations,
        backend=default_backend()
    )
    
    key = kdf.derive(password.encode())
    return key, salt


def get_encryption_key(config: EncryptionConfig) -> bytes:
    """Get the encryption key from the configuration.
    
    Args:
        config: Encryption configuration
        
    Returns:
        Encryption key
    """
    if config.key:
        # Key is provided in the configuration
        try:
            return base64.b64decode(config.key)
        except Exception as e:
            logger.error(f"Failed to decode encryption key: {e}")
            # Generate a random key
            return os.urandom(32)
    else:
        # Generate a random key
        return os.urandom(32)


def encrypt_data(data: bytes, config: EncryptionConfig) -> bytes:
    """Encrypt data using the specified algorithm.
    
    Args:
        data: Data to encrypt
        config: Encryption configuration
        
    Returns:
        Encrypted data
    """
    if config.algorithm == EncryptionAlgorithm.NONE:
        return data
    
    if not CRYPTOGRAPHY_AVAILABLE:
        logger.warning("Encryption requested but cryptography package not available")
        return data
    
    try:
        # Get the encryption key
        key = get_encryption_key(config)
        
        # Encrypt the data
        if config.algorithm == EncryptionAlgorithm.AES_GCM:
            # Generate a random nonce
            nonce = os.urandom(12)
            
            # Create an encryptor
            encryptor = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce),
                backend=default_backend()
            ).encryptor()
            
            # Add additional data for integrity
            if config.verify_integrity:
                encryptor.authenticate_additional_data(ENCRYPTION_HEADER)
            
            # Encrypt the data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Get the tag
            tag = encryptor.tag
            
            # Combine nonce, tag, and ciphertext
            encrypted = nonce + tag + ciphertext
            
        elif config.algorithm == EncryptionAlgorithm.AES_CBC:
            # Generate a random IV
            iv = os.urandom(16)
            
            # Create a padder
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            # Create an encryptor
            encryptor = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            ).encryptor()
            
            # Encrypt the data
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Calculate HMAC for integrity
            if config.verify_integrity:
                hmac_key = hashlib.sha256(key).digest()
                mac = hmac.new(hmac_key, iv + ciphertext, hashlib.sha256).digest()
                encrypted = iv + mac + ciphertext
            else:
                encrypted = iv + ciphertext
            
        elif config.algorithm == EncryptionAlgorithm.CHACHA20:
            # Generate a random nonce
            nonce = os.urandom(16)
            
            # Create an encryptor
            encryptor = Cipher(
                algorithms.ChaCha20(key, nonce),
                mode=None,
                backend=default_backend()
            ).encryptor()
            
            # Encrypt the data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Calculate HMAC for integrity
            if config.verify_integrity:
                hmac_key = hashlib.sha256(key).digest()
                mac = hmac.new(hmac_key, nonce + ciphertext, hashlib.sha256).digest()
                encrypted = nonce + mac + ciphertext
            else:
                encrypted = nonce + ciphertext
            
        else:
            # Unsupported algorithm
            logger.warning(f"Unsupported encryption algorithm: {config.algorithm}")
            return data
        
        # Add encryption header if configured
        if config.add_header:
            # Header format: AGNTRENC + algorithm (1 byte)
            algorithm_byte = {
                EncryptionAlgorithm.AES_GCM: b"\x01",
                EncryptionAlgorithm.AES_CBC: b"\x02",
                EncryptionAlgorithm.CHACHA20: b"\x03",
                EncryptionAlgorithm.XCHACHA20: b"\x04",
                EncryptionAlgorithm.FERNET: b"\x05"
            }.get(config.algorithm, b"\x00")
            
            # Add integrity flag
            integrity_byte = b"\x01" if config.verify_integrity else b"\x00"
            
            header = ENCRYPTION_HEADER + algorithm_byte + integrity_byte
            
            return header + encrypted
        else:
            return encrypted
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return data


def decrypt_data(data: bytes, config: Optional[EncryptionConfig] = None) -> bytes:
    """Decrypt data.
    
    Args:
        data: Data to decrypt
        config: Encryption configuration (optional, used if no header is present)
        
    Returns:
        Decrypted data
    """
    # Check if the data has an encryption header
    if len(data) >= 10 and data[:8] == ENCRYPTION_HEADER:
        # Extract algorithm and integrity flag from the header
        algorithm_byte = data[8:9]
        integrity_byte = data[9:10]
        
        # Determine the algorithm
        algorithm = {
            b"\x01": EncryptionAlgorithm.AES_GCM,
            b"\x02": EncryptionAlgorithm.AES_CBC,
            b"\x03": EncryptionAlgorithm.CHACHA20,
            b"\x04": EncryptionAlgorithm.XCHACHA20,
            b"\x05": EncryptionAlgorithm.FERNET
        }.get(algorithm_byte, EncryptionAlgorithm.NONE)
        
        # Determine if integrity verification is enabled
        verify_integrity = integrity_byte == b"\x01"
        
        # Skip the header
        data = data[10:]
    elif config is not None:
        # Use the provided configuration
        algorithm = config.algorithm
        verify_integrity = config.verify_integrity
    else:
        # No header and no configuration, return the data as is
        return data
    
    if algorithm == EncryptionAlgorithm.NONE:
        return data
    
    if not CRYPTOGRAPHY_AVAILABLE:
        logger.warning("Decryption requested but cryptography package not available")
        return data
    
    try:
        # Get the encryption key
        key = get_encryption_key(config) if config else os.urandom(32)
        
        # Decrypt the data
        if algorithm == EncryptionAlgorithm.AES_GCM:
            # Extract nonce, tag, and ciphertext
            nonce = data[:12]
            tag = data[12:28]
            ciphertext = data[28:]
            
            # Create a decryptor
            decryptor = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            ).decryptor()
            
            # Add additional data for integrity
            if verify_integrity:
                decryptor.authenticate_additional_data(ENCRYPTION_HEADER)
            
            # Decrypt the data
            return decryptor.update(ciphertext) + decryptor.finalize()
            
        elif algorithm == EncryptionAlgorithm.AES_CBC:
            # Extract IV, MAC (if present), and ciphertext
            iv = data[:16]
            
            if verify_integrity:
                mac = data[16:48]
                ciphertext = data[48:]
                
                # Verify the MAC
                hmac_key = hashlib.sha256(key).digest()
                expected_mac = hmac.new(hmac_key, iv + ciphertext, hashlib.sha256).digest()
                if not hmac.compare_digest(mac, expected_mac):
                    raise ValueError("MAC verification failed")
            else:
                ciphertext = data[16:]
            
            # Create a decryptor
            decryptor = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend()
            ).decryptor()
            
            # Decrypt the data
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()
            
        elif algorithm == EncryptionAlgorithm.CHACHA20:
            # Extract nonce, MAC (if present), and ciphertext
            nonce = data[:16]
            
            if verify_integrity:
                mac = data[16:48]
                ciphertext = data[48:]
                
                # Verify the MAC
                hmac_key = hashlib.sha256(key).digest()
                expected_mac = hmac.new(hmac_key, nonce + ciphertext, hashlib.sha256).digest()
                if not hmac.compare_digest(mac, expected_mac):
                    raise ValueError("MAC verification failed")
            else:
                ciphertext = data[16:]
            
            # Create a decryptor
            decryptor = Cipher(
                algorithms.ChaCha20(key, nonce),
                mode=None,
                backend=default_backend()
            ).decryptor()
            
            # Decrypt the data
            return decryptor.update(ciphertext) + decryptor.finalize()
            
        else:
            # Unsupported algorithm
            logger.warning(f"Unsupported decryption algorithm: {algorithm}")
            return data
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return data


def encrypt_text(text: str, encoding: str, config: EncryptionConfig) -> bytes:
    """Encrypt text using the specified algorithm.
    
    Args:
        text: Text to encrypt
        encoding: Text encoding
        config: Encryption configuration
        
    Returns:
        Encrypted data
    """
    # Convert text to bytes
    data = text.encode(encoding)
    
    # Encrypt the data
    return encrypt_data(data, config)


def decrypt_text(data: bytes, encoding: str, config: Optional[EncryptionConfig] = None) -> str:
    """Decrypt data to text.
    
    Args:
        data: Data to decrypt
        encoding: Text encoding
        config: Encryption configuration (optional, used if no header is present)
        
    Returns:
        Decrypted text
    """
    # Decrypt the data
    decrypted = decrypt_data(data, config)
    
    # Convert bytes to text
    return decrypted.decode(encoding)
