"""
Transport security hardening for the LLM Gateway.

This module provides TLS configuration, certificate pinning, and other
transport security features.
"""

import os
import ssl
import logging
import hashlib
import base64
from typing import Dict, List, Optional, Set, Union, Any
import httpx
import certifi
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


class TransportSecurityError(Exception):
    """Error raised for transport security operations."""
    pass


class TLSConfig:
    """TLS configuration for secure connections."""
    
    # Default cipher suites (TLS 1.3 and strong TLS 1.2)
    DEFAULT_CIPHER_SUITES = [
        # TLS 1.3 cipher suites
        "TLS_AES_256_GCM_SHA384",
        "TLS_AES_128_GCM_SHA256",
        "TLS_CHACHA20_POLY1305_SHA256",
        
        # Strong TLS 1.2 cipher suites
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
        "ECDHE-RSA-CHACHA20-POLY1305"
    ]
    
    def __init__(
        self,
        min_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_2,
        max_version: ssl.TLSVersion = ssl.TLSVersion.TLSv1_3,
        cipher_suites: Optional[List[str]] = None,
        verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED,
        check_hostname: bool = True,
        ca_certs_path: Optional[str] = None
    ):
        """
        Initialize TLS configuration.
        
        Args:
            min_version: Minimum TLS version
            max_version: Maximum TLS version
            cipher_suites: Allowed cipher suites
            verify_mode: Certificate verification mode
            check_hostname: Whether to check hostname
            ca_certs_path: Path to CA certificates file
        """
        self.min_version = min_version
        self.max_version = max_version
        self.cipher_suites = cipher_suites or self.DEFAULT_CIPHER_SUITES
        self.verify_mode = verify_mode
        self.check_hostname = check_hostname
        self.ca_certs_path = ca_certs_path or certifi.where()
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """
        Create an SSL context with the configured settings.
        
        Returns:
            SSL context
        """
        # Create SSL context
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Set TLS versions
        context.minimum_version = self.min_version
        context.maximum_version = self.max_version
        
        # Set cipher suites
        if hasattr(context, "set_ciphers"):
            context.set_ciphers(":".join(self.cipher_suites))
        
        # Set verification mode
        context.verify_mode = self.verify_mode
        context.check_hostname = self.check_hostname
        
        # Load CA certificates
        if self.ca_certs_path:
            context.load_verify_locations(self.ca_certs_path)
        
        # Disable compression (to prevent CRIME attack)
        context.options |= ssl.OP_NO_COMPRESSION
        
        # Enable OCSP stapling
        if hasattr(ssl, "OP_ENABLE_OCSP_STAPLING"):
            context.options |= ssl.OP_ENABLE_OCSP_STAPLING
        
        return context
    
    def create_httpx_client(self, **kwargs) -> httpx.Client:
        """
        Create an HTTPX client with the configured TLS settings.
        
        Args:
            **kwargs: Additional arguments for the HTTPX client
            
        Returns:
            HTTPX client
        """
        # Create SSL context
        ssl_context = self.create_ssl_context()
        
        # Create HTTPX client
        return httpx.Client(verify=ssl_context, **kwargs)
    
    async def create_async_httpx_client(self, **kwargs) -> httpx.AsyncClient:
        """
        Create an async HTTPX client with the configured TLS settings.
        
        Args:
            **kwargs: Additional arguments for the HTTPX client
            
        Returns:
            Async HTTPX client
        """
        # Create SSL context
        ssl_context = self.create_ssl_context()
        
        # Create async HTTPX client
        return httpx.AsyncClient(verify=ssl_context, **kwargs)


class CertificatePinningManager:
    """Certificate pinning manager for secure connections."""
    
    def __init__(self, pins: Optional[Dict[str, List[str]]] = None):
        """
        Initialize certificate pinning manager.
        
        Args:
            pins: Dictionary mapping hostnames to lists of pinned public key hashes
        """
        self.pins = pins or {}
    
    def add_pin(self, hostname: str, pin: str) -> None:
        """
        Add a certificate pin for a hostname.
        
        Args:
            hostname: Hostname
            pin: Public key hash in base64 format (sha256/base64)
        """
        if hostname not in self.pins:
            self.pins[hostname] = []
        
        if pin not in self.pins[hostname]:
            self.pins[hostname].append(pin)
            logger.info(f"Added certificate pin for {hostname}: {pin}")
    
    def remove_pin(self, hostname: str, pin: str) -> bool:
        """
        Remove a certificate pin for a hostname.
        
        Args:
            hostname: Hostname
            pin: Public key hash in base64 format (sha256/base64)
            
        Returns:
            True if pin was removed, False if not found
        """
        if hostname in self.pins and pin in self.pins[hostname]:
            self.pins[hostname].remove(pin)
            logger.info(f"Removed certificate pin for {hostname}: {pin}")
            
            # Remove hostname if no pins left
            if not self.pins[hostname]:
                del self.pins[hostname]
            
            return True
        
        return False
    
    def get_pins(self, hostname: str) -> List[str]:
        """
        Get certificate pins for a hostname.
        
        Args:
            hostname: Hostname
            
        Returns:
            List of public key hashes in base64 format (sha256/base64)
        """
        return self.pins.get(hostname, [])
    
    def verify_certificate(self, hostname: str, certificate: ssl.SSLObject) -> bool:
        """
        Verify a certificate against pinned public key hashes.
        
        Args:
            hostname: Hostname
            certificate: SSL certificate
            
        Returns:
            True if certificate matches a pinned hash, False otherwise
        """
        # Get pins for hostname
        pins = self.get_pins(hostname)
        if not pins:
            # No pins for this hostname, so it passes
            return True
        
        # Get certificate public key hash
        cert_hash = self._get_certificate_hash(certificate)
        
        # Check if hash matches any pin
        return cert_hash in pins
    
    def _get_certificate_hash(self, certificate: ssl.SSLObject) -> str:
        """
        Get the SHA-256 hash of a certificate's public key.
        
        Args:
            certificate: SSL certificate
            
        Returns:
            Public key hash in base64 format (sha256/base64)
        """
        # Get DER-encoded certificate
        der_cert = certificate.getpeercert(binary_form=True)
        
        # Parse certificate
        cert = x509.load_der_x509_certificate(der_cert, default_backend())
        
        # Get public key in DER format
        public_key = cert.public_key()
        public_key_der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Calculate SHA-256 hash
        digest = hashlib.sha256(public_key_der).digest()
        
        # Encode as base64
        return base64.b64encode(digest).decode("ascii")
    
    def create_ssl_context(self, hostname: str) -> ssl.SSLContext:
        """
        Create an SSL context with certificate pinning for a hostname.
        
        Args:
            hostname: Hostname
            
        Returns:
            SSL context with certificate pinning
        """
        # Create base SSL context
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Get pins for hostname
        pins = self.get_pins(hostname)
        if not pins:
            # No pins for this hostname, so just return the context
            return context
        
        # Set up certificate verification callback
        def verify_callback(conn, cert, errno, depth, result):
            if depth == 0:  # Only check the leaf certificate
                cert_hash = self._get_certificate_hash(cert)
                if cert_hash not in pins:
                    return False
            return result
        
        # Set verification callback
        context.verify_callback = verify_callback
        
        return context
    
    def create_httpx_client(self, hostname: str, **kwargs) -> httpx.Client:
        """
        Create an HTTPX client with certificate pinning for a hostname.
        
        Args:
            hostname: Hostname
            **kwargs: Additional arguments for the HTTPX client
            
        Returns:
            HTTPX client with certificate pinning
        """
        # Create SSL context with certificate pinning
        ssl_context = self.create_ssl_context(hostname)
        
        # Create HTTPX client
        return httpx.Client(verify=ssl_context, **kwargs)
    
    async def create_async_httpx_client(self, hostname: str, **kwargs) -> httpx.AsyncClient:
        """
        Create an async HTTPX client with certificate pinning for a hostname.
        
        Args:
            hostname: Hostname
            **kwargs: Additional arguments for the HTTPX client
            
        Returns:
            Async HTTPX client with certificate pinning
        """
        # Create SSL context with certificate pinning
        ssl_context = self.create_ssl_context(hostname)
        
        # Create async HTTPX client
        return httpx.AsyncClient(verify=ssl_context, **kwargs)
    
    @classmethod
    def from_certificate(cls, hostname: str, cert_path: str) -> "CertificatePinningManager":
        """
        Create a certificate pinning manager from a certificate file.
        
        Args:
            hostname: Hostname
            cert_path: Path to certificate file
            
        Returns:
            Certificate pinning manager
        """
        # Load certificate
        with open(cert_path, "rb") as f:
            cert_data = f.read()
        
        # Parse certificate
        cert = x509.load_pem_x509_certificate(cert_data, default_backend())
        
        # Get public key in DER format
        public_key = cert.public_key()
        public_key_der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Calculate SHA-256 hash
        digest = hashlib.sha256(public_key_der).digest()
        
        # Encode as base64
        pin = base64.b64encode(digest).decode("ascii")
        
        # Create manager with pin
        manager = cls()
        manager.add_pin(hostname, pin)
        
        return manager


class SecureTransportManager:
    """Manager for secure transport connections."""
    
    def __init__(
        self,
        tls_config: Optional[TLSConfig] = None,
        cert_pinning_manager: Optional[CertificatePinningManager] = None
    ):
        """
        Initialize secure transport manager.
        
        Args:
            tls_config: TLS configuration
            cert_pinning_manager: Certificate pinning manager
        """
        self.tls_config = tls_config or TLSConfig()
        self.cert_pinning_manager = cert_pinning_manager or CertificatePinningManager()
    
    def create_ssl_context(self, hostname: Optional[str] = None) -> ssl.SSLContext:
        """
        Create an SSL context with the configured settings.
        
        Args:
            hostname: Hostname for certificate pinning
            
        Returns:
            SSL context
        """
        if hostname and self.cert_pinning_manager.get_pins(hostname):
            # Use certificate pinning context
            return self.cert_pinning_manager.create_ssl_context(hostname)
        else:
            # Use TLS config context
            return self.tls_config.create_ssl_context()
    
    def create_httpx_client(self, hostname: Optional[str] = None, **kwargs) -> httpx.Client:
        """
        Create an HTTPX client with the configured security settings.
        
        Args:
            hostname: Hostname for certificate pinning
            **kwargs: Additional arguments for the HTTPX client
            
        Returns:
            HTTPX client
        """
        if hostname and self.cert_pinning_manager.get_pins(hostname):
            # Use certificate pinning client
            return self.cert_pinning_manager.create_httpx_client(hostname, **kwargs)
        else:
            # Use TLS config client
            return self.tls_config.create_httpx_client(**kwargs)
    
    async def create_async_httpx_client(self, hostname: Optional[str] = None, **kwargs) -> httpx.AsyncClient:
        """
        Create an async HTTPX client with the configured security settings.
        
        Args:
            hostname: Hostname for certificate pinning
            **kwargs: Additional arguments for the HTTPX client
            
        Returns:
            Async HTTPX client
        """
        if hostname and self.cert_pinning_manager.get_pins(hostname):
            # Use certificate pinning client
            return await self.cert_pinning_manager.create_async_httpx_client(hostname, **kwargs)
        else:
            # Use TLS config client
            return await self.tls_config.create_async_httpx_client(**kwargs)
    
    def verify_certificate(self, hostname: str, certificate: ssl.SSLObject) -> bool:
        """
        Verify a certificate against pinned public key hashes.
        
        Args:
            hostname: Hostname
            certificate: SSL certificate
            
        Returns:
            True if certificate matches a pinned hash, False otherwise
        """
        return self.cert_pinning_manager.verify_certificate(hostname, certificate)
    
    def add_certificate_pin(self, hostname: str, pin: str) -> None:
        """
        Add a certificate pin for a hostname.
        
        Args:
            hostname: Hostname
            pin: Public key hash in base64 format (sha256/base64)
        """
        self.cert_pinning_manager.add_pin(hostname, pin)
    
    def remove_certificate_pin(self, hostname: str, pin: str) -> bool:
        """
        Remove a certificate pin for a hostname.
        
        Args:
            hostname: Hostname
            pin: Public key hash in base64 format (sha256/base64)
            
        Returns:
            True if pin was removed, False if not found
        """
        return self.cert_pinning_manager.remove_pin(hostname, pin)
    
    def get_certificate_pins(self, hostname: str) -> List[str]:
        """
        Get certificate pins for a hostname.
        
        Args:
            hostname: Hostname
            
        Returns:
            List of public key hashes in base64 format (sha256/base64)
        """
        return self.cert_pinning_manager.get_pins(hostname)
