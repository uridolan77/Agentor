"""
Configuration models for filesystem interfaces.

This module provides Pydantic models for configuring filesystem interfaces.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class FilesystemConfig(BaseModel):
    """Base configuration for filesystem interfaces."""
    
    # Resilience settings
    retry_max_attempts: int = Field(3, description="Maximum number of retry attempts")
    retry_base_delay: float = Field(1.0, description="Base delay between retries in seconds")
    retry_max_delay: float = Field(30.0, description="Maximum delay between retries in seconds")
    retry_jitter: float = Field(0.1, description="Jitter factor for randomizing retry delays")
    
    # Timeout settings
    timeout_seconds: float = Field(30.0, description="Timeout for operations in seconds")
    timeout_strategy: str = Field("adaptive", description="Timeout strategy (fixed, adaptive, percentile)")
    
    # Circuit breaker settings
    circuit_breaker_failures: int = Field(5, description="Number of failures before opening circuit")
    circuit_breaker_recovery: int = Field(60, description="Recovery timeout in seconds")
    circuit_breaker_half_open_calls: int = Field(1, description="Maximum calls in half-open state")
    
    # Connection settings
    connection_timeout: int = Field(30, description="Connection timeout in seconds")
    connection_retry_attempts: int = Field(3, description="Connection retry attempts")
    connection_retry_delay: float = Field(2.0, description="Delay between connection retries")
    
    # Cache settings
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_ttl: int = Field(300, description="Cache TTL in seconds")
    cache_max_size: int = Field(1000, description="Maximum cache size")


class LocalFilesystemConfig(FilesystemConfig):
    """Configuration for local filesystem interface."""
    
    root_dir: Optional[str] = Field(None, description="Root directory")
    create_dirs: bool = Field(True, description="Create directories if they don't exist")
    follow_symlinks: bool = Field(True, description="Follow symbolic links")


class FTPFilesystemConfig(FilesystemConfig):
    """Configuration for FTP filesystem interface."""
    
    host: str = Field(..., description="FTP server hostname")
    port: int = Field(21, description="FTP server port")
    username: str = Field("anonymous", description="FTP username")
    password: str = Field("", description="FTP password")
    root_dir: Optional[str] = Field(None, description="Root directory on the FTP server")
    passive_mode: bool = Field(True, description="Use passive mode")
    secure: bool = Field(False, description="Use secure FTP (FTPS)")
    keepalive_interval: int = Field(60, description="Keepalive interval in seconds")


class SFTPFilesystemConfig(FilesystemConfig):
    """Configuration for SFTP filesystem interface."""
    
    host: str = Field(..., description="SFTP server hostname")
    port: int = Field(22, description="SFTP server port")
    username: str = Field(..., description="SFTP username")
    password: Optional[str] = Field(None, description="SFTP password")
    key_path: Optional[str] = Field(None, description="Path to private key file")
    key_passphrase: Optional[str] = Field(None, description="Passphrase for private key")
    root_dir: Optional[str] = Field(None, description="Root directory on the SFTP server")
    known_hosts_path: Optional[str] = Field(None, description="Path to known_hosts file")
    keepalive_interval: int = Field(60, description="Keepalive interval in seconds")


class S3FilesystemConfig(FilesystemConfig):
    """Configuration for S3 filesystem interface."""
    
    bucket: str = Field(..., description="S3 bucket name")
    prefix: Optional[str] = Field("", description="Prefix for all paths")
    region: Optional[str] = Field(None, description="AWS region")
    access_key: Optional[str] = Field(None, description="AWS access key")
    secret_key: Optional[str] = Field(None, description="AWS secret key")
    endpoint_url: Optional[str] = Field(None, description="Custom endpoint URL for S3-compatible storage")
    use_ssl: bool = Field(True, description="Use SSL for connections")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    max_pool_connections: int = Field(10, description="Maximum connection pool size")
    max_attempts: int = Field(3, description="Maximum number of retry attempts")


class AzureBlobFilesystemConfig(FilesystemConfig):
    """Configuration for Azure Blob Storage filesystem interface."""
    
    container: str = Field(..., description="Azure Blob Storage container name")
    prefix: Optional[str] = Field("", description="Prefix for all paths")
    connection_string: Optional[str] = Field(None, description="Azure Storage connection string")
    account_name: Optional[str] = Field(None, description="Azure Storage account name")
    account_key: Optional[str] = Field(None, description="Azure Storage account key")
    sas_token: Optional[str] = Field(None, description="Azure Storage SAS token")
    max_concurrency: int = Field(10, description="Maximum concurrent connections")


class GCSFilesystemConfig(FilesystemConfig):
    """Configuration for Google Cloud Storage filesystem interface."""
    
    bucket: str = Field(..., description="GCS bucket name")
    prefix: Optional[str] = Field("", description="Prefix for all paths")
    project_id: Optional[str] = Field(None, description="GCP project ID")
    credentials_path: Optional[str] = Field(None, description="Path to service account credentials file")
    credentials_json: Optional[str] = Field(None, description="Service account credentials JSON")
    retry_timeout: float = Field(120.0, description="Retry timeout in seconds")
    max_retry_delay: float = Field(60.0, description="Maximum retry delay in seconds")


class WebDAVFilesystemConfig(FilesystemConfig):
    """Configuration for WebDAV filesystem interface."""
    
    url: str = Field(..., description="WebDAV server URL")
    username: Optional[str] = Field(None, description="WebDAV username")
    password: Optional[str] = Field(None, description="WebDAV password")
    root_path: Optional[str] = Field("/", description="Root path on the WebDAV server")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    cert_path: Optional[str] = Field(None, description="Path to client certificate")
    timeout: int = Field(30, description="Connection timeout in seconds")
