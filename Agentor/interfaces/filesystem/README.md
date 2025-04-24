# Filesystem Interfaces

This package provides interfaces for interacting with various filesystem types, including local filesystems, remote filesystems (FTP, SFTP), and cloud storage (S3, Azure Blob Storage, Google Cloud Storage).

## Overview

The filesystem interfaces provide a consistent API for performing operations on different types of filesystems. All filesystem implementations adhere to the `FilesystemInterface` base class, which defines common operations such as reading, writing, listing directories, etc.

## Key Features

- **Consistent API**: All filesystem implementations provide the same interface, making it easy to switch between different types of filesystems.
- **Resilience Patterns**: Implementations include retry, circuit breaker, timeout, and other resilience patterns to handle transient errors.
- **Type-Safe Configuration**: Pydantic models are used for configuration, providing type validation and documentation.
- **Comprehensive Error Handling**: Detailed error information is provided through the `FilesystemResult` class.
- **Async/Await Support**: All operations are asynchronous, allowing for efficient I/O operations.

## Filesystem Types

- **Local Filesystem**: Access to the local filesystem.
- **Remote Filesystems**:
  - **FTP**: File Transfer Protocol
  - **SFTP**: SSH File Transfer Protocol
  - **WebDAV**: Web Distributed Authoring and Versioning
- **Cloud Storage**:
  - **S3**: Amazon S3 and compatible services
  - **Azure Blob Storage**: Microsoft Azure Blob Storage
  - **Google Cloud Storage**: Google Cloud Storage

## Usage

### Basic Usage

```python
from agentor.interfaces.filesystem.local import LocalFilesystem
from agentor.interfaces.filesystem.remote.ftp import FTPFilesystem
from agentor.interfaces.filesystem.cloud.s3 import S3Filesystem

# Create a local filesystem
local_fs = LocalFilesystem(name="local")

# Create an FTP filesystem
ftp_fs = FTPFilesystem(
    name="example-ftp",
    host="ftp.example.com",
    port=21,
    username="user",
    password="password",
    root_dir="/upload"
)

# Create an S3 filesystem
s3_fs = S3Filesystem(
    name="example-s3",
    bucket="my-bucket",
    prefix="my-prefix",
    region="us-west-2"
)

# Use the filesystems
async def main():
    # Connect to the FTP server
    await ftp_fs.connect()
    
    # Read a file
    result = await ftp_fs.read_text("example.txt")
    if result.success:
        print(f"File content: {result.data}")
    else:
        print(f"Error: {result.error}")
    
    # Write a file
    result = await local_fs.write_text("local.txt", "Hello, world!")
    
    # List a directory
    result = await s3_fs.list_dir("my-directory")
    if result.success:
        print(f"Directory contents: {result.data}")
    
    # Disconnect from the FTP server
    await ftp_fs.disconnect()
```

### Configuration

Filesystem implementations can be configured using Pydantic models:

```python
from agentor.interfaces.filesystem.config import FTPFilesystemConfig
from agentor.core.config import get_config_manager

# Set up configuration
config_manager = get_config_manager()

# Register configuration with the config manager
ftp_config = FTPFilesystemConfig(
    host="ftp.example.com",
    port=21,
    username="demo",
    password="password",
    root_dir="/upload",
    passive_mode=True,
    secure=False,
    retry_max_attempts=5,
    retry_base_delay=1.0,
    retry_max_delay=30.0,
    timeout_seconds=30.0,
    circuit_breaker_failures=3,
    circuit_breaker_recovery=60
)
config_manager.set_config_section("filesystem.ftp", ftp_config.dict())

# Create FTP filesystem (will use the registered configuration)
ftp_fs = FTPFilesystem(
    name="example-ftp",
    host=ftp_config.host,
    port=ftp_config.port,
    username=ftp_config.username,
    password=ftp_config.password,
    root_dir=ftp_config.root_dir
)
```

### Resilience Patterns

The filesystem implementations include various resilience patterns to handle transient errors:

- **Retry**: Automatically retry operations that fail due to transient errors.
- **Circuit Breaker**: Prevent cascading failures by failing fast when a service is unavailable.
- **Timeout**: Prevent operations from hanging indefinitely.

These patterns are configured through the configuration models and can be customized for each filesystem instance.

## Error Handling

All filesystem operations return a `FilesystemResult` object, which includes:

- `success`: A boolean indicating whether the operation was successful.
- `data`: The result data (if successful).
- `error`: An error object (if unsuccessful).

This allows for consistent error handling across all filesystem operations.

## Testing

The filesystem interfaces include comprehensive tests, including unit tests and integration tests. Mock implementations are provided for testing code that uses the filesystem interfaces without requiring actual filesystem access.

## Examples

See the `examples` directory for more examples of using the filesystem interfaces.
