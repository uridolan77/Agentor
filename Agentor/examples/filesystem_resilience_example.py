"""
Example demonstrating the enhanced filesystem interfaces with resilience patterns.

This example shows how to use:
- FTP filesystem with resilience patterns
- Configuration using Pydantic models
- Error handling with retry, circuit breaker, and timeout
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional

from agentor.interfaces.filesystem.remote.ftp import FTPFilesystem
from agentor.interfaces.filesystem.config import FTPFilesystemConfig
from agentor.core.config import get_config_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def main():
    """Run the filesystem resilience example."""
    logger.info("Starting filesystem resilience example")
    
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
    
    # Create FTP filesystem
    ftp = FTPFilesystem(
        name="example-ftp",
        host=ftp_config.host,
        port=ftp_config.port,
        username=ftp_config.username,
        password=ftp_config.password,
        root_dir=ftp_config.root_dir,
        passive_mode=ftp_config.passive_mode,
        secure=ftp_config.secure
    )
    
    try:
        # Connect to FTP server
        logger.info("Connecting to FTP server")
        result = await ftp.connect()
        if not result.success:
            logger.error(f"Failed to connect to FTP server: {result.error}")
            return
        
        # Create a directory
        logger.info("Creating directory")
        result = await ftp.create_dir("test_dir", exist_ok=True)
        if not result.success:
            logger.error(f"Failed to create directory: {result.error}")
            return
        
        # Write a file
        logger.info("Writing file")
        content = "Hello, world!"
        result = await ftp.write_text("test_dir/test.txt", content)
        if not result.success:
            logger.error(f"Failed to write file: {result.error}")
            return
        
        # Read the file
        logger.info("Reading file")
        result = await ftp.read_text("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to read file: {result.error}")
            return
        
        logger.info(f"File content: {result.data}")
        
        # List directory
        logger.info("Listing directory")
        result = await ftp.list_dir("test_dir")
        if not result.success:
            logger.error(f"Failed to list directory: {result.error}")
            return
        
        logger.info(f"Directory contents: {result.data}")
        
        # Get file info
        logger.info("Getting file info")
        result = await ftp.get_info("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to get file info: {result.error}")
            return
        
        logger.info(f"File info: {result.data}")
        
        # Delete the file
        logger.info("Deleting file")
        result = await ftp.delete_file("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to delete file: {result.error}")
            return
        
        # Delete the directory
        logger.info("Deleting directory")
        result = await ftp.delete_dir("test_dir")
        if not result.success:
            logger.error(f"Failed to delete directory: {result.error}")
            return
        
        logger.info("All operations completed successfully")
    except Exception as e:
        logger.exception(f"Error in example: {e}")
    finally:
        # Disconnect from FTP server
        logger.info("Disconnecting from FTP server")
        await ftp.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
