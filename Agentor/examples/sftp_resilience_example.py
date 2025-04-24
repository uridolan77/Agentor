"""
Example demonstrating the enhanced SFTP filesystem with resilience patterns.

This example shows how to use:
- SFTP filesystem with resilience patterns
- Configuration using Pydantic models
- Error handling with retry, circuit breaker, and timeout
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional

from agentor.interfaces.filesystem.remote.sftp_enhanced import SFTPEnhancedFilesystem
from agentor.interfaces.filesystem.config import SFTPFilesystemConfig
from agentor.core.config import get_config_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def main():
    """Run the SFTP resilience example."""
    logger.info("Starting SFTP resilience example")
    
    # Set up configuration
    config_manager = get_config_manager()
    
    # Register configuration with the config manager
    sftp_config = SFTPFilesystemConfig(
        host="sftp.example.com",
        port=22,
        username="demo",
        password="password",
        root_dir="/upload",
        key_path=None,
        key_passphrase=None,
        known_hosts_path=None,
        retry_max_attempts=5,
        retry_base_delay=1.0,
        retry_max_delay=30.0,
        timeout_seconds=30.0,
        circuit_breaker_failures=3,
        circuit_breaker_recovery=60
    )
    config_manager.set_config_section("filesystem.sftp", sftp_config.dict())
    
    # Create SFTP filesystem
    sftp = SFTPEnhancedFilesystem(
        name="example-sftp",
        host=sftp_config.host,
        port=sftp_config.port,
        username=sftp_config.username,
        password=sftp_config.password,
        root_dir=sftp_config.root_dir,
        private_key_path=sftp_config.key_path,
        private_key_passphrase=sftp_config.key_passphrase,
        known_hosts_path=sftp_config.known_hosts_path
    )
    
    try:
        # Connect to SFTP server
        logger.info("Connecting to SFTP server")
        result = await sftp.connect()
        if not result.success:
            logger.error(f"Failed to connect to SFTP server: {result.error}")
            return
        
        # Create a directory
        logger.info("Creating directory")
        result = await sftp.create_dir("test_dir", exist_ok=True)
        if not result.success:
            logger.error(f"Failed to create directory: {result.error}")
            return
        
        # Write a file
        logger.info("Writing file")
        content = "Hello, world!"
        result = await sftp.write_text("test_dir/test.txt", content)
        if not result.success:
            logger.error(f"Failed to write file: {result.error}")
            return
        
        # Read the file
        logger.info("Reading file")
        result = await sftp.read_text("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to read file: {result.error}")
            return
        
        logger.info(f"File content: {result.data}")
        
        # List directory
        logger.info("Listing directory")
        result = await sftp.list_dir("test_dir")
        if not result.success:
            logger.error(f"Failed to list directory: {result.error}")
            return
        
        logger.info(f"Directory contents: {result.data}")
        
        # Get file info
        logger.info("Getting file info")
        result = await sftp.get_info("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to get file info: {result.error}")
            return
        
        logger.info(f"File info: {result.data}")
        
        # Set modified time
        logger.info("Setting modified time")
        import time
        current_time = time.time()
        result = await sftp.set_modified_time("test_dir/test.txt", current_time)
        if not result.success:
            logger.error(f"Failed to set modified time: {result.error}")
            return
        
        # Get modified time
        logger.info("Getting modified time")
        result = await sftp.get_modified_time("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to get modified time: {result.error}")
            return
        
        logger.info(f"Modified time: {result.data}")
        
        # Copy the file
        logger.info("Copying file")
        result = await sftp.copy("test_dir/test.txt", "test_dir/test_copy.txt")
        if not result.success:
            logger.error(f"Failed to copy file: {result.error}")
            return
        
        # Move the file
        logger.info("Moving file")
        result = await sftp.move("test_dir/test_copy.txt", "test_dir/test_moved.txt")
        if not result.success:
            logger.error(f"Failed to move file: {result.error}")
            return
        
        # Delete the files
        logger.info("Deleting files")
        result = await sftp.delete_file("test_dir/test.txt")
        if not result.success:
            logger.error(f"Failed to delete file: {result.error}")
            return
        
        result = await sftp.delete_file("test_dir/test_moved.txt")
        if not result.success:
            logger.error(f"Failed to delete file: {result.error}")
            return
        
        # Delete the directory
        logger.info("Deleting directory")
        result = await sftp.delete_dir("test_dir")
        if not result.success:
            logger.error(f"Failed to delete directory: {result.error}")
            return
        
        logger.info("All operations completed successfully")
    except Exception as e:
        logger.exception(f"Error in example: {e}")
    finally:
        # Disconnect from SFTP server
        logger.info("Disconnecting from SFTP server")
        await sftp.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
