"""
Example comparing the original SFTP implementation with the enhanced version.

This example shows the differences between:
- Original SFTP implementation without resilience patterns
- Enhanced SFTP implementation with resilience patterns
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, List, Optional

from agentor.interfaces.filesystem.remote.sftp import SFTPFilesystem
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


async def test_original_sftp():
    """Test the original SFTP implementation."""
    logger.info("Testing original SFTP implementation")
    
    # Create SFTP filesystem
    sftp = SFTPFilesystem(
        name="original-sftp",
        host="sftp.example.com",
        port=22,
        username="demo",
        password="password",
        root_dir="/upload"
    )
    
    try:
        # Connect to SFTP server
        logger.info("Connecting to SFTP server")
        start_time = time.time()
        result = await sftp.connect()
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to connect to SFTP server: {result.error}")
            return
        
        logger.info(f"Connection time: {end_time - start_time:.2f} seconds")
        
        # Write a file
        logger.info("Writing file")
        content = "Hello, world!"
        start_time = time.time()
        result = await sftp.write_text("test.txt", content)
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to write file: {result.error}")
            return
        
        logger.info(f"Write time: {end_time - start_time:.2f} seconds")
        
        # Read the file
        logger.info("Reading file")
        start_time = time.time()
        result = await sftp.read_text("test.txt")
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to read file: {result.error}")
            return
        
        logger.info(f"Read time: {end_time - start_time:.2f} seconds")
        logger.info(f"File content: {result.data}")
        
        # Delete the file
        logger.info("Deleting file")
        start_time = time.time()
        result = await sftp.delete_file("test.txt")
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to delete file: {result.error}")
            return
        
        logger.info(f"Delete time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.exception(f"Error in original SFTP test: {e}")
    finally:
        # Disconnect from SFTP server
        logger.info("Disconnecting from SFTP server")
        await sftp.disconnect()


async def test_enhanced_sftp():
    """Test the enhanced SFTP implementation."""
    logger.info("Testing enhanced SFTP implementation")
    
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
        retry_max_attempts=3,
        retry_base_delay=1.0,
        retry_max_delay=5.0,
        timeout_seconds=10.0,
        circuit_breaker_failures=3,
        circuit_breaker_recovery=30
    )
    config_manager.set_config_section("filesystem.sftp", sftp_config.dict())
    
    # Create enhanced SFTP filesystem
    sftp = SFTPEnhancedFilesystem(
        name="enhanced-sftp",
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
        start_time = time.time()
        result = await sftp.connect()
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to connect to SFTP server: {result.error}")
            return
        
        logger.info(f"Connection time: {end_time - start_time:.2f} seconds")
        
        # Write a file
        logger.info("Writing file")
        content = "Hello, world!"
        start_time = time.time()
        result = await sftp.write_text("test.txt", content)
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to write file: {result.error}")
            return
        
        logger.info(f"Write time: {end_time - start_time:.2f} seconds")
        
        # Read the file
        logger.info("Reading file")
        start_time = time.time()
        result = await sftp.read_text("test.txt")
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to read file: {result.error}")
            return
        
        logger.info(f"Read time: {end_time - start_time:.2f} seconds")
        logger.info(f"File content: {result.data}")
        
        # Delete the file
        logger.info("Deleting file")
        start_time = time.time()
        result = await sftp.delete_file("test.txt")
        end_time = time.time()
        
        if not result.success:
            logger.error(f"Failed to delete file: {result.error}")
            return
        
        logger.info(f"Delete time: {end_time - start_time:.2f} seconds")
        
        # Simulate a failure scenario
        logger.info("Simulating a failure scenario")
        
        # Try to read a non-existent file
        logger.info("Trying to read a non-existent file")
        start_time = time.time()
        result = await sftp.read_text("nonexistent.txt")
        end_time = time.time()
        
        logger.info(f"Failed operation time: {end_time - start_time:.2f} seconds")
        logger.info(f"Error: {result.error}")
        
    except Exception as e:
        logger.exception(f"Error in enhanced SFTP test: {e}")
    finally:
        # Disconnect from SFTP server
        logger.info("Disconnecting from SFTP server")
        await sftp.disconnect()


async def main():
    """Run the SFTP comparison example."""
    logger.info("Starting SFTP comparison example")
    
    # Test original SFTP implementation
    await test_original_sftp()
    
    logger.info("-" * 80)
    
    # Test enhanced SFTP implementation
    await test_enhanced_sftp()
    
    logger.info("SFTP comparison example completed")


if __name__ == "__main__":
    asyncio.run(main())
