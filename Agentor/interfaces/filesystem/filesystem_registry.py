"""
Filesystem registry for the Agentor framework.

This module provides a registry for filesystem implementations, allowing them to be
accessed by name from anywhere in the application.
"""

from typing import Dict, List, Optional, Any
import time

from .base import FilesystemInterface, FilesystemResult
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class FilesystemRegistry:
    """Registry for filesystem implementations."""

    def __init__(self):
        self.filesystems: Dict[str, FilesystemInterface] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # Cleanup every 60 seconds

    def register(self, filesystem: FilesystemInterface) -> None:
        """Register a filesystem implementation."""
        if filesystem.name in self.filesystems:
            logger.warning(f"Overwriting existing filesystem with name {filesystem.name}")
        self.filesystems[filesystem.name] = filesystem
        logger.debug(f"Registered filesystem {filesystem.name}")

    def unregister(self, name: str) -> None:
        """Unregister a filesystem implementation."""
        if name in self.filesystems:
            del self.filesystems[name]
            logger.debug(f"Unregistered filesystem {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent filesystem {name}")

    def get(self, name: str) -> Optional[FilesystemInterface]:
        """Get a filesystem implementation by name."""
        return self.filesystems.get(name)

    def list_filesystems(self) -> List[str]:
        """List all registered filesystem names."""
        return list(self.filesystems.keys())

    async def close_all(self) -> None:
        """Close all filesystem connections."""
        for name, filesystem in self.filesystems.items():
            if filesystem.connected:
                try:
                    result = await filesystem.disconnect()
                    if result.success:
                        logger.debug(f"Closed filesystem connection {name}")
                    else:
                        logger.error(f"Failed to close filesystem connection {name}: {result.error}")
                except Exception as e:
                    logger.error(f"Error closing filesystem connection {name}: {e}")

    def clear(self) -> None:
        """Clear all registered filesystems."""
        self.filesystems.clear()
        logger.debug("Cleared all registered filesystems")

    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        # Implement idle connection cleanup logic here if needed
        pass


# Global filesystem registry
default_registry = FilesystemRegistry()
