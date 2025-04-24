"""
Network registry for the Agentor framework.

This module provides a registry for network clients, allowing them to be
accessed by name from anywhere in the application.
"""

from typing import Dict, List, Optional, Any
import time

from .network import NetworkInterface, NetworkResult
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class NetworkRegistry:
    """Registry for network clients."""

    def __init__(self):
        self.clients: Dict[str, NetworkInterface] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # Cleanup every 60 seconds

    def register(self, client: NetworkInterface) -> None:
        """Register a network client."""
        if client.name in self.clients:
            logger.warning(f"Overwriting existing network client with name {client.name}")
        self.clients[client.name] = client
        logger.debug(f"Registered network client {client.name}")

    def unregister(self, name: str) -> None:
        """Unregister a network client."""
        if name in self.clients:
            del self.clients[name]
            logger.debug(f"Unregistered network client {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent network client {name}")

    def get(self, name: str) -> Optional[NetworkInterface]:
        """Get a network client by name."""
        return self.clients.get(name)

    def list_clients(self) -> List[str]:
        """List all registered network client names."""
        return list(self.clients.keys())

    async def close_all(self) -> None:
        """Close all network connections."""
        for name, client in self.clients.items():
            if client.connected:
                try:
                    result = await client.disconnect()
                    if result.success:
                        logger.debug(f"Closed network connection {name}")
                    else:
                        logger.error(f"Failed to close network connection {name}: {result.error}")
                except Exception as e:
                    logger.error(f"Error closing network connection {name}: {e}")

    def clear(self) -> None:
        """Clear all registered network clients."""
        self.clients.clear()
        logger.debug("Cleared all registered network clients")

    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        self.last_cleanup = current_time
        
        # Implement idle connection cleanup logic here if needed
        pass


# Global network registry
default_registry = NetworkRegistry()
