"""
Database registry for the Agentor framework.

This module provides a registry for database connections, allowing them to be
accessed by name from anywhere in the application.
"""

from typing import Dict, List, Optional, Any
import asyncio
import time

from .base import DatabaseConnection, DatabaseResult, ConnectionError
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseRegistry:
    """Registry for database connections."""

    def __init__(self):
        self.connections: Dict[str, DatabaseConnection] = {}
        self.connection_pool: Dict[str, List[DatabaseConnection]] = {}
        self.pool_size: Dict[str, int] = {}
        self.idle_timeout: Dict[str, float] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 60.0  # Cleanup every 60 seconds

    def register(self, connection: DatabaseConnection) -> None:
        """Register a database connection."""
        if connection.name in self.connections:
            logger.warning(f"Overwriting existing database connection with name {connection.name}")
        self.connections[connection.name] = connection
        logger.debug(f"Registered database connection {connection.name}")

    def unregister(self, name: str) -> None:
        """Unregister a database connection."""
        if name in self.connections:
            del self.connections[name]
            logger.debug(f"Unregistered database connection {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent database connection {name}")

    def get(self, name: str) -> Optional[DatabaseConnection]:
        """Get a database connection by name."""
        return self.connections.get(name)

    def list_connections(self) -> List[str]:
        """List all registered database connection names."""
        return list(self.connections.keys())

    async def close_all(self) -> None:
        """Close all database connections."""
        for name, connection in self.connections.items():
            if connection.connected:
                try:
                    result = await connection.disconnect()
                    if result.success:
                        logger.debug(f"Closed database connection {name}")
                    else:
                        logger.error(f"Failed to close database connection {name}: {result.error}")
                except Exception as e:
                    logger.error(f"Error closing database connection {name}: {e}")

        # Close all pooled connections
        for name, pool in self.connection_pool.items():
            for connection in pool:
                if connection.connected:
                    try:
                        result = await connection.disconnect()
                        if not result.success:
                            logger.error(f"Failed to close pooled database connection {name}: {result.error}")
                    except Exception as e:
                        logger.error(f"Error closing pooled database connection {name}: {e}")
            self.connection_pool[name] = []

    def clear(self) -> None:
        """Clear all registered database connections."""
        self.connections.clear()
        logger.debug("Cleared all registered database connections")

    def register_pool(self, connection_factory, name: str, pool_size: int = 5, idle_timeout: float = 300.0) -> None:
        """Register a database connection pool."""
        if name in self.connection_pool:
            logger.warning(f"Overwriting existing database connection pool with name {name}")
        self.connection_pool[name] = []
        self.pool_size[name] = pool_size
        self.idle_timeout[name] = idle_timeout
        self._connection_factory = connection_factory
        logger.debug(f"Registered database connection pool {name} with size {pool_size}")

    async def get_from_pool(self, name: str) -> DatabaseResult:
        """Get a connection from the pool."""
        if name not in self.connection_pool:
            return DatabaseResult.error_result(f"Connection pool {name} not found")

        # Check if we need to clean up idle connections
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_idle_connections()
            self.last_cleanup = current_time

        # Try to get an existing connection from the pool
        pool = self.connection_pool[name]
        for i, connection in enumerate(pool):
            if not connection.connected:
                # Remove disconnected connections
                pool.pop(i)
                continue

            # Check if the connection is idle
            if current_time - connection.last_activity > self.idle_timeout[name]:
                # Close idle connections
                try:
                    await connection.disconnect()
                except Exception as e:
                    logger.error(f"Error closing idle connection in pool {name}: {e}")
                pool.pop(i)
                continue

            # Return the connection
            connection.last_activity = current_time
            return DatabaseResult.success_result(data=connection)

        # If we get here, there are no available connections in the pool
        # Check if we can create a new connection
        if len(pool) < self.pool_size[name]:
            try:
                # Create a new connection
                connection = self._connection_factory(name)
                result = await connection.connect()
                if not result.success:
                    return result

                # Add the connection to the pool
                connection.last_activity = current_time
                pool.append(connection)
                return DatabaseResult.success_result(data=connection)
            except Exception as e:
                logger.error(f"Error creating new connection in pool {name}: {e}")
                return DatabaseResult.error_result(ConnectionError(f"Failed to create new connection: {e}"))
        else:
            # Pool is full, wait for a connection to become available
            return DatabaseResult.error_result(f"Connection pool {name} is full")

    async def release_to_pool(self, name: str, connection: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        if name not in self.connection_pool:
            logger.warning(f"Attempted to release connection to non-existent pool {name}")
            return

        # Update the last activity time
        connection.last_activity = time.time()

    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections in all pools."""
        current_time = time.time()
        for name, pool in self.connection_pool.items():
            idle_timeout = self.idle_timeout[name]
            for i in range(len(pool) - 1, -1, -1):
                connection = pool[i]
                if not connection.connected or current_time - connection.last_activity > idle_timeout:
                    # Close and remove idle connections
                    try:
                        if connection.connected:
                            await connection.disconnect()
                    except Exception as e:
                        logger.error(f"Error closing idle connection in pool {name}: {e}")
                    pool.pop(i)


# Global database registry
default_registry = DatabaseRegistry()
