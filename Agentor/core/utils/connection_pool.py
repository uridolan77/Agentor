"""
Connection pooling utilities for the Agentor framework.

This module provides connection pooling for various database systems,
optimizing performance by reusing connections instead of creating new ones.
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Awaitable, Union, Tuple
import time
import logging
import asyncio
import functools
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Connection type


@dataclass
class PooledConnection(Generic[T]):
    """A pooled connection with metadata."""
    
    connection: T
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    in_use: bool = False
    
    def mark_used(self) -> None:
        """Mark the connection as in use."""
        self.in_use = True
        self.last_used = time.time()
        self.use_count += 1
    
    def mark_free(self) -> None:
        """Mark the connection as free."""
        self.in_use = False
        self.last_used = time.time()


class ConnectionPool(Generic[T], ABC):
    """Abstract base class for connection pools."""
    
    def __init__(
        self,
        min_size: int = 1,
        max_size: int = 10,
        max_idle_time: float = 300,
        max_lifetime: float = 3600,
        connection_timeout: float = 30.0,
        connection_validation_interval: float = 30.0
    ):
        """Initialize the connection pool.
        
        Args:
            min_size: Minimum number of connections
            max_size: Maximum number of connections
            max_idle_time: Maximum idle time in seconds
            max_lifetime: Maximum lifetime in seconds
            connection_timeout: Timeout for acquiring a connection
            connection_validation_interval: Interval for validating connections
        """
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        self.connection_timeout = connection_timeout
        self.connection_validation_interval = connection_validation_interval
        
        self.pool: List[PooledConnection[T]] = []
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)
        
        # Statistics
        self.created_connections = 0
        self.closed_connections = 0
        self.connection_timeouts = 0
        self.connection_errors = 0
        self.max_concurrent_connections = 0
        
        # Maintenance task
        self.maintenance_task = None
    
    @abstractmethod
    async def create_connection(self) -> T:
        """Create a new connection.
        
        Returns:
            A new connection
        """
        pass
    
    @abstractmethod
    async def close_connection(self, connection: T) -> None:
        """Close a connection.
        
        Args:
            connection: The connection to close
        """
        pass
    
    @abstractmethod
    async def validate_connection(self, connection: T) -> bool:
        """Validate a connection.
        
        Args:
            connection: The connection to validate
            
        Returns:
            True if the connection is valid
        """
        pass
    
    async def start(self) -> None:
        """Start the connection pool."""
        # Create initial connections
        async with self.lock:
            for _ in range(self.min_size):
                await self._add_connection()
        
        # Start maintenance task
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info(f"Started connection pool with {self.min_size} initial connections")
    
    async def stop(self) -> None:
        """Stop the connection pool."""
        # Cancel maintenance task
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self.lock:
            for pooled_conn in self.pool:
                try:
                    await self.close_connection(pooled_conn.connection)
                    self.closed_connections += 1
                except Exception as e:
                    logger.error(f"Error closing connection: {str(e)}")
            
            self.pool.clear()
        
        logger.info("Stopped connection pool")
    
    async def acquire(self) -> T:
        """Acquire a connection from the pool.
        
        Returns:
            A connection
            
        Raises:
            TimeoutError: If no connection could be acquired within the timeout
        """
        async with self.condition:
            # Try to find a free connection
            for pooled_conn in self.pool:
                if not pooled_conn.in_use:
                    # Validate the connection
                    try:
                        if await self.validate_connection(pooled_conn.connection):
                            pooled_conn.mark_used()
                            return pooled_conn.connection
                        else:
                            # Connection is invalid, remove it
                            self.pool.remove(pooled_conn)
                            await self.close_connection(pooled_conn.connection)
                            self.closed_connections += 1
                    except Exception as e:
                        logger.error(f"Error validating connection: {str(e)}")
                        # Connection is invalid, remove it
                        self.pool.remove(pooled_conn)
                        try:
                            await self.close_connection(pooled_conn.connection)
                            self.closed_connections += 1
                        except Exception as e2:
                            logger.error(f"Error closing invalid connection: {str(e2)}")
            
            # If we have room, create a new connection
            if len(self.pool) < self.max_size:
                try:
                    connection = await self._add_connection()
                    return connection
                except Exception as e:
                    logger.error(f"Error creating new connection: {str(e)}")
                    self.connection_errors += 1
            
            # Wait for a connection to become available
            try:
                # Wait with timeout
                start_time = time.time()
                while True:
                    # Check if we've exceeded the timeout
                    if time.time() - start_time > self.connection_timeout:
                        self.connection_timeouts += 1
                        raise TimeoutError(f"Timeout waiting for connection after {self.connection_timeout} seconds")
                    
                    # Wait for a notification
                    await asyncio.wait_for(
                        self.condition.wait(),
                        timeout=self.connection_timeout - (time.time() - start_time)
                    )
                    
                    # Try to find a free connection again
                    for pooled_conn in self.pool:
                        if not pooled_conn.in_use:
                            # Validate the connection
                            try:
                                if await self.validate_connection(pooled_conn.connection):
                                    pooled_conn.mark_used()
                                    return pooled_conn.connection
                                else:
                                    # Connection is invalid, remove it
                                    self.pool.remove(pooled_conn)
                                    await self.close_connection(pooled_conn.connection)
                                    self.closed_connections += 1
                            except Exception as e:
                                logger.error(f"Error validating connection: {str(e)}")
                                # Connection is invalid, remove it
                                self.pool.remove(pooled_conn)
                                try:
                                    await self.close_connection(pooled_conn.connection)
                                    self.closed_connections += 1
                                except Exception as e2:
                                    logger.error(f"Error closing invalid connection: {str(e2)}")
            except asyncio.TimeoutError:
                self.connection_timeouts += 1
                raise TimeoutError(f"Timeout waiting for connection after {self.connection_timeout} seconds")
    
    async def release(self, connection: T) -> None:
        """Release a connection back to the pool.
        
        Args:
            connection: The connection to release
        """
        async with self.condition:
            # Find the connection in the pool
            for pooled_conn in self.pool:
                if pooled_conn.connection is connection:
                    pooled_conn.mark_free()
                    # Notify waiters
                    self.condition.notify_all()
                    return
            
            # If the connection is not in the pool, close it
            try:
                await self.close_connection(connection)
            except Exception as e:
                logger.error(f"Error closing unknown connection: {str(e)}")
    
    async def _add_connection(self) -> T:
        """Add a new connection to the pool.
        
        Returns:
            The new connection
        """
        connection = await self.create_connection()
        pooled_conn = PooledConnection(connection=connection)
        pooled_conn.mark_used()
        self.pool.append(pooled_conn)
        self.created_connections += 1
        
        # Update statistics
        in_use_count = sum(1 for conn in self.pool if conn.in_use)
        self.max_concurrent_connections = max(self.max_concurrent_connections, in_use_count)
        
        return connection
    
    async def _maintenance_loop(self) -> None:
        """Maintenance loop for the connection pool."""
        while True:
            try:
                # Sleep for the validation interval
                await asyncio.sleep(self.connection_validation_interval)
                
                # Perform maintenance
                await self._perform_maintenance()
            except asyncio.CancelledError:
                logger.info("Connection pool maintenance task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in connection pool maintenance: {str(e)}")
    
    async def _perform_maintenance(self) -> None:
        """Perform maintenance on the connection pool."""
        async with self.lock:
            now = time.time()
            to_remove = []
            
            # Check each connection
            for pooled_conn in self.pool:
                # Skip connections in use
                if pooled_conn.in_use:
                    continue
                
                # Check if the connection has been idle for too long
                if now - pooled_conn.last_used > self.max_idle_time:
                    to_remove.append(pooled_conn)
                    continue
                
                # Check if the connection has exceeded its lifetime
                if now - pooled_conn.created_at > self.max_lifetime:
                    to_remove.append(pooled_conn)
                    continue
                
                # Validate the connection
                try:
                    if not await self.validate_connection(pooled_conn.connection):
                        to_remove.append(pooled_conn)
                except Exception as e:
                    logger.error(f"Error validating connection during maintenance: {str(e)}")
                    to_remove.append(pooled_conn)
            
            # Remove invalid connections
            for pooled_conn in to_remove:
                self.pool.remove(pooled_conn)
                try:
                    await self.close_connection(pooled_conn.connection)
                    self.closed_connections += 1
                except Exception as e:
                    logger.error(f"Error closing connection during maintenance: {str(e)}")
            
            # Ensure we have the minimum number of connections
            while len(self.pool) < self.min_size:
                try:
                    await self._add_connection()
                    # Mark the connection as free
                    self.pool[-1].mark_free()
                except Exception as e:
                    logger.error(f"Error creating connection during maintenance: {str(e)}")
                    self.connection_errors += 1
                    break
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary of connection pool statistics
        """
        async with self.lock:
            in_use_count = sum(1 for conn in self.pool if conn.in_use)
            free_count = len(self.pool) - in_use_count
            
            return {
                "size": len(self.pool),
                "min_size": self.min_size,
                "max_size": self.max_size,
                "in_use": in_use_count,
                "free": free_count,
                "created": self.created_connections,
                "closed": self.closed_connections,
                "timeouts": self.connection_timeouts,
                "errors": self.connection_errors,
                "max_concurrent": self.max_concurrent_connections
            }


class DatabaseConnectionPool(ConnectionPool[Any]):
    """Connection pool for database connections."""
    
    def __init__(
        self,
        connection_factory: Callable[[], Awaitable[Any]],
        connection_validator: Callable[[Any], Awaitable[bool]],
        connection_closer: Callable[[Any], Awaitable[None]],
        **kwargs
    ):
        """Initialize the database connection pool.
        
        Args:
            connection_factory: Function to create a new connection
            connection_validator: Function to validate a connection
            connection_closer: Function to close a connection
            **kwargs: Additional arguments for the connection pool
        """
        super().__init__(**kwargs)
        self.connection_factory = connection_factory
        self.connection_validator = connection_validator
        self.connection_closer = connection_closer
    
    async def create_connection(self) -> Any:
        """Create a new connection.
        
        Returns:
            A new connection
        """
        return await self.connection_factory()
    
    async def close_connection(self, connection: Any) -> None:
        """Close a connection.
        
        Args:
            connection: The connection to close
        """
        await self.connection_closer(connection)
    
    async def validate_connection(self, connection: Any) -> bool:
        """Validate a connection.
        
        Args:
            connection: The connection to validate
            
        Returns:
            True if the connection is valid
        """
        return await self.connection_validator(connection)


class ConnectionPoolManager:
    """Manager for multiple connection pools."""
    
    def __init__(self):
        """Initialize the connection pool manager."""
        self.pools: Dict[str, ConnectionPool] = {}
        self.lock = asyncio.Lock()
    
    async def create_pool(
        self,
        name: str,
        pool_class: type,
        **kwargs
    ) -> ConnectionPool:
        """Create a new connection pool.
        
        Args:
            name: The name of the pool
            pool_class: The connection pool class
            **kwargs: Arguments for the connection pool
            
        Returns:
            The connection pool
        """
        async with self.lock:
            if name in self.pools:
                raise ValueError(f"Pool {name} already exists")
            
            pool = pool_class(**kwargs)
            await pool.start()
            self.pools[name] = pool
            
            return pool
    
    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name.
        
        Args:
            name: The name of the pool
            
        Returns:
            The connection pool, or None if not found
        """
        async with self.lock:
            return self.pools.get(name)
    
    async def close_pool(self, name: str) -> bool:
        """Close a connection pool.
        
        Args:
            name: The name of the pool
            
        Returns:
            True if the pool was closed, False otherwise
        """
        async with self.lock:
            pool = self.pools.pop(name, None)
            
            if pool:
                await pool.stop()
                return True
            
            return False
    
    async def close_all(self) -> None:
        """Close all connection pools."""
        async with self.lock:
            for name, pool in list(self.pools.items()):
                await pool.stop()
            
            self.pools.clear()
    
    async def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connection pools.
        
        Returns:
            Dictionary of connection pool statistics
        """
        async with self.lock:
            stats = {}
            
            for name, pool in self.pools.items():
                stats[name] = await pool.get_stats()
            
            return stats


# Create a global connection pool manager
connection_pool_manager = ConnectionPoolManager()


async def get_connection(pool_name: str) -> Tuple[Any, Callable[[Any], Awaitable[None]]]:
    """Get a connection from a pool.
    
    Args:
        pool_name: The name of the pool
        
    Returns:
        A tuple of (connection, release_func)
        
    Raises:
        ValueError: If the pool does not exist
    """
    pool = await connection_pool_manager.get_pool(pool_name)
    
    if not pool:
        raise ValueError(f"Pool {pool_name} does not exist")
    
    connection = await pool.acquire()
    
    async def release():
        await pool.release(connection)
    
    return connection, release


class ConnectionPoolContext:
    """Context manager for acquiring a connection from a pool."""
    
    def __init__(self, pool_name: str):
        """Initialize the context manager.
        
        Args:
            pool_name: The name of the pool
        """
        self.pool_name = pool_name
        self.connection = None
        self.release_func = None
    
    async def __aenter__(self) -> Any:
        """Enter the context manager.
        
        Returns:
            The connection
        """
        self.connection, self.release_func = await get_connection(self.pool_name)
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        if self.release_func:
            await self.release_func()


def with_connection(pool_name: str):
    """Decorator for using a connection from a pool.
    
    Args:
        pool_name: The name of the pool
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with ConnectionPoolContext(pool_name) as connection:
                return await func(connection, *args, **kwargs)
        
        return wrapper
    
    return decorator
