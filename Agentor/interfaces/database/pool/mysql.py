"""
MySQL connection pool for the Agentor framework.

This module provides a specialized connection pool for MySQL databases.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import ConnectionPoolConfig, ConnectionValidationMode

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class MySqlConnectionPool:
    """MySQL connection pool with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: ConnectionPoolConfig,
        host: str,
        port: int,
        user: str,
        password: Optional[str] = None,
        database: Optional[str] = None,
        charset: str = "utf8mb4",
        **kwargs
    ):
        """Initialize the MySQL connection pool.
        
        Args:
            name: The name of the pool
            config: The connection pool configuration
            host: The MySQL host
            port: The MySQL port
            user: The MySQL user
            password: The MySQL password
            database: The MySQL database
            charset: The MySQL charset
            **kwargs: Additional connection parameters
        """
        self.name = name
        self.config = config
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.kwargs = kwargs
        
        # Connection pool
        self.pool: Optional[Any] = None
        self.connections: Dict[Any, Dict[str, Any]] = {}
        self.active_connections: Set[Any] = set()
        self.idle_connections: Set[Any] = set()
        
        # Connection pool lock
        self.pool_lock = asyncio.Lock()
        
        # Connection pool semaphore
        self.pool_semaphore = asyncio.Semaphore(config.max_size)
        
        # Connection pool condition
        self.pool_condition = asyncio.Condition(self.pool_lock)
        
        # Connection pool metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "waiting_requests": 0,
            "max_waiting_requests": 0,
            "total_wait_time": 0.0,
            "total_execution_time": 0.0,
            "total_operations": 0,
            "failed_operations": 0,
            "validation_failures": 0,
            "connection_timeouts": 0,
            "connection_errors": 0
        }
        
        # Connection pool health
        self.health = {
            "status": "initializing",
            "last_check": time.time(),
            "last_failure": None,
            "failure_count": 0,
            "consecutive_failures": 0,
            "last_error": None
        }
        
        # Connection validation function
        if config.validation_mode == ConnectionValidationMode.PING:
            self.validation_func = self._validate_connection_ping
        elif config.validation_mode == ConnectionValidationMode.QUERY:
            self.validation_func = self._validate_connection_query
        elif config.validation_mode == ConnectionValidationMode.CUSTOM:
            self.validation_func = self._validate_connection_custom
        else:
            self.validation_func = self._validate_connection_none
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            # Import the MySQL driver
            try:
                import aiomysql
            except ImportError:
                raise ImportError("aiomysql package is not installed. Install it with 'pip install aiomysql'")
            
            # Create the connection pool
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                charset=self.charset,
                minsize=self.config.min_size,
                maxsize=self.config.max_size,
                pool_recycle=int(self.config.max_lifetime),
                connect_timeout=self.config.connect_timeout,
                **self.kwargs
            )
            
            # Initialize connections
            for i in range(self.config.min_size):
                try:
                    # Acquire a connection
                    conn = await self.pool.acquire()
                    
                    # Store the connection
                    self.connections[conn] = {
                        "created_at": time.time(),
                        "last_activity": time.time(),
                        "validation_count": 0,
                        "last_validation": time.time(),
                        "validation_failures": 0,
                        "operations": 0,
                        "errors": 0
                    }
                    
                    # Add to idle connections
                    self.idle_connections.add(conn)
                    
                    # Update metrics
                    self.metrics["total_connections"] += 1
                    self.metrics["idle_connections"] += 1
                except Exception as e:
                    logger.error(f"Failed to initialize connection for pool {self.name}: {e}")
                    self.metrics["connection_errors"] += 1
            
            # Update health
            self.health["status"] = "healthy"
            
            logger.info(f"Initialized MySQL connection pool {self.name} with {len(self.connections)} connections")
        except Exception as e:
            logger.error(f"Failed to initialize MySQL connection pool {self.name}: {e}")
            self.health["status"] = "error"
            self.health["last_failure"] = time.time()
            self.health["failure_count"] += 1
            self.health["consecutive_failures"] += 1
            self.health["last_error"] = str(e)
            raise
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            # Close the pool
            self.pool.close()
            await self.pool.wait_closed()
            
            # Clear connections
            self.connections.clear()
            self.active_connections.clear()
            self.idle_connections.clear()
            
            # Update metrics
            self.metrics["active_connections"] = 0
            self.metrics["idle_connections"] = 0
            
            logger.info(f"Closed MySQL connection pool {self.name}")
    
    async def acquire(self) -> Tuple[Any, float]:
        """Acquire a connection from the pool.
        
        Returns:
            Tuple of (connection, acquisition time)
        """
        if not self.pool:
            raise RuntimeError(f"MySQL connection pool {self.name} is not initialized")
        
        # Update metrics
        self.metrics["waiting_requests"] += 1
        self.metrics["max_waiting_requests"] = max(
            self.metrics["max_waiting_requests"],
            self.metrics["waiting_requests"]
        )
        
        # Record start time
        start_time = time.time()
        
        try:
            # Acquire the semaphore
            async with self.pool_semaphore:
                # Acquire the lock
                async with self.pool_lock:
                    # Check if there are idle connections
                    if self.idle_connections:
                        # Get a connection from the idle pool
                        conn = self.idle_connections.pop()
                        
                        # Add to active connections
                        self.active_connections.add(conn)
                        
                        # Update connection info
                        self.connections[conn]["last_activity"] = time.time()
                        
                        # Update metrics
                        self.metrics["idle_connections"] -= 1
                        self.metrics["active_connections"] += 1
                        self.metrics["last_activity"] = time.time()
                        
                        # Calculate acquisition time
                        acquisition_time = time.time() - start_time
                        self.metrics["total_wait_time"] += acquisition_time
                        
                        return conn, acquisition_time
                    
                    # No idle connections, create a new one
                    try:
                        # Acquire a connection from the pool
                        conn = await self.pool.acquire()
                        
                        # Store the connection
                        if conn not in self.connections:
                            self.connections[conn] = {
                                "created_at": time.time(),
                                "last_activity": time.time(),
                                "validation_count": 0,
                                "last_validation": time.time(),
                                "validation_failures": 0,
                                "operations": 0,
                                "errors": 0
                            }
                            
                            # Update metrics
                            self.metrics["total_connections"] += 1
                        else:
                            # Update connection info
                            self.connections[conn]["last_activity"] = time.time()
                        
                        # Add to active connections
                        self.active_connections.add(conn)
                        
                        # Update metrics
                        self.metrics["active_connections"] += 1
                        self.metrics["last_activity"] = time.time()
                        
                        # Calculate acquisition time
                        acquisition_time = time.time() - start_time
                        self.metrics["total_wait_time"] += acquisition_time
                        
                        return conn, acquisition_time
                    except Exception as e:
                        logger.error(f"Failed to acquire connection from pool {self.name}: {e}")
                        self.metrics["connection_errors"] += 1
                        raise
        finally:
            # Update metrics
            self.metrics["waiting_requests"] -= 1
    
    async def release(self, conn: Any) -> None:
        """Release a connection back to the pool.
        
        Args:
            conn: The connection to release
        """
        if not self.pool:
            raise RuntimeError(f"MySQL connection pool {self.name} is not initialized")
        
        # Acquire the lock
        async with self.pool_lock:
            # Check if the connection is active
            if conn in self.active_connections:
                # Remove from active connections
                self.active_connections.remove(conn)
                
                # Check if the connection is still valid
                if await self.validation_func(conn):
                    # Add to idle connections
                    self.idle_connections.add(conn)
                    
                    # Update connection info
                    self.connections[conn]["last_activity"] = time.time()
                    
                    # Update metrics
                    self.metrics["active_connections"] -= 1
                    self.metrics["idle_connections"] += 1
                    self.metrics["last_activity"] = time.time()
                else:
                    # Connection is invalid, close it
                    try:
                        # Release the connection back to the pool
                        self.pool.release(conn)
                        
                        # Remove the connection
                        if conn in self.connections:
                            del self.connections[conn]
                        
                        # Update metrics
                        self.metrics["active_connections"] -= 1
                        self.metrics["validation_failures"] += 1
                    except Exception as e:
                        logger.error(f"Failed to release invalid connection from pool {self.name}: {e}")
            else:
                # Connection is not active, just release it
                try:
                    # Release the connection back to the pool
                    self.pool.release(conn)
                    
                    # Remove the connection
                    if conn in self.connections:
                        del self.connections[conn]
                except Exception as e:
                    logger.error(f"Failed to release connection from pool {self.name}: {e}")
            
            # Release the semaphore
            self.pool_semaphore.release()
    
    async def validate_connections(self) -> None:
        """Validate all connections in the pool."""
        if not self.pool:
            raise RuntimeError(f"MySQL connection pool {self.name} is not initialized")
        
        # Acquire the lock
        async with self.pool_lock:
            # Validate idle connections
            invalid_connections = set()
            
            for conn in self.idle_connections:
                # Check if the connection is still valid
                if not await self.validation_func(conn):
                    # Connection is invalid
                    invalid_connections.add(conn)
                    
                    # Update metrics
                    self.metrics["validation_failures"] += 1
                    
                    # Update connection info
                    if conn in self.connections:
                        self.connections[conn]["validation_failures"] += 1
            
            # Remove invalid connections
            for conn in invalid_connections:
                # Remove from idle connections
                self.idle_connections.remove(conn)
                
                # Release the connection back to the pool
                try:
                    self.pool.release(conn)
                    
                    # Remove the connection
                    if conn in self.connections:
                        del self.connections[conn]
                    
                    # Update metrics
                    self.metrics["idle_connections"] -= 1
                except Exception as e:
                    logger.error(f"Failed to release invalid connection from pool {self.name}: {e}")
            
            # Check if we need to create new connections
            if len(self.idle_connections) < self.config.min_size:
                # Calculate how many connections to create
                to_create = self.config.min_size - len(self.idle_connections)
                
                # Create new connections
                for i in range(to_create):
                    try:
                        # Acquire a connection
                        conn = await self.pool.acquire()
                        
                        # Store the connection
                        self.connections[conn] = {
                            "created_at": time.time(),
                            "last_activity": time.time(),
                            "validation_count": 0,
                            "last_validation": time.time(),
                            "validation_failures": 0,
                            "operations": 0,
                            "errors": 0
                        }
                        
                        # Add to idle connections
                        self.idle_connections.add(conn)
                        
                        # Update metrics
                        self.metrics["total_connections"] += 1
                        self.metrics["idle_connections"] += 1
                    except Exception as e:
                        logger.error(f"Failed to create connection for pool {self.name}: {e}")
                        self.metrics["connection_errors"] += 1
    
    async def scale_pool(self) -> None:
        """Scale the connection pool based on usage."""
        if not self.pool:
            raise RuntimeError(f"MySQL connection pool {self.name} is not initialized")
        
        # Acquire the lock
        async with self.pool_lock:
            # Calculate the current pool size
            current_size = len(self.active_connections) + len(self.idle_connections)
            
            # Calculate the current usage
            if current_size > 0:
                usage = len(self.active_connections) / current_size
            else:
                usage = 0
            
            # Check if we need to scale up
            if usage >= self.config.scale_up_threshold and current_size < self.config.max_size:
                # Calculate how many connections to add
                to_add = min(
                    self.config.scale_up_step,
                    self.config.max_size - current_size
                )
                
                # Add connections
                for i in range(to_add):
                    try:
                        # Acquire a connection
                        conn = await self.pool.acquire()
                        
                        # Store the connection
                        self.connections[conn] = {
                            "created_at": time.time(),
                            "last_activity": time.time(),
                            "validation_count": 0,
                            "last_validation": time.time(),
                            "validation_failures": 0,
                            "operations": 0,
                            "errors": 0
                        }
                        
                        # Add to idle connections
                        self.idle_connections.add(conn)
                        
                        # Update metrics
                        self.metrics["total_connections"] += 1
                        self.metrics["idle_connections"] += 1
                    except Exception as e:
                        logger.error(f"Failed to scale up connection pool {self.name}: {e}")
                        self.metrics["connection_errors"] += 1
            
            # Check if we need to scale down
            elif usage <= self.config.scale_down_threshold and current_size > self.config.min_size:
                # Calculate how many connections to remove
                to_remove = min(
                    self.config.scale_down_step,
                    current_size - self.config.min_size,
                    len(self.idle_connections)
                )
                
                # Remove connections
                for i in range(to_remove):
                    if not self.idle_connections:
                        break
                    
                    # Get a connection from the idle pool
                    conn = self.idle_connections.pop()
                    
                    # Release the connection back to the pool
                    try:
                        self.pool.release(conn)
                        
                        # Remove the connection
                        if conn in self.connections:
                            del self.connections[conn]
                        
                        # Update metrics
                        self.metrics["idle_connections"] -= 1
                    except Exception as e:
                        logger.error(f"Failed to scale down connection pool {self.name}: {e}")
    
    async def check_health(self) -> bool:
        """Check the health of the connection pool.
        
        Returns:
            True if the pool is healthy, False otherwise
        """
        if not self.pool:
            return False
        
        try:
            # Acquire a connection
            conn, _ = await self.acquire()
            
            try:
                # Check if the connection is valid
                if await self.validation_func(conn):
                    return True
                else:
                    return False
            finally:
                # Release the connection
                await self.release(conn)
        except Exception as e:
            logger.error(f"Health check failed for pool {self.name}: {e}")
            return False
    
    async def recover(self) -> None:
        """Attempt to recover the connection pool."""
        if not self.pool:
            # Try to initialize the pool
            await self.initialize()
            return
        
        # Acquire the lock
        async with self.pool_lock:
            # Close all connections
            for conn in list(self.connections.keys()):
                try:
                    # Release the connection back to the pool
                    self.pool.release(conn)
                except Exception:
                    pass
            
            # Clear connections
            self.connections.clear()
            self.active_connections.clear()
            self.idle_connections.clear()
            
            # Update metrics
            self.metrics["active_connections"] = 0
            self.metrics["idle_connections"] = 0
            
            # Create new connections
            for i in range(self.config.min_size):
                try:
                    # Acquire a connection
                    conn = await self.pool.acquire()
                    
                    # Store the connection
                    self.connections[conn] = {
                        "created_at": time.time(),
                        "last_activity": time.time(),
                        "validation_count": 0,
                        "last_validation": time.time(),
                        "validation_failures": 0,
                        "operations": 0,
                        "errors": 0
                    }
                    
                    # Add to idle connections
                    self.idle_connections.add(conn)
                    
                    # Update metrics
                    self.metrics["total_connections"] += 1
                    self.metrics["idle_connections"] += 1
                except Exception as e:
                    logger.error(f"Failed to recover connection for pool {self.name}: {e}")
                    self.metrics["connection_errors"] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _validate_connection_none(self, conn: Any) -> bool:
        """Validate a connection using no validation.
        
        Args:
            conn: The connection to validate
            
        Returns:
            True if the connection is valid, False otherwise
        """
        # No validation, always return True
        return True
    
    async def _validate_connection_ping(self, conn: Any) -> bool:
        """Validate a connection using ping.
        
        Args:
            conn: The connection to validate
            
        Returns:
            True if the connection is valid, False otherwise
        """
        try:
            # Ping the server
            await conn.ping()
            
            # Update connection info
            if conn in self.connections:
                self.connections[conn]["validation_count"] += 1
                self.connections[conn]["last_validation"] = time.time()
            
            return True
        except Exception as e:
            logger.error(f"Connection validation failed for pool {self.name}: {e}")
            
            # Update connection info
            if conn in self.connections:
                self.connections[conn]["validation_failures"] += 1
            
            return False
    
    async def _validate_connection_query(self, conn: Any) -> bool:
        """Validate a connection using a query.
        
        Args:
            conn: The connection to validate
            
        Returns:
            True if the connection is valid, False otherwise
        """
        try:
            # Execute the validation query
            async with conn.cursor() as cursor:
                await cursor.execute(self.config.validation_query or "SELECT 1")
                result = await cursor.fetchone()
                
                # Check the result
                if not result:
                    return False
            
            # Update connection info
            if conn in self.connections:
                self.connections[conn]["validation_count"] += 1
                self.connections[conn]["last_validation"] = time.time()
            
            return True
        except Exception as e:
            logger.error(f"Connection validation failed for pool {self.name}: {e}")
            
            # Update connection info
            if conn in self.connections:
                self.connections[conn]["validation_failures"] += 1
            
            return False
    
    async def _validate_connection_custom(self, conn: Any) -> bool:
        """Validate a connection using a custom validation function.
        
        Args:
            conn: The connection to validate
            
        Returns:
            True if the connection is valid, False otherwise
        """
        # Get the custom validation function
        validation_func = self.config.additional_settings.get("validation_func")
        
        if not validation_func:
            # No validation function, use ping
            return await self._validate_connection_ping(conn)
        
        try:
            # Call the validation function
            result = await validation_func(conn)
            
            # Update connection info
            if conn in self.connections:
                self.connections[conn]["validation_count"] += 1
                self.connections[conn]["last_validation"] = time.time()
            
            return result
        except Exception as e:
            logger.error(f"Custom connection validation failed for pool {self.name}: {e}")
            
            # Update connection info
            if conn in self.connections:
                self.connections[conn]["validation_failures"] += 1
            
            return False
