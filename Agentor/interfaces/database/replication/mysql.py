"""
MySQL replication manager for the Agentor framework.

This module provides a specialized manager for MySQL replication.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import ReplicationConfig, ReplicationMode, ReplicationRole, ServerConfig
from .manager import ReplicationManager

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class MySqlServer:
    """MySQL server in a replication setup."""
    
    def __init__(
        self,
        id: str,
        config: ServerConfig,
        connection_factory: Callable[[ServerConfig], T]
    ):
        """Initialize the MySQL server.
        
        Args:
            id: The ID of the server
            config: The server configuration
            connection_factory: Function to create a connection to the server
        """
        self.id = id
        self.config = config
        self.connection_factory = connection_factory
        
        # Connection pool
        self.pool: Optional[T] = None
        
        # Server status
        self.is_connected = False
        self.last_connected = 0.0
        self.last_error = None
        self.active_connections = 0
        
        # Replication status
        self.is_primary_server = False
        self.replication_lag = 0.0
        self.replication_status = {}
        
        # Server metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "health_checks": 0,
            "failed_health_checks": 0
        }
        
        # Server lock
        self.server_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the MySQL server."""
        logger.info(f"Initializing MySQL server {self.id}")
        
        try:
            # Create the connection pool
            self.pool = self.connection_factory(self.config)
            
            # Test the connection
            is_healthy = await self.is_healthy()
            
            if is_healthy:
                self.is_connected = True
                self.last_connected = time.time()
                logger.info(f"MySQL server {self.id} initialized successfully")
            else:
                logger.warning(f"MySQL server {self.id} is not healthy")
        except Exception as e:
            logger.error(f"Error initializing MySQL server {self.id}: {e}")
            self.last_error = str(e)
    
    async def close(self) -> None:
        """Close the MySQL server."""
        logger.info(f"Closing MySQL server {self.id}")
        
        # Close the connection pool
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logger.error(f"Error closing MySQL server {self.id}: {e}")
        
        self.is_connected = False
        logger.info(f"MySQL server {self.id} closed")
    
    async def get_connection(self) -> Optional[T]:
        """Get a connection from the server.
        
        Returns:
            A connection, or None if no connection is available
        """
        # Update metrics
        self.metrics["total_connections"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Check if the server is connected
        if not self.is_connected or not self.pool:
            self.metrics["failed_connections"] += 1
            return None
        
        try:
            # Get a connection from the pool
            conn, _ = await self.pool.acquire()
            
            # Update metrics
            self.active_connections += 1
            self.metrics["active_connections"] = self.active_connections
            
            return conn
        except Exception as e:
            logger.error(f"Error getting connection from MySQL server {self.id}: {e}")
            self.last_error = str(e)
            self.metrics["failed_connections"] += 1
            return None
    
    async def release_connection(self, conn: T) -> None:
        """Release a connection back to the server.
        
        Args:
            conn: The connection to release
        """
        # Check if the server is connected
        if not self.is_connected or not self.pool:
            return
        
        try:
            # Release the connection back to the pool
            await self.pool.release(conn)
            
            # Update metrics
            self.active_connections -= 1
            self.metrics["active_connections"] = self.active_connections
        except Exception as e:
            logger.error(f"Error releasing connection to MySQL server {self.id}: {e}")
            self.last_error = str(e)
    
    async def is_healthy(self) -> bool:
        """Check if the server is healthy.
        
        Returns:
            True if the server is healthy, False otherwise
        """
        # Update metrics
        self.metrics["health_checks"] += 1
        
        # Check if the server is connected
        if not self.pool:
            self.metrics["failed_health_checks"] += 1
            return False
        
        try:
            # Get a connection from the pool
            conn, _ = await self.pool.acquire()
            
            try:
                # Execute a simple query
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    
                    if result and result[0] == 1:
                        return True
                    else:
                        self.metrics["failed_health_checks"] += 1
                        return False
            finally:
                # Release the connection
                await self.pool.release(conn)
        except Exception as e:
            logger.error(f"Error checking health of MySQL server {self.id}: {e}")
            self.last_error = str(e)
            self.metrics["failed_health_checks"] += 1
            return False
    
    async def is_primary(self) -> bool:
        """Check if the server is a primary server.
        
        Returns:
            True if the server is a primary server, False otherwise
        """
        # Check if the server is connected
        if not self.pool:
            return False
        
        try:
            # Get a connection from the pool
            conn, _ = await self.pool.acquire()
            
            try:
                # Check if the server is a primary server
                async with conn.cursor() as cursor:
                    # Check if the server has read_only=0
                    await cursor.execute("SELECT @@read_only")
                    result = await cursor.fetchone()
                    
                    if result and result[0] == 0:
                        self.is_primary_server = True
                        return True
                    else:
                        self.is_primary_server = False
                        return False
            finally:
                # Release the connection
                await self.pool.release(conn)
        except Exception as e:
            logger.error(f"Error checking if MySQL server {self.id} is primary: {e}")
            self.last_error = str(e)
            return False
    
    async def promote_to_primary(self) -> bool:
        """Promote the server to a primary server.
        
        Returns:
            True if the promotion was successful, False otherwise
        """
        # Check if the server is connected
        if not self.pool:
            return False
        
        try:
            # Get a connection from the pool
            conn, _ = await self.pool.acquire()
            
            try:
                # Promote the server to a primary server
                async with conn.cursor() as cursor:
                    # Set read_only=0
                    await cursor.execute("SET GLOBAL read_only=0")
                    
                    # Check if the server is now a primary server
                    await cursor.execute("SELECT @@read_only")
                    result = await cursor.fetchone()
                    
                    if result and result[0] == 0:
                        self.is_primary_server = True
                        logger.info(f"MySQL server {self.id} promoted to primary")
                        return True
                    else:
                        logger.error(f"Failed to promote MySQL server {self.id} to primary")
                        return False
            finally:
                # Release the connection
                await self.pool.release(conn)
        except Exception as e:
            logger.error(f"Error promoting MySQL server {self.id} to primary: {e}")
            self.last_error = str(e)
            return False
    
    async def demote_to_replica(self) -> bool:
        """Demote the server to a replica server.
        
        Returns:
            True if the demotion was successful, False otherwise
        """
        # Check if the server is connected
        if not self.pool:
            return False
        
        try:
            # Get a connection from the pool
            conn, _ = await self.pool.acquire()
            
            try:
                # Demote the server to a replica server
                async with conn.cursor() as cursor:
                    # Set read_only=1
                    await cursor.execute("SET GLOBAL read_only=1")
                    
                    # Check if the server is now a replica server
                    await cursor.execute("SELECT @@read_only")
                    result = await cursor.fetchone()
                    
                    if result and result[0] == 1:
                        self.is_primary_server = False
                        logger.info(f"MySQL server {self.id} demoted to replica")
                        return True
                    else:
                        logger.error(f"Failed to demote MySQL server {self.id} to replica")
                        return False
            finally:
                # Release the connection
                await self.pool.release(conn)
        except Exception as e:
            logger.error(f"Error demoting MySQL server {self.id} to replica: {e}")
            self.last_error = str(e)
            return False
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get the replication status of the server.
        
        Returns:
            Dictionary of replication status
        """
        # Check if the server is connected
        if not self.pool:
            return {}
        
        try:
            # Get a connection from the pool
            conn, _ = await self.pool.acquire()
            
            try:
                # Get the replication status
                async with conn.cursor() as cursor:
                    if self.is_primary_server:
                        # Get primary status
                        await cursor.execute("SHOW MASTER STATUS")
                        result = await cursor.fetchone()
                        
                        if result:
                            status = {
                                "is_primary": True,
                                "binary_log_file": result[0],
                                "binary_log_position": result[1],
                                "binlog_do_db": result[2],
                                "binlog_ignore_db": result[3]
                            }
                            
                            # Get replica hosts
                            await cursor.execute("SHOW SLAVE HOSTS")
                            replicas = await cursor.fetchall()
                            
                            status["replicas"] = []
                            for replica in replicas:
                                status["replicas"].append({
                                    "server_id": replica[0],
                                    "host": replica[1],
                                    "port": replica[2],
                                    "master_id": replica[3]
                                })
                            
                            self.replication_status = status
                            return status
                        else:
                            return {"is_primary": True}
                    else:
                        # Get replica status
                        await cursor.execute("SHOW SLAVE STATUS")
                        result = await cursor.fetchone()
                        
                        if result:
                            # Get the column names
                            columns = [column[0] for column in cursor.description]
                            
                            # Create a dictionary of the result
                            status = dict(zip(columns, result))
                            
                            # Add some key fields
                            status["is_primary"] = False
                            status["is_replica"] = True
                            status["primary_host"] = status.get("Master_Host")
                            status["primary_port"] = status.get("Master_Port")
                            status["replication_lag"] = status.get("Seconds_Behind_Master")
                            status["io_running"] = status.get("Slave_IO_Running") == "Yes"
                            status["sql_running"] = status.get("Slave_SQL_Running") == "Yes"
                            status["last_error"] = status.get("Last_Error")
                            
                            # Update the replication lag
                            self.replication_lag = status.get("Seconds_Behind_Master", 0)
                            
                            self.replication_status = status
                            return status
                        else:
                            return {"is_primary": False, "is_replica": False}
            finally:
                # Release the connection
                await self.pool.release(conn)
        except Exception as e:
            logger.error(f"Error getting replication status of MySQL server {self.id}: {e}")
            self.last_error = str(e)
            return {}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the status of the server.
        
        Returns:
            Dictionary of server status
        """
        # Get the replication status
        replication_status = await self.get_replication_status()
        
        # Create the status dictionary
        status = {
            "id": self.id,
            "host": self.config.host,
            "port": self.config.port,
            "is_connected": self.is_connected,
            "last_connected": self.last_connected,
            "last_error": self.last_error,
            "active_connections": self.active_connections,
            "is_primary": self.is_primary_server,
            "replication_lag": self.replication_lag,
            "replication_status": replication_status,
            "metrics": self.metrics
        }
        
        return status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics


class MySqlReplicationManager(ReplicationManager[T, MySqlServer]):
    """MySQL replication manager with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: ReplicationConfig,
        connection_factory: Callable[[ServerConfig], T]
    ):
        """Initialize the MySQL replication manager.
        
        Args:
            name: The name of the manager
            config: The replication configuration
            connection_factory: Function to create a connection to a server
        """
        super().__init__(
            name=name,
            config=config,
            server_class=MySqlServer,
            connection_factory=connection_factory
        )
        
        # MySQL-specific metrics
        self.mysql_metrics = {
            "replication_lag": 0.0,
            "primary_connections": 0,
            "replica_connections": 0,
            "primary_queries": 0,
            "replica_queries": 0
        }
    
    async def get_read_connection(self) -> Tuple[Optional[T], Optional[MySqlServer]]:
        """Get a connection for read operations.
        
        Returns:
            Tuple of (connection, server)
        """
        # Get a connection using the base implementation
        conn, server = await super().get_read_connection()
        
        # Update MySQL-specific metrics
        if conn and server:
            if server.is_primary_server:
                self.mysql_metrics["primary_connections"] += 1
                self.mysql_metrics["primary_queries"] += 1
            else:
                self.mysql_metrics["replica_connections"] += 1
                self.mysql_metrics["replica_queries"] += 1
        
        return conn, server
    
    async def get_write_connection(self) -> Tuple[Optional[T], Optional[MySqlServer]]:
        """Get a connection for write operations.
        
        Returns:
            Tuple of (connection, server)
        """
        # Get a connection using the base implementation
        conn, server = await super().get_write_connection()
        
        # Update MySQL-specific metrics
        if conn and server:
            if server.is_primary_server:
                self.mysql_metrics["primary_connections"] += 1
                self.mysql_metrics["primary_queries"] += 1
            else:
                self.mysql_metrics["replica_connections"] += 1
                self.mysql_metrics["replica_queries"] += 1
        
        return conn, server
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get the status of the replication setup.
        
        Returns:
            Dictionary of replication status
        """
        # Get the base replication status
        status = await super().get_replication_status()
        
        # Add MySQL-specific status
        status["mysql_metrics"] = self.mysql_metrics
        
        # Calculate the average replication lag
        if self.replica_servers:
            total_lag = sum(server.replication_lag for server in self.replica_servers)
            avg_lag = total_lag / len(self.replica_servers)
            status["avg_replication_lag"] = avg_lag
            self.mysql_metrics["replication_lag"] = avg_lag
        
        return status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get replication manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Get the base metrics
        metrics = await super().get_metrics()
        
        # Add MySQL-specific metrics
        metrics.update(self.mysql_metrics)
        
        return metrics
