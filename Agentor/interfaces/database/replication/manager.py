"""
Replication manager for the Agentor framework.

This module provides a manager for database replication.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type, Tuple
import weakref

from .config import ReplicationConfig, ReplicationMode, ReplicationRole, ServerConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type
S = TypeVar('S')  # Server type


class ReplicationManager(Generic[T, S]):
    """Manager for database replication."""
    
    def __init__(
        self,
        name: str,
        config: ReplicationConfig,
        server_class: Type[S],
        connection_factory: Callable[[ServerConfig], T]
    ):
        """Initialize the replication manager.
        
        Args:
            name: The name of the manager
            config: The replication configuration
            server_class: The server class
            connection_factory: Function to create a connection to a server
        """
        self.name = name
        self.config = config
        self.server_class = server_class
        self.connection_factory = connection_factory
        
        # Servers
        self.servers: Dict[str, S] = {}
        self.primary_servers: List[S] = []
        self.replica_servers: List[S] = []
        
        # Current primary server
        self.current_primary: Optional[S] = None
        
        # Round-robin counter for load balancing
        self.round_robin_counter = 0
        
        # Manager lock
        self.manager_lock = asyncio.Lock()
        
        # Manager tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Manager metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_reads": 0,
            "total_writes": 0,
            "primary_reads": 0,
            "replica_reads": 0,
            "failed_reads": 0,
            "failed_writes": 0,
            "failovers": 0,
            "successful_failovers": 0,
            "failed_failovers": 0
        }
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
    
    async def initialize(self) -> None:
        """Initialize the replication manager."""
        logger.info(f"Initializing replication manager {self.name}")
        
        # Initialize servers
        for i, server_config in enumerate(self.config.servers):
            # Create a server ID
            server_id = f"{server_config.host}:{server_config.port}"
            
            # Create a server
            server = self.server_class(
                id=server_id,
                config=server_config,
                connection_factory=self.connection_factory
            )
            
            # Initialize the server
            await server.initialize()
            
            # Store the server
            self.servers[server_id] = server
            
            # Categorize the server
            if server_config.role == ReplicationRole.PRIMARY:
                self.primary_servers.append(server)
            elif server_config.role == ReplicationRole.REPLICA:
                self.replica_servers.append(server)
            else:  # Auto-detect role
                # Detect the role
                is_primary = await server.is_primary()
                
                if is_primary:
                    self.primary_servers.append(server)
                else:
                    self.replica_servers.append(server)
        
        # Set the current primary
        if self.primary_servers:
            self.current_primary = self.primary_servers[0]
        
        # Start health check task
        if self.config.failover_enabled:
            self.health_check_task = self.loop.create_task(self._health_check_loop())
        
        # Start monitoring task
        if self.config.monitoring_enabled:
            self.monitoring_task = self.loop.create_task(self._monitoring_loop())
        
        logger.info(f"Replication manager {self.name} initialized with {len(self.servers)} servers")
    
    async def close(self) -> None:
        """Close the replication manager."""
        logger.info(f"Closing replication manager {self.name}")
        
        # Cancel tasks
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close servers
        for server in self.servers.values():
            await server.close()
        
        # Clear servers
        self.servers.clear()
        self.primary_servers.clear()
        self.replica_servers.clear()
        self.current_primary = None
        
        logger.info(f"Replication manager {self.name} closed")
    
    async def get_read_connection(self) -> Tuple[Optional[T], Optional[S]]:
        """Get a connection for read operations.
        
        Returns:
            Tuple of (connection, server)
        """
        # Update metrics
        self.metrics["total_reads"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Check if read/write splitting is enabled
        if not self.config.read_write_splitting:
            # Use the primary server
            if self.current_primary:
                conn = await self.current_primary.get_connection()
                if conn:
                    # Update metrics
                    self.metrics["primary_reads"] += 1
                    return conn, self.current_primary
            
            # No primary server available
            self.metrics["failed_reads"] += 1
            return None, None
        
        # Use a replica server if available
        if self.replica_servers:
            # Get a replica server using the load balancing strategy
            server = await self._get_replica_server()
            
            if server:
                conn = await server.get_connection()
                if conn:
                    # Update metrics
                    self.metrics["replica_reads"] += 1
                    return conn, server
        
        # No replica server available, check if we can read from the primary
        if self.config.read_from_primary and self.current_primary:
            conn = await self.current_primary.get_connection()
            if conn:
                # Update metrics
                self.metrics["primary_reads"] += 1
                return conn, self.current_primary
        
        # No server available
        self.metrics["failed_reads"] += 1
        return None, None
    
    async def get_write_connection(self) -> Tuple[Optional[T], Optional[S]]:
        """Get a connection for write operations.
        
        Returns:
            Tuple of (connection, server)
        """
        # Update metrics
        self.metrics["total_writes"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Use the primary server
        if self.current_primary:
            conn = await self.current_primary.get_connection()
            if conn:
                return conn, self.current_primary
        
        # Check if we can write to a replica
        if self.config.write_to_replica and self.replica_servers:
            # Get a replica server using the load balancing strategy
            server = await self._get_replica_server()
            
            if server:
                conn = await server.get_connection()
                if conn:
                    return conn, server
        
        # No server available
        self.metrics["failed_writes"] += 1
        return None, None
    
    async def release_connection(self, conn: T, server: S) -> None:
        """Release a connection back to the server.
        
        Args:
            conn: The connection to release
            server: The server the connection belongs to
        """
        await server.release_connection(conn)
    
    async def failover(self) -> bool:
        """Perform a failover to a new primary server.
        
        Returns:
            True if the failover was successful, False otherwise
        """
        # Update metrics
        self.metrics["failovers"] += 1
        
        # Check if failover is enabled
        if not self.config.failover_enabled:
            logger.warning(f"Failover is disabled for replication manager {self.name}")
            self.metrics["failed_failovers"] += 1
            return False
        
        # Check if we have a current primary
        if not self.current_primary:
            logger.warning(f"No current primary server for replication manager {self.name}")
            self.metrics["failed_failovers"] += 1
            return False
        
        # Check if we have any replica servers
        if not self.replica_servers:
            logger.warning(f"No replica servers for replication manager {self.name}")
            self.metrics["failed_failovers"] += 1
            return False
        
        # Acquire the manager lock
        async with self.manager_lock:
            # Check if the current primary is still unhealthy
            is_healthy = await self.current_primary.is_healthy()
            if is_healthy:
                logger.info(f"Current primary server {self.current_primary.id} is healthy, no failover needed")
                return True
            
            # Find a healthy replica server
            for server in self.replica_servers:
                is_healthy = await server.is_healthy()
                if is_healthy:
                    # Promote the replica to primary
                    success = await server.promote_to_primary()
                    if success:
                        # Update the server lists
                        self.primary_servers.remove(self.current_primary)
                        self.replica_servers.remove(server)
                        self.replica_servers.append(self.current_primary)
                        self.primary_servers.append(server)
                        
                        # Update the current primary
                        self.current_primary = server
                        
                        logger.info(f"Failover successful, new primary server: {server.id}")
                        
                        # Update metrics
                        self.metrics["successful_failovers"] += 1
                        
                        return True
            
            # No healthy replica server found
            logger.error(f"No healthy replica server found for failover")
            self.metrics["failed_failovers"] += 1
            
            return False
    
    async def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all servers.
        
        Returns:
            Dictionary of server statuses
        """
        status = {}
        
        for server_id, server in self.servers.items():
            # Get the server status
            server_status = await server.get_status()
            
            # Add the server role
            if server in self.primary_servers:
                server_status["role"] = "primary"
            else:
                server_status["role"] = "replica"
            
            # Add the server status
            status[server_id] = server_status
        
        return status
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get the status of the replication setup.
        
        Returns:
            Dictionary of replication status
        """
        status = {
            "mode": self.config.mode,
            "primary_count": len(self.primary_servers),
            "replica_count": len(self.replica_servers),
            "current_primary": self.current_primary.id if self.current_primary else None,
            "read_write_splitting": self.config.read_write_splitting,
            "failover_enabled": self.config.failover_enabled,
            "metrics": self.metrics
        }
        
        return status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get replication manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _get_replica_server(self) -> Optional[S]:
        """Get a replica server using the load balancing strategy.
        
        Returns:
            A replica server, or None if no replica server is available
        """
        if not self.replica_servers:
            return None
        
        # Get the load balancing strategy
        strategy = self.config.load_balancing_strategy.lower()
        
        if strategy == "round_robin":
            # Round-robin strategy
            server = self.replica_servers[self.round_robin_counter % len(self.replica_servers)]
            self.round_robin_counter += 1
            return server
        elif strategy == "random":
            # Random strategy
            import random
            return random.choice(self.replica_servers)
        elif strategy == "weighted":
            # Weighted strategy
            total_weight = sum(server.config.weight for server in self.replica_servers)
            if total_weight <= 0:
                return self.replica_servers[0]
            
            import random
            r = random.uniform(0, total_weight)
            upto = 0
            
            for server in self.replica_servers:
                upto += server.config.weight
                if upto >= r:
                    return server
            
            return self.replica_servers[-1]
        elif strategy == "least_connections":
            # Least connections strategy
            return min(self.replica_servers, key=lambda s: s.active_connections)
        else:
            # Default to round-robin
            server = self.replica_servers[self.round_robin_counter % len(self.replica_servers)]
            self.round_robin_counter += 1
            return server
    
    async def _health_check_loop(self) -> None:
        """Health check loop for all servers."""
        try:
            while True:
                # Sleep for the health check interval
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check the health of all servers
                for server in self.servers.values():
                    try:
                        # Check if the server is healthy
                        is_healthy = await server.is_healthy()
                        
                        # Update the server status
                        if is_healthy:
                            logger.debug(f"Server {server.id} is healthy")
                        else:
                            logger.warning(f"Server {server.id} is unhealthy")
                            
                            # Check if this is the current primary
                            if server == self.current_primary:
                                logger.warning(f"Current primary server {server.id} is unhealthy, initiating failover")
                                
                                # Perform failover
                                await self.failover()
                    except Exception as e:
                        logger.error(f"Error checking health of server {server.id}: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in health check loop: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for all servers."""
        try:
            while True:
                # Sleep for the monitoring interval
                await asyncio.sleep(self.config.monitoring_interval)
                
                # Get the status of all servers
                server_status = await self.get_server_status()
                
                # Get the replication status
                replication_status = await self.get_replication_status()
                
                # Log the status
                logger.debug(f"Replication status: {replication_status}")
                logger.debug(f"Server status: {server_status}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in monitoring loop: {e}")
    
    def __del__(self):
        """Clean up resources when the manager is garbage collected."""
        # Cancel tasks
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
