"""
MySQL adapter with replication support for the Agentor framework.

This module provides a specialized adapter for MySQL with replication support.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from ..base import (
    DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from ..replication import (
    ReplicationConfig, ReplicationMode, ReplicationRole, ServerConfig,
    MySqlReplicationManager
)
from ..pool import (
    ConnectionPoolConfig, ConnectionValidationMode, MySqlConnectionPool
)
from .mysql import MySqlAdapter

logger = logging.getLogger(__name__)


class MySqlReplicationAdapter(MySqlAdapter):
    """MySQL adapter with replication support."""
    
    def __init__(
        self,
        name: str,
        replication_config: ReplicationConfig,
        **kwargs
    ):
        """Initialize the MySQL adapter with replication support.
        
        Args:
            name: The name of the adapter
            replication_config: The replication configuration
            **kwargs: Additional arguments for the MySQL adapter
        """
        super().__init__(name=name, **kwargs)
        
        # Replication configuration
        self.replication_config = replication_config
        
        # Replication manager
        self.replication_manager: Optional[MySqlReplicationManager] = None
        
        # Replication metrics
        self.replication_metrics = {
            "total_reads": 0,
            "total_writes": 0,
            "primary_reads": 0,
            "replica_reads": 0,
            "failed_reads": 0,
            "failed_writes": 0,
            "failovers": 0
        }
    
    async def connect(self) -> DatabaseResult:
        """Connect to the database.
        
        Returns:
            DatabaseResult indicating success or failure
        """
        # Check if already connected
        if self.connected:
            return DatabaseResult.success_result()
        
        try:
            # Create a connection factory
            def connection_factory(server_config: ServerConfig) -> MySqlConnectionPool:
                # Create a connection pool configuration
                pool_config = ConnectionPoolConfig(
                    min_size=1,
                    max_size=server_config.max_connections,
                    max_lifetime=3600.0,
                    connect_timeout=server_config.connection_timeout,
                    validation_mode=ConnectionValidationMode.PING,
                    validation_interval=60.0,
                    health_check_interval=60.0,
                    collect_metrics=True
                )
                
                # Create a connection pool
                pool = MySqlConnectionPool(
                    name=f"{server_config.host}:{server_config.port}",
                    config=pool_config,
                    host=server_config.host,
                    port=server_config.port,
                    user=server_config.user,
                    password=server_config.password,
                    database=server_config.database,
                    charset=self.charset
                )
                
                return pool
            
            # Create a replication manager
            self.replication_manager = MySqlReplicationManager(
                name=self.name,
                config=self.replication_config,
                connection_factory=connection_factory
            )
            
            # Initialize the replication manager
            await self.replication_manager.initialize()
            
            # Set the connection status
            self.connected = True
            
            # Get the replication status
            replication_status = await self.replication_manager.get_replication_status()
            logger.info(f"Connected to MySQL replication setup: {replication_status}")
            
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Error connecting to MySQL replication setup: {e}")
            return DatabaseResult.error_result(ConnectionError(str(e)))
    
    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the database.
        
        Returns:
            DatabaseResult indicating success or failure
        """
        # Check if connected
        if not self.connected:
            return DatabaseResult.success_result()
        
        try:
            # Close the replication manager
            if self.replication_manager:
                await self.replication_manager.close()
                self.replication_manager = None
            
            # Set the connection status
            self.connected = False
            
            logger.info(f"Disconnected from MySQL replication setup")
            
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Error disconnecting from MySQL replication setup: {e}")
            return DatabaseResult.error_result(ConnectionError(str(e)))
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query.
        
        Args:
            query: The query to execute
            params: The query parameters
            
        Returns:
            DatabaseResult indicating success or failure
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return DatabaseResult.error_result(ConnectionError("Not connected to the database"))
        
        # Update metrics
        self.replication_metrics["total_writes"] += 1
        
        try:
            # Get a write connection
            conn, server = await self.replication_manager.get_write_connection()
            if not conn or not server:
                self.replication_metrics["failed_writes"] += 1
                return DatabaseResult.error_result(ConnectionError("Failed to get a write connection"))
            
            try:
                # Execute the query
                async with conn.cursor() as cursor:
                    # Convert named parameters to positional parameters if needed
                    if params:
                        query, params = self._convert_params(query, params)
                    
                    # Execute the query
                    await cursor.execute(query, params or {})
                    
                    # Get the result
                    result = DatabaseResult.success_result(
                        affected_rows=cursor.rowcount,
                        last_insert_id=cursor.lastrowid
                    )
                    
                    return result
            finally:
                # Release the connection
                await self.replication_manager.release_connection(conn, server)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            self.replication_metrics["failed_writes"] += 1
            return DatabaseResult.error_result(QueryError(str(e)))
    
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single row from the database.
        
        Args:
            query: The query to execute
            params: The query parameters
            
        Returns:
            DatabaseResult with the row data
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return DatabaseResult.error_result(ConnectionError("Not connected to the database"))
        
        # Update metrics
        self.replication_metrics["total_reads"] += 1
        
        try:
            # Get a read connection
            conn, server = await self.replication_manager.get_read_connection()
            if not conn or not server:
                self.replication_metrics["failed_reads"] += 1
                return DatabaseResult.error_result(ConnectionError("Failed to get a read connection"))
            
            try:
                # Execute the query
                async with conn.cursor() as cursor:
                    # Convert named parameters to positional parameters if needed
                    if params:
                        query, params = self._convert_params(query, params)
                    
                    # Execute the query
                    await cursor.execute(query, params or {})
                    
                    # Fetch the row
                    row = await cursor.fetchone()
                    
                    # Get the result
                    if row:
                        # Convert the row to a dictionary
                        columns = [column[0] for column in cursor.description]
                        row_dict = dict(zip(columns, row))
                        
                        result = DatabaseResult.success_result(data=row_dict)
                    else:
                        result = DatabaseResult.success_result(data=None)
                    
                    # Update metrics
                    if server.is_primary_server:
                        self.replication_metrics["primary_reads"] += 1
                    else:
                        self.replication_metrics["replica_reads"] += 1
                    
                    return result
            finally:
                # Release the connection
                await self.replication_manager.release_connection(conn, server)
        except Exception as e:
            logger.error(f"Error fetching row: {e}")
            self.replication_metrics["failed_reads"] += 1
            return DatabaseResult.error_result(QueryError(str(e)))
    
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all rows from the database.
        
        Args:
            query: The query to execute
            params: The query parameters
            
        Returns:
            DatabaseResult with the rows data
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return DatabaseResult.error_result(ConnectionError("Not connected to the database"))
        
        # Update metrics
        self.replication_metrics["total_reads"] += 1
        
        try:
            # Get a read connection
            conn, server = await self.replication_manager.get_read_connection()
            if not conn or not server:
                self.replication_metrics["failed_reads"] += 1
                return DatabaseResult.error_result(ConnectionError("Failed to get a read connection"))
            
            try:
                # Execute the query
                async with conn.cursor() as cursor:
                    # Convert named parameters to positional parameters if needed
                    if params:
                        query, params = self._convert_params(query, params)
                    
                    # Execute the query
                    await cursor.execute(query, params or {})
                    
                    # Fetch all rows
                    rows = await cursor.fetchall()
                    
                    # Convert the rows to dictionaries
                    columns = [column[0] for column in cursor.description]
                    rows_dict = [dict(zip(columns, row)) for row in rows]
                    
                    # Get the result
                    result = DatabaseResult.success_result(data=rows_dict)
                    
                    # Update metrics
                    if server.is_primary_server:
                        self.replication_metrics["primary_reads"] += 1
                    else:
                        self.replication_metrics["replica_reads"] += 1
                    
                    return result
            finally:
                # Release the connection
                await self.replication_manager.release_connection(conn, server)
        except Exception as e:
            logger.error(f"Error fetching rows: {e}")
            self.replication_metrics["failed_reads"] += 1
            return DatabaseResult.error_result(QueryError(str(e)))
    
    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction.
        
        Returns:
            DatabaseResult indicating success or failure
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return DatabaseResult.error_result(ConnectionError("Not connected to the database"))
        
        # Check if already in a transaction
        if self.in_transaction:
            return DatabaseResult.error_result(TransactionError("Already in a transaction"))
        
        try:
            # Get a write connection
            conn, server = await self.replication_manager.get_write_connection()
            if not conn or not server:
                return DatabaseResult.error_result(ConnectionError("Failed to get a write connection"))
            
            try:
                # Begin a transaction
                async with conn.cursor() as cursor:
                    await cursor.execute("BEGIN")
                
                # Store the transaction connection and server
                self.transaction_connection = conn
                self.transaction_server = server
                self.in_transaction = True
                
                return DatabaseResult.success_result()
            except Exception as e:
                # Release the connection
                await self.replication_manager.release_connection(conn, server)
                
                logger.error(f"Error beginning transaction: {e}")
                return DatabaseResult.error_result(TransactionError(str(e)))
        except Exception as e:
            logger.error(f"Error beginning transaction: {e}")
            return DatabaseResult.error_result(TransactionError(str(e)))
    
    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction.
        
        Returns:
            DatabaseResult indicating success or failure
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return DatabaseResult.error_result(ConnectionError("Not connected to the database"))
        
        # Check if in a transaction
        if not self.in_transaction or not self.transaction_connection or not self.transaction_server:
            return DatabaseResult.error_result(TransactionError("Not in a transaction"))
        
        try:
            # Commit the transaction
            async with self.transaction_connection.cursor() as cursor:
                await cursor.execute("COMMIT")
            
            # Reset the transaction state
            conn = self.transaction_connection
            server = self.transaction_server
            self.transaction_connection = None
            self.transaction_server = None
            self.in_transaction = False
            
            # Release the connection
            await self.replication_manager.release_connection(conn, server)
            
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Error committing transaction: {e}")
            return DatabaseResult.error_result(TransactionError(str(e)))
    
    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction.
        
        Returns:
            DatabaseResult indicating success or failure
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return DatabaseResult.error_result(ConnectionError("Not connected to the database"))
        
        # Check if in a transaction
        if not self.in_transaction or not self.transaction_connection or not self.transaction_server:
            return DatabaseResult.error_result(TransactionError("Not in a transaction"))
        
        try:
            # Rollback the transaction
            async with self.transaction_connection.cursor() as cursor:
                await cursor.execute("ROLLBACK")
            
            # Reset the transaction state
            conn = self.transaction_connection
            server = self.transaction_server
            self.transaction_connection = None
            self.transaction_server = None
            self.in_transaction = False
            
            # Release the connection
            await self.replication_manager.release_connection(conn, server)
            
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Error rolling back transaction: {e}")
            return DatabaseResult.error_result(TransactionError(str(e)))
    
    async def get_replication_status(self) -> Dict[str, Any]:
        """Get the status of the replication setup.
        
        Returns:
            Dictionary of replication status
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return {}
        
        # Get the replication status
        replication_status = await self.replication_manager.get_replication_status()
        
        # Add adapter-specific metrics
        replication_status["adapter_metrics"] = self.replication_metrics
        
        return replication_status
    
    async def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all servers.
        
        Returns:
            Dictionary of server statuses
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return {}
        
        # Get the server status
        return await self.replication_manager.get_server_status()
    
    async def failover(self) -> bool:
        """Perform a failover to a new primary server.
        
        Returns:
            True if the failover was successful, False otherwise
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            return False
        
        # Update metrics
        self.replication_metrics["failovers"] += 1
        
        # Perform a failover
        return await self.replication_manager.failover()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Get the base metrics
        metrics = await super().get_metrics()
        
        # Add replication metrics
        metrics.update(self.replication_metrics)
        
        # Add replication manager metrics
        if self.replication_manager:
            replication_manager_metrics = await self.replication_manager.get_metrics()
            metrics["replication_manager"] = replication_manager_metrics
        
        return metrics
    
    async def stream_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream the results of a query.
        
        Args:
            query: The query to execute
            params: The query parameters
            chunk_size: The number of rows to fetch at a time
            
        Yields:
            Rows from the query
        """
        # Check if connected
        if not self.connected or not self.replication_manager:
            raise ConnectionError("Not connected to the database")
        
        # Update metrics
        self.replication_metrics["total_reads"] += 1
        
        # Get a read connection
        conn, server = await self.replication_manager.get_read_connection()
        if not conn or not server:
            self.replication_metrics["failed_reads"] += 1
            raise ConnectionError("Failed to get a read connection")
        
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Convert named parameters to positional parameters if needed
                if params:
                    query, params = self._convert_params(query, params)
                
                # Execute the query
                await cursor.execute(query, params or {})
                
                # Get the column names
                columns = [column[0] for column in cursor.description]
                
                # Fetch rows in chunks
                while True:
                    rows = await cursor.fetchmany(chunk_size)
                    if not rows:
                        break
                    
                    # Convert the rows to dictionaries
                    for row in rows:
                        yield dict(zip(columns, row))
                
                # Update metrics
                if server.is_primary_server:
                    self.replication_metrics["primary_reads"] += 1
                else:
                    self.replication_metrics["replica_reads"] += 1
        finally:
            # Release the connection
            await self.replication_manager.release_connection(conn, server)
