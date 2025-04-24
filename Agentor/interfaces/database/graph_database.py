"""
Graph database interfaces for the Agentor framework.

This module provides interfaces for interacting with graph databases,
such as Neo4j, Amazon Neptune, and ArangoDB.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar
import time
import json
import re

from .base import (
    DatabaseConnection, DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from .nosql import NoSqlType, NoSqlConnection
from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class GraphDatabase(NoSqlConnection):
    """Connection to a graph database."""

    def __init__(
        self,
        name: str,
        connection_string: str,
        **kwargs
    ):
        super().__init__(name=name, connection_string=connection_string, db_type=NoSqlType.GRAPH, **kwargs)
        self.client = None
        self.driver = None
        self.session = None

    async def connect(self) -> DatabaseResult:
        """Connect to the graph database."""
        try:
            # Determine the graph database type from the connection string
            if self.connection_string.startswith("neo4j"):
                return await self._connect_neo4j()
            elif self.connection_string.startswith("neptune"):
                return await self._connect_neptune()
            elif self.connection_string.startswith("arango"):
                return await self._connect_arango()
            else:
                return DatabaseResult.error_result(f"Unsupported graph database type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to connect to graph database: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to graph database: {e}"))

    async def _connect_neo4j(self) -> DatabaseResult:
        """Connect to a Neo4j database."""
        try:
            from neo4j import AsyncGraphDatabase
            
            # Parse the connection string
            # Format: neo4j://[username]:[password]@[host]:[port]
            self.driver = AsyncGraphDatabase.driver(self.connection_string)
            
            # Test the connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to Neo4j graph database: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "neo4j package is not installed. Install it with 'pip install neo4j'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to Neo4j: {e}"))

    async def _connect_neptune(self) -> DatabaseResult:
        """Connect to an Amazon Neptune database."""
        try:
            import gremlinpython
            from gremlinpython.driver.driver_remote_connection import DriverRemoteConnection
            from gremlinpython.process.anonymous_traversal import traversal
            
            # Parse the connection string
            # Format: neptune://[host]:[port]
            match = re.match(r"neptune://([^:]+)(?::(\d+))?", self.connection_string)
            if not match:
                return DatabaseResult.error_result("Invalid Neptune connection string")
            
            host, port = match.groups()
            port = port or "8182"
            
            # Create a connection
            connection = DriverRemoteConnection(f"wss://{host}:{port}/gremlin", "g")
            self.client = traversal().withRemote(connection)
            
            # Test the connection
            await self.client.V().limit(1).toList()
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to Neptune graph database: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "gremlinpython package is not installed. Install it with 'pip install gremlinpython'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Neptune: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to Neptune: {e}"))

    async def _connect_arango(self) -> DatabaseResult:
        """Connect to an ArangoDB database."""
        try:
            from aioarangodb import ArangoClient
            
            # Parse the connection string
            # Format: arango://[username]:[password]@[host]:[port]/[database]
            match = re.match(r"arango://(?:([^:]+):([^@]+)@)?([^:]+)(?::(\d+))?(?:/([^/]+))?", self.connection_string)
            if not match:
                return DatabaseResult.error_result("Invalid ArangoDB connection string")
            
            username, password, host, port, database = match.groups()
            port = port or "8529"
            username = username or "root"
            password = password or ""
            database = database or "_system"
            
            # Create a client
            self.client = ArangoClient(hosts=f"http://{host}:{port}")
            
            # Connect to the database
            self.db = await self.client.db(database, username=username, password=password)
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to ArangoDB graph database: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "aioarangodb package is not installed. Install it with 'pip install aioarangodb'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to ArangoDB: {e}"))

    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the graph database."""
        try:
            if self.connection_string.startswith("neo4j") and self.driver:
                await self.driver.close()
            elif self.connection_string.startswith("neptune") and self.client:
                # Gremlin Python doesn't have an async close method
                pass
            elif self.connection_string.startswith("arango") and self.client:
                await self.client.close()
            
            self.connected = False
            self.driver = None
            self.client = None
            logger.info(f"Disconnected from graph database: {self.name}")
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to disconnect from graph database: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to disconnect from graph database: {e}"))

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the graph database."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to graph database"))
        
        try:
            self.last_activity = time.time()
            params = params or {}
            
            if self.connection_string.startswith("neo4j"):
                async with self.driver.session() as session:
                    result = await session.run(query, params)
                    records = await result.values()
                    summary = await result.consume()
                    
                    return DatabaseResult.success_result(
                        data=records,
                        affected_rows=summary.counters.nodes_created + summary.counters.relationships_created,
                        metadata={
                            "nodes_created": summary.counters.nodes_created,
                            "nodes_deleted": summary.counters.nodes_deleted,
                            "relationships_created": summary.counters.relationships_created,
                            "relationships_deleted": summary.counters.relationships_deleted,
                            "properties_set": summary.counters.properties_set
                        }
                    )
            elif self.connection_string.startswith("neptune"):
                # For Neptune, we need to use Gremlin queries
                # The query should be a Gremlin query string
                result = await self.client.submit(query, params).all()
                return DatabaseResult.success_result(data=result)
            elif self.connection_string.startswith("arango"):
                # For ArangoDB, we can use AQL queries
                cursor = await self.db.aql.execute(query, bind_vars=params)
                result = [doc async for doc in cursor]
                return DatabaseResult.success_result(data=result)
            else:
                return DatabaseResult.error_result(f"Unsupported graph database type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to execute query on graph database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute query: {e}"))

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> DatabaseResult:
        """Execute a query multiple times with different parameters."""
        results = []
        total_affected = 0
        
        for params in params_list:
            result = await self.execute(query, params)
            results.append(result)
            if result.success:
                total_affected += result.affected_rows
        
        return DatabaseResult.success_result(
            data=results,
            affected_rows=total_affected
        )

    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single result from the graph database."""
        result = await self.execute(query, params)
        
        if not result.success:
            return result
        
        if not result.data:
            return DatabaseResult.success_result(data=None)
        
        # Return the first record
        return DatabaseResult.success_result(data=result.data[0] if result.data else None)

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all results from the graph database."""
        return await self.execute(query, params)

    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the graph database."""
        result = await self.fetch_one(query, params)
        
        if not result.success or result.data is None:
            return result
        
        # If the result is a list or tuple with one item, return that item
        if isinstance(result.data, (list, tuple)) and len(result.data) == 1:
            return DatabaseResult.success_result(data=result.data[0])
        
        # Otherwise, return the result as is
        return result

    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to graph database"))
        
        if self.in_transaction:
            return DatabaseResult.error_result(TransactionError("Transaction already in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("neo4j"):
                self.session = await self.driver.session()
                self.transaction = await self.session.begin_transaction()
                self.in_transaction = True
                return DatabaseResult.success_result()
            elif self.connection_string.startswith("arango"):
                self.transaction = await self.db.begin_transaction()
                self.in_transaction = True
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Transactions not supported for {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to begin transaction on graph database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to begin transaction: {e}"))

    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to graph database"))
        
        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("neo4j"):
                await self.transaction.commit()
                await self.session.close()
                self.transaction = None
                self.session = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            elif self.connection_string.startswith("arango"):
                await self.transaction.commit()
                self.transaction = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Transactions not supported for {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to commit transaction on graph database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to commit transaction: {e}"))

    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to graph database"))
        
        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("neo4j"):
                await self.transaction.rollback()
                await self.session.close()
                self.transaction = None
                self.session = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            elif self.connection_string.startswith("arango"):
                await self.transaction.abort()
                self.transaction = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Transactions not supported for {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to rollback transaction on graph database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to rollback transaction: {e}"))
