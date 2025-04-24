"""
Key-value store interfaces for the Agentor framework.

This module provides interfaces for interacting with key-value NoSQL databases,
such as Redis, DynamoDB, and Memcached.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
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
K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class KeyValueStore(NoSqlConnection, Generic[K, V]):
    """Connection to a key-value NoSQL database."""

    def __init__(
        self,
        name: str,
        connection_string: str,
        **kwargs
    ):
        super().__init__(name=name, connection_string=connection_string, db_type=NoSqlType.KEY_VALUE, **kwargs)
        self.client = None
        self.default_ttl = kwargs.get("default_ttl", 0)  # 0 means no expiration

    async def connect(self) -> DatabaseResult:
        """Connect to the key-value store."""
        try:
            # Determine the key-value store type from the connection string
            if self.connection_string.startswith("redis"):
                return await self._connect_redis()
            elif self.connection_string.startswith("dynamodb"):
                return await self._connect_dynamodb()
            elif self.connection_string.startswith("memcached"):
                return await self._connect_memcached()
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to connect to key-value store: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to key-value store: {e}"))

    async def _connect_redis(self) -> DatabaseResult:
        """Connect to a Redis database."""
        try:
            import redis.asyncio as redis
            
            # Parse the connection string
            # Format: redis://[[username]:[password]@][host][:port][/database]
            self.client = redis.from_url(self.connection_string)
            
            # Test the connection
            await self.client.ping()
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to Redis key-value store: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "redis package is not installed. Install it with 'pip install redis'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to Redis: {e}"))

    async def _connect_dynamodb(self) -> DatabaseResult:
        """Connect to a DynamoDB database."""
        try:
            import boto3
            import aioboto3
            
            # Parse the connection string
            # Format: dynamodb://[region]/[table]
            match = re.match(r"dynamodb://([^/]+)/([^/]+)", self.connection_string)
            if not match:
                return DatabaseResult.error_result("Invalid DynamoDB connection string")
            
            region, table = match.groups()
            
            # Create a session
            session = aioboto3.Session()
            self.client = await session.resource("dynamodb", region_name=region)
            self.table = await self.client.Table(table)
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to DynamoDB key-value store: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "boto3 and aioboto3 packages are not installed. Install them with 'pip install boto3 aioboto3'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to DynamoDB: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to DynamoDB: {e}"))

    async def _connect_memcached(self) -> DatabaseResult:
        """Connect to a Memcached database."""
        try:
            import aiomcache
            
            # Parse the connection string
            # Format: memcached://[host][:port]
            match = re.match(r"memcached://([^:]+)(?::(\d+))?", self.connection_string)
            if not match:
                return DatabaseResult.error_result("Invalid Memcached connection string")
            
            host, port = match.groups()
            port = int(port) if port else 11211
            
            self.client = aiomcache.Client(host, port)
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to Memcached key-value store: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "aiomcache package is not installed. Install it with 'pip install aiomcache'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Memcached: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to Memcached: {e}"))

    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the key-value store."""
        try:
            if self.client:
                if self.connection_string.startswith("redis"):
                    await self.client.close()
                elif self.connection_string.startswith("dynamodb"):
                    await self.client.close()
                elif self.connection_string.startswith("memcached"):
                    self.client.close()
            
            self.connected = False
            self.client = None
            logger.info(f"Disconnected from key-value store: {self.name}")
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to disconnect from key-value store: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to disconnect from key-value store: {e}"))

    async def get(self, key: K) -> DatabaseResult:
        """Get a value from the key-value store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                value = await self.client.get(str(key))
                if value:
                    try:
                        # Try to decode as JSON
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # If not JSON, decode as string
                        value = value.decode("utf-8")
                return DatabaseResult.success_result(data=value)
            elif self.connection_string.startswith("dynamodb"):
                response = await self.table.get_item(Key={"key": str(key)})
                item = response.get("Item")
                if item:
                    return DatabaseResult.success_result(data=item.get("value"))
                else:
                    return DatabaseResult.success_result(data=None)
            elif self.connection_string.startswith("memcached"):
                value = await self.client.get(str(key).encode("utf-8"))
                if value:
                    try:
                        # Try to decode as JSON
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # If not JSON, decode as string
                        value = value.decode("utf-8")
                return DatabaseResult.success_result(data=value)
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to get value for key {key}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to get value: {e}"))

    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> DatabaseResult:
        """Set a value in the key-value store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            ttl = ttl or self.default_ttl
            
            # Convert value to JSON if it's not a string
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
            
            if self.connection_string.startswith("redis"):
                if ttl > 0:
                    await self.client.setex(str(key), ttl, value)
                else:
                    await self.client.set(str(key), value)
                return DatabaseResult.success_result(affected_rows=1)
            elif self.connection_string.startswith("dynamodb"):
                item = {
                    "key": str(key),
                    "value": value
                }
                if ttl > 0:
                    item["ttl"] = int(time.time()) + ttl
                await self.table.put_item(Item=item)
                return DatabaseResult.success_result(affected_rows=1)
            elif self.connection_string.startswith("memcached"):
                if isinstance(value, str):
                    value = value.encode("utf-8")
                await self.client.set(str(key).encode("utf-8"), value, ttl if ttl > 0 else 0)
                return DatabaseResult.success_result(affected_rows=1)
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to set value for key {key}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to set value: {e}"))

    async def delete(self, key: K) -> DatabaseResult:
        """Delete a value from the key-value store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                result = await self.client.delete(str(key))
                return DatabaseResult.success_result(affected_rows=result)
            elif self.connection_string.startswith("dynamodb"):
                await self.table.delete_item(Key={"key": str(key)})
                return DatabaseResult.success_result(affected_rows=1)
            elif self.connection_string.startswith("memcached"):
                result = await self.client.delete(str(key).encode("utf-8"))
                return DatabaseResult.success_result(affected_rows=1 if result else 0)
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to delete value for key {key}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to delete value: {e}"))

    async def exists(self, key: K) -> DatabaseResult:
        """Check if a key exists in the key-value store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                result = await self.client.exists(str(key))
                return DatabaseResult.success_result(data=bool(result))
            elif self.connection_string.startswith("dynamodb"):
                response = await self.table.get_item(Key={"key": str(key)})
                return DatabaseResult.success_result(data="Item" in response)
            elif self.connection_string.startswith("memcached"):
                value = await self.client.get(str(key).encode("utf-8"))
                return DatabaseResult.success_result(data=value is not None)
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to check if key {key} exists: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to check if key exists: {e}"))

    async def increment(self, key: K, amount: int = 1) -> DatabaseResult:
        """Increment a value in the key-value store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                result = await self.client.incrby(str(key), amount)
                return DatabaseResult.success_result(data=result, affected_rows=1)
            elif self.connection_string.startswith("dynamodb"):
                response = await self.table.update_item(
                    Key={"key": str(key)},
                    UpdateExpression="SET #value = if_not_exists(#value, :zero) + :amount",
                    ExpressionAttributeNames={"#value": "value"},
                    ExpressionAttributeValues={":amount": amount, ":zero": 0},
                    ReturnValues="UPDATED_NEW"
                )
                return DatabaseResult.success_result(
                    data=response.get("Attributes", {}).get("value", amount),
                    affected_rows=1
                )
            elif self.connection_string.startswith("memcached"):
                # Memcached doesn't have atomic increment for non-numeric values
                # So we need to get, increment, and set
                value = await self.client.get(str(key).encode("utf-8"))
                if value:
                    try:
                        value = int(value)
                    except ValueError:
                        return DatabaseResult.error_result("Value is not an integer")
                else:
                    value = 0
                
                value += amount
                await self.client.set(str(key).encode("utf-8"), str(value).encode("utf-8"))
                return DatabaseResult.success_result(data=value, affected_rows=1)
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to increment value for key {key}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to increment value: {e}"))

    async def expire(self, key: K, ttl: int) -> DatabaseResult:
        """Set an expiration time for a key."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                result = await self.client.expire(str(key), ttl)
                return DatabaseResult.success_result(data=bool(result), affected_rows=1 if result else 0)
            elif self.connection_string.startswith("dynamodb"):
                response = await self.table.update_item(
                    Key={"key": str(key)},
                    UpdateExpression="SET #ttl = :ttl",
                    ExpressionAttributeNames={"#ttl": "ttl"},
                    ExpressionAttributeValues={":ttl": int(time.time()) + ttl},
                    ReturnValues="UPDATED_NEW"
                )
                return DatabaseResult.success_result(affected_rows=1)
            elif self.connection_string.startswith("memcached"):
                # Memcached doesn't support changing TTL without setting the value
                # So we need to get and set with the new TTL
                value = await self.client.get(str(key).encode("utf-8"))
                if value:
                    await self.client.set(str(key).encode("utf-8"), value, ttl)
                    return DatabaseResult.success_result(affected_rows=1)
                else:
                    return DatabaseResult.success_result(affected_rows=0)
            else:
                return DatabaseResult.error_result(f"Unsupported key-value store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to set expiration: {e}"))

    # Implement the abstract methods from DatabaseConnection
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the key-value store.
        
        For key-value stores, this is typically a JSON command like {"command": "get", "key": "mykey"}.
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            params = params or {}
            
            # Parse the query to determine the operation
            query_data = json.loads(query)
            command = query_data.get("command")
            
            if not command:
                return DatabaseResult.error_result("Query must include 'command'")
            
            if command == "get":
                key = query_data.get("key")
                if not key:
                    return DatabaseResult.error_result("Get command requires 'key'")
                return await self.get(key)
            elif command == "set":
                key = query_data.get("key")
                value = query_data.get("value")
                ttl = query_data.get("ttl", self.default_ttl)
                if not key or value is None:
                    return DatabaseResult.error_result("Set command requires 'key' and 'value'")
                return await self.set(key, value, ttl)
            elif command == "delete":
                key = query_data.get("key")
                if not key:
                    return DatabaseResult.error_result("Delete command requires 'key'")
                return await self.delete(key)
            elif command == "exists":
                key = query_data.get("key")
                if not key:
                    return DatabaseResult.error_result("Exists command requires 'key'")
                return await self.exists(key)
            elif command == "increment":
                key = query_data.get("key")
                amount = query_data.get("amount", 1)
                if not key:
                    return DatabaseResult.error_result("Increment command requires 'key'")
                return await self.increment(key, amount)
            elif command == "expire":
                key = query_data.get("key")
                ttl = query_data.get("ttl")
                if not key or ttl is None:
                    return DatabaseResult.error_result("Expire command requires 'key' and 'ttl'")
                return await self.expire(key, ttl)
            else:
                return DatabaseResult.error_result(f"Unsupported command: {command}")
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to execute query on key-value store: {self.name}, error: {e}")
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
        """Fetch a single value from the key-value store."""
        try:
            # Parse the query to determine the key
            query_data = json.loads(query)
            key = query_data.get("key")
            
            if not key:
                return DatabaseResult.error_result("Query must include 'key'")
            
            return await self.get(key)
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to fetch one value: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch one value: {e}"))

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch multiple values from the key-value store."""
        try:
            # Parse the query to determine the keys
            query_data = json.loads(query)
            keys = query_data.get("keys")
            
            if not keys:
                return DatabaseResult.error_result("Query must include 'keys'")
            
            results = {}
            for key in keys:
                result = await self.get(key)
                if result.success:
                    results[key] = result.data
            
            return DatabaseResult.success_result(data=results)
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to fetch all values: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch all values: {e}"))

    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the key-value store."""
        return await self.fetch_one(query, params)

    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction.
        
        Note: Not all key-value stores support transactions.
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                self.transaction = self.client.pipeline()
                self.in_transaction = True
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Transactions not supported for {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to begin transaction on key-value store: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to begin transaction: {e}"))

    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                results = await self.transaction.execute()
                self.transaction = None
                self.in_transaction = False
                return DatabaseResult.success_result(data=results)
            else:
                return DatabaseResult.error_result(f"Transactions not supported for {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to commit transaction on key-value store: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to commit transaction: {e}"))

    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to key-value store"))
        
        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("redis"):
                self.transaction.reset()
                self.transaction = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Transactions not supported for {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to rollback transaction on key-value store: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to rollback transaction: {e}"))
