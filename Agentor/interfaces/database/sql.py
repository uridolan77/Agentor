"""
SQL database interfaces for the Agentor framework.

This module provides interfaces for interacting with SQL databases,
including SQLite, PostgreSQL, MySQL, and others.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import re
import urllib.parse

from .base import (
    DatabaseConnection, DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class SqlDialect(Enum):
    """SQL dialects supported by the SQL database connection."""
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MSSQL = "mssql"
    ORACLE = "oracle"


class SqlConnection(DatabaseConnection):
    """Connection to a SQL database."""

    def __init__(
        self,
        name: str,
        connection_string: str,
        dialect: SqlDialect,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        **kwargs
    ):
        super().__init__(name=name, connection_string=connection_string, **kwargs)
        self.dialect = dialect
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool = None
        self.in_transaction = False
        self.transaction = None
        self.transaction_conn = None
        self.transaction_cursor = None

    async def connect(self) -> DatabaseResult:
        """Connect to the SQL database."""
        try:
            # Import the appropriate database driver based on the dialect
            if self.dialect == SqlDialect.SQLITE:
                try:
                    import aiosqlite
                    self.connection = await aiosqlite.connect(self.connection_string)
                    self.connection.row_factory = aiosqlite.Row
                except ImportError:
                    return DatabaseResult.error_result(
                        "aiosqlite package is not installed. Install it with 'pip install aiosqlite'"
                    )
            elif self.dialect == SqlDialect.POSTGRESQL:
                try:
                    import asyncpg
                    self.pool = await asyncpg.create_pool(
                        self.connection_string,
                        min_size=1,
                        max_size=self.pool_size,
                        max_inactive_connection_lifetime=self.pool_recycle,
                        **self.connection_params
                    )
                except ImportError:
                    return DatabaseResult.error_result(
                        "asyncpg package is not installed. Install it with 'pip install asyncpg'"
                    )
            elif self.dialect == SqlDialect.MYSQL:
                try:
                    import aiomysql
                    self.pool = await aiomysql.create_pool(
                        **self._parse_mysql_connection_string(),
                        minsize=1,
                        maxsize=self.pool_size,
                        **self.connection_params
                    )
                except ImportError:
                    return DatabaseResult.error_result(
                        "aiomysql package is not installed. Install it with 'pip install aiomysql'"
                    )
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")

            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to {self.dialect.value} database: {self.name}")
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to connect to {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to database: {e}"))

    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the SQL database."""
        try:
            if self.dialect == SqlDialect.SQLITE:
                if self.connection:
                    await self.connection.close()
            elif self.dialect in (SqlDialect.POSTGRESQL, SqlDialect.MYSQL):
                if self.pool:
                    await self.pool.close()

            self.connected = False
            self.connection = None
            self.pool = None
            logger.info(f"Disconnected from {self.dialect.value} database: {self.name}")
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to disconnect from {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to disconnect from database: {e}"))

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the SQL database."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        try:
            self.last_activity = time.time()
            params = params or {}

            if self.dialect == SqlDialect.SQLITE:
                # Convert named parameters to SQLite format (:param)
                query, params = self._convert_params_for_sqlite(query, params)
                cursor = await self.connection.execute(query, params)
                await self.connection.commit()
                return DatabaseResult.success_result(
                    affected_rows=cursor.rowcount,
                    last_insert_id=cursor.lastrowid
                )
            elif self.dialect == SqlDialect.POSTGRESQL:
                async with self.pool.acquire() as conn:
                    # Convert named parameters to PostgreSQL format ($1, $2, etc.)
                    query, params_list = self._convert_params_for_postgres(query, params)
                    result = await conn.execute(query, *params_list)
                    # Parse the result string (e.g., "INSERT 0 1")
                    affected_rows = self._parse_pg_result(result)
                    return DatabaseResult.success_result(affected_rows=affected_rows)
            elif self.dialect == SqlDialect.MYSQL:
                async with self.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        # Convert named parameters to MySQL format (%s)
                        query, params_list = self._convert_params_for_mysql(query, params)
                        await cursor.execute(query, params_list)
                        await conn.commit()
                        return DatabaseResult.success_result(
                            affected_rows=cursor.rowcount,
                            last_insert_id=cursor.lastrowid
                        )
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")
        except Exception as e:
            logger.error(f"Failed to execute query on {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute query: {e}"))

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> DatabaseResult:
        """Execute a query multiple times with different parameters."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        try:
            self.last_activity = time.time()

            if self.dialect == SqlDialect.SQLITE:
                # Convert named parameters to SQLite format (:param)
                query, _ = self._convert_params_for_sqlite(query, params_list[0] if params_list else {})
                # Convert each params dict to a tuple in the correct order
                params_tuples = [self._dict_to_tuple_for_sqlite(query, p) for p in params_list]
                await self.connection.executemany(query, params_tuples)
                await self.connection.commit()
                return DatabaseResult.success_result(affected_rows=len(params_list))
            elif self.dialect == SqlDialect.POSTGRESQL:
                async with self.pool.acquire() as conn:
                    # For PostgreSQL, we need to use an explicit transaction for executemany
                    async with conn.transaction():
                        total_affected = 0
                        for params in params_list:
                            query_pg, params_pg = self._convert_params_for_postgres(query, params)
                            result = await conn.execute(query_pg, *params_pg)
                            total_affected += self._parse_pg_result(result)
                        return DatabaseResult.success_result(affected_rows=total_affected)
            elif self.dialect == SqlDialect.MYSQL:
                async with self.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        # Convert named parameters to MySQL format (%s)
                        query, _ = self._convert_params_for_mysql(query, params_list[0] if params_list else {})
                        # Convert each params dict to a tuple in the correct order
                        params_tuples = [self._dict_to_tuple_for_mysql(query, p) for p in params_list]
                        await cursor.executemany(query, params_tuples)
                        await conn.commit()
                        return DatabaseResult.success_result(affected_rows=cursor.rowcount)
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")
        except Exception as e:
            logger.error(f"Failed to execute many queries on {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute many queries: {e}"))

    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single row from the SQL database."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        try:
            self.last_activity = time.time()
            params = params or {}

            if self.dialect == SqlDialect.SQLITE:
                query, params = self._convert_params_for_sqlite(query, params)
                cursor = await self.connection.execute(query, params)
                row = await cursor.fetchone()
                if row:
                    return DatabaseResult.success_result(data=dict(row))
                else:
                    return DatabaseResult.success_result(data=None)
            elif self.dialect == SqlDialect.POSTGRESQL:
                async with self.pool.acquire() as conn:
                    query, params_list = self._convert_params_for_postgres(query, params)
                    row = await conn.fetchrow(query, *params_list)
                    if row:
                        return DatabaseResult.success_result(data=dict(row))
                    else:
                        return DatabaseResult.success_result(data=None)
            elif self.dialect == SqlDialect.MYSQL:
                try:
                    import aiomysql
                    async with self.pool.acquire() as conn:
                        async with conn.cursor(aiomysql.DictCursor) as cursor:
                            query, params_list = self._convert_params_for_mysql(query, params)
                            await cursor.execute(query, params_list)
                            row = await cursor.fetchone()
                            return DatabaseResult.success_result(data=row)
                except ImportError:
                    return DatabaseResult.error_result(
                        "aiomysql package is not installed. Install it with 'pip install aiomysql'"
                    )
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")
        except Exception as e:
            logger.error(f"Failed to fetch one row from {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch one row: {e}"))

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all rows from the SQL database."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        try:
            self.last_activity = time.time()
            params = params or {}

            if self.dialect == SqlDialect.SQLITE:
                query, params = self._convert_params_for_sqlite(query, params)
                cursor = await self.connection.execute(query, params)
                rows = await cursor.fetchall()
                return DatabaseResult.success_result(data=[dict(row) for row in rows])
            elif self.dialect == SqlDialect.POSTGRESQL:
                async with self.pool.acquire() as conn:
                    query, params_list = self._convert_params_for_postgres(query, params)
                    rows = await conn.fetch(query, *params_list)
                    return DatabaseResult.success_result(data=[dict(row) for row in rows])
            elif self.dialect == SqlDialect.MYSQL:
                try:
                    import aiomysql
                    async with self.pool.acquire() as conn:
                        async with conn.cursor(aiomysql.DictCursor) as cursor:
                            query, params_list = self._convert_params_for_mysql(query, params)
                            await cursor.execute(query, params_list)
                            rows = await cursor.fetchall()
                            return DatabaseResult.success_result(data=rows)
                except ImportError:
                    return DatabaseResult.error_result(
                        "aiomysql package is not installed. Install it with 'pip install aiomysql'"
                    )
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")
        except Exception as e:
            logger.error(f"Failed to fetch all rows from {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch all rows: {e}"))

    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the SQL database."""
        result = await self.fetch_one(query, params)
        if not result.success:
            return result

        if result.data is None:
            return DatabaseResult.success_result(data=None)

        # Extract the first value from the row
        if isinstance(result.data, dict) and result.data:
            return DatabaseResult.success_result(data=next(iter(result.data.values())))
        else:
            return DatabaseResult.success_result(data=None)

    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        if self.in_transaction:
            return DatabaseResult.error_result(TransactionError("Transaction already in progress"))

        try:
            self.last_activity = time.time()

            if self.dialect == SqlDialect.SQLITE:
                await self.connection.execute("BEGIN TRANSACTION")
            elif self.dialect == SqlDialect.POSTGRESQL:
                self.transaction_conn = await self.pool.acquire()
                self.transaction = self.transaction_conn.transaction()
                await self.transaction.start()
            elif self.dialect == SqlDialect.MYSQL:
                self.transaction_conn = await self.pool.acquire()
                self.transaction_cursor = await self.transaction_conn.cursor()
                await self.transaction_conn.begin()
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")

            self.in_transaction = True
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to begin transaction on {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to begin transaction: {e}"))

    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))

        try:
            self.last_activity = time.time()

            if self.dialect == SqlDialect.SQLITE:
                await self.connection.commit()
            elif self.dialect == SqlDialect.POSTGRESQL:
                await self.transaction.commit()
                await self.pool.release(self.transaction_conn)
                self.transaction = None
                self.transaction_conn = None
            elif self.dialect == SqlDialect.MYSQL:
                await self.transaction_conn.commit()
                await self.transaction_cursor.close()
                self.pool.release(self.transaction_conn)
                self.transaction_cursor = None
                self.transaction_conn = None
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")

            self.in_transaction = False
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to commit transaction on {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to commit transaction: {e}"))

    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to database"))

        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))

        try:
            self.last_activity = time.time()

            if self.dialect == SqlDialect.SQLITE:
                await self.connection.rollback()
            elif self.dialect == SqlDialect.POSTGRESQL:
                await self.transaction.rollback()
                await self.pool.release(self.transaction_conn)
                self.transaction = None
                self.transaction_conn = None
            elif self.dialect == SqlDialect.MYSQL:
                await self.transaction_conn.rollback()
                await self.transaction_cursor.close()
                self.pool.release(self.transaction_conn)
                self.transaction_cursor = None
                self.transaction_conn = None
            else:
                return DatabaseResult.error_result(f"Unsupported SQL dialect: {self.dialect}")

            self.in_transaction = False
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to rollback transaction on {self.dialect.value} database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to rollback transaction: {e}"))

    def _parse_mysql_connection_string(self) -> Dict[str, Any]:
        """Parse a MySQL connection string into a dictionary of connection parameters."""
        # Example: mysql://user:password@host:port/database
        parsed = urllib.parse.urlparse(self.connection_string)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "db": parsed.path.lstrip("/"),
            "autocommit": False
        }

    def _convert_params_for_sqlite(self, query: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert named parameters to SQLite format (:param)."""
        # SQLite already uses :param format, so we just need to ensure all params are prefixed with :
        new_params = {}
        for key, value in params.items():
            if not key.startswith(":"):
                new_params[f":{key}"] = value
            else:
                new_params[key] = value
        return query, new_params

    def _dict_to_tuple_for_sqlite(self, query: str, params: Dict[str, Any]) -> Tuple:
        """Convert a parameters dictionary to a tuple for SQLite executemany."""
        # Extract parameter names from the query (assuming :param format)
        param_names = re.findall(r":([a-zA-Z0-9_]+)", query)
        return tuple(params.get(name, params.get(f":{name}")) for name in param_names)

    def _convert_params_for_postgres(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Convert named parameters to PostgreSQL format ($1, $2, etc.)."""
        # Replace :param with $1, $2, etc.
        param_names = []
        
        def replace_param(match):
            param_name = match.group(1)
            param_names.append(param_name)
            return f"${len(param_names)}"
        
        new_query = re.sub(r":([a-zA-Z0-9_]+)", replace_param, query)
        
        # Create a list of parameter values in the correct order
        params_list = [params[name] for name in param_names]
        
        return new_query, params_list

    def _convert_params_for_mysql(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Convert named parameters to MySQL format (%s)."""
        # Replace :param with %s and build a list of values
        param_names = []
        
        def replace_param(match):
            param_name = match.group(1)
            param_names.append(param_name)
            return "%s"
        
        new_query = re.sub(r":([a-zA-Z0-9_]+)", replace_param, query)
        
        # Create a list of parameter values in the correct order
        params_list = [params[name] for name in param_names]
        
        return new_query, params_list

    def _dict_to_tuple_for_mysql(self, query: str, params: Dict[str, Any]) -> Tuple:
        """Convert a parameters dictionary to a tuple for MySQL executemany."""
        # Extract parameter names from the query (assuming %s format after conversion)
        param_names = []
        re.sub(r":([a-zA-Z0-9_]+)", lambda m: param_names.append(m.group(1)) or "%s", query)
        return tuple(params[name] for name in param_names)

    def _parse_pg_result(self, result: str) -> int:
        """Parse a PostgreSQL result string to get the number of affected rows."""
        # Result format examples: "INSERT 0 1", "UPDATE 5", "DELETE 3"
        try:
            parts = result.split()
            if len(parts) >= 2 and parts[0] in ("INSERT", "UPDATE", "DELETE"):
                return int(parts[-1])
            return 0
        except (IndexError, ValueError):
            return 0
