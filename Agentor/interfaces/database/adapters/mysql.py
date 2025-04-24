"""
MySQL adapter for the Agentor framework.

This module provides a specialized adapter for MySQL databases with enhanced features.
"""

import time
from typing import Dict, Any, Optional, List, Tuple, TypeVar
import logging

from ..pool import ConnectionPoolConfig, ConnectionValidationMode, MySqlConnectionPool
from ..base import (
    DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from ..sql import SqlDialect
from ..enhanced.sql import EnhancedSqlConnection
from ..resilience import (
    with_database_resilience,
    DATABASE_CONNECTION_RETRY_CONFIG, DATABASE_QUERY_RETRY_CONFIG, DATABASE_TRANSACTION_RETRY_CONFIG
)
# Import TimeoutStrategy from the appropriate location
TimeoutStrategy = "ADAPTIVE"  # Placeholder for the actual import

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class MySqlAdapter(EnhancedSqlConnection):
    """MySQL adapter with enhanced features and MySQL-specific optimizations."""

    def __init__(
        self,
        name: str,
        connection_string: str,
        pool_size: int = 10,
        pool_recycle: int = 3600,
        charset: str = "utf8mb4",
        collation: str = "utf8mb4_unicode_ci",
        autocommit: bool = False,
        **kwargs
    ):
        """Initialize the MySQL adapter.

        Args:
            name: The name of the connection
            connection_string: The connection string for the database
            pool_size: The maximum number of connections in the pool
            pool_recycle: The connection recycle time in seconds
            charset: The MySQL charset to use
            collation: The MySQL collation to use
            autocommit: Whether to enable autocommit mode
            **kwargs: Additional connection parameters
        """
        # Ensure the dialect is MySQL
        super().__init__(
            name=name,
            dialect=SqlDialect.MYSQL,
            connection_string=connection_string,
            pool_size=pool_size,
            pool_recycle=pool_recycle,
            **kwargs
        )

        # MySQL-specific settings
        self.charset = charset
        self.collation = collation
        self.autocommit = autocommit

        # MySQL-specific metrics
        self.mysql_metrics = {
            "slow_queries": 0,
            "deadlocks": 0,
            "connection_errors": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "prepared_statements": 0,
            "stored_procedures": 0,
            "user_defined_functions": 0,
            "triggers": 0,
            "events": 0
        }

        # Connection pool configuration
        self.pool_config = ConnectionPoolConfig(
            min_size=1,
            max_size=pool_size,
            max_lifetime=pool_recycle,
            connect_timeout=30.0,
            validation_mode=ConnectionValidationMode.PING,
            validation_interval=60.0,
            health_check_interval=60.0,
            collect_metrics=True
        )

        # Create the pool
        self.pool = None

    @with_database_resilience(
        database="mysql",
        operation="connect",
        retry_config=DATABASE_CONNECTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=False
    )
    async def connect(self) -> DatabaseResult:
        """Connect to the MySQL database with resilience patterns."""
        try:
            # Check if the pool is initialized
            if not self.pool:
                return DatabaseResult.error_result(ConnectionError("Connection pool is not initialized"))

            # Parse the connection string
            conn_params = self._parse_mysql_connection_string()

            # Create the connection pool
            self.pool = MySqlConnectionPool(
                name=self.name,
                config=self.pool_config,
                host=conn_params["host"],
                port=conn_params["port"],
                user=conn_params["user"],
                password=conn_params.get("password"),
                database=conn_params.get("db"),
                charset=self.charset,
                autocommit=self.autocommit,
                **self.connection_params
            )

            # Initialize the pool
            await self.pool.initialize()

            # Test the connection
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Set the collation
                    await cursor.execute(f"SET NAMES {self.charset} COLLATE {self.collation}")

                    # Test the connection
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    if not result or result[0] != 1:
                        return DatabaseResult.error_result("Failed to connect to MySQL database")

            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to MySQL database: {self.name}")

            return DatabaseResult.success_result()
        except Exception as e:
            self.mysql_metrics["connection_errors"] += 1
            logger.error(f"Failed to connect to MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to MySQL database: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="execute",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the MySQL database with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()

            try:
                async with conn.cursor() as cursor:
                    # Convert named parameters to MySQL format (%s)
                    query, params_list = self._convert_params_for_mysql(query, params or {})

                    # Execute the query
                    await cursor.execute(query, params_list)

                    # Commit if not in a transaction
                    if not self.in_transaction and not self.autocommit:
                        await conn.commit()

                    # Get the result
                    result = DatabaseResult.success_result(
                        affected_rows=cursor.rowcount,
                        last_insert_id=cursor.lastrowid
                    )
            finally:
                # Release the connection
                await self.pool.release(conn)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for deadlocks
            if "deadlock" in str(e).lower():
                self.mysql_metrics["deadlocks"] += 1

            logger.error(f"Failed to execute query on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute query: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="fetch_one",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single row from the MySQL database with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Convert named parameters to MySQL format (%s)
                    query, params_list = self._convert_params_for_mysql(query, params or {})

                    # Execute the query
                    await cursor.execute(query, params_list)

                    # Fetch the result
                    row = await cursor.fetchone()

                    # Get the result
                    result = DatabaseResult.success_result(data=row)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to fetch one row from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch one row: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="fetch_all",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=60.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all rows from the MySQL database with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Convert named parameters to MySQL format (%s)
                    query, params_list = self._convert_params_for_mysql(query, params or {})

                    # Execute the query
                    await cursor.execute(query, params_list)

                    # Fetch the results
                    rows = await cursor.fetchall()

                    # Get the result
                    result = DatabaseResult.success_result(data=rows)
            finally:
                # Release the connection
                await self.pool.release(conn)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to fetch all rows from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch all rows: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="fetch_value",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the MySQL database with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Convert named parameters to MySQL format (%s)
                    query, params_list = self._convert_params_for_mysql(query, params or {})

                    # Execute the query
                    await cursor.execute(query, params_list)

                    # Fetch the result
                    row = await cursor.fetchone()

                    # Get the result
                    if row and len(row) > 0:
                        result = DatabaseResult.success_result(data=row[0])
                    else:
                        result = DatabaseResult.success_result(data=None)
            finally:
                # Release the connection
                await self.pool.release(conn)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to fetch value from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch value: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="execute_many",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=60.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> DatabaseResult:
        """Execute a query multiple times with different parameters."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        if not params_list:
            return DatabaseResult.success_result(affected_rows=0)

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Convert named parameters to MySQL format (%s)
                    query, _ = self._convert_params_for_mysql(query, params_list[0])

                    # Convert each params dict to a tuple in the correct order
                    params_tuples = [self._dict_to_tuple_for_mysql(query, p) for p in params_list]

                    # Execute the query
                    await cursor.executemany(query, params_tuples)

                    # Commit if not in a transaction
                    if not self.in_transaction and not self.autocommit:
                        await conn.commit()

                    # Get the result
                    result = DatabaseResult.success_result(affected_rows=cursor.rowcount)
            finally:
                # Release the connection
                await self.pool.release(conn)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow query detected: {query} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for deadlocks
            if "deadlock" in str(e).lower():
                self.mysql_metrics["deadlocks"] += 1

            logger.error(f"Failed to execute many queries on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute many queries: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="begin_transaction",
        retry_config=DATABASE_TRANSACTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        if self.in_transaction:
            return DatabaseResult.error_result(TransactionError("Transaction already in progress"))

        try:
            # Acquire a connection from the pool
            self.transaction_conn = await self.pool.acquire()
            self.transaction_cursor = await self.transaction_conn.cursor()

            # Begin the transaction
            await self.transaction_cursor.execute("START TRANSACTION")

            self.in_transaction = True
            return DatabaseResult.success_result()
        except Exception as e:
            # Clean up
            if self.transaction_conn:
                self.pool.release(self.transaction_conn)
                self.transaction_conn = None
                self.transaction_cursor = None

            logger.error(f"Failed to begin transaction on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to begin transaction: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="commit_transaction",
        retry_config=DATABASE_TRANSACTION_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))

        try:
            # Commit the transaction
            await self.transaction_conn.commit()

            # Clean up
            self.pool.release(self.transaction_conn)
            self.transaction_conn = None
            self.transaction_cursor = None

            self.in_transaction = False
            return DatabaseResult.success_result()
        except Exception as e:
            # Try to roll back
            try:
                await self.transaction_conn.rollback()
            except Exception:
                pass

            # Clean up
            if self.transaction_conn:
                self.pool.release(self.transaction_conn)
                self.transaction_conn = None
                self.transaction_cursor = None

            self.in_transaction = False
            logger.error(f"Failed to commit transaction on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to commit transaction: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="rollback_transaction",
        retry_config=DATABASE_TRANSACTION_RETRY_CONFIG,
        timeout_seconds=10.0,
        timeout_strategy=TimeoutStrategy.FIXED,
        use_circuit_breaker=False,
        use_bulkhead=False
    )
    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction with resilience patterns."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))

        try:
            # Rollback the transaction
            await self.transaction_conn.rollback()

            # Clean up
            self.pool.release(self.transaction_conn)
            self.transaction_conn = None
            self.transaction_cursor = None

            self.in_transaction = False
            return DatabaseResult.success_result()
        except Exception as e:
            # Clean up
            if self.transaction_conn:
                self.pool.release(self.transaction_conn)
                self.transaction_conn = None
                self.transaction_cursor = None

            self.in_transaction = False
            logger.error(f"Failed to rollback transaction on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to rollback transaction: {e}"))

    async def execute_prepared_statement(
        self,
        statement_name: str,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> DatabaseResult:
        """Execute a prepared statement on the MySQL database.

        Args:
            statement_name: The name of the prepared statement
            query: The SQL query to prepare
            params: The parameters for the query

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        self.mysql_metrics["prepared_statements"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Convert named parameters to MySQL format (%s)
                    query, params_list = self._convert_params_for_mysql(query, params or {})

                    # Prepare the statement
                    await cursor.execute(f"PREPARE {statement_name} FROM '{query}'")

                    # Create the EXECUTE statement with parameters
                    param_placeholders = ", ".join(["?"] * len(params_list))
                    execute_query = f"EXECUTE {statement_name} USING {param_placeholders}"

                    # Execute the prepared statement
                    await cursor.execute(execute_query, params_list)

                    # Deallocate the prepared statement
                    await cursor.execute(f"DEALLOCATE PREPARE {statement_name}")

                    # Commit if not in a transaction
                    if not self.in_transaction and not self.autocommit:
                        await conn.commit()

                    # Get the result
                    result = DatabaseResult.success_result(
                        affected_rows=cursor.rowcount,
                        last_insert_id=cursor.lastrowid
                    )
            finally:
                # Release the connection
                await self.pool.release(conn)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow prepared statement detected: {statement_name} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to execute prepared statement on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute prepared statement: {e}"))

    async def get_server_info(self) -> DatabaseResult:
        """Get information about the MySQL server.

        Returns:
            The server information
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Get server variables
                    await cursor.execute("SHOW VARIABLES")
                    variables = await cursor.fetchall()

                    # Get server status
                    await cursor.execute("SHOW STATUS")
                    status = await cursor.fetchall()

                    # Get server version
                    await cursor.execute("SELECT VERSION() AS version")
                    version = await cursor.fetchone()

                    # Get server information
                    server_info = {
                        "version": version["version"],
                        "variables": {row["Variable_name"]: row["Value"] for row in variables},
                        "status": {row["Variable_name"]: row["Value"] for row in status}
                    }

                    return DatabaseResult.success_result(data=server_info)
        except Exception as e:
            logger.error(f"Failed to get server info from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to get server info: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="call_procedure",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=60.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def call_procedure(
        self,
        procedure_name: str,
        params: Optional[List[Any]] = None,
        fetch_results: bool = True
    ) -> DatabaseResult:
        """Call a stored procedure on the MySQL database.

        Args:
            procedure_name: The name of the stored procedure
            params: The parameters for the procedure
            fetch_results: Whether to fetch the results

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        self.mysql_metrics["stored_procedures"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Build the CALL statement
                    params = params or []
                    param_placeholders = ", ".join(["?" if p is not None else "NULL" for p in params])
                    call_query = f"CALL {procedure_name}({param_placeholders})"

                    # Execute the CALL statement
                    await cursor.execute(call_query, [p for p in params if p is not None])

                    # Fetch the results if requested
                    if fetch_results:
                        rows = await cursor.fetchall()

                        # Check if there are more result sets
                        more_results = True
                        all_results = [rows]

                        while more_results:
                            try:
                                more_results = await cursor.nextset()
                                if more_results:
                                    rows = await cursor.fetchall()
                                    all_results.append(rows)
                            except Exception:
                                more_results = False

                        # Return the results
                        if len(all_results) == 1:
                            result = DatabaseResult.success_result(data=all_results[0])
                        else:
                            result = DatabaseResult.success_result(data=all_results)
                    else:
                        # Return success without data
                        result = DatabaseResult.success_result(affected_rows=cursor.rowcount)

                    # Commit if not in a transaction
                    if not self.in_transaction and not self.autocommit:
                        await conn.commit()

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow stored procedure detected: {procedure_name} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to call stored procedure on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to call stored procedure: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="call_function",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def call_function(
        self,
        function_name: str,
        params: Optional[List[Any]] = None
    ) -> DatabaseResult:
        """Call a user-defined function on the MySQL database.

        Args:
            function_name: The name of the user-defined function
            params: The parameters for the function

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        self.mysql_metrics["user_defined_functions"] += 1
        start_time = time.time()

        try:
            # Acquire a connection from the pool
            conn, _ = await self.pool.acquire()
            try:
                async with conn.cursor() as cursor:
                    # Build the SELECT statement
                    params = params or []
                    param_placeholders = ", ".join(["?" if p is not None else "NULL" for p in params])
                    select_query = f"SELECT {function_name}({param_placeholders})"

                    # Execute the SELECT statement
                    await cursor.execute(select_query, [p for p in params if p is not None])

                    # Fetch the result
                    row = await cursor.fetchone()

                    # Return the result
                    if row and len(row) > 0:
                        result = DatabaseResult.success_result(data=row[0])
                    else:
                        result = DatabaseResult.success_result(data=None)
            finally:
                # Release the connection
                await self.pool.release(conn)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            # Check for slow queries
            if execution_time > 1.0:  # 1 second threshold for slow queries
                self.mysql_metrics["slow_queries"] += 1
                logger.warning(f"Slow user-defined function detected: {function_name} ({execution_time:.2f}s)")

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to call user-defined function on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to call user-defined function: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="create_trigger",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def create_trigger(
        self,
        trigger_name: str,
        table_name: str,
        timing: str,
        event: str,
        body: str
    ) -> DatabaseResult:
        """Create a trigger on the MySQL database.

        Args:
            trigger_name: The name of the trigger
            table_name: The name of the table
            timing: The timing of the trigger (BEFORE or AFTER)
            event: The event that triggers the trigger (INSERT, UPDATE, or DELETE)
            body: The body of the trigger

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Validate parameters
        timing = timing.upper()
        if timing not in ["BEFORE", "AFTER"]:
            return DatabaseResult.error_result(QueryError(f"Invalid trigger timing: {timing}"))

        event = event.upper()
        if event not in ["INSERT", "UPDATE", "DELETE"]:
            return DatabaseResult.error_result(QueryError(f"Invalid trigger event: {event}"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        self.mysql_metrics["triggers"] += 1
        start_time = time.time()

        try:
            # Build the CREATE TRIGGER statement
            create_trigger_query = f"""
            CREATE TRIGGER {trigger_name}
            {timing} {event} ON {table_name}
            FOR EACH ROW
            {body}
            """

            # Execute the CREATE TRIGGER statement
            result = await self.execute(create_trigger_query)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to create trigger on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to create trigger: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="drop_trigger",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def drop_trigger(
        self,
        trigger_name: str
    ) -> DatabaseResult:
        """Drop a trigger from the MySQL database.

        Args:
            trigger_name: The name of the trigger

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Build the DROP TRIGGER statement
            drop_trigger_query = f"DROP TRIGGER IF EXISTS {trigger_name}"

            # Execute the DROP TRIGGER statement
            result = await self.execute(drop_trigger_query)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to drop trigger from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to drop trigger: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="create_event",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def create_event(
        self,
        event_name: str,
        schedule: str,
        body: str,
        on_completion: str = "NOT PRESERVE",
        status: str = "ENABLED",
        comment: Optional[str] = None
    ) -> DatabaseResult:
        """Create an event on the MySQL database.

        Args:
            event_name: The name of the event
            schedule: The schedule of the event (e.g., "AT '2023-01-01 00:00:00'" or "EVERY 1 DAY")
            body: The body of the event
            on_completion: The on completion clause (PRESERVE or NOT PRESERVE)
            status: The status of the event (ENABLED, DISABLED, or SLAVESIDE_DISABLED)
            comment: A comment for the event

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Validate parameters
        on_completion = on_completion.upper()
        if on_completion not in ["PRESERVE", "NOT PRESERVE"]:
            return DatabaseResult.error_result(QueryError(f"Invalid on completion clause: {on_completion}"))

        status = status.upper()
        if status not in ["ENABLED", "DISABLED", "SLAVESIDE_DISABLED"]:
            return DatabaseResult.error_result(QueryError(f"Invalid event status: {status}"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        self.mysql_metrics["events"] += 1
        start_time = time.time()

        try:
            # Build the CREATE EVENT statement
            create_event_query = f"""
            CREATE EVENT {event_name}
            ON SCHEDULE {schedule}
            ON COMPLETION {on_completion}
            {status}
            {f"COMMENT '{comment}'" if comment else ""}
            DO
            {body}
            """

            # Execute the CREATE EVENT statement
            result = await self.execute(create_event_query)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to create event on MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to create event: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="drop_event",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def drop_event(
        self,
        event_name: str
    ) -> DatabaseResult:
        """Drop an event from the MySQL database.

        Args:
            event_name: The name of the event

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Build the DROP EVENT statement
            drop_event_query = f"DROP EVENT IF EXISTS {event_name}"

            # Execute the DROP EVENT statement
            result = await self.execute(drop_event_query)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to drop event from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to drop event: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="get_triggers",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get_triggers(
        self,
        database_name: Optional[str] = None,
        table_name: Optional[str] = None
    ) -> DatabaseResult:
        """Get triggers from the MySQL database.

        Args:
            database_name: The name of the database (optional)
            table_name: The name of the table (optional)

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Build the SHOW TRIGGERS statement
            query = "SHOW TRIGGERS"
            params = {}

            if database_name:
                query += " FROM :database_name"
                params["database_name"] = database_name

            if table_name:
                query += " LIKE :table_name"
                params["table_name"] = table_name

            # Execute the SHOW TRIGGERS statement
            result = await self.fetch_all(query, params)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to get triggers from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to get triggers: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="get_events",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get_events(
        self,
        database_name: Optional[str] = None,
        event_name: Optional[str] = None
    ) -> DatabaseResult:
        """Get events from the MySQL database.

        Args:
            database_name: The name of the database (optional)
            event_name: The name of the event (optional)

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Build the SHOW EVENTS statement
            query = "SHOW EVENTS"
            params = {}

            if database_name:
                query += " FROM :database_name"
                params["database_name"] = database_name

            if event_name:
                query += " LIKE :event_name"
                params["event_name"] = event_name

            # Execute the SHOW EVENTS statement
            result = await self.fetch_all(query, params)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to get events from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to get events: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="get_procedures",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get_procedures(
        self,
        database_name: Optional[str] = None,
        procedure_name: Optional[str] = None
    ) -> DatabaseResult:
        """Get stored procedures from the MySQL database.

        Args:
            database_name: The name of the database (optional)
            procedure_name: The name of the procedure (optional)

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Build the SHOW PROCEDURE STATUS statement
            query = "SHOW PROCEDURE STATUS"
            params = {}

            if database_name or procedure_name:
                query += " WHERE"

                if database_name:
                    query += " Db = :database_name"
                    params["database_name"] = database_name

                    if procedure_name:
                        query += " AND"

                if procedure_name:
                    query += " Name = :procedure_name"
                    params["procedure_name"] = procedure_name

            # Execute the SHOW PROCEDURE STATUS statement
            result = await self.fetch_all(query, params)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to get stored procedures from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to get stored procedures: {e}"))

    @with_database_resilience(
        database="mysql",
        operation="get_functions",
        retry_config=DATABASE_QUERY_RETRY_CONFIG,
        timeout_seconds=30.0,
        timeout_strategy=TimeoutStrategy.ADAPTIVE,
        use_circuit_breaker=True,
        use_bulkhead=True
    )
    async def get_functions(
        self,
        database_name: Optional[str] = None,
        function_name: Optional[str] = None
    ) -> DatabaseResult:
        """Get user-defined functions from the MySQL database.

        Args:
            database_name: The name of the database (optional)
            function_name: The name of the function (optional)

        Returns:
            The result of the operation
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to MySQL database"))

        # Update metrics
        self.pool_metrics["total_queries"] += 1
        start_time = time.time()

        try:
            # Build the SHOW FUNCTION STATUS statement
            query = "SHOW FUNCTION STATUS"
            params = {}

            if database_name or function_name:
                query += " WHERE"

                if database_name:
                    query += " Db = :database_name"
                    params["database_name"] = database_name

                    if function_name:
                        query += " AND"

                if function_name:
                    query += " Name = :function_name"
                    params["function_name"] = function_name

            # Execute the SHOW FUNCTION STATUS statement
            result = await self.fetch_all(query, params)

            # Update metrics
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            return result
        except Exception as e:
            # Update metrics
            self.pool_metrics["failed_queries"] += 1
            execution_time = time.time() - start_time
            self.pool_metrics["total_execution_time"] += execution_time

            logger.error(f"Failed to get user-defined functions from MySQL database: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to get user-defined functions: {e}"))

    def _parse_mysql_connection_string(self) -> Dict[str, Any]:
        """Parse the MySQL connection string.

        Returns:
            Dictionary of connection parameters
        """
        # Parse the connection string
        if not self.connection_string.startswith("mysql"):
            raise ValueError(f"Invalid MySQL connection string: {self.connection_string}")

        # Remove the protocol prefix
        conn_str = self.connection_string.replace("mysql://", "")

        # Extract the credentials, host, port, and database
        credentials_host_port, database = conn_str.split("/", 1) if "/" in conn_str else (conn_str, "")

        if "@" in credentials_host_port:
            credentials, host_port = credentials_host_port.split("@", 1)
        else:
            credentials, host_port = "", credentials_host_port

        if ":" in credentials:
            user, password = credentials.split(":", 1)
        else:
            user, password = credentials, ""

        if ":" in host_port:
            host, port = host_port.split(":", 1)
            try:
                port = int(port)
            except ValueError:
                port = 3306
        else:
            host, port = host_port, 3306

        # Build the connection parameters
        conn_params = {
            "host": host,
            "port": port,
            "user": user,
            "db": database
        }

        # Add password if provided
        if password:
            conn_params["password"] = password

        return conn_params

    def _convert_params_for_mysql(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Convert named parameters to MySQL format (%s).

        Args:
            query: The SQL query with named parameters
            params: The parameters for the query

        Returns:
            Tuple of (converted query, list of parameter values)
        """
        # Find all named parameters in the query
        param_names = []
        converted_query = query

        # Replace :param with %s
        for param_name in params.keys():
            if f":{param_name}" in converted_query:
                param_names.append(param_name)
                converted_query = converted_query.replace(f":{param_name}", "%s")

        # Create a list of parameter values in the correct order
        param_values = [params[name] for name in param_names]

        return converted_query, param_values

    def _dict_to_tuple_for_mysql(self, query: str, params: Dict[str, Any]) -> List[Any]:
        """Convert a dictionary of parameters to a tuple for MySQL.

        Args:
            query: The SQL query with named parameters
            params: The parameters for the query

        Returns:
            List of parameter values in the correct order
        """
        # Convert the query and get the parameter values
        _, param_values = self._convert_params_for_mysql(query, params)

        return param_values

    async def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = await super().get_metrics()
        metrics["mysql_metrics"] = self.mysql_metrics
        return metrics
