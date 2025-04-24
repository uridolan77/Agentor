"""
MySQL user-defined function manager for the Agentor framework.

This module provides a specialized manager for MySQL user-defined functions.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import FunctionConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class FunctionMetadata:
    """Metadata for a user-defined function."""
    
    def __init__(
        self,
        name: str,
        schema: str,
        params: List[Dict[str, Any]],
        returns: Dict[str, Any],
        body: str,
        created: str,
        modified: str,
        definer: str,
        security_type: str,
        is_deterministic: bool,
        data_access: str,
        comment: str
    ):
        """Initialize the function metadata.
        
        Args:
            name: The name of the function
            schema: The schema of the function
            params: The parameters of the function
            returns: The return type of the function
            body: The body of the function
            created: The creation date of the function
            modified: The last modification date of the function
            definer: The definer of the function
            security_type: The security type of the function
            is_deterministic: Whether the function is deterministic
            data_access: The data access of the function
            comment: The comment of the function
        """
        self.name = name
        self.schema = schema
        self.params = params
        self.returns = returns
        self.body = body
        self.created = created
        self.modified = modified
        self.definer = definer
        self.security_type = security_type
        self.is_deterministic = is_deterministic
        self.data_access = data_access
        self.comment = comment
        self.cached_at = time.time()
    
    def is_expired(self, ttl: float) -> bool:
        """Check if the metadata is expired.
        
        Args:
            ttl: The TTL in seconds
            
        Returns:
            True if the metadata is expired, False otherwise
        """
        return time.time() > self.cached_at + ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the metadata to a dictionary.
        
        Returns:
            Dictionary of metadata
        """
        return {
            "name": self.name,
            "schema": self.schema,
            "params": self.params,
            "returns": self.returns,
            "body": self.body,
            "created": self.created,
            "modified": self.modified,
            "definer": self.definer,
            "security_type": self.security_type,
            "is_deterministic": self.is_deterministic,
            "data_access": self.data_access,
            "comment": self.comment,
            "cached_at": self.cached_at
        }


class MySqlFunctionManager:
    """MySQL user-defined function manager with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: FunctionConfig,
        connection_func: Optional[Callable[[], T]] = None
    ):
        """Initialize the MySQL user-defined function manager.
        
        Args:
            name: The name of the manager
            config: The function configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Function metadata cache
        self.metadata_cache: Dict[str, FunctionMetadata] = {}
        
        # Manager lock
        self.manager_lock = asyncio.Lock()
        
        # Manager metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "max_execution_time": 0.0,
            "min_execution_time": float('inf'),
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the user-defined function manager."""
        logger.info(f"Initialized MySQL user-defined function manager {self.name}")
    
    async def close(self) -> None:
        """Close the user-defined function manager."""
        # Clear the metadata cache
        async with self.manager_lock:
            self.metadata_cache.clear()
        
        logger.info(f"Closed MySQL user-defined function manager {self.name}")
    
    async def call_function(
        self,
        function_name: str,
        params: Optional[List[Any]] = None,
        schema: Optional[str] = None
    ) -> Tuple[bool, Optional[Any], Optional[Exception]]:
        """Call a user-defined function.
        
        Args:
            function_name: The name of the function
            params: The parameters for the function
            schema: The schema of the function, or None for the default schema
            
        Returns:
            Tuple of (success, result, error)
        """
        # Update metrics
        self.metrics["total_calls"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            self.metrics["failed_calls"] += 1
            return False, None, error
        
        # Get the function metadata
        metadata = await self.get_function_metadata(function_name, schema)
        
        # Execute the function
        start_time = time.time()
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the SELECT statement
            params = params or []
            param_placeholders = ", ".join(["?" if p is not None else "NULL" for p in params])
            full_function_name = f"{schema + '.' if schema else ''}{function_name}"
            select_query = f"SELECT {full_function_name}({param_placeholders})"
            
            # Log the call if configured
            if self.config.log_calls:
                logger.info(f"Calling function: {select_query} with params: {params}")
            
            # Execute the function
            async with conn.cursor() as cursor:
                # Execute the SELECT statement
                await cursor.execute(select_query, [p for p in params if p is not None])
                
                # Fetch the result
                row = await cursor.fetchone()
                
                # Get the result
                result = row[0] if row and len(row) > 0 else None
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update metrics
            self.metrics["successful_calls"] += 1
            self.metrics["total_execution_time"] += execution_time
            self.metrics["avg_execution_time"] = self.metrics["total_execution_time"] / self.metrics["successful_calls"]
            self.metrics["max_execution_time"] = max(self.metrics["max_execution_time"], execution_time)
            self.metrics["min_execution_time"] = min(self.metrics["min_execution_time"], execution_time)
            
            # Log the result if configured
            if self.config.log_results:
                logger.info(f"Function result: {result}")
            
            # Return the result
            return True, result, None
        except Exception as e:
            logger.error(f"Error calling function {function_name}: {e}")
            
            # Update metrics
            self.metrics["failed_calls"] += 1
            
            # Return the error
            return False, None, e
    
    async def create_function(
        self,
        function_name: str,
        params: List[Dict[str, Any]],
        returns: Dict[str, Any],
        body: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None,
        is_deterministic: bool = False,
        data_access: str = "CONTAINS SQL",
        security_type: str = "DEFINER",
        comment: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Create a user-defined function.
        
        Args:
            function_name: The name of the function
            params: The parameters for the function
            returns: The return type of the function
            body: The body of the function
            schema: The schema of the function, or None for the default schema
            definer: The definer of the function, or None for the current user
            is_deterministic: Whether the function is deterministic
            data_access: The data access of the function (CONTAINS SQL, NO SQL, READS SQL DATA, MODIFIES SQL DATA)
            security_type: The security type of the function (DEFINER or INVOKER)
            comment: The comment of the function, or None for no comment
            
        Returns:
            Tuple of (success, error)
        """
        # Check if creation is allowed
        if not self.config.allow_create:
            error = Exception("Creating functions is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the CREATE FUNCTION statement
            full_function_name = f"{schema + '.' if schema else ''}{function_name}"
            
            # Build the parameter list
            param_list = []
            for param in params:
                param_name = param.get("name")
                param_type = param.get("type")
                
                if not param_name or not param_type:
                    error = Exception("Parameter name and type are required")
                    return False, error
                
                param_list.append(f"{param_name} {param_type}")
            
            param_string = ", ".join(param_list)
            
            # Build the return type
            return_type = returns.get("type")
            if not return_type:
                error = Exception("Return type is required")
                return False, error
            
            # Build the definer clause
            definer_clause = f"DEFINER = {definer}" if definer else ""
            
            # Build the deterministic clause
            deterministic_clause = "DETERMINISTIC" if is_deterministic else "NOT DETERMINISTIC"
            
            # Build the comment clause
            comment_clause = f"COMMENT '{comment}'" if comment else ""
            
            # Build the CREATE FUNCTION statement
            create_query = f"""
                CREATE {definer_clause} FUNCTION {full_function_name}({param_string})
                RETURNS {return_type}
                {deterministic_clause}
                {data_access}
                SQL SECURITY {security_type}
                {comment_clause}
                {body}
            """
            
            # Execute the CREATE FUNCTION statement
            async with conn.cursor() as cursor:
                await cursor.execute(create_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{function_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            return True, None
        except Exception as e:
            logger.error(f"Error creating function {function_name}: {e}")
            return False, e
    
    async def alter_function(
        self,
        function_name: str,
        params: List[Dict[str, Any]],
        returns: Dict[str, Any],
        body: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None,
        is_deterministic: bool = False,
        data_access: str = "CONTAINS SQL",
        security_type: str = "DEFINER",
        comment: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Alter a user-defined function.
        
        Args:
            function_name: The name of the function
            params: The parameters for the function
            returns: The return type of the function
            body: The body of the function
            schema: The schema of the function, or None for the default schema
            definer: The definer of the function, or None for the current user
            is_deterministic: Whether the function is deterministic
            data_access: The data access of the function (CONTAINS SQL, NO SQL, READS SQL DATA, MODIFIES SQL DATA)
            security_type: The security type of the function (DEFINER or INVOKER)
            comment: The comment of the function, or None for no comment
            
        Returns:
            Tuple of (success, error)
        """
        # Check if alteration is allowed
        if not self.config.allow_alter:
            error = Exception("Altering functions is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Drop the function first
            full_function_name = f"{schema + '.' if schema else ''}{function_name}"
            drop_query = f"DROP FUNCTION IF EXISTS {full_function_name}"
            
            async with conn.cursor() as cursor:
                await cursor.execute(drop_query)
            
            # Create the function again
            return await self.create_function(
                function_name=function_name,
                params=params,
                returns=returns,
                body=body,
                schema=schema,
                definer=definer,
                is_deterministic=is_deterministic,
                data_access=data_access,
                security_type=security_type,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Error altering function {function_name}: {e}")
            return False, e
    
    async def drop_function(
        self,
        function_name: str,
        schema: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Drop a user-defined function.
        
        Args:
            function_name: The name of the function
            schema: The schema of the function, or None for the default schema
            
        Returns:
            Tuple of (success, error)
        """
        # Check if dropping is allowed
        if not self.config.allow_drop:
            error = Exception("Dropping functions is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the DROP FUNCTION statement
            full_function_name = f"{schema + '.' if schema else ''}{function_name}"
            drop_query = f"DROP FUNCTION IF EXISTS {full_function_name}"
            
            # Execute the DROP FUNCTION statement
            async with conn.cursor() as cursor:
                await cursor.execute(drop_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{function_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            return True, None
        except Exception as e:
            logger.error(f"Error dropping function {function_name}: {e}")
            return False, e
    
    async def get_function_metadata(
        self,
        function_name: str,
        schema: Optional[str] = None
    ) -> Optional[FunctionMetadata]:
        """Get metadata for a user-defined function.
        
        Args:
            function_name: The name of the function
            schema: The schema of the function, or None for the default schema
            
        Returns:
            The function metadata, or None if not found
        """
        # Check if we have a connection function
        if not self.connection_func:
            return None
        
        # Check if metadata caching is enabled
        if self.config.cache_metadata:
            # Check if the metadata is in the cache
            cache_key = f"{schema or ''}.{function_name}".lower()
            
            async with self.manager_lock:
                if cache_key in self.metadata_cache:
                    metadata = self.metadata_cache[cache_key]
                    
                    # Check if the metadata is expired
                    if not metadata.is_expired(self.config.cache_ttl):
                        # Update metrics
                        self.metrics["cache_hits"] += 1
                        
                        return metadata
        
        # Update metrics
        self.metrics["cache_misses"] += 1
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the function metadata
            async with conn.cursor() as cursor:
                # Get the function from information_schema
                query = """
                    SELECT
                        ROUTINE_SCHEMA,
                        ROUTINE_NAME,
                        ROUTINE_TYPE,
                        DTD_IDENTIFIER,
                        ROUTINE_BODY,
                        ROUTINE_DEFINITION,
                        PARAMETER_STYLE,
                        IS_DETERMINISTIC,
                        SQL_DATA_ACCESS,
                        SECURITY_TYPE,
                        CREATED,
                        LAST_ALTERED,
                        SQL_MODE,
                        ROUTINE_COMMENT,
                        DEFINER
                    FROM
                        information_schema.ROUTINES
                    WHERE
                        ROUTINE_TYPE = 'FUNCTION'
                        AND ROUTINE_NAME = %s
                """
                
                params = [function_name]
                
                if schema:
                    query += " AND ROUTINE_SCHEMA = %s"
                    params.append(schema)
                
                await cursor.execute(query, params)
                result = await cursor.fetchone()
                
                if not result:
                    return None
                
                # Get the function parameters
                param_query = """
                    SELECT
                        PARAMETER_NAME,
                        PARAMETER_MODE,
                        DATA_TYPE,
                        CHARACTER_MAXIMUM_LENGTH,
                        NUMERIC_PRECISION,
                        NUMERIC_SCALE
                    FROM
                        information_schema.PARAMETERS
                    WHERE
                        SPECIFIC_NAME = %s
                        AND SPECIFIC_SCHEMA = %s
                        AND PARAMETER_NAME IS NOT NULL
                    ORDER BY
                        ORDINAL_POSITION
                """
                
                await cursor.execute(param_query, [function_name, result[0]])
                param_results = await cursor.fetchall()
                
                # Parse the parameters
                params = []
                for param in param_results:
                    param_info = {
                        "name": param[0],
                        "mode": param[1],
                        "type": param[2],
                        "max_length": param[3],
                        "precision": param[4],
                        "scale": param[5]
                    }
                    params.append(param_info)
                
                # Get the return type
                return_query = """
                    SELECT
                        DATA_TYPE,
                        CHARACTER_MAXIMUM_LENGTH,
                        NUMERIC_PRECISION,
                        NUMERIC_SCALE
                    FROM
                        information_schema.PARAMETERS
                    WHERE
                        SPECIFIC_NAME = %s
                        AND SPECIFIC_SCHEMA = %s
                        AND PARAMETER_NAME IS NULL
                """
                
                await cursor.execute(return_query, [function_name, result[0]])
                return_result = await cursor.fetchone()
                
                # Parse the return type
                returns = {
                    "type": return_result[0],
                    "max_length": return_result[1],
                    "precision": return_result[2],
                    "scale": return_result[3]
                }
                
                # Create the metadata
                metadata = FunctionMetadata(
                    name=result[1],
                    schema=result[0],
                    params=params,
                    returns=returns,
                    body=result[5],
                    created=str(result[10]),
                    modified=str(result[11]),
                    definer=result[14],
                    security_type=result[9],
                    is_deterministic=result[7] == "YES",
                    data_access=result[8],
                    comment=result[13]
                )
                
                # Cache the metadata if enabled
                if self.config.cache_metadata:
                    async with self.manager_lock:
                        self.metadata_cache[cache_key] = metadata
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting function metadata for {function_name}: {e}")
            return None
    
    async def list_functions(
        self,
        schema: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List user-defined functions.
        
        Args:
            schema: The schema to list functions from, or None for all schemas
            pattern: A pattern to filter function names, or None for all functions
            
        Returns:
            List of function metadata
        """
        # Check if we have a connection function
        if not self.connection_func:
            return []
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the functions
            async with conn.cursor() as cursor:
                # Build the query
                query = """
                    SELECT
                        ROUTINE_SCHEMA,
                        ROUTINE_NAME,
                        ROUTINE_TYPE,
                        DTD_IDENTIFIER,
                        ROUTINE_BODY,
                        PARAMETER_STYLE,
                        IS_DETERMINISTIC,
                        SQL_DATA_ACCESS,
                        SECURITY_TYPE,
                        CREATED,
                        LAST_ALTERED,
                        SQL_MODE,
                        ROUTINE_COMMENT,
                        DEFINER
                    FROM
                        information_schema.ROUTINES
                    WHERE
                        ROUTINE_TYPE = 'FUNCTION'
                """
                
                params = []
                
                if schema:
                    query += " AND ROUTINE_SCHEMA = %s"
                    params.append(schema)
                
                if pattern:
                    query += " AND ROUTINE_NAME LIKE %s"
                    params.append(pattern)
                
                query += " ORDER BY ROUTINE_SCHEMA, ROUTINE_NAME"
                
                await cursor.execute(query, params)
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                functions = []
                for result in results:
                    function = {
                        "schema": result[0],
                        "name": result[1],
                        "type": result[2],
                        "data_type": result[3],
                        "body_type": result[4],
                        "parameter_style": result[5],
                        "is_deterministic": result[6],
                        "data_access": result[7],
                        "security_type": result[8],
                        "created": str(result[9]),
                        "modified": str(result[10]),
                        "sql_mode": result[11],
                        "comment": result[12],
                        "definer": result[13]
                    }
                    functions.append(function)
                
                return functions
        except Exception as e:
            logger.error(f"Error listing functions: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get user-defined function manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
