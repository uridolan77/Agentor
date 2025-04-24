"""
MySQL stored procedure manager for the Agentor framework.

This module provides a specialized manager for MySQL stored procedures.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import ProcedureConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class ProcedureMetadata:
    """Metadata for a stored procedure."""
    
    def __init__(
        self,
        name: str,
        schema: str,
        params: List[Dict[str, Any]],
        returns: Optional[Dict[str, Any]],
        body: str,
        created: str,
        modified: str,
        definer: str,
        security_type: str,
        comment: str
    ):
        """Initialize the procedure metadata.
        
        Args:
            name: The name of the procedure
            schema: The schema of the procedure
            params: The parameters of the procedure
            returns: The return type of the procedure, if any
            body: The body of the procedure
            created: The creation date of the procedure
            modified: The last modification date of the procedure
            definer: The definer of the procedure
            security_type: The security type of the procedure
            comment: The comment of the procedure
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
            "comment": self.comment,
            "cached_at": self.cached_at
        }


class MySqlProcedureManager:
    """MySQL stored procedure manager with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: ProcedureConfig,
        connection_func: Optional[Callable[[], T]] = None
    ):
        """Initialize the MySQL stored procedure manager.
        
        Args:
            name: The name of the manager
            config: The procedure configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Procedure metadata cache
        self.metadata_cache: Dict[str, ProcedureMetadata] = {}
        
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
            "total_rows": 0,
            "avg_rows": 0.0,
            "max_rows": 0,
            "min_rows": float('inf'),
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the stored procedure manager."""
        logger.info(f"Initialized MySQL stored procedure manager {self.name}")
    
    async def close(self) -> None:
        """Close the stored procedure manager."""
        # Clear the metadata cache
        async with self.manager_lock:
            self.metadata_cache.clear()
        
        logger.info(f"Closed MySQL stored procedure manager {self.name}")
    
    async def call_procedure(
        self,
        procedure_name: str,
        params: Optional[List[Any]] = None,
        schema: Optional[str] = None
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[Exception]]:
        """Call a stored procedure.
        
        Args:
            procedure_name: The name of the procedure
            params: The parameters for the procedure
            schema: The schema of the procedure, or None for the default schema
            
        Returns:
            Tuple of (success, results, error)
        """
        # Update metrics
        self.metrics["total_calls"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            self.metrics["failed_calls"] += 1
            return False, None, error
        
        # Get the procedure metadata
        metadata = await self.get_procedure_metadata(procedure_name, schema)
        
        # Execute the procedure
        start_time = time.time()
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the CALL statement
            params = params or []
            param_placeholders = ", ".join(["?" if p is not None else "NULL" for p in params])
            full_procedure_name = f"{schema + '.' if schema else ''}{procedure_name}"
            call_query = f"CALL {full_procedure_name}({param_placeholders})"
            
            # Log the call if configured
            if self.config.log_calls:
                logger.info(f"Calling procedure: {call_query} with params: {params}")
            
            # Execute the procedure
            async with conn.cursor() as cursor:
                # Execute the CALL statement
                await cursor.execute(call_query, [p for p in params if p is not None])
                
                # Fetch all results
                all_results = []
                
                # MySQL procedures can return multiple result sets
                current_result = await cursor.fetchall()
                if current_result:
                    all_results.append(current_result)
                
                # Check for more results
                while await cursor.nextset():
                    current_result = await cursor.fetchall()
                    if current_result:
                        all_results.append(current_result)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update metrics
            self.metrics["successful_calls"] += 1
            self.metrics["total_execution_time"] += execution_time
            self.metrics["avg_execution_time"] = self.metrics["total_execution_time"] / self.metrics["successful_calls"]
            self.metrics["max_execution_time"] = max(self.metrics["max_execution_time"], execution_time)
            self.metrics["min_execution_time"] = min(self.metrics["min_execution_time"], execution_time)
            
            # Calculate total rows
            total_rows = sum(len(result) for result in all_results)
            self.metrics["total_rows"] += total_rows
            self.metrics["avg_rows"] = self.metrics["total_rows"] / self.metrics["successful_calls"]
            self.metrics["max_rows"] = max(self.metrics["max_rows"], total_rows)
            self.metrics["min_rows"] = min(self.metrics["min_rows"], total_rows)
            
            # Log the results if configured
            if self.config.log_results:
                logger.info(f"Procedure results: {all_results}")
            
            # Return the results
            if len(all_results) == 1:
                return True, all_results[0], None
            else:
                return True, all_results, None
        except Exception as e:
            logger.error(f"Error calling procedure {procedure_name}: {e}")
            
            # Update metrics
            self.metrics["failed_calls"] += 1
            
            # Return the error
            return False, None, e
    
    async def create_procedure(
        self,
        procedure_name: str,
        params: List[Dict[str, Any]],
        body: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None,
        security_type: str = "DEFINER",
        comment: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Create a stored procedure.
        
        Args:
            procedure_name: The name of the procedure
            params: The parameters for the procedure
            body: The body of the procedure
            schema: The schema of the procedure, or None for the default schema
            definer: The definer of the procedure, or None for the current user
            security_type: The security type of the procedure (DEFINER or INVOKER)
            comment: The comment of the procedure, or None for no comment
            
        Returns:
            Tuple of (success, error)
        """
        # Check if creation is allowed
        if not self.config.allow_create:
            error = Exception("Creating procedures is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the CREATE PROCEDURE statement
            full_procedure_name = f"{schema + '.' if schema else ''}{procedure_name}"
            
            # Build the parameter list
            param_list = []
            for param in params:
                param_mode = param.get("mode", "IN").upper()
                param_name = param.get("name")
                param_type = param.get("type")
                
                if not param_name or not param_type:
                    error = Exception("Parameter name and type are required")
                    return False, error
                
                param_list.append(f"{param_mode} {param_name} {param_type}")
            
            param_string = ", ".join(param_list)
            
            # Build the definer clause
            definer_clause = f"DEFINER = {definer}" if definer else ""
            
            # Build the comment clause
            comment_clause = f"COMMENT '{comment}'" if comment else ""
            
            # Build the CREATE PROCEDURE statement
            create_query = f"""
                CREATE {definer_clause} PROCEDURE {full_procedure_name}({param_string})
                SQL SECURITY {security_type}
                {comment_clause}
                {body}
            """
            
            # Execute the CREATE PROCEDURE statement
            async with conn.cursor() as cursor:
                await cursor.execute(create_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{procedure_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            return True, None
        except Exception as e:
            logger.error(f"Error creating procedure {procedure_name}: {e}")
            return False, e
    
    async def alter_procedure(
        self,
        procedure_name: str,
        params: List[Dict[str, Any]],
        body: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None,
        security_type: str = "DEFINER",
        comment: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Alter a stored procedure.
        
        Args:
            procedure_name: The name of the procedure
            params: The parameters for the procedure
            body: The body of the procedure
            schema: The schema of the procedure, or None for the default schema
            definer: The definer of the procedure, or None for the current user
            security_type: The security type of the procedure (DEFINER or INVOKER)
            comment: The comment of the procedure, or None for no comment
            
        Returns:
            Tuple of (success, error)
        """
        # Check if alteration is allowed
        if not self.config.allow_alter:
            error = Exception("Altering procedures is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Drop the procedure first
            full_procedure_name = f"{schema + '.' if schema else ''}{procedure_name}"
            drop_query = f"DROP PROCEDURE IF EXISTS {full_procedure_name}"
            
            async with conn.cursor() as cursor:
                await cursor.execute(drop_query)
            
            # Create the procedure again
            return await self.create_procedure(
                procedure_name=procedure_name,
                params=params,
                body=body,
                schema=schema,
                definer=definer,
                security_type=security_type,
                comment=comment
            )
        except Exception as e:
            logger.error(f"Error altering procedure {procedure_name}: {e}")
            return False, e
    
    async def drop_procedure(
        self,
        procedure_name: str,
        schema: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Drop a stored procedure.
        
        Args:
            procedure_name: The name of the procedure
            schema: The schema of the procedure, or None for the default schema
            
        Returns:
            Tuple of (success, error)
        """
        # Check if dropping is allowed
        if not self.config.allow_drop:
            error = Exception("Dropping procedures is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the DROP PROCEDURE statement
            full_procedure_name = f"{schema + '.' if schema else ''}{procedure_name}"
            drop_query = f"DROP PROCEDURE IF EXISTS {full_procedure_name}"
            
            # Execute the DROP PROCEDURE statement
            async with conn.cursor() as cursor:
                await cursor.execute(drop_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{procedure_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            return True, None
        except Exception as e:
            logger.error(f"Error dropping procedure {procedure_name}: {e}")
            return False, e
    
    async def get_procedure_metadata(
        self,
        procedure_name: str,
        schema: Optional[str] = None
    ) -> Optional[ProcedureMetadata]:
        """Get metadata for a stored procedure.
        
        Args:
            procedure_name: The name of the procedure
            schema: The schema of the procedure, or None for the default schema
            
        Returns:
            The procedure metadata, or None if not found
        """
        # Check if we have a connection function
        if not self.connection_func:
            return None
        
        # Check if metadata caching is enabled
        if self.config.cache_metadata:
            # Check if the metadata is in the cache
            cache_key = f"{schema or ''}.{procedure_name}".lower()
            
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
            
            # Get the procedure metadata
            async with conn.cursor() as cursor:
                # Get the procedure from information_schema
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
                        ROUTINE_TYPE = 'PROCEDURE'
                        AND ROUTINE_NAME = %s
                """
                
                params = [procedure_name]
                
                if schema:
                    query += " AND ROUTINE_SCHEMA = %s"
                    params.append(schema)
                
                await cursor.execute(query, params)
                result = await cursor.fetchone()
                
                if not result:
                    return None
                
                # Get the procedure parameters
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
                    ORDER BY
                        ORDINAL_POSITION
                """
                
                await cursor.execute(param_query, [procedure_name, result[0]])
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
                
                # Create the metadata
                metadata = ProcedureMetadata(
                    name=result[1],
                    schema=result[0],
                    params=params,
                    returns=None,  # Procedures don't return values
                    body=result[5],
                    created=str(result[10]),
                    modified=str(result[11]),
                    definer=result[14],
                    security_type=result[9],
                    comment=result[13]
                )
                
                # Cache the metadata if enabled
                if self.config.cache_metadata:
                    async with self.manager_lock:
                        self.metadata_cache[cache_key] = metadata
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting procedure metadata for {procedure_name}: {e}")
            return None
    
    async def list_procedures(
        self,
        schema: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List stored procedures.
        
        Args:
            schema: The schema to list procedures from, or None for all schemas
            pattern: A pattern to filter procedure names, or None for all procedures
            
        Returns:
            List of procedure metadata
        """
        # Check if we have a connection function
        if not self.connection_func:
            return []
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the procedures
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
                        ROUTINE_TYPE = 'PROCEDURE'
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
                procedures = []
                for result in results:
                    procedure = {
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
                    procedures.append(procedure)
                
                return procedures
        except Exception as e:
            logger.error(f"Error listing procedures: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get stored procedure manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
