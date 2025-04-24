"""
MySQL trigger manager for the Agentor framework.

This module provides a specialized manager for MySQL triggers.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import TriggerConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class TriggerMetadata:
    """Metadata for a trigger."""
    
    def __init__(
        self,
        name: str,
        schema: str,
        table: str,
        event: str,
        timing: str,
        body: str,
        created: str,
        definer: str,
        character_set: str,
        collation: str,
        sql_mode: str
    ):
        """Initialize the trigger metadata.
        
        Args:
            name: The name of the trigger
            schema: The schema of the trigger
            table: The table of the trigger
            event: The event of the trigger (INSERT, UPDATE, DELETE)
            timing: The timing of the trigger (BEFORE, AFTER)
            body: The body of the trigger
            created: The creation date of the trigger
            definer: The definer of the trigger
            character_set: The character set of the trigger
            collation: The collation of the trigger
            sql_mode: The SQL mode of the trigger
        """
        self.name = name
        self.schema = schema
        self.table = table
        self.event = event
        self.timing = timing
        self.body = body
        self.created = created
        self.definer = definer
        self.character_set = character_set
        self.collation = collation
        self.sql_mode = sql_mode
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
            "table": self.table,
            "event": self.event,
            "timing": self.timing,
            "body": self.body,
            "created": self.created,
            "definer": self.definer,
            "character_set": self.character_set,
            "collation": self.collation,
            "sql_mode": self.sql_mode,
            "cached_at": self.cached_at
        }


class MySqlTriggerManager:
    """MySQL trigger manager with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: TriggerConfig,
        connection_func: Optional[Callable[[], T]] = None
    ):
        """Initialize the MySQL trigger manager.
        
        Args:
            name: The name of the manager
            config: The trigger configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Trigger metadata cache
        self.metadata_cache: Dict[str, TriggerMetadata] = {}
        
        # Manager lock
        self.manager_lock = asyncio.Lock()
        
        # Manager metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the trigger manager."""
        logger.info(f"Initialized MySQL trigger manager {self.name}")
    
    async def close(self) -> None:
        """Close the trigger manager."""
        # Clear the metadata cache
        async with self.manager_lock:
            self.metadata_cache.clear()
        
        logger.info(f"Closed MySQL trigger manager {self.name}")
    
    async def create_trigger(
        self,
        trigger_name: str,
        table_name: str,
        event: str,
        timing: str,
        body: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Create a trigger.
        
        Args:
            trigger_name: The name of the trigger
            table_name: The name of the table
            event: The event (INSERT, UPDATE, DELETE)
            timing: The timing (BEFORE, AFTER)
            body: The body of the trigger
            schema: The schema of the trigger, or None for the default schema
            definer: The definer of the trigger, or None for the current user
            
        Returns:
            Tuple of (success, error)
        """
        # Check if creation is allowed
        if not self.config.allow_create:
            error = Exception("Creating triggers is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["last_activity"] = time.time()
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the CREATE TRIGGER statement
            full_trigger_name = f"{schema + '.' if schema else ''}{trigger_name}"
            full_table_name = f"{schema + '.' if schema else ''}{table_name}"
            
            # Validate the event
            event = event.upper()
            if event not in ["INSERT", "UPDATE", "DELETE"]:
                error = Exception(f"Invalid event: {event}")
                self.metrics["failed_operations"] += 1
                return False, error
            
            # Validate the timing
            timing = timing.upper()
            if timing not in ["BEFORE", "AFTER"]:
                error = Exception(f"Invalid timing: {timing}")
                self.metrics["failed_operations"] += 1
                return False, error
            
            # Build the definer clause
            definer_clause = f"DEFINER = {definer}" if definer else ""
            
            # Build the CREATE TRIGGER statement
            create_query = f"""
                CREATE {definer_clause} TRIGGER {full_trigger_name}
                {timing} {event} ON {full_table_name}
                FOR EACH ROW
                {body}
            """
            
            # Log the operation if configured
            if self.config.log_operations:
                logger.info(f"Creating trigger: {create_query}")
            
            # Execute the CREATE TRIGGER statement
            async with conn.cursor() as cursor:
                await cursor.execute(create_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{trigger_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            # Update metrics
            self.metrics["successful_operations"] += 1
            
            return True, None
        except Exception as e:
            logger.error(f"Error creating trigger {trigger_name}: {e}")
            
            # Update metrics
            self.metrics["failed_operations"] += 1
            
            return False, e
    
    async def drop_trigger(
        self,
        trigger_name: str,
        schema: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Drop a trigger.
        
        Args:
            trigger_name: The name of the trigger
            schema: The schema of the trigger, or None for the default schema
            
        Returns:
            Tuple of (success, error)
        """
        # Check if dropping is allowed
        if not self.config.allow_drop:
            error = Exception("Dropping triggers is not allowed")
            return False, error
        
        # Check if we have a connection function
        if not self.connection_func:
            error = Exception("No connection function provided")
            return False, error
        
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["last_activity"] = time.time()
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Build the DROP TRIGGER statement
            full_trigger_name = f"{schema + '.' if schema else ''}{trigger_name}"
            drop_query = f"DROP TRIGGER IF EXISTS {full_trigger_name}"
            
            # Log the operation if configured
            if self.config.log_operations:
                logger.info(f"Dropping trigger: {drop_query}")
            
            # Execute the DROP TRIGGER statement
            async with conn.cursor() as cursor:
                await cursor.execute(drop_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{trigger_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            # Update metrics
            self.metrics["successful_operations"] += 1
            
            return True, None
        except Exception as e:
            logger.error(f"Error dropping trigger {trigger_name}: {e}")
            
            # Update metrics
            self.metrics["failed_operations"] += 1
            
            return False, e
    
    async def get_trigger_metadata(
        self,
        trigger_name: str,
        schema: Optional[str] = None
    ) -> Optional[TriggerMetadata]:
        """Get metadata for a trigger.
        
        Args:
            trigger_name: The name of the trigger
            schema: The schema of the trigger, or None for the default schema
            
        Returns:
            The trigger metadata, or None if not found
        """
        # Check if we have a connection function
        if not self.connection_func:
            return None
        
        # Check if metadata caching is enabled
        if self.config.cache_metadata:
            # Check if the metadata is in the cache
            cache_key = f"{schema or ''}.{trigger_name}".lower()
            
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
            
            # Get the trigger metadata
            async with conn.cursor() as cursor:
                # Get the trigger from information_schema
                query = """
                    SELECT
                        TRIGGER_SCHEMA,
                        TRIGGER_NAME,
                        EVENT_OBJECT_TABLE,
                        ACTION_TIMING,
                        EVENT_MANIPULATION,
                        ACTION_STATEMENT,
                        CREATED,
                        DEFINER,
                        CHARACTER_SET_CLIENT,
                        COLLATION_CONNECTION,
                        SQL_MODE
                    FROM
                        information_schema.TRIGGERS
                    WHERE
                        TRIGGER_NAME = %s
                """
                
                params = [trigger_name]
                
                if schema:
                    query += " AND TRIGGER_SCHEMA = %s"
                    params.append(schema)
                
                await cursor.execute(query, params)
                result = await cursor.fetchone()
                
                if not result:
                    return None
                
                # Create the metadata
                metadata = TriggerMetadata(
                    name=result[1],
                    schema=result[0],
                    table=result[2],
                    timing=result[3],
                    event=result[4],
                    body=result[5],
                    created=str(result[6]),
                    definer=result[7],
                    character_set=result[8],
                    collation=result[9],
                    sql_mode=result[10]
                )
                
                # Cache the metadata if enabled
                if self.config.cache_metadata:
                    async with self.manager_lock:
                        self.metadata_cache[cache_key] = metadata
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting trigger metadata for {trigger_name}: {e}")
            return None
    
    async def list_triggers(
        self,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List triggers.
        
        Args:
            schema: The schema to list triggers from, or None for all schemas
            table: The table to list triggers for, or None for all tables
            pattern: A pattern to filter trigger names, or None for all triggers
            
        Returns:
            List of trigger metadata
        """
        # Check if we have a connection function
        if not self.connection_func:
            return []
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the triggers
            async with conn.cursor() as cursor:
                # Build the query
                query = """
                    SELECT
                        TRIGGER_SCHEMA,
                        TRIGGER_NAME,
                        EVENT_OBJECT_TABLE,
                        ACTION_TIMING,
                        EVENT_MANIPULATION,
                        ACTION_STATEMENT,
                        CREATED,
                        DEFINER,
                        CHARACTER_SET_CLIENT,
                        COLLATION_CONNECTION,
                        SQL_MODE
                    FROM
                        information_schema.TRIGGERS
                    WHERE
                        1=1
                """
                
                params = []
                
                if schema:
                    query += " AND TRIGGER_SCHEMA = %s"
                    params.append(schema)
                
                if table:
                    query += " AND EVENT_OBJECT_TABLE = %s"
                    params.append(table)
                
                if pattern:
                    query += " AND TRIGGER_NAME LIKE %s"
                    params.append(pattern)
                
                query += " ORDER BY TRIGGER_SCHEMA, EVENT_OBJECT_TABLE, TRIGGER_NAME"
                
                await cursor.execute(query, params)
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                triggers = []
                for result in results:
                    trigger = {
                        "schema": result[0],
                        "name": result[1],
                        "table": result[2],
                        "timing": result[3],
                        "event": result[4],
                        "body": result[5],
                        "created": str(result[6]),
                        "definer": result[7],
                        "character_set": result[8],
                        "collation": result[9],
                        "sql_mode": result[10]
                    }
                    triggers.append(trigger)
                
                return triggers
        except Exception as e:
            logger.error(f"Error listing triggers: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get trigger manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
