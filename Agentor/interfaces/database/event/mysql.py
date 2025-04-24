"""
MySQL event scheduling manager for the Agentor framework.

This module provides a specialized manager for MySQL event scheduling.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import EventConfig

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class EventMetadata:
    """Metadata for an event."""
    
    def __init__(
        self,
        name: str,
        schema: str,
        body: str,
        definer: str,
        execute_at: Optional[str],
        interval_value: Optional[int],
        interval_field: Optional[str],
        starts: Optional[str],
        ends: Optional[str],
        status: str,
        on_completion: str,
        created: str,
        last_altered: str,
        last_executed: Optional[str],
        sql_mode: str,
        comment: str
    ):
        """Initialize the event metadata.
        
        Args:
            name: The name of the event
            schema: The schema of the event
            body: The body of the event
            definer: The definer of the event
            execute_at: The execution time of the event, or None for recurring events
            interval_value: The interval value of the event, or None for one-time events
            interval_field: The interval field of the event, or None for one-time events
            starts: The start time of the event, or None for immediate start
            ends: The end time of the event, or None for no end
            status: The status of the event (ENABLED, DISABLED, SLAVESIDE_DISABLED)
            on_completion: The on completion behavior of the event (PRESERVE, NOT PRESERVE)
            created: The creation date of the event
            last_altered: The last alteration date of the event
            last_executed: The last execution date of the event, or None if never executed
            sql_mode: The SQL mode of the event
            comment: The comment of the event
        """
        self.name = name
        self.schema = schema
        self.body = body
        self.definer = definer
        self.execute_at = execute_at
        self.interval_value = interval_value
        self.interval_field = interval_field
        self.starts = starts
        self.ends = ends
        self.status = status
        self.on_completion = on_completion
        self.created = created
        self.last_altered = last_altered
        self.last_executed = last_executed
        self.sql_mode = sql_mode
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
            "body": self.body,
            "definer": self.definer,
            "execute_at": self.execute_at,
            "interval_value": self.interval_value,
            "interval_field": self.interval_field,
            "starts": self.starts,
            "ends": self.ends,
            "status": self.status,
            "on_completion": self.on_completion,
            "created": self.created,
            "last_altered": self.last_altered,
            "last_executed": self.last_executed,
            "sql_mode": self.sql_mode,
            "comment": self.comment,
            "cached_at": self.cached_at
        }


class MySqlEventManager:
    """MySQL event scheduling manager with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: EventConfig,
        connection_func: Optional[Callable[[], T]] = None
    ):
        """Initialize the MySQL event scheduling manager.
        
        Args:
            name: The name of the manager
            config: The event configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Event metadata cache
        self.metadata_cache: Dict[str, EventMetadata] = {}
        
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
        """Initialize the event scheduling manager."""
        logger.info(f"Initialized MySQL event scheduling manager {self.name}")
    
    async def close(self) -> None:
        """Close the event scheduling manager."""
        # Clear the metadata cache
        async with self.manager_lock:
            self.metadata_cache.clear()
        
        logger.info(f"Closed MySQL event scheduling manager {self.name}")
    
    async def create_event(
        self,
        event_name: str,
        body: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None,
        execute_at: Optional[str] = None,
        interval_value: Optional[int] = None,
        interval_field: Optional[str] = None,
        starts: Optional[str] = None,
        ends: Optional[str] = None,
        status: str = "ENABLED",
        on_completion: str = "NOT PRESERVE",
        comment: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Create an event.
        
        Args:
            event_name: The name of the event
            body: The body of the event
            schema: The schema of the event, or None for the default schema
            definer: The definer of the event, or None for the current user
            execute_at: The execution time of the event, or None for recurring events
            interval_value: The interval value of the event, or None for one-time events
            interval_field: The interval field of the event, or None for one-time events
            starts: The start time of the event, or None for immediate start
            ends: The end time of the event, or None for no end
            status: The status of the event (ENABLED, DISABLED, SLAVESIDE_DISABLED)
            on_completion: The on completion behavior of the event (PRESERVE, NOT PRESERVE)
            comment: The comment of the event, or None for no comment
            
        Returns:
            Tuple of (success, error)
        """
        # Check if creation is allowed
        if not self.config.allow_create:
            error = Exception("Creating events is not allowed")
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
            
            # Build the CREATE EVENT statement
            full_event_name = f"{schema + '.' if schema else ''}{event_name}"
            
            # Validate the status
            status = status.upper()
            if status not in ["ENABLED", "DISABLED", "SLAVESIDE_DISABLED"]:
                error = Exception(f"Invalid status: {status}")
                self.metrics["failed_operations"] += 1
                return False, error
            
            # Validate the on completion
            on_completion = on_completion.upper()
            if on_completion not in ["PRESERVE", "NOT PRESERVE"]:
                error = Exception(f"Invalid on completion: {on_completion}")
                self.metrics["failed_operations"] += 1
                return False, error
            
            # Build the definer clause
            definer_clause = f"DEFINER = {definer}" if definer else ""
            
            # Build the schedule clause
            if execute_at:
                # One-time event
                schedule_clause = f"AT '{execute_at}'"
            elif interval_value and interval_field:
                # Recurring event
                schedule_clause = f"EVERY {interval_value} {interval_field}"
                
                if starts:
                    schedule_clause += f" STARTS '{starts}'"
                
                if ends:
                    schedule_clause += f" ENDS '{ends}'"
            else:
                error = Exception("Either execute_at or interval_value and interval_field must be provided")
                self.metrics["failed_operations"] += 1
                return False, error
            
            # Build the comment clause
            comment_clause = f"COMMENT '{comment}'" if comment else ""
            
            # Build the CREATE EVENT statement
            create_query = f"""
                CREATE {definer_clause} EVENT {full_event_name}
                ON SCHEDULE {schedule_clause}
                ON COMPLETION {on_completion}
                {comment_clause}
                DO {body}
            """
            
            # Log the operation if configured
            if self.config.log_operations:
                logger.info(f"Creating event: {create_query}")
            
            # Execute the CREATE EVENT statement
            async with conn.cursor() as cursor:
                await cursor.execute(create_query)
                
                # Set the event status if not ENABLED
                if status != "ENABLED":
                    await cursor.execute(f"ALTER EVENT {full_event_name} {status}")
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{event_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            # Update metrics
            self.metrics["successful_operations"] += 1
            
            return True, None
        except Exception as e:
            logger.error(f"Error creating event {event_name}: {e}")
            
            # Update metrics
            self.metrics["failed_operations"] += 1
            
            return False, e
    
    async def alter_event(
        self,
        event_name: str,
        schema: Optional[str] = None,
        definer: Optional[str] = None,
        execute_at: Optional[str] = None,
        interval_value: Optional[int] = None,
        interval_field: Optional[str] = None,
        starts: Optional[str] = None,
        ends: Optional[str] = None,
        status: Optional[str] = None,
        on_completion: Optional[str] = None,
        comment: Optional[str] = None,
        body: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Alter an event.
        
        Args:
            event_name: The name of the event
            schema: The schema of the event, or None for the default schema
            definer: The definer of the event, or None for no change
            execute_at: The execution time of the event, or None for no change
            interval_value: The interval value of the event, or None for no change
            interval_field: The interval field of the event, or None for no change
            starts: The start time of the event, or None for no change
            ends: The end time of the event, or None for no change
            status: The status of the event, or None for no change
            on_completion: The on completion behavior of the event, or None for no change
            comment: The comment of the event, or None for no change
            body: The body of the event, or None for no change
            
        Returns:
            Tuple of (success, error)
        """
        # Check if alteration is allowed
        if not self.config.allow_alter:
            error = Exception("Altering events is not allowed")
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
            
            # Build the ALTER EVENT statement
            full_event_name = f"{schema + '.' if schema else ''}{event_name}"
            
            # Build the definer clause
            definer_clause = f"DEFINER = {definer}" if definer else ""
            
            # Build the schedule clause
            schedule_clause = ""
            if execute_at:
                # One-time event
                schedule_clause = f"ON SCHEDULE AT '{execute_at}'"
            elif interval_value and interval_field:
                # Recurring event
                schedule_clause = f"ON SCHEDULE EVERY {interval_value} {interval_field}"
                
                if starts:
                    schedule_clause += f" STARTS '{starts}'"
                
                if ends:
                    schedule_clause += f" ENDS '{ends}'"
            
            # Build the on completion clause
            on_completion_clause = f"ON COMPLETION {on_completion}" if on_completion else ""
            
            # Build the comment clause
            comment_clause = f"COMMENT '{comment}'" if comment else ""
            
            # Build the body clause
            body_clause = f"DO {body}" if body else ""
            
            # Build the ALTER EVENT statement
            alter_query = f"ALTER EVENT {full_event_name}"
            
            # Add clauses
            clauses = []
            if definer_clause:
                clauses.append(definer_clause)
            if schedule_clause:
                clauses.append(schedule_clause)
            if on_completion_clause:
                clauses.append(on_completion_clause)
            if comment_clause:
                clauses.append(comment_clause)
            if body_clause:
                clauses.append(body_clause)
            
            if not clauses:
                error = Exception("No changes specified")
                self.metrics["failed_operations"] += 1
                return False, error
            
            alter_query += " " + " ".join(clauses)
            
            # Log the operation if configured
            if self.config.log_operations:
                logger.info(f"Altering event: {alter_query}")
            
            # Execute the ALTER EVENT statement
            async with conn.cursor() as cursor:
                await cursor.execute(alter_query)
                
                # Set the event status if specified
                if status:
                    status = status.upper()
                    if status not in ["ENABLED", "DISABLED", "SLAVESIDE_DISABLED"]:
                        error = Exception(f"Invalid status: {status}")
                        self.metrics["failed_operations"] += 1
                        return False, error
                    
                    await cursor.execute(f"ALTER EVENT {full_event_name} {status}")
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{event_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            # Update metrics
            self.metrics["successful_operations"] += 1
            
            return True, None
        except Exception as e:
            logger.error(f"Error altering event {event_name}: {e}")
            
            # Update metrics
            self.metrics["failed_operations"] += 1
            
            return False, e
    
    async def drop_event(
        self,
        event_name: str,
        schema: Optional[str] = None
    ) -> Tuple[bool, Optional[Exception]]:
        """Drop an event.
        
        Args:
            event_name: The name of the event
            schema: The schema of the event, or None for the default schema
            
        Returns:
            Tuple of (success, error)
        """
        # Check if dropping is allowed
        if not self.config.allow_drop:
            error = Exception("Dropping events is not allowed")
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
            
            # Build the DROP EVENT statement
            full_event_name = f"{schema + '.' if schema else ''}{event_name}"
            drop_query = f"DROP EVENT IF EXISTS {full_event_name}"
            
            # Log the operation if configured
            if self.config.log_operations:
                logger.info(f"Dropping event: {drop_query}")
            
            # Execute the DROP EVENT statement
            async with conn.cursor() as cursor:
                await cursor.execute(drop_query)
            
            # Invalidate the metadata cache
            async with self.manager_lock:
                cache_key = f"{schema or ''}.{event_name}".lower()
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
            
            # Update metrics
            self.metrics["successful_operations"] += 1
            
            return True, None
        except Exception as e:
            logger.error(f"Error dropping event {event_name}: {e}")
            
            # Update metrics
            self.metrics["failed_operations"] += 1
            
            return False, e
    
    async def get_event_metadata(
        self,
        event_name: str,
        schema: Optional[str] = None
    ) -> Optional[EventMetadata]:
        """Get metadata for an event.
        
        Args:
            event_name: The name of the event
            schema: The schema of the event, or None for the default schema
            
        Returns:
            The event metadata, or None if not found
        """
        # Check if we have a connection function
        if not self.connection_func:
            return None
        
        # Check if metadata caching is enabled
        if self.config.cache_metadata:
            # Check if the metadata is in the cache
            cache_key = f"{schema or ''}.{event_name}".lower()
            
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
            
            # Get the event metadata
            async with conn.cursor() as cursor:
                # Get the event from information_schema
                query = """
                    SELECT
                        EVENT_SCHEMA,
                        EVENT_NAME,
                        DEFINER,
                        EVENT_BODY,
                        EVENT_DEFINITION,
                        EVENT_TYPE,
                        EXECUTE_AT,
                        INTERVAL_VALUE,
                        INTERVAL_FIELD,
                        STARTS,
                        ENDS,
                        STATUS,
                        ON_COMPLETION,
                        CREATED,
                        LAST_ALTERED,
                        LAST_EXECUTED,
                        EVENT_COMMENT,
                        SQL_MODE
                    FROM
                        information_schema.EVENTS
                    WHERE
                        EVENT_NAME = %s
                """
                
                params = [event_name]
                
                if schema:
                    query += " AND EVENT_SCHEMA = %s"
                    params.append(schema)
                
                await cursor.execute(query, params)
                result = await cursor.fetchone()
                
                if not result:
                    return None
                
                # Create the metadata
                metadata = EventMetadata(
                    name=result[1],
                    schema=result[0],
                    definer=result[2],
                    body=result[4],
                    execute_at=str(result[6]) if result[6] else None,
                    interval_value=result[7],
                    interval_field=result[8],
                    starts=str(result[9]) if result[9] else None,
                    ends=str(result[10]) if result[10] else None,
                    status=result[11],
                    on_completion=result[12],
                    created=str(result[13]),
                    last_altered=str(result[14]),
                    last_executed=str(result[15]) if result[15] else None,
                    sql_mode=result[17],
                    comment=result[16]
                )
                
                # Cache the metadata if enabled
                if self.config.cache_metadata:
                    async with self.manager_lock:
                        self.metadata_cache[cache_key] = metadata
                
                return metadata
        except Exception as e:
            logger.error(f"Error getting event metadata for {event_name}: {e}")
            return None
    
    async def list_events(
        self,
        schema: Optional[str] = None,
        pattern: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List events.
        
        Args:
            schema: The schema to list events from, or None for all schemas
            pattern: A pattern to filter event names, or None for all events
            status: The status to filter events by, or None for all statuses
            
        Returns:
            List of event metadata
        """
        # Check if we have a connection function
        if not self.connection_func:
            return []
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the events
            async with conn.cursor() as cursor:
                # Build the query
                query = """
                    SELECT
                        EVENT_SCHEMA,
                        EVENT_NAME,
                        DEFINER,
                        EVENT_BODY,
                        EVENT_DEFINITION,
                        EVENT_TYPE,
                        EXECUTE_AT,
                        INTERVAL_VALUE,
                        INTERVAL_FIELD,
                        STARTS,
                        ENDS,
                        STATUS,
                        ON_COMPLETION,
                        CREATED,
                        LAST_ALTERED,
                        LAST_EXECUTED,
                        EVENT_COMMENT,
                        SQL_MODE
                    FROM
                        information_schema.EVENTS
                    WHERE
                        1=1
                """
                
                params = []
                
                if schema:
                    query += " AND EVENT_SCHEMA = %s"
                    params.append(schema)
                
                if pattern:
                    query += " AND EVENT_NAME LIKE %s"
                    params.append(pattern)
                
                if status:
                    query += " AND STATUS = %s"
                    params.append(status)
                
                query += " ORDER BY EVENT_SCHEMA, EVENT_NAME"
                
                await cursor.execute(query, params)
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                events = []
                for result in results:
                    event = {
                        "schema": result[0],
                        "name": result[1],
                        "definer": result[2],
                        "body": result[4],
                        "type": result[5],
                        "execute_at": str(result[6]) if result[6] else None,
                        "interval_value": result[7],
                        "interval_field": result[8],
                        "starts": str(result[9]) if result[9] else None,
                        "ends": str(result[10]) if result[10] else None,
                        "status": result[11],
                        "on_completion": result[12],
                        "created": str(result[13]),
                        "last_altered": str(result[14]),
                        "last_executed": str(result[15]) if result[15] else None,
                        "comment": result[16],
                        "sql_mode": result[17]
                    }
                    events.append(event)
                
                return events
        except Exception as e:
            logger.error(f"Error listing events: {e}")
            return []
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event scheduling manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
