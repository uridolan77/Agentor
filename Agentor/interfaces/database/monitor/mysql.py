"""
MySQL performance monitor for the Agentor framework.

This module provides a specialized performance monitor for MySQL databases.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set
import weakref

from .config import MonitoringConfig, MonitoringLevel

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Connection type


class QueryStats:
    """Statistics for a query."""
    
    def __init__(self, query: str):
        """Initialize the query statistics.
        
        Args:
            query: The query
        """
        self.query = query
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.total_rows = 0
        self.min_rows = float('inf')
        self.max_rows = 0
        self.last_executed = 0.0
        self.slow_count = 0
        self.very_slow_count = 0
        self.error_count = 0
        self.last_error = None
        self.last_plan = None
    
    def add_execution(self, execution_time: float, rows: int, error: Optional[Exception] = None, plan: Optional[Dict[str, Any]] = None) -> None:
        """Add an execution to the statistics.
        
        Args:
            execution_time: The execution time in seconds
            rows: The number of rows affected or returned
            error: The error, if any
            plan: The query plan, if any
        """
        self.count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.total_rows += rows
        self.min_rows = min(self.min_rows, rows)
        self.max_rows = max(self.max_rows, rows)
        self.last_executed = time.time()
        
        if error:
            self.error_count += 1
            self.last_error = str(error)
        
        if plan:
            self.last_plan = plan
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the statistics to a dictionary.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "query": self.query,
            "count": self.count,
            "total_time": self.total_time,
            "avg_time": self.total_time / self.count if self.count > 0 else 0.0,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "total_rows": self.total_rows,
            "avg_rows": self.total_rows / self.count if self.count > 0 else 0.0,
            "min_rows": self.min_rows if self.min_rows != float('inf') else 0,
            "max_rows": self.max_rows,
            "last_executed": self.last_executed,
            "slow_count": self.slow_count,
            "very_slow_count": self.very_slow_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_plan": self.last_plan
        }


class MySqlPerformanceMonitor:
    """MySQL performance monitor with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: MonitoringConfig,
        connection_func: Optional[Callable[[], T]] = None
    ):
        """Initialize the MySQL performance monitor.
        
        Args:
            name: The name of the monitor
            config: The monitoring configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Query statistics
        self.query_stats: Dict[str, QueryStats] = {}
        
        # Table statistics
        self.table_stats: Dict[str, Dict[str, Any]] = {}
        
        # Index statistics
        self.index_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Monitor lock
        self.monitor_lock = asyncio.Lock()
        
        # Monitor metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_queries": 0,
            "slow_queries": 0,
            "very_slow_queries": 0,
            "total_execution_time": 0.0,
            "avg_execution_time": 0.0,
            "max_execution_time": 0.0,
            "min_execution_time": float('inf'),
            "total_rows": 0,
            "avg_rows": 0.0,
            "max_rows": 0,
            "min_rows": float('inf'),
            "total_alerts": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the performance monitor."""
        logger.info(f"Initialized MySQL performance monitor {self.name}")
        
        # Collect initial table and index statistics if configured
        if self.config.collect_table_stats or self.config.collect_index_stats:
            await self.collect_database_stats()
    
    async def close(self) -> None:
        """Close the performance monitor."""
        logger.info(f"Closed MySQL performance monitor {self.name}")
    
    async def record_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        rows: int = 0,
        error: Optional[Exception] = None
    ) -> None:
        """Record a query execution.
        
        Args:
            query: The query
            params: The query parameters
            execution_time: The execution time in seconds
            rows: The number of rows affected or returned
            error: The error, if any
        """
        # Check if the query should be monitored
        if not self.config.should_monitor_query(query):
            return
        
        # Check if the query should be sampled
        if not self.config.should_sample_query(query):
            return
        
        # Update metrics
        self.metrics["total_queries"] += 1
        self.metrics["total_execution_time"] += execution_time
        self.metrics["avg_execution_time"] = self.metrics["total_execution_time"] / self.metrics["total_queries"]
        self.metrics["max_execution_time"] = max(self.metrics["max_execution_time"], execution_time)
        self.metrics["min_execution_time"] = min(self.metrics["min_execution_time"], execution_time)
        self.metrics["total_rows"] += rows
        self.metrics["avg_rows"] = self.metrics["total_rows"] / self.metrics["total_queries"]
        self.metrics["max_rows"] = max(self.metrics["max_rows"], rows)
        self.metrics["min_rows"] = min(self.metrics["min_rows"], rows)
        self.metrics["last_activity"] = time.time()
        
        # Check if the query is slow
        is_slow = execution_time >= self.config.slow_query_threshold
        is_very_slow = execution_time >= self.config.very_slow_query_threshold
        
        if is_slow:
            self.metrics["slow_queries"] += 1
        
        if is_very_slow:
            self.metrics["very_slow_queries"] += 1
        
        # Log slow queries if configured
        if is_slow and self.config.log_slow_queries:
            logger.warning(f"Slow query ({execution_time:.2f}s): {query}")
        
        # Check if we should alert on the query
        if self.config.should_alert_on_query(query, execution_time):
            self.metrics["total_alerts"] += 1
            logger.error(f"Alert: Slow query ({execution_time:.2f}s): {query}")
        
        # Collect the query plan if configured
        plan = None
        if self.config.should_collect_query_plan(query):
            plan = await self.get_query_plan(query, params)
        
        # Log the query plan if configured
        if plan and self.config.log_query_plans:
            logger.info(f"Query plan for {query}: {json.dumps(plan, indent=2)}")
        
        # Update query statistics
        async with self.monitor_lock:
            # Get or create the query statistics
            if query not in self.query_stats:
                self.query_stats[query] = QueryStats(query)
            
            # Update the statistics
            stats = self.query_stats[query]
            stats.add_execution(execution_time, rows, error, plan)
            
            if is_slow:
                stats.slow_count += 1
            
            if is_very_slow:
                stats.very_slow_count += 1
        
        # Log query statistics if configured
        if self.config.log_query_stats:
            logger.info(f"Query statistics for {query}: {json.dumps(stats.to_dict(), indent=2)}")
    
    async def get_query_plan(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get the query plan for a query.
        
        Args:
            query: The query
            params: The query parameters
            
        Returns:
            The query plan, or None if not available
        """
        # Check if we have a connection function
        if not self.connection_func:
            return None
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Execute EXPLAIN
            async with conn.cursor() as cursor:
                # Convert the query to EXPLAIN
                explain_query = f"EXPLAIN {query}"
                
                # Execute the EXPLAIN query
                await cursor.execute(explain_query, params or {})
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to a dictionary
                plan = []
                for row in results:
                    plan_row = {}
                    for i, column in enumerate(cursor.description):
                        plan_row[column[0]] = row[i]
                    plan.append(plan_row)
                
                return {"plan": plan}
        except Exception as e:
            logger.error(f"Error getting query plan for {query}: {e}")
            return None
    
    async def collect_database_stats(self) -> None:
        """Collect database statistics."""
        # Check if we have a connection function
        if not self.connection_func:
            return
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Collect table statistics if configured
            if self.config.collect_table_stats:
                await self._collect_table_stats(conn)
            
            # Collect index statistics if configured
            if self.config.collect_index_stats:
                await self._collect_index_stats(conn)
        except Exception as e:
            logger.error(f"Error collecting database statistics: {e}")
    
    async def _collect_table_stats(self, conn: T) -> None:
        """Collect table statistics.
        
        Args:
            conn: The database connection
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Get table statistics
                await cursor.execute("""
                    SELECT
                        table_schema,
                        table_name,
                        engine,
                        table_rows,
                        avg_row_length,
                        data_length,
                        index_length,
                        auto_increment,
                        create_time,
                        update_time
                    FROM
                        information_schema.tables
                    WHERE
                        table_schema NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
                """)
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Update table statistics
                async with self.monitor_lock:
                    for row in results:
                        table_name = f"{row[0]}.{row[1]}"
                        self.table_stats[table_name] = {
                            "schema": row[0],
                            "name": row[1],
                            "engine": row[2],
                            "rows": row[3],
                            "avg_row_length": row[4],
                            "data_length": row[5],
                            "index_length": row[6],
                            "auto_increment": row[7],
                            "create_time": row[8],
                            "update_time": row[9],
                            "collected_at": time.time()
                        }
        except Exception as e:
            logger.error(f"Error collecting table statistics: {e}")
    
    async def _collect_index_stats(self, conn: T) -> None:
        """Collect index statistics.
        
        Args:
            conn: The database connection
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Get index statistics
                await cursor.execute("""
                    SELECT
                        table_schema,
                        table_name,
                        index_name,
                        non_unique,
                        seq_in_index,
                        column_name,
                        cardinality,
                        nullable,
                        index_type
                    FROM
                        information_schema.statistics
                    WHERE
                        table_schema NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
                    ORDER BY
                        table_schema, table_name, index_name, seq_in_index
                """)
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Update index statistics
                async with self.monitor_lock:
                    for row in results:
                        table_name = f"{row[0]}.{row[1]}"
                        index_name = row[2]
                        
                        if table_name not in self.index_stats:
                            self.index_stats[table_name] = {}
                        
                        if index_name not in self.index_stats[table_name]:
                            self.index_stats[table_name][index_name] = {
                                "schema": row[0],
                                "table": row[1],
                                "name": index_name,
                                "non_unique": row[3],
                                "columns": [],
                                "cardinality": row[6],
                                "nullable": row[7],
                                "type": row[8],
                                "collected_at": time.time()
                            }
                        
                        # Add the column
                        self.index_stats[table_name][index_name]["columns"].append({
                            "seq": row[4],
                            "name": row[5]
                        })
        except Exception as e:
            logger.error(f"Error collecting index statistics: {e}")
    
    async def get_query_stats(self, query: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get query statistics.
        
        Args:
            query: The query, or None for all queries
            
        Returns:
            Dictionary or list of dictionaries of query statistics
        """
        async with self.monitor_lock:
            if query:
                # Get statistics for a specific query
                if query in self.query_stats:
                    return self.query_stats[query].to_dict()
                else:
                    return {}
            else:
                # Get statistics for all queries
                return [stats.to_dict() for stats in self.query_stats.values()]
    
    async def get_table_stats(self, table: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get table statistics.
        
        Args:
            table: The table, or None for all tables
            
        Returns:
            Dictionary or dictionary of dictionaries of table statistics
        """
        async with self.monitor_lock:
            if table:
                # Get statistics for a specific table
                if table in self.table_stats:
                    return self.table_stats[table]
                else:
                    return {}
            else:
                # Get statistics for all tables
                return self.table_stats
    
    async def get_index_stats(self, table: Optional[str] = None, index: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
        """Get index statistics.
        
        Args:
            table: The table, or None for all tables
            index: The index, or None for all indexes
            
        Returns:
            Dictionary, dictionary of dictionaries, or dictionary of dictionaries of dictionaries of index statistics
        """
        async with self.monitor_lock:
            if table:
                if table in self.index_stats:
                    if index:
                        # Get statistics for a specific index
                        if index in self.index_stats[table]:
                            return self.index_stats[table][index]
                        else:
                            return {}
                    else:
                        # Get statistics for all indexes in a table
                        return self.index_stats[table]
                else:
                    return {}
            else:
                # Get statistics for all indexes in all tables
                return self.index_stats
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance monitor metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
