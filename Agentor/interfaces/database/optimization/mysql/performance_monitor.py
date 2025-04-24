"""
MySQL performance monitor for the Agentor framework.

This module provides a specialized monitor for MySQL performance.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
import weakref

from ..config import PerformanceMonitoringConfig, OptimizationLevel

logger = logging.getLogger(__name__)


class MySqlPerformanceMonitor:
    """MySQL performance monitor with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: PerformanceMonitoringConfig,
        connection_func: Optional[Callable[[], Any]] = None
    ):
        """Initialize the MySQL performance monitor.
        
        Args:
            name: The name of the monitor
            config: The performance monitoring configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {}
        self.historical_metrics: List[Dict[str, Any]] = []
        self.slow_queries: List[Dict[str, Any]] = []
        
        # Monitor metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_collections": 0,
            "slow_queries_detected": 0,
            "high_load_alerts": 0
        }
        
        # Monitor lock
        self.monitor_lock = asyncio.Lock()
        
        # Monitor task
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
    
    async def initialize(self) -> None:
        """Initialize the performance monitor."""
        logger.info(f"Initialized MySQL performance monitor {self.name}")
        
        # Start the monitoring task
        self.monitoring_task = self.loop.create_task(self._monitoring_loop())
    
    async def close(self) -> None:
        """Close the performance monitor."""
        # Cancel the monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        # Clear metrics
        self.performance_metrics.clear()
        self.historical_metrics.clear()
        self.slow_queries.clear()
        
        logger.info(f"Closed MySQL performance monitor {self.name}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Check if we have metrics
        if not self.performance_metrics:
            # Collect metrics
            await self._collect_metrics()
        
        return self.performance_metrics
    
    async def get_historical_metrics(self) -> List[Dict[str, Any]]:
        """Get historical performance metrics.
        
        Returns:
            List of historical performance metrics
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        return self.historical_metrics
    
    async def get_slow_queries(self) -> List[Dict[str, Any]]:
        """Get slow queries.
        
        Returns:
            List of slow queries
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        return self.slow_queries
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance monitor metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for performance monitoring."""
        try:
            while True:
                # Sleep for the monitoring interval
                await asyncio.sleep(self.config.monitoring_interval)
                
                # Collect metrics
                await self._collect_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Collect detailed metrics
                if time.time() % self.config.detailed_monitoring_interval < self.config.monitoring_interval:
                    await self._collect_detailed_metrics()
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in monitoring loop: {e}")
    
    async def _collect_metrics(self) -> None:
        """Collect performance metrics."""
        # Check if we have a connection function
        if not self.connection_func:
            return
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Update metrics
            self.metrics["total_collections"] += 1
            
            # Get server status
            server_status = await self._get_server_status(conn)
            
            # Calculate metrics
            metrics = {
                "timestamp": time.time(),
                "queries_per_second": int(server_status.get("Questions", 0)) / self.config.monitoring_interval,
                "slow_queries": int(server_status.get("Slow_queries", 0)),
                "threads_connected": int(server_status.get("Threads_connected", 0)),
                "threads_running": int(server_status.get("Threads_running", 0)),
                "connection_errors": int(server_status.get("Connection_errors_internal", 0)),
                "aborted_clients": int(server_status.get("Aborted_clients", 0)),
                "aborted_connections": int(server_status.get("Aborted_connects", 0)),
                "table_locks_waited": int(server_status.get("Table_locks_waited", 0)),
                "table_locks_immediate": int(server_status.get("Table_locks_immediate", 0)),
                "innodb_buffer_pool_read_requests": int(server_status.get("Innodb_buffer_pool_read_requests", 0)),
                "innodb_buffer_pool_reads": int(server_status.get("Innodb_buffer_pool_reads", 0)),
                "innodb_row_lock_waits": int(server_status.get("Innodb_row_lock_waits", 0)),
                "innodb_row_lock_time": int(server_status.get("Innodb_row_lock_time", 0)),
                "created_tmp_disk_tables": int(server_status.get("Created_tmp_disk_tables", 0)),
                "created_tmp_tables": int(server_status.get("Created_tmp_tables", 0)),
                "select_full_join": int(server_status.get("Select_full_join", 0)),
                "select_range_check": int(server_status.get("Select_range_check", 0)),
                "sort_merge_passes": int(server_status.get("Sort_merge_passes", 0))
            }
            
            # Calculate derived metrics
            if metrics["innodb_buffer_pool_read_requests"] > 0:
                metrics["buffer_pool_hit_ratio"] = 1.0 - (metrics["innodb_buffer_pool_reads"] / metrics["innodb_buffer_pool_read_requests"])
            else:
                metrics["buffer_pool_hit_ratio"] = 1.0
            
            if metrics["table_locks_immediate"] + metrics["table_locks_waited"] > 0:
                metrics["table_lock_wait_ratio"] = metrics["table_locks_waited"] / (metrics["table_locks_immediate"] + metrics["table_locks_waited"])
            else:
                metrics["table_lock_wait_ratio"] = 0.0
            
            if metrics["created_tmp_tables"] > 0:
                metrics["tmp_disk_table_ratio"] = metrics["created_tmp_disk_tables"] / metrics["created_tmp_tables"]
            else:
                metrics["tmp_disk_table_ratio"] = 0.0
            
            # Calculate load
            metrics["load"] = min(1.0, (
                metrics["threads_running"] / max(metrics["threads_connected"], 1) +
                metrics["table_lock_wait_ratio"] +
                (1.0 - metrics["buffer_pool_hit_ratio"]) +
                metrics["tmp_disk_table_ratio"]
            ) / 4.0)
            
            # Store the metrics
            self.performance_metrics = metrics
            
            # Add to historical metrics
            self.historical_metrics.append(metrics)
            
            # Limit the number of historical metrics
            max_history = int(3600 / self.config.monitoring_interval)  # 1 hour of history
            if len(self.historical_metrics) > max_history:
                self.historical_metrics = self.historical_metrics[-max_history:]
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _collect_detailed_metrics(self) -> None:
        """Collect detailed performance metrics."""
        # Check if we have a connection function
        if not self.connection_func:
            return
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Collect slow queries
            if self.config.collect_query_metrics:
                await self._collect_slow_queries(conn)
            
            # Collect index metrics
            if self.config.collect_index_metrics:
                await self._collect_index_metrics(conn)
            
            # Collect server metrics
            if self.config.collect_server_metrics:
                await self._collect_server_metrics(conn)
        except Exception as e:
            logger.error(f"Error collecting detailed performance metrics: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for performance alerts."""
        # Check if we have metrics
        if not self.performance_metrics:
            return
        
        # Check for slow queries
        if self.config.alert_on_slow_queries:
            slow_queries = self.performance_metrics.get("slow_queries", 0)
            if slow_queries > 0:
                logger.warning(f"Alert: {slow_queries} slow queries detected")
                self.metrics["slow_queries_detected"] += slow_queries
        
        # Check for high load
        if self.config.alert_on_high_load:
            load = self.performance_metrics.get("load", 0.0)
            if load > self.config.alert_threshold:
                logger.warning(f"Alert: High server load detected ({load:.2f})")
                self.metrics["high_load_alerts"] += 1
    
    async def _get_server_status(self, conn: Any) -> Dict[str, Any]:
        """Get server status.
        
        Args:
            conn: The database connection
            
        Returns:
            Dictionary of server status
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                await cursor.execute("SHOW GLOBAL STATUS")
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to a dictionary
                status = {}
                for row in results:
                    status[row[0]] = row[1]
                
                return status
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
            return {}
    
    async def _collect_slow_queries(self, conn: Any) -> None:
        """Collect slow queries.
        
        Args:
            conn: The database connection
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Get slow queries from the slow query log
                await cursor.execute("""
                    SELECT
                        start_time,
                        user_host,
                        query_time,
                        lock_time,
                        rows_sent,
                        rows_examined,
                        db,
                        sql_text
                    FROM
                        mysql.slow_log
                    WHERE
                        query_time >= %s
                    ORDER BY
                        query_time DESC
                    LIMIT 100
                """, (self.config.slow_query_threshold,))
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                slow_queries = []
                for row in results:
                    query = {}
                    for i, column_name in enumerate(cursor.description):
                        query[column_name[0]] = row[i]
                    slow_queries.append(query)
                
                # Store the slow queries
                self.slow_queries = slow_queries
                
                # Update metrics
                self.metrics["slow_queries_detected"] += len(slow_queries)
        except Exception as e:
            logger.error(f"Error collecting slow queries: {e}")
    
    async def _collect_index_metrics(self, conn: Any) -> None:
        """Collect index metrics.
        
        Args:
            conn: The database connection
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Get the database name
                await cursor.execute("SELECT DATABASE()")
                db_result = await cursor.fetchone()
                db_name = db_result[0]
                
                # Get index metrics from information_schema
                await cursor.execute("""
                    SELECT
                        t.TABLE_NAME AS table_name,
                        t.TABLE_ROWS AS table_rows,
                        t.DATA_LENGTH AS data_length,
                        t.INDEX_LENGTH AS index_length,
                        i.INDEX_NAME AS index_name,
                        i.COLUMN_NAME AS column_name,
                        i.SEQ_IN_INDEX AS seq_in_index,
                        i.CARDINALITY AS cardinality,
                        s.ROWS_READ AS rows_read,
                        s.ROWS_INSERTED AS rows_inserted,
                        s.ROWS_UPDATED AS rows_updated,
                        s.ROWS_DELETED AS rows_deleted
                    FROM
                        information_schema.TABLES t
                    JOIN
                        information_schema.STATISTICS i
                    ON
                        t.TABLE_SCHEMA = i.TABLE_SCHEMA
                        AND t.TABLE_NAME = i.TABLE_NAME
                    LEFT JOIN
                        information_schema.INDEX_STATISTICS s
                    ON
                        i.TABLE_SCHEMA = s.INDEX_SCHEMA
                        AND i.TABLE_NAME = s.TABLE_NAME
                        AND i.INDEX_NAME = s.INDEX_NAME
                    WHERE
                        t.TABLE_SCHEMA = %s
                    ORDER BY
                        t.TABLE_NAME, i.INDEX_NAME, i.SEQ_IN_INDEX
                """, (db_name,))
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                index_metrics = []
                for row in results:
                    metric = {}
                    for i, column_name in enumerate(cursor.description):
                        metric[column_name[0]] = row[i]
                    index_metrics.append(metric)
                
                # Add index metrics to performance metrics
                self.performance_metrics["index_metrics"] = index_metrics
        except Exception as e:
            logger.error(f"Error collecting index metrics: {e}")
    
    async def _collect_server_metrics(self, conn: Any) -> None:
        """Collect server metrics.
        
        Args:
            conn: The database connection
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Get server metrics
                await cursor.execute("""
                    SELECT
                        @@version AS version,
                        @@version_comment AS version_comment,
                        @@innodb_buffer_pool_size AS buffer_pool_size,
                        @@innodb_buffer_pool_instances AS buffer_pool_instances,
                        @@innodb_log_file_size AS log_file_size,
                        @@innodb_flush_log_at_trx_commit AS flush_log_at_trx_commit,
                        @@innodb_flush_method AS flush_method,
                        @@max_connections AS max_connections,
                        @@thread_cache_size AS thread_cache_size,
                        @@query_cache_size AS query_cache_size,
                        @@query_cache_type AS query_cache_type,
                        @@tmp_table_size AS tmp_table_size,
                        @@max_heap_table_size AS max_heap_table_size,
                        @@innodb_io_capacity AS io_capacity,
                        @@innodb_io_capacity_max AS io_capacity_max
                """)
                
                # Fetch the results
                result = await cursor.fetchone()
                
                # Convert the result to a dictionary
                server_metrics = {}
                for i, column_name in enumerate(cursor.description):
                    server_metrics[column_name[0]] = result[i]
                
                # Add server metrics to performance metrics
                self.performance_metrics["server_metrics"] = server_metrics
        except Exception as e:
            logger.error(f"Error collecting server metrics: {e}")
    
    def __del__(self):
        """Clean up resources when the monitor is garbage collected."""
        # Cancel the monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
