"""
MySQL server optimizer for the Agentor framework.

This module provides a specialized optimizer for MySQL server configuration.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
import weakref

from ..config import ServerOptimizationConfig, OptimizationLevel

logger = logging.getLogger(__name__)


class MySqlServerOptimizer:
    """MySQL server optimizer with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: ServerOptimizationConfig,
        connection_func: Optional[Callable[[], Any]] = None
    ):
        """Initialize the MySQL server optimizer.
        
        Args:
            name: The name of the optimizer
            config: The server optimization configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Server cache
        self.server_variables: Dict[str, Any] = {}
        self.server_status: Dict[str, Any] = {}
        self.server_stats: Dict[str, Any] = {}
        
        # Optimizer metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_analyses": 0,
            "recommended_settings": 0,
            "applied_settings": 0
        }
        
        # Optimizer lock
        self.optimizer_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the server optimizer."""
        logger.info(f"Initialized MySQL server optimizer {self.name}")
    
    async def close(self) -> None:
        """Close the server optimizer."""
        # Clear caches
        self.server_variables.clear()
        self.server_status.clear()
        self.server_stats.clear()
        
        logger.info(f"Closed MySQL server optimizer {self.name}")
    
    async def optimize_server(self) -> List[Dict[str, Any]]:
        """Optimize server settings.
        
        Returns:
            List of server setting recommendations
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        self.metrics["total_analyses"] += 1
        
        # Check if we have a connection function
        if not self.connection_func:
            return []
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the server variables
            server_variables = await self._get_server_variables(conn)
            
            # Get the server status
            server_status = await self._get_server_status(conn)
            
            # Get the server statistics
            server_stats = await self._get_server_stats(conn)
            
            # Analyze the server
            analysis = await self.analyze_server()
            
            # Generate recommendations
            recommendations = []
            
            # Check for basic optimizations
            if self.config.level in [OptimizationLevel.BASIC, OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE]:
                # Check innodb_buffer_pool_size
                buffer_pool_size = int(server_variables.get("innodb_buffer_pool_size", 0))
                server_memory = server_stats.get("server_memory", 0)
                
                if server_memory > 0 and buffer_pool_size < server_memory * 0.5:
                    # Recommend 50-80% of server memory for buffer pool
                    recommended_size = int(server_memory * 0.7)
                    
                    recommendations.append({
                        "variable": "innodb_buffer_pool_size",
                        "current_value": buffer_pool_size,
                        "recommended_value": recommended_size,
                        "reason": "InnoDB buffer pool size is too small",
                        "sql": f"SET GLOBAL innodb_buffer_pool_size = {recommended_size}"
                    })
                
                # Check query_cache_size
                query_cache_size = int(server_variables.get("query_cache_size", 0))
                query_cache_type = int(server_variables.get("query_cache_type", 0))
                
                if query_cache_type == 1 and query_cache_size < 20 * 1024 * 1024:
                    # Recommend at least 20MB for query cache
                    recommended_size = 20 * 1024 * 1024
                    
                    recommendations.append({
                        "variable": "query_cache_size",
                        "current_value": query_cache_size,
                        "recommended_value": recommended_size,
                        "reason": "Query cache size is too small",
                        "sql": f"SET GLOBAL query_cache_size = {recommended_size}"
                    })
                
                # Check max_connections
                max_connections = int(server_variables.get("max_connections", 0))
                max_used_connections = int(server_status.get("Max_used_connections", 0))
                
                if max_used_connections > max_connections * 0.8:
                    # Recommend increasing max_connections
                    recommended_value = int(max_connections * 1.5)
                    
                    recommendations.append({
                        "variable": "max_connections",
                        "current_value": max_connections,
                        "recommended_value": recommended_value,
                        "reason": "Max connections is too low",
                        "sql": f"SET GLOBAL max_connections = {recommended_value}"
                    })
            
            # Check for moderate optimizations
            if self.config.level in [OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE]:
                # Check innodb_log_file_size
                log_file_size = int(server_variables.get("innodb_log_file_size", 0))
                buffer_pool_size = int(server_variables.get("innodb_buffer_pool_size", 0))
                
                if buffer_pool_size > 0 and log_file_size < buffer_pool_size * 0.25:
                    # Recommend 25% of buffer pool size for log file size
                    recommended_size = int(buffer_pool_size * 0.25)
                    
                    recommendations.append({
                        "variable": "innodb_log_file_size",
                        "current_value": log_file_size,
                        "recommended_value": recommended_size,
                        "reason": "InnoDB log file size is too small",
                        "sql": f"SET GLOBAL innodb_log_file_size = {recommended_size}"
                    })
                
                # Check innodb_flush_log_at_trx_commit
                flush_log = int(server_variables.get("innodb_flush_log_at_trx_commit", 1))
                
                if flush_log == 1 and analysis.get("write_intensive", False):
                    # Recommend 2 for write-intensive workloads
                    recommendations.append({
                        "variable": "innodb_flush_log_at_trx_commit",
                        "current_value": flush_log,
                        "recommended_value": 2,
                        "reason": "Write-intensive workload detected",
                        "sql": "SET GLOBAL innodb_flush_log_at_trx_commit = 2"
                    })
                
                # Check innodb_flush_method
                flush_method = server_variables.get("innodb_flush_method", "")
                
                if flush_method != "O_DIRECT" and server_stats.get("os_type") == "Linux":
                    # Recommend O_DIRECT for Linux
                    recommendations.append({
                        "variable": "innodb_flush_method",
                        "current_value": flush_method,
                        "recommended_value": "O_DIRECT",
                        "reason": "O_DIRECT is recommended for Linux",
                        "sql": "SET GLOBAL innodb_flush_method = 'O_DIRECT'"
                    })
            
            # Check for aggressive optimizations
            if self.config.level == OptimizationLevel.AGGRESSIVE:
                # Check innodb_buffer_pool_instances
                buffer_pool_instances = int(server_variables.get("innodb_buffer_pool_instances", 1))
                buffer_pool_size = int(server_variables.get("innodb_buffer_pool_size", 0))
                
                if buffer_pool_size > 1024 * 1024 * 1024 and buffer_pool_instances < 8:
                    # Recommend 8 instances for large buffer pools
                    recommendations.append({
                        "variable": "innodb_buffer_pool_instances",
                        "current_value": buffer_pool_instances,
                        "recommended_value": 8,
                        "reason": "Large buffer pool detected",
                        "sql": "SET GLOBAL innodb_buffer_pool_instances = 8"
                    })
                
                # Check innodb_io_capacity
                io_capacity = int(server_variables.get("innodb_io_capacity", 200))
                
                if io_capacity == 200 and server_stats.get("disk_type") == "SSD":
                    # Recommend higher IO capacity for SSDs
                    recommendations.append({
                        "variable": "innodb_io_capacity",
                        "current_value": io_capacity,
                        "recommended_value": 2000,
                        "reason": "SSD detected",
                        "sql": "SET GLOBAL innodb_io_capacity = 2000"
                    })
                
                # Check tmp_table_size and max_heap_table_size
                tmp_table_size = int(server_variables.get("tmp_table_size", 0))
                max_heap_table_size = int(server_variables.get("max_heap_table_size", 0))
                created_tmp_disk_tables = int(server_status.get("Created_tmp_disk_tables", 0))
                created_tmp_tables = int(server_status.get("Created_tmp_tables", 0))
                
                if created_tmp_disk_tables > 0 and created_tmp_tables > 0:
                    tmp_disk_ratio = created_tmp_disk_tables / created_tmp_tables
                    
                    if tmp_disk_ratio > 0.25:
                        # Recommend increasing tmp_table_size and max_heap_table_size
                        recommended_size = min(256 * 1024 * 1024, tmp_table_size * 2)
                        
                        recommendations.append({
                            "variable": "tmp_table_size",
                            "current_value": tmp_table_size,
                            "recommended_value": recommended_size,
                            "reason": "Too many temporary tables created on disk",
                            "sql": f"SET GLOBAL tmp_table_size = {recommended_size}"
                        })
                        
                        recommendations.append({
                            "variable": "max_heap_table_size",
                            "current_value": max_heap_table_size,
                            "recommended_value": recommended_size,
                            "reason": "Too many temporary tables created on disk",
                            "sql": f"SET GLOBAL max_heap_table_size = {recommended_size}"
                        })
            
            # Update metrics
            self.metrics["recommended_settings"] += len(recommendations)
            
            # Apply settings if auto-configure is enabled
            if self.config.auto_configure_server and recommendations:
                applied_settings = await self._apply_settings(conn, recommendations)
                self.metrics["applied_settings"] += applied_settings
            
            return recommendations
        except Exception as e:
            logger.error(f"Error optimizing server settings: {e}")
            return []
    
    async def analyze_server(self) -> Dict[str, Any]:
        """Analyze server settings.
        
        Returns:
            Dictionary of analysis results
        """
        # Check if server analysis is enabled
        if not self.config.analyze_server:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Check if we have a connection function
        if not self.connection_func:
            return {}
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the server variables
            server_variables = await self._get_server_variables(conn)
            
            # Get the server status
            server_status = await self._get_server_status(conn)
            
            # Get the server statistics
            server_stats = await self._get_server_stats(conn)
            
            # Analyze the server
            analysis = {
                "server_version": server_variables.get("version", ""),
                "server_memory": server_stats.get("server_memory", 0),
                "server_cpu_cores": server_stats.get("server_cpu_cores", 0),
                "os_type": server_stats.get("os_type", ""),
                "disk_type": server_stats.get("disk_type", ""),
                "buffer_pool_size": int(server_variables.get("innodb_buffer_pool_size", 0)),
                "buffer_pool_usage": float(server_status.get("Innodb_buffer_pool_pages_data", 0)) / float(server_status.get("Innodb_buffer_pool_pages_total", 1)),
                "query_cache_hit_ratio": float(server_status.get("Qcache_hits", 0)) / (float(server_status.get("Qcache_hits", 0)) + float(server_status.get("Com_select", 1))),
                "table_open_cache_hit_ratio": 1.0 - float(server_status.get("Opened_tables", 0)) / float(server_status.get("Opened_table_definitions", 1)),
                "connection_usage": float(server_status.get("Max_used_connections", 0)) / float(server_variables.get("max_connections", 1)),
                "slow_queries": int(server_status.get("Slow_queries", 0)),
                "aborted_connections": int(server_status.get("Aborted_connects", 0)),
                "aborted_clients": int(server_status.get("Aborted_clients", 0)),
                "tmp_disk_tables_ratio": float(server_status.get("Created_tmp_disk_tables", 0)) / float(server_status.get("Created_tmp_tables", 1)),
                "handler_read_ratio": float(server_status.get("Handler_read_rnd", 0)) / float(server_status.get("Handler_read_rnd_next", 1)),
                "write_intensive": float(server_status.get("Com_insert", 0) + server_status.get("Com_update", 0) + server_status.get("Com_delete", 0)) > float(server_status.get("Com_select", 0))
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing server settings: {e}")
            return {}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server optimizer metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _get_server_variables(self, conn: Any) -> Dict[str, Any]:
        """Get server variables.
        
        Args:
            conn: The database connection
            
        Returns:
            Dictionary of server variables
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                await cursor.execute("SHOW GLOBAL VARIABLES")
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to a dictionary
                variables = {}
                for row in results:
                    variables[row[0]] = row[1]
                
                # Store the server variables
                self.server_variables = variables
                
                return variables
        except Exception as e:
            logger.error(f"Error getting server variables: {e}")
            return {}
    
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
                
                # Store the server status
                self.server_status = status
                
                return status
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
            return {}
    
    async def _get_server_stats(self, conn: Any) -> Dict[str, Any]:
        """Get server statistics.
        
        Args:
            conn: The database connection
            
        Returns:
            Dictionary of server statistics
        """
        try:
            # Check if server statistics collection is enabled
            if not self.config.collect_server_stats:
                return {}
            
            # Get server memory
            server_memory = 0
            os_type = "Unknown"
            disk_type = "Unknown"
            server_cpu_cores = 0
            
            # Execute the query to get server information
            async with conn.cursor() as cursor:
                # Get server version
                await cursor.execute("SELECT VERSION()")
                version_result = await cursor.fetchone()
                server_version = version_result[0]
                
                # Try to get server memory and CPU cores
                try:
                    # Check if the server is Linux
                    if "linux" in server_version.lower():
                        os_type = "Linux"
                        
                        # Get server memory
                        await cursor.execute("SELECT @@innodb_buffer_pool_size")
                        buffer_pool_result = await cursor.fetchone()
                        buffer_pool_size = int(buffer_pool_result[0])
                        
                        # Estimate server memory (buffer pool is usually 70-80% of server memory)
                        server_memory = int(buffer_pool_size / 0.7)
                        
                        # Get CPU cores
                        await cursor.execute("SELECT @@innodb_buffer_pool_instances")
                        instances_result = await cursor.fetchone()
                        instances = int(instances_result[0])
                        
                        # Estimate CPU cores (buffer pool instances is usually 1 per core)
                        server_cpu_cores = max(instances, 1)
                        
                        # Try to determine disk type
                        await cursor.execute("SELECT @@innodb_io_capacity")
                        io_capacity_result = await cursor.fetchone()
                        io_capacity = int(io_capacity_result[0])
                        
                        # IO capacity > 1000 usually indicates SSD
                        disk_type = "SSD" if io_capacity > 1000 else "HDD"
                    elif "win" in server_version.lower():
                        os_type = "Windows"
                        
                        # Similar estimates for Windows
                        await cursor.execute("SELECT @@innodb_buffer_pool_size")
                        buffer_pool_result = await cursor.fetchone()
                        buffer_pool_size = int(buffer_pool_result[0])
                        
                        server_memory = int(buffer_pool_size / 0.7)
                        
                        await cursor.execute("SELECT @@innodb_buffer_pool_instances")
                        instances_result = await cursor.fetchone()
                        instances = int(instances_result[0])
                        
                        server_cpu_cores = max(instances, 1)
                        
                        await cursor.execute("SELECT @@innodb_io_capacity")
                        io_capacity_result = await cursor.fetchone()
                        io_capacity = int(io_capacity_result[0])
                        
                        disk_type = "SSD" if io_capacity > 1000 else "HDD"
                except Exception:
                    # Fallback to defaults
                    pass
            
            # Create the server stats
            stats = {
                "server_memory": server_memory,
                "os_type": os_type,
                "disk_type": disk_type,
                "server_cpu_cores": server_cpu_cores
            }
            
            # Store the server stats
            self.server_stats = stats
            
            return stats
        except Exception as e:
            logger.error(f"Error getting server statistics: {e}")
            return {}
    
    async def _apply_settings(self, conn: Any, recommendations: List[Dict[str, Any]]) -> int:
        """Apply server settings based on recommendations.
        
        Args:
            conn: The database connection
            recommendations: The server setting recommendations
            
        Returns:
            Number of settings applied
        """
        applied_settings = 0
        
        for recommendation in recommendations:
            try:
                # Execute the SQL statement
                async with conn.cursor() as cursor:
                    await cursor.execute(recommendation.get("sql"))
                
                applied_settings += 1
                logger.info(f"Applied server setting: {recommendation.get('sql')}")
            except Exception as e:
                logger.error(f"Error applying server setting: {e}")
        
        return applied_settings
