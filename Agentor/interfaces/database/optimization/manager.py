"""
Optimization manager for the Agentor framework.

This module provides a manager for database optimization.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Type
import weakref

from .config import OptimizationConfig, OptimizationLevel

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Optimizer type


class OptimizationManager(Generic[T]):
    """Manager for database optimization."""
    
    def __init__(
        self,
        name: str,
        config: OptimizationConfig,
        optimizer_class: Type[T]
    ):
        """Initialize the optimization manager.
        
        Args:
            name: The name of the manager
            config: The optimization configuration
            optimizer_class: The optimizer class
        """
        self.name = name
        self.config = config
        self.optimizer_class = optimizer_class
        
        # Optimizers
        self.query_optimizer: Optional[T] = None
        self.index_optimizer: Optional[T] = None
        self.server_optimizer: Optional[T] = None
        self.performance_monitor: Optional[T] = None
        
        # Manager tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Manager metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "optimized_queries": 0,
            "optimized_indexes": 0,
            "optimized_server_settings": 0,
            "performance_alerts": 0
        }
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
    
    async def initialize(self) -> None:
        """Initialize the optimization manager."""
        logger.info(f"Initializing optimization manager {self.name}")
        
        # Check if optimization is enabled
        if not self.config.enabled:
            logger.info(f"Optimization is disabled for manager {self.name}")
            return
        
        # Initialize query optimizer
        if self.config.query_optimization.enabled:
            self.query_optimizer = self.optimizer_class(
                name=f"{self.name}_query",
                config=self.config.query_optimization
            )
            await self.query_optimizer.initialize()
        
        # Initialize index optimizer
        if self.config.index_optimization.enabled:
            self.index_optimizer = self.optimizer_class(
                name=f"{self.name}_index",
                config=self.config.index_optimization
            )
            await self.index_optimizer.initialize()
        
        # Initialize server optimizer
        if self.config.server_optimization.enabled:
            self.server_optimizer = self.optimizer_class(
                name=f"{self.name}_server",
                config=self.config.server_optimization
            )
            await self.server_optimizer.initialize()
        
        # Initialize performance monitor
        if self.config.performance_monitoring.enabled:
            self.performance_monitor = self.optimizer_class(
                name=f"{self.name}_performance",
                config=self.config.performance_monitoring
            )
            await self.performance_monitor.initialize()
            
            # Start monitoring task
            self.monitoring_task = self.loop.create_task(self._monitoring_loop())
        
        logger.info(f"Optimization manager {self.name} initialized")
    
    async def close(self) -> None:
        """Close the optimization manager."""
        logger.info(f"Closing optimization manager {self.name}")
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        # Close optimizers
        if self.query_optimizer:
            await self.query_optimizer.close()
            self.query_optimizer = None
        
        if self.index_optimizer:
            await self.index_optimizer.close()
            self.index_optimizer = None
        
        if self.server_optimizer:
            await self.server_optimizer.close()
            self.server_optimizer = None
        
        if self.performance_monitor:
            await self.performance_monitor.close()
            self.performance_monitor = None
        
        logger.info(f"Optimization manager {self.name} closed")
    
    async def optimize_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Optimize a query.
        
        Args:
            query: The query to optimize
            params: The query parameters
            
        Returns:
            The optimized query
        """
        # Check if query optimization is enabled
        if not self.config.enabled or not self.config.query_optimization.enabled or not self.query_optimizer:
            return query
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Optimize the query
            optimized_query = await self.query_optimizer.optimize_query(query, params)
            
            # Check if the query was optimized
            if optimized_query != query:
                self.metrics["optimized_queries"] += 1
            
            return optimized_query
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            return query
    
    async def analyze_query(self, query: str, params: Optional[Dict[str, Any]] = None, execution_time: float = 0.0) -> Dict[str, Any]:
        """Analyze a query.
        
        Args:
            query: The query to analyze
            params: The query parameters
            execution_time: The execution time of the query in seconds
            
        Returns:
            Dictionary of analysis results
        """
        # Check if query optimization is enabled
        if not self.config.enabled or not self.config.query_optimization.enabled or not self.query_optimizer:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Analyze the query
            return await self.query_optimizer.analyze_query(query, params, execution_time)
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {}
    
    async def get_query_plan(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the query plan for a query.
        
        Args:
            query: The query to get the plan for
            params: The query parameters
            
        Returns:
            Dictionary of query plan
        """
        # Check if query optimization is enabled
        if not self.config.enabled or not self.config.query_optimization.enabled or not self.query_optimizer:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Get the query plan
            return await self.query_optimizer.get_query_plan(query, params)
        except Exception as e:
            logger.error(f"Error getting query plan: {e}")
            return {}
    
    async def optimize_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Optimize indexes for a table.
        
        Args:
            table_name: The name of the table
            
        Returns:
            List of index recommendations
        """
        # Check if index optimization is enabled
        if not self.config.enabled or not self.config.index_optimization.enabled or not self.index_optimizer:
            return []
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Optimize the indexes
            recommendations = await self.index_optimizer.optimize_indexes(table_name)
            
            # Update metrics
            self.metrics["optimized_indexes"] += len(recommendations)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error optimizing indexes: {e}")
            return []
    
    async def analyze_indexes(self, table_name: str) -> Dict[str, Any]:
        """Analyze indexes for a table.
        
        Args:
            table_name: The name of the table
            
        Returns:
            Dictionary of analysis results
        """
        # Check if index optimization is enabled
        if not self.config.enabled or not self.config.index_optimization.enabled or not self.index_optimizer:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Analyze the indexes
            return await self.index_optimizer.analyze_indexes(table_name)
        except Exception as e:
            logger.error(f"Error analyzing indexes: {e}")
            return {}
    
    async def optimize_server(self) -> List[Dict[str, Any]]:
        """Optimize server settings.
        
        Returns:
            List of server setting recommendations
        """
        # Check if server optimization is enabled
        if not self.config.enabled or not self.config.server_optimization.enabled or not self.server_optimizer:
            return []
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Optimize the server settings
            recommendations = await self.server_optimizer.optimize_server()
            
            # Update metrics
            self.metrics["optimized_server_settings"] += len(recommendations)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error optimizing server settings: {e}")
            return []
    
    async def analyze_server(self) -> Dict[str, Any]:
        """Analyze server settings.
        
        Returns:
            Dictionary of analysis results
        """
        # Check if server optimization is enabled
        if not self.config.enabled or not self.config.server_optimization.enabled or not self.server_optimizer:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Analyze the server settings
            return await self.server_optimizer.analyze_server()
        except Exception as e:
            logger.error(f"Error analyzing server settings: {e}")
            return {}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Check if performance monitoring is enabled
        if not self.config.enabled or not self.config.performance_monitoring.enabled or not self.performance_monitor:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        try:
            # Get the performance metrics
            return await self.performance_monitor.get_performance_metrics()
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get optimization manager metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()
        
        # Add optimizer metrics
        if self.query_optimizer:
            query_metrics = await self.query_optimizer.get_metrics()
            metrics["query_optimizer"] = query_metrics
        
        if self.index_optimizer:
            index_metrics = await self.index_optimizer.get_metrics()
            metrics["index_optimizer"] = index_metrics
        
        if self.server_optimizer:
            server_metrics = await self.server_optimizer.get_metrics()
            metrics["server_optimizer"] = server_metrics
        
        if self.performance_monitor:
            performance_metrics = await self.performance_monitor.get_metrics()
            metrics["performance_monitor"] = performance_metrics
        
        return metrics
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for performance monitoring."""
        try:
            while True:
                # Sleep for the monitoring interval
                await asyncio.sleep(self.config.performance_monitoring.monitoring_interval)
                
                # Check if performance monitoring is enabled
                if not self.config.enabled or not self.config.performance_monitoring.enabled or not self.performance_monitor:
                    continue
                
                try:
                    # Get performance metrics
                    metrics = await self.performance_monitor.get_performance_metrics()
                    
                    # Check for alerts
                    if self.config.performance_monitoring.alert_on_slow_queries:
                        slow_queries = metrics.get("slow_queries", 0)
                        if slow_queries > 0:
                            logger.warning(f"Alert: {slow_queries} slow queries detected")
                            self.metrics["performance_alerts"] += 1
                    
                    if self.config.performance_monitoring.alert_on_high_load:
                        load = metrics.get("load", 0.0)
                        if load > self.config.performance_monitoring.alert_threshold:
                            logger.warning(f"Alert: High server load detected ({load:.2f})")
                            self.metrics["performance_alerts"] += 1
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Unexpected error in monitoring loop: {e}")
    
    def __del__(self):
        """Clean up resources when the manager is garbage collected."""
        # Cancel monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
