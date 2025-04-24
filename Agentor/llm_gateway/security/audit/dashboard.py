"""
Security dashboards for the LLM Gateway.

This module provides security dashboards for monitoring and analyzing security
events and metrics in the LLM Gateway.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import asyncio

from agentor.llm_gateway.security.audit.logging import (
    LogLevel,
    LogCategory,
    LogEvent,
    SecureLogger
)
from agentor.llm_gateway.security.audit.trail import (
    AuditAction,
    AuditResource,
    AuditOutcome,
    AuditEvent,
    AuditTrail
)

logger = logging.getLogger(__name__)


class TimeRange(Enum):
    """Time ranges for dashboard metrics."""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"


class MetricAggregation(Enum):
    """Aggregation methods for dashboard metrics."""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    DISTINCT = "distinct"


class DashboardMetric:
    """Metric for security dashboards."""
    
    def __init__(
        self,
        name: str,
        description: str,
        value: Union[int, float, str, List[Any], Dict[str, Any]],
        time_range: TimeRange,
        aggregation: MetricAggregation,
        category: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dashboard metric.
        
        Args:
            name: Metric name
            description: Metric description
            value: Metric value
            time_range: Time range for the metric
            aggregation: Aggregation method for the metric
            category: Metric category
            timestamp: Metric timestamp
            metadata: Additional metadata
        """
        self.name = name
        self.description = description
        self.value = value
        self.time_range = time_range
        self.aggregation = aggregation
        self.category = category
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "time_range": self.time_range.value,
            "aggregation": self.aggregation.value,
            "category": self.category,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class SecurityDashboard:
    """Dashboard for security monitoring."""
    
    def __init__(
        self,
        name: str,
        description: str,
        secure_logger: Optional[SecureLogger] = None,
        audit_trail: Optional[AuditTrail] = None,
        refresh_interval: int = 60,  # seconds
        max_cache_age: int = 300,  # seconds
        metrics_config: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize security dashboard.
        
        Args:
            name: Dashboard name
            description: Dashboard description
            secure_logger: Secure logger
            audit_trail: Audit trail
            refresh_interval: Refresh interval in seconds
            max_cache_age: Maximum cache age in seconds
            metrics_config: Configuration for dashboard metrics
        """
        self.name = name
        self.description = description
        self.secure_logger = secure_logger
        self.audit_trail = audit_trail
        self.refresh_interval = refresh_interval
        self.max_cache_age = max_cache_age
        self.metrics_config = metrics_config or []
        
        # Cached metrics
        self.metrics_cache = {}
        self.last_refresh = 0
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def refresh_metrics(self, force: bool = False) -> None:
        """
        Refresh dashboard metrics.
        
        Args:
            force: Whether to force refresh
        """
        # Check if refresh is needed
        current_time = time.time()
        if not force and current_time - self.last_refresh < self.refresh_interval:
            return
        
        # Acquire lock
        async with self.lock:
            # Check again after acquiring lock
            if not force and current_time - self.last_refresh < self.refresh_interval:
                return
            
            # Clear cache
            self.metrics_cache = {}
            
            # Refresh metrics
            for config in self.metrics_config:
                try:
                    # Get metric
                    metric = await self._calculate_metric(config)
                    
                    # Add to cache
                    if metric:
                        self.metrics_cache[metric.name] = metric
                
                except Exception as e:
                    logger.error(f"Error calculating metric {config.get('name', 'unknown')}: {e}")
            
            # Update last refresh time
            self.last_refresh = current_time
    
    async def _calculate_metric(self, config: Dict[str, Any]) -> Optional[DashboardMetric]:
        """
        Calculate a dashboard metric.
        
        Args:
            config: Metric configuration
            
        Returns:
            Dashboard metric or None if calculation failed
        """
        # Get metric parameters
        name = config.get("name")
        description = config.get("description", "")
        time_range = TimeRange(config.get("time_range", "last_day"))
        aggregation = MetricAggregation(config.get("aggregation", "count"))
        category = config.get("category", "security")
        source = config.get("source", "audit")  # "audit" or "log"
        filters = config.get("filters", {})
        
        if not name:
            logger.warning("Metric configuration missing name")
            return None
        
        # Calculate time range
        start_time = None
        end_time = time.time()
        
        if time_range == TimeRange.LAST_HOUR:
            start_time = end_time - 3600
        elif time_range == TimeRange.LAST_DAY:
            start_time = end_time - 86400
        elif time_range == TimeRange.LAST_WEEK:
            start_time = end_time - 604800
        elif time_range == TimeRange.LAST_MONTH:
            start_time = end_time - 2592000
        elif time_range == TimeRange.CUSTOM:
            start_time = config.get("start_time")
            end_time = config.get("end_time", end_time)
        
        # Get data based on source
        data = []
        
        if source == "audit" and self.audit_trail:
            # Get audit events
            events = await self.audit_trail.get_events(
                start_time=start_time,
                end_time=end_time,
                user_id=filters.get("user_id"),
                action=AuditAction(filters["action"]) if "action" in filters else None,
                resource=AuditResource(filters["resource"]) if "resource" in filters else None,
                outcome=AuditOutcome(filters["outcome"]) if "outcome" in filters else None,
                limit=10000
            )
            
            data = events
        
        elif source == "log" and self.secure_logger:
            # This is a simplified example
            # In a real implementation, you would query logs from the secure logger
            pass
        
        # Calculate metric value based on aggregation
        value = None
        
        if aggregation == MetricAggregation.COUNT:
            value = len(data)
        
        elif aggregation == MetricAggregation.DISTINCT:
            # Get distinct values for a field
            field = config.get("field")
            if field:
                distinct_values = set()
                for item in data:
                    if hasattr(item, field):
                        distinct_values.add(getattr(item, field))
                    elif isinstance(item, dict) and field in item:
                        distinct_values.add(item[field])
                
                value = len(distinct_values)
            else:
                value = 0
        
        elif aggregation in [MetricAggregation.SUM, MetricAggregation.AVERAGE, MetricAggregation.MIN, MetricAggregation.MAX]:
            # Get values for a field
            field = config.get("field")
            if field:
                values = []
                for item in data:
                    if hasattr(item, field):
                        field_value = getattr(item, field)
                        if isinstance(field_value, (int, float)):
                            values.append(field_value)
                    elif isinstance(item, dict) and field in item:
                        field_value = item[field]
                        if isinstance(field_value, (int, float)):
                            values.append(field_value)
                
                if values:
                    if aggregation == MetricAggregation.SUM:
                        value = sum(values)
                    elif aggregation == MetricAggregation.AVERAGE:
                        value = sum(values) / len(values)
                    elif aggregation == MetricAggregation.MIN:
                        value = min(values)
                    elif aggregation == MetricAggregation.MAX:
                        value = max(values)
                else:
                    value = 0
            else:
                value = 0
        
        # Create metric
        return DashboardMetric(
            name=name,
            description=description,
            value=value,
            time_range=time_range,
            aggregation=aggregation,
            category=category,
            metadata={
                "config": config,
                "data_count": len(data)
            }
        )
    
    async def get_metrics(
        self,
        category: Optional[str] = None,
        refresh: bool = False
    ) -> List[DashboardMetric]:
        """
        Get dashboard metrics.
        
        Args:
            category: Category filter
            refresh: Whether to refresh metrics
            
        Returns:
            List of dashboard metrics
        """
        # Refresh metrics if needed
        if refresh or time.time() - self.last_refresh > self.max_cache_age:
            await self.refresh_metrics(force=refresh)
        
        # Get metrics
        async with self.lock:
            metrics = list(self.metrics_cache.values())
        
        # Filter by category
        if category:
            metrics = [m for m in metrics if m.category == category]
        
        return metrics
    
    async def get_metric(self, name: str, refresh: bool = False) -> Optional[DashboardMetric]:
        """
        Get a dashboard metric by name.
        
        Args:
            name: Metric name
            refresh: Whether to refresh metrics
            
        Returns:
            Dashboard metric or None if not found
        """
        # Refresh metrics if needed
        if refresh or time.time() - self.last_refresh > self.max_cache_age:
            await self.refresh_metrics(force=refresh)
        
        # Get metric
        async with self.lock:
            return self.metrics_cache.get(name)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dashboard to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "refresh_interval": self.refresh_interval,
            "max_cache_age": self.max_cache_age,
            "metrics_config": self.metrics_config,
            "last_refresh": self.last_refresh
        }


class DashboardManager:
    """Manager for security dashboards."""
    
    def __init__(
        self,
        secure_logger: Optional[SecureLogger] = None,
        audit_trail: Optional[AuditTrail] = None,
        dashboards: Optional[List[SecurityDashboard]] = None
    ):
        """
        Initialize dashboard manager.
        
        Args:
            secure_logger: Secure logger
            audit_trail: Audit trail
            dashboards: List of security dashboards
        """
        self.secure_logger = secure_logger
        self.audit_trail = audit_trail
        self.dashboards = dashboards or []
        
        # Create default dashboards if none provided
        if not self.dashboards:
            self._create_default_dashboards()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    def _create_default_dashboards(self) -> None:
        """Create default security dashboards."""
        # Security overview dashboard
        security_overview = SecurityDashboard(
            name="security_overview",
            description="Security overview dashboard",
            secure_logger=self.secure_logger,
            audit_trail=self.audit_trail,
            metrics_config=[
                {
                    "name": "total_security_events",
                    "description": "Total security events",
                    "time_range": "last_day",
                    "aggregation": "count",
                    "category": "security",
                    "source": "audit",
                    "filters": {
                        "action": "authenticate"
                    }
                },
                {
                    "name": "failed_logins",
                    "description": "Failed login attempts",
                    "time_range": "last_day",
                    "aggregation": "count",
                    "category": "security",
                    "source": "audit",
                    "filters": {
                        "action": "login",
                        "outcome": "failure"
                    }
                },
                {
                    "name": "blocked_requests",
                    "description": "Blocked requests",
                    "time_range": "last_day",
                    "aggregation": "count",
                    "category": "security",
                    "source": "audit",
                    "filters": {
                        "action": "block"
                    }
                }
            ]
        )
        
        # User activity dashboard
        user_activity = SecurityDashboard(
            name="user_activity",
            description="User activity dashboard",
            secure_logger=self.secure_logger,
            audit_trail=self.audit_trail,
            metrics_config=[
                {
                    "name": "active_users",
                    "description": "Active users",
                    "time_range": "last_day",
                    "aggregation": "distinct",
                    "category": "user",
                    "source": "audit",
                    "field": "user_id"
                },
                {
                    "name": "total_logins",
                    "description": "Total login attempts",
                    "time_range": "last_day",
                    "aggregation": "count",
                    "category": "user",
                    "source": "audit",
                    "filters": {
                        "action": "login"
                    }
                },
                {
                    "name": "content_generation",
                    "description": "Content generation requests",
                    "time_range": "last_day",
                    "aggregation": "count",
                    "category": "user",
                    "source": "audit",
                    "filters": {
                        "action": "generate"
                    }
                }
            ]
        )
        
        # Add dashboards
        self.dashboards.append(security_overview)
        self.dashboards.append(user_activity)
    
    async def get_dashboard(self, name: str) -> Optional[SecurityDashboard]:
        """
        Get a dashboard by name.
        
        Args:
            name: Dashboard name
            
        Returns:
            Security dashboard or None if not found
        """
        for dashboard in self.dashboards:
            if dashboard.name == name:
                return dashboard
        
        return None
    
    async def get_dashboards(self) -> List[SecurityDashboard]:
        """
        Get all dashboards.
        
        Returns:
            List of security dashboards
        """
        return self.dashboards.copy()
    
    async def add_dashboard(self, dashboard: SecurityDashboard) -> None:
        """
        Add a dashboard.
        
        Args:
            dashboard: Security dashboard
        """
        async with self.lock:
            # Check if dashboard with same name already exists
            for i, existing in enumerate(self.dashboards):
                if existing.name == dashboard.name:
                    # Replace existing dashboard
                    self.dashboards[i] = dashboard
                    return
            
            # Add new dashboard
            self.dashboards.append(dashboard)
    
    async def remove_dashboard(self, name: str) -> bool:
        """
        Remove a dashboard.
        
        Args:
            name: Dashboard name
            
        Returns:
            True if dashboard was removed, False if not found
        """
        async with self.lock:
            for i, dashboard in enumerate(self.dashboards):
                if dashboard.name == name:
                    del self.dashboards[i]
                    return True
            
            return False
    
    async def refresh_all_dashboards(self) -> None:
        """Refresh all dashboards."""
        for dashboard in self.dashboards:
            await dashboard.refresh_metrics(force=True)
    
    async def get_all_metrics(
        self,
        category: Optional[str] = None,
        refresh: bool = False
    ) -> Dict[str, List[DashboardMetric]]:
        """
        Get metrics from all dashboards.
        
        Args:
            category: Category filter
            refresh: Whether to refresh metrics
            
        Returns:
            Dictionary mapping dashboard names to lists of metrics
        """
        result = {}
        
        for dashboard in self.dashboards:
            metrics = await dashboard.get_metrics(category=category, refresh=refresh)
            result[dashboard.name] = metrics
        
        return result
