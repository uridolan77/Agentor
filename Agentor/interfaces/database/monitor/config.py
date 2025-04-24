"""
Performance monitoring configuration for the Agentor framework.

This module provides configuration classes for database performance monitoring.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable, Pattern
import re
from pydantic import BaseModel, Field, validator


class MonitoringLevel(str, Enum):
    """Monitoring levels."""
    
    NONE = "none"  # No monitoring
    BASIC = "basic"  # Basic monitoring
    DETAILED = "detailed"  # Detailed monitoring
    FULL = "full"  # Full monitoring
    CUSTOM = "custom"  # Custom monitoring level


class MonitoringConfig(BaseModel):
    """Configuration for database performance monitoring."""
    
    # Monitoring settings
    enabled: bool = Field(True, description="Whether monitoring is enabled")
    level: MonitoringLevel = Field(MonitoringLevel.BASIC, description="Monitoring level")
    
    # Slow query settings
    slow_query_threshold: float = Field(1.0, description="Threshold for slow queries in seconds")
    very_slow_query_threshold: float = Field(10.0, description="Threshold for very slow queries in seconds")
    
    # Sampling settings
    sample_rate: float = Field(1.0, description="Sampling rate for queries (0.0-1.0)")
    min_sample_rate: float = Field(0.01, description="Minimum sampling rate for queries (0.0-1.0)")
    
    # Logging settings
    log_slow_queries: bool = Field(True, description="Whether to log slow queries")
    log_query_plans: bool = Field(False, description="Whether to log query plans")
    log_query_stats: bool = Field(False, description="Whether to log query statistics")
    
    # Alert settings
    alert_on_slow_queries: bool = Field(False, description="Whether to alert on slow queries")
    alert_threshold: float = Field(5.0, description="Threshold for alerting in seconds")
    
    # Collection settings
    collect_query_plans: bool = Field(False, description="Whether to collect query plans")
    collect_table_stats: bool = Field(False, description="Whether to collect table statistics")
    collect_index_stats: bool = Field(False, description="Whether to collect index statistics")
    
    # Metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("sample_rate")
    def validate_sample_rate(cls, v):
        """Validate that the sample rate is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Sample rate ({v}) must be between 0 and 1")
        
        return v
    
    @validator("min_sample_rate")
    def validate_min_sample_rate(cls, v, values):
        """Validate that the minimum sample rate is between 0 and 1 and less than or equal to the sample rate."""
        if v < 0 or v > 1:
            raise ValueError(f"Minimum sample rate ({v}) must be between 0 and 1")
        
        sample_rate = values.get("sample_rate")
        if sample_rate is not None and v > sample_rate:
            raise ValueError(f"Minimum sample rate ({v}) must be less than or equal to sample rate ({sample_rate})")
        
        return v
    
    def should_monitor_query(self, query: str) -> bool:
        """Check if a query should be monitored based on the configuration.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query should be monitored, False otherwise
        """
        # Check if monitoring is enabled
        if not self.enabled:
            return False
        
        # Check the monitoring level
        if self.level == MonitoringLevel.NONE:
            return False
        
        # Get the custom monitoring function
        custom_func = self.additional_settings.get("should_monitor_func")
        if custom_func:
            return custom_func(query)
        
        # Default to True for all queries
        return True
    
    def should_sample_query(self, query: str) -> bool:
        """Check if a query should be sampled based on the configuration.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query should be sampled, False otherwise
        """
        # Check if monitoring is enabled
        if not self.enabled:
            return False
        
        # Check the monitoring level
        if self.level == MonitoringLevel.NONE:
            return False
        
        # Get the custom sampling function
        custom_func = self.additional_settings.get("should_sample_func")
        if custom_func:
            return custom_func(query)
        
        # Sample based on the sample rate
        import random
        return random.random() < self.sample_rate
    
    def should_collect_query_plan(self, query: str) -> bool:
        """Check if a query plan should be collected based on the configuration.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query plan should be collected, False otherwise
        """
        # Check if monitoring is enabled
        if not self.enabled:
            return False
        
        # Check if query plan collection is enabled
        if not self.collect_query_plans:
            return False
        
        # Check the monitoring level
        if self.level == MonitoringLevel.NONE or self.level == MonitoringLevel.BASIC:
            return False
        
        # Get the custom collection function
        custom_func = self.additional_settings.get("should_collect_query_plan_func")
        if custom_func:
            return custom_func(query)
        
        # Default to True for all queries
        return True
    
    def should_alert_on_query(self, query: str, execution_time: float) -> bool:
        """Check if an alert should be triggered for a query based on the configuration.
        
        Args:
            query: The query to check
            execution_time: The execution time of the query in seconds
            
        Returns:
            True if an alert should be triggered, False otherwise
        """
        # Check if monitoring is enabled
        if not self.enabled:
            return False
        
        # Check if alerting is enabled
        if not self.alert_on_slow_queries:
            return False
        
        # Check if the query is slow enough to trigger an alert
        if execution_time < self.alert_threshold:
            return False
        
        # Get the custom alerting function
        custom_func = self.additional_settings.get("should_alert_func")
        if custom_func:
            return custom_func(query, execution_time)
        
        # Default to True for all slow queries
        return True
