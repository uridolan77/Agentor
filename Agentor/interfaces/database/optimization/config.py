"""
Optimization configuration for the Agentor framework.

This module provides configuration classes for database optimization.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable, Pattern
import re
from pydantic import BaseModel, Field, validator


class OptimizationLevel(str, Enum):
    """Optimization levels."""
    
    NONE = "none"  # No optimization
    BASIC = "basic"  # Basic optimization
    MODERATE = "moderate"  # Moderate optimization
    AGGRESSIVE = "aggressive"  # Aggressive optimization
    CUSTOM = "custom"  # Custom optimization level


class QueryOptimizationConfig(BaseModel):
    """Configuration for query optimization."""
    
    # Query optimization settings
    enabled: bool = Field(True, description="Whether query optimization is enabled")
    level: OptimizationLevel = Field(OptimizationLevel.MODERATE, description="Optimization level")
    
    # Query rewriting settings
    rewrite_queries: bool = Field(True, description="Whether to rewrite queries")
    add_missing_indexes: bool = Field(True, description="Whether to add missing indexes to queries")
    optimize_joins: bool = Field(True, description="Whether to optimize joins")
    optimize_where_clauses: bool = Field(True, description="Whether to optimize WHERE clauses")
    optimize_order_by: bool = Field(True, description="Whether to optimize ORDER BY clauses")
    optimize_group_by: bool = Field(True, description="Whether to optimize GROUP BY clauses")
    optimize_limit: bool = Field(True, description="Whether to optimize LIMIT clauses")
    
    # Query analysis settings
    analyze_queries: bool = Field(True, description="Whether to analyze queries")
    slow_query_threshold: float = Field(1.0, description="Threshold for slow queries in seconds")
    very_slow_query_threshold: float = Field(10.0, description="Threshold for very slow queries in seconds")
    
    # Query plan settings
    collect_query_plans: bool = Field(True, description="Whether to collect query plans")
    analyze_query_plans: bool = Field(True, description="Whether to analyze query plans")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("slow_query_threshold")
    def validate_slow_query_threshold(cls, v):
        """Validate that the slow query threshold is positive."""
        if v <= 0:
            raise ValueError(f"Slow query threshold ({v}) must be positive")
        
        return v
    
    @validator("very_slow_query_threshold")
    def validate_very_slow_query_threshold(cls, v, values):
        """Validate that the very slow query threshold is greater than the slow query threshold."""
        slow_query_threshold = values.get("slow_query_threshold")
        if slow_query_threshold is not None and v <= slow_query_threshold:
            raise ValueError(f"Very slow query threshold ({v}) must be greater than slow query threshold ({slow_query_threshold})")
        
        return v


class IndexOptimizationConfig(BaseModel):
    """Configuration for index optimization."""
    
    # Index optimization settings
    enabled: bool = Field(True, description="Whether index optimization is enabled")
    level: OptimizationLevel = Field(OptimizationLevel.MODERATE, description="Optimization level")
    
    # Index analysis settings
    analyze_indexes: bool = Field(True, description="Whether to analyze indexes")
    collect_index_stats: bool = Field(True, description="Whether to collect index statistics")
    
    # Index recommendation settings
    recommend_indexes: bool = Field(True, description="Whether to recommend indexes")
    recommend_composite_indexes: bool = Field(True, description="Whether to recommend composite indexes")
    recommend_covering_indexes: bool = Field(True, description="Whether to recommend covering indexes")
    
    # Index creation settings
    auto_create_indexes: bool = Field(False, description="Whether to automatically create recommended indexes")
    max_indexes_per_table: int = Field(5, description="Maximum number of indexes per table")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("max_indexes_per_table")
    def validate_max_indexes_per_table(cls, v):
        """Validate that the maximum indexes per table is positive."""
        if v <= 0:
            raise ValueError(f"Maximum indexes per table ({v}) must be positive")
        
        return v


class ServerOptimizationConfig(BaseModel):
    """Configuration for server optimization."""
    
    # Server optimization settings
    enabled: bool = Field(True, description="Whether server optimization is enabled")
    level: OptimizationLevel = Field(OptimizationLevel.MODERATE, description="Optimization level")
    
    # Server analysis settings
    analyze_server: bool = Field(True, description="Whether to analyze server configuration")
    collect_server_stats: bool = Field(True, description="Whether to collect server statistics")
    
    # Server recommendation settings
    recommend_server_settings: bool = Field(True, description="Whether to recommend server settings")
    
    # Server configuration settings
    auto_configure_server: bool = Field(False, description="Whether to automatically configure server settings")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")


class PerformanceMonitoringConfig(BaseModel):
    """Configuration for performance monitoring."""
    
    # Performance monitoring settings
    enabled: bool = Field(True, description="Whether performance monitoring is enabled")
    level: OptimizationLevel = Field(OptimizationLevel.MODERATE, description="Monitoring level")
    
    # Monitoring interval settings
    monitoring_interval: float = Field(60.0, description="Monitoring interval in seconds")
    detailed_monitoring_interval: float = Field(300.0, description="Detailed monitoring interval in seconds")
    
    # Metrics collection settings
    collect_query_metrics: bool = Field(True, description="Whether to collect query metrics")
    collect_index_metrics: bool = Field(True, description="Whether to collect index metrics")
    collect_server_metrics: bool = Field(True, description="Whether to collect server metrics")
    
    # Alert settings
    alert_on_slow_queries: bool = Field(True, description="Whether to alert on slow queries")
    alert_on_high_load: bool = Field(True, description="Whether to alert on high server load")
    alert_threshold: float = Field(0.8, description="Threshold for alerting (0.0-1.0)")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("monitoring_interval")
    def validate_monitoring_interval(cls, v):
        """Validate that the monitoring interval is positive."""
        if v <= 0:
            raise ValueError(f"Monitoring interval ({v}) must be positive")
        
        return v
    
    @validator("detailed_monitoring_interval")
    def validate_detailed_monitoring_interval(cls, v, values):
        """Validate that the detailed monitoring interval is greater than the monitoring interval."""
        monitoring_interval = values.get("monitoring_interval")
        if monitoring_interval is not None and v < monitoring_interval:
            raise ValueError(f"Detailed monitoring interval ({v}) must be greater than or equal to monitoring interval ({monitoring_interval})")
        
        return v
    
    @validator("alert_threshold")
    def validate_alert_threshold(cls, v):
        """Validate that the alert threshold is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Alert threshold ({v}) must be between 0 and 1")
        
        return v


class OptimizationConfig(BaseModel):
    """Configuration for database optimization."""
    
    # Optimization settings
    enabled: bool = Field(True, description="Whether optimization is enabled")
    level: OptimizationLevel = Field(OptimizationLevel.MODERATE, description="Optimization level")
    
    # Query optimization settings
    query_optimization: QueryOptimizationConfig = Field(default_factory=QueryOptimizationConfig, description="Query optimization configuration")
    
    # Index optimization settings
    index_optimization: IndexOptimizationConfig = Field(default_factory=IndexOptimizationConfig, description="Index optimization configuration")
    
    # Server optimization settings
    server_optimization: ServerOptimizationConfig = Field(default_factory=ServerOptimizationConfig, description="Server optimization configuration")
    
    # Performance monitoring settings
    performance_monitoring: PerformanceMonitoringConfig = Field(default_factory=PerformanceMonitoringConfig, description="Performance monitoring configuration")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("query_optimization", "index_optimization", "server_optimization", "performance_monitoring", pre=True)
    def validate_sub_configs(cls, v, values):
        """Validate that the sub-configurations have the same level as the main configuration."""
        if isinstance(v, dict) and "level" not in v and "level" in values:
            v["level"] = values["level"]
        
        return v
