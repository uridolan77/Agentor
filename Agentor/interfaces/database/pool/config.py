"""
Connection pool configuration for the Agentor framework.

This module provides configuration classes for database connection pools.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class ConnectionValidationMode(str, Enum):
    """Connection validation modes."""
    
    NONE = "none"  # No validation
    PING = "ping"  # Ping the server
    QUERY = "query"  # Execute a query
    CUSTOM = "custom"  # Custom validation function


class ConnectionPoolConfig(BaseModel):
    """Configuration for database connection pools."""
    
    # Pool size settings
    min_size: int = Field(1, description="Minimum number of connections in the pool")
    max_size: int = Field(10, description="Maximum number of connections in the pool")
    target_size: Optional[int] = Field(None, description="Target number of connections in the pool")
    
    # Connection settings
    max_idle_time: float = Field(300.0, description="Maximum idle time in seconds")
    max_lifetime: float = Field(3600.0, description="Maximum lifetime in seconds")
    connect_timeout: float = Field(10.0, description="Connection timeout in seconds")
    
    # Validation settings
    validation_mode: ConnectionValidationMode = Field(ConnectionValidationMode.PING, description="Connection validation mode")
    validation_query: Optional[str] = Field(None, description="Query to use for validation")
    validation_interval: float = Field(60.0, description="Validation interval in seconds")
    
    # Retry settings
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retry attempts in seconds")
    
    # Health check settings
    health_check_interval: float = Field(60.0, description="Health check interval in seconds")
    health_check_timeout: float = Field(5.0, description="Health check timeout in seconds")
    
    # Scaling settings
    scale_up_threshold: float = Field(0.8, description="Threshold for scaling up the pool")
    scale_down_threshold: float = Field(0.2, description="Threshold for scaling down the pool")
    scale_up_step: int = Field(2, description="Number of connections to add when scaling up")
    scale_down_step: int = Field(1, description="Number of connections to remove when scaling down")
    
    # Metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("target_size", always=True)
    def validate_target_size(cls, v, values):
        """Validate that the target size is between min_size and max_size."""
        if v is None:
            # Default to the average of min_size and max_size
            min_size = values.get("min_size", 1)
            max_size = values.get("max_size", 10)
            return (min_size + max_size) // 2
        
        min_size = values.get("min_size", 1)
        max_size = values.get("max_size", 10)
        
        if v < min_size:
            raise ValueError(f"Target size ({v}) must be greater than or equal to min_size ({min_size})")
        
        if v > max_size:
            raise ValueError(f"Target size ({v}) must be less than or equal to max_size ({max_size})")
        
        return v
    
    @validator("validation_query")
    def validate_validation_query(cls, v, values):
        """Validate that the validation query is provided when validation mode is QUERY."""
        validation_mode = values.get("validation_mode")
        
        if validation_mode == ConnectionValidationMode.QUERY and not v:
            raise ValueError("Validation query must be provided when validation mode is QUERY")
        
        return v
    
    @validator("scale_up_threshold", "scale_down_threshold")
    def validate_thresholds(cls, v):
        """Validate that the thresholds are between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError(f"Threshold ({v}) must be between 0 and 1")
        
        return v
    
    @validator("scale_down_threshold")
    def validate_scale_down_threshold(cls, v, values):
        """Validate that the scale down threshold is less than the scale up threshold."""
        scale_up_threshold = values.get("scale_up_threshold")
        
        if scale_up_threshold is not None and v >= scale_up_threshold:
            raise ValueError(f"Scale down threshold ({v}) must be less than scale up threshold ({scale_up_threshold})")
        
        return v
