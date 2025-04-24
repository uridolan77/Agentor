"""
User-defined function configuration for the Agentor framework.

This module provides configuration classes for database user-defined function management.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class FunctionConfig(BaseModel):
    """Configuration for database user-defined function management."""
    
    # Function settings
    cache_metadata: bool = Field(True, description="Whether to cache function metadata")
    cache_ttl: float = Field(3600.0, description="TTL for cached metadata in seconds")
    
    # Execution settings
    timeout: float = Field(30.0, description="Timeout for function execution in seconds")
    
    # Logging settings
    log_calls: bool = Field(False, description="Whether to log function calls")
    log_results: bool = Field(False, description="Whether to log function results")
    
    # Security settings
    allow_create: bool = Field(True, description="Whether to allow creating functions")
    allow_alter: bool = Field(True, description="Whether to allow altering functions")
    allow_drop: bool = Field(True, description="Whether to allow dropping functions")
    
    # Metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("timeout")
    def validate_timeout(cls, v):
        """Validate that the timeout is positive."""
        if v <= 0:
            raise ValueError(f"Timeout ({v}) must be positive")
        
        return v
    
    @validator("cache_ttl")
    def validate_cache_ttl(cls, v):
        """Validate that the cache TTL is positive."""
        if v <= 0:
            raise ValueError(f"Cache TTL ({v}) must be positive")
        
        return v
