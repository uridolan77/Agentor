"""
Stored procedure configuration for the Agentor framework.

This module provides configuration classes for database stored procedure management.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class ProcedureConfig(BaseModel):
    """Configuration for database stored procedure management."""
    
    # Procedure settings
    cache_metadata: bool = Field(True, description="Whether to cache procedure metadata")
    cache_ttl: float = Field(3600.0, description="TTL for cached metadata in seconds")
    
    # Execution settings
    timeout: float = Field(30.0, description="Timeout for procedure execution in seconds")
    max_rows: int = Field(10000, description="Maximum number of rows to return")
    
    # Logging settings
    log_calls: bool = Field(False, description="Whether to log procedure calls")
    log_results: bool = Field(False, description="Whether to log procedure results")
    
    # Security settings
    allow_create: bool = Field(True, description="Whether to allow creating procedures")
    allow_alter: bool = Field(True, description="Whether to allow altering procedures")
    allow_drop: bool = Field(True, description="Whether to allow dropping procedures")
    
    # Metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("max_rows")
    def validate_max_rows(cls, v):
        """Validate that the maximum rows is positive."""
        if v <= 0:
            raise ValueError(f"Maximum rows ({v}) must be positive")
        
        return v
    
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
