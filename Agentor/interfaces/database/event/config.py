"""
Event scheduling configuration for the Agentor framework.

This module provides configuration classes for database event scheduling.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class EventConfig(BaseModel):
    """Configuration for database event scheduling."""
    
    # Event settings
    cache_metadata: bool = Field(True, description="Whether to cache event metadata")
    cache_ttl: float = Field(3600.0, description="TTL for cached metadata in seconds")
    
    # Logging settings
    log_operations: bool = Field(False, description="Whether to log event operations")
    
    # Security settings
    allow_create: bool = Field(True, description="Whether to allow creating events")
    allow_alter: bool = Field(True, description="Whether to allow altering events")
    allow_drop: bool = Field(True, description="Whether to allow dropping events")
    
    # Metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("cache_ttl")
    def validate_cache_ttl(cls, v):
        """Validate that the cache TTL is positive."""
        if v <= 0:
            raise ValueError(f"Cache TTL ({v}) must be positive")
        
        return v
