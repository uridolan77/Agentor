"""
Trigger configuration for the Agentor framework.

This module provides configuration classes for database trigger management.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class TriggerConfig(BaseModel):
    """Configuration for database trigger management."""
    
    # Trigger settings
    cache_metadata: bool = Field(True, description="Whether to cache trigger metadata")
    cache_ttl: float = Field(3600.0, description="TTL for cached metadata in seconds")
    
    # Logging settings
    log_operations: bool = Field(False, description="Whether to log trigger operations")
    
    # Security settings
    allow_create: bool = Field(True, description="Whether to allow creating triggers")
    allow_drop: bool = Field(True, description="Whether to allow dropping triggers")
    
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
