"""
Batch processing configuration for the Agentor framework.

This module provides configuration classes for database batch processing.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class BatchStrategy(str, Enum):
    """Batch processing strategies."""
    
    NONE = "none"  # No batching
    SIZE = "size"  # Batch by size
    TIME = "time"  # Batch by time
    HYBRID = "hybrid"  # Batch by size or time, whichever comes first
    CUSTOM = "custom"  # Custom batching strategy


class BatchProcessingConfig(BaseModel):
    """Configuration for database batch processing."""
    
    # Batch settings
    enabled: bool = Field(True, description="Whether batching is enabled")
    strategy: BatchStrategy = Field(BatchStrategy.HYBRID, description="Batch strategy")
    
    # Batch size settings
    batch_size: int = Field(100, description="Maximum number of operations in a batch")
    min_batch_size: int = Field(1, description="Minimum number of operations in a batch")
    
    # Batch time settings
    batch_time: float = Field(1.0, description="Maximum time to wait for a batch in seconds")
    min_batch_time: float = Field(0.01, description="Minimum time to wait for a batch in seconds")
    
    # Batch execution settings
    max_retries: int = Field(3, description="Maximum number of retries for a batch")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    
    # Batch metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate that the batch size is positive."""
        if v <= 0:
            raise ValueError(f"Batch size ({v}) must be positive")
        
        return v
    
    @validator("min_batch_size")
    def validate_min_batch_size(cls, v, values):
        """Validate that the minimum batch size is positive and less than or equal to the batch size."""
        if v <= 0:
            raise ValueError(f"Minimum batch size ({v}) must be positive")
        
        batch_size = values.get("batch_size")
        if batch_size is not None and v > batch_size:
            raise ValueError(f"Minimum batch size ({v}) must be less than or equal to batch size ({batch_size})")
        
        return v
    
    @validator("batch_time")
    def validate_batch_time(cls, v):
        """Validate that the batch time is positive."""
        if v <= 0:
            raise ValueError(f"Batch time ({v}) must be positive")
        
        return v
    
    @validator("min_batch_time")
    def validate_min_batch_time(cls, v, values):
        """Validate that the minimum batch time is positive and less than or equal to the batch time."""
        if v <= 0:
            raise ValueError(f"Minimum batch time ({v}) must be positive")
        
        batch_time = values.get("batch_time")
        if batch_time is not None and v > batch_time:
            raise ValueError(f"Minimum batch time ({v}) must be less than or equal to batch time ({batch_time})")
        
        return v
    
    def should_batch_operation(self, operation: str) -> bool:
        """Check if an operation should be batched based on the configuration.
        
        Args:
            operation: The operation to check
            
        Returns:
            True if the operation should be batched, False otherwise
        """
        # Check if batching is enabled
        if not self.enabled:
            return False
        
        # Check the batch strategy
        if self.strategy == BatchStrategy.NONE:
            return False
        
        # Get the custom batching function
        custom_func = self.additional_settings.get("should_batch_func")
        if custom_func:
            return custom_func(operation)
        
        # Default to True for all operations
        return True
    
    def get_batch_size(self, operation: str) -> int:
        """Get the batch size for an operation based on the configuration.
        
        Args:
            operation: The operation to get the batch size for
            
        Returns:
            The batch size
        """
        # Get the custom batch size function
        custom_func = self.additional_settings.get("get_batch_size_func")
        if custom_func:
            size = custom_func(operation)
            
            # Clamp the size to the min/max range
            return max(self.min_batch_size, min(self.batch_size, size))
        
        # Use the default batch size
        return self.batch_size
    
    def get_batch_time(self, operation: str) -> float:
        """Get the batch time for an operation based on the configuration.
        
        Args:
            operation: The operation to get the batch time for
            
        Returns:
            The batch time in seconds
        """
        # Get the custom batch time function
        custom_func = self.additional_settings.get("get_batch_time_func")
        if custom_func:
            time = custom_func(operation)
            
            # Clamp the time to the min/max range
            return max(self.min_batch_time, min(self.batch_time, time))
        
        # Use the default batch time
        return self.batch_time
