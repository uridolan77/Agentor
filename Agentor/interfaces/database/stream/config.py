"""
Streaming configuration for the Agentor framework.

This module provides configuration classes for database streaming.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class StreamStrategy(str, Enum):
    """Streaming strategies."""
    
    NONE = "none"  # No streaming
    CHUNK = "chunk"  # Stream by chunks
    ROW = "row"  # Stream by rows
    CUSTOM = "custom"  # Custom streaming strategy


class StreamingConfig(BaseModel):
    """Configuration for database streaming."""
    
    # Stream settings
    enabled: bool = Field(True, description="Whether streaming is enabled")
    strategy: StreamStrategy = Field(StreamStrategy.CHUNK, description="Stream strategy")
    
    # Chunk settings
    chunk_size: int = Field(1000, description="Number of rows in a chunk")
    min_chunk_size: int = Field(1, description="Minimum number of rows in a chunk")
    
    # Buffer settings
    buffer_size: int = Field(10, description="Number of chunks to buffer")
    min_buffer_size: int = Field(1, description="Minimum number of chunks to buffer")
    
    # Timeout settings
    chunk_timeout: float = Field(10.0, description="Timeout for getting a chunk in seconds")
    query_timeout: float = Field(300.0, description="Timeout for the entire query in seconds")
    
    # Metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        """Validate that the chunk size is positive."""
        if v <= 0:
            raise ValueError(f"Chunk size ({v}) must be positive")
        
        return v
    
    @validator("min_chunk_size")
    def validate_min_chunk_size(cls, v, values):
        """Validate that the minimum chunk size is positive and less than or equal to the chunk size."""
        if v <= 0:
            raise ValueError(f"Minimum chunk size ({v}) must be positive")
        
        chunk_size = values.get("chunk_size")
        if chunk_size is not None and v > chunk_size:
            raise ValueError(f"Minimum chunk size ({v}) must be less than or equal to chunk size ({chunk_size})")
        
        return v
    
    @validator("buffer_size")
    def validate_buffer_size(cls, v):
        """Validate that the buffer size is positive."""
        if v <= 0:
            raise ValueError(f"Buffer size ({v}) must be positive")
        
        return v
    
    @validator("min_buffer_size")
    def validate_min_buffer_size(cls, v, values):
        """Validate that the minimum buffer size is positive and less than or equal to the buffer size."""
        if v <= 0:
            raise ValueError(f"Minimum buffer size ({v}) must be positive")
        
        buffer_size = values.get("buffer_size")
        if buffer_size is not None and v > buffer_size:
            raise ValueError(f"Minimum buffer size ({v}) must be less than or equal to buffer size ({buffer_size})")
        
        return v
    
    def should_stream_query(self, query: str) -> bool:
        """Check if a query should be streamed based on the configuration.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query should be streamed, False otherwise
        """
        # Check if streaming is enabled
        if not self.enabled:
            return False
        
        # Check the stream strategy
        if self.strategy == StreamStrategy.NONE:
            return False
        
        # Get the custom streaming function
        custom_func = self.additional_settings.get("should_stream_func")
        if custom_func:
            return custom_func(query)
        
        # Check if the query is a SELECT query
        return query.strip().upper().startswith("SELECT")
    
    def get_chunk_size(self, query: str) -> int:
        """Get the chunk size for a query based on the configuration.
        
        Args:
            query: The query to get the chunk size for
            
        Returns:
            The chunk size
        """
        # Get the custom chunk size function
        custom_func = self.additional_settings.get("get_chunk_size_func")
        if custom_func:
            size = custom_func(query)
            
            # Clamp the size to the min/max range
            return max(self.min_chunk_size, min(self.chunk_size, size))
        
        # Use the default chunk size
        return self.chunk_size
    
    def get_buffer_size(self, query: str) -> int:
        """Get the buffer size for a query based on the configuration.
        
        Args:
            query: The query to get the buffer size for
            
        Returns:
            The buffer size
        """
        # Get the custom buffer size function
        custom_func = self.additional_settings.get("get_buffer_size_func")
        if custom_func:
            size = custom_func(query)
            
            # Clamp the size to the min/max range
            return max(self.min_buffer_size, min(self.buffer_size, size))
        
        # Use the default buffer size
        return self.buffer_size
