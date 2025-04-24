"""
Query cache configuration for the Agentor framework.

This module provides configuration classes for database query caching.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable, Pattern
import re
from pydantic import BaseModel, Field, validator


class CacheStrategy(str, Enum):
    """Cache strategies."""
    
    NONE = "none"  # No caching
    ALL = "all"  # Cache all queries
    SELECT_ONLY = "select_only"  # Cache only SELECT queries
    PATTERN = "pattern"  # Cache queries matching a pattern
    CUSTOM = "custom"  # Custom caching strategy


class QueryCacheConfig(BaseModel):
    """Configuration for database query caching."""
    
    # Cache settings
    enabled: bool = Field(True, description="Whether caching is enabled")
    strategy: CacheStrategy = Field(CacheStrategy.SELECT_ONLY, description="Cache strategy")
    
    # Cache size settings
    max_size: int = Field(1000, description="Maximum number of cached queries")
    max_memory: int = Field(100 * 1024 * 1024, description="Maximum memory usage in bytes")
    
    # Cache TTL settings
    default_ttl: float = Field(300.0, description="Default TTL for cached queries in seconds")
    min_ttl: float = Field(1.0, description="Minimum TTL for cached queries in seconds")
    max_ttl: float = Field(3600.0, description="Maximum TTL for cached queries in seconds")
    
    # Cache invalidation settings
    invalidate_on_write: bool = Field(True, description="Whether to invalidate cache on write queries")
    invalidate_tables: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Tables to invalidate when specific tables are modified"
    )
    
    # Cache patterns
    include_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns of queries to include in caching"
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns of queries to exclude from caching"
    )
    
    # Cache metrics settings
    collect_metrics: bool = Field(True, description="Whether to collect metrics")
    metrics_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("include_patterns", "exclude_patterns")
    def validate_patterns(cls, v):
        """Validate that the patterns are valid regular expressions."""
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError(f"Invalid regular expression: {pattern}")
        
        return v
    
    def should_cache_query(self, query: str) -> bool:
        """Check if a query should be cached based on the configuration.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query should be cached, False otherwise
        """
        # Check if caching is enabled
        if not self.enabled:
            return False
        
        # Check the cache strategy
        if self.strategy == CacheStrategy.NONE:
            return False
        elif self.strategy == CacheStrategy.ALL:
            return True
        elif self.strategy == CacheStrategy.SELECT_ONLY:
            # Check if the query is a SELECT query
            return query.strip().upper().startswith("SELECT")
        elif self.strategy == CacheStrategy.PATTERN:
            # Check if the query matches any include pattern
            for pattern in self.include_patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    # Check if the query matches any exclude pattern
                    for exclude_pattern in self.exclude_patterns:
                        if re.search(exclude_pattern, query, re.IGNORECASE):
                            return False
                    
                    return True
            
            return False
        elif self.strategy == CacheStrategy.CUSTOM:
            # Use the custom cache function
            custom_func = self.additional_settings.get("should_cache_func")
            if custom_func:
                return custom_func(query)
            
            return False
        
        return False
    
    def get_cache_ttl(self, query: str) -> float:
        """Get the TTL for a query based on the configuration.
        
        Args:
            query: The query to get the TTL for
            
        Returns:
            The TTL in seconds
        """
        # Get the custom TTL function
        custom_func = self.additional_settings.get("get_ttl_func")
        if custom_func:
            ttl = custom_func(query)
            
            # Clamp the TTL to the min/max range
            return max(self.min_ttl, min(self.max_ttl, ttl))
        
        # Use the default TTL
        return self.default_ttl
    
    def should_invalidate_on_query(self, query: str) -> bool:
        """Check if the cache should be invalidated based on a query.
        
        Args:
            query: The query to check
            
        Returns:
            True if the cache should be invalidated, False otherwise
        """
        # Check if invalidation on write is enabled
        if not self.invalidate_on_write:
            return False
        
        # Check if the query is a write query
        query = query.strip().upper()
        if query.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE", "TRUNCATE")):
            return True
        
        return False
    
    def get_tables_to_invalidate(self, query: str) -> List[str]:
        """Get the tables to invalidate based on a query.
        
        Args:
            query: The query to check
            
        Returns:
            List of tables to invalidate
        """
        # Get the custom invalidation function
        custom_func = self.additional_settings.get("get_tables_to_invalidate_func")
        if custom_func:
            return custom_func(query)
        
        # Extract the table name from the query
        query = query.strip().upper()
        
        # Simple regex to extract table name from common SQL statements
        # This is not a complete SQL parser, but it works for simple cases
        table_match = None
        
        if query.startswith("INSERT"):
            table_match = re.search(r"INSERT\s+INTO\s+`?(\w+)`?", query)
        elif query.startswith("UPDATE"):
            table_match = re.search(r"UPDATE\s+`?(\w+)`?", query)
        elif query.startswith("DELETE"):
            table_match = re.search(r"DELETE\s+FROM\s+`?(\w+)`?", query)
        elif query.startswith("REPLACE"):
            table_match = re.search(r"REPLACE\s+INTO\s+`?(\w+)`?", query)
        elif query.startswith("TRUNCATE"):
            table_match = re.search(r"TRUNCATE\s+TABLE\s+`?(\w+)`?", query)
        
        if table_match:
            table_name = table_match.group(1)
            
            # Get the tables to invalidate for this table
            tables_to_invalidate = [table_name]
            
            # Add any additional tables from the configuration
            if table_name in self.invalidate_tables:
                tables_to_invalidate.extend(self.invalidate_tables[table_name])
            
            return tables_to_invalidate
        
        return []
