"""
Cached tool registry for the Agentor framework.

This module provides a tool registry with caching for improved performance.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import asyncio
import time
import hashlib
import json

from agentor.core.interfaces.tool import ITool, IToolRegistry, ToolResult
from agentor.agents.enhanced_tools import EnhancedToolRegistry
from agentor.core.utils.caching import Cache, InMemoryCache, CacheStrategy

logger = logging.getLogger(__name__)


class CachedToolRegistry(IToolRegistry):
    """Tool registry with caching for faster tool lookups and schema retrieval."""
    
    def __init__(
        self,
        base_registry: IToolRegistry,
        cache_ttl: int = 3600,
        schema_cache_ttl: int = 86400,
        result_cache_ttl: int = 300,
        max_cache_size: int = 1000,
        cache_strategy: CacheStrategy = CacheStrategy.LRU,
        enable_result_caching: bool = True
    ):
        """Initialize the cached tool registry.
        
        Args:
            base_registry: The base tool registry to wrap
            cache_ttl: Time-to-live for cache entries in seconds
            schema_cache_ttl: Time-to-live for schema cache entries in seconds
            result_cache_ttl: Time-to-live for result cache entries in seconds
            max_cache_size: Maximum number of items to store in the cache
            cache_strategy: Cache eviction strategy
            enable_result_caching: Whether to cache tool execution results
        """
        self.base_registry = base_registry
        self.cache_ttl = cache_ttl
        self.schema_cache_ttl = schema_cache_ttl
        self.result_cache_ttl = result_cache_ttl
        self.enable_result_caching = enable_result_caching
        
        # Create caches
        self.tool_cache = Cache(
            primary_backend=InMemoryCache(
                max_size=max_cache_size,
                default_ttl=cache_ttl,
                strategy=cache_strategy
            )
        )
        
        self.schema_cache = Cache(
            primary_backend=InMemoryCache(
                max_size=max_cache_size,
                default_ttl=schema_cache_ttl,
                strategy=cache_strategy
            )
        )
        
        self.result_cache = Cache(
            primary_backend=InMemoryCache(
                max_size=max_cache_size,
                default_ttl=result_cache_ttl,
                strategy=cache_strategy
            )
        )
    
    def register_tool(self, tool: ITool) -> None:
        """Register a tool.
        
        Args:
            tool: The tool to register
        """
        # Register with the base registry
        self.base_registry.register_tool(tool)
        
        # Invalidate caches
        asyncio.create_task(self.tool_cache.delete(tool.name))
        asyncio.create_task(self.schema_cache.delete(tool.name))
    
    async def get_tool(self, name: str, version: Optional[str] = None) -> Optional[ITool]:
        """Get a tool by name and optional version.
        
        Args:
            name: The name of the tool
            version: Optional version of the tool
            
        Returns:
            The tool, or None if not found
        """
        # Create a cache key
        cache_key = name
        if version:
            cache_key = f"{name}:{version}"
        
        # Try to get from cache
        cached_tool = await self.tool_cache.get(cache_key)
        if cached_tool is not None:
            return cached_tool
        
        # Get from base registry
        tool = await self.base_registry.get_tool(name, version)
        
        # Cache the result
        if tool is not None:
            await self.tool_cache.set(cache_key, tool)
        
        return tool
    
    async def get_tools(self) -> Dict[str, ITool]:
        """Get all tools.
        
        Returns:
            A dictionary of tool names to tools
        """
        # Get from base registry
        tools = await self.base_registry.get_tools()
        
        # Cache each tool
        for name, tool in tools.items():
            await self.tool_cache.set(name, tool)
        
        return tools
    
    async def get_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get the schema for a tool.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            The tool schema
            
        Raises:
            KeyError: If the tool is not registered
        """
        # Try to get from cache
        cached_schema = await self.schema_cache.get(tool_name)
        if cached_schema is not None:
            return cached_schema
        
        # Get from base registry
        schema = await self.base_registry.get_schema(tool_name)
        
        # Cache the result
        await self.schema_cache.set(tool_name, schema)
        
        return schema
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool.
        
        Args:
            tool_name: The name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            The result of executing the tool
            
        Raises:
            KeyError: If the tool is not registered
        """
        # Check if result caching is enabled
        if not self.enable_result_caching:
            return await self.base_registry.execute_tool(tool_name, **kwargs)
        
        # Create a cache key
        cache_key = self._create_execution_cache_key(tool_name, kwargs)
        
        # Try to get from cache
        cached_result = await self.result_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for tool execution: {tool_name}")
            return cached_result
        
        # Execute the tool
        result = await self.base_registry.execute_tool(tool_name, **kwargs)
        
        # Cache the result if successful and cacheable
        if result.success and self._is_cacheable_result(tool_name, kwargs, result):
            await self.result_cache.set(cache_key, result)
        
        return result
    
    def _create_execution_cache_key(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Create a cache key for tool execution.
        
        Args:
            tool_name: The name of the tool
            kwargs: The tool arguments
            
        Returns:
            A cache key
        """
        # Convert kwargs to a stable string representation
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        
        # Create a hash
        key = f"{tool_name}:{kwargs_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_cacheable_result(self, tool_name: str, kwargs: Dict[str, Any], result: ToolResult) -> bool:
        """Check if a tool result is cacheable.
        
        Args:
            tool_name: The name of the tool
            kwargs: The tool arguments
            result: The tool result
            
        Returns:
            True if the result is cacheable
        """
        # Don't cache errors
        if not result.success:
            return False
        
        # Don't cache results with no data
        if not result.data:
            return False
        
        # Don't cache results for tools that are likely to change
        non_cacheable_tools = {
            "weather",  # Weather changes frequently
            "news",     # News changes frequently
            "random",   # Random results shouldn't be cached
            "time",     # Time changes constantly
            "date"      # Date changes daily
        }
        
        if tool_name.lower() in non_cacheable_tools:
            return False
        
        # Check for time-sensitive keywords in the tool name
        time_sensitive_keywords = {"current", "latest", "today", "now", "live", "real-time", "realtime"}
        if any(keyword in tool_name.lower() for keyword in time_sensitive_keywords):
            return False
        
        return True
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        tool_stats = await self.tool_cache.get_stats()
        schema_stats = await self.schema_cache.get_stats()
        result_stats = await self.result_cache.get_stats()
        
        return {
            "tool_cache": tool_stats,
            "schema_cache": schema_stats,
            "result_cache": result_stats
        }
    
    async def clear_caches(self) -> None:
        """Clear all caches."""
        await self.tool_cache.clear()
        await self.schema_cache.clear()
        await self.result_cache.clear()
        
        logger.info("Cleared all tool registry caches")


def create_cached_registry(
    base_registry: Optional[IToolRegistry] = None,
    **kwargs
) -> CachedToolRegistry:
    """Create a cached tool registry.
    
    Args:
        base_registry: The base tool registry to wrap
        **kwargs: Additional arguments for the cached registry
        
    Returns:
        A cached tool registry
    """
    if base_registry is None:
        base_registry = EnhancedToolRegistry()
    
    return CachedToolRegistry(base_registry, **kwargs)
