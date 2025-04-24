"""
Pytest configuration and fixtures for the Agentor test suite.

This module provides common fixtures that can be used across different test modules.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import json
import pickle
from typing import Dict, Any, List, Optional

from agentor.core.utils.caching import (
    CacheEntry,
    CacheStrategy,
    InMemoryCache,
    RedisCache,
    Cache
)


# ===== Cache Fixtures =====

@pytest.fixture
def cache_entry():
    """Create a cache entry for testing."""
    return CacheEntry(
        value="test_value",
        expiry=time.time() + 60,  # 60 seconds from now
        created_at=time.time(),
        last_accessed=time.time(),
        access_count=0
    )


@pytest.fixture
def in_memory_cache():
    """Create an in-memory cache for testing."""
    return InMemoryCache(
        max_size=100,
        default_ttl=60,
        strategy=CacheStrategy.LRU
    )


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_client = AsyncMock()
    
    # Mock storage for simulating Redis
    storage = {}
    
    # Mock get method
    async def mock_get(key):
        return storage.get(key)
    
    # Mock set method
    async def mock_set(key, value, ex=None):
        storage[key] = value
        return True
    
    # Mock delete method
    async def mock_delete(*keys):
        deleted = 0
        for key in keys:
            if key in storage:
                del storage[key]
                deleted += 1
        return deleted
    
    # Mock keys method
    async def mock_keys(pattern):
        import fnmatch
        return [k for k in storage.keys() if fnmatch.fnmatch(k, pattern)]
    
    # Assign mock methods
    mock_client.get = mock_get
    mock_client.set = mock_set
    mock_client.delete = mock_delete
    mock_client.keys = mock_keys
    
    return mock_client


@pytest.fixture
def redis_cache(mock_redis_client):
    """Create a Redis cache with a mock client for testing."""
    with patch('redis.asyncio.from_url', return_value=mock_redis_client):
        cache = RedisCache(
            redis_url="redis://localhost:6379/0",
            default_ttl=60,
            prefix="test:"
        )
        # Replace the Redis client with our mock
        cache.redis = mock_redis_client
        return cache


@pytest.fixture
def multi_backend_cache(in_memory_cache, redis_cache):
    """Create a multi-backend cache for testing."""
    return Cache(
        primary_backend=in_memory_cache,
        secondary_backend=redis_cache,
        default_ttl=60
    )


# ===== Mock Classes for Testing =====

class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, name="mock-llm", latency=0.1):
        """Initialize the mock LLM.
        
        Args:
            name: The name of the LLM
            latency: Simulated latency in seconds
        """
        self.name = name
        self.latency = latency
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the prompt.
        
        Args:
            prompt: The prompt to respond to
            **kwargs: Additional arguments
            
        Returns:
            The generated response
        """
        self.call_count += 1
        await asyncio.sleep(self.latency)  # Simulate API latency
        return f"Response to: {prompt}"


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockLLM()


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name="mock-agent", llm=None):
        """Initialize the mock agent.
        
        Args:
            name: The name of the agent
            llm: The LLM to use
        """
        self.name = name
        self.llm = llm or MockLLM()
        self.call_count = 0
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process the input text.
        
        Args:
            input_text: The input text to process
            
        Returns:
            The processing result
        """
        self.call_count += 1
        response = await self.llm.generate(input_text)
        return {
            "input": input_text,
            "output": response,
            "agent": self.name
        }


@pytest.fixture
def mock_agent(mock_llm):
    """Create a mock agent for testing."""
    return MockAgent(llm=mock_llm)


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name="mock-tool", latency=0.05):
        """Initialize the mock tool.
        
        Args:
            name: The name of the tool
            latency: Simulated latency in seconds
        """
        self.name = name
        self.latency = latency
        self.call_count = 0
    
    async def execute(self, input_data: Any) -> Any:
        """Execute the tool.
        
        Args:
            input_data: The input data
            
        Returns:
            The execution result
        """
        self.call_count += 1
        await asyncio.sleep(self.latency)  # Simulate execution latency
        
        if isinstance(input_data, dict):
            return {k: f"processed_{v}" for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [f"processed_{item}" for item in input_data]
        else:
            return f"processed_{input_data}"


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    return MockTool()
