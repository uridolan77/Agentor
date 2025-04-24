"""
Unit tests for the core caching utilities.

This module provides comprehensive tests for the caching system, including:
- CacheEntry tests
- InMemoryCache tests with different eviction strategies
- RedisCache tests with mocked Redis client
- High-level Cache interface tests
- Cached decorator tests
- Performance benchmarks
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pickle
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from agentor.core.utils.caching import (
    CacheEntry,
    CacheStrategy,
    CacheBackend,
    InMemoryCache,
    RedisCache,
    Cache,
    cached
)


# ===== Fixtures =====

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
def in_memory_cache_lru():
    """Create an in-memory cache with LRU strategy for testing."""
    return InMemoryCache(
        max_size=5,
        default_ttl=60,
        strategy=CacheStrategy.LRU
    )


@pytest.fixture
def in_memory_cache_lfu():
    """Create an in-memory cache with LFU strategy for testing."""
    return InMemoryCache(
        max_size=5,
        default_ttl=60,
        strategy=CacheStrategy.LFU
    )


@pytest.fixture
def in_memory_cache_fifo():
    """Create an in-memory cache with FIFO strategy for testing."""
    return InMemoryCache(
        max_size=5,
        default_ttl=60,
        strategy=CacheStrategy.FIFO
    )


@pytest.fixture
def in_memory_cache_ttl():
    """Create an in-memory cache with TTL strategy for testing."""
    return InMemoryCache(
        max_size=5,
        default_ttl=60,
        strategy=CacheStrategy.TTL
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
def multi_backend_cache(in_memory_cache_lru, redis_cache):
    """Create a multi-backend cache for testing."""
    return Cache(
        primary_backend=in_memory_cache_lru,
        secondary_backend=redis_cache,
        default_ttl=60
    )


# ===== CacheEntry Tests =====

def test_cache_entry_creation(cache_entry):
    """Test creating a cache entry."""
    assert cache_entry.value == "test_value"
    assert cache_entry.access_count == 0
    assert cache_entry.is_expired == False


def test_cache_entry_expiry():
    """Test cache entry expiry."""
    # Create an already expired entry
    expired_entry = CacheEntry(
        value="test_value",
        expiry=time.time() - 10  # 10 seconds in the past
    )
    
    assert expired_entry.is_expired == True


def test_cache_entry_access(cache_entry):
    """Test accessing a cache entry."""
    # Record the initial access time
    initial_access_time = cache_entry.last_accessed
    
    # Wait a small amount of time
    time.sleep(0.01)
    
    # Access the entry
    cache_entry.access()
    
    # Check that the access count increased
    assert cache_entry.access_count == 1
    
    # Check that the last accessed time was updated
    assert cache_entry.last_accessed > initial_access_time


# ===== InMemoryCache Tests =====

@pytest.mark.asyncio
async def test_in_memory_cache_get_set(in_memory_cache_lru):
    """Test getting and setting values in the in-memory cache."""
    # Test cache miss
    result = await in_memory_cache_lru.get("test_key")
    assert result is None
    
    # Test cache set
    await in_memory_cache_lru.set("test_key", "test_value")
    
    # Test cache hit
    result = await in_memory_cache_lru.get("test_key")
    assert result is not None
    assert result.value == "test_value"


@pytest.mark.asyncio
async def test_in_memory_cache_delete(in_memory_cache_lru):
    """Test deleting values from the in-memory cache."""
    # Set a value
    await in_memory_cache_lru.set("test_key", "test_value")
    
    # Verify it's there
    result = await in_memory_cache_lru.get("test_key")
    assert result is not None
    
    # Delete it
    deleted = await in_memory_cache_lru.delete("test_key")
    assert deleted == True
    
    # Verify it's gone
    result = await in_memory_cache_lru.get("test_key")
    assert result is None
    
    # Try to delete a non-existent key
    deleted = await in_memory_cache_lru.delete("non_existent_key")
    assert deleted == False


@pytest.mark.asyncio
async def test_in_memory_cache_clear(in_memory_cache_lru):
    """Test clearing the in-memory cache."""
    # Set some values
    await in_memory_cache_lru.set("key1", "value1")
    await in_memory_cache_lru.set("key2", "value2")
    
    # Verify they're there
    assert (await in_memory_cache_lru.get("key1")) is not None
    assert (await in_memory_cache_lru.get("key2")) is not None
    
    # Clear the cache
    cleared = await in_memory_cache_lru.clear()
    assert cleared == True
    
    # Verify they're gone
    assert (await in_memory_cache_lru.get("key1")) is None
    assert (await in_memory_cache_lru.get("key2")) is None


@pytest.mark.asyncio
async def test_in_memory_cache_expiry(in_memory_cache_lru):
    """Test cache entry expiry in the in-memory cache."""
    # Set a value with a short TTL
    await in_memory_cache_lru.set("test_key", "test_value", ttl=0.1)
    
    # Verify it's there
    result = await in_memory_cache_lru.get("test_key")
    assert result is not None
    
    # Wait for it to expire
    await asyncio.sleep(0.2)
    
    # Verify it's gone
    result = await in_memory_cache_lru.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_in_memory_cache_stats(in_memory_cache_lru):
    """Test getting cache statistics from the in-memory cache."""
    # Set some values
    await in_memory_cache_lru.set("key1", "value1")
    await in_memory_cache_lru.set("key2", "value2")
    
    # Get some values (hits)
    await in_memory_cache_lru.get("key1")
    await in_memory_cache_lru.get("key2")
    
    # Get a non-existent value (miss)
    await in_memory_cache_lru.get("key3")
    
    # Delete a value
    await in_memory_cache_lru.delete("key1")
    
    # Get the stats
    stats = await in_memory_cache_lru.get_stats()
    
    # Check the stats
    assert stats["size"] == 1
    assert stats["max_size"] == 5
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["sets"] == 2
    assert stats["deletes"] == 1
    assert stats["evictions"] == 0
    assert stats["hit_ratio"] == 2/3


@pytest.mark.asyncio
async def test_in_memory_cache_lru_eviction(in_memory_cache_lru):
    """Test LRU eviction strategy in the in-memory cache."""
    # Fill the cache
    for i in range(5):
        await in_memory_cache_lru.set(f"key{i}", f"value{i}")
    
    # Access some keys to update their LRU status
    await in_memory_cache_lru.get("key0")
    await in_memory_cache_lru.get("key2")
    await in_memory_cache_lru.get("key4")
    
    # Add a new key to trigger eviction
    await in_memory_cache_lru.set("new_key", "new_value")
    
    # The least recently used keys should be evicted (key1 and key3)
    assert (await in_memory_cache_lru.get("key0")) is not None
    assert (await in_memory_cache_lru.get("key1")) is None  # Evicted
    assert (await in_memory_cache_lru.get("key2")) is not None
    assert (await in_memory_cache_lru.get("key3")) is None  # Evicted
    assert (await in_memory_cache_lru.get("key4")) is not None
    assert (await in_memory_cache_lru.get("new_key")) is not None


@pytest.mark.asyncio
async def test_in_memory_cache_lfu_eviction(in_memory_cache_lfu):
    """Test LFU eviction strategy in the in-memory cache."""
    # Fill the cache
    for i in range(5):
        await in_memory_cache_lfu.set(f"key{i}", f"value{i}")
    
    # Access some keys multiple times to update their frequency
    for _ in range(3):
        await in_memory_cache_lfu.get("key0")
    for _ in range(2):
        await in_memory_cache_lfu.get("key2")
    await in_memory_cache_lfu.get("key4")
    
    # Add a new key to trigger eviction
    await in_memory_cache_lfu.set("new_key", "new_value")
    
    # The least frequently used keys should be evicted (key1 and key3)
    assert (await in_memory_cache_lfu.get("key0")) is not None
    assert (await in_memory_cache_lfu.get("key1")) is None  # Evicted
    assert (await in_memory_cache_lfu.get("key2")) is not None
    assert (await in_memory_cache_lfu.get("key3")) is None  # Evicted
    assert (await in_memory_cache_lfu.get("key4")) is not None
    assert (await in_memory_cache_lfu.get("new_key")) is not None


@pytest.mark.asyncio
async def test_in_memory_cache_fifo_eviction(in_memory_cache_fifo):
    """Test FIFO eviction strategy in the in-memory cache."""
    # Fill the cache
    for i in range(5):
        await in_memory_cache_fifo.set(f"key{i}", f"value{i}")
        await asyncio.sleep(0.01)  # Ensure different creation times
    
    # Add a new key to trigger eviction
    await in_memory_cache_fifo.set("new_key", "new_value")
    
    # The first key added should be evicted (key0)
    assert (await in_memory_cache_fifo.get("key0")) is None  # Evicted
    assert (await in_memory_cache_fifo.get("key1")) is not None
    assert (await in_memory_cache_fifo.get("key2")) is not None
    assert (await in_memory_cache_fifo.get("key3")) is not None
    assert (await in_memory_cache_fifo.get("key4")) is not None
    assert (await in_memory_cache_fifo.get("new_key")) is not None


@pytest.mark.asyncio
async def test_in_memory_cache_ttl_eviction(in_memory_cache_ttl):
    """Test TTL eviction strategy in the in-memory cache."""
    # Fill the cache with different TTLs
    await in_memory_cache_ttl.set("key0", "value0", ttl=0.5)
    await in_memory_cache_ttl.set("key1", "value1", ttl=0.4)
    await in_memory_cache_ttl.set("key2", "value2", ttl=0.3)
    await in_memory_cache_ttl.set("key3", "value3", ttl=0.2)
    await in_memory_cache_ttl.set("key4", "value4", ttl=0.1)
    
    # Add a new key to trigger eviction
    await in_memory_cache_ttl.set("new_key", "new_value")
    
    # The key with the shortest TTL should be evicted (key4)
    assert (await in_memory_cache_ttl.get("key0")) is not None
    assert (await in_memory_cache_ttl.get("key1")) is not None
    assert (await in_memory_cache_ttl.get("key2")) is not None
    assert (await in_memory_cache_ttl.get("key3")) is not None
    assert (await in_memory_cache_ttl.get("key4")) is None  # Evicted
    assert (await in_memory_cache_ttl.get("new_key")) is not None


# ===== RedisCache Tests =====

@pytest.mark.asyncio
async def test_redis_cache_get_set(redis_cache):
    """Test getting and setting values in the Redis cache."""
    # Test cache miss
    result = await redis_cache.get("test_key")
    assert result is None
    
    # Test cache set
    await redis_cache.set("test_key", "test_value")
    
    # Test cache hit
    result = await redis_cache.get("test_key")
    assert result is not None
    assert result.value == "test_value"


@pytest.mark.asyncio
async def test_redis_cache_delete(redis_cache):
    """Test deleting values from the Redis cache."""
    # Set a value
    await redis_cache.set("test_key", "test_value")
    
    # Verify it's there
    result = await redis_cache.get("test_key")
    assert result is not None
    
    # Delete it
    deleted = await redis_cache.delete("test_key")
    assert deleted == True
    
    # Verify it's gone
    result = await redis_cache.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_redis_cache_clear(redis_cache):
    """Test clearing the Redis cache."""
    # Set some values
    await redis_cache.set("key1", "value1")
    await redis_cache.set("key2", "value2")
    
    # Verify they're there
    assert (await redis_cache.get("key1")) is not None
    assert (await redis_cache.get("key2")) is not None
    
    # Clear the cache
    cleared = await redis_cache.clear()
    assert cleared == True
    
    # Verify they're gone
    assert (await redis_cache.get("key1")) is None
    assert (await redis_cache.get("key2")) is None


@pytest.mark.asyncio
async def test_redis_cache_stats(redis_cache):
    """Test getting cache statistics from the Redis cache."""
    # Set some values
    await redis_cache.set("key1", "value1")
    await redis_cache.set("key2", "value2")
    
    # Get some values (hits)
    await redis_cache.get("key1")
    await redis_cache.get("key2")
    
    # Get a non-existent value (miss)
    await redis_cache.get("key3")
    
    # Delete a value
    await redis_cache.delete("key1")
    
    # Get the stats
    stats = await redis_cache.get_stats()
    
    # Check the stats
    assert stats["size"] == 1
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["sets"] == 2
    assert stats["deletes"] == 1
    assert stats["hit_ratio"] == 2/3


# ===== Cache (High-level interface) Tests =====

@pytest.mark.asyncio
async def test_cache_get_set(multi_backend_cache):
    """Test getting and setting values in the high-level cache."""
    # Test cache miss
    result = await multi_backend_cache.get("test_key")
    assert result is None
    
    # Test cache set
    await multi_backend_cache.set("test_key", "test_value")
    
    # Test cache hit
    result = await multi_backend_cache.get("test_key")
    assert result == "test_value"


@pytest.mark.asyncio
async def test_cache_delete(multi_backend_cache):
    """Test deleting values from the high-level cache."""
    # Set a value
    await multi_backend_cache.set("test_key", "test_value")
    
    # Verify it's there
    result = await multi_backend_cache.get("test_key")
    assert result == "test_value"
    
    # Delete it
    deleted = await multi_backend_cache.delete("test_key")
    assert deleted == True
    
    # Verify it's gone
    result = await multi_backend_cache.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_clear(multi_backend_cache):
    """Test clearing the high-level cache."""
    # Set some values
    await multi_backend_cache.set("key1", "value1")
    await multi_backend_cache.set("key2", "value2")
    
    # Verify they're there
    assert (await multi_backend_cache.get("key1")) == "value1"
    assert (await multi_backend_cache.get("key2")) == "value2"
    
    # Clear the cache
    cleared = await multi_backend_cache.clear()
    assert cleared == True
    
    # Verify they're gone
    assert (await multi_backend_cache.get("key1")) is None
    assert (await multi_backend_cache.get("key2")) is None


@pytest.mark.asyncio
async def test_cache_key_serializer(multi_backend_cache):
    """Test the key serializer in the high-level cache."""
    # Test with a string key
    await multi_backend_cache.set("string_key", "string_value")
    assert (await multi_backend_cache.get("string_key")) == "string_value"
    
    # Test with a complex key (dictionary)
    complex_key = {"id": 123, "name": "test"}
    await multi_backend_cache.set(complex_key, "complex_value")
    assert (await multi_backend_cache.get(complex_key)) == "complex_value"
    
    # Test with another complex key (list)
    list_key = [1, 2, 3]
    await multi_backend_cache.set(list_key, "list_value")
    assert (await multi_backend_cache.get(list_key)) == "list_value"


@pytest.mark.asyncio
async def test_cache_secondary_backend(multi_backend_cache):
    """Test the secondary backend in the high-level cache."""
    # Set a value
    await multi_backend_cache.set("test_key", "test_value")
    
    # Clear the primary backend directly
    await multi_backend_cache.primary.clear()
    
    # The value should still be available from the secondary backend
    result = await multi_backend_cache.get("test_key")
    assert result == "test_value"
    
    # The value should now be in the primary backend again
    primary_entry = await multi_backend_cache.primary.get(multi_backend_cache.key_serializer("test_key"))
    assert primary_entry is not None
    assert primary_entry.value == "test_value"


# ===== Cached Decorator Tests =====

@pytest.mark.asyncio
async def test_cached_decorator_sync_function():
    """Test the cached decorator with a synchronous function."""
    call_count = 0
    
    @cached(ttl=60)
    def test_function(arg1, arg2=None):
        nonlocal call_count
        call_count += 1
        return f"{arg1}-{arg2}"
    
    # First call should execute the function
    result1 = await test_function("test", arg2="value")
    assert result1 == "test-value"
    assert call_count == 1
    
    # Second call with the same arguments should use the cache
    result2 = await test_function("test", arg2="value")
    assert result2 == "test-value"
    assert call_count == 1  # Still 1
    
    # Call with different arguments should execute the function again
    result3 = await test_function("test", arg2="different")
    assert result3 == "test-different"
    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_decorator_async_function():
    """Test the cached decorator with an asynchronous function."""
    call_count = 0
    
    @cached(ttl=60)
    async def test_async_function(arg):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return f"async-{arg}"
    
    # First call should execute the function
    result1 = await test_async_function("test")
    assert result1 == "async-test"
    assert call_count == 1
    
    # Second call with the same arguments should use the cache
    result2 = await test_async_function("test")
    assert result2 == "async-test"
    assert call_count == 1  # Still 1
    
    # Call with different arguments should execute the function again
    result3 = await test_async_function("different")
    assert result3 == "async-different"
    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_decorator_custom_key_builder():
    """Test the cached decorator with a custom key builder."""
    call_count = 0
    
    def custom_key_builder(arg1, arg2=None):
        # Only use arg1 for the cache key
        return f"custom:{arg1}"
    
    @cached(ttl=60, key_builder=custom_key_builder)
    def test_function(arg1, arg2=None):
        nonlocal call_count
        call_count += 1
        return f"{arg1}-{arg2}"
    
    # First call should execute the function
    result1 = await test_function("test", arg2="value1")
    assert result1 == "test-value1"
    assert call_count == 1
    
    # Second call with the same arg1 but different arg2 should use the cache
    # because our custom key builder only uses arg1
    result2 = await test_function("test", arg2="value2")
    assert result2 == "test-value1"  # Note: returns cached value, not new value
    assert call_count == 1  # Still 1
    
    # Call with different arg1 should execute the function again
    result3 = await test_function("different", arg2="value")
    assert result3 == "different-value"
    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_decorator_invalidate():
    """Test the invalidate method of the cached decorator."""
    call_count = 0
    
    @cached(ttl=60)
    def test_function(arg):
        nonlocal call_count
        call_count += 1
        return f"result-{arg}"
    
    # First call should execute the function
    result1 = await test_function("test")
    assert result1 == "result-test"
    assert call_count == 1
    
    # Second call should use the cache
    result2 = await test_function("test")
    assert result2 == "result-test"
    assert call_count == 1  # Still 1
    
    # Invalidate the cache for this argument
    await test_function.invalidate("test")
    
    # Next call should execute the function again
    result3 = await test_function("test")
    assert result3 == "result-test"
    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_decorator_invalidate_all():
    """Test the invalidate_all method of the cached decorator."""
    call_count = 0
    
    @cached(ttl=60)
    def test_function(arg):
        nonlocal call_count
        call_count += 1
        return f"result-{arg}"
    
    # Call with different arguments
    await test_function("arg1")
    await test_function("arg2")
    assert call_count == 2
    
    # Call again, should use cache
    await test_function("arg1")
    await test_function("arg2")
    assert call_count == 2  # Still 2
    
    # Invalidate all cache entries
    await test_function.invalidate_all()
    
    # Call again, should execute the function
    await test_function("arg1")
    await test_function("arg2")
    assert call_count == 4


# ===== Performance Benchmarks =====

@pytest.mark.asyncio
async def test_cache_performance_benchmark():
    """Benchmark the performance of different cache strategies."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Create caches with different strategies
    caches = {
        "LRU": InMemoryCache(max_size=10000, strategy=CacheStrategy.LRU),
        "LFU": InMemoryCache(max_size=10000, strategy=CacheStrategy.LFU),
        "FIFO": InMemoryCache(max_size=10000, strategy=CacheStrategy.FIFO),
        "TTL": InMemoryCache(max_size=10000, strategy=CacheStrategy.TTL)
    }
    
    # Number of operations
    num_ops = 100000
    
    # Results
    results = {}
    
    for name, cache in caches.items():
        # Measure set performance
        start_time = time.time()
        for i in range(num_ops):
            await cache.set(f"key{i % 1000}", f"value{i}")
        set_time = time.time() - start_time
        
        # Measure get performance (with 80% hit rate)
        start_time = time.time()
        for i in range(num_ops):
            if i % 5 == 0:  # 20% miss rate
                await cache.get(f"key{i + 1000}")  # Miss
            else:
                await cache.get(f"key{i % 1000}")  # Hit
        get_time = time.time() - start_time
        
        # Store results
        results[name] = {
            "set_time": set_time,
            "get_time": get_time,
            "total_time": set_time + get_time,
            "ops_per_second": num_ops / (set_time + get_time)
        }
    
    # Print results
    print("\nCache Performance Benchmark Results:")
    print("====================================")
    for name, result in results.items():
        print(f"{name} Strategy:")
        print(f"  Set Time: {result['set_time']:.4f}s")
        print(f"  Get Time: {result['get_time']:.4f}s")
        print(f"  Total Time: {result['total_time']:.4f}s")
        print(f"  Ops/Second: {result['ops_per_second']:.2f}")
        print()
    
    # Assert something to make pytest happy
    assert True
