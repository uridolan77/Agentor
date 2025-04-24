"""
Benchmarks for the caching system.

This module provides benchmarks to evaluate the performance of the caching system
under different conditions, including:
- Different cache strategies
- Different cache sizes
- Different workload patterns
- Different key and value sizes
"""

import pytest
import asyncio
import time
import random
import string
import json
import statistics
from typing import Dict, Any, List, Optional, Tuple

from agentor.core.utils.caching import (
    CacheStrategy,
    InMemoryCache,
    RedisCache,
    Cache,
    cached
)


# ===== Utility Functions =====

def generate_random_string(length: int) -> str:
    """Generate a random string of the specified length.
    
    Args:
        length: The length of the string
        
    Returns:
        A random string
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def generate_random_dict(num_keys: int, value_length: int) -> Dict[str, str]:
    """Generate a random dictionary with the specified number of keys.
    
    Args:
        num_keys: The number of keys in the dictionary
        value_length: The length of each value
        
    Returns:
        A random dictionary
    """
    return {
        f"key_{i}": generate_random_string(value_length)
        for i in range(num_keys)
    }


async def measure_operation_time(
    operation,
    num_iterations: int = 1000,
    warmup_iterations: int = 100
) -> Tuple[float, float, float]:
    """Measure the time taken by an operation.
    
    Args:
        operation: The operation to measure
        num_iterations: The number of iterations
        warmup_iterations: The number of warmup iterations
        
    Returns:
        A tuple of (average_time, min_time, max_time)
    """
    # Warmup
    for _ in range(warmup_iterations):
        await operation()
    
    # Measure
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        await operation()
        end_time = time.time()
        times.append(end_time - start_time)
    
    return (
        statistics.mean(times),
        min(times),
        max(times)
    )


# ===== Benchmark Tests =====

@pytest.mark.asyncio
async def test_cache_strategy_benchmark():
    """Benchmark different cache strategies."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Parameters
    num_iterations = 10000
    warmup_iterations = 1000
    cache_size = 1000
    num_keys = 2000  # More than cache size to force evictions
    
    # Create caches with different strategies
    caches = {
        "LRU": InMemoryCache(max_size=cache_size, strategy=CacheStrategy.LRU),
        "LFU": InMemoryCache(max_size=cache_size, strategy=CacheStrategy.LFU),
        "FIFO": InMemoryCache(max_size=cache_size, strategy=CacheStrategy.FIFO),
        "TTL": InMemoryCache(max_size=cache_size, strategy=CacheStrategy.TTL)
    }
    
    # Results
    results = {}
    
    for name, cache in caches.items():
        print(f"\nBenchmarking {name} strategy...")
        
        # Generate keys and values
        keys = [f"key_{i}" for i in range(num_keys)]
        values = [f"value_{i}" for i in range(num_keys)]
        
        # Measure set performance
        async def set_operation():
            key = random.choice(keys)
            value = random.choice(values)
            await cache.set(key, value)
        
        set_avg, set_min, set_max = await measure_operation_time(
            set_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Fill the cache for get operations
        for i in range(min(cache_size, num_keys)):
            await cache.set(keys[i], values[i])
        
        # Measure get performance (with hits and misses)
        async def get_operation():
            # 80% hits, 20% misses
            if random.random() < 0.8:
                key = random.choice(keys[:cache_size])  # Hit
            else:
                key = random.choice(keys[cache_size:])  # Miss
            await cache.get(key)
        
        get_avg, get_min, get_max = await measure_operation_time(
            get_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Measure delete performance
        async def delete_operation():
            key = random.choice(keys)
            await cache.delete(key)
        
        delete_avg, delete_min, delete_max = await measure_operation_time(
            delete_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Store results
        results[name] = {
            "set": {
                "avg": set_avg,
                "min": set_min,
                "max": set_max,
                "ops_per_sec": 1 / set_avg
            },
            "get": {
                "avg": get_avg,
                "min": get_min,
                "max": get_max,
                "ops_per_sec": 1 / get_avg
            },
            "delete": {
                "avg": delete_avg,
                "min": delete_min,
                "max": delete_max,
                "ops_per_sec": 1 / delete_avg
            }
        }
    
    # Print results
    print("\nCache Strategy Benchmark Results:")
    print("================================")
    for name, result in results.items():
        print(f"\n{name} Strategy:")
        print(f"  Set: {result['set']['avg']*1000:.3f}ms avg, {result['set']['ops_per_sec']:.0f} ops/sec")
        print(f"  Get: {result['get']['avg']*1000:.3f}ms avg, {result['get']['ops_per_sec']:.0f} ops/sec")
        print(f"  Delete: {result['delete']['avg']*1000:.3f}ms avg, {result['delete']['ops_per_sec']:.0f} ops/sec")
    
    # Find the fastest strategy for each operation
    fastest_set = min(results.items(), key=lambda x: x[1]["set"]["avg"])
    fastest_get = min(results.items(), key=lambda x: x[1]["get"]["avg"])
    fastest_delete = min(results.items(), key=lambda x: x[1]["delete"]["avg"])
    
    print("\nFastest Strategies:")
    print(f"  Set: {fastest_set[0]} ({fastest_set[1]['set']['ops_per_sec']:.0f} ops/sec)")
    print(f"  Get: {fastest_get[0]} ({fastest_get[1]['get']['ops_per_sec']:.0f} ops/sec)")
    print(f"  Delete: {fastest_delete[0]} ({fastest_delete[1]['delete']['ops_per_sec']:.0f} ops/sec)")
    
    # Assert something to make pytest happy
    assert True


@pytest.mark.asyncio
async def test_cache_size_benchmark():
    """Benchmark different cache sizes."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Parameters
    num_iterations = 5000
    warmup_iterations = 500
    cache_sizes = [100, 1000, 10000, 100000]
    
    # Results
    results = {}
    
    for cache_size in cache_sizes:
        print(f"\nBenchmarking cache size {cache_size}...")
        
        # Create a cache
        cache = InMemoryCache(max_size=cache_size, strategy=CacheStrategy.LRU)
        
        # Generate keys (twice the cache size to ensure evictions)
        num_keys = cache_size * 2
        keys = [f"key_{i}" for i in range(num_keys)]
        values = [f"value_{i}" for i in range(num_keys)]
        
        # Measure set performance
        async def set_operation():
            key = random.choice(keys)
            value = random.choice(values)
            await cache.set(key, value)
        
        set_avg, set_min, set_max = await measure_operation_time(
            set_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Fill the cache for get operations
        for i in range(min(cache_size, num_keys)):
            await cache.set(keys[i], values[i])
        
        # Measure get performance (with hits and misses)
        async def get_operation():
            # 80% hits, 20% misses
            if random.random() < 0.8:
                key = random.choice(keys[:cache_size])  # Hit
            else:
                key = random.choice(keys[cache_size:])  # Miss
            await cache.get(key)
        
        get_avg, get_min, get_max = await measure_operation_time(
            get_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Store results
        results[cache_size] = {
            "set": {
                "avg": set_avg,
                "min": set_min,
                "max": set_max,
                "ops_per_sec": 1 / set_avg
            },
            "get": {
                "avg": get_avg,
                "min": get_min,
                "max": get_max,
                "ops_per_sec": 1 / get_avg
            }
        }
    
    # Print results
    print("\nCache Size Benchmark Results:")
    print("============================")
    for size, result in results.items():
        print(f"\nCache Size {size}:")
        print(f"  Set: {result['set']['avg']*1000:.3f}ms avg, {result['set']['ops_per_sec']:.0f} ops/sec")
        print(f"  Get: {result['get']['avg']*1000:.3f}ms avg, {result['get']['ops_per_sec']:.0f} ops/sec")
    
    # Assert something to make pytest happy
    assert True


@pytest.mark.asyncio
async def test_workload_pattern_benchmark():
    """Benchmark different workload patterns."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Parameters
    num_iterations = 10000
    warmup_iterations = 1000
    cache_size = 10000
    
    # Create a cache
    cache = InMemoryCache(max_size=cache_size, strategy=CacheStrategy.LRU)
    
    # Workload patterns
    patterns = {
        "Uniform": lambda keys: random.choice(keys),  # Uniform distribution
        "Zipfian": lambda keys: keys[min(int(random.paretovariate(1.5)), len(keys) - 1)],  # Power law distribution
        "Sequential": lambda keys, i=[-1]: keys[(i[0] := (i[0] + 1) % len(keys))],  # Sequential access
        "Clustered": lambda keys: random.choice(keys[max(0, random.randint(0, len(keys)) - 100):min(len(keys), random.randint(0, len(keys)) + 100)])  # Clustered access
    }
    
    # Results
    results = {}
    
    for name, pattern_func in patterns.items():
        print(f"\nBenchmarking {name} workload pattern...")
        
        # Generate keys and values
        num_keys = cache_size * 2
        keys = [f"key_{i}" for i in range(num_keys)]
        values = [f"value_{i}" for i in range(num_keys)]
        
        # Fill the cache
        for i in range(min(cache_size, num_keys)):
            await cache.set(keys[i], values[i])
        
        # Measure get performance
        async def get_operation():
            key = pattern_func(keys)
            await cache.get(key)
        
        get_avg, get_min, get_max = await measure_operation_time(
            get_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Measure set performance
        async def set_operation():
            key = pattern_func(keys)
            value = random.choice(values)
            await cache.set(key, value)
        
        set_avg, set_min, set_max = await measure_operation_time(
            set_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Store results
        results[name] = {
            "get": {
                "avg": get_avg,
                "min": get_min,
                "max": get_max,
                "ops_per_sec": 1 / get_avg
            },
            "set": {
                "avg": set_avg,
                "min": set_min,
                "max": set_max,
                "ops_per_sec": 1 / set_avg
            }
        }
        
        # Clear the cache for the next pattern
        await cache.clear()
    
    # Print results
    print("\nWorkload Pattern Benchmark Results:")
    print("==================================")
    for name, result in results.items():
        print(f"\n{name} Pattern:")
        print(f"  Get: {result['get']['avg']*1000:.3f}ms avg, {result['get']['ops_per_sec']:.0f} ops/sec")
        print(f"  Set: {result['set']['avg']*1000:.3f}ms avg, {result['set']['ops_per_sec']:.0f} ops/sec")
    
    # Assert something to make pytest happy
    assert True


@pytest.mark.asyncio
async def test_key_value_size_benchmark():
    """Benchmark different key and value sizes."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Parameters
    num_iterations = 5000
    warmup_iterations = 500
    cache_size = 1000
    
    # Key and value sizes to test
    sizes = [
        {"key_size": 10, "value_size": 100},
        {"key_size": 10, "value_size": 1000},
        {"key_size": 10, "value_size": 10000},
        {"key_size": 100, "value_size": 100},
        {"key_size": 1000, "value_size": 100}
    ]
    
    # Results
    results = {}
    
    for size_config in sizes:
        key_size = size_config["key_size"]
        value_size = size_config["value_size"]
        name = f"Key={key_size}B, Value={value_size}B"
        
        print(f"\nBenchmarking {name}...")
        
        # Create a cache
        cache = InMemoryCache(max_size=cache_size, strategy=CacheStrategy.LRU)
        
        # Generate keys and values
        keys = [generate_random_string(key_size) for _ in range(cache_size)]
        values = [generate_random_string(value_size) for _ in range(cache_size)]
        
        # Measure set performance
        async def set_operation():
            key = random.choice(keys)
            value = random.choice(values)
            await cache.set(key, value)
        
        set_avg, set_min, set_max = await measure_operation_time(
            set_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Fill the cache for get operations
        for i in range(cache_size):
            await cache.set(keys[i], values[i])
        
        # Measure get performance
        async def get_operation():
            key = random.choice(keys)
            await cache.get(key)
        
        get_avg, get_min, get_max = await measure_operation_time(
            get_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Store results
        results[name] = {
            "set": {
                "avg": set_avg,
                "min": set_min,
                "max": set_max,
                "ops_per_sec": 1 / set_avg
            },
            "get": {
                "avg": get_avg,
                "min": get_min,
                "max": get_max,
                "ops_per_sec": 1 / get_avg
            }
        }
    
    # Print results
    print("\nKey/Value Size Benchmark Results:")
    print("================================")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Set: {result['set']['avg']*1000:.3f}ms avg, {result['set']['ops_per_sec']:.0f} ops/sec")
        print(f"  Get: {result['get']['avg']*1000:.3f}ms avg, {result['get']['ops_per_sec']:.0f} ops/sec")
    
    # Assert something to make pytest happy
    assert True


@pytest.mark.asyncio
async def test_cached_decorator_benchmark():
    """Benchmark the cached decorator with different configurations."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Parameters
    num_iterations = 5000
    warmup_iterations = 500
    
    # Function to cache
    async def expensive_function(arg1, arg2=None):
        # Simulate an expensive operation
        await asyncio.sleep(0.01)
        return f"{arg1}-{arg2}"
    
    # Cache configurations
    configs = {
        "Default": {
            "ttl": None,
            "key_builder": None,
            "cache_instance": None
        },
        "Custom TTL": {
            "ttl": 30,
            "key_builder": None,
            "cache_instance": None
        },
        "Custom Key Builder": {
            "ttl": None,
            "key_builder": lambda arg1, arg2=None: f"custom:{arg1}",
            "cache_instance": None
        },
        "Custom Cache": {
            "ttl": None,
            "key_builder": None,
            "cache_instance": Cache(
                primary_backend=InMemoryCache(
                    max_size=10000,
                    default_ttl=60,
                    strategy=CacheStrategy.LRU
                )
            )
        }
    }
    
    # Results
    results = {}
    
    for name, config in configs.items():
        print(f"\nBenchmarking {name} configuration...")
        
        # Create the cached function
        cached_func = cached(
            ttl=config["ttl"],
            key_builder=config["key_builder"],
            cache_instance=config["cache_instance"]
        )(expensive_function)
        
        # Generate arguments
        args1 = [f"arg1_{i}" for i in range(100)]
        args2 = [f"arg2_{i}" for i in range(100)]
        
        # Measure first-call performance (cache misses)
        async def first_call_operation():
            arg1 = random.choice(args1)
            arg2 = random.choice(args2)
            await cached_func(arg1, arg2)
        
        first_call_avg, first_call_min, first_call_max = await measure_operation_time(
            first_call_operation,
            num_iterations=num_iterations // 10,  # Fewer iterations for expensive operation
            warmup_iterations=warmup_iterations // 10
        )
        
        # Measure repeated-call performance (cache hits)
        # Use a fixed set of arguments to ensure cache hits
        fixed_args = [(args1[0], args2[0]) for _ in range(num_iterations)]
        
        # Prime the cache
        await cached_func(args1[0], args2[0])
        
        async def repeated_call_operation():
            arg1, arg2 = fixed_args[0]
            await cached_func(arg1, arg2)
        
        repeated_call_avg, repeated_call_min, repeated_call_max = await measure_operation_time(
            repeated_call_operation,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Store results
        results[name] = {
            "first_call": {
                "avg": first_call_avg,
                "min": first_call_min,
                "max": first_call_max,
                "ops_per_sec": 1 / first_call_avg
            },
            "repeated_call": {
                "avg": repeated_call_avg,
                "min": repeated_call_min,
                "max": repeated_call_max,
                "ops_per_sec": 1 / repeated_call_avg
            },
            "speedup": first_call_avg / repeated_call_avg
        }
    
    # Print results
    print("\nCached Decorator Benchmark Results:")
    print("==================================")
    for name, result in results.items():
        print(f"\n{name} Configuration:")
        print(f"  First Call: {result['first_call']['avg']*1000:.3f}ms avg, {result['first_call']['ops_per_sec']:.0f} ops/sec")
        print(f"  Repeated Call: {result['repeated_call']['avg']*1000:.3f}ms avg, {result['repeated_call']['ops_per_sec']:.0f} ops/sec")
        print(f"  Speedup: {result['speedup']:.2f}x")
    
    # Assert something to make pytest happy
    assert True
