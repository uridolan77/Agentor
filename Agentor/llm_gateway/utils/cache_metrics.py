"""
Enhanced metrics for cache operations.

This module provides specialized metrics for monitoring cache operations, including:
- Cache hit/miss rates
- Cache efficiency metrics
- Size and retention monitoring
- Semantic similarity metrics
- Cache performance metrics
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
import functools
from datetime import datetime, timedelta
from enum import Enum

from prometheus_client import Counter, Histogram, Gauge, Summary

from agentor.llm_gateway.utils.metrics import registry
from agentor.llm_gateway.utils.llm_metrics import (
    LLM_CACHE_HITS, LLM_CACHE_MISSES, LLM_CACHE_HIT_RATIO
)

logger = logging.getLogger(__name__)


# Additional cache-specific metrics
CACHE_SIZE = Gauge(
    'cache_size_entries',
    'Current number of entries in the cache',
    ['cache_type', 'provider', 'model'],
    registry=registry
)

CACHE_CAPACITY = Gauge(
    'cache_capacity_entries',
    'Maximum capacity of the cache',
    ['cache_type', 'provider', 'model'],
    registry=registry
)

CACHE_USAGE_RATIO = Gauge(
    'cache_usage_ratio',
    'Ratio of current cache size to maximum capacity',
    ['cache_type', 'provider', 'model'],
    registry=registry
)

CACHE_EVICTIONS = Counter(
    'cache_evictions_total',
    'Number of cache entries evicted',
    ['cache_type', 'provider', 'model', 'reason'],
    registry=registry
)

CACHE_CLEANING_TIME = Histogram(
    'cache_cleaning_time_seconds',
    'Time taken to clean the cache',
    ['cache_type', 'provider', 'model'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

CACHE_LOOKUP_TIME = Histogram(
    'cache_lookup_time_seconds',
    'Time taken to look up entries in the cache',
    ['cache_type', 'provider', 'model', 'result'],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
    registry=registry
)

CACHE_SIMILARITY_SCORES = Histogram(
    'cache_similarity_scores',
    'Similarity scores for cache lookups',
    ['cache_type', 'provider', 'model'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
    registry=registry
)

CACHE_TTL_REMAINING = Histogram(
    'cache_ttl_remaining_seconds',
    'Remaining TTL for cache hits',
    ['cache_type', 'provider', 'model'],
    buckets=(1, 10, 60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400),
    registry=registry
)

CACHE_MEMORY_USAGE = Gauge(
    'cache_memory_usage_bytes',
    'Estimated memory usage of the cache',
    ['cache_type', 'provider', 'model'],
    registry=registry
)


class EvictionReason(str, Enum):
    """Reasons for evicting entries from the cache."""
    EXPIRED = "expired"
    LRU = "lru"
    MANUAL = "manual"
    FULL = "full"
    ERROR = "error"


class CacheType(str, Enum):
    """Types of caches."""
    EXACT = "exact"
    SEMANTIC = "semantic"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class CacheMetricsManager:
    """Manager for cache metrics.
    
    This class provides methods for tracking and reporting cache metrics,
    making it easier to monitor cache performance and behavior.
    """
    
    def __init__(
        self, 
        cache_type: str = "semantic", 
        provider: str = "generic",
        model: str = "generic",
        auto_update_interval: int = 60
    ):
        """Initialize the cache metrics manager.
        
        Args:
            cache_type: The type of cache (e.g., "semantic", "exact")
            provider: The LLM provider (e.g., "openai", "anthropic")
            model: The model name
            auto_update_interval: Interval for automatic metrics updates (in seconds)
        """
        self.cache_type = cache_type
        self.provider = provider
        self.model = model
        self.auto_update_interval = auto_update_interval
        self.last_update = time.time()
        self._running = False
        self._cache = None
        self._update_task = None
    
    def attach_to_cache(self, cache) -> None:
        """Attach to a cache instance for metrics collection.
        
        Args:
            cache: The cache instance to monitor
        """
        self._cache = cache
        
        # Initialize metrics
        self._update_size_metrics()
        
        # Start automatic updates if enabled
        if self.auto_update_interval > 0:
            self.start_auto_updates()
    
    def start_auto_updates(self) -> None:
        """Start automatic periodic updates of metrics."""
        if self._running:
            return
            
        self._running = True
        
        async def update_loop():
            while self._running:
                try:
                    await self.update_all_metrics()
                    await asyncio.sleep(self.auto_update_interval)
                except Exception as e:
                    logger.error(f"Error updating cache metrics: {e}")
                    await asyncio.sleep(5)  # Shorter backoff on error
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self._update_task = asyncio.create_task(update_loop())
        else:
            logger.warning("No running event loop for cache metrics auto-updates")
    
    def stop_auto_updates(self) -> None:
        """Stop automatic periodic updates of metrics."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None
    
    async def update_all_metrics(self) -> None:
        """Update all metrics for the cache."""
        if not self._cache:
            logger.warning("No cache attached, cannot update metrics")
            return
            
        self._update_size_metrics()
        await self._update_memory_usage()
        self.last_update = time.time()
    
    def _update_size_metrics(self) -> None:
        """Update cache size metrics."""
        if not self._cache:
            return
            
        try:
            # Update current size
            current_size = len(self._cache.cache)
            CACHE_SIZE.labels(
                cache_type=self.cache_type,
                provider=self.provider,
                model=self.model
            ).set(current_size)
            
            # Update capacity if available
            if hasattr(self._cache, 'cache_size'):
                capacity = self._cache.cache_size
                CACHE_CAPACITY.labels(
                    cache_type=self.cache_type,
                    provider=self.provider,
                    model=self.model
                ).set(capacity)
                
                # Update usage ratio
                if capacity > 0:
                    usage_ratio = current_size / capacity
                    CACHE_USAGE_RATIO.labels(
                        cache_type=self.cache_type,
                        provider=self.provider,
                        model=self.model
                    ).set(usage_ratio)
        except Exception as e:
            logger.error(f"Error updating size metrics: {e}")
    
    async def _update_memory_usage(self) -> None:
        """Estimate and update the memory usage of the cache."""
        if not self._cache:
            return
            
        try:
            # This is a very rough estimate based on the number of entries
            # In a real implementation, you'd want to do more precise measurement
            # based on the actual size of the cache entries
            
            import sys
            import json
            
            # Sample a few entries to estimate average size
            sample_size = min(10, len(self._cache.cache))
            total_size = 0
            
            if sample_size > 0:
                keys = list(self._cache.cache.keys())[:sample_size]
                
                for key in keys:
                    # Estimate the size of the key
                    key_size = sys.getsizeof(key)
                    
                    # Estimate the size of the entry
                    entry = self._cache.cache[key]
                    entry_json = json.dumps(entry["response"].dict())
                    entry_size = sys.getsizeof(entry_json)
                    
                    # Estimate the size of the embedding if available
                    embedding_size = 0
                    if hasattr(self._cache, 'embeddings') and key in self._cache.embeddings:
                        embedding = self._cache.embeddings[key]
                        embedding_size = sys.getsizeof(embedding) * 1.5  # Account for overhead
                    
                    total_size += key_size + entry_size + embedding_size
                
                # Calculate average entry size and extrapolate
                avg_entry_size = total_size / sample_size
                estimated_total = avg_entry_size * len(self._cache.cache)
                
                # Add a safety factor for any internal structures
                estimated_total *= 1.2
                
                # Update the metric
                CACHE_MEMORY_USAGE.labels(
                    cache_type=self.cache_type,
                    provider=self.provider,
                    model=self.model
                ).set(estimated_total)
        except Exception as e:
            logger.error(f"Error updating memory usage metric: {e}")
    
    def track_eviction(self, reason: EvictionReason, count: int = 1) -> None:
        """Track cache evictions.
        
        Args:
            reason: The reason for the eviction
            count: The number of entries evicted
        """
        CACHE_EVICTIONS.labels(
            cache_type=self.cache_type,
            provider=self.provider,
            model=self.model,
            reason=reason.value
        ).inc(count)
    
    def track_similarity(self, similarity_score: float) -> None:
        """Track similarity scores for cache lookups.
        
        Args:
            similarity_score: The cosine similarity score
        """
        CACHE_SIMILARITY_SCORES.labels(
            cache_type=self.cache_type,
            provider=self.provider,
            model=self.model
        ).observe(similarity_score)
    
    def track_ttl_remaining(self, seconds_remaining: float) -> None:
        """Track remaining TTL for cache hits.
        
        Args:
            seconds_remaining: The number of seconds remaining before expiry
        """
        CACHE_TTL_REMAINING.labels(
            cache_type=self.cache_type,
            provider=self.provider,
            model=self.model
        ).observe(seconds_remaining)
    
    def track_cleaning_time(self, duration: float) -> None:
        """Track the time taken to clean the cache.
        
        Args:
            duration: The time taken in seconds
        """
        CACHE_CLEANING_TIME.labels(
            cache_type=self.cache_type,
            provider=self.provider,
            model=self.model
        ).observe(duration)
    
    def get_cache_hit_ratio(self) -> float:
        """Get the current cache hit ratio.
        
        Returns:
            The cache hit ratio (0.0 to 1.0)
        """
        hits = LLM_CACHE_HITS.labels(
            provider=self.provider,
            model=self.model,
            cache_type=self.cache_type
        )._value.get()
        
        misses = LLM_CACHE_MISSES.labels(
            provider=self.provider,
            model=self.model,
            cache_type=self.cache_type
        )._value.get()
        
        total = hits + misses
        
        if total > 0:
            return hits / total
        else:
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of cache metrics.
        
        Returns:
            A dictionary containing the metrics summary
        """
        hit_ratio = self.get_cache_hit_ratio()
        
        if not self._cache:
            return {
                "cache_type": self.cache_type,
                "provider": self.provider,
                "model": self.model,
                "hit_ratio": hit_ratio,
                "error": "No cache attached"
            }
        
        result = {
            "cache_type": self.cache_type,
            "provider": self.provider,
            "model": self.model,
            "hit_ratio": hit_ratio,
            "size": len(self._cache.cache),
            "last_updated": self.last_update
        }
        
        # Add capacity if available
        if hasattr(self._cache, 'cache_size'):
            result["capacity"] = self._cache.cache_size
            result["usage_ratio"] = result["size"] / result["capacity"]
        
        # Add threshold if available
        if hasattr(self._cache, 'threshold'):
            result["threshold"] = self._cache.threshold
        
        # Add TTL if available
        if hasattr(self._cache, 'ttl'):
            result["ttl"] = self._cache.ttl
        
        return result


def time_cache_lookup(cache_type: str = "semantic"):
    """Decorator to time cache lookups.
    
    Args:
        cache_type: The type of cache
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract provider and model from instance
            instance = args[0]
            provider = getattr(instance, 'provider', 'generic')
            model = getattr(instance, 'model', 'generic')
            
            start_time = time.time()
            
            # Execute the cache lookup
            result = await func(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record timing
            CACHE_LOOKUP_TIME.labels(
                cache_type=cache_type,
                provider=provider,
                model=model,
                result="hit" if result else "miss"
            ).observe(duration)
            
            return result
        
        return wrapper
    
    return decorator