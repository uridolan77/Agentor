"""
Performance monitoring utilities for the Agentor framework.

This module provides tools for monitoring and optimizing performance,
including profiling, metrics collection, and performance analysis.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple, TypeVar, Generic
import time
import logging
import asyncio
import functools
import inspect
import traceback
import statistics
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Return type


@dataclass
class PerformanceMetric:
    """A performance metric with statistics."""
    
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0
    samples: List[float] = field(default_factory=list)
    max_samples: int = 100
    
    @property
    def avg_time(self) -> float:
        """Get the average time.
        
        Returns:
            The average time
        """
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def median_time(self) -> float:
        """Get the median time.
        
        Returns:
            The median time
        """
        return statistics.median(self.samples) if self.samples else 0.0
    
    @property
    def p95_time(self) -> float:
        """Get the 95th percentile time.
        
        Returns:
            The 95th percentile time
        """
        if not self.samples:
            return 0.0
        
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[idx]
    
    @property
    def p99_time(self) -> float:
        """Get the 99th percentile time.
        
        Returns:
            The 99th percentile time
        """
        if not self.samples:
            return 0.0
        
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx]
    
    def add_sample(self, time: float) -> None:
        """Add a time sample.
        
        Args:
            time: The time sample
        """
        self.count += 1
        self.total_time += time
        self.min_time = min(self.min_time, time)
        self.max_time = max(self.max_time, time)
        self.last_time = time
        
        # Add to samples, keeping the most recent
        self.samples.append(time)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "count": self.count,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "median_time": self.median_time,
            "p95_time": self.p95_time,
            "p99_time": self.p99_time,
            "last_time": self.last_time
        }


class PerformanceMonitor:
    """Performance monitor for tracking execution times."""
    
    def __init__(self, max_samples: int = 100):
        """Initialize the performance monitor.
        
        Args:
            max_samples: Maximum number of samples to keep per metric
        """
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.lock = asyncio.Lock()
        self.max_samples = max_samples
    
    async def record(self, name: str, time: float) -> None:
        """Record a time sample.
        
        Args:
            name: The metric name
            time: The time sample
        """
        async with self.lock:
            if name not in self.metrics:
                self.metrics[name] = PerformanceMetric(name=name, max_samples=self.max_samples)
            
            self.metrics[name].add_sample(time)
    
    async def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get a metric by name.
        
        Args:
            name: The metric name
            
        Returns:
            The metric, or None if not found
        """
        async with self.lock:
            return self.metrics.get(name)
    
    async def get_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get all metrics.
        
        Returns:
            Dictionary of metrics
        """
        async with self.lock:
            return self.metrics.copy()
    
    async def clear(self) -> None:
        """Clear all metrics."""
        async with self.lock:
            self.metrics.clear()
    
    async def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all metrics.
        
        Returns:
            Dictionary of metric statistics
        """
        async with self.lock:
            return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring execution time.
        
        Args:
            name: The metric name
            
        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            asyncio.create_task(self.record(name, elapsed))
    
    @asynccontextmanager
    async def measure_async(self, name: str):
        """Async context manager for measuring execution time.
        
        Args:
            name: The metric name
            
        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            await self.record(name, elapsed)


# Create a global performance monitor
performance_monitor = PerformanceMonitor()


def measure(name: Optional[str] = None):
    """Decorator for measuring function execution time.
    
    Args:
        name: The metric name (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        metric_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor.measure(metric_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with performance_monitor.measure_async(metric_name):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


class Profiler:
    """Profiler for detailed performance analysis."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the profiler.
        
        Args:
            enabled: Whether the profiler is enabled
        """
        self.enabled = enabled
        self.current_span = None
        self.spans = []
        self.start_time = None
    
    def start(self) -> None:
        """Start the profiler."""
        if not self.enabled:
            return
        
        self.start_time = time.time()
        self.spans = []
    
    def stop(self) -> Dict[str, Any]:
        """Stop the profiler.
        
        Returns:
            Profiling results
        """
        if not self.enabled or self.start_time is None:
            return {}
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Calculate statistics
        span_stats = {}
        for span in self.spans:
            name = span["name"]
            if name not in span_stats:
                span_stats[name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0,
                    "times": []
                }
            
            elapsed = span["end"] - span["start"]
            span_stats[name]["count"] += 1
            span_stats[name]["total_time"] += elapsed
            span_stats[name]["min_time"] = min(span_stats[name]["min_time"], elapsed)
            span_stats[name]["max_time"] = max(span_stats[name]["max_time"], elapsed)
            span_stats[name]["times"].append(elapsed)
        
        # Calculate percentages and averages
        for name, stats in span_stats.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["percentage"] = (stats["total_time"] / total_time) * 100 if total_time > 0 else 0
            
            # Calculate percentiles
            if stats["times"]:
                sorted_times = sorted(stats["times"])
                stats["median_time"] = sorted_times[len(sorted_times) // 2]
                stats["p95_time"] = sorted_times[int(len(sorted_times) * 0.95)]
                stats["p99_time"] = sorted_times[int(len(sorted_times) * 0.99)]
            
            # Remove raw times
            del stats["times"]
        
        return {
            "total_time": total_time,
            "spans": span_stats
        }
    
    @contextmanager
    def span(self, name: str):
        """Context manager for measuring a span.
        
        Args:
            name: The span name
            
        Yields:
            None
        """
        if not self.enabled or self.start_time is None:
            yield
            return
        
        # Record the span
        span = {
            "name": name,
            "start": time.time(),
            "end": None
        }
        
        # Save the parent span
        parent = self.current_span
        self.current_span = span
        
        try:
            yield
        finally:
            # Record the end time
            span["end"] = time.time()
            
            # Add to spans
            self.spans.append(span)
            
            # Restore the parent span
            self.current_span = parent
    
    @asynccontextmanager
    async def async_span(self, name: str):
        """Async context manager for measuring a span.
        
        Args:
            name: The span name
            
        Yields:
            None
        """
        if not self.enabled or self.start_time is None:
            yield
            return
        
        # Record the span
        span = {
            "name": name,
            "start": time.time(),
            "end": None
        }
        
        # Save the parent span
        parent = self.current_span
        self.current_span = span
        
        try:
            yield
        finally:
            # Record the end time
            span["end"] = time.time()
            
            # Add to spans
            self.spans.append(span)
            
            # Restore the parent span
            self.current_span = parent


class HotspotDetector:
    """Detector for performance hotspots."""
    
    def __init__(
        self,
        threshold_ms: float = 100.0,
        sample_rate: float = 0.01,
        max_hotspots: int = 10
    ):
        """Initialize the hotspot detector.
        
        Args:
            threshold_ms: Threshold in milliseconds for considering a function a hotspot
            sample_rate: Sampling rate (0.0 to 1.0)
            max_hotspots: Maximum number of hotspots to track
        """
        self.threshold_ms = threshold_ms
        self.sample_rate = sample_rate
        self.max_hotspots = max_hotspots
        
        self.hotspots = {}
        self.lock = asyncio.Lock()
    
    async def record(self, func_name: str, elapsed_ms: float, stack: Optional[List[str]] = None) -> None:
        """Record a function execution.
        
        Args:
            func_name: The function name
            elapsed_ms: The elapsed time in milliseconds
            stack: The call stack
        """
        # Skip if below threshold
        if elapsed_ms < self.threshold_ms:
            return
        
        # Skip based on sampling rate
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return
        
        async with self.lock:
            if func_name not in self.hotspots:
                # If we're at the limit, remove the least significant hotspot
                if len(self.hotspots) >= self.max_hotspots:
                    min_hotspot = min(self.hotspots.items(), key=lambda x: x[1]["total_time"])
                    del self.hotspots[min_hotspot[0]]
                
                self.hotspots[func_name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "max_time": 0.0,
                    "samples": [],
                    "stacks": []
                }
            
            self.hotspots[func_name]["count"] += 1
            self.hotspots[func_name]["total_time"] += elapsed_ms
            self.hotspots[func_name]["max_time"] = max(self.hotspots[func_name]["max_time"], elapsed_ms)
            
            # Keep a limited number of samples
            samples = self.hotspots[func_name]["samples"]
            samples.append(elapsed_ms)
            if len(samples) > 10:
                samples.pop(0)
            
            # Keep a limited number of stacks
            if stack:
                stacks = self.hotspots[func_name]["stacks"]
                stacks.append(stack)
                if len(stacks) > 5:
                    stacks.pop(0)
    
    async def get_hotspots(self) -> Dict[str, Dict[str, Any]]:
        """Get the detected hotspots.
        
        Returns:
            Dictionary of hotspots
        """
        async with self.lock:
            # Calculate average times
            result = {}
            for name, data in self.hotspots.items():
                result[name] = {
                    "count": data["count"],
                    "total_time": data["total_time"],
                    "avg_time": data["total_time"] / data["count"] if data["count"] > 0 else 0,
                    "max_time": data["max_time"],
                    "samples": data["samples"],
                    "stacks": data["stacks"]
                }
            
            return result
    
    async def clear(self) -> None:
        """Clear all hotspots."""
        async with self.lock:
            self.hotspots.clear()


def detect_hotspot(threshold_ms: float = 100.0):
    """Decorator for detecting performance hotspots.
    
    Args:
        threshold_ms: Threshold in milliseconds for considering a function a hotspot
        
    Returns:
        Decorated function
    """
    # Create a detector
    detector = HotspotDetector(threshold_ms=threshold_ms)
    
    def decorator(func):
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= threshold_ms:
                    stack = traceback.format_stack()[:-1]  # Exclude this frame
                    asyncio.create_task(detector.record(func_name, elapsed_ms, stack))
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= threshold_ms:
                    stack = traceback.format_stack()[:-1]  # Exclude this frame
                    await detector.record(func_name, elapsed_ms, stack)
        
        # Add the detector to the function
        if asyncio.iscoroutinefunction(func):
            async_wrapper.detector = detector
            return async_wrapper
        else:
            wrapper.detector = detector
            return wrapper
    
    return decorator


class AdaptiveThrottler:
    """Throttler that adapts based on performance metrics."""
    
    def __init__(
        self,
        initial_rate: float = 100.0,
        min_rate: float = 10.0,
        max_rate: float = 1000.0,
        target_latency_ms: float = 100.0,
        adjustment_factor: float = 0.1,
        window_size: int = 10
    ):
        """Initialize the adaptive throttler.
        
        Args:
            initial_rate: Initial rate limit (operations per second)
            min_rate: Minimum rate limit
            max_rate: Maximum rate limit
            target_latency_ms: Target latency in milliseconds
            adjustment_factor: Rate adjustment factor
            window_size: Window size for latency measurements
        """
        self.rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_latency_ms = target_latency_ms
        self.adjustment_factor = adjustment_factor
        self.window_size = window_size
        
        self.latencies = []
        self.last_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to proceed.
        
        This method throttles the caller based on the current rate limit.
        """
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            
            # Calculate the minimum time between operations
            min_interval = 1.0 / self.rate
            
            # If we need to wait, do so
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            
            self.last_time = time.time()
    
    async def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement.
        
        Args:
            latency_ms: The latency in milliseconds
        """
        async with self.lock:
            # Add to the window
            self.latencies.append(latency_ms)
            if len(self.latencies) > self.window_size:
                self.latencies.pop(0)
            
            # Adjust the rate based on the average latency
            if self.latencies:
                avg_latency = sum(self.latencies) / len(self.latencies)
                
                if avg_latency > self.target_latency_ms:
                    # Latency is too high, decrease the rate
                    self.rate = max(
                        self.min_rate,
                        self.rate * (1.0 - self.adjustment_factor)
                    )
                else:
                    # Latency is acceptable, increase the rate
                    self.rate = min(
                        self.max_rate,
                        self.rate * (1.0 + self.adjustment_factor)
                    )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get throttler statistics.
        
        Returns:
            Dictionary of throttler statistics
        """
        async with self.lock:
            return {
                "rate": self.rate,
                "min_rate": self.min_rate,
                "max_rate": self.max_rate,
                "target_latency_ms": self.target_latency_ms,
                "avg_latency_ms": sum(self.latencies) / len(self.latencies) if self.latencies else 0,
                "window_size": self.window_size,
                "current_window_size": len(self.latencies)
            }


@asynccontextmanager
async def throttled_operation(throttler: AdaptiveThrottler, record_latency: bool = True):
    """Async context manager for throttled operations.
    
    Args:
        throttler: The throttler to use
        record_latency: Whether to record the operation latency
        
    Yields:
        None
    """
    # Acquire permission to proceed
    await throttler.acquire()
    
    start_time = time.time()
    try:
        yield
    finally:
        if record_latency:
            elapsed_ms = (time.time() - start_time) * 1000
            await throttler.record_latency(elapsed_ms)
