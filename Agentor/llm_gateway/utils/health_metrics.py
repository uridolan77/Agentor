"""
System health monitoring for the LLM Gateway Agent System.

This module provides metrics and utilities for monitoring the health of the system, including:
- System resource usage
- Service health checks
- Dependency health monitoring
- Circuit breaker status
- Rate limiting status
- Quota monitoring
"""

import time
import logging
import asyncio
import os
import platform
import psutil
from typing import Dict, Any, List, Optional, Callable, Union, Set
from enum import Enum
from datetime import datetime

from prometheus_client import Counter, Histogram, Gauge, Summary

from agentor.llm_gateway.utils.metrics import (
    CIRCUIT_BREAKER_STATE, registry
)

logger = logging.getLogger(__name__)


# System health metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'Memory usage in bytes',
    registry=registry
)

SYSTEM_MEMORY_USAGE_PERCENT = Gauge(
    'system_memory_usage_percent',
    'Memory usage percentage',
    registry=registry
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_bytes',
    'Disk usage in bytes',
    ['path'],
    registry=registry
)

SYSTEM_DISK_USAGE_PERCENT = Gauge(
    'system_disk_usage_percent',
    'Disk usage percentage',
    ['path'],
    registry=registry
)

SYSTEM_NETWORK_IO = Counter(
    'system_network_io_bytes',
    'Network IO in bytes',
    ['direction'],
    registry=registry
)

SYSTEM_OPEN_FILES = Gauge(
    'system_open_files',
    'Number of open files',
    registry=registry
)

SYSTEM_OPEN_CONNECTIONS = Gauge(
    'system_open_connections',
    'Number of open network connections',
    ['state'],
    registry=registry
)

# Service health metrics
SERVICE_HEALTH = Gauge(
    'service_health',
    'Health status of services (0=unhealthy, 1=degraded, 2=healthy)',
    ['service'],
    registry=registry
)

SERVICE_UPTIME = Gauge(
    'service_uptime_seconds',
    'Service uptime in seconds',
    ['service'],
    registry=registry
)

SERVICE_LAST_CHECK = Gauge(
    'service_last_check_timestamp',
    'Timestamp of the last health check',
    ['service'],
    registry=registry
)

# Dependency health metrics
DEPENDENCY_HEALTH = Gauge(
    'dependency_health',
    'Health status of dependencies (0=unhealthy, 1=degraded, 2=healthy)',
    ['dependency', 'endpoint'],
    registry=registry
)

DEPENDENCY_LATENCY = Histogram(
    'dependency_latency_seconds',
    'Latency of dependency calls',
    ['dependency', 'endpoint'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

DEPENDENCY_ERRORS = Counter(
    'dependency_errors_total',
    'Number of errors from dependencies',
    ['dependency', 'endpoint', 'error_type'],
    registry=registry
)

# Circuit breaker metrics
CIRCUIT_BREAKER_TRIP_COUNT = Counter(
    'circuit_breaker_trip_count_total',
    'Number of times a circuit breaker has tripped',
    ['name', 'provider'],
    registry=registry
)

CIRCUIT_BREAKER_SUCCESS_COUNT = Counter(
    'circuit_breaker_success_count_total',
    'Number of successful calls through a circuit breaker',
    ['name', 'provider'],
    registry=registry
)

CIRCUIT_BREAKER_FAILURE_COUNT = Counter(
    'circuit_breaker_failure_count_total',
    'Number of failed calls through a circuit breaker',
    ['name', 'provider'],
    registry=registry
)

CIRCUIT_BREAKER_REJECTED_COUNT = Counter(
    'circuit_breaker_rejected_count_total',
    'Number of rejected calls due to an open circuit breaker',
    ['name', 'provider'],
    registry=registry
)

CIRCUIT_BREAKER_HALF_OPEN_COUNT = Counter(
    'circuit_breaker_half_open_count_total',
    'Number of times a circuit breaker has gone to half-open state',
    ['name', 'provider'],
    registry=registry
)

# Rate limiting metrics
RATE_LIMIT_EXCEEDED_COUNT = Counter(
    'rate_limit_exceeded_count_total',
    'Number of times a rate limit has been exceeded',
    ['provider', 'limit_type'],
    registry=registry
)

RATE_LIMIT_CURRENT = Gauge(
    'rate_limit_current',
    'Current rate limit usage',
    ['provider', 'limit_type'],
    registry=registry
)

RATE_LIMIT_MAX = Gauge(
    'rate_limit_max',
    'Maximum rate limit',
    ['provider', 'limit_type'],
    registry=registry
)

# Quota metrics
QUOTA_USAGE = Gauge(
    'quota_usage',
    'Current quota usage',
    ['provider', 'quota_type'],
    registry=registry
)

QUOTA_LIMIT = Gauge(
    'quota_limit',
    'Quota limit',
    ['provider', 'quota_type'],
    registry=registry
)

QUOTA_RESET_TIME = Gauge(
    'quota_reset_time_seconds',
    'Time until quota reset in seconds',
    ['provider', 'quota_type'],
    registry=registry
)

# Process metrics
PROCESS_CPU_USAGE = Gauge(
    'process_cpu_usage_percent',
    'Process CPU usage percentage',
    registry=registry
)

PROCESS_MEMORY_USAGE = Gauge(
    'process_memory_usage_bytes',
    'Process memory usage in bytes',
    registry=registry
)

PROCESS_THREAD_COUNT = Gauge(
    'process_thread_count',
    'Number of threads in the process',
    registry=registry
)

PROCESS_OPEN_FDS = Gauge(
    'process_open_fds',
    'Number of open file descriptors',
    registry=registry
)


class HealthStatus(int, Enum):
    """Health status of a service or dependency."""
    UNHEALTHY = 0
    DEGRADED = 1
    HEALTHY = 2


class SystemHealthMonitor:
    """Monitor system health metrics."""

    def __init__(self, interval: int = 60):
        """Initialize the system health monitor.

        Args:
            interval: The interval in seconds between health checks
        """
        self.interval = interval
        self.running = False
        self.task = None
        self.start_time = time.time()

    async def start(self):
        """Start the health monitoring."""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("System health monitoring started")

    async def stop(self):
        """Stop the health monitoring."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info("System health monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_process_metrics()

                # Update service uptime
                uptime = time.time() - self.start_time
                SERVICE_UPTIME.labels(service="llm_gateway").set(uptime)

                # Wait for the next interval
                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in system health monitoring: {str(e)}")
                await asyncio.sleep(self.interval)

    def _collect_system_metrics(self):
        """Collect system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        SYSTEM_CPU_USAGE.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.used)
        SYSTEM_MEMORY_USAGE_PERCENT.set(memory.percent)

        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                SYSTEM_DISK_USAGE.labels(path=partition.mountpoint).set(usage.used)
                SYSTEM_DISK_USAGE_PERCENT.labels(path=partition.mountpoint).set(usage.percent)
            except PermissionError:
                # Skip partitions that can't be accessed
                pass

        # Network IO
        net_io = psutil.net_io_counters()
        SYSTEM_NETWORK_IO.labels(direction="sent").inc(net_io.bytes_sent)
        SYSTEM_NETWORK_IO.labels(direction="received").inc(net_io.bytes_recv)

        # Open files
        try:
            open_files = len(psutil.Process().open_files())
            SYSTEM_OPEN_FILES.set(open_files)
        except psutil.AccessDenied:
            # Skip if access is denied
            pass

        # Open connections
        try:
            connections = psutil.Process().connections()
            conn_states = {}
            for conn in connections:
                state = conn.status
                conn_states[state] = conn_states.get(state, 0) + 1

            for state, count in conn_states.items():
                SYSTEM_OPEN_CONNECTIONS.labels(state=state).set(count)
        except psutil.AccessDenied:
            # Skip if access is denied
            pass

    def _collect_process_metrics(self):
        """Collect process metrics."""
        process = psutil.Process()

        # CPU usage
        try:
            cpu_percent = process.cpu_percent(interval=1)
            PROCESS_CPU_USAGE.set(cpu_percent)
        except psutil.AccessDenied:
            pass

        # Memory usage
        try:
            memory_info = process.memory_info()
            PROCESS_MEMORY_USAGE.set(memory_info.rss)
        except psutil.AccessDenied:
            pass

        # Thread count
        try:
            thread_count = len(process.threads())
            PROCESS_THREAD_COUNT.set(thread_count)
        except psutil.AccessDenied:
            pass

        # Open file descriptors
        try:
            if platform.system() != 'Windows':
                open_fds = process.num_fds()
                PROCESS_OPEN_FDS.set(open_fds)
        except (psutil.AccessDenied, AttributeError):
            pass


class ServiceHealthCheck:
    """Health check for a service."""

    def __init__(
        self,
        service_name: str,
        check_func: Callable[[], Union[bool, HealthStatus]],
        interval: int = 60
    ):
        """Initialize the service health check.

        Args:
            service_name: The name of the service
            check_func: A function that returns the health status
            interval: The interval in seconds between health checks
        """
        self.service_name = service_name
        self.check_func = check_func
        self.interval = interval
        self.running = False
        self.task = None

    async def start(self):
        """Start the health check."""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._check_loop())
        logger.info(f"Health check for service {self.service_name} started")

    async def stop(self):
        """Stop the health check."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info(f"Health check for service {self.service_name} stopped")

    async def _check_loop(self):
        """Main health check loop."""
        while self.running:
            try:
                # Run the health check
                status = await self._run_check()

                # Update the health status
                if isinstance(status, bool):
                    status = HealthStatus.HEALTHY if status else HealthStatus.UNHEALTHY

                SERVICE_HEALTH.labels(service=self.service_name).set(status.value)

                # Update the last check timestamp
                SERVICE_LAST_CHECK.labels(service=self.service_name).set(time.time())

                # Wait for the next interval
                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in health check for service {self.service_name}: {str(e)}")
                SERVICE_HEALTH.labels(service=self.service_name).set(HealthStatus.UNHEALTHY.value)
                await asyncio.sleep(self.interval)

    async def _run_check(self) -> HealthStatus:
        """Run the health check.

        Returns:
            The health status
        """
        if asyncio.iscoroutinefunction(self.check_func):
            status = await self.check_func()
        else:
            status = self.check_func()

        return status


class DependencyHealthCheck:
    """Health check for a dependency."""

    def __init__(
        self,
        dependency_name: str,
        endpoint: str,
        check_func: Callable[[], Union[bool, HealthStatus]],
        interval: int = 60
    ):
        """Initialize the dependency health check.

        Args:
            dependency_name: The name of the dependency
            endpoint: The endpoint to check
            check_func: A function that returns the health status
            interval: The interval in seconds between health checks
        """
        self.dependency_name = dependency_name
        self.endpoint = endpoint
        self.check_func = check_func
        self.interval = interval
        self.running = False
        self.task = None

    async def start(self):
        """Start the health check."""
        if self.running:
            return

        self.running = True
        self.task = asyncio.create_task(self._check_loop())
        logger.info(f"Health check for dependency {self.dependency_name} ({self.endpoint}) started")

    async def stop(self):
        """Stop the health check."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info(f"Health check for dependency {self.dependency_name} ({self.endpoint}) stopped")

    async def _check_loop(self):
        """Main health check loop."""
        while self.running:
            try:
                # Run the health check
                start_time = time.time()
                status = await self._run_check()
                duration = time.time() - start_time

                # Update the health status
                if isinstance(status, bool):
                    status = HealthStatus.HEALTHY if status else HealthStatus.UNHEALTHY

                DEPENDENCY_HEALTH.labels(
                    dependency=self.dependency_name,
                    endpoint=self.endpoint
                ).set(status.value)

                # Update the latency
                DEPENDENCY_LATENCY.labels(
                    dependency=self.dependency_name,
                    endpoint=self.endpoint
                ).observe(duration)

                # Wait for the next interval
                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(
                    f"Error in health check for dependency {self.dependency_name} "
                    f"({self.endpoint}): {str(e)}"
                )

                DEPENDENCY_HEALTH.labels(
                    dependency=self.dependency_name,
                    endpoint=self.endpoint
                ).set(HealthStatus.UNHEALTHY.value)

                DEPENDENCY_ERRORS.labels(
                    dependency=self.dependency_name,
                    endpoint=self.endpoint,
                    error_type=type(e).__name__
                ).inc()

                await asyncio.sleep(self.interval)

    async def _run_check(self) -> HealthStatus:
        """Run the health check.

        Returns:
            The health status
        """
        if asyncio.iscoroutinefunction(self.check_func):
            status = await self.check_func()
        else:
            status = self.check_func()

        return status


class CircuitBreakerMonitor:
    """Monitor circuit breaker status."""

    def __init__(self):
        """Initialize the circuit breaker monitor."""
        self.breakers = {}

    def register_breaker(self, name: str, provider: str, breaker):
        """Register a circuit breaker for monitoring.

        Args:
            name: The name of the circuit breaker
            provider: The provider (e.g., "openai")
            breaker: The circuit breaker object
        """
        self.breakers[(name, provider)] = breaker
        logger.info(f"Registered circuit breaker: {name} ({provider})")

    def unregister_breaker(self, name: str, provider: str):
        """Unregister a circuit breaker.

        Args:
            name: The name of the circuit breaker
            provider: The provider
        """
        if (name, provider) in self.breakers:
            del self.breakers[(name, provider)]
            logger.info(f"Unregistered circuit breaker: {name} ({provider})")

    def update_metrics(self):
        """Update circuit breaker metrics."""
        for (name, provider), breaker in self.breakers.items():
            # Update state
            state_value = 0  # Closed
            if breaker.state == "open":
                state_value = 1
            elif breaker.state == "half-open":
                state_value = 2

            CIRCUIT_BREAKER_STATE.labels(name=name, provider=provider).set(state_value)

            # Update counters - directly set the values instead of incrementing by difference
            if hasattr(breaker, 'success_count'):
                # Set the counter directly to the current value
                current_success = CIRCUIT_BREAKER_SUCCESS_COUNT.labels(
                    name=name,
                    provider=provider
                )
                # Calculate the increment needed
                increment = max(0, breaker.success_count - current_success._value.get())
                if increment > 0:
                    current_success.inc(increment)

            if hasattr(breaker, 'failure_count'):
                # Set the counter directly to the current value
                current_failure = CIRCUIT_BREAKER_FAILURE_COUNT.labels(
                    name=name,
                    provider=provider
                )
                # Calculate the increment needed
                increment = max(0, breaker.failure_count - current_failure._value.get())
                if increment > 0:
                    current_failure.inc(increment)

            if hasattr(breaker, 'rejected_count'):
                # Set the counter directly to the current value
                current_rejected = CIRCUIT_BREAKER_REJECTED_COUNT.labels(
                    name=name,
                    provider=provider
                )
                # Calculate the increment needed
                increment = max(0, breaker.rejected_count - current_rejected._value.get())
                if increment > 0:
                    current_rejected.inc(increment)


class RateLimitMonitor:
    """Monitor rate limit status."""

    def update_rate_limit(
        self,
        provider: str,
        limit_type: str,
        current: int,
        maximum: int,
        exceeded: bool = False
    ):
        """Update rate limit metrics.

        Args:
            provider: The provider (e.g., "openai")
            limit_type: The type of rate limit (e.g., "requests_per_minute")
            current: The current usage
            maximum: The maximum allowed
            exceeded: Whether the rate limit was exceeded
        """
        RATE_LIMIT_CURRENT.labels(
            provider=provider,
            limit_type=limit_type
        ).set(current)

        RATE_LIMIT_MAX.labels(
            provider=provider,
            limit_type=limit_type
        ).set(maximum)

        if exceeded:
            RATE_LIMIT_EXCEEDED_COUNT.labels(
                provider=provider,
                limit_type=limit_type
            ).inc()


class QuotaMonitor:
    """Monitor quota status."""

    def update_quota(
        self,
        provider: str,
        quota_type: str,
        usage: int,
        limit: int,
        reset_time: Optional[int] = None
    ):
        """Update quota metrics.

        Args:
            provider: The provider (e.g., "openai")
            quota_type: The type of quota (e.g., "tokens_per_day")
            usage: The current usage
            limit: The quota limit
            reset_time: The time until reset in seconds (optional)
        """
        QUOTA_USAGE.labels(
            provider=provider,
            quota_type=quota_type
        ).set(usage)

        QUOTA_LIMIT.labels(
            provider=provider,
            quota_type=quota_type
        ).set(limit)

        if reset_time is not None:
            QUOTA_RESET_TIME.labels(
                provider=provider,
                quota_type=quota_type
            ).set(reset_time)


# Create global instances
system_monitor = SystemHealthMonitor()
circuit_breaker_monitor = CircuitBreakerMonitor()
rate_limit_monitor = RateLimitMonitor()
quota_monitor = QuotaMonitor()


async def start_health_monitoring():
    """Start all health monitoring."""
    await system_monitor.start()

    # Register service health checks
    service_check = ServiceHealthCheck(
        service_name="llm_gateway",
        check_func=lambda: True,  # Simple check that always returns healthy
        interval=60
    )
    await service_check.start()

    logger.info("Health monitoring started")


async def stop_health_monitoring():
    """Stop all health monitoring."""
    await system_monitor.stop()
    logger.info("Health monitoring stopped")
