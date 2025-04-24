from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    push_to_gateway
)
import time
import functools
import logging
import asyncio

logger = logging.getLogger(__name__)

# Create a registry for metrics
registry = CollectorRegistry()

# Define metrics
LLM_REQUESTS = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['provider', 'model', 'status'],
    registry=registry
)

LLM_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM request latency',
    ['provider', 'model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

AGENT_EXECUTIONS = Counter(
    'agent_executions_total',
    'Agent executions',
    ['agent_name', 'status'],
    registry=registry
)

AGENT_LATENCY = Histogram(
    'agent_execution_latency_seconds',
    'Agent execution latency',
    ['agent_name'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

TOKEN_USAGE = Counter(
    'token_usage_total',
    'Token usage',
    ['provider', 'model', 'type'],
    registry=registry
)

CACHE_OPERATIONS = Counter(
    'cache_operations_total',
    'Cache operations',
    ['operation', 'result'],
    registry=registry
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active requests',
    ['endpoint'],
    registry=registry
)

REQUEST_SIZE = Summary(
    'request_size_bytes',
    'Request size in bytes',
    ['endpoint'],
    registry=registry
)

RESPONSE_SIZE = Summary(
    'response_size_bytes',
    'Response size in bytes',
    ['endpoint'],
    registry=registry
)

# Additional metrics
LLM_COST = Counter(
    'llm_cost_usd',
    'Cost of LLM requests in USD',
    ['provider', 'model'],
    registry=registry
)

TOOL_EXECUTIONS = Counter(
    'tool_executions_total',
    'Tool executions',
    ['tool_name', 'status'],
    registry=registry
)

TOOL_LATENCY = Histogram(
    'tool_execution_latency_seconds',
    'Tool execution latency',
    ['tool_name'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=registry
)

MEMORY_OPERATIONS = Counter(
    'memory_operations_total',
    'Memory operations',
    ['operation', 'memory_type'],
    registry=registry
)

CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type'],
    registry=registry
)

ROUTER_DECISIONS = Counter(
    'router_decisions_total',
    'Router decisions',
    ['router_name', 'destination', 'confidence_level'],
    registry=registry
)

CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['name', 'provider'],
    registry=registry
)

RESPONSE_QUALITY = Histogram(
    'response_quality_score',
    'Quality score of responses',
    ['provider', 'model'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)


def track_latency(metric, labels):
    """Decorator to track the latency of a function.

    Args:
        metric: The Histogram metric to update
        labels: The labels for the metric
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
                logger.debug(f"Function {func.__name__} took {duration:.2f} seconds")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
                logger.debug(f"Function {func.__name__} took {duration:.2f} seconds")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_llm_request(provider, model):
    """Decorator to track an LLM request.

    Args:
        provider: The LLM provider
        model: The LLM model
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                LLM_REQUESTS.labels(provider=provider, model=model, status="success").inc()

                # Track token usage
                if hasattr(result, 'usage'):
                    for token_type, count in result.usage.items():
                        TOKEN_USAGE.labels(
                            provider=provider,
                            model=model,
                            type=token_type
                        ).inc(count)

                return result
            except Exception:
                LLM_REQUESTS.labels(provider=provider, model=model, status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                LLM_LATENCY.labels(provider=provider, model=model).observe(duration)

        return wrapper

    return decorator


def track_agent_execution(agent_name):
    """Decorator to track an agent execution.

    Args:
        agent_name: The name of the agent
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="success").inc()
                return result
            except Exception:
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                AGENT_LATENCY.labels(agent_name=agent_name).observe(duration)

        return wrapper

    return decorator


def track_tool_execution(tool_name):
    """Decorator to track a tool execution.

    Args:
        tool_name: The name of the tool
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                TOOL_EXECUTIONS.labels(tool_name=tool_name, status="success").inc()
                return result
            except Exception:
                TOOL_EXECUTIONS.labels(tool_name=tool_name, status="error").inc()
                raise
            finally:
                duration = time.time() - start_time
                TOOL_LATENCY.labels(tool_name=tool_name).observe(duration)

        return wrapper

    return decorator


def track_memory_operation(operation, memory_type):
    """Decorator to track a memory operation.

    Args:
        operation: The type of operation (e.g., "read", "write")
        memory_type: The type of memory (e.g., "short_term", "long_term")
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            MEMORY_OPERATIONS.labels(operation=operation, memory_type=memory_type).inc()
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            MEMORY_OPERATIONS.labels(operation=operation, memory_type=memory_type).inc()
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_router_decision(router_name):
    """Decorator to track a router decision.

    Args:
        router_name: The name of the router
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Extract destination and confidence from the result
            destination = getattr(result, "intent", "unknown")
            confidence = getattr(result, "confidence", 0.0)

            # Determine confidence level
            if confidence >= 0.8:
                confidence_level = "high"
            elif confidence >= 0.5:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            # Record the decision
            ROUTER_DECISIONS.labels(
                router_name=router_name,
                destination=destination,
                confidence_level=confidence_level
            ).inc()

            return result

        return wrapper

    return decorator


def update_circuit_breaker_state(name, provider, state):
    """Update the circuit breaker state metric.

    Args:
        name: The name of the circuit breaker
        provider: The provider (e.g., "openai")
        state: The state (0=closed, 1=open, 2=half-open)
    """
    CIRCUIT_BREAKER_STATE.labels(name=name, provider=provider).set(state)


def record_response_quality(provider, model, quality_score):
    """Record the quality of a response.

    Args:
        provider: The LLM provider
        model: The model name
        quality_score: The quality score (0.0-1.0)
    """
    RESPONSE_QUALITY.labels(provider=provider, model=model).observe(quality_score)


def push_metrics(gateway, job):
    """Push metrics to a Prometheus push gateway.

    Args:
        gateway: The gateway URL
        job: The job name
    """
    try:
        push_to_gateway(gateway, job=job, registry=registry)
        logger.info(f"Pushed metrics to gateway: {gateway}, job: {job}")
    except Exception as e:
        logger.error(f"Failed to push metrics to gateway: {e}")


class MetricsMiddleware:
    """Middleware for collecting API metrics."""

    def __init__(self, app):
        """Initialize the middleware.

        Args:
            app: The FastAPI application
        """
        self.app = app

    async def __call__(self, scope, receive, send):
        """Process an incoming request.

        Args:
            scope: The ASGI scope
            receive: The ASGI receive function
            send: The ASGI send function
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract the path and method
        path = scope["path"]

        # Increment active requests
        ACTIVE_REQUESTS.labels(endpoint=path).inc()

        # Start timing

        # Process the request
        response_status = "unknown"
        response_size = 0

        # Wrap the send function to capture the status code and response size
        original_send = send

        async def wrapped_send(message):
            nonlocal response_status, response_size

            if message["type"] == "http.response.start":
                # Extract the status code
                status_code = message.get("status", 200)
                response_status = f"{status_code}"

            elif message["type"] == "http.response.body":
                # Extract the response body size
                body = message.get("body", b"")
                response_size += len(body)

            await original_send(message)

        try:
            # Process the request
            await self.app(scope, receive, wrapped_send)
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.labels(endpoint=path).dec()

            # Record response size
            RESPONSE_SIZE.labels(endpoint=path).observe(response_size)
