"""
Example demonstrating the enhanced error handling and resilience features in Agentor.

This example shows how to use:
- Adaptive retry strategies
- Error correlation
- Bulkhead pattern for resource isolation
- Timeout management
- Circuit breaker pattern
"""

import asyncio
import logging
import time
import random
import sys
from typing import Dict, Any, List, Optional

from agentor.llm_gateway.utils.error_handler import ErrorContext, ErrorSeverity, ErrorCategory
from agentor.llm_gateway.utils.error_correlation import error_correlator, setup_error_correlation
from agentor.llm_gateway.utils.retry import (
    RetryConfig, RetryStrategy, retry_async, retry_async_decorator,
    NETWORK_RETRY_CONFIG, RATE_LIMIT_RETRY_CONFIG, ADAPTIVE_RETRY_CONFIG
)
from agentor.llm_gateway.utils.bulkhead import bulkhead_manager, with_bulkhead
from agentor.llm_gateway.utils.timeout import timeout_manager, with_timeout, TimeoutStrategy
from agentor.llm_gateway.utils.circuit_breaker import CircuitBreaker, CircuitBreakerState
from agentor.llm_gateway.utils.degradation import DegradationManager, DegradationLevel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Simulated services with different failure modes
class NetworkService:
    """Simulated service with network failures."""
    
    def __init__(self, failure_rate: float = 0.3):
        """Initialize the service.
        
        Args:
            failure_rate: Probability of a failure (0.0 to 1.0)
        """
        self.failure_rate = failure_rate
        self.circuit_breaker = CircuitBreaker(
            name="network_service",
            failure_threshold=5,
            recovery_timeout=10.0,
            half_open_max_calls=3
        )
    
    @retry_async_decorator("network_service", "fetch_data", NETWORK_RETRY_CONFIG)
    @with_timeout("network_service", "fetch_data", TimeoutStrategy.ADAPTIVE, 2.0, 10.0)
    async def fetch_data(self, request_id: str) -> Dict[str, Any]:
        """Fetch data from the service.
        
        Args:
            request_id: The ID of the request
            
        Returns:
            The fetched data
            
        Raises:
            ConnectionError: If a network error occurs
            TimeoutError: If the request times out
        """
        # Check if the circuit breaker is open
        if not self.circuit_breaker.allow_request():
            logger.warning(f"Circuit breaker is open for network_service")
            raise ConnectionError("Circuit breaker is open")
        
        try:
            # Simulate network latency
            delay = random.uniform(0.1, 3.0)
            await asyncio.sleep(delay)
            
            # Simulate failures
            if random.random() < self.failure_rate:
                # Record the failure in the circuit breaker
                self.circuit_breaker.record_failure()
                
                # Raise a network error
                raise ConnectionError(f"Network error while fetching data for {request_id}")
            
            # Record the success in the circuit breaker
            self.circuit_breaker.record_success()
            
            # Return the data
            return {
                "request_id": request_id,
                "timestamp": time.time(),
                "data": f"Data for {request_id}"
            }
        
        except Exception as e:
            # Record the failure in the circuit breaker
            self.circuit_breaker.record_failure()
            
            # Re-raise the exception
            raise


class RateLimitedService:
    """Simulated service with rate limiting."""
    
    def __init__(self, rate_limit: int = 5, window_size: float = 10.0):
        """Initialize the service.
        
        Args:
            rate_limit: Maximum number of requests in the window
            window_size: Size of the rate limiting window in seconds
        """
        self.rate_limit = rate_limit
        self.window_size = window_size
        self.requests = []
    
    @retry_async_decorator("rate_limited_service", "process_request", RATE_LIMIT_RETRY_CONFIG)
    @with_bulkhead("rate_limited_service", max_concurrent=3, max_queue_size=5)
    async def process_request(self, request_id: str) -> Dict[str, Any]:
        """Process a request.
        
        Args:
            request_id: The ID of the request
            
        Returns:
            The processed data
            
        Raises:
            Exception: If the rate limit is exceeded
        """
        # Check rate limit
        current_time = time.time()
        
        # Remove old requests
        self.requests = [t for t in self.requests if current_time - t < self.window_size]
        
        # Check if we're over the limit
        if len(self.requests) >= self.rate_limit:
            raise Exception(f"Rate limit exceeded for {request_id}")
        
        # Add the current request
        self.requests.append(current_time)
        
        # Simulate processing
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        # Return the result
        return {
            "request_id": request_id,
            "timestamp": current_time,
            "result": f"Processed {request_id}"
        }


class UnreliableService:
    """Simulated service with various failure modes."""
    
    def __init__(self, failure_rate: float = 0.4):
        """Initialize the service.
        
        Args:
            failure_rate: Probability of a failure (0.0 to 1.0)
        """
        self.failure_rate = failure_rate
        self.degradation_manager = DegradationManager()
    
    @retry_async_decorator("unreliable_service", "execute_operation", ADAPTIVE_RETRY_CONFIG)
    @with_timeout("unreliable_service", "execute_operation", TimeoutStrategy.DYNAMIC, 1.0, 5.0)
    async def execute_operation(self, operation_id: str) -> Dict[str, Any]:
        """Execute an operation.
        
        Args:
            operation_id: The ID of the operation
            
        Returns:
            The operation result
            
        Raises:
            Exception: If the operation fails
        """
        # Check degradation level
        degradation_level = self.degradation_manager.get_current_level()
        
        # Adjust behavior based on degradation level
        if degradation_level == DegradationLevel.CRITICAL:
            # In critical degradation, fail fast
            raise Exception(f"Service in critical degradation, rejecting {operation_id}")
        
        elif degradation_level == DegradationLevel.SEVERE:
            # In severe degradation, increase failure rate
            adjusted_failure_rate = min(0.8, self.failure_rate * 2)
        
        elif degradation_level == DegradationLevel.MODERATE:
            # In moderate degradation, use normal failure rate
            adjusted_failure_rate = self.failure_rate
        
        else:
            # In normal operation, reduce failure rate
            adjusted_failure_rate = max(0.1, self.failure_rate / 2)
        
        # Simulate variable latency
        if degradation_level in (DegradationLevel.SEVERE, DegradationLevel.CRITICAL):
            # Higher latency in degraded states
            delay = random.uniform(1.0, 4.0)
        else:
            delay = random.uniform(0.2, 2.0)
        
        await asyncio.sleep(delay)
        
        # Simulate failures
        if random.random() < adjusted_failure_rate:
            # Update degradation level based on failures
            self.degradation_manager.record_failure()
            
            # Create an error context
            error_context = ErrorContext(
                component="unreliable_service",
                operation="execute_operation",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SERVICE_ERROR,
                metadata={"operation_id": operation_id}
            )
            
            # Raise an exception
            error = Exception(f"Operation {operation_id} failed")
            
            # Add to error correlation
            await error_correlator.add_error(error, error_context)
            
            raise error
        
        # Record success
        self.degradation_manager.record_success()
        
        # Return the result
        return {
            "operation_id": operation_id,
            "timestamp": time.time(),
            "result": f"Executed {operation_id}",
            "degradation_level": degradation_level.value
        }


async def network_service_example():
    """Example using the network service with retries and circuit breaker."""
    logger.info("=== Network Service Example ===")
    
    # Create the service
    service = NetworkService(failure_rate=0.3)
    
    # Make multiple requests
    for i in range(20):
        request_id = f"request-{i}"
        
        try:
            logger.info(f"Sending request {request_id}")
            result = await service.fetch_data(request_id)
            logger.info(f"Request {request_id} succeeded: {result}")
        
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
        
        # Check circuit breaker state
        state = service.circuit_breaker.get_state()
        if state != CircuitBreakerState.CLOSED:
            logger.warning(
                f"Circuit breaker state: {state.value}, "
                f"failures: {service.circuit_breaker.failure_count}, "
                f"last failure: {service.circuit_breaker.last_failure_time}"
            )
        
        # Small delay between requests
        await asyncio.sleep(0.5)


async def rate_limited_service_example():
    """Example using the rate limited service with bulkhead pattern."""
    logger.info("\n=== Rate Limited Service Example ===")
    
    # Create the service
    service = RateLimitedService(rate_limit=5, window_size=10.0)
    
    # Create tasks for concurrent requests
    tasks = []
    for i in range(15):
        request_id = f"request-{i}"
        task = asyncio.create_task(process_rate_limited_request(service, request_id))
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Get bulkhead stats
    stats = bulkhead_manager.get_stats("rate_limited_service")
    logger.info(f"Bulkhead stats: {stats}")


async def process_rate_limited_request(service: RateLimitedService, request_id: str):
    """Process a request to the rate limited service.
    
    Args:
        service: The rate limited service
        request_id: The ID of the request
    """
    try:
        logger.info(f"Sending request {request_id}")
        result = await service.process_request(request_id)
        logger.info(f"Request {request_id} succeeded: {result}")
    
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")


async def unreliable_service_example():
    """Example using the unreliable service with adaptive timeouts and degradation."""
    logger.info("\n=== Unreliable Service Example ===")
    
    # Create the service
    service = UnreliableService(failure_rate=0.4)
    
    # Make multiple requests
    for i in range(30):
        operation_id = f"operation-{i}"
        
        try:
            logger.info(f"Executing operation {operation_id}")
            result = await service.execute_operation(operation_id)
            logger.info(f"Operation {operation_id} succeeded: {result}")
        
        except Exception as e:
            logger.error(f"Operation {operation_id} failed: {str(e)}")
        
        # Check degradation level
        level = service.degradation_manager.get_current_level()
        if level != DegradationLevel.NORMAL:
            logger.warning(
                f"Degradation level: {level.value}, "
                f"failure rate: {service.degradation_manager.get_failure_rate():.2f}"
            )
        
        # Get timeout stats
        if i % 10 == 0:
            stats = timeout_manager.get_stats("unreliable_service", "execute_operation")
            logger.info(f"Timeout stats: {stats}")
        
        # Small delay between operations
        await asyncio.sleep(0.2)


async def error_correlation_example():
    """Example demonstrating error correlation."""
    logger.info("\n=== Error Correlation Example ===")
    
    # Set up error correlation
    await setup_error_correlation()
    
    # Create error contexts
    components = ["api", "database", "auth", "cache"]
    operations = ["get", "create", "update", "delete", "validate"]
    categories = [
        ErrorCategory.NETWORK_ERROR,
        ErrorCategory.TIMEOUT_ERROR,
        ErrorCategory.SERVICE_ERROR,
        ErrorCategory.VALIDATION_ERROR
    ]
    
    # Generate some errors
    for i in range(50):
        component = random.choice(components)
        operation = random.choice(operations)
        category = random.choice(categories)
        
        # Create an error context
        error_context = ErrorContext(
            component=component,
            operation=operation,
            severity=ErrorSeverity.ERROR,
            category=category,
            metadata={"error_id": f"error-{i}"}
        )
        
        # Create an error
        error = Exception(f"Error in {component}.{operation}: {category.value}")
        
        # Add to error correlation
        await error_correlator.add_error(error, error_context)
        
        # Small delay between errors
        await asyncio.sleep(0.1)
    
    # Get error patterns
    patterns = error_correlator.get_all_patterns()
    logger.info(f"Detected {len(patterns)} error patterns:")
    
    for pattern in patterns:
        logger.info(f"Pattern: {pattern.pattern_id}")
        logger.info(f"Description: {pattern.description}")
        logger.info(f"Count: {pattern.count}")
        logger.info(f"First seen: {pattern.first_seen}")
        logger.info(f"Last seen: {pattern.last_seen}")
        logger.info("---")
    
    # Get recent errors
    recent_errors = error_correlator.get_recent_errors(5)
    logger.info(f"Recent errors:")
    
    for error in recent_errors:
        logger.info(f"Error ID: {error.error_id}")
        logger.info(f"Type: {error.error_type}")
        logger.info(f"Message: {error.message}")
        logger.info(f"Component: {error.component}")
        logger.info(f"Operation: {error.operation}")
        logger.info(f"Category: {error.category.value}")
        logger.info("---")


async def main():
    """Run all resilience examples."""
    try:
        await network_service_example()
        await rate_limited_service_example()
        await unreliable_service_example()
        await error_correlation_example()
    
    except KeyboardInterrupt:
        logger.info("Examples interrupted")
    
    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
