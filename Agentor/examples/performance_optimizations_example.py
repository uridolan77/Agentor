"""
Example demonstrating performance optimizations in the Agentor framework.

This example shows how to use the performance optimization utilities:
- Caching for expensive operations
- Connection pooling for database operations
- Batching for vector database operations
- Performance monitoring and profiling
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional
import numpy as np

from agentor.core.utils.caching import (
    Cache,
    InMemoryCache,
    cached
)
from agentor.core.utils.connection_pool import (
    ConnectionPool,
    DatabaseConnectionPool,
    connection_pool_manager,
    with_connection
)
from agentor.components.memory.batch_operations import (
    BatchedVectorDB,
    VectorDBBatchProcessor
)
from agentor.core.utils.performance import (
    performance_monitor,
    measure,
    Profiler,
    detect_hotspot,
    AdaptiveThrottler,
    throttled_operation
)
from agentor.components.memory.vector_db import InMemoryVectorDB
from agentor.components.memory.embedding import MockEmbeddingProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example 1: Caching for expensive operations
@cached(ttl=60)
async def expensive_calculation(a: int, b: int) -> int:
    """Simulate an expensive calculation.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Result of the calculation
    """
    logger.info(f"Performing expensive calculation: {a} * {b}")
    await asyncio.sleep(1)  # Simulate expensive operation
    return a * b


# Example 2: Connection pooling for database operations
class MockDatabase:
    """Mock database for demonstration purposes."""
    
    def __init__(self, connection_string: str):
        """Initialize the mock database.
        
        Args:
            connection_string: The connection string
        """
        self.connection_string = connection_string
        self.query_count = 0
    
    async def connect(self):
        """Connect to the database.
        
        Returns:
            A database connection
        """
        logger.info(f"Connecting to database: {self.connection_string}")
        await asyncio.sleep(0.5)  # Simulate connection time
        return {"connection_id": random.randint(1000, 9999), "db": self}
    
    async def close(self, connection):
        """Close a database connection.
        
        Args:
            connection: The connection to close
        """
        logger.info(f"Closing database connection: {connection['connection_id']}")
        await asyncio.sleep(0.1)  # Simulate closing time
    
    async def validate(self, connection) -> bool:
        """Validate a database connection.
        
        Args:
            connection: The connection to validate
            
        Returns:
            True if the connection is valid
        """
        return True
    
    async def query(self, connection, query: str) -> List[Dict[str, Any]]:
        """Execute a query.
        
        Args:
            connection: The database connection
            query: The query to execute
            
        Returns:
            Query results
        """
        logger.info(f"Executing query on connection {connection['connection_id']}: {query}")
        self.query_count += 1
        await asyncio.sleep(0.2)  # Simulate query time
        return [{"result": f"Data for {query}"}]


# Create a mock database
mock_db = MockDatabase("postgresql://localhost:5432/mydb")


# Create a connection pool
async def setup_db_pool():
    """Set up the database connection pool."""
    return await connection_pool_manager.create_pool(
        name="mock_db",
        pool_class=DatabaseConnectionPool,
        connection_factory=mock_db.connect,
        connection_validator=mock_db.validate,
        connection_closer=mock_db.close,
        min_size=2,
        max_size=5
    )


# Example function using the connection pool
@with_connection("mock_db")
async def execute_query(connection, query: str) -> List[Dict[str, Any]]:
    """Execute a query using a pooled connection.
    
    Args:
        connection: The database connection
        query: The query to execute
        
    Returns:
        Query results
    """
    return await mock_db.query(connection, query)


# Example 3: Batching for vector database operations
async def setup_vector_db():
    """Set up the vector database with batching."""
    # Create a vector database
    vector_db = InMemoryVectorDB(dimension=384)
    
    # Wrap with batching
    batched_db = BatchedVectorDB(
        vector_db=vector_db,
        add_batch_size=50,
        max_wait_time=0.1
    )
    
    return batched_db


# Example 4: Performance monitoring
@measure("generate_embedding")
async def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for a text.
    
    Args:
        text: The text to embed
        
    Returns:
        The embedding vector
    """
    # Simulate embedding generation
    await asyncio.sleep(0.05)
    return [random.random() for _ in range(384)]


# Example 5: Hotspot detection
@detect_hotspot(threshold_ms=50)
async def potential_hotspot(iterations: int) -> int:
    """Function that might be a performance hotspot.
    
    Args:
        iterations: Number of iterations
        
    Returns:
        Result of the computation
    """
    result = 0
    for i in range(iterations):
        result += i
        if i % 1000 == 0:
            await asyncio.sleep(0.001)  # Simulate some work
    
    return result


# Example 6: Adaptive throttling
async def run_throttled_operations(throttler: AdaptiveThrottler, count: int):
    """Run operations with adaptive throttling.
    
    Args:
        throttler: The throttler to use
        count: Number of operations to run
    """
    for i in range(count):
        async with throttled_operation(throttler):
            # Simulate an operation with variable latency
            latency = 0.05 + (0.1 * random.random())
            await asyncio.sleep(latency)
            logger.info(f"Completed throttled operation {i+1}/{count}")


async def run_caching_example():
    """Run the caching example."""
    logger.info("\n=== Caching Example ===")
    
    # First call (cache miss)
    start_time = time.time()
    result1 = await expensive_calculation(123, 456)
    elapsed1 = time.time() - start_time
    
    logger.info(f"First call result: {result1}, time: {elapsed1:.2f}s")
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = await expensive_calculation(123, 456)
    elapsed2 = time.time() - start_time
    
    logger.info(f"Second call result: {result2}, time: {elapsed2:.2f}s")
    
    # Different parameters (cache miss)
    start_time = time.time()
    result3 = await expensive_calculation(456, 789)
    elapsed3 = time.time() - start_time
    
    logger.info(f"Third call result: {result3}, time: {elapsed3:.2f}s")
    
    # Get cache stats
    cache_stats = await expensive_calculation.cache.get_stats()
    logger.info(f"Cache stats: {cache_stats}")


async def run_connection_pool_example():
    """Run the connection pool example."""
    logger.info("\n=== Connection Pool Example ===")
    
    # Set up the connection pool
    await setup_db_pool()
    
    # Execute queries concurrently
    queries = [
        "SELECT * FROM users",
        "SELECT * FROM products",
        "SELECT * FROM orders",
        "SELECT * FROM customers",
        "SELECT * FROM inventory"
    ]
    
    # Execute queries sequentially first
    logger.info("Executing queries sequentially...")
    start_time = time.time()
    for query in queries:
        await execute_query(query)
    sequential_time = time.time() - start_time
    
    logger.info(f"Sequential execution time: {sequential_time:.2f}s")
    
    # Execute queries concurrently
    logger.info("Executing queries concurrently...")
    start_time = time.time()
    tasks = [execute_query(query) for query in queries]
    await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    
    logger.info(f"Concurrent execution time: {concurrent_time:.2f}s")
    
    # Get connection pool stats
    pool_stats = await connection_pool_manager.get_stats()
    logger.info(f"Connection pool stats: {pool_stats}")


async def run_vector_db_batching_example():
    """Run the vector database batching example."""
    logger.info("\n=== Vector DB Batching Example ===")
    
    # Set up the vector database
    batched_db = await setup_vector_db()
    
    # Create some test vectors
    vectors = []
    for i in range(200):
        vectors.append({
            "id": f"vec-{i}",
            "vector": [random.random() for _ in range(384)],
            "metadata": {"text": f"Text for vector {i}", "category": random.choice(["A", "B", "C"])}
        })
    
    # Add vectors individually first
    logger.info("Adding vectors individually...")
    start_time = time.time()
    for vector in vectors[:50]:
        await batched_db.vector_db.add(**vector)
    individual_time = time.time() - start_time
    
    logger.info(f"Individual addition time: {individual_time:.2f}s")
    
    # Clear the database
    await batched_db.vector_db.clear()
    
    # Add vectors with batching
    logger.info("Adding vectors with batching...")
    start_time = time.time()
    tasks = [batched_db.add(**vector) for vector in vectors]
    await asyncio.gather(*tasks)
    batched_time = time.time() - start_time
    
    logger.info(f"Batched addition time: {batched_time:.2f}s")
    
    # Get batching stats
    batching_stats = await batched_db.get_stats()
    logger.info(f"Batching stats: {batching_stats}")


async def run_performance_monitoring_example():
    """Run the performance monitoring example."""
    logger.info("\n=== Performance Monitoring Example ===")
    
    # Generate embeddings
    for i in range(20):
        await generate_embedding(f"Text sample {i}")
    
    # Get performance metrics
    metrics = await performance_monitor.get_stats()
    logger.info(f"Performance metrics: {metrics}")
    
    # Use a profiler
    profiler = Profiler()
    profiler.start()
    
    with profiler.span("outer_operation"):
        # Simulate some work
        await asyncio.sleep(0.1)
        
        with profiler.span("inner_operation_1"):
            await asyncio.sleep(0.2)
        
        with profiler.span("inner_operation_2"):
            await asyncio.sleep(0.15)
    
    profile_results = profiler.stop()
    logger.info(f"Profiler results: {profile_results}")


async def run_hotspot_detection_example():
    """Run the hotspot detection example."""
    logger.info("\n=== Hotspot Detection Example ===")
    
    # Run the potential hotspot function with different loads
    await potential_hotspot(10000)
    await potential_hotspot(50000)
    await potential_hotspot(100000)
    
    # Get hotspot information
    hotspots = await potential_hotspot.detector.get_hotspots()
    logger.info(f"Detected hotspots: {hotspots}")


async def run_adaptive_throttling_example():
    """Run the adaptive throttling example."""
    logger.info("\n=== Adaptive Throttling Example ===")
    
    # Create a throttler
    throttler = AdaptiveThrottler(
        initial_rate=10.0,
        min_rate=5.0,
        max_rate=50.0,
        target_latency_ms=100.0
    )
    
    # Run throttled operations
    await run_throttled_operations(throttler, 20)
    
    # Get throttler stats
    throttler_stats = await throttler.get_stats()
    logger.info(f"Throttler stats: {throttler_stats}")


async def main():
    """Run all the examples."""
    logger.info("=== Performance Optimizations Examples ===")
    
    # Run the caching example
    await run_caching_example()
    
    # Run the connection pool example
    await run_connection_pool_example()
    
    # Run the vector database batching example
    await run_vector_db_batching_example()
    
    # Run the performance monitoring example
    await run_performance_monitoring_example()
    
    # Run the hotspot detection example
    await run_hotspot_detection_example()
    
    # Run the adaptive throttling example
    await run_adaptive_throttling_example()
    
    # Clean up
    await connection_pool_manager.close_all()


if __name__ == "__main__":
    asyncio.run(main())
