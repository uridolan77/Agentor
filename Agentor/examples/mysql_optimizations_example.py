"""
Example demonstrating MySQL optimizations.

This example shows how to use the MySQL adapter with advanced optimizations:
- Query caching
- Batch processing
- Streaming results
- Performance monitoring
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any, Optional, List, AsyncIterator

from agentor.interfaces.database import (
    MySqlAdapter,
    create_mysql_adapter
)
from agentor.interfaces.database.cache import (
    QueryCacheConfig,
    CacheStrategy,
    MySqlQueryCache
)
from agentor.interfaces.database.batch import (
    BatchProcessingConfig,
    BatchStrategy,
    MySqlBatchProcessor
)
from agentor.interfaces.database.stream import (
    StreamingConfig,
    StreamStrategy,
    MySqlStreamProcessor
)
from agentor.interfaces.database.monitor import (
    MonitoringConfig,
    MonitoringLevel,
    MySqlPerformanceMonitor
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_query_caching():
    """Test query caching."""
    logger.info("Testing query caching")
    
    # Create a query cache configuration
    cache_config = QueryCacheConfig(
        enabled=True,
        strategy=CacheStrategy.SELECT_ONLY,
        max_size=1000,
        max_memory=100 * 1024 * 1024,
        default_ttl=300.0,
        invalidate_on_write=True,
        invalidate_tables={
            "users": ["user_roles", "user_permissions"],
            "products": ["product_categories", "product_tags"]
        },
        include_patterns=[
            r"SELECT.*FROM\s+users",
            r"SELECT.*FROM\s+products"
        ],
        exclude_patterns=[
            r"SELECT.*FOR\s+UPDATE"
        ]
    )
    
    # Create a MySQL query cache
    cache = MySqlQueryCache(
        name="test_cache",
        config=cache_config
    )
    
    try:
        # Initialize the cache
        logger.info("Initializing the query cache")
        await cache.initialize()
        
        # Test caching
        logger.info("Testing cache operations")
        
        # Cache a query result
        query = "SELECT * FROM users WHERE id = :id"
        params = {"id": 1}
        key = f"{query}_{params['id']}"
        result = {"id": 1, "name": "John Doe", "email": "john@example.com"}
        
        await cache.set(key, result, query)
        
        # Get the result from cache
        hit, cached_result = await cache.get(key)
        logger.info(f"Cache hit: {hit}, Result: {cached_result}")
        
        # Test cache invalidation
        logger.info("Testing cache invalidation")
        
        # Invalidate a specific key
        await cache.invalidate(key)
        
        hit, cached_result = await cache.get(key)
        logger.info(f"After invalidation - Cache hit: {hit}, Result: {cached_result}")
        
        # Cache the result again
        await cache.set(key, result, query)
        
        # Invalidate based on a query
        update_query = "UPDATE users SET name = 'Jane Doe' WHERE id = 1"
        await cache.invalidate_query(update_query)
        
        hit, cached_result = await cache.get(key)
        logger.info(f"After query invalidation - Cache hit: {hit}, Result: {cached_result}")
        
        # Cache the result again
        await cache.set(key, result, query)
        
        # Invalidate specific tables
        await cache.invalidate_tables(["users"])
        
        hit, cached_result = await cache.get(key)
        logger.info(f"After table invalidation - Cache hit: {hit}, Result: {cached_result}")
        
        # Get cache metrics
        metrics = await cache.get_metrics()
        logger.info(f"Cache metrics: {metrics}")
        
    finally:
        # Close the cache
        logger.info("Closing the query cache")
        await cache.close()


async def test_batch_processing():
    """Test batch processing."""
    logger.info("Testing batch processing")
    
    # Create a batch processing configuration
    batch_config = BatchProcessingConfig(
        enabled=True,
        strategy=BatchStrategy.HYBRID,
        batch_size=100,
        min_batch_size=10,
        batch_time=1.0,
        min_batch_time=0.1,
        max_retries=3,
        retry_delay=1.0
    )
    
    # Create a function to execute a batch of operations
    async def execute_batch(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of operations.
        
        Args:
            operations: The operations to execute
            
        Returns:
            The results of the operations
        """
        logger.info(f"Executing batch of {len(operations)} operations")
        
        # Simulate batch execution
        await asyncio.sleep(0.5)
        
        # Return results
        return [{"id": op["id"], "success": True} for op in operations]
    
    # Create a MySQL batch processor
    batch_processor = MySqlBatchProcessor(
        name="test_batch",
        config=batch_config,
        execute_batch_func=execute_batch
    )
    
    try:
        # Initialize the batch processor
        logger.info("Initializing the batch processor")
        await batch_processor.initialize()
        
        # Test batch processing
        logger.info("Testing batch operations")
        
        # Create tasks for batch operations
        tasks = []
        for i in range(250):
            tasks.append(
                batch_processor.add_operation(
                    "insert",
                    {"id": i, "name": f"Item {i}", "value": random.randint(1, 100)}
                )
            )
        
        # Wait for all operations to complete
        results = await asyncio.gather(*tasks)
        
        # Check the results
        success_count = sum(1 for result in results if result["success"])
        logger.info(f"Batch operations completed: {success_count}/{len(results)} successful")
        
        # Get batch processor metrics
        metrics = await batch_processor.get_metrics()
        logger.info(f"Batch processor metrics: {metrics}")
        
    finally:
        # Close the batch processor
        logger.info("Closing the batch processor")
        await batch_processor.close()


async def test_streaming_results():
    """Test streaming results."""
    logger.info("Testing streaming results")
    
    # Create a streaming configuration
    stream_config = StreamingConfig(
        enabled=True,
        strategy=StreamStrategy.CHUNK,
        chunk_size=50,
        min_chunk_size=10,
        buffer_size=5,
        min_buffer_size=2,
        chunk_timeout=5.0,
        query_timeout=60.0
    )
    
    # Create a function to execute a query and return a stream of chunks
    async def execute_query(query: str, params: Dict[str, Any], chunk_size: int) -> AsyncIterator[List[Dict[str, Any]]]:
        """Execute a query and return a stream of chunks.
        
        Args:
            query: The query to execute
            params: The query parameters
            chunk_size: The chunk size
            
        Returns:
            An async iterator of chunks
        """
        logger.info(f"Executing query: {query} with chunk size {chunk_size}")
        
        # Simulate a large result set
        total_rows = 1000
        
        # Generate chunks
        for i in range(0, total_rows, chunk_size):
            # Simulate query execution time
            await asyncio.sleep(0.1)
            
            # Generate a chunk of results
            chunk = []
            for j in range(i, min(i + chunk_size, total_rows)):
                chunk.append({"id": j, "name": f"Item {j}", "value": random.randint(1, 100)})
            
            # Yield the chunk
            yield chunk
    
    # Create a MySQL stream processor
    stream_processor = MySqlStreamProcessor(
        name="test_stream",
        config=stream_config,
        execute_query_func=execute_query
    )
    
    try:
        # Initialize the stream processor
        logger.info("Initializing the stream processor")
        await stream_processor.initialize()
        
        # Test streaming
        logger.info("Testing streaming")
        
        # Stream a query by chunks
        logger.info("Streaming by chunks")
        query = "SELECT * FROM large_table WHERE value > :min_value"
        params = {"min_value": 50}
        
        chunk_count = 0
        row_count = 0
        
        async for chunk in stream_processor.stream_query(query, params):
            chunk_count += 1
            row_count += len(chunk)
            logger.info(f"Received chunk {chunk_count} with {len(chunk)} rows")
        
        logger.info(f"Streaming completed: {chunk_count} chunks, {row_count} rows")
        
        # Stream a query by rows
        logger.info("Streaming by rows")
        stream_processor.config.strategy = StreamStrategy.ROW
        
        row_count = 0
        
        async for row in stream_processor.stream_query(query, params):
            row_count += 1
            if row_count % 100 == 0:
                logger.info(f"Processed {row_count} rows")
        
        logger.info(f"Streaming completed: {row_count} rows")
        
        # Get stream processor metrics
        metrics = await stream_processor.get_metrics()
        logger.info(f"Stream processor metrics: {metrics}")
        
    finally:
        # Close the stream processor
        logger.info("Closing the stream processor")
        await stream_processor.close()


async def test_performance_monitoring():
    """Test performance monitoring."""
    logger.info("Testing performance monitoring")
    
    # Create a monitoring configuration
    monitor_config = MonitoringConfig(
        enabled=True,
        level=MonitoringLevel.DETAILED,
        slow_query_threshold=1.0,
        very_slow_query_threshold=5.0,
        sample_rate=1.0,
        log_slow_queries=True,
        log_query_plans=True,
        log_query_stats=True,
        alert_on_slow_queries=True,
        alert_threshold=5.0,
        collect_query_plans=True,
        collect_table_stats=True,
        collect_index_stats=True
    )
    
    # Create a MySQL performance monitor
    monitor = MySqlPerformanceMonitor(
        name="test_monitor",
        config=monitor_config
    )
    
    try:
        # Initialize the performance monitor
        logger.info("Initializing the performance monitor")
        await monitor.initialize()
        
        # Test monitoring
        logger.info("Testing monitoring")
        
        # Record some queries
        queries = [
            "SELECT * FROM users WHERE id = 1",
            "SELECT * FROM products WHERE category_id = 2",
            "UPDATE users SET name = 'Jane Doe' WHERE id = 1",
            "INSERT INTO products (name, price) VALUES ('New Product', 99.99)",
            "SELECT * FROM orders WHERE user_id = 1 AND status = 'pending'"
        ]
        
        for i, query in enumerate(queries):
            # Simulate different execution times
            execution_time = random.uniform(0.1, 10.0)
            rows = random.randint(1, 100)
            
            # Record the query
            await monitor.record_query(
                query=query,
                params={"param": i},
                execution_time=execution_time,
                rows=rows
            )
            
            logger.info(f"Recorded query: {query} ({execution_time:.2f}s, {rows} rows)")
        
        # Get query statistics
        query_stats = await monitor.get_query_stats()
        logger.info(f"Query statistics: {query_stats}")
        
        # Get monitor metrics
        metrics = await monitor.get_metrics()
        logger.info(f"Monitor metrics: {metrics}")
        
    finally:
        # Close the performance monitor
        logger.info("Closing the performance monitor")
        await monitor.close()


async def test_mysql_adapter_with_optimizations():
    """Test the MySQL adapter with all optimizations."""
    logger.info("Testing MySQL adapter with all optimizations")
    
    # Create a MySQL adapter with optimizations
    mysql = create_mysql_adapter({
        "name": "mysql_optimized",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": False,
        "mysql_pool_min_size": 2,
        "mysql_pool_max_size": 10,
        "mysql_pool_recycle": 3600,
        "mysql_cache_enabled": True,
        "mysql_cache_strategy": "select_only",
        "mysql_cache_ttl": 300,
        "mysql_batch_enabled": True,
        "mysql_batch_size": 100,
        "mysql_batch_time": 1.0,
        "mysql_stream_enabled": True,
        "mysql_stream_chunk_size": 1000,
        "mysql_monitor_enabled": True,
        "mysql_monitor_level": "detailed"
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create test tables
        logger.info("Creating test tables")
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await mysql.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                category_id INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        logger.info("Inserting test data")
        
        # Insert users
        for i in range(100):
            await mysql.execute(
                "INSERT INTO users (name, email) VALUES (:name, :email)",
                {"name": f"User {i}", "email": f"user{i}@example.com"}
            )
        
        # Insert products
        for i in range(1000):
            await mysql.execute(
                "INSERT INTO products (name, price, category_id) VALUES (:name, :price, :category_id)",
                {
                    "name": f"Product {i}",
                    "price": random.uniform(10.0, 1000.0),
                    "category_id": random.randint(1, 10)
                }
            )
        
        # Test query caching
        logger.info("Testing query caching")
        
        # Execute a query that should be cached
        start_time = time.time()
        result1 = await mysql.fetch_one(
            "SELECT * FROM users WHERE id = :id",
            {"id": 1}
        )
        first_query_time = time.time() - start_time
        
        logger.info(f"First query time: {first_query_time:.6f}s")
        
        # Execute the same query again (should be cached)
        start_time = time.time()
        result2 = await mysql.fetch_one(
            "SELECT * FROM users WHERE id = :id",
            {"id": 1}
        )
        second_query_time = time.time() - start_time
        
        logger.info(f"Second query time: {second_query_time:.6f}s")
        logger.info(f"Cache speedup: {first_query_time / second_query_time:.2f}x")
        
        # Test batch processing
        logger.info("Testing batch processing")
        
        # Execute batch operations
        batch_tasks = []
        for i in range(100, 200):
            batch_tasks.append(
                mysql.execute(
                    "INSERT INTO users (name, email) VALUES (:name, :email)",
                    {"name": f"Batch User {i}", "email": f"batchuser{i}@example.com"}
                )
            )
        
        batch_results = await asyncio.gather(*batch_tasks)
        batch_success = sum(1 for result in batch_results if result.success)
        
        logger.info(f"Batch operations: {batch_success}/{len(batch_results)} successful")
        
        # Test streaming
        logger.info("Testing streaming")
        
        # Stream a large query
        row_count = 0
        async for row in mysql.stream_query(
            "SELECT * FROM products WHERE price > :min_price",
            {"min_price": 500.0}
        ):
            row_count += 1
        
        logger.info(f"Streamed {row_count} rows")
        
        # Get metrics
        metrics = await mysql.get_metrics()
        logger.info(f"MySQL adapter metrics: {metrics}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS users")
            await mysql.execute("DROP TABLE IF EXISTS products")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL optimizations examples."""
    logger.info("Starting MySQL optimizations examples")
    
    # Test query caching
    await test_query_caching()
    logger.info("-" * 80)
    
    # Test batch processing
    await test_batch_processing()
    logger.info("-" * 80)
    
    # Test streaming results
    await test_streaming_results()
    logger.info("-" * 80)
    
    # Test performance monitoring
    await test_performance_monitoring()
    logger.info("-" * 80)
    
    # Test MySQL adapter with all optimizations
    await test_mysql_adapter_with_optimizations()
    
    logger.info("MySQL optimizations examples completed")


if __name__ == "__main__":
    asyncio.run(main())
