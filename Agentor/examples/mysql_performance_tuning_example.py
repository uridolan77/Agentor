"""
Example demonstrating MySQL performance tuning.

This example shows how to use the MySQL adapter with performance tuning features:
- Query optimization
- Index optimization
- Server configuration tuning
- Performance monitoring and analysis
"""

import asyncio
import logging
import sys
import time
import random
from typing import Dict, Any, Optional, List, Tuple

from agentor.interfaces.database import (
    MySqlAdapter,
    create_mysql_adapter
)
from agentor.interfaces.database.optimization import (
    OptimizationConfig,
    OptimizationLevel,
    MySqlOptimizationManager
)
from agentor.interfaces.database.optimization.config import (
    QueryOptimizationConfig,
    IndexOptimizationConfig,
    ServerOptimizationConfig,
    PerformanceMonitoringConfig
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


async def test_query_optimization():
    """Test query optimization."""
    logger.info("Testing query optimization")
    
    # Create a MySQL adapter
    mysql = create_mysql_adapter({
        "name": "mysql_query_optimization",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": True
    })
    
    try:
        # Connect to the database
        logger.info("Connecting to the database")
        result = await mysql.connect()
        if not result.success:
            logger.error(f"Failed to connect to the database: {result.error}")
            return
        
        logger.info("Connected to the database")
        
        # Create an optimization configuration
        optimization_config = OptimizationConfig(
            enabled=True,
            level=OptimizationLevel.AGGRESSIVE,
            query_optimization=QueryOptimizationConfig(
                enabled=True,
                level=OptimizationLevel.AGGRESSIVE,
                rewrite_queries=True,
                add_missing_indexes=True,
                optimize_joins=True,
                optimize_where_clauses=True,
                optimize_order_by=True,
                optimize_group_by=True,
                optimize_limit=True,
                analyze_queries=True,
                slow_query_threshold=1.0,
                very_slow_query_threshold=5.0,
                collect_query_plans=True,
                analyze_query_plans=True
            ),
            index_optimization=IndexOptimizationConfig(
                enabled=True,
                level=OptimizationLevel.MODERATE,
                analyze_indexes=True,
                collect_index_stats=True,
                recommend_indexes=True,
                recommend_composite_indexes=True,
                recommend_covering_indexes=True,
                auto_create_indexes=False,
                max_indexes_per_table=5
            ),
            server_optimization=ServerOptimizationConfig(
                enabled=True,
                level=OptimizationLevel.MODERATE,
                analyze_server=True,
                collect_server_stats=True,
                recommend_server_settings=True,
                auto_configure_server=False
            ),
            performance_monitoring=PerformanceMonitoringConfig(
                enabled=True,
                level=OptimizationLevel.MODERATE,
                monitoring_interval=60.0,
                detailed_monitoring_interval=300.0,
                collect_query_metrics=True,
                collect_index_metrics=True,
                collect_server_metrics=True,
                alert_on_slow_queries=True,
                alert_on_high_load=True,
                alert_threshold=0.8
            )
        )
        
        # Create an optimization manager
        optimization_manager = MySqlOptimizationManager(
            name="test_optimization",
            config=optimization_config,
            connection_func=lambda: mysql.pool
        )
        
        # Initialize the optimization manager
        await optimization_manager.initialize()
        
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
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                amount DECIMAL(10, 2) NOT NULL,
                status VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Insert test data
        logger.info("Inserting test data")
        
        # Insert users
        for i in range(1000):
            await mysql.execute(
                "INSERT INTO users (name, email) VALUES (:name, :email)",
                {"name": f"User {i}", "email": f"user{i}@example.com"}
            )
        
        # Insert orders
        for i in range(5000):
            await mysql.execute(
                "INSERT INTO orders (user_id, amount, status) VALUES (:user_id, :amount, :status)",
                {
                    "user_id": random.randint(1, 1000),
                    "amount": random.uniform(10.0, 1000.0),
                    "status": random.choice(["pending", "completed", "cancelled"])
                }
            )
        
        # Test query optimization
        logger.info("Testing query optimization")
        
        # Define some test queries
        test_queries = [
            # Query with SELECT *
            "SELECT * FROM users WHERE id = 1",
            
            # Query with inefficient JOIN
            "SELECT u.name, o.amount FROM users u, orders o WHERE u.id = o.user_id AND o.status = 'completed'",
            
            # Query with inefficient WHERE clause
            "SELECT * FROM orders WHERE amount != 100",
            
            # Query without LIMIT
            "SELECT * FROM users ORDER BY created_at DESC",
            
            # Query with inefficient GROUP BY
            "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id"
        ]
        
        # Optimize and analyze each query
        for query in test_queries:
            logger.info(f"Original query: {query}")
            
            # Optimize the query
            optimized_query = await optimization_manager.optimize_query(query)
            logger.info(f"Optimized query: {optimized_query}")
            
            # Execute the original query
            start_time = time.time()
            result = await mysql.execute(query)
            execution_time = time.time() - start_time
            
            # Analyze the query
            analysis = await optimization_manager.analyze_query(query, None, execution_time)
            logger.info(f"Query analysis: {analysis}")
            
            # Get the query plan
            query_plan = await optimization_manager.get_query_plan(query)
            logger.info(f"Query plan: {query_plan}")
        
        # Test index optimization
        logger.info("Testing index optimization")
        
        # Optimize indexes for the orders table
        index_recommendations = await optimization_manager.optimize_indexes("orders")
        logger.info(f"Index recommendations: {index_recommendations}")
        
        # Analyze indexes for the orders table
        index_analysis = await optimization_manager.analyze_indexes("orders")
        logger.info(f"Index analysis: {index_analysis}")
        
        # Test server optimization
        logger.info("Testing server optimization")
        
        # Optimize server settings
        server_recommendations = await optimization_manager.optimize_server()
        logger.info(f"Server recommendations: {server_recommendations}")
        
        # Analyze server settings
        server_analysis = await optimization_manager.analyze_server()
        logger.info(f"Server analysis: {server_analysis}")
        
        # Test performance monitoring
        logger.info("Testing performance monitoring")
        
        # Get performance metrics
        performance_metrics = await optimization_manager.get_performance_metrics()
        logger.info(f"Performance metrics: {performance_metrics}")
        
        # Get historical metrics
        historical_metrics = await optimization_manager.get_historical_metrics()
        logger.info(f"Historical metrics: {len(historical_metrics)} records")
        
        # Get slow queries
        slow_queries = await optimization_manager.get_slow_queries()
        logger.info(f"Slow queries: {len(slow_queries)} records")
        
        # Get optimization metrics
        optimization_metrics = await optimization_manager.get_metrics()
        logger.info(f"Optimization metrics: {optimization_metrics}")
        
        # Close the optimization manager
        await optimization_manager.close()
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS orders")
            await mysql.execute("DROP TABLE IF EXISTS users")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def test_mysql_adapter_with_optimization():
    """Test the MySQL adapter with optimization."""
    logger.info("Testing MySQL adapter with optimization")
    
    # Create a MySQL adapter with optimization
    mysql = create_mysql_adapter({
        "name": "mysql_optimization",
        "mysql_user": "root",  # Replace with your MySQL user
        "mysql_password": "",  # Replace with your MySQL password
        "mysql_host": "localhost",
        "mysql_port": 3306,
        "mysql_database": "test",  # Replace with your MySQL database
        "mysql_charset": "utf8mb4",
        "mysql_collation": "utf8mb4_unicode_ci",
        "mysql_autocommit": True,
        "mysql_optimization_enabled": True,
        "mysql_optimization_level": "aggressive",
        "mysql_query_optimization_enabled": True,
        "mysql_index_optimization_enabled": True,
        "mysql_server_optimization_enabled": True,
        "mysql_performance_monitoring_enabled": True
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
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                stock INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert test data
        logger.info("Inserting test data")
        
        # Insert products
        for i in range(1000):
            await mysql.execute(
                "INSERT INTO products (name, price, stock) VALUES (:name, :price, :stock)",
                {
                    "name": f"Product {i}",
                    "price": random.uniform(10.0, 1000.0),
                    "stock": random.randint(1, 100)
                }
            )
        
        # Test optimized queries
        logger.info("Testing optimized queries")
        
        # Define some test queries
        test_queries = [
            # Query with SELECT *
            "SELECT * FROM products WHERE id = 1",
            
            # Query with inefficient WHERE clause
            "SELECT * FROM products WHERE price != 100",
            
            # Query without LIMIT
            "SELECT * FROM products ORDER BY created_at DESC",
            
            # Query with inefficient GROUP BY
            "SELECT stock, COUNT(*) FROM products GROUP BY stock"
        ]
        
        # Execute each query
        for query in test_queries:
            logger.info(f"Executing query: {query}")
            
            # Execute the query
            result = await mysql.execute_optimized(query)
            
            if result.success:
                logger.info(f"Query executed successfully")
            else:
                logger.error(f"Failed to execute query: {result.error}")
        
        # Test index optimization
        logger.info("Testing index optimization")
        
        # Optimize indexes for the products table
        index_recommendations = await mysql.optimize_indexes("products")
        logger.info(f"Index recommendations: {index_recommendations}")
        
        # Test server optimization
        logger.info("Testing server optimization")
        
        # Optimize server settings
        server_recommendations = await mysql.optimize_server()
        logger.info(f"Server recommendations: {server_recommendations}")
        
        # Test performance monitoring
        logger.info("Testing performance monitoring")
        
        # Get performance metrics
        performance_metrics = await mysql.get_performance_metrics()
        logger.info(f"Performance metrics: {performance_metrics}")
        
        # Get optimization metrics
        optimization_metrics = await mysql.get_optimization_metrics()
        logger.info(f"Optimization metrics: {optimization_metrics}")
        
    finally:
        # Clean up
        if mysql.connected:
            # Drop the tables
            logger.info("Dropping the tables")
            await mysql.execute("DROP TABLE IF EXISTS products")
            
            # Disconnect
            logger.info("Disconnecting from the database")
            await mysql.disconnect()


async def main():
    """Run the MySQL performance tuning examples."""
    logger.info("Starting MySQL performance tuning examples")
    
    # Test query optimization
    await test_query_optimization()
    logger.info("-" * 80)
    
    # Test MySQL adapter with optimization
    await test_mysql_adapter_with_optimization()
    
    logger.info("MySQL performance tuning examples completed")


if __name__ == "__main__":
    asyncio.run(main())
