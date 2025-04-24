"""
Example demonstrating the cached tool registry in Agentor.

This example shows how to use the cached tool registry to improve performance
by caching tool lookups, schemas, and execution results.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional

from agentor.agents.enhanced_tools import (
    EnhancedTool, 
    EnhancedToolRegistry, 
    ToolResult, 
    WeatherTool, 
    NewsTool, 
    CalculatorTool
)
from agentor.agents.tools.cached_registry import (
    CachedToolRegistry,
    create_cached_registry
)
from agentor.core.utils.caching import CacheStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SlowTool(EnhancedTool):
    """A tool that is intentionally slow for demonstration purposes."""
    
    def __init__(self, delay: float = 1.0):
        """Initialize the slow tool.
        
        Args:
            delay: The delay in seconds
        """
        super().__init__(
            name="slow_tool",
            description="A tool that is intentionally slow",
            version="1.0.0"
        )
        self.delay = delay
    
    async def run(self, input_text: str) -> ToolResult:
        """Run the tool with a delay.
        
        Args:
            input_text: The input text
            
        Returns:
            The result of running the tool
        """
        logger.info(f"Running slow tool with input: {input_text}")
        
        # Simulate a slow operation
        await asyncio.sleep(self.delay)
        
        return ToolResult(
            success=True,
            data={
                "input": input_text,
                "output": f"Processed: {input_text}",
                "timestamp": time.time()
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        return {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "The input text to process"
                }
            },
            "required": ["input_text"]
        }


class RandomTool(EnhancedTool):
    """A tool that returns random results."""
    
    def __init__(self):
        """Initialize the random tool."""
        super().__init__(
            name="random_tool",
            description="A tool that returns random results",
            version="1.0.0"
        )
    
    async def run(self, seed: Optional[int] = None) -> ToolResult:
        """Run the tool and return random results.
        
        Args:
            seed: Optional random seed
            
        Returns:
            The result of running the tool
        """
        logger.info(f"Running random tool with seed: {seed}")
        
        # Set the seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Generate random data
        return ToolResult(
            success=True,
            data={
                "random_number": random.random(),
                "random_choice": random.choice(["A", "B", "C", "D", "E"]),
                "random_sample": random.sample(range(100), 5),
                "timestamp": time.time()
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        return {
            "type": "object",
            "properties": {
                "seed": {
                    "type": "integer",
                    "description": "Optional random seed"
                }
            }
        }


async def run_without_caching():
    """Run tools without caching."""
    logger.info("\n=== Running Without Caching ===")
    
    # Create a regular tool registry
    registry = EnhancedToolRegistry()
    
    # Register tools
    registry.register_tool(SlowTool())
    registry.register_tool(CalculatorTool())
    registry.register_tool(RandomTool())
    
    # Execute the slow tool multiple times
    logger.info("Executing slow tool multiple times...")
    start_time = time.time()
    
    for i in range(5):
        result = await registry.execute_tool("slow_tool", input_text=f"Test {i}")
        logger.info(f"Result {i}: {result.data}")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time without caching: {elapsed:.2f}s")
    
    # Execute the calculator tool multiple times
    logger.info("\nExecuting calculator tool multiple times...")
    start_time = time.time()
    
    for i in range(5):
        result = await registry.execute_tool("calculator", expression="2 + 2")
        logger.info(f"Result {i}: {result.data}")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time without caching: {elapsed:.2f}s")
    
    # Execute the random tool multiple times
    logger.info("\nExecuting random tool multiple times...")
    start_time = time.time()
    
    for i in range(5):
        result = await registry.execute_tool("random_tool", seed=42)
        logger.info(f"Result {i}: {result.data}")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time without caching: {elapsed:.2f}s")


async def run_with_caching():
    """Run tools with caching."""
    logger.info("\n=== Running With Caching ===")
    
    # Create a base tool registry
    base_registry = EnhancedToolRegistry()
    
    # Register tools
    base_registry.register_tool(SlowTool())
    base_registry.register_tool(CalculatorTool())
    base_registry.register_tool(RandomTool())
    
    # Create a cached registry
    cached_registry = CachedToolRegistry(
        base_registry=base_registry,
        cache_ttl=60,
        result_cache_ttl=30,
        max_cache_size=100,
        cache_strategy=CacheStrategy.LRU,
        enable_result_caching=True
    )
    
    # Execute the slow tool multiple times
    logger.info("Executing slow tool multiple times...")
    start_time = time.time()
    
    for i in range(5):
        result = await cached_registry.execute_tool("slow_tool", input_text=f"Test {i % 2}")
        logger.info(f"Result {i}: {result.data}")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time with caching: {elapsed:.2f}s")
    
    # Execute the calculator tool multiple times
    logger.info("\nExecuting calculator tool multiple times...")
    start_time = time.time()
    
    for i in range(5):
        result = await cached_registry.execute_tool("calculator", expression="2 + 2")
        logger.info(f"Result {i}: {result.data}")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time with caching: {elapsed:.2f}s")
    
    # Execute the random tool multiple times
    logger.info("\nExecuting random tool multiple times...")
    start_time = time.time()
    
    for i in range(5):
        result = await cached_registry.execute_tool("random_tool", seed=42)
        logger.info(f"Result {i}: {result.data}")
    
    elapsed = time.time() - start_time
    logger.info(f"Total time with caching: {elapsed:.2f}s")
    
    # Get cache stats
    cache_stats = await cached_registry.get_cache_stats()
    logger.info(f"\nCache stats: {cache_stats}")


async def run_cache_invalidation():
    """Demonstrate cache invalidation."""
    logger.info("\n=== Cache Invalidation ===")
    
    # Create a cached registry
    cached_registry = create_cached_registry(
        result_cache_ttl=5  # Short TTL for demonstration
    )
    
    # Register tools
    cached_registry.register_tool(SlowTool(delay=0.5))
    
    # Execute the tool and cache the result
    logger.info("Executing tool and caching result...")
    result1 = await cached_registry.execute_tool("slow_tool", input_text="Cache me")
    logger.info(f"Result: {result1.data}")
    
    # Execute again (should be cached)
    logger.info("\nExecuting again (should be cached)...")
    start_time = time.time()
    result2 = await cached_registry.execute_tool("slow_tool", input_text="Cache me")
    elapsed = time.time() - start_time
    logger.info(f"Result: {result2.data}")
    logger.info(f"Time: {elapsed:.2f}s")
    
    # Wait for cache to expire
    logger.info("\nWaiting for cache to expire...")
    await asyncio.sleep(6)
    
    # Execute again (should miss cache)
    logger.info("\nExecuting again (should miss cache)...")
    start_time = time.time()
    result3 = await cached_registry.execute_tool("slow_tool", input_text="Cache me")
    elapsed = time.time() - start_time
    logger.info(f"Result: {result3.data}")
    logger.info(f"Time: {elapsed:.2f}s")
    
    # Clear the cache
    logger.info("\nClearing the cache...")
    await cached_registry.clear_caches()
    
    # Execute again (should miss cache)
    logger.info("\nExecuting again (should miss cache)...")
    start_time = time.time()
    result4 = await cached_registry.execute_tool("slow_tool", input_text="Cache me")
    elapsed = time.time() - start_time
    logger.info(f"Result: {result4.data}")
    logger.info(f"Time: {elapsed:.2f}s")


async def main():
    """Run the example."""
    logger.info("=== Cached Tool Registry Example ===")
    
    # Run without caching
    await run_without_caching()
    
    # Run with caching
    await run_with_caching()
    
    # Demonstrate cache invalidation
    await run_cache_invalidation()


if __name__ == "__main__":
    asyncio.run(main())
