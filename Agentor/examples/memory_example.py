"""
Example demonstrating the enhanced memory systems in Agentor.

This example shows how to use the different memory types:
- Episodic memory for storing sequences of events
- Semantic memory for storing knowledge and facts
- Procedural memory for storing learned behaviors
- Unified memory that combines all three types
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

from agentor.components.memory.embedding import MockEmbeddingProvider
from agentor.components.memory.episodic_memory import EpisodicMemory
from agentor.components.memory.semantic_memory import SemanticMemory
from agentor.components.memory.procedural_memory import ProceduralMemory
from agentor.components.memory.unified_memory import UnifiedMemory, MemoryType
from agentor.components.memory.forgetting import (
    ThresholdForgetting,
    TimeForgetting,
    EbbinghausForgetting,
    CompositeForgetting,
    ForgettingConfig,
    ForgettingManager
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def episodic_memory_example():
    """Example of using episodic memory."""
    logger.info("=== Episodic Memory Example ===")
    
    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)
    
    # Create an episodic memory
    memory = EpisodicMemory(
        embedding_provider=embedding_provider,
        max_episodes=100,
        forgetting_threshold=0.2
    )
    
    # Create an episode
    episode = await memory.create_episode(
        metadata={"title": "Shopping Trip", "location": "Grocery Store"}
    )
    logger.info(f"Created episode: {episode.id}")
    
    # Add events to the episode
    await memory.add({
        "action": "enter_store",
        "observation": "The store is busy today.",
        "timestamp": time.time()
    })
    
    await memory.add({
        "action": "pick_up_item",
        "item": "apples",
        "quantity": 5,
        "observation": "The apples look fresh.",
        "timestamp": time.time() + 60
    })
    
    await memory.add({
        "action": "pick_up_item",
        "item": "bread",
        "quantity": 1,
        "observation": "The bread is on sale.",
        "timestamp": time.time() + 120
    })
    
    await memory.add({
        "action": "checkout",
        "total_cost": 15.75,
        "observation": "The checkout was quick.",
        "timestamp": time.time() + 180
    })
    
    # End the episode
    await memory.end_episode()
    
    # Create another episode
    episode2 = await memory.create_episode(
        metadata={"title": "Coffee Shop Visit", "location": "Downtown Cafe"}
    )
    
    # Add events to the second episode
    await memory.add({
        "action": "enter_cafe",
        "observation": "The cafe is quiet and cozy.",
        "timestamp": time.time() + 3600
    })
    
    await memory.add({
        "action": "order_drink",
        "item": "latte",
        "size": "medium",
        "observation": "The barista was friendly.",
        "timestamp": time.time() + 3660
    })
    
    await memory.add({
        "action": "sit_down",
        "observation": "Found a nice table by the window.",
        "timestamp": time.time() + 3720
    })
    
    # End the second episode
    await memory.end_episode()
    
    # Query episodes by text similarity
    logger.info("\nQuerying episodes by text similarity:")
    results = await memory.get({"text": "grocery shopping for apples"})
    
    for result in results:
        logger.info(f"Episode: {result['id']}")
        logger.info(f"Title: {result['metadata'].get('title')}")
        logger.info(f"Events: {len(result['events'])}")
        logger.info("---")
    
    # Query episodes by time range
    logger.info("\nQuerying episodes by time range:")
    start_time = time.time()
    end_time = time.time() + 200  # Just the first episode
    
    results = await memory.get({"start_time": start_time, "end_time": end_time})
    
    for result in results:
        logger.info(f"Episode: {result['id']}")
        logger.info(f"Title: {result['metadata'].get('title')}")
        logger.info(f"Events: {len(result['events'])}")
        logger.info("---")


async def semantic_memory_example():
    """Example of using semantic memory."""
    logger.info("\n=== Semantic Memory Example ===")
    
    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)
    
    # Create a semantic memory
    memory = SemanticMemory(
        embedding_provider=embedding_provider,
        max_nodes=1000,
        forgetting_threshold=0.2,
        similarity_threshold=0.85
    )
    
    # Add knowledge to memory
    await memory.add({
        "text": "Paris is the capital of France.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })
    
    await memory.add({
        "text": "The Eiffel Tower is in Paris.",
        "category": "geography",
        "importance": 0.7,
        "confidence": 1.0
    })
    
    await memory.add({
        "text": "Python is a programming language.",
        "category": "technology",
        "importance": 0.9,
        "confidence": 1.0
    })
    
    await memory.add({
        "text": "Machine learning is a subset of artificial intelligence.",
        "category": "technology",
        "importance": 0.8,
        "confidence": 1.0
    })
    
    # Query by text similarity
    logger.info("\nQuerying by text similarity:")
    results = await memory.get({"text": "What is the capital of France?"})
    
    for result in results[:2]:  # Show top 2 results
        logger.info(f"Node: {result['id']}")
        logger.info(f"Content: {result['content'].get('text')}")
        logger.info(f"Similarity: {result.get('similarity', 'N/A')}")
        logger.info("---")
    
    # Query by category
    logger.info("\nQuerying by category:")
    results = await memory.get({"category": "technology"})
    
    for result in results:
        logger.info(f"Node: {result['id']}")
        logger.info(f"Content: {result['content'].get('text')}")
        logger.info("---")
    
    # Link related nodes
    nodes = list(memory.nodes.values())
    if len(nodes) >= 2:
        await memory.link_nodes(nodes[0].id, nodes[1].id)
        
        # Get related nodes
        logger.info(f"\nNodes related to {nodes[0].id}:")
        related = await memory.get_related_nodes(nodes[0].id)
        
        for result in related:
            logger.info(f"Node: {result['id']}")
            logger.info(f"Content: {result['content'].get('text')}")
            logger.info("---")


async def procedural_memory_example():
    """Example of using procedural memory."""
    logger.info("\n=== Procedural Memory Example ===")
    
    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)
    
    # Create a procedural memory
    memory = ProceduralMemory(
        embedding_provider=embedding_provider,
        max_procedures=100,
        retention_threshold=0.2
    )
    
    # Store a function
    async def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    proc_id = await memory.store_function(
        name="Greeting",
        description="A function that greets a person by name",
        func=greet,
        tags=["greeting", "social"]
    )
    
    logger.info(f"Stored function with ID: {proc_id}")
    
    # Store code
    code_id = await memory.store_code(
        name="Calculate Sum",
        description="Calculate the sum of two numbers",
        code="""
def calculate_sum(a, b):
    return a + b

result = calculate_sum(args[0], args[1])
""",
        tags=["math", "calculation"]
    )
    
    logger.info(f"Stored code with ID: {code_id}")
    
    # Store steps
    steps_id = await memory.store_steps(
        name="Make Coffee",
        description="Steps to make a cup of coffee",
        steps=[
            {"description": "Boil water", "time_required": 120},
            {"description": "Add coffee grounds to filter", "time_required": 30},
            {"description": "Pour hot water over grounds", "time_required": 60},
            {"description": "Wait for coffee to brew", "time_required": 180},
            {"description": "Pour coffee into cup", "time_required": 15},
            {"description": "Add sugar and cream if desired", "time_required": 20}
        ],
        tags=["cooking", "beverage"]
    )
    
    logger.info(f"Stored steps with ID: {steps_id}")
    
    # Execute a procedure
    result = await memory.execute_procedure(proc_id, "World")
    logger.info(f"Executed greeting procedure: {result}")
    
    # Query by text similarity
    logger.info("\nQuerying by text similarity:")
    results = await memory.get({"text": "How to make coffee"})
    
    for result in results[:2]:  # Show top 2 results
        logger.info(f"Procedure: {result['id']}")
        logger.info(f"Name: {result['name']}")
        logger.info(f"Description: {result['description']}")
        logger.info(f"Similarity: {result.get('similarity', 'N/A')}")
        logger.info("---")
    
    # Query by tags
    logger.info("\nQuerying by tags:")
    results = await memory.get({"tags": "math"})
    
    for result in results:
        logger.info(f"Procedure: {result['id']}")
        logger.info(f"Name: {result['name']}")
        logger.info(f"Description: {result['description']}")
        logger.info("---")


async def unified_memory_example():
    """Example of using unified memory."""
    logger.info("\n=== Unified Memory Example ===")
    
    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)
    
    # Create a unified memory
    memory = UnifiedMemory(
        embedding_provider=embedding_provider,
        consolidation_interval=3600
    )
    
    # Add episodic memory
    await memory.create_episode(
        metadata={"title": "Morning Routine", "location": "Home"}
    )
    
    await memory.add_to_episode({
        "action": "wake_up",
        "time": "7:00 AM",
        "observation": "Feeling refreshed after a good night's sleep."
    })
    
    await memory.add_to_episode({
        "action": "make_breakfast",
        "items": ["eggs", "toast", "coffee"],
        "observation": "The coffee smells great."
    })
    
    await memory.end_episode()
    
    # Add semantic memory
    await memory.add({
        "text": "Coffee contains caffeine, which is a stimulant.",
        "category": "nutrition",
        "importance": 0.7,
        "confidence": 1.0,
        "memory_type": "semantic"
    })
    
    await memory.add({
        "text": "Eggs are a good source of protein.",
        "category": "nutrition",
        "importance": 0.6,
        "confidence": 1.0,
        "memory_type": "semantic"
    })
    
    # Add procedural memory
    await memory.add({
        "name": "Brew Coffee",
        "description": "Steps to brew a perfect cup of coffee",
        "steps": [
            {"description": "Grind coffee beans", "time_required": 30},
            {"description": "Heat water to 200°F", "time_required": 120},
            {"description": "Add grounds to filter", "time_required": 15},
            {"description": "Pour water over grounds", "time_required": 30},
            {"description": "Wait for coffee to brew", "time_required": 180}
        ],
        "tags": ["cooking", "beverage"],
        "memory_type": "procedural"
    })
    
    # Search across all memory types
    logger.info("\nSearching across all memory types:")
    results = await memory.search("coffee", memory_type=MemoryType.ALL)
    
    for result in results[:3]:  # Show top 3 results
        memory_type = result.get('memory_type', 'unknown')
        logger.info(f"Memory Type: {memory_type}")
        
        if memory_type == 'episodic':
            logger.info(f"Episode: {result['id']}")
            logger.info(f"Title: {result['metadata'].get('title')}")
            for event in result.get('events', [])[:2]:  # Show first 2 events
                logger.info(f"  Event: {event.get('action')}")
        
        elif memory_type == 'semantic':
            logger.info(f"Knowledge: {result['content'].get('text')}")
        
        elif memory_type == 'procedural':
            logger.info(f"Procedure: {result['name']}")
            logger.info(f"Description: {result['description']}")
        
        logger.info("---")
    
    # Search specific memory type
    logger.info("\nSearching only semantic memory:")
    results = await memory.search("nutrition", memory_type=MemoryType.SEMANTIC)
    
    for result in results:
        logger.info(f"Knowledge: {result['content'].get('text')}")
        logger.info(f"Category: {result['content'].get('category')}")
        logger.info("---")


async def forgetting_example():
    """Example of using forgetting mechanisms."""
    logger.info("\n=== Forgetting Mechanisms Example ===")
    
    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)
    
    # Create a semantic memory
    memory = SemanticMemory(
        embedding_provider=embedding_provider,
        max_nodes=10,  # Small limit to demonstrate forgetting
        forgetting_threshold=0.3
    )
    
    # Add knowledge with varying importance
    for i in range(15):  # Add more items than the max to trigger forgetting
        importance = 0.1 + (i % 10) / 10.0  # Vary importance from 0.1 to 1.0
        
        await memory.add({
            "text": f"This is fact #{i} with importance {importance:.1f}",
            "category": "test",
            "importance": importance,
            "confidence": 1.0
        })
    
    # Check how many items were kept
    results = await memory.get({})
    logger.info(f"After adding 15 items with max_nodes=10, {len(results)} items remain")
    
    # Show the remaining items
    logger.info("\nRemaining items (should be the most important ones):")
    for result in results:
        logger.info(f"Text: {result['content'].get('text')}")
        logger.info(f"Importance: {result['content'].get('importance')}")
        logger.info("---")
    
    # Create a composite forgetting mechanism
    threshold_forgetting = ThresholdForgetting(threshold=0.3)
    time_forgetting = TimeForgetting(half_life=3600)  # 1 hour half-life
    ebbinghaus_forgetting = EbbinghausForgetting()
    
    composite = CompositeForgetting([
        (threshold_forgetting, 0.5),
        (time_forgetting, 0.3),
        (ebbinghaus_forgetting, 0.2)
    ])
    
    # Create a forgetting manager
    forgetting_config = ForgettingConfig(
        mechanism=composite,
        check_interval=60,  # Check every minute
        max_items=5  # Keep only 5 items
    )
    
    forgetting_manager = ForgettingManager(forgetting_config)
    
    # Create a new memory for testing the forgetting manager
    test_memory = {}
    
    # Add items with varying importance and timestamps
    for i in range(10):
        importance = 0.1 + (i % 10) / 10.0
        item_id = f"item-{i}"
        
        # Create items with different creation times
        created_at = time.time() - (10 - i) * 3600  # Older items first
        
        test_memory[item_id] = type('MemoryItem', (), {
            'importance': importance,
            'created_at': created_at,
            'updated_at': created_at,
            'content': f"Test item #{i}"
        })
    
    # Apply forgetting
    remaining = await forgetting_manager.apply_forgetting(test_memory)
    
    # Show the remaining items
    logger.info("\nAfter applying forgetting manager (should keep 5 items):")
    logger.info(f"Items remaining: {len(remaining)}")
    
    for item_id, item in remaining.items():
        logger.info(f"Item: {item_id}")
        logger.info(f"Content: {item.content}")
        logger.info(f"Importance: {item.importance}")
        logger.info("---")


async def main():
    """Run all memory examples."""
    await episodic_memory_example()
    await semantic_memory_example()
    await procedural_memory_example()
    await unified_memory_example()
    await forgetting_example()


if __name__ == "__main__":
    asyncio.run(main())
