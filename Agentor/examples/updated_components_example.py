"""
Example demonstrating the updated components using the standardized interfaces.

This example shows how to use the updated memory components, LLM providers,
agent implementations, and tools with the new standardized interfaces.
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional

from agentor.core.init import framework_lifespan
from agentor.core.plugin import get_plugin_registry
from agentor.core.di import inject, get_container
from agentor.core.registry import get_component_registry
from agentor.core.interfaces.memory import IMemory, IEpisodicMemory, ISemanticMemory
from agentor.core.interfaces.llm import ILLM, LLMRequest
from agentor.core.interfaces.agent import IAgent, AgentInput
from agentor.core.interfaces.tool import ITool, IToolRegistry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@inject
async def memory_example(
    simple_memory: IMemory,
    episodic_memory: IEpisodicMemory,
    semantic_memory: ISemanticMemory
):
    """Example of using the updated memory components.
    
    Args:
        simple_memory: The simple memory component (injected)
        episodic_memory: The episodic memory component (injected)
        semantic_memory: The semantic memory component (injected)
    """
    logger.info("=== Memory Example ===")
    
    # Use simple memory
    logger.info("Using simple memory")
    await simple_memory.add({"text": "This is a simple memory item", "timestamp": time.time()})
    results = await simple_memory.get({})
    logger.info(f"Retrieved {len(results)} items from simple memory")
    
    # Use episodic memory
    logger.info("\nUsing episodic memory")
    episode_id = await episodic_memory.add_episode({
        "title": "Shopping Trip",
        "location": "Grocery Store"
    })
    logger.info(f"Created episode: {episode_id}")
    
    await episodic_memory.add({
        "action": "enter_store",
        "observation": "The store is busy today.",
        "timestamp": time.time(),
        "episode_id": episode_id
    })
    
    await episodic_memory.add({
        "action": "checkout",
        "total_cost": 15.75,
        "observation": "The checkout was quick.",
        "timestamp": time.time() + 180,
        "episode_id": episode_id
    })
    
    episode = await episodic_memory.get_episode(episode_id)
    logger.info(f"Retrieved episode: {episode['title']}")
    
    # Use semantic memory
    logger.info("\nUsing semantic memory")
    knowledge_id = await semantic_memory.add_knowledge({
        "text": "Paris is the capital of France.",
        "category": "geography",
        "importance": 0.8
    })
    logger.info(f"Added knowledge: {knowledge_id}")
    
    await semantic_memory.add_knowledge({
        "text": "The Eiffel Tower is in Paris.",
        "category": "geography",
        "importance": 0.7
    })
    
    results = await semantic_memory.search_knowledge("Paris")
    logger.info(f"Found {len(results)} items about Paris")
    for item in results:
        logger.info(f"  - {item.get('text')}")


@inject
async def llm_example(llm: ILLM):
    """Example of using the updated LLM providers.
    
    Args:
        llm: The LLM provider (injected)
    """
    logger.info("\n=== LLM Example ===")
    
    # Create a request
    request = LLMRequest(
        prompt="What is the capital of France?",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )
    
    # Generate a response
    try:
        response = await llm.generate(request)
        logger.info(f"Generated response: {response.text}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Token usage: {response.usage}")
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")


@inject
async def agent_example(agent: IAgent):
    """Example of using the updated agent implementations.
    
    Args:
        agent: The agent (injected)
    """
    logger.info("\n=== Agent Example ===")
    
    # Run the agent
    output = await agent.run("What is the weather in New York?")
    logger.info(f"Agent response: {output.response}")
    
    # Run the agent with context
    output = await agent.run(
        "What should I wear?",
        {"location": "New York", "temperature": 72, "conditions": "sunny"}
    )
    logger.info(f"Agent response with context: {output.response}")


@inject
async def tool_example(tool_registry: IToolRegistry):
    """Example of using the updated tools.
    
    Args:
        tool_registry: The tool registry (injected)
    """
    logger.info("\n=== Tool Example ===")
    
    # Get the available tools
    tools = tool_registry.get_tools()
    logger.info(f"Available tools: {', '.join(tools.keys())}")
    
    # Use the weather tool
    weather_tool = tool_registry.get_tool("weather")
    if weather_tool:
        result = await weather_tool.run(location="New York")
        logger.info(f"Weather in New York: {result.data}")
    
    # Use the calculator tool
    calculator_tool = tool_registry.get_tool("calculator")
    if calculator_tool:
        result = await calculator_tool.run(expression="2 + 2 * 3")
        logger.info(f"Calculation result: {result.data}")


@inject
async def router_example(router: IAgent):
    """Example of using the updated router.
    
    Args:
        router: The router (injected)
    """
    logger.info("\n=== Router Example ===")
    
    # Run the router with different queries
    queries = [
        "What is the weather in New York?",
        "What is 2 + 2?",
        "Tell me the latest news about technology"
    ]
    
    for query in queries:
        output = await router.run(query)
        logger.info(f"Query: {query}")
        logger.info(f"Response: {output.response}")
        logger.info(f"Metadata: {output.metadata}")
        logger.info("---")


async def run_examples():
    """Run all examples."""
    # Get the component registry
    component_registry = get_component_registry()
    
    # Get the container
    container = get_container()
    
    # Register components in the container
    container.register_instance(
        IMemory,
        component_registry.get_memory_provider("simple")
    )
    
    container.register_instance(
        IEpisodicMemory,
        component_registry.get_memory_provider("episodic")
    )
    
    container.register_instance(
        ISemanticMemory,
        component_registry.get_memory_provider("semantic")
    )
    
    container.register_instance(
        ILLM,
        component_registry.get_llm_provider("openai")
    )
    
    container.register_instance(
        IAgent,
        component_registry.get_agent_provider("base")
    )
    
    container.register_instance(
        IToolRegistry,
        component_registry.get_tool_registry()
    )
    
    # Run the examples
    await memory_example()
    
    # Only run the LLM example if the OpenAI API key is set
    if os.environ.get("OPENAI_API_KEY"):
        await llm_example()
    else:
        logger.warning("Skipping LLM example because OPENAI_API_KEY is not set")
    
    await agent_example()
    await tool_example()
    
    # Get the hierarchical router
    router = component_registry.get_agent_provider("hierarchical_router")
    if router:
        container.register_instance(IAgent, router, name="router")
        await router_example()
    else:
        logger.warning("Skipping router example because hierarchical_router is not available")


async def main():
    """Main function."""
    async with framework_lifespan():
        await run_examples()


if __name__ == "__main__":
    asyncio.run(main())
