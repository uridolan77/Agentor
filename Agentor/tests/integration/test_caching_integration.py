"""
Integration tests for the caching system.

This module tests how the caching system integrates with other components
of the Agentor framework, including:
- LLM caching integration
- Agent caching integration
- Tool caching integration
- Performance impact of caching
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import json
import random
from typing import Dict, Any, List, Optional

from agentor.core.utils.caching import (
    CacheStrategy,
    InMemoryCache,
    Cache,
    cached
)


# ===== Mock Classes for Integration Testing =====

class MockLLM:
    """Mock LLM for testing caching integration."""
    
    def __init__(self, name="mock-llm", latency=0.1):
        """Initialize the mock LLM.
        
        Args:
            name: The name of the LLM
            latency: Simulated latency in seconds
        """
        self.name = name
        self.latency = latency
        self.call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to the prompt.
        
        Args:
            prompt: The prompt to respond to
            **kwargs: Additional arguments
            
        Returns:
            The generated response
        """
        self.call_count += 1
        await asyncio.sleep(self.latency)  # Simulate API latency
        return f"Response to: {prompt}"


class MockAgent:
    """Mock agent for testing caching integration."""
    
    def __init__(self, name="mock-agent", llm=None):
        """Initialize the mock agent.
        
        Args:
            name: The name of the agent
            llm: The LLM to use
        """
        self.name = name
        self.llm = llm or MockLLM()
        self.call_count = 0
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process the input text.
        
        Args:
            input_text: The input text to process
            
        Returns:
            The processing result
        """
        self.call_count += 1
        response = await self.llm.generate(input_text)
        return {
            "input": input_text,
            "output": response,
            "agent": self.name
        }


class MockTool:
    """Mock tool for testing caching integration."""
    
    def __init__(self, name="mock-tool", latency=0.05):
        """Initialize the mock tool.
        
        Args:
            name: The name of the tool
            latency: Simulated latency in seconds
        """
        self.name = name
        self.latency = latency
        self.call_count = 0
    
    async def execute(self, input_data: Any) -> Any:
        """Execute the tool.
        
        Args:
            input_data: The input data
            
        Returns:
            The execution result
        """
        self.call_count += 1
        await asyncio.sleep(self.latency)  # Simulate execution latency
        
        if isinstance(input_data, dict):
            return {k: f"processed_{v}" for k, v in input_data.items()}
        elif isinstance(input_data, list):
            return [f"processed_{item}" for item in input_data]
        else:
            return f"processed_{input_data}"


# ===== Fixtures =====

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockLLM()


@pytest.fixture
def mock_agent(mock_llm):
    """Create a mock agent for testing."""
    return MockAgent(llm=mock_llm)


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    return MockTool()


@pytest.fixture
def cache():
    """Create a cache for testing."""
    return Cache(
        primary_backend=InMemoryCache(
            max_size=1000,
            default_ttl=60,
            strategy=CacheStrategy.LRU
        )
    )


# ===== Integration Tests =====

@pytest.mark.asyncio
async def test_llm_caching_integration(mock_llm, cache):
    """Test caching integration with an LLM."""
    
    # Create a cached version of the LLM's generate method
    @cached(cache_instance=cache)
    async def cached_generate(prompt: str, **kwargs):
        return await mock_llm.generate(prompt, **kwargs)
    
    # First call should execute the LLM
    response1 = await cached_generate("Hello, world!")
    assert response1 == "Response to: Hello, world!"
    assert mock_llm.call_count == 1
    
    # Second call with the same prompt should use the cache
    response2 = await cached_generate("Hello, world!")
    assert response2 == "Response to: Hello, world!"
    assert mock_llm.call_count == 1  # Still 1
    
    # Call with a different prompt should execute the LLM again
    response3 = await cached_generate("How are you?")
    assert response3 == "Response to: How are you?"
    assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_agent_caching_integration(mock_agent, cache):
    """Test caching integration with an agent."""
    
    # Create a cached version of the agent's process method
    @cached(cache_instance=cache)
    async def cached_process(input_text: str):
        return await mock_agent.process(input_text)
    
    # First call should execute the agent
    result1 = await cached_process("Hello, agent!")
    assert result1["input"] == "Hello, agent!"
    assert result1["output"] == "Response to: Hello, agent!"
    assert mock_agent.call_count == 1
    assert mock_agent.llm.call_count == 1
    
    # Second call with the same input should use the cache
    result2 = await cached_process("Hello, agent!")
    assert result2["input"] == "Hello, agent!"
    assert result2["output"] == "Response to: Hello, agent!"
    assert mock_agent.call_count == 1  # Still 1
    assert mock_agent.llm.call_count == 1  # Still 1
    
    # Call with a different input should execute the agent again
    result3 = await cached_process("New input")
    assert result3["input"] == "New input"
    assert result3["output"] == "Response to: New input"
    assert mock_agent.call_count == 2
    assert mock_agent.llm.call_count == 2


@pytest.mark.asyncio
async def test_tool_caching_integration(mock_tool, cache):
    """Test caching integration with a tool."""
    
    # Create a cached version of the tool's execute method
    @cached(cache_instance=cache)
    async def cached_execute(input_data: Any):
        return await mock_tool.execute(input_data)
    
    # Test with a string input
    result1 = await cached_execute("test_input")
    assert result1 == "processed_test_input"
    assert mock_tool.call_count == 1
    
    # Second call with the same input should use the cache
    result2 = await cached_execute("test_input")
    assert result2 == "processed_test_input"
    assert mock_tool.call_count == 1  # Still 1
    
    # Test with a dictionary input
    dict_input = {"key1": "value1", "key2": "value2"}
    result3 = await cached_execute(dict_input)
    assert result3 == {"key1": "processed_value1", "key2": "processed_value2"}
    assert mock_tool.call_count == 2
    
    # Second call with the same dictionary should use the cache
    result4 = await cached_execute(dict_input)
    assert result4 == {"key1": "processed_value1", "key2": "processed_value2"}
    assert mock_tool.call_count == 2  # Still 2


@pytest.mark.asyncio
async def test_caching_performance_impact():
    """Test the performance impact of caching."""
    # Skip in normal test runs
    pytest.skip("Performance benchmark - run manually")
    
    # Create a mock LLM with high latency
    slow_llm = MockLLM(latency=0.5)
    
    # Create a cache
    cache = Cache(
        primary_backend=InMemoryCache(
            max_size=1000,
            default_ttl=60,
            strategy=CacheStrategy.LRU
        )
    )
    
    # Create a cached version of the LLM's generate method
    @cached(cache_instance=cache)
    async def cached_generate(prompt: str, **kwargs):
        return await slow_llm.generate(prompt, **kwargs)
    
    # Generate a list of prompts with some repetition
    num_prompts = 100
    unique_prompts = [f"Prompt {i}" for i in range(20)]
    prompts = [random.choice(unique_prompts) for _ in range(num_prompts)]
    
    # Measure time without caching
    start_time = time.time()
    for prompt in prompts:
        await slow_llm.generate(prompt)
    uncached_time = time.time() - start_time
    
    # Reset the LLM
    slow_llm.call_count = 0
    
    # Measure time with caching
    start_time = time.time()
    for prompt in prompts:
        await cached_generate(prompt)
    cached_time = time.time() - start_time
    
    # Calculate statistics
    cache_hit_rate = 1 - (slow_llm.call_count / num_prompts)
    speedup = uncached_time / cached_time
    
    # Print results
    print("\nCaching Performance Impact:")
    print("==========================")
    print(f"Number of prompts: {num_prompts}")
    print(f"Unique prompts: {len(unique_prompts)}")
    print(f"Cache hit rate: {cache_hit_rate:.2%}")
    print(f"Time without caching: {uncached_time:.2f}s")
    print(f"Time with caching: {cached_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Assert that caching provides a significant speedup
    assert cached_time < uncached_time
    assert slow_llm.call_count < num_prompts


@pytest.mark.asyncio
async def test_multi_component_caching():
    """Test caching with multiple components working together."""
    # Create components
    llm = MockLLM(latency=0.1)
    agent = MockAgent(llm=llm)
    tool = MockTool(latency=0.05)
    
    # Create a cache
    cache = Cache(
        primary_backend=InMemoryCache(
            max_size=1000,
            default_ttl=60,
            strategy=CacheStrategy.LRU
        )
    )
    
    # Create cached versions of the methods
    @cached(cache_instance=cache)
    async def cached_generate(prompt: str, **kwargs):
        return await llm.generate(prompt, **kwargs)
    
    @cached(cache_instance=cache)
    async def cached_process(input_text: str):
        # Use the cached LLM inside the agent
        response = await cached_generate(input_text)
        agent.call_count += 1
        return {
            "input": input_text,
            "output": response,
            "agent": agent.name
        }
    
    @cached(cache_instance=cache)
    async def cached_execute(input_data: Any):
        return await tool.execute(input_data)
    
    # Simulate a workflow: process input, then execute tool with the result
    async def workflow(input_text: str):
        # Process the input with the agent
        process_result = await cached_process(input_text)
        
        # Execute the tool with the agent's output
        tool_result = await cached_execute(process_result["output"])
        
        return {
            "input": input_text,
            "process_result": process_result,
            "tool_result": tool_result
        }
    
    # First run of the workflow
    start_time = time.time()
    result1 = await workflow("Test input")
    first_run_time = time.time() - start_time
    
    assert llm.call_count == 1
    assert agent.call_count == 1
    assert tool.call_count == 1
    
    # Second run with the same input (should use cache for everything)
    start_time = time.time()
    result2 = await workflow("Test input")
    second_run_time = time.time() - start_time
    
    assert llm.call_count == 1  # Still 1
    assert agent.call_count == 1  # Still 1
    assert tool.call_count == 1  # Still 1
    
    # Verify the results are the same
    assert result1 == result2
    
    # The second run should be significantly faster
    assert second_run_time < first_run_time / 5  # At least 5x faster
    
    # Third run with different input
    start_time = time.time()
    result3 = await workflow("Different input")
    
    assert llm.call_count == 2  # Increased
    assert agent.call_count == 2  # Increased
    assert tool.call_count == 2  # Increased
    
    # Verify the results are different
    assert result3 != result1
