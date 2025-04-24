"""Tests for the Agentor agents.

This module contains tests for the various agent implementations provided by the Agentor framework.
"""

import pytest
import unittest
import asyncio
from unittest.mock import MagicMock, patch

from agentor.agents import (
    Agent, AgentInput, AgentOutput, EnhancedAgent,
    RuleBasedAgent, UtilityBasedAgent, ReactiveAgent, MemoryEnhancedAgent
)
from agentor.agents.state_models import (
    BaseAgentState, RuleBasedAgentState, UtilityBasedAgentState, ReactiveAgentState
)


class TestAgentImplementation(Agent):
    """A test agent implementation for unit testing."""

    def __init__(self, name=None):
        super().__init__(name)
        self.test_result = "Test result"
        self.perceive_called = False
        self.decide_called = False
        self.act_called = False

    def decide(self):
        """Return a fixed action for testing."""
        self.decide_called = True
        return "test_action"

    async def perceive(self):
        """Return a fixed perception for testing."""
        self.perceive_called = True
        return {"test_perception": "value"}

    async def act(self, action_name):
        """Return a fixed result for testing."""
        self.act_called = True
        if action_name == "test_action":
            return self.test_result
        raise ValueError(f"Unknown action: {action_name}")

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Add a test field to the context."""
        input_data.context["test_field"] = "test_value"
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Add a test field to the metadata."""
        output_data.metadata["test_field"] = "test_value"
        return output_data


@pytest.mark.asyncio
async def test_agent_run():
    """Test the Agent.run method."""
    # Create the agent
    agent = TestAgentImplementation(name="TestAgent")

    # Run the agent
    result = await agent.run("Test query", {"initial_context": "value"})

    # Check the result
    assert result.response == "Test result"
    assert result.metadata["agent_name"] == "TestAgent"
    assert result.metadata["test_field"] == "test_value"
    assert "state" in result.metadata
    assert result.metadata["state"]["current_query"] == "Test query"
    assert result.metadata["state"]["current_context"]["initial_context"] == "value"
    assert result.metadata["state"]["current_context"]["test_field"] == "test_value"


@pytest.mark.asyncio
async def test_agent_run_once():
    """Test the Agent.run_once method."""
    # Create the agent
    agent = TestAgentImplementation(name="TestAgent")

    # Run the agent once
    result = await agent.run_once()

    # Check the result
    assert result == "Test result"
    assert agent.perceive_called
    assert agent.decide_called
    assert agent.act_called


@pytest.mark.asyncio
async def test_agent_lifecycle():
    """Test the agent lifecycle methods."""
    # Create the agent
    agent = TestAgentImplementation(name="TestAgent")

    # Initialize the agent
    await agent.initialize()
    assert agent._start_time is not None

    # Shutdown the agent
    await agent.shutdown()


class TestRuleBasedAgent:
    """Tests for the RuleBasedAgent class."""

    @pytest.mark.asyncio
    async def test_rule_based_agent_initialization(self):
        """Test RuleBasedAgent initialization."""
        # Create a rule-based agent
        agent = RuleBasedAgent(name="test_rule_agent")

        # Test agent properties
        assert agent.name == "test_rule_agent"
        assert isinstance(agent.state, dict)

    @pytest.mark.asyncio
    async def test_rule_based_agent_rules(self):
        """Test adding and using rules in a RuleBasedAgent."""
        # Create a rule-based agent
        agent = RuleBasedAgent(name="test_rule_agent")

        # Add a test rule
        @agent.rule(priority=10)
        def test_rule(agent):
            return "test" in agent.state.get("current_query", "")

        # Register a test action
        agent.register_action("test_rule", lambda a: "Rule triggered")

        # Test with a matching query
        result = await agent.run("This is a test query")
        assert result.response == "Rule triggered"

        # Test with a non-matching query
        result = await agent.run("This is a query")
        assert result.response != "Rule triggered"


class TestUtilityBasedAgent:
    """Tests for the UtilityBasedAgent class."""

    @pytest.mark.asyncio
    async def test_utility_based_agent_initialization(self):
        """Test UtilityBasedAgent initialization."""
        # Create a utility-based agent
        agent = UtilityBasedAgent(name="test_utility_agent")

        # Test agent properties
        assert agent.name == "test_utility_agent"
        assert isinstance(agent.state, dict)

    @pytest.mark.asyncio
    async def test_utility_based_agent_utilities(self):
        """Test adding and using utilities in a UtilityBasedAgent."""
        # Create a utility-based agent
        agent = UtilityBasedAgent(name="test_utility_agent")

        # Add a test utility function
        @agent.utility
        def test_utility(agent):
            if "test" in agent.state.get("current_query", ""):
                return 0.8
            return 0.2

        # Register a test action
        agent.register_action("test_utility", lambda a: "Utility action triggered")

        # Test with a high utility query
        result = await agent.run("This is a test query")
        assert result.response == "Utility action triggered"


class TestReactiveAgent:
    """Tests for the ReactiveAgent class."""

    @pytest.mark.asyncio
    async def test_reactive_agent_initialization(self):
        """Test ReactiveAgent initialization."""
        # Create a reactive agent
        agent = ReactiveAgent(name="test_reactive_agent")

        # Test agent properties
        assert agent.name == "test_reactive_agent"
        assert isinstance(agent.state, dict)

    @pytest.mark.asyncio
    async def test_reactive_agent_behaviors(self):
        """Test adding and using behaviors in a ReactiveAgent."""
        # Create a reactive agent
        agent = ReactiveAgent(name="test_reactive_agent")

        # Add a test behavior
        @agent.behavior(priority=10)
        def test_behavior(agent):
            return "test" in agent.state.get("current_query", "")

        # Register a test action
        agent.register_action("test_behavior", lambda a: "Behavior triggered")

        # Test with a matching query
        result = await agent.run("This is a test query")
        assert result.response == "Behavior triggered"


class TestMemoryEnhancedAgent:
    """Tests for the MemoryEnhancedAgent class."""

    @pytest.mark.asyncio
    async def test_memory_enhanced_agent_initialization(self):
        """Test MemoryEnhancedAgent initialization."""
        # Create a memory-enhanced agent
        agent = MemoryEnhancedAgent(name="test_memory_agent")

        # Test agent properties
        assert agent.name == "test_memory_agent"
        assert isinstance(agent.state, dict)

    @pytest.mark.asyncio
    async def test_memory_enhanced_agent_memory(self):
        """Test memory operations in a MemoryEnhancedAgent."""
        # Create a memory-enhanced agent with a mock memory
        agent = MemoryEnhancedAgent(name="test_memory_agent")

        # Mock the memory methods
        agent.memory = MagicMock()
        agent.memory.add = MagicMock(return_value=asyncio.Future())
        agent.memory.add.return_value.set_result(None)

        agent.memory.get = MagicMock(return_value=asyncio.Future())
        agent.memory.get.return_value.set_result([{"text": "Test memory", "timestamp": 123456789.0}])

        # Add a memory entry
        await agent.add_memory("Test memory")
        agent.memory.add.assert_called_once()

        # Retrieve memories
        memories = await agent.get_memories("test")
        agent.memory.get.assert_called_once()
        assert len(memories) == 1
        assert memories[0]["text"] == "Test memory"


if __name__ == "__main__":
    pytest.main()