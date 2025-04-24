"""
Tests for the AbstractAgent class.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional

from agentor.agents.abstract_agent import AbstractAgent
from agentor.core.interfaces.agent import AgentInput, AgentOutput


class TestAgent(AbstractAgent):
    """Test implementation of AbstractAgent for testing."""
    
    def __init__(self, name=None):
        super().__init__(name)
        self.perceive_called = False
        self.decide_called = False
        self.act_called = False
        self.initialize_called = False
        self.shutdown_called = False
        self.test_action_result = "Test action result"
    
    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Test implementation of preprocess."""
        input_data.context["preprocessed"] = True
        return input_data
    
    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Test implementation of postprocess."""
        output_data.metadata["postprocessed"] = True
        return output_data
    
    async def perceive(self) -> Dict[str, Any]:
        """Test implementation of perceive."""
        self.perceive_called = True
        return {"test": "perception"}
    
    def decide(self) -> str:
        """Test implementation of decide."""
        self.decide_called = True
        return "test_action"
    
    async def act(self, action_name: str) -> Any:
        """Test implementation of act."""
        self.act_called = True
        if action_name == "test_action":
            return self.test_action_result
        raise ValueError(f"Unknown action: {action_name}")
    
    async def run_once(self) -> Any:
        """Test implementation of run_once."""
        await self.perceive()
        action = self.decide()
        return await self.act(action)
    
    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Test implementation of run."""
        input_data = AgentInput(query=query, context=context or {})
        
        # Preprocess the input
        processed_input = await self.preprocess(input_data)
        
        # Run the agent's core logic
        self.state['current_query'] = processed_input.query
        self.state['current_context'] = processed_input.context
        
        result = await self.run_once()
        
        # Create the output
        output_data = AgentOutput(
            response=result,
            metadata={
                "agent_name": self.name,
                "state": self.state
            }
        )
        
        # Postprocess the output
        processed_output = await self.postprocess(output_data)
        
        return processed_output
    
    async def initialize(self) -> None:
        """Test implementation of initialize."""
        self.initialize_called = True
    
    async def shutdown(self) -> None:
        """Test implementation of shutdown."""
        self.shutdown_called = True


@pytest.mark.asyncio
async def test_abstract_agent_initialization():
    """Test that the AbstractAgent can be initialized."""
    agent = TestAgent(name="TestAgent")
    assert agent.name == "TestAgent"
    assert isinstance(agent.state, dict)
    assert isinstance(agent.sensors, dict)
    assert isinstance(agent.actions, dict)
    assert isinstance(agent.tools, dict)
    assert isinstance(agent.resources, dict)


@pytest.mark.asyncio
async def test_abstract_agent_lifecycle():
    """Test the AbstractAgent lifecycle."""
    agent = TestAgent(name="TestAgent")
    
    # Test the lifespan context manager
    async with agent.lifespan() as a:
        assert a is agent
        assert agent.initialize_called
    
    assert agent.shutdown_called


@pytest.mark.asyncio
async def test_abstract_agent_run():
    """Test the AbstractAgent run method."""
    agent = TestAgent(name="TestAgent")
    
    # Register a test action
    agent.register_action("test_action", lambda a: "Test action result")
    
    # Run the agent
    result = await agent.run("Test query", {"test": "context"})
    
    # Check that the agent ran correctly
    assert agent.perceive_called
    assert agent.decide_called
    assert agent.act_called
    
    # Check the result
    assert result.response == "Test action result"
    assert result.metadata["agent_name"] == "TestAgent"
    assert result.metadata["postprocessed"] is True
    
    # Check that the state was updated
    assert agent.state["current_query"] == "Test query"
    assert agent.state["current_context"]["test"] == "context"
    assert agent.state["current_context"]["preprocessed"] is True


@pytest.mark.asyncio
async def test_abstract_agent_tools():
    """Test the AbstractAgent tool methods."""
    agent = TestAgent(name="TestAgent")
    
    # Create a mock tool
    class MockTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description
        
        async def run(self, **kwargs):
            return {"result": "Mock tool result"}
    
    # Register the tool
    tool = MockTool("mock_tool", "A mock tool")
    agent.register_tool(tool)
    
    # Check that the tool was registered
    assert "mock_tool" in agent.tools
    assert agent.tools["mock_tool"] is tool
    
    # Execute the tool
    result = await agent.execute_tools([(tool, {"param": "value"})])
    
    # Check the result
    assert len(result) == 1
    assert result[0]["result"] == "Mock tool result"


@pytest.mark.asyncio
async def test_abstract_agent_sensors():
    """Test the AbstractAgent sensor methods."""
    agent = TestAgent(name="TestAgent")
    
    # Register a synchronous sensor
    agent.register_sensor("sync_sensor", lambda a: "Sync sensor result")
    
    # Register an asynchronous sensor
    async def async_sensor(a):
        await asyncio.sleep(0.1)
        return "Async sensor result"
    
    agent.register_sensor("async_sensor", async_sensor)
    
    # Override the perceive method to use the registered sensors
    original_perceive = agent.perceive
    
    async def test_perceive():
        perception = {}
        # Run all sensors in parallel
        sensor_tasks = []
        for name, sensor in agent.sensors.items():
            if asyncio.iscoroutinefunction(sensor):
                sensor_tasks.append(agent._run_async_sensor(name, sensor))
            else:
                perception[name] = sensor(agent)
        
        # Gather results from async sensors
        if sensor_tasks:
            sensor_results = await asyncio.gather(*sensor_tasks, return_exceptions=True)
            for name, result in sensor_results:
                if isinstance(result, Exception):
                    perception[name] = {"error": str(result)}
                else:
                    perception[name] = result
        
        return perception
    
    agent.perceive = test_perceive
    
    # Run the perceive method
    perception = await agent.perceive()
    
    # Check the results
    assert perception["sync_sensor"] == "Sync sensor result"
    assert perception["async_sensor"] == "Async sensor result"
    
    # Restore the original perceive method
    agent.perceive = original_perceive


@pytest.mark.asyncio
async def test_abstract_agent_resources():
    """Test the AbstractAgent resource methods."""
    agent = TestAgent(name="TestAgent")
    
    # Create a mock resource
    class MockResource:
        def __init__(self):
            self.value = "Mock resource value"
            self.closed = False
        
        def close(self):
            self.closed = True
    
    # Create a mock cleanup function
    @asyncio.contextmanager
    async def cleanup_func(resource):
        try:
            yield resource
        finally:
            resource.close()
    
    # Register the resource
    resource = MockResource()
    agent.register_resource("mock_resource", resource, cleanup_func)
    
    # Check that the resource was registered
    assert "mock_resource" in agent.resources
    
    # Use the resource
    async with await agent.get_resource("mock_resource") as r:
        assert r is resource
        assert r.value == "Mock resource value"
        assert not r.closed
    
    # Check that the resource was closed
    assert resource.closed
