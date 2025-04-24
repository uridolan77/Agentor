"""
Abstract base agent implementation for the Agentor framework.

This module provides an abstract base class for all agent implementations,
defining the common interface and lifecycle methods that all agents should implement.
The AbstractAgent class serves as the foundation for the entire agent ecosystem,
ensuring consistent behavior and interaction patterns across different agent types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional, Tuple, AsyncContextManager, TypeVar, Generic, Union
import uuid
import asyncio
import time
import logging
import traceback
from contextlib import asynccontextmanager
from pydantic import BaseModel

from agentor.agents.tools.base import BaseTool, ToolResult
from agentor.core.interfaces.agent import AgentInput, AgentOutput, IAgent
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.di import Inject, inject

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception class for all agent-related errors."""
    pass


class ResourceNotFoundError(AgentError):
    """Exception raised when a resource is not found."""
    pass


class ToolExecutionError(AgentError):
    """Exception raised when a tool execution fails."""
    pass


class SensorError(AgentError):
    """Exception raised when a sensor fails."""
    pass


class ActionError(AgentError):
    """Exception raised when an action fails."""
    pass


class AsyncResource:
    """A wrapper for an async resource with a cleanup function.
    
    This class provides a convenient way to manage async resources with
    proper cleanup handling, ensuring resources are properly released
    when they are no longer needed.
    """

    def __init__(self, resource: Any, cleanup_func: Callable):
        """Initialize the async resource.

        Args:
            resource: The resource to wrap
            cleanup_func: A function that takes the resource and returns an async context manager
        """
        self.resource = resource
        self.cleanup_func = cleanup_func

    @asynccontextmanager
    async def __call__(self):
        """Get the resource as an async context manager.

        Returns:
            An async context manager for the resource
            
        Raises:
            Exception: If an error occurs while creating or cleaning up the resource
        """
        try:
            async with self.cleanup_func(self.resource) as resource:
                yield resource
        except Exception as e:
            logger.error(f"Error managing async resource: {e}")
            raise


class AbstractAgent(ABC, IAgent):
    """Abstract base class for all agents.

    This class defines the common interface and lifecycle methods that all agents
    should implement. It provides a foundation for building different types of agents
    with consistent behavior and implements the IAgent interface to ensure compatibility
    with the agent ecosystem.
    
    Attributes:
        state: A dictionary storing the agent's current state
        sensors: A dictionary of registered sensor functions
        actions: A dictionary of registered action functions
        tools: A dictionary of registered tools
        resources: A dictionary of registered async resources
    """

    def __init__(self, name: Optional[str] = None, tool_registry: Optional[IToolRegistry] = None):
        """Initialize the agent.

        Args:
            name: The name of the agent, or None to generate a random name
            tool_registry: Optional tool registry to use for automatically registering tools
        """
        self._name = name or f"Agent-{uuid.uuid4().hex[:8]}"
        self.state: Dict[str, Any] = {}
        self.sensors: Dict[str, Callable] = {}
        self.actions: Dict[str, Callable] = {}
        self.tools: Dict[str, BaseTool] = {}
        self.resources: Dict[str, AsyncResource] = {}
        self._start_time = time.time()
        self._tool_registry = tool_registry
        
        # Register tools from the tool registry if provided
        if tool_registry:
            for name, tool in tool_registry.get_tools().items():
                self.register_tool(tool)

    @property
    def name(self) -> str:
        """Get the name of the agent.

        Returns:
            The name of the agent
        """
        return self._name
        
    @property
    def version(self) -> str:
        """Get the version of the agent.
        
        Returns:
            The version of the agent
        """
        return "0.1.0"
        
    @property
    def description(self) -> str:
        """Get the description of the agent.
        
        Returns:
            The description of the agent
        """
        return "Abstract base agent"

    def register_sensor(self, name: str, sensor_func: Callable) -> None:
        """Register a sensor function.

        Args:
            name: The name of the sensor
            sensor_func: A function that takes the agent as input and returns sensor data
            
        Raises:
            ValueError: If a sensor with the same name is already registered
        """
        if name in self.sensors:
            logger.warning(f"Overwriting existing sensor: {name}")
        self.sensors[name] = sensor_func
        logger.debug(f"Registered sensor: {name}")

    def register_action(self, name: str, action_func: Callable) -> None:
        """Register an action function.

        Args:
            name: The name of the action
            action_func: A function that takes the agent as input and performs an action
            
        Raises:
            ValueError: If an action with the same name is already registered
        """
        if name in self.actions:
            logger.warning(f"Overwriting existing action: {name}")
        self.actions[name] = action_func
        logger.debug(f"Registered action: {name}")

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: The tool to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self.tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_resource(self, name: str, resource: Any, cleanup_func: Callable) -> None:
        """Register an async resource.

        Args:
            name: The name of the resource
            resource: The resource to register
            cleanup_func: A function that takes the resource and returns an async context manager
            
        Raises:
            ValueError: If a resource with the same name is already registered
        """
        if name in self.resources:
            logger.warning(f"Overwriting existing resource: {name}")
        self.resources[name] = AsyncResource(resource, cleanup_func)
        logger.debug(f"Registered resource: {name}")

    async def get_resource(self, name: str) -> AsyncContextManager:
        """Get an async resource.

        Args:
            name: The name of the resource

        Returns:
            An async context manager for the resource
            
        Raises:
            ResourceNotFoundError: If the resource is not found
        """
        if name not in self.resources:
            raise ResourceNotFoundError(f"Resource not found: {name}")
        try:
            return self.resources[name]()
        except Exception as e:
            logger.error(f"Error accessing resource '{name}': {e}")
            raise ResourceNotFoundError(f"Error accessing resource '{name}': {str(e)}")

    async def execute_tools(self, tools_to_run: List[Tuple[BaseTool, Dict[str, Any]]]) -> List[ToolResult]:
        """Execute multiple tools in parallel.

        Args:
            tools_to_run: A list of (tool, params) tuples to run

        Returns:
            A list of tool results
            
        Raises:
            ToolExecutionError: If any tool execution fails
        """
        try:
            return await asyncio.gather(
                *(tool.run(**params) for tool, params in tools_to_run),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error executing tools: {e}")
            raise ToolExecutionError(f"Error executing tools: {str(e)}")

    async def _run_async_sensor(self, name: str, sensor: Callable) -> Tuple[str, Any]:
        """Run an async sensor and return its name and result.

        Args:
            name: The name of the sensor
            sensor: The sensor function

        Returns:
            A tuple of (name, result)
        """
        try:
            result = await sensor(self)
            return name, result
        except Exception as e:
            logger.error(f"Error running sensor '{name}': {e}")
            return name, SensorError(f"Error running sensor '{name}': {str(e)}")

    @abstractmethod
    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        This is a good place to put IO-bound operations.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        pass

    @abstractmethod
    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        This is a good place to put IO-bound operations.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        pass

    @abstractmethod
    async def perceive(self) -> Dict[str, Any]:
        """Collect data from all sensors asynchronously.

        Returns:
            A dictionary of sensor readings
            
        Raises:
            SensorError: If a critical sensor fails
        """
        pass

    @abstractmethod
    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        pass

    @abstractmethod
    async def act(self, action_name: str) -> Any:
        """Execute the specified action asynchronously.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
            
        Raises:
            ActionError: If the action fails or doesn't exist
        """
        pass

    @abstractmethod
    async def run_once(self) -> Any:
        """Run one perception-decision-action cycle asynchronously.

        Returns:
            The result of the action
        """
        pass

    @abstractmethod
    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the agent with the given input.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            The agent's response
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent.

        This method is called when the agent is created. It should set up any
        resources needed by the agent.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the agent.

        This method is called when the agent is destroyed. It should clean up any
        resources used by the agent.
        """
        pass

    @asynccontextmanager
    async def lifespan(self) -> AsyncContextManager:
        """Manage the agent's lifecycle.

        This context manager initializes the agent and cleans up resources when done.
        
        Yields:
            The agent instance
        """
        try:
            # Initialize the agent
            await self.initialize()
            self._start_time = time.time()
            yield self
        except Exception as e:
            logger.error(f"Error during agent lifespan: {e}")
            raise
        finally:
            # Clean up resources
            try:
                await self.shutdown()
            except Exception as e:
                logger.error(f"Error during agent shutdown: {e}")
