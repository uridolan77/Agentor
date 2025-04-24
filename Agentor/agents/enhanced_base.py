"""
Enhanced base agent implementation for the Agentor framework.

This module provides an enhanced base agent implementation that uses the standardized
interfaces and dependency injection system. The Agent class provides a concrete
implementation of the AbstractAgent interface, while the EnhancedAgent class adds
plugin support and more extensive capabilities.
"""

from typing import Dict, Any, Callable, List, Optional, Tuple, TypeVar, Union
import asyncio
import time
import logging
import traceback

from agentor.core.interfaces.agent import IAgent, AgentInput, AgentOutput
from agentor.core.interfaces.tool import ITool, ToolResult, IToolRegistry
from agentor.core.di import inject, Inject
from agentor.core.plugin import Plugin
from agentor.components.decision import IDecisionPolicy
from agentor.agents.abstract_agent import AbstractAgent, AsyncResource, AgentError, ActionError, SensorError, ResourceNotFoundError, ToolExecutionError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigurationError(AgentError):
    """Exception raised for errors in agent configuration."""
    pass


class DecisionError(AgentError):
    """Exception raised for errors in the decision-making process."""
    pass


class Agent(AbstractAgent):
    """Base class for all agents.

    This class implements the AbstractAgent interface and provides a concrete
    implementation of the common agent functionality. It handles the basic
    lifecycle of an agent, including initialization, execution, and cleanup.
    
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
            tool_registry: Optional tool registry to use for tool registration
        """
        super().__init__(name, tool_registry)
        logger.debug(f"Created agent: {self.name}")
        
    @property
    def description(self) -> str:
        """Get the description of the agent.
        
        Returns:
            The description of the agent
        """
        return "Base agent implementation with core functionality"

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        This is a good place to put IO-bound operations.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        This is a good place to put async operations.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        return output_data

    async def perceive(self) -> Dict[str, Any]:
        """Collect data from all sensors asynchronously.

        Returns:
            A dictionary of sensor readings
            
        Raises:
            SensorError: If a critical sensor fails
        """
        perception = {}
        # Run all sensors in parallel
        sensor_tasks = []
        for name, sensor in self.sensors.items():
            if asyncio.iscoroutinefunction(sensor):
                sensor_tasks.append(self._run_async_sensor(name, sensor))
            else:
                try:
                    perception[name] = sensor(self)
                except Exception as e:
                    logger.error(f"Error running sensor '{name}': {e}")
                    perception[name] = {"error": str(e)}

        # Gather results from async sensors
        if sensor_tasks:
            sensor_results = await asyncio.gather(*sensor_tasks, return_exceptions=True)
            for name, result in sensor_results:
                if isinstance(result, Exception):
                    perception[name] = {"error": str(result)}
                else:
                    perception[name] = result

        self.state['last_perception'] = perception
        return perception

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement decide()")

    async def act(self, action_name: str) -> Any:
        """Execute the specified action asynchronously.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
            
        Raises:
            ActionError: If the action is not found or fails to execute
        """
        if action_name not in self.actions:
            raise ActionError(f"Unknown action: {action_name}")

        try:
            action_func = self.actions[action_name]
            if asyncio.iscoroutinefunction(action_func):
                return await action_func(self)
            else:
                return action_func(self)
        except Exception as e:
            logger.error(f"Error executing action '{action_name}': {e}")
            raise ActionError(f"Error executing action '{action_name}': {str(e)}")

    async def run_once(self) -> Any:
        """Run one perception-decision-action cycle asynchronously.

        Returns:
            The result of the action
            
        Raises:
            SensorError: If perception fails
            DecisionError: If decision-making fails
            ActionError: If action execution fails
        """
        try:
            # First, perceive the environment
            await self.perceive()
            
            # Next, make a decision based on perception
            try:
                action = self.decide()
            except Exception as e:
                logger.error(f"Decision error: {e}")
                raise DecisionError(f"Failed to make a decision: {str(e)}")
            
            # Finally, execute the action
            return await self.act(action)
        except Exception as e:
            logger.error(f"Error in run_once: {e}")
            raise

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the agent with the given input.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            The agent's response
            
        Raises:
            AgentError: If an error occurs during agent execution
        """
        try:
            input_data = AgentInput(query=query, context=context or {})

            # Preprocess the input
            processed_input = await self.preprocess(input_data)

            # Run the agent's core logic
            self.state['current_query'] = processed_input.query
            self.state['current_context'] = processed_input.context
            self.state['start_time'] = time.time()

            result = await self.run_once()

            # Calculate execution time
            execution_time = time.time() - self.state['start_time']

            # Create the output
            output_data = AgentOutput(
                response=str(result) if result is not None else "",
                metadata={
                    "agent_name": self.name,
                    "execution_time": execution_time,
                    "state": self.state
                }
            )

            # Postprocess the output
            processed_output = await self.postprocess(output_data)

            return processed_output
        except Exception as e:
            logger.error(f"Error running agent '{self.name}': {e}")
            # Return a graceful error response
            return AgentOutput(
                response=f"An error occurred while processing your request: {str(e)}",
                metadata={
                    "agent_name": self.name,
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "traceback": traceback.format_exc()
                }
            )

    async def initialize(self) -> None:
        """Initialize the agent.

        This method is called when the agent is created. It should set up any
        resources needed by the agent.
        """
        logger.info(f"Initializing agent: {self.name}")
        self._start_time = time.time()
        self.state['initialization_time'] = time.time()

    async def shutdown(self) -> None:
        """Shutdown the agent.

        This method is called when the agent is destroyed. It should clean up any
        resources used by the agent.
        """
        logger.info(f"Shutting down agent: {self.name}")
        self.state['shutdown_time'] = time.time()
        self.state['total_uptime'] = time.time() - self._start_time


class EnhancedAgent(Agent, Plugin):
    """Enhanced base class for all agents.

    This class implements both the AbstractAgent interface and the Plugin interface,
    making it compatible with both the agent ecosystem and the plugin system.
    It adds support for tool registration, decision policies, and more extensive
    logging and error handling.
    
    Attributes:
        _tool_registry: A registry of available tools
        _decision_policy: An optional policy for making decisions
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tool_registry: Optional[IToolRegistry] = None,
        decision_policy: Optional[IDecisionPolicy] = None
    ):
        """Initialize the agent.

        Args:
            name: The name of the agent
            tool_registry: Optional tool registry to use
            decision_policy: Optional decision policy to use
        """
        super().__init__(name, tool_registry)
        self._decision_policy = decision_policy

        logger.debug(f"Created enhanced agent: {self.name}")
        
        # Record initialization parameters in state
        self.state['agent_type'] = self.__class__.__name__
        self.state['has_decision_policy'] = self._decision_policy is not None
        self.state['has_tool_registry'] = self._tool_registry is not None
        
        if self._decision_policy:
            self.state['decision_policy_type'] = self._decision_policy.__class__.__name__

    @property
    def version(self) -> str:
        """Get the version of the agent.

        Returns:
            The version of the agent
        """
        return "0.2.0"  # Updated version number to reflect enhancements

    @property
    def description(self) -> str:
        """Get the description of the agent.

        Returns:
            The description of the agent
        """
        return "Enhanced base agent with plugin support and extended capabilities"

    def set_decision_policy(self, policy: IDecisionPolicy) -> None:
        """Set the decision policy for the agent.

        Args:
            policy: The decision policy to use
            
        Raises:
            ValueError: If the policy is None
        """
        if policy is None:
            raise ValueError("Decision policy cannot be None")
            
        self._decision_policy = policy
        self.state['decision_policy_type'] = self._decision_policy.__class__.__name__
        logger.debug(f"Set decision policy: {policy.__class__.__name__}")

    def decide(self) -> str:
        """Make a decision based on the current state.

        If a decision policy is set, it will be used to make the decision.
        Otherwise, subclasses must implement this method.

        Returns:
            The name of the action to take
            
        Raises:
            NotImplementedError: If no decision policy is set and the method is not overridden
            DecisionError: If the decision policy fails
        """
        if self._decision_policy is not None:
            try:
                start_time = time.time()
                decision = self._decision_policy.decide(self)
                execution_time = time.time() - start_time
                
                # Store metrics about the decision
                self.state['last_decision_time'] = execution_time
                self.state['last_decision'] = decision
                
                return decision
            except Exception as e:
                logger.error(f"Decision policy failed: {e}")
                raise DecisionError(f"Decision policy failed: {str(e)}")

        # If no decision policy is set, subclasses must implement this method
        raise NotImplementedError("Subclasses must implement decide() or set a decision policy")

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        This is a good place to put IO-bound operations.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        # Record the input data in the state
        self.state['input_query'] = input_data.query
        self.state['input_context_keys'] = list(input_data.context.keys()) if input_data.context else []
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        This is a good place to put async operations.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        # Add additional metadata to the output
        output_data.metadata["agent_version"] = self.version
        output_data.metadata["agent_description"] = self.description
        output_data.metadata["tools_available"] = list(self.tools.keys())
        output_data.metadata["actions_available"] = list(self.actions.keys())
        
        return output_data

    # Override lifecycle methods to add enhanced initialization and shutdown
    async def initialize(self) -> None:
        """Initialize the agent.

        This method is called when the agent is created. It should set up any
        resources needed by the agent. The enhanced version adds validation and
        more detailed logging.
        
        Raises:
            ConfigurationError: If the agent is not properly configured
        """
        logger.info(f"Initializing enhanced agent: {self.name}")
        
        # Validate configuration
        if hasattr(self, 'validate_configuration'):
            try:
                await self.validate_configuration()
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")
                raise ConfigurationError(f"Agent configuration validation failed: {str(e)}")
        
        await super().initialize()
        
        logger.info(f"Enhanced agent {self.name} successfully initialized")

    async def shutdown(self) -> None:
        """Shutdown the agent.

        This method is called when the agent is destroyed. It should clean up any
        resources used by the agent. The enhanced version adds more graceful error
        handling during shutdown.
        """
        logger.info(f"Shutting down enhanced agent: {self.name}")
        
        try:
            await super().shutdown()
            logger.info(f"Enhanced agent {self.name} successfully shut down")
        except Exception as e:
            logger.error(f"Error during enhanced agent shutdown: {e}")
            # Log but don't re-raise to ensure shutdown continues

    # Plugin interface implementation
    def initialize_plugin(self, registry) -> None:
        """Initialize the plugin.

        This method is called when the plugin is registered with the plugin system.

        Args:
            registry: The plugin registry
        """
        logger.info(f"Initializing agent plugin: {self.name}")
        self.state['plugin_initialized'] = True
        self.state['plugin_registry'] = registry.__class__.__name__

    def shutdown_plugin(self) -> None:
        """Shutdown the plugin.

        This method is called when the plugin is unregistered from the plugin system.
        """
        logger.info(f"Shutting down agent plugin: {self.name}")
        self.state['plugin_shutdown'] = True


class EnhancedAgentFactory:
    """Factory for creating enhanced agents.
    
    This factory provides a central point for registering, creating, and managing
    different types of agents. It supports dependency injection for agent creation,
    making it easy to provide the required dependencies to agents.
    """

    def __init__(self):
        """Initialize the agent factory."""
        self.agent_types: Dict[str, type] = {}
        logger.info("Agent factory initialized")

    def register_agent_type(self, name: str, agent_class: type) -> None:
        """Register an agent type.

        Args:
            name: The name of the agent type
            agent_class: The agent class
            
        Raises:
            ValueError: If the agent class is not a subclass of EnhancedAgent
            ValueError: If an agent type with the same name is already registered
        """
        if not issubclass(agent_class, EnhancedAgent):
            raise ValueError(f"Agent class must be a subclass of EnhancedAgent: {agent_class.__name__}")

        if name in self.agent_types:
            logger.warning(f"Overwriting existing agent type: {name}")
            
        self.agent_types[name] = agent_class
        logger.info(f"Registered agent type: {name} ({agent_class.__name__})")

    @inject
    async def create_agent(
        self,
        agent_type: str,
        config: Dict[str, Any] = None,
        tool_registry: Optional[IToolRegistry] = None,
        decision_policy: Optional[IDecisionPolicy] = None
    ) -> EnhancedAgent:
        """Create an agent.

        Args:
            agent_type: The type of agent to create
            config: The configuration for the agent
            tool_registry: Optional tool registry to use
            decision_policy: Optional decision policy to use

        Returns:
            The created agent

        Raises:
            ValueError: If the agent type is not registered
            ConfigurationError: If the agent could not be created with the provided configuration
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = self.agent_types[agent_type]
        logger.debug(f"Creating agent of type {agent_type} ({agent_class.__name__})")

        try:
            # Create the agent
            agent = agent_class(
                tool_registry=tool_registry,
                decision_policy=decision_policy,
                **(config or {})
            )
            
            # Initialize the agent
            await agent.initialize()
            
            logger.info(f"Successfully created agent of type {agent_type}: {agent.name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent of type {agent_type}: {e}")
            raise ConfigurationError(f"Failed to create agent of type {agent_type}: {str(e)}")

    def get_agent_types(self) -> List[str]:
        """Get the available agent types.

        Returns:
            A list of available agent types
        """
        return list(self.agent_types.keys())
        
    def get_agent_class(self, agent_type: str) -> Optional[type]:
        """Get the agent class for a given agent type.
        
        Args:
            agent_type: The type of agent to get the class for
            
        Returns:
            The agent class, or None if the agent type is not registered
        """
        return self.agent_types.get(agent_type)


# Global agent factory
agent_factory = EnhancedAgentFactory()


def get_agent_factory() -> EnhancedAgentFactory:
    """Get the global agent factory.

    Returns:
        The global agent factory
    """
    return agent_factory
