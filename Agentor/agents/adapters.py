"""
Adapters for existing agent implementations to use the new standardized interfaces.

This module provides adapter classes that wrap existing agent implementations
to make them compatible with the new standardized interfaces.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
import asyncio
import time
from contextlib import asynccontextmanager

from agentor.agents.enhanced_base import Agent as OldAgent
from agentor.core.interfaces.agent import AgentInput as OldAgentInput
from agentor.core.interfaces.agent import AgentOutput as OldAgentOutput
from agentor.components.learning.deep_q_learning import DeepQLearningAgent
from agentor.components.learning.ppo_agent import PPOAgent
from agentor.llm_gateway.agents.router import SemanticRouter
from agentor.llm_gateway.agents.hierarchical_router import HierarchicalRouter, RuleBasedRouter

from agentor.core.interfaces.agent import (
    IAgent, IMultiAgent, IAgentRouter, IAgentFactory,
    AgentInput, AgentOutput
)
from abc import ABC, abstractmethod
from agentor.core.interfaces.tool import ITool, IToolRegistry
from agentor.core.plugin import Plugin
from agentor.core.registry import get_component_registry
from agentor.core.di import inject, get_container

logger = logging.getLogger(__name__)


def convert_input(input_data: AgentInput) -> OldAgentInput:
    """Convert a new AgentInput to an old AgentInput.

    Args:
        input_data: The new input to convert

    Returns:
        The converted old input
    """
    return OldAgentInput(
        query=input_data.query,
        context=input_data.context or {}
    )


def convert_output(output_data: OldAgentOutput) -> AgentOutput:
    """Convert an old AgentOutput to a new AgentOutput.

    Args:
        output_data: The old output to convert

    Returns:
        The converted new output
    """
    return AgentOutput(
        response=output_data.response,
        metadata=output_data.metadata
    )


class AgentAdapter(IAgent):
    """Adapter for the base Agent class."""

    def __init__(self, agent: OldAgent):
        """Initialize the agent adapter.

        Args:
            agent: The agent implementation to adapt
        """
        self.agent = agent

    @property
    def name(self) -> str:
        """Get the name of the agent.

        Returns:
            The name of the agent
        """
        return self.agent.name

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        old_input = convert_input(input_data)
        old_output = await self.agent.preprocess(old_input)
        return AgentInput(
            query=old_output.query,
            context=old_output.context
        )

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        old_output = OldAgentOutput(
            response=output_data.response,
            metadata=output_data.metadata or {}
        )
        old_processed = await self.agent.postprocess(old_output)
        return AgentOutput(
            response=old_processed.response,
            metadata=old_processed.metadata
        )

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        return self.agent.decide()

    async def act(self, action_name: str) -> Any:
        """Execute the specified action.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
        """
        return await self.agent.act(action_name)

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the agent with a query and context.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            The agent's response
        """
        old_output = await self.agent.run(query, context)
        return convert_output(old_output)


class DeepQLearningAgentAdapter(AgentAdapter):
    """Adapter for the DeepQLearningAgent class."""

    def __init__(self, agent: Optional[DeepQLearningAgent] = None, **kwargs):
        """Initialize the Deep Q-Learning agent adapter.

        Args:
            agent: The Deep Q-Learning agent implementation to adapt, or None to create a new one
            **kwargs: Additional arguments to pass to the DeepQLearningAgent constructor
        """
        super().__init__(agent or DeepQLearningAgent(**kwargs))
        self.dql_agent = self.agent

    async def train(self, num_episodes: int, max_steps: int = 100) -> Dict[str, Any]:
        """Train the agent.

        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode

        Returns:
            Training statistics
        """
        return await self.dql_agent.train(num_episodes, max_steps)

    async def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model to
        """
        await self.dql_agent.save_model(path)

    async def load_model(self, path: str) -> None:
        """Load the model from a file.

        Args:
            path: Path to load the model from
        """
        await self.dql_agent.load_model(path)


class PPOAgentAdapter(AgentAdapter):
    """Adapter for the PPOAgent class."""

    def __init__(self, agent: Optional[PPOAgent] = None, **kwargs):
        """Initialize the PPO agent adapter.

        Args:
            agent: The PPO agent implementation to adapt, or None to create a new one
            **kwargs: Additional arguments to pass to the PPOAgent constructor
        """
        super().__init__(agent or PPOAgent(**kwargs))
        self.ppo_agent = self.agent

    async def train(self, num_episodes: int, max_steps: int = 100) -> Dict[str, Any]:
        """Train the agent.

        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode

        Returns:
            Training statistics
        """
        return await self.ppo_agent.train(num_episodes, max_steps)

    async def save_model(self, path: str) -> None:
        """Save the model to a file.

        Args:
            path: Path to save the model to
        """
        await self.ppo_agent.save_model(path)

    async def load_model(self, path: str) -> None:
        """Load the model from a file.

        Args:
            path: Path to load the model from
        """
        await self.ppo_agent.load_model(path)


class RouteHistory:
    """History of routing decisions."""

    def __init__(self):
        """Initialize the route history."""
        self.routes: List[Dict[str, Any]] = []

    def add_route(self, router_name: str, agent_name: str, confidence: float, query: str, timestamp: Optional[float] = None):
        """Add a routing decision to the history.

        Args:
            router_name: The name of the router
            agent_name: The name of the selected agent
            confidence: The confidence score
            query: The query that was routed
            timestamp: Optional timestamp (defaults to current time)
        """
        self.routes.append({
            "router": router_name,
            "agent": agent_name,
            "confidence": confidence,
            "query": query,
            "timestamp": timestamp or time.time()
        })

    def get_last_route(self) -> Optional[Dict[str, Any]]:
        """Get the last routing decision.

        Returns:
            The last routing decision, or None if no routes exist
        """
        if not self.routes:
            return None
        return self.routes[-1]

    def get_routes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the routing history.

        Args:
            limit: Optional limit on the number of routes to return

        Returns:
            The routing history
        """
        if limit is None:
            return self.routes
        return self.routes[-limit:]

    def clear(self):
        """Clear the routing history."""
        self.routes = []


class BaseRouterAdapter(IAgentRouter, ABC):
    """Base class for router adapters.

    This class provides common functionality for router adapters.
    """

    def __init__(self, enable_tracing: bool = True):
        """Initialize the base router adapter.

        Args:
            enable_tracing: Whether to enable route tracing
        """
        self._agents: Dict[str, IAgent] = {}
        self._resources: Dict[str, Any] = {}
        self._enable_tracing = enable_tracing
        self._route_history = RouteHistory() if enable_tracing else None

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the router.

        Returns:
            The name of the router
        """
        pass

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before routing.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after routing.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        return output_data

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        return "route"

    async def act(self, action_name: str) -> Any:
        """Execute the specified action.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
        """
        if action_name != "route":
            raise ValueError(f"Unknown action: {action_name}")

        # This will be handled by the run method
        return None

    def get_route_history(self) -> Optional[RouteHistory]:
        """Get the route history.

        Returns:
            The route history, or None if tracing is disabled
        """
        return self._route_history

    def enable_tracing(self, enable: bool = True):
        """Enable or disable route tracing.

        Args:
            enable: Whether to enable tracing
        """
        if enable and self._route_history is None:
            self._route_history = RouteHistory()
            self._enable_tracing = True
        elif not enable:
            self._route_history = None
            self._enable_tracing = False

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the router with a query and context.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            The router's response
        """
        input_data = AgentInput(query=query, context=context or {})
        agent, confidence = await self.route(input_data)

        # Add routing information to the context
        if context is None:
            context = {}

        # Record the routing decision in the history
        if self._enable_tracing and self._route_history is not None:
            self._route_history.add_route(
                router_name=self.name,
                agent_name=agent.name,
                confidence=confidence,
                query=query
            )

        # Add routing information to the context
        routing_info = {
            "name": self.name,
            "selected_agent": agent.name,
            "confidence": confidence,
            "timestamp": time.time()
        }

        # Add route history if available
        if self._enable_tracing and self._route_history is not None:
            routing_info["history"] = self._route_history.get_routes(limit=5)

        context["router"] = routing_info

        # Run the selected agent
        return await agent.run(query, context)

    @abstractmethod
    async def route(self, input_data: AgentInput) -> Tuple[IAgent, float]:
        """Route an input to the appropriate agent.

        Args:
            input_data: The input data to route

        Returns:
            A tuple of (agent, confidence)
        """
        pass

    def register_resource(self, name: str, resource: Any, cleanup_func: Optional[Callable] = None) -> None:
        """Register a resource for cleanup.

        Args:
            name: The name of the resource
            resource: The resource to register
            cleanup_func: A function to call when cleaning up the resource
        """
        self._resources[name] = (resource, cleanup_func)
        logger.debug(f"Registered resource: {name}")

    async def cleanup_resources(self) -> None:
        """Clean up all registered resources.

        This method ensures that all resources are attempted to be cleaned up,
        even if some cleanup operations fail. Errors are logged but do not
        prevent other resources from being cleaned up.
        """
        # Make a copy of the resources to avoid modification during iteration
        resources_to_cleanup = list(self._resources.items())
        cleanup_errors = []

        # Track successfully cleaned up resources
        cleaned_up = []

        # First pass: try to clean up each resource
        for name, (resource, cleanup_func) in resources_to_cleanup:
            try:
                if cleanup_func is not None:
                    if asyncio.iscoroutinefunction(cleanup_func):
                        await cleanup_func(resource)
                    else:
                        cleanup_func(resource)
                logger.debug(f"Cleaned up resource: {name}")
                cleaned_up.append(name)
            except Exception as e:
                error_msg = f"Error cleaning up resource {name}: {str(e)}"
                logger.error(error_msg)
                cleanup_errors.append((name, error_msg))

        # Remove successfully cleaned up resources
        for name in cleaned_up:
            if name in self._resources:
                del self._resources[name]

        # Log summary
        if cleanup_errors:
            logger.warning(f"Failed to clean up {len(cleanup_errors)} resources: {', '.join(name for name, _ in cleanup_errors)}")

        if cleaned_up:
            logger.info(f"Successfully cleaned up {len(cleaned_up)} resources")

        # Return remaining resources for potential manual cleanup
        return cleanup_errors

    @asynccontextmanager
    async def lifespan(self):
        """Manage the router's lifecycle.

        This context manager initializes the router and cleans up resources when done.
        """
        try:
            # Initialize the router
            yield self
        finally:
            # Clean up resources
            await self.cleanup_resources()


class SemanticRouterAdapter(BaseRouterAdapter):
    """Adapter for the SemanticRouter class."""

    def __init__(self, router: Optional[SemanticRouter] = None, **kwargs):
        """Initialize the semantic router adapter.

        Args:
            router: The semantic router implementation to adapt, or None to create a new one
            **kwargs: Additional arguments to pass to the SemanticRouter constructor
        """
        super().__init__()
        self.router = router or SemanticRouter(**kwargs)
        # Register the router as a resource for cleanup
        if hasattr(self.router, 'close') and callable(self.router.close):
            self.register_resource('router', self.router, self.router.close)

    @property
    def name(self) -> str:
        """Get the name of the router.

        Returns:
            The name of the router
        """
        return "semantic_router"

    async def route(self, input_data: AgentInput) -> Tuple[IAgent, float]:
        """Route an input to the appropriate agent.

        Args:
            input_data: The input data to route

        Returns:
            A tuple of (agent, confidence)
        """
        # Convert the input
        old_input = convert_input(input_data)

        # Get the old agents from the router
        old_agents = self.router.agents

        # Create a mapping from old agent names to new agents
        agent_map = {}
        for name, old_agent in old_agents.items():
            if name in self._agents:
                agent_map[name] = self._agents[name]
            else:
                # Create a new adapter for the old agent
                adapter = AgentAdapter(old_agent)
                self._agents[name] = adapter
                agent_map[name] = adapter

        # Route the input
        agent_name, confidence = await self.router.route_query(old_input.query)

        # Get the selected agent
        if agent_name not in agent_map:
            raise ValueError(f"Unknown agent: {agent_name}")

        return agent_map[agent_name], confidence

    def register_agent(self, agent: IAgent, embeddings: Optional[List[float]] = None) -> None:
        """Register an agent with the router.

        Args:
            agent: The agent to register
            embeddings: Optional embeddings for the agent
        """
        # Store the agent
        self._agents[agent.name] = agent

        # Create an old agent adapter if needed
        if isinstance(agent, AgentAdapter):
            old_agent = agent.agent
        else:
            # Create a wrapper for the new agent
            old_agent = OldAgent(agent.name)
            old_agent.decide = agent.decide
            old_agent.act = agent.act
            old_agent.run = lambda query, context=None: agent.run(query, context)

        # Register the old agent with the router
        self.router.register_agent(old_agent, embeddings)


class RuleBasedRouterAdapter(BaseRouterAdapter):
    """Adapter for the RuleBasedRouter class."""

    def __init__(self, router: Optional[RuleBasedRouter] = None, **kwargs):
        """Initialize the rule-based router adapter.

        Args:
            router: The rule-based router implementation to adapt, or None to create a new one
            **kwargs: Additional arguments to pass to the RuleBasedRouter constructor
        """
        super().__init__()
        self.router = router or RuleBasedRouter(**kwargs)
        # Register the router as a resource for cleanup
        if hasattr(self.router, 'close') and callable(self.router.close):
            self.register_resource('router', self.router, self.router.close)

    @property
    def name(self) -> str:
        """Get the name of the router.

        Returns:
            The name of the router
        """
        return "rule_based_router"

    async def route(self, input_data: AgentInput) -> Tuple[IAgent, float]:
        """Route an input to the appropriate agent.

        Args:
            input_data: The input data to route

        Returns:
            A tuple of (agent, confidence)
        """
        # Convert the input
        old_input = convert_input(input_data)

        # Get the old agents from the router
        old_agents = self.router.agents

        # Create a mapping from old agent names to new agents
        agent_map = {}
        for name, old_agent in old_agents.items():
            if name in self._agents:
                agent_map[name] = self._agents[name]
            else:
                # Create a new adapter for the old agent
                adapter = AgentAdapter(old_agent)
                self._agents[name] = adapter
                agent_map[name] = adapter

        # Route the input
        agent_name, confidence = await self.router.route_query(old_input.query)

        # Get the selected agent
        if agent_name not in agent_map:
            raise ValueError(f"Unknown agent: {agent_name}")

        return agent_map[agent_name], confidence

    def register_agent(self, agent: IAgent, rules: List[str]) -> None:
        """Register an agent with the router.

        Args:
            agent: The agent to register
            rules: Rules for when to use this agent
        """
        # Store the agent
        self._agents[agent.name] = agent

        # Create an old agent adapter if needed
        if isinstance(agent, AgentAdapter):
            old_agent = agent.agent
        else:
            # Create a wrapper for the new agent
            old_agent = OldAgent(agent.name)
            old_agent.decide = agent.decide
            old_agent.act = agent.act
            old_agent.run = lambda query, context=None: agent.run(query, context)

        # Register the old agent with the router
        self.router.register_agent(old_agent, rules)


class HierarchicalRouterAdapter(BaseRouterAdapter):
    """Adapter for the HierarchicalRouter class."""

    def __init__(
        self,
        semantic_router: Optional[SemanticRouterAdapter] = None,
        rule_router: Optional[RuleBasedRouterAdapter] = None,
        fallback_agent: Optional[IAgent] = None,
        router: Optional[HierarchicalRouter] = None
    ):
        """Initialize the hierarchical router adapter.

        Args:
            semantic_router: The semantic router adapter
            rule_router: The rule-based router adapter
            fallback_agent: The fallback agent
            router: The hierarchical router implementation to adapt, or None to create a new one
        """
        super().__init__()

        if router is not None:
            self.router = router

            # Create adapters for the routers and fallback agent
            if semantic_router is None:
                semantic_router = SemanticRouterAdapter(router.semantic_router)

            if rule_router is None:
                rule_router = RuleBasedRouterAdapter(router.rule_router)

            if fallback_agent is None:
                fallback_agent = AgentAdapter(router.fallback_agent)
        else:
            # Ensure we have all the required components
            if semantic_router is None or rule_router is None or fallback_agent is None:
                raise ValueError("Must provide semantic_router, rule_router, and fallback_agent")

            # Get the old routers and fallback agent
            old_semantic_router = semantic_router.router
            old_rule_router = rule_router.router

            # Get the old fallback agent
            if isinstance(fallback_agent, AgentAdapter):
                old_fallback_agent = fallback_agent.agent
            else:
                # Create a wrapper for the new agent
                old_fallback_agent = OldAgent(fallback_agent.name)
                old_fallback_agent.decide = fallback_agent.decide
                old_fallback_agent.act = fallback_agent.act
                old_fallback_agent.run = lambda query, context=None: fallback_agent.run(query, context)

            # Create the hierarchical router
            self.router = HierarchicalRouter(
                semantic_router=old_semantic_router,
                rule_router=old_rule_router,
                fallback_agent=old_fallback_agent
            )

        self.semantic_router = semantic_router
        self.rule_router = rule_router
        self.fallback_agent = fallback_agent

        # Register resources for cleanup
        if hasattr(self.router, 'close') and callable(self.router.close):
            self.register_resource('router', self.router, self.router.close)

    @property
    def name(self) -> str:
        """Get the name of the router.

        Returns:
            The name of the router
        """
        return "hierarchical_router"

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before routing.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after routing.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        return output_data

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        return "route"

    async def act(self, action_name: str) -> Any:
        """Execute the specified action.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
        """
        if action_name != "route":
            raise ValueError(f"Unknown action: {action_name}")

        # This will be handled by the run method
        return None

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the router with a query and context.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            The router's response
        """
        old_input = OldAgentInput(query=query, context=context or {})
        old_output = await self.router.route(old_input)
        return convert_output(old_output)

    async def route(self, input_data: AgentInput) -> Tuple[IAgent, float]:
        """Route an input to the appropriate agent.

        Args:
            input_data: The input data to route

        Returns:
            A tuple of (agent, confidence)
        """
        # Try the semantic router first
        try:
            agent, confidence = await self.semantic_router.route(input_data)
            if confidence >= 0.7:  # High confidence threshold
                return agent, confidence
        except Exception as e:
            logger.warning(f"Semantic router failed: {str(e)}")

        # Try the rule-based router next
        try:
            agent, confidence = await self.rule_router.route(input_data)
            if confidence >= 0.5:  # Medium confidence threshold
                return agent, confidence
        except Exception as e:
            logger.warning(f"Rule-based router failed: {str(e)}")

        # Fall back to the fallback agent
        return self.fallback_agent, 0.0


# Agent plugins

class BaseAgentPlugin(Plugin):
    """Plugin for the base Agent class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "base_agent"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Base agent implementation"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Create a simple agent adapter
        agent = OldAgent(name="base_agent")
        agent.register_action("default", lambda a: "Default response")
        agent.decide = lambda: "default"

        agent_adapter = AgentAdapter(agent)

        # Register the agent provider
        component_registry = get_component_registry()
        component_registry.register_agent_provider("base", agent_adapter)

        logger.info("Registered base agent provider")


class DeepQLearningAgentPlugin(Plugin):
    """Plugin for the DeepQLearningAgent class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "deep_q_learning_agent"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Agent that learns using Deep Q-Learning"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        try:
            # Create the agent adapter
            agent_adapter = DeepQLearningAgentAdapter(
                state_dim=10,
                action_dim=5,
                hidden_dim=128
            )

            # Register the agent provider
            component_registry = get_component_registry()
            component_registry.register_agent_provider("deep_q_learning", agent_adapter)

            logger.info("Registered Deep Q-Learning agent provider")
        except ImportError:
            logger.warning("PyTorch not available, skipping Deep Q-Learning agent registration")


class PPOAgentPlugin(Plugin):
    """Plugin for the PPOAgent class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "ppo_agent"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Agent that learns using Proximal Policy Optimization"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        try:
            # Create the agent adapter
            agent_adapter = PPOAgentAdapter(
                state_dim=10,
                action_dim=5,
                hidden_dim=128
            )

            # Register the agent provider
            component_registry = get_component_registry()
            component_registry.register_agent_provider("ppo", agent_adapter)

            logger.info("Registered PPO agent provider")
        except ImportError:
            logger.warning("PyTorch not available, skipping PPO agent registration")


class SemanticRouterPlugin(Plugin):
    """Plugin for the SemanticRouter class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "semantic_router"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Router that uses semantic similarity to route queries to agents"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Create the router adapter
        router_adapter = SemanticRouterAdapter()

        # Register the agent provider
        component_registry = get_component_registry()
        component_registry.register_agent_provider("semantic_router", router_adapter)

        # Store the adapter for cleanup
        self.router_adapter = router_adapter

        logger.info("Registered semantic router provider")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        if hasattr(self, 'router_adapter'):
            # Store the adapter for later cleanup
            # The actual cleanup will be handled by the plugin registry
            # which should properly manage the event loop
            logger.info("Marked semantic router resources for cleanup")

            # Register a cleanup function with the plugin registry
            from agentor.core.plugin import get_plugin_registry
            registry = get_plugin_registry()
            registry.register_cleanup_task(self.router_adapter.cleanup_resources)


class RuleBasedRouterPlugin(Plugin):
    """Plugin for the RuleBasedRouter class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "rule_based_router"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Router that uses rules to route queries to agents"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Create the router adapter
        router_adapter = RuleBasedRouterAdapter()

        # Register the agent provider
        component_registry = get_component_registry()
        component_registry.register_agent_provider("rule_based_router", router_adapter)

        # Store the adapter for cleanup
        self.router_adapter = router_adapter

        logger.info("Registered rule-based router provider")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        if hasattr(self, 'router_adapter'):
            # Store the adapter for later cleanup
            # The actual cleanup will be handled by the plugin registry
            # which should properly manage the event loop
            logger.info("Marked rule-based router resources for cleanup")

            # Register a cleanup function with the plugin registry
            from agentor.core.plugin import get_plugin_registry
            registry = get_plugin_registry()
            registry.register_cleanup_task(self.router_adapter.cleanup_resources)


class HierarchicalRouterPlugin(Plugin):
    """Plugin for the HierarchicalRouter class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "hierarchical_router"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Router that uses multiple routing strategies with fallback"

    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies of the plugin.

        Returns:
            A list of plugin names that this plugin depends on
        """
        return ["semantic_router", "rule_based_router", "base_agent"]

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Get the component registry
        component_registry = get_component_registry()

        # Get the semantic router
        semantic_router = component_registry.get_agent_provider("semantic_router")
        if not isinstance(semantic_router, SemanticRouterAdapter):
            semantic_router = SemanticRouterAdapter()

        # Get the rule-based router
        rule_router = component_registry.get_agent_provider("rule_based_router")
        if not isinstance(rule_router, RuleBasedRouterAdapter):
            rule_router = RuleBasedRouterAdapter()

        # Get the fallback agent
        fallback_agent = component_registry.get_agent_provider("base")

        # Create the hierarchical router adapter
        router_adapter = HierarchicalRouterAdapter(
            semantic_router=semantic_router,
            rule_router=rule_router,
            fallback_agent=fallback_agent
        )

        # Register the agent provider
        component_registry.register_agent_provider("hierarchical_router", router_adapter)

        # Store the adapter for cleanup
        self.router_adapter = router_adapter

        logger.info("Registered hierarchical router provider")

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        if hasattr(self, 'router_adapter'):
            # Store the adapter for later cleanup
            # The actual cleanup will be handled by the plugin registry
            # which should properly manage the event loop
            logger.info("Marked hierarchical router resources for cleanup")

            # Register a cleanup function with the plugin registry
            from agentor.core.plugin import get_plugin_registry
            registry = get_plugin_registry()
            registry.register_cleanup_task(self.router_adapter.cleanup_resources)
