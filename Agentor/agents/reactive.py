"""
Reactive agent implementation for the Agentor framework.

This module provides a reactive agent implementation that uses the standardized
interfaces and dependency injection system. Reactive agents operate on a simple
stimulus-response model, executing behaviors when their associated conditions are met.
These agents are particularly well-suited for environments where clear-cut rules can
be defined for agent behavior.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
import logging
import asyncio
import time
from functools import wraps

from agentor.agents.enhanced_base import EnhancedAgent, DecisionError
from agentor.agents.state_models import ReactiveAgentState
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.interfaces.agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class BehaviorExecutionError(DecisionError):
    """Exception raised when a behavior fails to execute."""
    pass


class BehaviorConditionError(DecisionError):
    """Exception raised when a behavior condition fails to evaluate."""
    pass


class Behavior:
    """A behavior for a reactive agent.
    
    A behavior consists of a condition and an action. When the condition is met,
    the action is executed. Behaviors can have priorities, allowing more important
    behaviors to take precedence.
    
    Attributes:
        name: The name of the behavior
        condition: A function that evaluates whether the behavior should be triggered
        action: The name of the action to take if the condition is met
        priority: The priority of the behavior (higher values = higher priority)
    """

    def __init__(
        self,
        name: str,
        condition: Callable[["ReactiveAgent"], bool],
        action: str,
        priority: int = 1
    ):
        """Initialize the behavior.

        Args:
            name: The name of the behavior
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the behavior (higher values = higher priority)
            
        Raises:
            ValueError: If the name is empty or if the priority is not a positive integer
        """
        if not name:
            raise ValueError("Behavior name cannot be empty")
        if priority <= 0:
            raise ValueError("Behavior priority must be a positive integer")
            
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority
        self.last_triggered_time: Optional[float] = None
        self.trigger_count: int = 0


def behavior(name: str, action: str, priority: int = 1):
    """Decorator for defining behaviors.

    This decorator allows defining behaviors directly as methods on the agent class,
    making it easier to organize and maintain behaviors that are closely tied to
    the agent's implementation.
    
    Args:
        name: The name of the behavior
        action: The name of the action to take if the condition is met
        priority: The priority of the behavior (higher values = higher priority)

    Returns:
        A decorator function
        
    Example:
        >>> class MyAgent(ReactiveAgent):
        ...     @behavior(name="high_temperature", action="activate_cooling", priority=2)
        ...     def check_temperature(self) -> bool:
        ...         return self.state.get("temperature", 0) > 30
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        # Add behavior metadata to the function
        wrapper.is_behavior = True
        wrapper.behavior_name = name
        wrapper.action = action
        wrapper.priority = priority

        return wrapper

    return decorator


class ReactiveAgent(EnhancedAgent):
    """Reactive agent implementation.
    
    Reactive agents operate on a simple stimulus-response model, executing
    behaviors when their associated conditions are met. They are particularly
    useful for scenarios where clear rules can define the agent's behavior.
    
    Key features:
    - Behaviors with conditions and actions
    - Priority-based behavior selection
    - Support for both method-based and function-based behavior definitions
    
    Examples:
        >>> agent = ReactiveAgent("reactive_agent")
        >>> agent.add_behavior(
        ...     name="high_temperature",
        ...     condition=lambda agent: agent.state.get("temperature", 0) > 30,
        ...     action="cooling_system",
        ...     priority=2
        ... )
        >>> agent.add_behavior(
        ...     name="low_light",
        ...     condition=lambda agent: agent.state.get("light_level", 100) < 20,
        ...     action="turn_on_light",
        ...     priority=3
        ... )
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tool_registry: Optional[IToolRegistry] = None
    ):
        """Initialize the reactive agent.

        Args:
            name: The name of the agent
            tool_registry: Optional tool registry to use
        """
        super().__init__(name, tool_registry)
        self.state_model = ReactiveAgentState()
        self.behaviors: List[Behavior] = []

        # Register behaviors from methods decorated with @behavior
        self._register_decorated_behaviors()
        
        # Minimum time between behavior evaluations (seconds)
        self.min_evaluation_interval: float = 0.1
        self._last_evaluation_time: float = 0
        
        # Default behavior setup
        self._default_behavior_name: str = "default_behavior"
        self._register_default_actions()
        
    @property
    def version(self) -> str:
        """Get the version of the agent.
        
        Returns:
            The version of the agent
        """
        return "0.2.0"
    
    @property
    def description(self) -> str:
        """Get the description of the agent.
        
        Returns:
            The description of the agent
        """
        return "Reactive agent that executes behaviors when their conditions are met"
        
    def _register_default_actions(self) -> None:
        """Register default actions for the agent."""
        # Register a default no-op action
        if "no_op" not in self.actions:
            self.register_action("no_op", lambda agent: "No operation performed")

    def _register_decorated_behaviors(self) -> None:
        """Register behaviors from methods decorated with @behavior."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, 'is_behavior') and attr.is_behavior:
                self.add_behavior(
                    name=attr.behavior_name,
                    condition=attr,
                    action=attr.action,
                    priority=attr.priority
                )
                logger.debug(f"Registered decorated behavior: {attr.behavior_name}")

    def add_behavior(
        self,
        name: str,
        condition: Callable[["ReactiveAgent"], bool],
        action: str,
        priority: int = 1
    ) -> None:
        """Add a behavior to the agent.

        Args:
            name: The name of the behavior
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the behavior (higher values = higher priority)
            
        Raises:
            ValueError: If a behavior with the same name already exists
            ValueError: If the action is not registered
        """
        # Check if behavior with this name already exists
        if any(b.name == name for b in self.behaviors):
            logger.warning(f"Overwriting existing behavior: {name}")
            self.behaviors = [b for b in self.behaviors if b.name != name]
        
        # Validate that the action exists
        if action not in self.actions and action != self._default_behavior_name:
            raise ValueError(f"Action '{action}' is not registered")
        
        # Create and register the behavior
        behavior = Behavior(name, condition, action, priority)
        self.behaviors.append(behavior)
        self.state_model.behavior_priorities[name] = priority
        
        # Sort behaviors by priority (highest first)
        self.behaviors.sort(key=lambda b: b.priority, reverse=True)
        
        logger.debug(f"Added behavior '{name}' with priority {priority} for action '{action}'")
        
    def remove_behavior(self, name: str) -> bool:
        """Remove a behavior from the agent.
        
        Args:
            name: The name of the behavior to remove
            
        Returns:
            True if the behavior was removed, False if it wasn't found
        """
        initial_count = len(self.behaviors)
        self.behaviors = [b for b in self.behaviors if b.name != name]
        
        if name in self.state_model.behavior_priorities:
            del self.state_model.behavior_priorities[name]
            
        if len(self.behaviors) < initial_count:
            logger.debug(f"Removed behavior: {name}")
            return True
        return False
        
    def get_behaviors(self) -> List[Dict[str, Any]]:
        """Get information about all registered behaviors.
        
        Returns:
            A list of dictionaries with behavior information
        """
        return [
            {
                "name": b.name,
                "priority": b.priority,
                "action": b.action,
                "last_triggered": b.last_triggered_time,
                "trigger_count": b.trigger_count
            }
            for b in self.behaviors
        ]

    def evaluate_behaviors(self) -> List[Behavior]:
        """Evaluate all behaviors and return those with conditions that are true.
        
        Returns:
            A list of behaviors with conditions that are true, sorted by priority
            
        Raises:
            BehaviorConditionError: If behavior condition evaluation fails
        """
        # Track the current evaluation time
        current_time = time.time()
        self._last_evaluation_time = current_time
        
        applicable_behaviors = []
        failed_behaviors = []
        
        for behavior in self.behaviors:
            try:
                if behavior.condition(self):
                    applicable_behaviors.append(behavior)
                    behavior.last_triggered_time = current_time
                    behavior.trigger_count += 1
            except Exception as e:
                logger.error(f"Error evaluating behavior '{behavior.name}': {str(e)}")
                failed_behaviors.append((behavior.name, str(e)))
        
        if failed_behaviors and not applicable_behaviors:
            # Only raise an error if no behaviors are applicable and some failed
            error_details = "; ".join([f"{name}: {error}" for name, error in failed_behaviors])
            raise BehaviorConditionError(f"Error evaluating behaviors: {error_details}")
        
        return applicable_behaviors

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
            
        Raises:
            DecisionError: If no applicable behaviors are found and default action is unavailable
        """
        # Reset applicable behaviors
        self.state_model.applicable_behaviors = []
        
        # Rate limiting: ensure min time between evaluations
        current_time = time.time()
        time_since_last_eval = current_time - self._last_evaluation_time
        if time_since_last_eval < self.min_evaluation_interval:
            # Return the last selected behavior's action if it exists
            if self.state_model.selected_behavior:
                return self.state_model.last_action
            
        # Find all applicable behaviors
        try:
            applicable_behaviors = self.evaluate_behaviors()
            
            # Update the state model with applicable behavior names
            self.state_model.applicable_behaviors = [b.name for b in applicable_behaviors]
            
            if not applicable_behaviors:
                logger.info("No applicable behaviors found, using default action")
                self.state_model.selected_behavior = None
                default_action = self._default_action()
                self.state_model.last_action = default_action
                return default_action
    
            # Get the highest-priority behavior
            selected_behavior = applicable_behaviors[0]
            self.state_model.selected_behavior = selected_behavior.name
    
            logger.debug(f"Selected behavior '{selected_behavior.name}' with priority {selected_behavior.priority}")
    
            # Return the action of the highest-priority behavior
            self.state_model.last_action = selected_behavior.action
            return selected_behavior.action
            
        except Exception as e:
            logger.error(f"Error in decision making: {str(e)}")
            raise DecisionError(f"Failed to make decision: {str(e)}")

    def _default_action(self) -> str:
        """Get the default action when no behaviors apply.

        Returns:
            The name of the default action
            
        Raises:
            DecisionError: If no actions are registered
        """
        # First check if no_op is registered
        if "no_op" in self.actions:
            return "no_op"
            
        # Otherwise, use the first registered action
        if self.actions:
            return list(self.actions.keys())[0]
        else:
            raise DecisionError("No actions registered and no applicable behaviors found")

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        # Update the state model
        self.state_model.current_query = input_data.query
        self.state_model.current_context = input_data.context

        # Check for behavior-specific data in context
        if "behaviors" in input_data.context:
            behavior_data = input_data.context["behaviors"]
            if isinstance(behavior_data, dict):
                # Set behavior-specific state variables
                for behavior_name, data in behavior_data.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            state_key = f"behavior.{behavior_name}.{key}"
                            self.state[state_key] = value
        
        # Reset behaviors tracking
        self.state_model.applicable_behaviors = []
        self.state_model.selected_behavior = None
        self.state_model.last_action = None

        return await super().preprocess(input_data)

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        # Add behavior information to the metadata
        output_data.metadata["reactive"] = {
            "applicable_behaviors": self.state_model.applicable_behaviors,
            "selected_behavior": self.state_model.selected_behavior,
            "behavior_priorities": self.state_model.behavior_priorities,
            "last_action": self.state_model.last_action
        }
        
        # Add information about all behaviors
        output_data.metadata["reactive"]["behaviors"] = self.get_behaviors()

        return await super().postprocess(output_data)

    @property
    def state(self) -> Dict[str, Any]:
        """Get the agent's state.

        Returns:
            The agent's state
        """
        return self.state_model.dict()
        
    async def act(self, action_name: str) -> Any:
        """Execute the specified action asynchronously.

        Args:
            action_name: The name of the action to execute

        Returns:
            The result of the action
            
        Raises:
            BehaviorExecutionError: If the action execution fails
        """
        try:
            result = await super().act(action_name)
            return result
        except Exception as e:
            logger.error(f"Error executing action '{action_name}' for behavior: {str(e)}")
            raise BehaviorExecutionError(f"Error executing action '{action_name}': {str(e)}")
            
    async def initialize(self) -> None:
        """Initialize the agent.
        
        This method sets up any resources needed by the agent and validates
        that the agent has the necessary components to function properly.
        
        Raises:
            ConfigurationError: If the agent is not properly configured
        """
        await super().initialize()
        
        # Log information about behaviors
        if self.behaviors:
            behavior_info = ", ".join([f"{b.name} (priority: {b.priority})" for b in self.behaviors])
            logger.info(f"Reactive agent {self.name} initialized with behaviors: {behavior_info}")
        else:
            logger.info(f"Reactive agent {self.name} initialized with no behaviors")
            
    async def validate_configuration(self) -> None:
        """Validate the agent configuration.
        
        This method checks that the agent has all the necessary components
        to function properly.
        
        Raises:
            ConfigurationError: If the agent is not properly configured
        """
        # Check for any potential configuration issues
        problematic_behaviors = []
        
        # Check that all referenced actions exist
        for behavior in self.behaviors:
            if behavior.action not in self.actions and behavior.action != self._default_behavior_name:
                problematic_behaviors.append(f"Behavior '{behavior.name}' references non-existent action '{behavior.action}'")
        
        if problematic_behaviors:
            issues = "\n- ".join(problematic_behaviors)
            logger.warning(f"Potential configuration issues detected:\n- {issues}")
