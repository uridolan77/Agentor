"""Utility-based agent implementation for the Agentor framework.

This module provides a utility-based agent implementation that uses the standardized
interfaces and dependency injection system.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
import asyncio
from functools import wraps

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents.state_models import UtilityBasedAgentState
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.interfaces.agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class UtilityFunction:
    """A utility function for a utility-based agent."""

    def __init__(
        self,
        name: str,
        action: str,
        function: Callable[["UtilityBasedAgent"], float],
        threshold: float = 0.0
    ):
        """Initialize a utility function.

        Args:
            name: The name of the utility function
            action: The name of the action associated with this utility function
            function: A function that takes an agent and returns a utility value
            threshold: The minimum utility value required to consider this action
        """
        self.name = name
        self.action = action
        self.function = function
        self.threshold = threshold


def utility(name: str, action: str, threshold: float = 0.0):
    """Decorator for defining utility functions.

    Args:
        name: The name of the utility function
        action: The name of the action associated with this utility function
        threshold: The minimum utility value required to consider this action

    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        # Add utility function metadata to the function
        wrapper.is_utility = True
        wrapper.utility_name = name
        wrapper.action = action
        wrapper.threshold = threshold

        return wrapper

    return decorator


class UtilityBasedAgent(EnhancedAgent):
    """Utility-based agent implementation."""

    def __init__(
        self,
        name: Optional[str] = None,
        tool_registry: Optional[IToolRegistry] = None
    ):
        """Initialize the utility-based agent.

        Args:
            name: The name of the agent
            tool_registry: Optional tool registry to use
        """
        super().__init__(name, tool_registry)
        self.state_model = UtilityBasedAgentState()
        self.utility_functions: List[UtilityFunction] = []

        # Register utility functions from methods decorated with @utility
        self._register_decorated_utility_functions()

    def _register_decorated_utility_functions(self):
        """Register utility functions from methods decorated with @utility."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, 'is_utility') and attr.is_utility:
                self.add_utility_function(
                    name=attr.utility_name,
                    action=attr.action,
                    function=attr,
                    threshold=attr.threshold
                )

    def add_utility_function(
        self,
        name: str,
        action: str,
        function: Callable[["UtilityBasedAgent"], float],
        threshold: float = 0.0
    ):
        """Add a utility function to the agent.

        Args:
            name: The name of the utility function
            action: The name of the action associated with this utility function
            function: A function that takes an agent and returns a utility value
            threshold: The minimum utility value required to consider this action
        """
        self.utility_functions.append(UtilityFunction(name, action, function, threshold))
        logger.debug(f"Added utility function '{name}' for action '{action}' with threshold {threshold}")

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        # Calculate utilities for all actions
        action_utilities = {}
        for util_func in self.utility_functions:
            utility_value = util_func.function(self)
            action_utilities[util_func.action] = utility_value

            # Store in state model
            self.state_model.action_utilities[util_func.action] = utility_value

        # Filter actions by threshold
        valid_actions = {
            action: utility
            for action, utility in action_utilities.items()
            if utility >= self.get_utility_threshold(action)
        }

        if not valid_actions:
            logger.warning("No actions with utility above threshold, using default action")
            return self._default_action()

        # Select the action with the highest utility
        selected_action = max(valid_actions.items(), key=lambda x: x[1])[0]
        self.state_model.selected_action = selected_action
        self.state_model.last_utility = valid_actions[selected_action]

        logger.debug(f"Selected action '{selected_action}' with utility {valid_actions[selected_action]}")

        return selected_action

    def get_utility_threshold(self, action: str) -> float:
        """Get the utility threshold for an action.

        Args:
            action: The name of the action

        Returns:
            The utility threshold
        """
        # Find the utility function for this action
        for util_func in self.utility_functions:
            if util_func.action == action:
                return util_func.threshold

        # Default threshold
        return self.state_model.utility_threshold

    def _default_action(self) -> str:
        """Get the default action when no actions have utility above threshold.

        Returns:
            The name of the default action
        """
        # Override this method in subclasses to provide a custom default action
        if self.actions:
            return list(self.actions.keys())[0]
        else:
            raise ValueError("No actions registered and no actions with utility above threshold")

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

        # Reset utilities
        self.state_model.action_utilities = {}
        self.state_model.selected_action = None
        self.state_model.last_utility = None

        return await super().preprocess(input_data)

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        # Add utility information to the metadata
        output_data.metadata["utility_based"] = {
            "action_utilities": self.state_model.action_utilities,
            "selected_action": self.state_model.selected_action,
            "utility": self.state_model.last_utility
        }

        return await super().postprocess(output_data)

    @property
    def state(self) -> Dict[str, Any]:
        """Get the agent's state.

        Returns:
            The agent's state
        """
        return self.state_model.dict()

    @state.setter
    def state(self, value: Dict[str, Any]):
        """Set the agent's state.

        Args:
            value: The new state
        """
        # Update the state model with the new values
        for key, val in value.items():
            if hasattr(self.state_model, key):
                setattr(self.state_model, key, val)