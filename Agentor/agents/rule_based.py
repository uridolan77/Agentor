"""Rule-based agent implementation for the Agentor framework.

This module provides a rule-based agent implementation that uses the standardized
interfaces and dependency injection system.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
import asyncio
from functools import wraps

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents.state_models import RuleBasedAgentState
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.interfaces.agent import AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class Rule:
    """A rule for a rule-based agent."""

    def __init__(
        self,
        name: str,
        condition: Callable[["RuleBasedAgent"], bool],
        action: str,
        priority: int = 1
    ):
        """Initialize a rule.

        Args:
            name: The name of the rule
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the rule (higher values = higher priority)
        """
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority


def rule(name: str, action: str, priority: int = 1):
    """Decorator for defining rules.

    Args:
        name: The name of the rule
        action: The name of the action to take if the condition is met
        priority: The priority of the rule (higher values = higher priority)

    Returns:
        A decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        # Add rule metadata to the function
        wrapper.is_rule = True
        wrapper.rule_name = name
        wrapper.action = action
        wrapper.priority = priority

        return wrapper

    return decorator


class RuleBasedAgent(EnhancedAgent):
    """Rule-based agent implementation."""

    def __init__(
        self,
        name: Optional[str] = None,
        tool_registry: Optional[IToolRegistry] = None
    ):
        """Initialize the rule-based agent.

        Args:
            name: The name of the agent
            tool_registry: Optional tool registry to use
        """
        super().__init__(name, tool_registry)
        self.state_model = RuleBasedAgentState()
        self.rules: List[Rule] = []

        # Register rules from methods decorated with @rule
        self._register_decorated_rules()

    def _register_decorated_rules(self):
        """Register rules from methods decorated with @rule."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, 'is_rule') and attr.is_rule:
                self.add_rule(
                    name=attr.rule_name,
                    condition=attr,
                    action=attr.action,
                    priority=attr.priority
                )

    def add_rule(
        self,
        name: str,
        condition: Callable[["RuleBasedAgent"], bool],
        action: str,
        priority: int = 1
    ):
        """Add a rule to the agent.

        Args:
            name: The name of the rule
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the rule (higher values = higher priority)
        """
        self.rules.append(Rule(name, condition, action, priority))
        self.state_model.rule_priorities[name] = priority
        logger.debug(f"Added rule '{name}' with priority {priority} for action '{action}'")

    def decide(self) -> str:
        """Make a decision based on the current state.

        Returns:
            The name of the action to take
        """
        # Find all applicable rules
        applicable_rules = []
        for rule in self.rules:
            if rule.condition(self):
                applicable_rules.append(rule)
                self.state_model.applicable_rules.append(rule.name)

        if not applicable_rules:
            logger.warning("No applicable rules found, using default action")
            return self._default_action()

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        # Get the highest-priority rule
        selected_rule = applicable_rules[0]
        self.state_model.selected_rule = selected_rule.name

        logger.debug(f"Selected rule '{selected_rule.name}' with priority {selected_rule.priority}")

        # Return the action of the highest-priority rule
        return selected_rule.action

    def _default_action(self) -> str:
        """Get the default action when no rules apply.

        Returns:
            The name of the default action
        """
        # Override this method in subclasses to provide a custom default action
        if self.actions:
            return list(self.actions.keys())[0]
        else:
            raise ValueError("No actions registered and no applicable rules found")

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

        # Reset applicable rules
        self.state_model.applicable_rules = []
        self.state_model.selected_rule = None

        return await super().preprocess(input_data)

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        # Add rule information to the metadata
        output_data.metadata["rule_based"] = {
            "applicable_rules": self.state_model.applicable_rules,
            "selected_rule": self.state_model.selected_rule
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