from typing import Callable, List, Tuple, Any, Dict, Optional, Protocol, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class IDecisionPolicy(ABC):
    """Interface for decision policies.

    A decision policy determines what action an agent should take based on its current state.
    """

    @abstractmethod
    def decide(self, agent) -> str:
        """Make a decision for the agent.

        Args:
            agent: The agent to make a decision for

        Returns:
            The name of the action to take
        """
        pass


class Rule:
    """A rule for a rule-based decision engine."""

    def __init__(self, condition: Callable, action: str, priority: int = 1):
        """Initialize a rule.

        Args:
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the rule (higher values = higher priority)
        """
        self.condition = condition
        self.action = action
        self.priority = priority


class RuleBasedDecisionPolicy(IDecisionPolicy):
    """A decision policy that uses rules to make decisions."""

    def __init__(self):
        """Initialize the decision policy."""
        self.rules: List[Rule] = []

    def add_rule(self, condition: Callable, action: str, priority: int = 1):
        """Add a rule to the decision policy.

        Args:
            condition: A function that takes an agent and returns a boolean
            action: The name of the action to take if the condition is met
            priority: The priority of the rule (higher values = higher priority)
        """
        self.rules.append(Rule(condition, action, priority))

    def decide(self, agent) -> str:
        """Make a decision for the agent.

        Args:
            agent: The agent to make a decision for

        Returns:
            The name of the action to take
        """
        # Find all applicable rules
        applicable_rules = []
        for rule in self.rules:
            if rule.condition(agent):
                applicable_rules.append(rule)

        if not applicable_rules:
            raise ValueError("No applicable rules found")

        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)

        # Return the action of the highest-priority rule
        return applicable_rules[0].action


class PatternMatchingDecisionPolicy(IDecisionPolicy):
    """A decision policy that uses pattern matching on the query to make decisions."""

    def __init__(self, default_action: str = "default"):
        """Initialize the decision policy.

        Args:
            default_action: The default action to take if no patterns match
        """
        self.patterns: List[Tuple[str, str]] = []
        self.default_action = default_action

    def add_pattern(self, pattern: str, action: str):
        """Add a pattern to the decision policy.

        Args:
            pattern: A string pattern to match in the query
            action: The name of the action to take if the pattern is found
        """
        self.patterns.append((pattern, action))

    def decide(self, agent) -> str:
        """Make a decision for the agent.

        Args:
            agent: The agent to make a decision for

        Returns:
            The name of the action to take
        """
        # Get the query from the agent's state
        query = agent.state.get('current_query', '').lower()

        # Check each pattern
        for pattern, action in self.patterns:
            if pattern.lower() in query:
                return action

        # Return the default action if no patterns match
        return self.default_action


class MLBasedDecisionPolicy(IDecisionPolicy):
    """A decision policy that uses a machine learning model to make decisions."""

    def __init__(self, model, default_action: str = "default"):
        """Initialize the decision policy.

        Args:
            model: The machine learning model to use
            default_action: The default action to take if the model fails
        """
        self.model = model
        self.default_action = default_action

    def decide(self, agent) -> str:
        """Make a decision for the agent.

        Args:
            agent: The agent to make a decision for

        Returns:
            The name of the action to take
        """
        try:
            # Extract features from the agent's state
            features = self._extract_features(agent)

            # Use the model to predict the action
            action = self.model.predict(features)

            return action
        except Exception as e:
            logger.error(f"Error using ML model for decision: {str(e)}")
            return self.default_action

    def _extract_features(self, agent) -> Any:
        """Extract features from the agent's state for the model.

        Args:
            agent: The agent to extract features from

        Returns:
            The extracted features
        """
        # This is a placeholder - in a real implementation, this would extract
        # relevant features from the agent's state for the model
        return {"query": agent.state.get('current_query', '')}


class CompositeDecisionPolicy(IDecisionPolicy):
    """A decision policy that combines multiple policies."""

    def __init__(self, fallback_action: str = "default"):
        """Initialize the decision policy.

        Args:
            fallback_action: The action to take if all policies fail
        """
        self.policies: List[IDecisionPolicy] = []
        self.fallback_action = fallback_action

    def add_policy(self, policy: IDecisionPolicy):
        """Add a policy to the composite.

        Args:
            policy: The policy to add
        """
        self.policies.append(policy)

    def decide(self, agent) -> str:
        """Make a decision for the agent.

        Args:
            agent: The agent to make a decision for

        Returns:
            The name of the action to take
        """
        for policy in self.policies:
            try:
                return policy.decide(agent)
            except Exception as e:
                logger.warning(f"Policy {policy.__class__.__name__} failed: {str(e)}")

        # If all policies fail, return the fallback action
        return self.fallback_action


# For backward compatibility
RuleBasedDecisionEngine = RuleBasedDecisionPolicy
