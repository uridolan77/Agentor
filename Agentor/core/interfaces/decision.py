"""
Decision interfaces for the Agentor framework.

This module defines the interfaces for decision components in the Agentor framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DecisionPolicyProvider(Protocol):
    """Protocol for decision policy providers."""
    
    def decide(self, agent: Any) -> str:
        """Make a decision based on the agent's state.
        
        Args:
            agent: The agent to make a decision for
            
        Returns:
            The name of the action to take
        """
        ...


class IDecisionPolicy(ABC):
    """Interface for decision policy components."""
    
    @abstractmethod
    def decide(self, agent: Any) -> str:
        """Make a decision based on the agent's state.
        
        Args:
            agent: The agent to make a decision for
            
        Returns:
            The name of the action to take
        """
        pass


class IDecisionStrategy(ABC):
    """Interface for decision strategy components.
    
    Decision strategies are more complex than policies and can involve
    multiple steps, planning, or other advanced decision-making techniques.
    """
    
    @abstractmethod
    async def plan(self, agent: Any, goal: str) -> List[str]:
        """Create a plan to achieve a goal.
        
        Args:
            agent: The agent to create a plan for
            goal: The goal to achieve
            
        Returns:
            A list of action names to take
        """
        pass
    
    @abstractmethod
    async def evaluate(self, agent: Any, plan: List[str]) -> float:
        """Evaluate a plan.
        
        Args:
            agent: The agent to evaluate the plan for
            plan: The plan to evaluate
            
        Returns:
            A score for the plan
        """
        pass
    
    @abstractmethod
    async def execute(self, agent: Any, plan: List[str]) -> Any:
        """Execute a plan.
        
        Args:
            agent: The agent to execute the plan for
            plan: The plan to execute
            
        Returns:
            The result of executing the plan
        """
        pass
