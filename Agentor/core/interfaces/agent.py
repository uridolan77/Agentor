"""
Agent interfaces for the Agentor framework.

This module defines the interfaces for agent components in the Agentor framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple, AsyncContextManager, Protocol, runtime_checkable
from pydantic import BaseModel

from agentor.core.interfaces.tool import ToolResult


class AgentInput(BaseModel):
    """Input data for an agent."""
    
    query: str
    """The query to process."""
    
    context: Optional[Dict[str, Any]] = None
    """Additional context for the query."""


class AgentOutput(BaseModel):
    """Output data from an agent."""
    
    response: str
    """The agent's response."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata for the response."""


@runtime_checkable
class AgentProvider(Protocol):
    """Protocol for agent providers."""
    
    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the agent with a query and context.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The agent's response
        """
        ...


class IAgent(ABC):
    """Interface for agent components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the agent.
        
        Returns:
            The name of the agent
        """
        pass
    
    @abstractmethod
    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.
        
        Args:
            input_data: The input data to preprocess
            
        Returns:
            The preprocessed input data
        """
        pass
    
    @abstractmethod
    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.
        
        Args:
            output_data: The output data to postprocess
            
        Returns:
            The postprocessed output data
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
        """Execute the specified action.
        
        Args:
            action_name: The name of the action to execute
            
        Returns:
            The result of the action
        """
        pass
    
    @abstractmethod
    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        """Run the agent with a query and context.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The agent's response
        """
        pass


class IMultiAgent(IAgent):
    """Interface for multi-agent systems."""
    
    @abstractmethod
    async def add_agent(self, agent: IAgent) -> None:
        """Add an agent to the system.
        
        Args:
            agent: The agent to add
        """
        pass
    
    @abstractmethod
    async def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the system.
        
        Args:
            agent_name: The name of the agent to remove
        """
        pass
    
    @abstractmethod
    async def get_agent(self, agent_name: str) -> Optional[IAgent]:
        """Get an agent by name.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The agent, or None if not found
        """
        pass
    
    @abstractmethod
    async def route(self, input_data: AgentInput) -> Tuple[IAgent, float]:
        """Route an input to the appropriate agent.
        
        Args:
            input_data: The input data to route
            
        Returns:
            A tuple of (agent, confidence)
        """
        pass


class IAgentFactory(ABC):
    """Interface for agent factories."""
    
    @abstractmethod
    async def create_agent(self, agent_type: str, config: Dict[str, Any]) -> IAgent:
        """Create an agent.
        
        Args:
            agent_type: The type of agent to create
            config: The configuration for the agent
            
        Returns:
            The created agent
        """
        pass
    
    @abstractmethod
    async def get_agent_types(self) -> List[str]:
        """Get the available agent types.
        
        Returns:
            A list of available agent types
        """
        pass
