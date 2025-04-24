"""
Base classes for multi-agent coordination.

This module provides base classes and common utilities for multi-agent coordination.
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from dataclasses import dataclass

from agentor.core.interfaces.agent import IAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Message between agents."""
    
    sender: str
    receiver: str
    content: str
    timestamp: float = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize the timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create a message from a dictionary.
        
        Args:
            data: The dictionary data
            
        Returns:
            The created message
        """
        return cls(
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata")
        )


class AgentRegistry:
    """Registry of agents."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents: Dict[str, IAgent] = {}
    
    def register(self, agent: IAgent) -> None:
        """Register an agent.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.name] = agent
        logger.info(f"Registered agent {agent.name}")
    
    def unregister(self, agent_name: str) -> None:
        """Unregister an agent.
        
        Args:
            agent_name: The name of the agent to unregister
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Unregistered agent {agent_name}")
    
    def get(self, agent_name: str) -> Optional[IAgent]:
        """Get an agent by name.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The agent, or None if not found
        """
        return self.agents.get(agent_name)
    
    def get_all(self) -> Dict[str, IAgent]:
        """Get all registered agents.
        
        Returns:
            Dictionary mapping agent names to agents
        """
        return self.agents.copy()
    
    def clear(self) -> None:
        """Clear the registry."""
        self.agents.clear()
        logger.info("Cleared agent registry")


class CoordinationContext:
    """Context for agent coordination."""
    
    def __init__(self):
        """Initialize the coordination context."""
        self.data: Dict[str, Any] = {}
        self.messages: List[AgentMessage] = []
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context.
        
        Args:
            key: The key
            value: The value
        """
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context.
        
        Args:
            key: The key
            default: The default value to return if the key is not found
            
        Returns:
            The value, or the default if not found
        """
        return self.data.get(key, default)
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the context.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
    
    def get_messages(self, agent_name: Optional[str] = None) -> List[AgentMessage]:
        """Get messages from the context.
        
        Args:
            agent_name: Optional name of the agent to filter by
            
        Returns:
            List of messages
        """
        if agent_name is None:
            return self.messages.copy()
        
        return [
            message
            for message in self.messages
            if message.sender == agent_name or message.receiver == agent_name
        ]
    
    def clear(self) -> None:
        """Clear the context."""
        self.data.clear()
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "data": self.data,
            "messages": [message.to_dict() for message in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationContext':
        """Create a context from a dictionary.
        
        Args:
            data: The dictionary data
            
        Returns:
            The created context
        """
        context = cls()
        context.data = data.get("data", {})
        context.messages = [
            AgentMessage.from_dict(message_data)
            for message_data in data.get("messages", [])
        ]
        return context
