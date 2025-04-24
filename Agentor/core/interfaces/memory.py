"""
Memory interfaces for the Agentor framework.

This module defines the interfaces for memory components in the Agentor framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic, Protocol, runtime_checkable
from pydantic import BaseModel


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory providers."""
    
    async def add(self, item: Dict[str, Any]) -> None:
        """Add an item to memory.
        
        Args:
            item: The item to add to memory
        """
        ...
    
    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get items from memory that match a query.
        
        Args:
            query: The query to match
            limit: Maximum number of items to return
            
        Returns:
            A list of matching items
        """
        ...
    
    async def clear(self) -> None:
        """Clear all items from memory."""
        ...


class MemoryItem(BaseModel):
    """Base model for memory items."""
    
    id: Optional[str] = None
    """Unique identifier for the item."""
    
    content: Dict[str, Any]
    """The content of the item."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Metadata for the item."""


class MemoryQuery(BaseModel):
    """Base model for memory queries."""
    
    filter: Optional[Dict[str, Any]] = None
    """Filter criteria for the query."""
    
    limit: int = 10
    """Maximum number of items to return."""
    
    offset: int = 0
    """Offset for pagination."""
    
    sort: Optional[List[Dict[str, str]]] = None
    """Sort criteria for the query."""


class IMemory(ABC):
    """Interface for memory components."""
    
    @abstractmethod
    async def add(self, item: Dict[str, Any]) -> None:
        """Add an item to memory.
        
        Args:
            item: The item to add to memory
        """
        pass
    
    @abstractmethod
    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get items from memory that match a query.
        
        Args:
            query: The query to match
            limit: Maximum number of items to return
            
        Returns:
            A list of matching items
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all items from memory."""
        pass


class IEpisodicMemory(IMemory):
    """Interface for episodic memory components."""
    
    @abstractmethod
    async def add_episode(self, episode: Dict[str, Any]) -> str:
        """Add an episode to memory.
        
        Args:
            episode: The episode to add
            
        Returns:
            The ID of the added episode
        """
        pass
    
    @abstractmethod
    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get an episode by ID.
        
        Args:
            episode_id: The ID of the episode
            
        Returns:
            The episode, or None if not found
        """
        pass
    
    @abstractmethod
    async def get_episodes(
        self, 
        filter: Optional[Dict[str, Any]] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get episodes that match a filter.
        
        Args:
            filter: The filter to match
            limit: Maximum number of episodes to return
            
        Returns:
            A list of matching episodes
        """
        pass


class ISemanticMemory(IMemory):
    """Interface for semantic memory components."""
    
    @abstractmethod
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """Add knowledge to memory.
        
        Args:
            knowledge: The knowledge to add
            
        Returns:
            The ID of the added knowledge
        """
        pass
    
    @abstractmethod
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID.
        
        Args:
            knowledge_id: The ID of the knowledge
            
        Returns:
            The knowledge, or None if not found
        """
        pass
    
    @abstractmethod
    async def search_knowledge(
        self, 
        query: str, 
        filter: Optional[Dict[str, Any]] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for knowledge.
        
        Args:
            query: The search query
            filter: Optional filter criteria
            limit: Maximum number of results to return
            
        Returns:
            A list of matching knowledge items
        """
        pass
    
    @abstractmethod
    async def search_knowledge_hybrid(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search for knowledge.
        
        Args:
            query: The search query
            filter: Optional filter criteria
            limit: Maximum number of results to return
            alpha: Weight between vector (alpha) and text (1-alpha) search
            
        Returns:
            A list of matching knowledge items
        """
        pass


class IProceduralMemory(IMemory):
    """Interface for procedural memory components."""
    
    @abstractmethod
    async def add_procedure(self, procedure: Dict[str, Any]) -> str:
        """Add a procedure to memory.
        
        Args:
            procedure: The procedure to add
            
        Returns:
            The ID of the added procedure
        """
        pass
    
    @abstractmethod
    async def get_procedure(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Get a procedure by ID.
        
        Args:
            procedure_id: The ID of the procedure
            
        Returns:
            The procedure, or None if not found
        """
        pass
    
    @abstractmethod
    async def find_procedures(
        self, 
        query: str, 
        filter: Optional[Dict[str, Any]] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find procedures that match a query.
        
        Args:
            query: The search query
            filter: Optional filter criteria
            limit: Maximum number of results to return
            
        Returns:
            A list of matching procedures
        """
        pass
    
    @abstractmethod
    async def execute_procedure(
        self, 
        procedure_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a procedure.
        
        Args:
            procedure_id: The ID of the procedure to execute
            context: The execution context
            
        Returns:
            The result of the procedure execution
        """
        pass


class IUnifiedMemory(IMemory):
    """Interface for unified memory components."""
    
    @abstractmethod
    async def add(self, item: Dict[str, Any], memory_type: Optional[str] = None) -> None:
        """Add an item to memory.
        
        Args:
            item: The item to add to memory
            memory_type: The type of memory to add to, or None to auto-detect
        """
        pass
    
    @abstractmethod
    async def get(
        self, 
        query: Dict[str, Any], 
        memory_type: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get items from memory that match a query.
        
        Args:
            query: The query to match
            memory_type: The type of memory to search, or None to search all
            limit: Maximum number of items to return
            
        Returns:
            A list of matching items
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        memory_type: Optional[str] = None, 
        filter: Optional[Dict[str, Any]] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for items in memory.
        
        Args:
            query: The search query
            memory_type: The type of memory to search, or None to search all
            filter: Optional filter criteria
            limit: Maximum number of results to return
            
        Returns:
            A list of matching items
        """
        pass
    
    @abstractmethod
    async def consolidate_memories(self) -> None:
        """Consolidate memories across all memory subsystems."""
        pass
