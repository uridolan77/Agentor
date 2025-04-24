"""
Adapters for existing memory components to use the new standardized interfaces.

This module provides adapter classes that wrap existing memory implementations
to make them compatible with the new standardized interfaces.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging

from agentor.components.memory import (
    Memory, SimpleMemory, VectorMemory,
    EpisodicMemory, SemanticMemory, ProceduralMemory, UnifiedMemory
)
from agentor.core.interfaces.memory import (
    IMemory, IEpisodicMemory, ISemanticMemory, IProceduralMemory, IUnifiedMemory
)
from agentor.core.plugin import Plugin
from agentor.core.registry import get_component_registry

logger = logging.getLogger(__name__)


class MemoryAdapter(IMemory):
    """Adapter for the base Memory class."""
    
    def __init__(self, memory: Memory):
        """Initialize the memory adapter.
        
        Args:
            memory: The memory implementation to adapt
        """
        self.memory = memory
    
    async def add(self, item: Dict[str, Any]) -> None:
        """Add an item to memory.
        
        Args:
            item: The item to add to memory
        """
        await self.memory.add(item)
    
    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get items from memory that match a query.
        
        Args:
            query: The query to match
            limit: Maximum number of items to return
            
        Returns:
            A list of matching items
        """
        return await self.memory.get(query, limit)
    
    async def clear(self) -> None:
        """Clear all items from memory."""
        await self.memory.clear()


class SimpleMemoryAdapter(MemoryAdapter):
    """Adapter for the SimpleMemory class."""
    
    def __init__(self, memory: Optional[SimpleMemory] = None):
        """Initialize the simple memory adapter.
        
        Args:
            memory: The simple memory implementation to adapt, or None to create a new one
        """
        super().__init__(memory or SimpleMemory())


class VectorMemoryAdapter(MemoryAdapter):
    """Adapter for the VectorMemory class."""
    
    def __init__(self, memory: Optional[VectorMemory] = None):
        """Initialize the vector memory adapter.
        
        Args:
            memory: The vector memory implementation to adapt, or None to create a new one
        """
        super().__init__(memory or VectorMemory())


class EpisodicMemoryAdapter(MemoryAdapter, IEpisodicMemory):
    """Adapter for the EpisodicMemory class."""
    
    def __init__(self, memory: Optional[EpisodicMemory] = None):
        """Initialize the episodic memory adapter.
        
        Args:
            memory: The episodic memory implementation to adapt, or None to create a new one
        """
        super().__init__(memory or EpisodicMemory())
        self.episodic_memory = self.memory
    
    async def add_episode(self, episode: Dict[str, Any]) -> str:
        """Add an episode to memory.
        
        Args:
            episode: The episode to add
            
        Returns:
            The ID of the added episode
        """
        return await self.episodic_memory.add_episode(episode)
    
    async def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Get an episode by ID.
        
        Args:
            episode_id: The ID of the episode
            
        Returns:
            The episode, or None if not found
        """
        return await self.episodic_memory.get_episode(episode_id)
    
    async def get_episodes_by_time(
        self, 
        start_time: Optional[float] = None, 
        end_time: Optional[float] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get episodes by time range.
        
        Args:
            start_time: The start time (Unix timestamp), or None for no lower bound
            end_time: The end time (Unix timestamp), or None for no upper bound
            limit: Maximum number of episodes to return
            
        Returns:
            A list of episodes in the time range
        """
        return await self.episodic_memory.get_episodes_by_time(start_time, end_time, limit)
    
    async def search_episodes(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for episodes.
        
        Args:
            query: The search query
            limit: Maximum number of episodes to return
            
        Returns:
            A list of matching episodes
        """
        return await self.episodic_memory.search(query, limit)


class SemanticMemoryAdapter(MemoryAdapter, ISemanticMemory):
    """Adapter for the SemanticMemory class."""
    
    def __init__(self, memory: Optional[SemanticMemory] = None):
        """Initialize the semantic memory adapter.
        
        Args:
            memory: The semantic memory implementation to adapt, or None to create a new one
        """
        super().__init__(memory or SemanticMemory())
        self.semantic_memory = self.memory
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """Add knowledge to memory.
        
        Args:
            knowledge: The knowledge to add
            
        Returns:
            The ID of the added knowledge
        """
        return await self.semantic_memory.add_knowledge(knowledge)
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID.
        
        Args:
            knowledge_id: The ID of the knowledge
            
        Returns:
            The knowledge, or None if not found
        """
        return await self.semantic_memory.get_knowledge(knowledge_id)
    
    async def search_knowledge(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for knowledge.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of matching knowledge items
        """
        return await self.semantic_memory.search(query, limit)
    
    async def update_knowledge(
        self, 
        knowledge_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update knowledge.
        
        Args:
            knowledge_id: The ID of the knowledge to update
            updates: The updates to apply
            
        Returns:
            The updated knowledge, or None if not found
        """
        return await self.semantic_memory.update_knowledge(knowledge_id, updates)


class ProceduralMemoryAdapter(MemoryAdapter, IProceduralMemory):
    """Adapter for the ProceduralMemory class."""
    
    def __init__(self, memory: Optional[ProceduralMemory] = None):
        """Initialize the procedural memory adapter.
        
        Args:
            memory: The procedural memory implementation to adapt, or None to create a new one
        """
        super().__init__(memory or ProceduralMemory())
        self.procedural_memory = self.memory
    
    async def add_procedure(self, procedure: Dict[str, Any]) -> str:
        """Add a procedure to memory.
        
        Args:
            procedure: The procedure to add
            
        Returns:
            The ID of the added procedure
        """
        return await self.procedural_memory.add_procedure(procedure)
    
    async def get_procedure(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Get a procedure by ID.
        
        Args:
            procedure_id: The ID of the procedure
            
        Returns:
            The procedure, or None if not found
        """
        return await self.procedural_memory.get_procedure(procedure_id)
    
    async def search_procedures(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for procedures.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of matching procedures
        """
        return await self.procedural_memory.search(query, limit)
    
    async def execute_procedure(
        self, 
        procedure_id: str, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a procedure.
        
        Args:
            procedure_id: The ID of the procedure to execute
            inputs: The inputs for the procedure
            
        Returns:
            The result of executing the procedure
        """
        return await self.procedural_memory.execute_procedure(procedure_id, inputs)


class UnifiedMemoryAdapter(MemoryAdapter, IUnifiedMemory):
    """Adapter for the UnifiedMemory class."""
    
    def __init__(self, memory: Optional[UnifiedMemory] = None):
        """Initialize the unified memory adapter.
        
        Args:
            memory: The unified memory implementation to adapt, or None to create a new one
        """
        super().__init__(memory or UnifiedMemory())
        self.unified_memory = self.memory
    
    async def add(self, item: Dict[str, Any], memory_type: Optional[str] = None) -> None:
        """Add an item to memory.
        
        Args:
            item: The item to add to memory
            memory_type: The type of memory to add to, or None to auto-detect
        """
        await self.unified_memory.add(item, memory_type)
    
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
        return await self.unified_memory.get(query, memory_type, limit)
    
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
        return await self.unified_memory.search(query, memory_type, filter, limit)
    
    async def consolidate_memories(self) -> None:
        """Consolidate memories across all memory subsystems."""
        await self.unified_memory.consolidate_memories()


# Memory plugins

class SimpleMemoryPlugin(Plugin):
    """Plugin for the SimpleMemory class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "simple_memory"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Simple in-memory storage for agent memory"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Create the memory adapter
        memory_adapter = SimpleMemoryAdapter()
        
        # Register the memory provider
        component_registry = get_component_registry()
        component_registry.register_memory_provider("simple", memory_adapter)
        
        logger.info("Registered simple memory provider")


class VectorMemoryPlugin(Plugin):
    """Plugin for the VectorMemory class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "vector_memory"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Vector-based memory storage for semantic search"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Create the memory adapter
        memory_adapter = VectorMemoryAdapter()
        
        # Register the memory provider
        component_registry = get_component_registry()
        component_registry.register_memory_provider("vector", memory_adapter)
        
        logger.info("Registered vector memory provider")


class EpisodicMemoryPlugin(Plugin):
    """Plugin for the EpisodicMemory class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "episodic_memory"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Episodic memory for storing sequences of events"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Create the memory adapter
        memory_adapter = EpisodicMemoryAdapter()
        
        # Register the memory provider
        component_registry = get_component_registry()
        component_registry.register_memory_provider("episodic", memory_adapter)
        
        logger.info("Registered episodic memory provider")


class SemanticMemoryPlugin(Plugin):
    """Plugin for the SemanticMemory class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "semantic_memory"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Semantic memory for storing knowledge and facts"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Create the memory adapter
        memory_adapter = SemanticMemoryAdapter()
        
        # Register the memory provider
        component_registry = get_component_registry()
        component_registry.register_memory_provider("semantic", memory_adapter)
        
        logger.info("Registered semantic memory provider")


class ProceduralMemoryPlugin(Plugin):
    """Plugin for the ProceduralMemory class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "procedural_memory"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Procedural memory for storing learned behaviors and skills"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Create the memory adapter
        memory_adapter = ProceduralMemoryAdapter()
        
        # Register the memory provider
        component_registry = get_component_registry()
        component_registry.register_memory_provider("procedural", memory_adapter)
        
        logger.info("Registered procedural memory provider")


class UnifiedMemoryPlugin(Plugin):
    """Plugin for the UnifiedMemory class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "unified_memory"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Unified memory system combining episodic, semantic, and procedural memory"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Create the memory adapter
        memory_adapter = UnifiedMemoryAdapter()
        
        # Register the memory provider
        component_registry = get_component_registry()
        component_registry.register_memory_provider("unified", memory_adapter)
        
        logger.info("Registered unified memory provider")
