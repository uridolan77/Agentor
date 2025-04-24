"""
Temporal Memory for the Agentor framework.

This module provides temporal memory capabilities that enhance memory management
with time-based retrieval, decay, and importance scoring mechanisms.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import heapq

from agentor.components.memory.base import Memory
from agentor.components.memory.semantic_memory import KnowledgeNode

logger = logging.getLogger(__name__)

class TemporalMemoryNode(KnowledgeNode):
    """Extended knowledge node with temporal properties."""
    
    def __init__(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        decay_rate: float = 0.05,
        last_accessed: Optional[datetime] = None
    ):
        super().__init__(content, importance, metadata)
        self.created_at = datetime.now()
        self.last_accessed = last_accessed or self.created_at
        self.access_count = 0
        self.decay_rate = decay_rate
        
    def access(self) -> None:
        """Record an access to this memory node."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def get_current_importance(self) -> float:
        """Calculate importance with temporal decay applied."""
        days_since_access = (datetime.now() - self.last_accessed).days
        decay_factor = 1.0 - (self.decay_rate * days_since_access)
        decay_factor = max(0.1, decay_factor)  # Minimum decay
        
        # Apply frequency boost
        frequency_boost = min(0.5, self.access_count * 0.05)
        
        return min(1.0, self.importance * decay_factor + frequency_boost)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        data = super().to_dict()
        data.update({
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "decay_rate": self.decay_rate
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalMemoryNode":
        """Create node from dictionary."""
        node = cls(
            content=data["content"],
            importance=data["importance"],
            metadata=data.get("metadata", {})
        )
        
        if "created_at" in data:
            node.created_at = datetime.fromisoformat(data["created_at"])
        if "last_accessed" in data:
            node.last_accessed = datetime.fromisoformat(data["last_accessed"])
        if "access_count" in data:
            node.access_count = data["access_count"]
        if "decay_rate" in data:
            node.decay_rate = data["decay_rate"]
            
        return node


class TemporalMemory(Memory):
    """Memory implementation with time-based retrieval and decay mechanisms."""
    
    def __init__(
        self,
        max_nodes: int = 1000,
        default_decay_rate: float = 0.05,
        retention_period: int = 90,  # days
        forgetting_threshold: float = 0.2
    ):
        """Initialize temporal memory.
        
        Args:
            max_nodes: Maximum number of memory nodes to store
            default_decay_rate: Default rate at which memories decay over time
            retention_period: Maximum number of days to retain memories
            forgetting_threshold: Importance threshold below which memories are forgotten
        """
        self.nodes: List[TemporalMemoryNode] = []
        self.max_nodes = max_nodes
        self.default_decay_rate = default_decay_rate
        self.retention_period = retention_period  # days
        self.forgetting_threshold = forgetting_threshold
        
    async def add(self, content: str, importance: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a memory node.
        
        Args:
            content: Content of the memory
            importance: Initial importance score
            metadata: Additional metadata
            
        Returns:
            ID of the created memory node
        """
        # Create node
        node = TemporalMemoryNode(
            content=content,
            importance=importance,
            metadata=metadata or {},
            decay_rate=self.default_decay_rate
        )
        
        # Add to storage
        self.nodes.append(node)
        
        # Check capacity and remove least important if needed
        if len(self.nodes) > self.max_nodes:
            await self._forget_least_important()
            
        return node.id
    
    async def get(self, node_id: str) -> Optional[TemporalMemoryNode]:
        """Get a memory node by ID.
        
        Args:
            node_id: The ID of the node to retrieve
            
        Returns:
            The memory node if found, None otherwise
        """
        for node in self.nodes:
            if node.id == node_id:
                node.access()
                return node
        return None
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        min_importance: float = 0.0
    ) -> List[TemporalMemoryNode]:
        """Search for memory nodes.
        
        Args:
            query: Search query
            limit: Maximum number of results
            time_range: Optional tuple of (start_time, end_time) for filtering
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memory nodes
        """
        # Filter by time range if provided
        candidates = self.nodes
        if time_range:
            start_time, end_time = time_range
            candidates = [
                node for node in candidates 
                if start_time <= node.created_at <= end_time
            ]
        
        # Filter by minimum importance (using decayed importance)
        candidates = [
            node for node in candidates 
            if node.get_current_importance() >= min_importance
        ]
        
        # Simple text matching for now (could be enhanced with embeddings)
        # Score based on text relevance and current importance
        scored_nodes = []
        for node in candidates:
            # Simple text match score
            text_match = query.lower() in node.content.lower()
            match_score = 0.8 if text_match else 0.1
            
            # Combine with importance
            current_importance = node.get_current_importance()
            final_score = (match_score * 0.7) + (current_importance * 0.3)
            
            scored_nodes.append((final_score, node))
            
        # Sort by score and take top results
        scored_nodes.sort(reverse=True)
        results = [node for _, node in scored_nodes[:limit]]
        
        # Mark nodes as accessed
        for node in results:
            node.access()
            
        return results
    
    async def update(self, node_id: str, **updates) -> bool:
        """Update a memory node.
        
        Args:
            node_id: ID of the node to update
            **updates: Fields to update
            
        Returns:
            True if successfully updated, False otherwise
        """
        node = await self.get(node_id)
        if not node:
            return False
            
        # Update fields
        if "content" in updates:
            node.content = updates["content"]
        if "importance" in updates:
            node.importance = updates["importance"]
        if "metadata" in updates and updates["metadata"]:
            node.metadata.update(updates["metadata"])
        if "decay_rate" in updates:
            node.decay_rate = updates["decay_rate"]
            
        return True
    
    async def delete(self, node_id: str) -> bool:
        """Delete a memory node.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                self.nodes.pop(i)
                return True
        return False
    
    async def clear(self) -> None:
        """Clear all memory nodes."""
        self.nodes = []
        
    async def _forget_least_important(self) -> None:
        """Forget the least important memories based on current importance."""
        if not self.nodes:
            logger.debug("No nodes to forget")
            return
            
        # Calculate current importance for all nodes
        nodes_with_importance = [
            (node.get_current_importance(), node) 
            for node in self.nodes
        ]
        
        # Sort by importance (ascending)
        nodes_with_importance.sort()
        
        # Make sure we have nodes to forget
        if not nodes_with_importance:
            logger.warning("No nodes with importance found")
            return
            
        # Remove lowest importance node
        _, node_to_forget = nodes_with_importance[0]
        logger.debug(f"Forgetting node with ID {node_to_forget.id} (importance: {node_to_forget.get_current_importance()})")
        await self.delete(node_to_forget.id)
        
    async def run_maintenance(self) -> int:
        """Run maintenance tasks like forgetting old or unimportant memories.
        
        Returns:
            Number of memories forgotten
        """
        forgotten_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.retention_period)
        
        # Identify nodes to forget
        nodes_to_forget = []
        for node in self.nodes:
            current_importance = node.get_current_importance()
            
            # Forget if below threshold or too old
            if (current_importance < self.forgetting_threshold or 
                node.created_at < cutoff_date):
                nodes_to_forget.append(node.id)
                
        # Delete identified nodes
        for node_id in nodes_to_forget:
            success = await self.delete(node_id)
            if success:
                forgotten_count += 1
                
        return forgotten_count