"""
Working Memory for the Agentor framework.

This module provides working memory capabilities that simulate human short-term
memory with capacity constraints, recency effects, and cognitive load management.
"""

from typing import Dict, Any, List, Optional, Tuple, Deque
import logging
from datetime import datetime
from collections import deque
import uuid
import json

from agentor.components.memory.base import Memory

logger = logging.getLogger(__name__)

class WorkingMemoryItem:
    """An item in working memory."""
    
    def __init__(
        self,
        content: Any,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general"
    ):
        """Initialize a working memory item.
        
        Args:
            content: The content to store
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            category: Category of the memory item
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.importance = importance
        self.metadata = metadata or {}
        self.category = category
        self.created_at = datetime.now()
        self.last_accessed = self.created_at
        self.access_count = 0
        
    def access(self) -> None:
        """Record an access to this memory item."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "id": self.id,
            "content": self._serialize_content(),
            "importance": self.importance,
            "metadata": self.metadata,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
        
    def _serialize_content(self) -> Any:
        """Serialize content for storage in dictionary."""
        if isinstance(self.content, (str, int, float, bool, list, dict)) or self.content is None:
            return self.content
        
        # Try to use the object's to_dict method if available
        if hasattr(self.content, "to_dict") and callable(getattr(self.content, "to_dict")):
            return self.content.to_dict()
            
        # Fallback to string representation
        return str(self.content)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemoryItem":
        """Create from a dictionary."""
        item = cls(
            content=data["content"],
            importance=data["importance"],
            metadata=data.get("metadata", {}),
            category=data.get("category", "general")
        )
        item.id = data["id"]
        
        if "created_at" in data:
            item.created_at = datetime.fromisoformat(data["created_at"])
        if "last_accessed" in data:
            item.last_accessed = datetime.fromisoformat(data["last_accessed"])
        if "access_count" in data:
            item.access_count = data["access_count"]
            
        return item


class WorkingMemory(Memory):
    """Working memory implementation with capacity constraints.
    
    This simulates human working memory with limited capacity,
    cognitive load management, and recency effects.
    """
    
    def __init__(
        self,
        capacity: int = 7,  # Based on Miller's "magical number 7±2"
        max_cognitive_load: float = 10.0,
        categories: Optional[List[str]] = None
    ):
        """Initialize working memory.
        
        Args:
            capacity: Maximum number of items to store
            max_cognitive_load: Maximum cognitive load (sum of importance)
            categories: Optional list of memory categories
        """
        self.capacity = capacity
        self.max_cognitive_load = max_cognitive_load
        self.categories = categories or ["general", "task", "context", "goal"]
        
        # Memory items by category
        self.items: Dict[str, List[WorkingMemoryItem]] = {
            category: [] for category in self.categories
        }
        
        # Recent access queue for LRU-like behavior
        self.recent_access: Deque[str] = deque(maxlen=capacity * 2)
        
    async def add(
        self,
        content: Any,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        category: str = "general"
    ) -> str:
        """Add an item to working memory.
        
        Args:
            content: The content to store
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            category: Category of the memory item
            
        Returns:
            ID of the created memory item
        """
        # Ensure valid category
        if category not in self.categories:
            category = "general"
            
        # Create item
        item = WorkingMemoryItem(
            content=content,
            importance=importance,
            metadata=metadata or {},
            category=category
        )
        
        # Add to category
        self.items[category].append(item)
        
        # Update recent access
        self.recent_access.append(item.id)
        
        # Check capacity and cognitive load
        await self._manage_constraints()
        
        return item.id
    
    async def get(self, item_id: str) -> Optional[WorkingMemoryItem]:
        """Get an item by ID.
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        for category, items in self.items.items():
            for item in items:
                if item.id == item_id:
                    item.access()
                    self.recent_access.append(item.id)
                    return item
        return None
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[WorkingMemoryItem]:
        """Search for items.
        
        Args:
            query: Search query
            limit: Maximum number of results
            category: Optional category filter
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memory items
        """
        results = []
        
        # Determine categories to search in
        categories = [category] if category in self.categories else self.categories
        
        # Search each category
        for cat in categories:
            for item in self.items[cat]:
                if item.importance < min_importance:
                    continue
                    
                # Simple text matching
                content_str = str(item.content)
                if query.lower() in content_str.lower():
                    results.append(item)
                    # Mark as accessed
                    item.access()
                    self.recent_access.append(item.id)
        
        # Sort by recency (using recent_access queue) and take top results
        sorted_results = sorted(
            results,
            key=lambda x: -list(self.recent_access).index(x.id) if x.id in self.recent_access else float('-inf')
        )
        
        return sorted_results[:limit]
    
    async def update(
        self,
        item_id: str,
        content: Optional[Any] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None
    ) -> bool:
        """Update an item.
        
        Args:
            item_id: ID of the item to update
            content: New content (if None, keeps existing)
            importance: New importance score (if None, keeps existing)
            metadata: Metadata to update (if None, keeps existing)
            category: New category (if None, keeps existing)
            
        Returns:
            True if successfully updated, False otherwise
        """
        item = await self.get(item_id)
        if not item:
            return False
            
        # Find item's current category
        current_category = None
        for cat, items in self.items.items():
            for i, mem_item in enumerate(items):
                if mem_item.id == item_id:
                    current_category = cat
                    break
            if current_category:
                break
                
        if not current_category:
            return False
        
        # Update fields
        if content is not None:
            item.content = content
        if importance is not None:
            item.importance = importance
        if metadata is not None:
            item.metadata.update(metadata)
            
        # Handle category change
        if category is not None and category != current_category and category in self.categories:
            # Remove from old category
            self.items[current_category] = [
                i for i in self.items[current_category] if i.id != item_id
            ]
            # Add to new category
            self.items[category].append(item)
            
        # Mark as accessed
        item.access()
        self.recent_access.append(item_id)
        
        # Check constraints
        await self._manage_constraints()
            
        return True
    
    async def delete(self, item_id: str) -> bool:
        """Delete an item.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        for category in self.categories:
            for i, item in enumerate(self.items[category]):
                if item.id == item_id:
                    self.items[category].pop(i)
                    # Remove from recent access queue
                    self.recent_access = deque(
                        [i for i in self.recent_access if i != item_id],
                        maxlen=self.recent_access.maxlen
                    )
                    return True
        return False
    
    async def clear(self, category: Optional[str] = None) -> None:
        """Clear items.
        
        Args:
            category: Optional category to clear (if None, clears all)
        """
        if category is None:
            # Clear all categories
            for cat in self.categories:
                self.items[cat] = []
            self.recent_access.clear()
        elif category in self.categories:
            # Clear specific category
            item_ids = [item.id for item in self.items[category]]
            self.items[category] = []
            # Update recent access queue
            self.recent_access = deque(
                [i for i in self.recent_access if i not in item_ids],
                maxlen=self.recent_access.maxlen
            )
    
    def get_cognitive_load(self) -> float:
        """Calculate current cognitive load.
        
        Returns:
            Current cognitive load as sum of item importances
        """
        load = 0.0
        for category in self.categories:
            for item in self.items[category]:
                load += item.importance
        return load
    
    def get_category_load(self, category: str) -> float:
        """Calculate cognitive load for a specific category.
        
        Args:
            category: Category to calculate load for
            
        Returns:
            Cognitive load for the category
        """
        if category not in self.categories:
            return 0.0
            
        return sum(item.importance for item in self.items[category])
    
    def get_total_items(self) -> int:
        """Get total number of items in working memory.
        
        Returns:
            Total number of items
        """
        return sum(len(items) for items in self.items.values())
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get a summary of working memory state.
        
        Returns:
            Dictionary with memory summary
        """
        total_items = self.get_total_items()
        cognitive_load = self.get_cognitive_load()
        
        category_stats = {}
        for category in self.categories:
            items_count = len(self.items[category])
            category_load = self.get_category_load(category)
            category_stats[category] = {
                "items_count": items_count,
                "cognitive_load": category_load
            }
            
        return {
            "total_items": total_items,
            "capacity": self.capacity,
            "cognitive_load": cognitive_load,
            "max_cognitive_load": self.max_cognitive_load,
            "categories": category_stats
        }
        
    async def _manage_constraints(self) -> None:
        """Manage capacity and cognitive load constraints."""
        # Check total items against capacity
        total_items = self.get_total_items()
        
        # Calculate cognitive load
        cognitive_load = self.get_cognitive_load()
        
        # If we're within limits, nothing to do
        if total_items <= self.capacity and cognitive_load <= self.max_cognitive_load:
            return
            
        # We need to forget some items to stay within constraints
        # Get all items across categories with their importance
        all_items = []
        for category in self.categories:
            for item in self.items[category]:
                # Score is based on:
                # 1. Importance (higher is better)
                # 2. Recency (more recent is better)
                # 3. Access count (more access is better)
                
                # Recency factor: position in recent_access queue (normalized to 0-1)
                recency = 0.0
                if item.id in self.recent_access:
                    idx = list(self.recent_access).index(item.id)
                    recency = 1.0 - (idx / len(self.recent_access))
                
                # Access count factor (capped at 10)
                access_factor = min(1.0, item.access_count / 10.0)
                
                # Combined score - higher means MORE likely to be kept
                score = (item.importance * 0.5) + (recency * 0.3) + (access_factor * 0.2)
                
                all_items.append((score, item, category))
        
        # Sort by score (ascending, so least important items are first)
        all_items.sort()
        
        # Remove items until we're within constraints
        while all_items and (self.get_total_items() > self.capacity or 
                            self.get_cognitive_load() > self.max_cognitive_load):
            # Remove lowest scored item
            _, item_to_forget, category = all_items.pop(0)
            await self.delete(item_to_forget.id)