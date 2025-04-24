"""
Memory Forgetting Mechanisms for the Agentor framework.

This module provides various forgetting mechanisms to prevent memory bloat and
prioritize important memories over less important ones.
"""

from typing import Dict, Any, List, Optional, Tuple, Protocol, Callable, TypeVar
import time
import math
import random
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryItem(Protocol):
    """Protocol for items that can be forgotten."""
    
    importance: float
    """Importance score for this item (0.0 to 1.0)."""
    
    created_at: float
    """When the item was created."""
    
    updated_at: float
    """When the item was last updated."""


class ForgettingMechanism(ABC):
    """Base class for forgetting mechanisms."""
    
    @abstractmethod
    async def should_forget(self, item: MemoryItem) -> bool:
        """Determine if an item should be forgotten.
        
        Args:
            item: The memory item to evaluate
            
        Returns:
            True if the item should be forgotten, False otherwise
        """
        pass
    
    @abstractmethod
    async def prioritize(self, items: List[T]) -> List[T]:
        """Prioritize items based on forgetting criteria.
        
        Args:
            items: The items to prioritize
            
        Returns:
            The items sorted by priority (most important first)
        """
        pass


class ThresholdForgetting(ForgettingMechanism):
    """Simple threshold-based forgetting mechanism.
    
    Forgets items with importance below a threshold.
    """
    
    def __init__(self, threshold: float = 0.2):
        """Initialize the threshold forgetting mechanism.
        
        Args:
            threshold: The importance threshold below which items are forgotten
        """
        self.threshold = threshold
    
    async def should_forget(self, item: MemoryItem) -> bool:
        """Determine if an item should be forgotten based on importance threshold.
        
        Args:
            item: The memory item to evaluate
            
        Returns:
            True if the item's importance is below the threshold
        """
        return item.importance < self.threshold
    
    async def prioritize(self, items: List[T]) -> List[T]:
        """Prioritize items based on importance.
        
        Args:
            items: The items to prioritize
            
        Returns:
            The items sorted by importance (most important first)
        """
        return sorted(items, key=lambda x: getattr(x, 'importance', 0.0), reverse=True)


class TimeForgetting(ForgettingMechanism):
    """Time-based forgetting mechanism.
    
    Forgets items based on their age, with older items more likely to be forgotten.
    """
    
    def __init__(self, half_life: float = 604800.0):  # Default: 1 week in seconds
        """Initialize the time forgetting mechanism.
        
        Args:
            half_life: The half-life of memories in seconds
        """
        self.half_life = half_life
    
    async def should_forget(self, item: MemoryItem) -> bool:
        """Determine if an item should be forgotten based on age and importance.
        
        Args:
            item: The memory item to evaluate
            
        Returns:
            True if the item should be forgotten based on age
        """
        age = time.time() - item.updated_at
        
        # Calculate decay factor based on age
        decay_factor = math.exp(-age / self.half_life)
        
        # Adjust by importance (more important items decay slower)
        adjusted_decay = decay_factor * (0.5 + 0.5 * item.importance)
        
        # Random chance of forgetting based on decay
        return random.random() > adjusted_decay
    
    async def prioritize(self, items: List[T]) -> List[T]:
        """Prioritize items based on recency and importance.
        
        Args:
            items: The items to prioritize
            
        Returns:
            The items sorted by a combination of recency and importance
        """
        now = time.time()
        
        def priority_score(item):
            age = now - getattr(item, 'updated_at', now)
            importance = getattr(item, 'importance', 0.5)
            
            # Calculate decay factor based on age
            decay_factor = math.exp(-age / self.half_life)
            
            # Combine with importance
            return decay_factor * (0.5 + 0.5 * importance)
        
        return sorted(items, key=priority_score, reverse=True)


class EbbinghausForgetting(ForgettingMechanism):
    """Ebbinghaus forgetting curve mechanism.
    
    Models memory retention based on the Ebbinghaus forgetting curve,
    which describes how memory retention decreases over time.
    """
    
    def __init__(self, strength: float = 0.5, stability: float = 0.5):
        """Initialize the Ebbinghaus forgetting mechanism.
        
        Args:
            strength: The initial strength of memories (0.0 to 1.0)
            stability: How quickly memories stabilize with repetition (0.0 to 1.0)
        """
        self.strength = strength
        self.stability = stability
    
    async def should_forget(self, item: MemoryItem) -> bool:
        """Determine if an item should be forgotten based on the forgetting curve.
        
        Args:
            item: The memory item to evaluate
            
        Returns:
            True if the item should be forgotten
        """
        age = time.time() - item.updated_at
        
        # Calculate retention based on the forgetting curve
        # R = e^(-t/S) where R is retention, t is time, and S is stability
        stability = self.stability * (1.0 + item.importance)
        retention = math.exp(-age / (stability * 86400))  # Convert stability to seconds
        
        # Adjust by importance
        adjusted_retention = retention * (0.5 + 0.5 * item.importance)
        
        # Random chance of forgetting based on retention
        return random.random() > adjusted_retention
    
    async def prioritize(self, items: List[T]) -> List[T]:
        """Prioritize items based on the forgetting curve.
        
        Args:
            items: The items to prioritize
            
        Returns:
            The items sorted by retention (highest retention first)
        """
        now = time.time()
        
        def retention_score(item):
            age = now - getattr(item, 'updated_at', now)
            importance = getattr(item, 'importance', 0.5)
            
            # Calculate stability based on importance
            stability = self.stability * (1.0 + importance)
            
            # Calculate retention
            retention = math.exp(-age / (stability * 86400))
            
            # Adjust by importance
            return retention * (0.5 + 0.5 * importance)
        
        return sorted(items, key=retention_score, reverse=True)


class CompositeForgetting(ForgettingMechanism):
    """Composite forgetting mechanism that combines multiple mechanisms.
    
    Allows for combining different forgetting strategies with weights.
    """
    
    def __init__(self, mechanisms: List[Tuple[ForgettingMechanism, float]]):
        """Initialize the composite forgetting mechanism.
        
        Args:
            mechanisms: List of (mechanism, weight) tuples
        """
        self.mechanisms = mechanisms
        
        # Normalize weights
        total_weight = sum(weight for _, weight in mechanisms)
        self.normalized_weights = [weight / total_weight for _, weight in mechanisms]
    
    async def should_forget(self, item: MemoryItem) -> bool:
        """Determine if an item should be forgotten based on weighted votes.
        
        Args:
            item: The memory item to evaluate
            
        Returns:
            True if the weighted vote exceeds 0.5
        """
        votes = []
        
        # Get votes from all mechanisms
        for (mechanism, _), weight in zip(self.mechanisms, self.normalized_weights):
            should_forget = await mechanism.should_forget(item)
            votes.append((1.0 if should_forget else 0.0) * weight)
        
        # Calculate weighted vote
        weighted_vote = sum(votes)
        
        return weighted_vote > 0.5
    
    async def prioritize(self, items: List[T]) -> List[T]:
        """Prioritize items based on weighted scores from all mechanisms.
        
        Args:
            items: The items to prioritize
            
        Returns:
            The items sorted by combined priority score
        """
        # Get prioritized lists from all mechanisms
        prioritized_lists = []
        for mechanism, _ in self.mechanisms:
            prioritized = await mechanism.prioritize(items.copy())
            prioritized_lists.append(prioritized)
        
        # Calculate scores based on position in each list
        scores = {id(item): 0.0 for item in items}
        
        for prioritized, weight in zip(prioritized_lists, self.normalized_weights):
            for i, item in enumerate(prioritized):
                # Score is based on position (higher position = higher score)
                position_score = 1.0 - (i / len(prioritized))
                scores[id(item)] += position_score * weight
        
        # Sort by combined score
        return sorted(items, key=lambda x: scores[id(x)], reverse=True)


@dataclass
class ForgettingConfig:
    """Configuration for memory forgetting."""
    
    mechanism: ForgettingMechanism
    """The forgetting mechanism to use."""
    
    check_interval: int = 3600
    """How often to check for items to forget (in seconds)."""
    
    max_items: Optional[int] = None
    """Maximum number of items to keep (None for unlimited)."""


class ForgettingManager:
    """Manager for applying forgetting mechanisms to memory systems."""
    
    def __init__(self, config: ForgettingConfig):
        """Initialize the forgetting manager.
        
        Args:
            config: The forgetting configuration
        """
        self.config = config
        self.last_check = time.time()
        self.lock = asyncio.Lock()
    
    async def apply_forgetting(self, items: Dict[str, MemoryItem]) -> Dict[str, MemoryItem]:
        """Apply forgetting to a collection of items.
        
        Args:
            items: Dictionary of items to apply forgetting to
            
        Returns:
            Dictionary of remaining items after forgetting
        """
        async with self.lock:
            now = time.time()
            
            # Only check if enough time has passed
            if now - self.last_check < self.config.check_interval:
                return items
            
            self.last_check = now
            
            # Convert to list for processing
            item_list = list(items.values())
            
            # Apply forgetting mechanism to each item
            remaining_items = {}
            for item_id, item in items.items():
                should_forget = await self.config.mechanism.should_forget(item)
                
                if not should_forget:
                    remaining_items[item_id] = item
            
            # If we still have too many items, prioritize and keep only the top ones
            if self.config.max_items and len(remaining_items) > self.config.max_items:
                prioritized = await self.config.mechanism.prioritize(list(remaining_items.values()))
                
                # Keep only the top items
                top_items = prioritized[:self.config.max_items]
                
                # Convert back to dictionary
                remaining_items = {
                    item_id: item
                    for item_id, item in items.items()
                    if item in top_items
                }
            
            # Log forgetting statistics
            forgotten_count = len(items) - len(remaining_items)
            if forgotten_count > 0:
                logger.info(f"Forgot {forgotten_count} items ({len(remaining_items)} remaining)")
            
            return remaining_items
