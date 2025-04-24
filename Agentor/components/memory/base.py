"""
Base memory classes for the Agentor framework.

This module provides the base memory classes used by other memory implementations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import json
import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Memory(ABC):
    """Base class for agent memory systems."""

    @abstractmethod
    async def add(self, item: Dict[str, Any]):
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
    async def clear(self):
        """Clear all items from memory."""
        pass


class SimpleMemory(Memory):
    """A simple in-memory implementation of Memory."""

    def __init__(self, max_items: int = 1000):
        """Initialize the memory.

        Args:
            max_items: The maximum number of items to store
        """
        self.items: List[Dict[str, Any]] = []
        self.max_items = max_items

    async def add(self, item: Dict[str, Any]):
        # Add timestamp if not present
        if 'timestamp' not in item:
            item['timestamp'] = time.time()

        # Add the item
        self.items.append(item)

        # Trim if necessary
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        # Simple exact matching
        results = []
        for item in self.items:
            matches = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    matches = False
                    break
            if matches:
                results.append(item)
                if len(results) >= limit:
                    break

        return results

    async def clear(self):
        self.items = []


class VectorMemory(Memory):
    """A vector-based memory implementation using embeddings."""

    def __init__(self, embedding_provider, max_items: int = 1000):
        """Initialize the memory.

        Args:
            embedding_provider: Provider for generating embeddings
            max_items: The maximum number of items to store
        """
        self.items: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.max_items = max_items
        self.embedding_provider = embedding_provider

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    async def add(self, item: Dict[str, Any]):
        # Add timestamp if not present
        if 'timestamp' not in item:
            item['timestamp'] = time.time()

        # Get the embedding for the text field
        if 'text' in item:
            embedding = await self.embedding_provider.get_embedding(item['text'])
        else:
            # If no text field, use a JSON representation of the item
            embedding = await self.embedding_provider.get_embedding(json.dumps(item))

        # Add the item and embedding
        self.items.append(item)
        self.embeddings.append(embedding)

        # Trim if necessary
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]
            self.embeddings = self.embeddings[-self.max_items:]

    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        if not self.items:
            return []

        # If the query has a text field, use it for semantic matching
        if 'text' in query:
            query_embedding = await self.embedding_provider.get_embedding(query['text'])

            # Calculate similarities
            similarities = [
                self._cosine_similarity(query_embedding, embedding)
                for embedding in self.embeddings
            ]

            # Sort items by similarity
            pairs = sorted(
                zip(similarities, self.items),
                key=lambda x: x[0],
                reverse=True
            )

            # Return the top items
            return [item for _, item in pairs[:limit]]

        # Otherwise, fall back to exact matching using SimpleMemory's implementation
        results = []
        for item in self.items:
            matches = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    matches = False
                    break
            if matches:
                results.append(item)
                if len(results) >= limit:
                    break

        return results

    async def clear(self):
        self.items = []
        self.embeddings = []
