"""
Semantic Memory implementation for the Agentor framework.

Semantic memory stores knowledge, facts, and concepts that the agent has learned.
It allows for storing and retrieving information based on meaning rather than exact matches.

This module provides the KnowledgeNode class and a SemanticMemory implementation
that uses vector databases for efficient storage and retrieval of semantic embeddings.
"""

from typing import Dict, Any, List, Optional, Set
import time
import asyncio
from dataclasses import dataclass, field
import logging

from agentor.components.memory.base import Memory
from agentor.components.memory.vector_db import InMemoryVectorDB
from agentor.llm_gateway.utils.metrics import track_memory_operation

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """A node in the semantic knowledge graph."""

    id: str
    """Unique identifier for the node."""

    content: Dict[str, Any]
    """The content of the node."""

    embedding: Optional[List[float]] = None
    """Vector embedding of the node for semantic search."""

    created_at: float = field(default_factory=time.time)
    """When the node was created."""

    updated_at: float = field(default_factory=time.time)
    """When the node was last updated."""

    importance: float = 0.5
    """Importance score for this node (0.0 to 1.0)."""

    confidence: float = 1.0
    """Confidence score for this node (0.0 to 1.0)."""

    related_nodes: Set[str] = field(default_factory=set)
    """IDs of related nodes."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary.

        Returns:
            Dictionary representation of the node
        """
        return {
            'id': self.id,
            'content': self.content,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'importance': self.importance,
            'confidence': self.confidence,
            'related_nodes': list(self.related_nodes)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create a node from a dictionary.

        Args:
            data: Dictionary representation of a node

        Returns:
            A KnowledgeNode object
        """
        node = cls(
            id=data['id'],
            content=data['content'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            importance=data['importance'],
            confidence=data['confidence']
        )
        node.related_nodes = set(data['related_nodes'])
        return node

    def update_content(self, content: Dict[str, Any]) -> None:
        """Update the content of this node.

        Args:
            content: The new content
        """
        self.content = content
        self.updated_at = time.time()
        # Embedding will need to be regenerated
        self.embedding = None


# Enhanced semantic memory implementation has been merged into this file


class SemanticMemory(Memory):
    """Semantic memory implementation that stores knowledge and facts.

    This implementation uses vector databases for efficient storage and retrieval
    of semantic embeddings, supporting hybrid search and advanced filtering.
    """

    def __init__(
        self,
        embedding_provider=None,
        vector_db=None,
        max_nodes: int = 10000,
        forgetting_threshold: float = 0.2,
        similarity_threshold: float = 0.85,
        consolidation_interval: int = 86400,  # 24 hours in seconds
    ):
        """Initialize the semantic memory.

        Args:
            embedding_provider: Provider for generating embeddings
            vector_db: Vector database provider for storing embeddings (optional)
            max_nodes: Maximum number of nodes to store
            forgetting_threshold: Importance threshold below which nodes may be forgotten
            similarity_threshold: Similarity threshold for considering nodes as duplicates
            consolidation_interval: How often to consolidate memories (in seconds)
        """
        # Create an in-memory vector database if none is provided
        if vector_db is None:
            vector_db = InMemoryVectorDB(dimension=384 if embedding_provider else 0)

        self.nodes: Dict[str, KnowledgeNode] = {}
        self.vector_db = vector_db
        self.embedding_provider = embedding_provider
        self.max_nodes = max_nodes
        self.forgetting_threshold = forgetting_threshold
        self.similarity_threshold = similarity_threshold
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = time.time()
        self.lock = asyncio.Lock()

    @track_memory_operation("add", "semantic")
    async def add(self, item: Dict[str, Any]):
        """Add a knowledge item to semantic memory.

        Args:
            item: The knowledge item to add
        """
        async with self.lock:
            # Generate a node ID if not provided
            node_id = item.get('id')
            if node_id is None:
                node_id = f"node-{int(time.time())}-{hash(str(item))}"[:16]

            # Check if this is an update to an existing node
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.update_content(item)
                logger.info(f"Updated node {node_id}")
            else:
                # Create a new node
                node = KnowledgeNode(
                    id=node_id,
                    content=item,
                    importance=item.get('importance', 0.5),
                    confidence=item.get('confidence', 1.0)
                )
                self.nodes[node_id] = node
                logger.info(f"Added new node {node_id}")

            # Generate embedding if we have a provider
            if self.embedding_provider and not node.embedding:
                # Create a text representation of the node
                node_text = self._node_to_text(node)

                # Generate the embedding
                node.embedding = await self.embedding_provider.get_embedding(node_text)

                # Store in vector database
                await self.vector_db.add(
                    id=node.id,
                    vector=node.embedding,
                    metadata={
                        'content': node.content,
                        'importance': node.importance,
                        'confidence': node.confidence,
                        'created_at': node.created_at,
                        'updated_at': node.updated_at,
                        'text': node_text
                    }
                )

                # Check for similar nodes to establish relationships
                await self._find_and_link_related_nodes(node)

            # Check if we need to consolidate memories
            if time.time() - self.last_consolidation > self.consolidation_interval:
                await self.consolidate_memories()

    @track_memory_operation("get", "semantic")
    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get knowledge nodes that match a query.

        Args:
            query: The query to match
            limit: Maximum number of nodes to return

        Returns:
            A list of matching nodes as dictionaries
        """
        results = []

        # If querying by node ID
        if 'id' in query:
            node = self.nodes.get(query['id'])
            if node:
                results.append(node.to_dict())
            return results

        # If querying by semantic similarity
        if 'text' in query and self.embedding_provider:
            query_embedding = await self.embedding_provider.get_embedding(query['text'])

            # Prepare filters
            filters = {}
            for key, value in query.items():
                if key not in ['text', 'limit', 'threshold', 'hybrid']:
                    filters[key] = value

            # Determine if we should use hybrid search
            use_hybrid = query.get('hybrid', False)

            if use_hybrid:
                # Use hybrid search (vector + text)
                vector_results = await self.vector_db.search_hybrid(
                    query_text=query['text'],
                    query_vector=query_embedding,
                    limit=limit,
                    filters=filters if filters else None,
                    alpha=query.get('alpha', 0.5)  # Default to equal weighting
                )
            else:
                # Use vector search only
                vector_results = await self.vector_db.search(
                    query_vector=query_embedding,
                    limit=limit,
                    filters=filters if filters else None
                )

            # Convert results to node dictionaries
            threshold = query.get('threshold', 0.0)
            for result in vector_results:
                if result['score'] >= threshold:
                    # Get the node from memory or create from vector DB result
                    node_id = result['id']
                    if node_id in self.nodes:
                        node_dict = self.nodes[node_id].to_dict()
                    else:
                        # Create a node dict from the vector DB result
                        metadata = result['metadata']
                        node_dict = {
                            'id': node_id,
                            'content': metadata.get('content', {}),
                            'created_at': metadata.get('created_at', 0),
                            'updated_at': metadata.get('updated_at', 0),
                            'importance': metadata.get('importance', 0.5),
                            'confidence': metadata.get('confidence', 1.0),
                            'related_nodes': []
                        }

                    # Add similarity score
                    node_dict['similarity'] = result['score']
                    results.append(node_dict)

        # If querying by content fields without semantic search
        elif any(key not in ['limit', 'hybrid', 'alpha'] for key in query.keys()):
            # Use the vector DB for filtering if possible
            if self.embedding_provider:
                filters = {}
                for key, value in query.items():
                    if key not in ['limit', 'hybrid', 'alpha']:
                        filters[f"content.{key}"] = value

                # Use a dummy vector for pure metadata filtering
                dummy_vector = [0.0] * self.vector_db.dimension
                vector_results = await self.vector_db.search(
                    query_vector=dummy_vector,
                    limit=limit,
                    filters=filters
                )

                # Convert results to node dictionaries
                for result in vector_results:
                    node_id = result['id']
                    if node_id in self.nodes:
                        results.append(self.nodes[node_id].to_dict())
            else:
                # Fall back to in-memory filtering
                for node in self.nodes.values():
                    matches = True
                    for key, value in query.items():
                        if key in ['limit', 'hybrid', 'alpha']:
                            continue

                        if key not in node.content or node.content[key] != value:
                            matches = False
                            break

                    if matches:
                        results.append(node.to_dict())
                        if len(results) >= limit:
                            break

        # Otherwise, return most recently updated nodes
        else:
            sorted_nodes = sorted(
                self.nodes.values(),
                key=lambda node: node.updated_at,
                reverse=True
            )

            for node in sorted_nodes[:limit]:
                results.append(node.to_dict())

        return results

    @track_memory_operation("clear", "semantic")
    async def clear(self):
        """Clear all nodes from memory."""
        async with self.lock:
            # Clear in-memory nodes
            self.nodes = {}

            # TODO: Clear vector database
            # This would require a method to clear all vectors in the database
            # which is not currently part of the VectorDBProvider interface

    async def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a specific node by ID.

        Args:
            node_id: The ID of the node to get

        Returns:
            The node, or None if not found
        """
        # First check in-memory cache
        node = self.nodes.get(node_id)
        if node:
            return node

        # If not in memory, try to get from vector DB
        result = await self.vector_db.get(node_id)
        if result:
            metadata = result['metadata']
            node = KnowledgeNode(
                id=node_id,
                content=metadata.get('content', {}),
                embedding=result.get('vector'),
                created_at=metadata.get('created_at', time.time()),
                updated_at=metadata.get('updated_at', time.time()),
                importance=metadata.get('importance', 0.5),
                confidence=metadata.get('confidence', 1.0)
            )

            # Cache the node in memory
            self.nodes[node_id] = node
            return node

        return None

    async def update_importance(self, node_id: str, importance: float) -> bool:
        """Update the importance score of a node.

        Args:
            node_id: The ID of the node to update
            importance: The new importance score (0.0 to 1.0)

        Returns:
            True if the node was updated, False otherwise
        """
        async with self.lock:
            node = await self.get_node(node_id)
            if node is None:
                return False

            node.importance = max(0.0, min(1.0, importance))

            # Update in vector DB if we have an embedding
            if node.embedding:
                # Get the current metadata
                result = await self.vector_db.get(node_id)
                if result:
                    metadata = result['metadata']
                    metadata['importance'] = node.importance

                    # Update the vector DB entry
                    await self.vector_db.add(
                        id=node_id,
                        vector=node.embedding,
                        metadata=metadata
                    )

            return True

    async def update_confidence(self, node_id: str, confidence: float) -> bool:
        """Update the confidence score of a node.

        Args:
            node_id: The ID of the node to update
            confidence: The new confidence score (0.0 to 1.0)

        Returns:
            True if the node was updated, False otherwise
        """
        async with self.lock:
            node = await self.get_node(node_id)
            if node is None:
                return False

            node.confidence = max(0.0, min(1.0, confidence))

            # Update in vector DB if we have an embedding
            if node.embedding:
                # Get the current metadata
                result = await self.vector_db.get(node_id)
                if result:
                    metadata = result['metadata']
                    metadata['confidence'] = node.confidence

                    # Update the vector DB entry
                    await self.vector_db.add(
                        id=node_id,
                        vector=node.embedding,
                        metadata=metadata
                    )

            return True

    async def link_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Create a bidirectional link between two nodes.

        Args:
            node_id1: The ID of the first node
            node_id2: The ID of the second node

        Returns:
            True if the nodes were linked, False otherwise
        """
        async with self.lock:
            node1 = await self.get_node(node_id1)
            node2 = await self.get_node(node_id2)

            if node1 is None or node2 is None:
                return False

            node1.related_nodes.add(node_id2)
            node2.related_nodes.add(node_id1)

            # Note: Vector DBs typically don't store graph relationships directly
            # We keep these relationships in memory

            return True

    async def unlink_nodes(self, node_id1: str, node_id2: str) -> bool:
        """Remove a bidirectional link between two nodes.

        Args:
            node_id1: The ID of the first node
            node_id2: The ID of the second node

        Returns:
            True if the nodes were unlinked, False otherwise
        """
        async with self.lock:
            node1 = await self.get_node(node_id1)
            node2 = await self.get_node(node_id2)

            if node1 is None or node2 is None:
                return False

            if node_id2 in node1.related_nodes:
                node1.related_nodes.remove(node_id2)

            if node_id1 in node2.related_nodes:
                node2.related_nodes.remove(node_id1)

            return True

    async def get_related_nodes(self, node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get nodes related to a specific node.

        Args:
            node_id: The ID of the node
            limit: Maximum number of related nodes to return

        Returns:
            A list of related nodes as dictionaries
        """
        node = await self.get_node(node_id)
        if node is None:
            return []

        related_nodes = []
        for related_id in node.related_nodes:
            related_node = await self.get_node(related_id)
            if related_node:
                related_nodes.append(related_node.to_dict())
                if len(related_nodes) >= limit:
                    break

        return related_nodes

    async def search_hybrid(
        self,
        query_text: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search using both text and vector similarity.

        Args:
            query_text: The text query
            limit: Maximum number of results to return
            filters: Optional metadata filters
            alpha: Weight between vector (alpha) and text (1-alpha) search

        Returns:
            A list of matching nodes as dictionaries
        """
        if not self.embedding_provider:
            logger.warning("Hybrid search requires an embedding provider")
            return []

        # Generate embedding for the query
        query_embedding = await self.embedding_provider.get_embedding(query_text)

        # Perform hybrid search
        vector_results = await self.vector_db.search_hybrid(
            query_text=query_text,
            query_vector=query_embedding,
            limit=limit,
            filters=filters,
            alpha=alpha
        )

        # Convert results to node dictionaries
        results = []
        for result in vector_results:
            # Get the node from memory or create from vector DB result
            node_id = result['id']
            node = await self.get_node(node_id)

            if node:
                node_dict = node.to_dict()
                node_dict['similarity'] = result['score']
                results.append(node_dict)

        return results

    async def consolidate_memories(self):
        """Consolidate memories by removing less important nodes if needed."""
        async with self.lock:
            # Only consolidate if we have more nodes than the maximum
            if len(self.nodes) <= self.max_nodes:
                self.last_consolidation = time.time()
                return

            # Sort nodes by importance
            sorted_nodes = sorted(
                self.nodes.values(),
                key=lambda node: node.importance
            )

            # Remove the least important nodes
            nodes_to_remove = len(self.nodes) - self.max_nodes
            for node in sorted_nodes[:nodes_to_remove]:
                # Only remove nodes below the forgetting threshold
                if node.importance <= self.forgetting_threshold:
                    # Remove links to this node from other nodes
                    for other_node in self.nodes.values():
                        if node.id in other_node.related_nodes:
                            other_node.related_nodes.remove(node.id)

                    # Remove from vector DB
                    if node.embedding:
                        await self.vector_db.delete(node.id)

                    # Remove from memory
                    del self.nodes[node.id]

            self.last_consolidation = time.time()

    async def _find_and_link_related_nodes(self, node: KnowledgeNode):
        """Find and link semantically related nodes.

        Args:
            node: The node to find relations for
        """
        if not node.embedding or not self.embedding_provider:
            return

        # Search for similar nodes
        similar_nodes = await self.vector_db.search(
            query_vector=node.embedding,
            limit=10,  # Limit to 10 similar nodes
            filters=None
        )

        # Link to similar nodes above the threshold
        for result in similar_nodes:
            # Skip the node itself
            if result['id'] == node.id:
                continue

            # Only link if similarity is above threshold
            if result['score'] >= self.similarity_threshold:
                # Get the node
                similar_node = await self.get_node(result['id'])
                if similar_node:
                    # Create bidirectional link
                    node.related_nodes.add(similar_node.id)
                    similar_node.related_nodes.add(node.id)

    def _node_to_text(self, node: KnowledgeNode) -> str:
        """Convert a node to a text representation for embedding.

        Args:
            node: The node to convert

        Returns:
            A text representation of the node
        """
        # Start with the text field if it exists
        if 'text' in node.content:
            text = node.content['text']
        else:
            # Otherwise, create a text representation from the content
            parts = []
            for key, value in node.content.items():
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, (int, float, bool)):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    parts.append(f"{key}: {', '.join(value)}")

            text = " ".join(parts)

        return text

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            The cosine similarity (between -1 and 1)
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        # Calculate dot product
        dot_product = sum(x * y for x, y in zip(a, b))

        # Calculate magnitudes
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
