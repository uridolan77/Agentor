"""
Unit tests for the enhanced semantic memory with vector database integration.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List

from agentor.components.memory.embedding import MockEmbeddingProvider
from agentor.components.memory.vector_db import InMemoryVectorDB
from agentor.components.memory.semantic_memory import SemanticMemory as EnhancedSemanticMemory


@pytest.fixture
def embedding_provider():
    """Create a mock embedding provider for testing."""
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def vector_db():
    """Create an in-memory vector database for testing."""
    return InMemoryVectorDB(dimension=384)


@pytest.fixture
def memory(embedding_provider, vector_db):
    """Create an enhanced semantic memory for testing."""
    return EnhancedSemanticMemory(
        vector_db=vector_db,
        embedding_provider=embedding_provider,
        max_nodes=100,
        forgetting_threshold=0.2,
        similarity_threshold=0.85
    )


@pytest.mark.asyncio
async def test_add_and_retrieve(memory):
    """Test adding and retrieving items from memory."""
    # Add an item
    await memory.add({
        "text": "Paris is the capital of France.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })

    # Retrieve by exact match
    results = await memory.get({"category": "geography"})
    assert len(results) == 1
    assert results[0]["content"]["text"] == "Paris is the capital of France."

    # Add more items
    await memory.add({
        "text": "Rome is the capital of Italy.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })

    await memory.add({
        "text": "Berlin is the capital of Germany.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })

    # Retrieve all geography items
    results = await memory.get({"category": "geography"})
    assert len(results) == 3

    # Retrieve by semantic search
    results = await memory.get({"text": "capital of France"})
    assert len(results) > 0
    assert results[0]["content"]["text"] == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_hybrid_search(memory):
    """Test hybrid search functionality."""
    # Add items
    await memory.add({
        "text": "Machine learning is a subset of artificial intelligence.",
        "category": "technology",
        "importance": 0.8,
        "confidence": 1.0
    })

    await memory.add({
        "text": "Deep learning uses neural networks with multiple layers.",
        "category": "technology",
        "importance": 0.7,
        "confidence": 1.0
    })

    await memory.add({
        "text": "Python is a popular programming language for AI development.",
        "category": "technology",
        "importance": 0.6,
        "confidence": 1.0
    })

    # Perform hybrid search
    results = await memory.search_hybrid("neural networks in AI")
    assert len(results) > 0

    # The most relevant result should be about deep learning
    assert "neural networks" in results[0]["content"]["text"].lower()


@pytest.mark.asyncio
async def test_node_relationships(memory):
    """Test establishing relationships between nodes."""
    # Add related items
    await memory.add({
        "id": "node1",
        "text": "The Eiffel Tower is in Paris.",
        "category": "landmarks",
        "importance": 0.8,
        "confidence": 1.0
    })

    await memory.add({
        "id": "node2",
        "text": "The Louvre Museum is in Paris.",
        "category": "landmarks",
        "importance": 0.8,
        "confidence": 1.0
    })

    # Manually link nodes
    result = await memory.link_nodes("node1", "node2")
    assert result is True

    # Get related nodes
    related = await memory.get_related_nodes("node1")
    assert len(related) == 1
    assert related[0]["id"] == "node2"

    # Unlink nodes
    result = await memory.unlink_nodes("node1", "node2")
    assert result is True

    # Check that nodes are unlinked
    related = await memory.get_related_nodes("node1")
    assert len(related) == 0


@pytest.mark.asyncio
async def test_importance_and_confidence(memory):
    """Test updating importance and confidence scores."""
    # Add an item
    await memory.add({
        "id": "test_node",
        "text": "This is a test node.",
        "category": "test",
        "importance": 0.5,
        "confidence": 0.5
    })

    # Update importance
    result = await memory.update_importance("test_node", 0.8)
    assert result is True

    # Update confidence
    result = await memory.update_confidence("test_node", 0.9)
    assert result is True

    # Retrieve the node
    node = await memory.get_node("test_node")
    assert node is not None
    assert node.importance == 0.8
    assert node.confidence == 0.9


@pytest.mark.asyncio
async def test_consolidation(memory):
    """Test memory consolidation."""
    # Set a small max_nodes value
    memory.max_nodes = 3

    # Add more items than max_nodes
    for i in range(5):
        await memory.add({
            "id": f"node{i}",
            "text": f"Test node {i}",
            "category": "test",
            "importance": 0.1 * i,  # Lower importance for earlier nodes
            "confidence": 1.0
        })

    # Force consolidation
    await memory.consolidate_memories()

    # Check that low-importance nodes were removed
    assert len(memory.nodes) <= 3

    # The highest importance nodes should remain
    assert "node4" in memory.nodes
    assert "node3" in memory.nodes
