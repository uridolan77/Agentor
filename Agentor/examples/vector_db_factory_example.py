"""
Example demonstrating the VectorDBFactory for creating different vector database providers.

This example shows how to use the VectorDBFactory to create and use different
vector database providers with the enhanced semantic memory system.
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional

from agentor.components.memory.embedding import MockEmbeddingProvider, OpenAIEmbeddingProvider
from agentor.components.memory.semantic_memory import SemanticMemory as EnhancedSemanticMemory
from agentor.components.memory.vector_db_factory import VectorDBFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def vector_db_factory_example():
    """Example using the VectorDBFactory to create different vector database providers."""
    logger.info("\n=== Vector DB Factory Example ===")

    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)

    # Example 1: In-memory vector database
    logger.info("\n--- In-Memory Vector DB Example ---")

    # Create an in-memory vector database using the factory
    vector_db = VectorDBFactory.create(
        db_type="in_memory",
        dimension=384
    )

    # Create an enhanced semantic memory
    memory = EnhancedSemanticMemory(
        vector_db=vector_db,
        embedding_provider=embedding_provider,
        max_nodes=1000,
        forgetting_threshold=0.2,
        similarity_threshold=0.85
    )

    # Add knowledge to memory
    await memory.add({
        "text": "Paris is the capital of France.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })

    await memory.add({
        "text": "Rome is the capital of Italy.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })

    # Perform a semantic search
    logger.info("Semantic search for 'capital of France':")
    results = await memory.get({"text": "capital of France"})
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    # Example 2: FAISS vector database
    logger.info("\n--- FAISS Vector DB Example ---")

    try:
        # Create a FAISS vector database using the factory
        vector_db = VectorDBFactory.create(
            db_type="faiss",
            index_name="agentor_example",
            dimension=384,
            index_type="IndexFlatL2",
            save_path="./data"
        )

        # Create an enhanced semantic memory
        memory = EnhancedSemanticMemory(
            vector_db=vector_db,
            embedding_provider=embedding_provider,
            max_nodes=1000,
            forgetting_threshold=0.2,
            similarity_threshold=0.85
        )

        # Add knowledge to memory
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

        # Perform a semantic search
        logger.info("Semantic search for 'neural networks':")
        results = await memory.get({"text": "neural networks"})
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    except Exception as e:
        logger.error(f"FAISS example failed: {str(e)}")

    # Example 3: ChromaDB vector database
    logger.info("\n--- ChromaDB Vector DB Example ---")

    try:
        # Create a ChromaDB vector database using the factory
        vector_db = VectorDBFactory.create(
            db_type="chroma",
            collection_name="agentor_example",
            persist_directory="./data/chromadb"
        )

        # Create an enhanced semantic memory
        memory = EnhancedSemanticMemory(
            vector_db=vector_db,
            embedding_provider=embedding_provider,
            max_nodes=1000,
            forgetting_threshold=0.2,
            similarity_threshold=0.85
        )

        # Add knowledge to memory
        await memory.add({
            "text": "Renewable energy sources include solar, wind, and hydroelectric power.",
            "category": "environment",
            "importance": 0.8,
            "confidence": 1.0
        })

        await memory.add({
            "text": "Electric vehicles produce fewer emissions than gasoline-powered cars.",
            "category": "environment",
            "importance": 0.7,
            "confidence": 1.0
        })

        # Perform a semantic search
        logger.info("Semantic search for 'renewable energy':")
        results = await memory.get({"text": "renewable energy"})
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

        # Perform a hybrid search
        logger.info("\nHybrid search for 'clean energy sources':")
        results = await memory.search_hybrid("clean energy sources")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    except Exception as e:
        logger.error(f"ChromaDB example failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(vector_db_factory_example())
