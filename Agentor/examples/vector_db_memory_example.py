"""
Example demonstrating the enhanced semantic memory with vector database integration.

This example shows how to use different vector database providers with the
enhanced semantic memory system, including:
- In-memory vector database for testing
- Qdrant for local or cloud-based vector storage
- Pinecone for cloud-based vector storage
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional

from agentor.components.memory.embedding import MockEmbeddingProvider, OpenAIEmbeddingProvider
from agentor.components.memory.vector_db import InMemoryVectorDB, QdrantProvider, PineconeProvider
from agentor.components.memory.semantic_memory import SemanticMemory as EnhancedSemanticMemory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def in_memory_vector_db_example():
    """Example using the in-memory vector database."""
    logger.info("\n=== In-Memory Vector DB Example ===")

    # Create an embedding provider
    embedding_provider = MockEmbeddingProvider(dimension=384)

    # Create an in-memory vector database
    vector_db = InMemoryVectorDB(dimension=384)

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
        "text": "The Eiffel Tower is in Paris.",
        "category": "geography",
        "importance": 0.7,
        "confidence": 1.0
    })

    await memory.add({
        "text": "The Louvre Museum is in Paris.",
        "category": "culture",
        "importance": 0.6,
        "confidence": 1.0
    })

    await memory.add({
        "text": "Rome is the capital of Italy.",
        "category": "geography",
        "importance": 0.8,
        "confidence": 1.0
    })

    await memory.add({
        "text": "The Colosseum is in Rome.",
        "category": "culture",
        "importance": 0.7,
        "confidence": 1.0
    })

    # Perform a semantic search
    logger.info("Semantic search for 'Paris landmarks':")
    results = await memory.get({"text": "Paris landmarks"})
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    # Perform a hybrid search
    logger.info("\nHybrid search for 'Paris landmarks':")
    results = await memory.search_hybrid("Paris landmarks")
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    # Filter by category
    logger.info("\nFilter by category 'culture':")
    results = await memory.get({"category": "culture"})
    for i, result in enumerate(results):
        logger.info(f"  {i+1}. {result['content'].get('text')}")


async def qdrant_example():
    """Example using Qdrant vector database."""
    logger.info("\n=== Qdrant Vector DB Example ===")

    # Check if we should skip this example
    if not os.environ.get("QDRANT_ENABLED"):
        logger.info("Skipping Qdrant example. Set QDRANT_ENABLED=1 to run it.")
        return

    try:
        # Create an embedding provider
        embedding_provider = MockEmbeddingProvider(dimension=384)

        # Create a Qdrant vector database
        vector_db = QdrantProvider(
            collection_name="agentor_example",
            dimension=384,
            create_collection_if_not_exists=True
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

        await memory.add({
            "text": "Python is a popular programming language for AI development.",
            "category": "technology",
            "importance": 0.6,
            "confidence": 1.0
        })

        await memory.add({
            "text": "TensorFlow and PyTorch are popular deep learning frameworks.",
            "category": "technology",
            "importance": 0.7,
            "confidence": 1.0
        })

        # Perform a semantic search
        logger.info("Semantic search for 'neural networks':")
        results = await memory.get({"text": "neural networks"})
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

        # Perform a hybrid search
        logger.info("\nHybrid search for 'machine learning frameworks':")
        results = await memory.search_hybrid("machine learning frameworks")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    except Exception as e:
        logger.error(f"Error in Qdrant example: {str(e)}")


async def pinecone_example():
    """Example using Pinecone vector database."""
    logger.info("\n=== Pinecone Vector DB Example ===")

    # Check if we should skip this example
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.info("Skipping Pinecone example. Set PINECONE_API_KEY environment variable to run it.")
        return

    try:
        # Get Pinecone environment
        environment = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")

        # Create an embedding provider (OpenAI for better quality)
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            embedding_provider = OpenAIEmbeddingProvider(
                api_key=openai_api_key,
                model="text-embedding-ada-002"
            )
            dimension = 1536  # OpenAI embedding dimension
        else:
            logger.info("Using mock embeddings for Pinecone example. Set OPENAI_API_KEY for better results.")
            embedding_provider = MockEmbeddingProvider(dimension=384)
            dimension = 384

        # Create a Pinecone vector database
        vector_db = PineconeProvider(
            api_key=api_key,
            environment=environment,
            index_name="agentor-example",
            dimension=dimension,
            create_index_if_not_exists=True
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
            "text": "Climate change is causing global temperatures to rise.",
            "category": "environment",
            "importance": 0.9,
            "confidence": 1.0
        })

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

        await memory.add({
            "text": "Recycling helps reduce waste and conserve natural resources.",
            "category": "environment",
            "importance": 0.6,
            "confidence": 1.0
        })

        # Perform a semantic search
        logger.info("Semantic search for 'renewable energy':")
        results = await memory.get({"text": "renewable energy"})
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

        # Perform a hybrid search
        logger.info("\nHybrid search for 'reducing carbon emissions':")
        results = await memory.search_hybrid("reducing carbon emissions")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['content'].get('text')} (similarity: {result.get('similarity', 0):.3f})")

    except Exception as e:
        logger.error(f"Error in Pinecone example: {str(e)}")


async def main():
    """Run all examples."""
    await in_memory_vector_db_example()
    await qdrant_example()
    await pinecone_example()


if __name__ == "__main__":
    asyncio.run(main())
