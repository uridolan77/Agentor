"""
Vector Database Factory for the Agentor framework.

This module provides a factory class for creating vector database providers,
making it easy to switch between different vector database implementations.
"""

from typing import Dict, Any, Optional

from agentor.components.memory.vector_db import (
    VectorDBProvider,
    InMemoryVectorDB,
    PineconeProvider,
    QdrantProvider
)

# Import the new providers (will be implemented)
from agentor.components.memory.vector_db_milvus import MilvusProvider
from agentor.components.memory.vector_db_weaviate import WeaviateProvider
from agentor.components.memory.vector_db_faiss import FAISSProvider
from agentor.components.memory.vector_db_chroma import ChromaDBProvider


class VectorDBFactory:
    """Factory class for creating vector database providers."""

    @staticmethod
    def create(db_type: str, **kwargs) -> VectorDBProvider:
        """Create a vector database provider.

        Args:
            db_type: The type of vector database to create
            **kwargs: Additional arguments to pass to the provider constructor

        Returns:
            A vector database provider instance

        Raises:
            ValueError: If the specified database type is not supported
        """
        if db_type == "pinecone":
            return PineconeProvider(**kwargs)
        elif db_type == "milvus":
            return MilvusProvider(**kwargs)
        elif db_type == "weaviate":
            return WeaviateProvider(**kwargs)
        elif db_type == "faiss":
            return FAISSProvider(**kwargs)
        elif db_type == "chroma":
            return ChromaDBProvider(**kwargs)
        elif db_type == "qdrant":
            return QdrantProvider(**kwargs)
        elif db_type == "in_memory":
            return InMemoryVectorDB(**kwargs)
        else:
            raise ValueError(f"Unknown vector DB type: {db_type}")
