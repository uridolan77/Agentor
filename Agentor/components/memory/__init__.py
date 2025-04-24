"""
Memory module for the Agentor framework.

This module provides various memory implementations for agents, including:
- Episodic memory for storing sequences of events
- Semantic memory for storing knowledge and facts
- Procedural memory for storing learned behaviors and skills
- Vector database integration for efficient semantic search
- Adapters for using memory components with the standardized interfaces
"""

# Import base memory classes
from .base import Memory, SimpleMemory, VectorMemory
from .episodic_memory import EpisodicMemory, Episode
from .semantic_memory import SemanticMemory, KnowledgeNode
from .procedural_memory import ProceduralMemory, Procedure
# Enhanced semantic memory has been merged into semantic_memory
from .vector_db import (
    VectorDBProvider,
    InMemoryVectorDB,
    PineconeProvider,
    QdrantProvider
)
from .vector_db_milvus import MilvusProvider
from .vector_db_weaviate import WeaviateProvider
from .vector_db_faiss import FAISSProvider
from .vector_db_chroma import ChromaDBProvider
from .vector_db_factory import VectorDBFactory
from .adapters import (
    MemoryAdapter,
    SimpleMemoryAdapter,
    VectorMemoryAdapter,
    EpisodicMemoryAdapter,
    SemanticMemoryAdapter,
    ProceduralMemoryAdapter,
    UnifiedMemoryAdapter,
    SimpleMemoryPlugin,
    VectorMemoryPlugin,
    EpisodicMemoryPlugin,
    SemanticMemoryPlugin,
    ProceduralMemoryPlugin,
    UnifiedMemoryPlugin
)

__all__ = [
    # Base memory classes
    'Memory',
    'SimpleMemory',
    'VectorMemory',
    'EpisodicMemory',
    'Episode',
    'SemanticMemory',
    'KnowledgeNode',
    'ProceduralMemory',
    'Procedure',

    # Vector database providers
    'VectorDBProvider',
    'InMemoryVectorDB',
    'PineconeProvider',
    'QdrantProvider',
    'MilvusProvider',
    'WeaviateProvider',
    'FAISSProvider',
    'ChromaDBProvider',
    'VectorDBFactory',

    # Memory adapters
    'MemoryAdapter',
    'SimpleMemoryAdapter',
    'VectorMemoryAdapter',
    'EpisodicMemoryAdapter',
    'SemanticMemoryAdapter',
    'ProceduralMemoryAdapter',
    'UnifiedMemoryAdapter',

    # Memory plugins
    'SimpleMemoryPlugin',
    'VectorMemoryPlugin',
    'EpisodicMemoryPlugin',
    'SemanticMemoryPlugin',
    'ProceduralMemoryPlugin',
    'UnifiedMemoryPlugin',
]
