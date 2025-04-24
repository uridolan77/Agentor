# Vector Database Integration for Agentor

This module provides integrations with specialized vector databases for efficient storage and retrieval of embeddings in the Agentor framework.

## Overview

The vector database integration enhances Agentor's memory systems with:

- Efficient storage and retrieval of vector embeddings
- Approximate nearest neighbor (ANN) search for fast similarity queries
- Hybrid search combining semantic and keyword-based approaches
- Scalable solutions for large embedding collections
- Support for multiple vector database providers

## Supported Vector Databases

### InMemoryVectorDB

A simple in-memory vector database for testing and development. Useful for small-scale applications and prototyping.

```python
from agentor.components.memory.vector_db import InMemoryVectorDB

vector_db = InMemoryVectorDB(dimension=384)
```

### Pinecone

Cloud-native vector database with high scalability and performance.

```python
from agentor.components.memory.vector_db import PineconeProvider

vector_db = PineconeProvider(
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="your-index",
    dimension=1536,
    create_index_if_not_exists=True
)
```

### Qdrant

Open-source vector database with rich filtering capabilities.

```python
from agentor.components.memory.vector_db import QdrantProvider

vector_db = QdrantProvider(
    collection_name="your-collection",
    url="http://localhost:6333",  # Optional, for remote Qdrant
    dimension=1536,
    create_collection_if_not_exists=True
)
```

### Milvus

Distributed vector database for similarity search with high scalability.

```python
from agentor.components.memory import MilvusProvider

vector_db = MilvusProvider(
    collection_name="your-collection",
    uri="localhost:19530",  # Milvus server URI
    dimension=1536,
    create_collection_if_not_exists=True
)
```

### Weaviate

Knowledge graph with vector search capabilities, supporting semantic search.

```python
from agentor.components.memory import WeaviateProvider

vector_db = WeaviateProvider(
    class_name="YourClass",
    url="http://localhost:8080",  # Weaviate server URL
    dimension=1536,
    create_class_if_not_exists=True
)
```

### FAISS

Facebook AI Similarity Search (FAISS) for efficient similarity search and clustering of dense vectors.

```python
from agentor.components.memory import FAISSProvider

vector_db = FAISSProvider(
    index_name="your-index",
    dimension=1536,
    index_type="IndexFlatL2",  # or "IndexHNSWFlat" for approximate search
    save_path="/path/to/save"  # Optional, to persist the index
)
```

### ChromaDB

Open-source embedding database designed for document embeddings and retrieval.

```python
from agentor.components.memory import ChromaDBProvider

vector_db = ChromaDBProvider(
    collection_name="your-collection",
    persist_directory="/path/to/persist",  # Optional, to persist the database
    create_collection_if_not_exists=True
)
```

## Using with Semantic Memory

The vector database providers are designed to work with the `SemanticMemory` class:

```python
from agentor.components.memory.embedding import OpenAIEmbeddingProvider
from agentor.components.memory.vector_db import QdrantProvider
from agentor.components.memory.semantic_memory import SemanticMemory

# Create an embedding provider
embedding_provider = OpenAIEmbeddingProvider(
    api_key="your-openai-api-key",
    model="text-embedding-ada-002"
)

# Create a vector database provider
vector_db = QdrantProvider(
    collection_name="semantic-memory",
    dimension=1536,
    create_collection_if_not_exists=True
)

# Create a semantic memory
memory = SemanticMemory(
    vector_db=vector_db,
    embedding_provider=embedding_provider,
    max_nodes=10000,
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

# Perform a semantic search
results = await memory.get({"text": "capital of France"})

# Perform a hybrid search (combining vector and text search)
results = await memory.search_hybrid("European capitals")
```

## Hybrid Search

Hybrid search combines vector similarity with text-based search for improved results:

```python
# Perform a hybrid search with custom weighting
results = await memory.search_hybrid(
    query_text="renewable energy sources",
    limit=10,
    filters={"category": "environment"},
    alpha=0.7  # 70% weight on vector similarity, 30% on text matching
)
```

## Filtering

All vector database providers support filtering based on metadata:

```python
# Filter by category
results = await memory.get({
    "text": "capital cities",
    "category": "geography"
})

# Multiple filters
results = await memory.get({
    "text": "landmarks",
    "category": "tourism",
    "importance": 0.8
})
```

## Using the VectorDBFactory

The `VectorDBFactory` provides a convenient way to create vector database providers without having to import each provider class directly:

```python
from agentor.components.memory import VectorDBFactory

# Create an in-memory vector database
vector_db = VectorDBFactory.create(
    db_type="in_memory",
    dimension=384
)

# Create a FAISS vector database
vector_db = VectorDBFactory.create(
    db_type="faiss",
    index_name="my_index",
    dimension=1536,
    index_type="IndexFlatL2"
)

# Create a Pinecone vector database
vector_db = VectorDBFactory.create(
    db_type="pinecone",
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="your-index",
    dimension=1536
)
```

Supported database types:

- `"in_memory"`: InMemoryVectorDB
- `"pinecone"`: PineconeProvider
- `"qdrant"`: QdrantProvider
- `"milvus"`: MilvusProvider
- `"weaviate"`: WeaviateProvider
- `"faiss"`: FAISSProvider
- `"chroma"`: ChromaDBProvider

## Extending with New Providers

To add support for additional vector databases, create a new class that implements the `VectorDBProvider` interface:

```python
from agentor.components.memory.vector_db import VectorDBProvider

class MyCustomVectorDB(VectorDBProvider):
    # Implement the required methods
    async def add(self, id, vector, metadata=None):
        # Implementation

    async def get(self, id):
        # Implementation

    async def delete(self, id):
        # Implementation

    async def search(self, query_vector, limit=10, filters=None):
        # Implementation

    async def search_hybrid(self, query_text, query_vector, limit=10, filters=None, alpha=0.5):
        # Implementation
```

Then, update the `VectorDBFactory` to support your new provider:

```python
from agentor.components.memory.vector_db_factory import VectorDBFactory
from my_module import MyCustomVectorDB

# Extend the factory with your custom provider
class ExtendedVectorDBFactory(VectorDBFactory):
    @staticmethod
    def create(db_type: str, **kwargs):
        if db_type == "my_custom_db":
            return MyCustomVectorDB(**kwargs)
        else:
            return VectorDBFactory.create(db_type, **kwargs)
```

## Performance Considerations

- For large-scale applications, use cloud-based solutions like Pinecone
- For self-hosted options, Qdrant provides excellent performance
- Use batch operations when adding multiple vectors
- Consider index sharding for very large collections
- Monitor memory usage when using the in-memory provider
