"""
Vector Database integration for the Agentor framework.

This module provides integrations with specialized vector databases for efficient
storage and retrieval of embeddings, including:
- Pinecone: Cloud-native vector database with high scalability
- Qdrant: Open-source vector database with rich filtering
- Weaviate: Knowledge graph with vector search capabilities
- Milvus: Distributed vector database for similarity search
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import logging
import time
import numpy as np
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VectorDBProvider(ABC):
    """Base class for vector database providers."""

    @abstractmethod
    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to the database.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector from the database.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        pass

    @abstractmethod
    async def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search using both text and vector similarity.

        Args:
            query_text: The text query for keyword search
            query_vector: The query vector for semantic search
            limit: Maximum number of results to return
            filters: Optional metadata filters
            alpha: Weight between vector (alpha) and text (1-alpha) search

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        pass

    @abstractmethod
    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection/index.

        Args:
            name: The name of the collection/index
            dimension: The dimension of the vectors
            **kwargs: Additional provider-specific parameters

        Returns:
            True if the collection was created, False otherwise
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection/index.

        Args:
            name: The name of the collection/index

        Returns:
            True if the collection was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections/indexes.

        Returns:
            A list of collection/index names
        """
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection/index exists.

        Args:
            name: The name of the collection/index

        Returns:
            True if the collection exists, False otherwise
        """
        pass


class PineconeProvider(VectorDBProvider):
    """Vector database provider using Pinecone."""

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 1536,
        namespace: str = "",
        create_index_if_not_exists: bool = True
    ):
        """Initialize the Pinecone provider.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Dimension of the vectors
            namespace: Optional namespace within the index
            create_index_if_not_exists: Whether to create the index if it doesn't exist
        """
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone package is required for PineconeProvider. Install with 'pip install pinecone-client'")

        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.namespace = namespace
        self.create_index_if_not_exists = create_index_if_not_exists
        self.pinecone = pinecone

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Check if index exists and create if needed
        if create_index_if_not_exists and index_name not in pinecone.list_indexes():
            logger.info(f"Creating Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )

        # Connect to the index
        self.index = pinecone.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to Pinecone.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.index.upsert(
                vectors=[(id, vector, metadata or {})],
                namespace=self.namespace
            )
        )
        logger.debug(f"Added vector {id} to Pinecone")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from Pinecone.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.index.fetch(ids=[id], namespace=self.namespace)
        )

        if id in response['vectors']:
            vector_data = response['vectors'][id]
            return {
                'id': id,
                'vector': vector_data['values'],
                'metadata': vector_data['metadata'],
                'score': 1.0  # Perfect match for direct retrieval
            }

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector from Pinecone.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.index.delete(ids=[id], namespace=self.namespace)
        )
        logger.debug(f"Deleted vector {id} from Pinecone")
        return True

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.index.query(
                vector=query_vector,
                top_k=limit,
                namespace=self.namespace,
                filter=filters
            )
        )

        results = []
        for match in response['matches']:
            results.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            })

        return results

    async def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search using both text and vector similarity in Pinecone.

        Note: This implementation depends on Pinecone's hybrid search capability,
        which requires the index to be configured with text indexing.

        Args:
            query_text: The text query for keyword search
            query_vector: The query vector for semantic search
            limit: Maximum number of results to return
            filters: Optional metadata filters
            alpha: Weight between vector (alpha) and text (1-alpha) search

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Check if the index supports hybrid search
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_vector,
                    top_k=limit,
                    namespace=self.namespace,
                    filter=filters,
                    alpha=alpha,
                    query_text=query_text
                )
            )

            results = []
            for match in response['matches']:
                results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata']
                })

            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
            return await self.search(query_vector, limit, filters)

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new index in Pinecone.

        Args:
            name: The name of the index
            dimension: The dimension of the vectors
            **kwargs: Additional parameters (e.g., metric)

        Returns:
            True if the index was created, False otherwise
        """
        try:
            # Get metric
            metric = kwargs.get('metric', 'cosine')

            # Check if index already exists
            if name in self.pinecone.list_indexes():
                logger.warning(f"Index {name} already exists in Pinecone")
                return False

            # Create the index
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.pinecone.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric
                )
            )

            logger.info(f"Created Pinecone index: {name}")
            return True

        except Exception as e:
            logger.error(f"Error creating Pinecone index {name}: {str(e)}")
            return False

    async def delete_collection(self, name: str) -> bool:
        """Delete an index from Pinecone.

        Args:
            name: The name of the index

        Returns:
            True if the index was deleted, False otherwise
        """
        try:
            # Check if index exists
            if name not in self.pinecone.list_indexes():
                logger.warning(f"Index {name} does not exist in Pinecone")
                return False

            # Delete the index
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.pinecone.delete_index(name)
            )

            logger.info(f"Deleted Pinecone index: {name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting Pinecone index {name}: {str(e)}")
            return False

    async def list_collections(self) -> List[str]:
        """List all indexes in Pinecone.

        Returns:
            A list of index names
        """
        try:
            loop = asyncio.get_event_loop()
            indexes = await loop.run_in_executor(
                None,
                lambda: self.pinecone.list_indexes()
            )

            return indexes

        except Exception as e:
            logger.error(f"Error listing Pinecone indexes: {str(e)}")
            return []

    async def collection_exists(self, name: str) -> bool:
        """Check if an index exists in Pinecone.

        Args:
            name: The name of the index

        Returns:
            True if the index exists, False otherwise
        """
        try:
            indexes = await self.list_collections()
            return name in indexes

        except Exception as e:
            logger.error(f"Error checking if Pinecone index {name} exists: {str(e)}")
            return False


class QdrantProvider(VectorDBProvider):
    """Vector database provider using Qdrant."""

    def __init__(
        self,
        collection_name: str,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = 1536,
        distance: str = "Cosine",
        create_collection_if_not_exists: bool = True
    ):
        """Initialize the Qdrant provider.

        Args:
            collection_name: Name of the Qdrant collection
            url: URL of the Qdrant server (None for local)
            api_key: API key for Qdrant Cloud (optional)
            dimension: Dimension of the vectors
            distance: Distance metric to use (Cosine, Euclid, Dot)
            create_collection_if_not_exists: Whether to create the collection if it doesn't exist
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError:
            raise ImportError("Qdrant package is required for QdrantProvider. Install with 'pip install qdrant-client'")

        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = distance
        self.create_collection_if_not_exists = create_collection_if_not_exists
        self.url = url
        self.api_key = api_key

        # Import here to avoid module-level import issues
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        self.models = models
        self.QdrantClient = QdrantClient

        # Connect to Qdrant
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(location=":memory:")

        # Create collection if needed
        if create_collection_if_not_exists:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=self.models.VectorParams(
                        size=dimension,
                        distance=self.models.Distance[distance]
                    )
                )

        logger.info(f"Connected to Qdrant collection: {collection_name}")

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to Qdrant.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    self.models.PointStruct(
                        id=id,
                        vector=vector,
                        payload=metadata or {}
                    )
                ]
            )
        )
        logger.debug(f"Added vector {id} to Qdrant")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from Qdrant.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_vectors=True
            )
        )

        if response:
            point = response[0]
            return {
                'id': point.id,
                'vector': point.vector,
                'metadata': point.payload,
                'score': 1.0  # Perfect match for direct retrieval
            }

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector from Qdrant.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.models.PointIdsList(
                    points=[id]
                )
            )
        )
        logger.debug(f"Deleted vector {id} from Qdrant")
        return True

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Convert filters to Qdrant filter format if provided
        qdrant_filter = None
        if filters:
            qdrant_filter = self._convert_filters(filters)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter
            )
        )

        results = []
        for scored_point in response:
            results.append({
                'id': scored_point.id,
                'score': scored_point.score,
                'metadata': scored_point.payload
            })

        return results

    async def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search using both text and vector similarity in Qdrant.

        Args:
            query_text: The text query for keyword search
            query_vector: The query vector for semantic search
            limit: Maximum number of results to return
            filters: Optional metadata filters
            alpha: Weight between vector (alpha) and text (1-alpha) search

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Convert filters to Qdrant filter format if provided
        qdrant_filter = None
        if filters:
            qdrant_filter = self._convert_filters(filters)

        try:
            # Use Qdrant's hybrid search if available
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search_batch(
                    collection_name=self.collection_name,
                    requests=[
                        self.models.SearchRequest(
                            vector=query_vector,
                            limit=limit,
                            filter=qdrant_filter,
                            with_payload=True,
                            params={"exact": False}
                        ),
                        self.models.SearchRequest(
                            text=query_text,
                            limit=limit,
                            filter=qdrant_filter,
                            with_payload=True
                        )
                    ]
                )
            )

            # Combine vector and text search results with weighted scores
            vector_results = {point.id: point for point in response[0]}
            text_results = {point.id: point for point in response[1]}

            # Get all unique IDs
            all_ids = set(vector_results.keys()) | set(text_results.keys())

            # Combine scores
            results = []
            for id in all_ids:
                vector_score = vector_results[id].score if id in vector_results else 0.0
                text_score = text_results[id].score if id in text_results else 0.0

                # Weighted combination
                combined_score = alpha * vector_score + (1 - alpha) * text_score

                # Get metadata from either result
                metadata = (
                    vector_results[id].payload if id in vector_results
                    else text_results[id].payload
                )

                results.append({
                    'id': id,
                    'score': combined_score,
                    'metadata': metadata
                })

            # Sort by combined score and limit results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
            return await self.search(query_vector, limit, filters)

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection in Qdrant.

        Args:
            name: The name of the collection
            dimension: The dimension of the vectors
            **kwargs: Additional parameters (e.g., distance)

        Returns:
            True if the collection was created, False otherwise
        """
        try:
            # Get distance metric
            distance = kwargs.get('distance', 'Cosine')

            # Check if collection already exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if name in collection_names:
                logger.warning(f"Collection {name} already exists in Qdrant")
                return False

            # Create the collection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.create_collection(
                    collection_name=name,
                    vectors_config=self.models.VectorParams(
                        size=dimension,
                        distance=self.models.Distance[distance]
                    )
                )
            )

            logger.info(f"Created Qdrant collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Error creating Qdrant collection {name}: {str(e)}")
            return False

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection from Qdrant.

        Args:
            name: The name of the collection

        Returns:
            True if the collection was deleted, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if name not in collection_names:
                logger.warning(f"Collection {name} does not exist in Qdrant")
                return False

            # Delete the collection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_collection(collection_name=name)
            )

            logger.info(f"Deleted Qdrant collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting Qdrant collection {name}: {str(e)}")
            return False

    async def list_collections(self) -> List[str]:
        """List all collections in Qdrant.

        Returns:
            A list of collection names
        """
        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                None,
                lambda: self.client.get_collections().collections
            )

            return [collection.name for collection in collections]

        except Exception as e:
            logger.error(f"Error listing Qdrant collections: {str(e)}")
            return []

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists in Qdrant.

        Args:
            name: The name of the collection

        Returns:
            True if the collection exists, False otherwise
        """
        try:
            collections = await self.list_collections()
            return name in collections

        except Exception as e:
            logger.error(f"Error checking if Qdrant collection {name} exists: {str(e)}")
            return False

    def _convert_filters(self, filters: Dict[str, Any]) -> Any:
        """Convert generic filters to Qdrant filter format.

        Args:
            filters: Generic filter dictionary

        Returns:
            Qdrant filter object
        """
        # This is a simplified implementation
        # In a real implementation, you would convert the filters to Qdrant's filter format
        conditions = []

        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    self.models.FieldCondition(
                        key=key,
                        match=self.models.MatchAny(any=value)
                    )
                )
            else:
                conditions.append(
                    self.models.FieldCondition(
                        key=key,
                        match=self.models.MatchValue(value=value)
                    )
                )

        if len(conditions) == 1:
            return conditions[0]
        else:
            return self.models.Filter(
                must=conditions
            )


class InMemoryVectorDB(VectorDBProvider):
    """Simple in-memory vector database for testing and development."""

    def __init__(self, dimension: int = 1536):
        """Initialize the in-memory vector database.

        Args:
            dimension: Dimension of the vectors
        """
        self.dimension = dimension
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self.collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.lock = asyncio.Lock()

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to the in-memory database.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        async with self.lock:
            self.vectors[id] = {
                'vector': vector,
                'metadata': metadata or {}
            }
        logger.debug(f"Added vector {id} to in-memory database")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from the in-memory database.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        async with self.lock:
            if id in self.vectors:
                return {
                    'id': id,
                    'vector': self.vectors[id]['vector'],
                    'metadata': self.vectors[id]['metadata'],
                    'score': 1.0  # Perfect match for direct retrieval
                }

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector from the in-memory database.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        async with self.lock:
            if id in self.vectors:
                del self.vectors[id]
                logger.debug(f"Deleted vector {id} from in-memory database")
                return True

        return False

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in the in-memory database.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        results = []

        async with self.lock:
            for id, data in self.vectors.items():
                # Apply filters if provided
                if filters and not self._matches_filters(data['metadata'], filters):
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, data['vector'])

                results.append({
                    'id': id,
                    'vector': data['vector'],
                    'metadata': data['metadata'],
                    'score': similarity
                })

        # Sort by similarity (highest first) and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    async def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid search using both text and vector similarity.

        Args:
            query_text: The text query for keyword search
            query_vector: The query vector for semantic search
            limit: Maximum number of results to return
            filters: Optional metadata filters
            alpha: Weight between vector (alpha) and text (1-alpha) search

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Get vector search results
        vector_results = await self.search(query_vector, limit=None, filters=filters)
        vector_scores = {result['id']: result['score'] for result in vector_results}

        # Perform text search (simple keyword matching)
        text_results = []
        query_terms = query_text.lower().split()

        async with self.lock:
            for id, data in self.vectors.items():
                # Apply filters if provided
                if filters and not self._matches_filters(data['metadata'], filters):
                    continue

                # Simple text matching score based on metadata
                text_score = self._text_match_score(data['metadata'], query_terms)

                text_results.append({
                    'id': id,
                    'score': text_score,
                    'metadata': data['metadata']
                })

        text_scores = {result['id']: result['score'] for result in text_results}

        # Combine scores
        all_ids = set(vector_scores.keys()) | set(text_scores.keys())
        combined_results = []

        for id in all_ids:
            vector_score = vector_scores.get(id, 0.0)
            text_score = text_scores.get(id, 0.0)

            # Weighted combination
            combined_score = alpha * vector_score + (1 - alpha) * text_score

            # Get metadata
            metadata = None
            for result in vector_results:
                if result['id'] == id:
                    metadata = result['metadata']
                    break

            if metadata is None:
                for result in text_results:
                    if result['id'] == id:
                        metadata = result['metadata']
                        break

            combined_results.append({
                'id': id,
                'score': combined_score,
                'metadata': metadata
            })

        # Sort by combined score and limit results
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)

        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a_np, b_np) / (norm_a * norm_b)

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False

            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False

        return True

    def _text_match_score(self, metadata: Dict[str, Any], query_terms: List[str]) -> float:
        """Calculate a simple text match score based on metadata."""
        if not query_terms:
            return 0.0

        # Convert metadata to a single string for searching
        metadata_text = json.dumps(metadata).lower()

        # Count matching terms
        matches = sum(1 for term in query_terms if term in metadata_text)

        # Calculate score as percentage of matching terms
        return matches / len(query_terms) if query_terms else 0.0

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection.

        Args:
            name: The name of the collection
            dimension: The dimension of the vectors
            **kwargs: Additional parameters (ignored for in-memory)

        Returns:
            True if the collection was created, False otherwise
        """
        async with self.lock:
            if name in self.collections:
                logger.warning(f"Collection {name} already exists")
                return False

            self.collections[name] = {}
            logger.info(f"Created in-memory collection: {name}")
            return True

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: The name of the collection

        Returns:
            True if the collection was deleted, False otherwise
        """
        async with self.lock:
            if name not in self.collections:
                logger.warning(f"Collection {name} does not exist")
                return False

            del self.collections[name]
            logger.info(f"Deleted in-memory collection: {name}")
            return True

    async def list_collections(self) -> List[str]:
        """List all collections.

        Returns:
            A list of collection names
        """
        async with self.lock:
            return list(self.collections.keys())

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: The name of the collection

        Returns:
            True if the collection exists, False otherwise
        """
        async with self.lock:
            return name in self.collections
