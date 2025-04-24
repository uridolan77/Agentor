"""
Milvus Vector Database Provider for the Agentor framework.

This module provides integration with Milvus, a distributed vector database
for similarity search, supporting both local and cloud deployments.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
import time
import json
import uuid
from abc import ABC, abstractmethod

from agentor.components.memory.vector_db import VectorDBProvider

logger = logging.getLogger(__name__)


class MilvusProvider(VectorDBProvider):
    """Vector database provider using Milvus."""

    def __init__(
        self,
        collection_name: str,
        uri: str = "localhost:19530",
        user: str = "",
        password: str = "",
        dimension: int = 1536,
        metric_type: str = "COSINE",
        create_collection_if_not_exists: bool = True
    ):
        """Initialize the Milvus provider.

        Args:
            collection_name: Name of the Milvus collection
            uri: URI of the Milvus server
            user: Username for authentication (optional)
            password: Password for authentication (optional)
            dimension: Dimension of the vectors
            metric_type: Distance metric to use (COSINE, L2, IP)
            create_collection_if_not_exists: Whether to create the collection if it doesn't exist
        """
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        except ImportError:
            raise ImportError("Milvus package is required for MilvusProvider. Install with 'pip install pymilvus'")

        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.create_collection_if_not_exists = create_collection_if_not_exists
        self.uri = uri
        self.user = user
        self.password = password

        # Import here to avoid module-level import issues
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        self.connections = connections
        self.Collection = Collection
        self.FieldSchema = FieldSchema
        self.CollectionSchema = CollectionSchema
        self.DataType = DataType
        self.utility = utility

        # Connect to Milvus
        self._connect()

        # Check if collection exists and create if needed
        if create_collection_if_not_exists and not self.utility.has_collection(collection_name):
            self._create_collection()

        # Get the collection
        self.collection = self.Collection(collection_name)
        logger.info(f"Connected to Milvus collection: {collection_name}")

    def _connect(self):
        """Connect to Milvus server."""
        self.connections.connect(
            alias="default",
            uri=self.uri,
            user=self.user,
            password=self.password
        )
        logger.debug(f"Connected to Milvus server at {self.uri}")

    def _create_collection(self):
        """Create a new collection in Milvus."""
        # Define fields for the collection
        fields = [
            self.FieldSchema(name="id", dtype=self.DataType.VARCHAR, is_primary=True, max_length=100),
            self.FieldSchema(name="vector", dtype=self.DataType.FLOAT_VECTOR, dim=self.dimension),
            self.FieldSchema(name="metadata", dtype=self.DataType.JSON)
        ]

        # Create collection schema
        schema = self.CollectionSchema(fields=fields)

        # Create collection
        collection = self.Collection(
            name=self.collection_name,
            schema=schema,
            using="default"
        )

        # Create index on vector field
        index_params = {
            "metric_type": self.metric_type,
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        logger.info(f"Created Milvus collection: {self.collection_name}")

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to Milvus.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.collection.insert([
                [id],
                [vector],
                [metadata or {}]
            ])
        )
        logger.debug(f"Added vector {id} to Milvus")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from Milvus.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                expr=f'id == "{id}"',
                output_fields=["id", "vector", "metadata"]
            )
        )

        if response:
            return {
                'id': response[0]['id'],
                'vector': response[0]['vector'],
                'metadata': response[0]['metadata'],
                'score': 1.0  # Perfect match for direct retrieval
            }

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector from Milvus.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.collection.delete(expr=f'id == "{id}"')
        )

        if response and response.delete_count > 0:
            logger.debug(f"Deleted vector {id} from Milvus")
            return True

        return False

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Milvus.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Convert filters to Milvus expression format if provided
        expr = None
        if filters:
            expr = self._convert_filters(filters)

        # Prepare search parameters
        search_params = {
            "metric_type": self.metric_type,
            "params": {"ef": 64}
        }

        # Load the collection if not loaded
        if not self.collection.is_loaded:
            self.collection.load()

        # Perform the search
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["metadata"]
            )
        )

        results = []
        for hits in response:
            for hit in hits:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'metadata': hit.entity.get('metadata', {})
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
        """Perform a hybrid search combining vector similarity with text matching.

        Args:
            query_text: The text query
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters
            alpha: Weight of vector search vs text search (0.0 to 1.0)

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Milvus doesn't have built-in hybrid search, so we'll implement a basic version
        # by doing a vector search and then re-ranking based on text similarity
        try:
            # First, do a vector search with a higher limit
            vector_results = await self.search(
                query_vector=query_vector,
                limit=limit * 2,  # Get more results for re-ranking
                filters=filters
            )

            # If no results or alpha is 1.0 (vector only), return vector results
            if not vector_results or alpha >= 1.0:
                return vector_results[:limit]

            # Re-rank results based on text similarity
            for result in vector_results:
                # Get text from metadata if available
                text = ""
                metadata = result.get('metadata', {})
                if 'text' in metadata:
                    text = metadata['text']
                elif 'content' in metadata and isinstance(metadata['content'], dict) and 'text' in metadata['content']:
                    text = metadata['content']['text']

                # Calculate simple text similarity (this is a basic implementation)
                # In a real implementation, you might want to use a more sophisticated text similarity measure
                text_score = self._calculate_text_similarity(query_text, text)

                # Combine scores with alpha weighting
                combined_score = alpha * result['score'] + (1 - alpha) * text_score
                result['score'] = combined_score

            # Sort by combined score and return top results
            vector_results.sort(key=lambda x: x['score'], reverse=True)
            return vector_results[:limit]

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
            return await self.search(query_vector, limit, filters)

    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """Calculate a simple text similarity score.

        Args:
            query: The query text
            text: The text to compare against

        Returns:
            A similarity score between 0.0 and 1.0
        """
        if not query or not text:
            return 0.0

        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        text_lower = text.lower()

        # Split into words
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())

        # Calculate Jaccard similarity
        if not query_words or not text_words:
            return 0.0

        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))

        return intersection / union if union > 0 else 0.0

    def _convert_filters(self, filters: Dict[str, Any]) -> str:
        """Convert dictionary filters to Milvus expression format.

        Args:
            filters: Dictionary of filters

        Returns:
            Milvus expression string
        """
        expressions = []

        for key, value in filters.items():
            # Handle different value types
            if isinstance(value, str):
                # For string values, use JSON_CONTAINS for partial matching
                expressions.append(f'JSON_CONTAINS(metadata, "{{\"{key}\": \"{value}\"}}")')
            elif isinstance(value, (int, float, bool)):
                # For numeric and boolean values
                expressions.append(f'JSON_CONTAINS(metadata, "{{\"{key}\": {value}}}")')
            elif isinstance(value, list):
                # For list values, check if any item in the list matches
                or_expressions = []
                for item in value:
                    if isinstance(item, str):
                        or_expressions.append(f'JSON_CONTAINS(metadata, "{{\"{key}\": \"{item}\"}}")')
                    else:
                        or_expressions.append(f'JSON_CONTAINS(metadata, "{{\"{key}\": {item}}}")')
                if or_expressions:
                    expressions.append(f"({' OR '.join(or_expressions)})")

        # Combine all expressions with AND
        return " AND ".join(expressions) if expressions else ""

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection in Milvus.

        Args:
            name: The name of the collection
            dimension: The dimension of the vectors
            **kwargs: Additional parameters for collection creation

        Returns:
            True if the collection was created, False otherwise
        """
        try:
            # Check if collection already exists
            if self.utility.has_collection(name):
                logger.warning(f"Collection {name} already exists in Milvus")
                return False

            # Get metric type
            metric_type = kwargs.get('metric_type', self.metric_type)

            # Define fields for the collection
            fields = [
                self.FieldSchema(name="id", dtype=self.DataType.VARCHAR, is_primary=True, max_length=100),
                self.FieldSchema(name="vector", dtype=self.DataType.FLOAT_VECTOR, dim=dimension),
                self.FieldSchema(name="metadata", dtype=self.DataType.JSON)
            ]

            # Create collection schema
            schema = self.CollectionSchema(fields=fields)

            # Create collection
            collection = self.Collection(
                name=name,
                schema=schema,
                using="default"
            )

            # Create index on vector field
            index_params = {
                "metric_type": metric_type,
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"Created Milvus collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Milvus collection {name}: {str(e)}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists in Milvus.

        Args:
            name: The name of the collection

        Returns:
            True if the collection exists, False otherwise
        """
        return self.utility.has_collection(name)
