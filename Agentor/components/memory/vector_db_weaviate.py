"""
Weaviate Vector Database Provider for the Agentor framework.

This module provides integration with Weaviate, a knowledge graph with vector
search capabilities, supporting both local and cloud deployments.
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


class WeaviateProvider(VectorDBProvider):
    """Vector database provider using Weaviate."""

    def __init__(
        self,
        class_name: str,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        create_class_if_not_exists: bool = True
    ):
        """Initialize the Weaviate provider.

        Args:
            class_name: Name of the Weaviate class
            url: URL of the Weaviate server
            api_key: API key for authentication (optional)
            dimension: Dimension of the vectors
            create_class_if_not_exists: Whether to create the class if it doesn't exist
        """
        try:
            import weaviate
        except ImportError:
            raise ImportError("Weaviate package is required for WeaviateProvider. Install with 'pip install weaviate-client'")

        self.class_name = class_name
        self.dimension = dimension
        self.create_class_if_not_exists = create_class_if_not_exists
        self.url = url
        self.api_key = api_key

        # Import here to avoid module-level import issues
        import weaviate
        self.weaviate = weaviate

        # Connect to Weaviate
        self._connect()

        # Check if class exists and create if needed
        if create_class_if_not_exists and not self._class_exists(class_name):
            self._create_class()

        logger.info(f"Connected to Weaviate class: {class_name}")

    def _connect(self):
        """Connect to Weaviate server."""
        auth_config = None
        if self.api_key:
            auth_config = self.weaviate.auth.AuthApiKey(api_key=self.api_key)

        self.client = self.weaviate.Client(
            url=self.url,
            auth_client_secret=auth_config
        )
        logger.debug(f"Connected to Weaviate server at {self.url}")

    def _class_exists(self, class_name: str) -> bool:
        """Check if a class exists in Weaviate.

        Args:
            class_name: The name of the class

        Returns:
            True if the class exists, False otherwise
        """
        try:
            schema = self.client.schema.get()
            classes = schema.get('classes', [])
            return any(cls.get('class') == class_name for cls in classes)
        except Exception as e:
            logger.error(f"Error checking if class exists: {str(e)}")
            return False

    def _create_class(self):
        """Create a new class in Weaviate."""
        class_obj = {
            "class": self.class_name,
            "vectorizer": "none",  # We'll provide our own vectors
            "properties": [
                {
                    "name": "metadata",
                    "dataType": ["text"],
                    "description": "Metadata stored as JSON string"
                }
            ]
        }

        try:
            self.client.schema.create_class(class_obj)
            logger.info(f"Created Weaviate class: {self.class_name}")
        except Exception as e:
            logger.error(f"Error creating Weaviate class: {str(e)}")
            raise

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to Weaviate.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        # Convert metadata to JSON string
        metadata_str = json.dumps(metadata or {})

        # Prepare data object
        data_object = {
            "metadata": metadata_str
        }

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.data_object.create(
                data_object=data_object,
                class_name=self.class_name,
                uuid=id,
                vector=vector
            )
        )
        logger.debug(f"Added vector {id} to Weaviate")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from Weaviate.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.data_object.get_by_id(
                    uuid=id,
                    class_name=self.class_name,
                    with_vector=True
                )
            )

            if response:
                # Parse metadata from JSON string
                metadata = {}
                if 'metadata' in response['properties']:
                    try:
                        metadata = json.loads(response['properties']['metadata'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse metadata for object {id}")

                return {
                    'id': id,
                    'vector': response.get('vector', []),
                    'metadata': metadata,
                    'score': 1.0  # Perfect match for direct retrieval
                }

        except Exception as e:
            logger.error(f"Error getting object from Weaviate: {str(e)}")

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector from Weaviate.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.data_object.delete(
                    uuid=id,
                    class_name=self.class_name
                )
            )
            logger.debug(f"Deleted vector {id} from Weaviate")
            return True
        except Exception as e:
            logger.error(f"Error deleting object from Weaviate: {str(e)}")
            return False

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Weaviate.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        try:
            # Convert filters to Weaviate where filter if provided
            where_filter = None
            if filters:
                where_filter = self._convert_filters(filters)

            # Prepare GraphQL query
            query = self.client.query.get(
                class_name=self.class_name,
                properties=["metadata"]
            ).with_additional(["id", "vector", "certainty"])

            # Add where filter if provided
            if where_filter:
                query = query.with_where(where_filter)

            # Add vector search
            query = query.with_near_vector({
                "vector": query_vector,
                "certainty": 0.7  # Minimum certainty threshold
            }).with_limit(limit)

            # Execute query
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: query.do()
            )

            results = []
            if response and 'data' in response and 'Get' in response['data'] and self.class_name in response['data']['Get']:
                for item in response['data']['Get'][self.class_name]:
                    # Parse metadata from JSON string
                    metadata = {}
                    if 'metadata' in item:
                        try:
                            metadata = json.loads(item['metadata'])
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse metadata for search result")

                    results.append({
                        'id': item['_additional']['id'],
                        'score': item['_additional'].get('certainty', 0.0),
                        'metadata': metadata,
                        'vector': item['_additional'].get('vector', [])
                    })

            return results

        except Exception as e:
            logger.error(f"Error searching in Weaviate: {str(e)}")
            return []

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
        try:
            # Convert filters to Weaviate where filter if provided
            where_filter = None
            if filters:
                where_filter = self._convert_filters(filters)

            # Prepare GraphQL query
            query = self.client.query.get(
                class_name=self.class_name,
                properties=["metadata"]
            ).with_additional(["id", "vector", "certainty"])

            # Add where filter if provided
            if where_filter:
                query = query.with_where(where_filter)

            # Add hybrid search
            hybrid_config = {
                "query": query_text,
                "alpha": alpha,
                "vector": query_vector
            }
            query = query.with_hybrid(hybrid_config).with_limit(limit)

            # Execute query
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: query.do()
            )

            results = []
            if response and 'data' in response and 'Get' in response['data'] and self.class_name in response['data']['Get']:
                for item in response['data']['Get'][self.class_name]:
                    # Parse metadata from JSON string
                    metadata = {}
                    if 'metadata' in item:
                        try:
                            metadata = json.loads(item['metadata'])
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse metadata for search result")

                    results.append({
                        'id': item['_additional']['id'],
                        'score': item['_additional'].get('certainty', 0.0),
                        'metadata': metadata,
                        'vector': item['_additional'].get('vector', [])
                    })

            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
            return await self.search(query_vector, limit, filters)

    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary filters to Weaviate where filter format.

        Args:
            filters: Dictionary of filters

        Returns:
            Weaviate where filter object
        """
        # Since we store metadata as a JSON string, we need to use text operators
        # This is a simplified implementation that works for basic use cases
        operands = []

        for key, value in filters.items():
            # For each filter, create a text contains operator
            if isinstance(value, str):
                # For string values
                operands.append({
                    "path": ["metadata"],
                    "operator": "Like",
                    "valueText": f"*\"{key}\"\\s*:\\s*\"{value}\"*"
                })
            elif isinstance(value, (int, float, bool)):
                # For numeric and boolean values
                operands.append({
                    "path": ["metadata"],
                    "operator": "Like",
                    "valueText": f"*\"{key}\"\\s*:\\s*{value}*"
                })
            elif isinstance(value, list):
                # For list values, check if any item in the list matches
                list_operands = []
                for item in value:
                    if isinstance(item, str):
                        list_operands.append({
                            "path": ["metadata"],
                            "operator": "Like",
                            "valueText": f"*\"{key}\"\\s*:\\s*\"{item}\"*"
                        })
                    else:
                        list_operands.append({
                            "path": ["metadata"],
                            "operator": "Like",
                            "valueText": f"*\"{key}\"\\s*:\\s*{item}*"
                        })
                if list_operands:
                    operands.append({
                        "operator": "Or",
                        "operands": list_operands
                    })

        # Combine all operands with AND
        if len(operands) > 1:
            return {
                "operator": "And",
                "operands": operands
            }
        elif len(operands) == 1:
            return operands[0]
        else:
            return {}

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new class in Weaviate.

        Args:
            name: The name of the class
            dimension: The dimension of the vectors
            **kwargs: Additional parameters for class creation

        Returns:
            True if the class was created, False otherwise
        """
        try:
            # Check if class already exists
            if self._class_exists(name):
                logger.warning(f"Class {name} already exists in Weaviate")
                return False

            # Create class object
            class_obj = {
                "class": name,
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Metadata stored as JSON string"
                    }
                ]
            }

            # Create the class
            self.client.schema.create_class(class_obj)
            logger.info(f"Created Weaviate class: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create Weaviate class {name}: {str(e)}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if a class exists in Weaviate.

        Args:
            name: The name of the class

        Returns:
            True if the class exists, False otherwise
        """
        return self._class_exists(name)
