"""
ChromaDB Vector Database Provider for the Agentor framework.

This module provides integration with ChromaDB, an open-source embedding database
designed for document embeddings and retrieval.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
import time
import json
import os
from abc import ABC, abstractmethod

from agentor.components.memory.vector_db import VectorDBProvider

logger = logging.getLogger(__name__)


class ChromaDBProvider(VectorDBProvider):
    """Vector database provider using ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        client_settings: Optional[Dict[str, Any]] = None,
        embedding_function=None,
        create_collection_if_not_exists: bool = True
    ):
        """Initialize the ChromaDB provider.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (optional)
            client_settings: Settings for the ChromaDB client (optional)
            embedding_function: Function to generate embeddings (optional)
            create_collection_if_not_exists: Whether to create the collection if it doesn't exist
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaDB package is required for ChromaDBProvider. Install with 'pip install chromadb'")

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client_settings = client_settings or {}
        self.embedding_function = embedding_function
        self.create_collection_if_not_exists = create_collection_if_not_exists

        # Import here to avoid module-level import issues
        import chromadb
        self.chromadb = chromadb

        # Connect to ChromaDB
        self._connect()

        # Get or create the collection
        self._get_or_create_collection()

        logger.info(f"Connected to ChromaDB collection: {collection_name}")

    def _connect(self):
        """Connect to ChromaDB."""
        if self.persist_directory:
            self.client = self.chromadb.PersistentClient(
                path=self.persist_directory,
                settings=self.client_settings
            )
        else:
            self.client = self.chromadb.Client(settings=self.client_settings)
        logger.debug(f"Connected to ChromaDB")

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        try:
            # Try to get the collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.debug(f"Found existing ChromaDB collection: {self.collection_name}")
        except Exception as e:
            if self.create_collection_if_not_exists:
                # Create the collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Created ChromaDB collection: {self.collection_name}")
            else:
                logger.error(f"ChromaDB collection {self.collection_name} does not exist and create_collection_if_not_exists is False")
                raise

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to ChromaDB.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        # ChromaDB requires documents when adding embeddings
        # If no text is provided in metadata, we'll use a placeholder
        document = ""
        if metadata:
            if 'text' in metadata:
                document = metadata['text']
            elif 'content' in metadata and isinstance(metadata['content'], dict) and 'text' in metadata['content']:
                document = metadata['content']['text']

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.collection.add(
                ids=[id],
                embeddings=[vector],
                metadatas=[metadata or {}],
                documents=[document]
            )
        )
        logger.debug(f"Added vector {id} to ChromaDB")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from ChromaDB.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.collection.get(
                ids=[id],
                include=["embeddings", "metadatas", "documents"]
            )
        )

        if response and response['ids'] and len(response['ids']) > 0:
            idx = response['ids'].index(id) if id in response['ids'] else -1
            if idx >= 0:
                return {
                    'id': id,
                    'vector': response['embeddings'][idx] if 'embeddings' in response and idx < len(response['embeddings']) else [],
                    'metadata': response['metadatas'][idx] if 'metadatas' in response and idx < len(response['metadatas']) else {},
                    'score': 1.0  # Perfect match for direct retrieval
                }

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector from ChromaDB.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.delete(ids=[id])
            )
            logger.debug(f"Deleted vector {id} from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector from ChromaDB: {str(e)}")
            return False

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        # Convert filters to ChromaDB where format if provided
        where = None
        if filters:
            where = self._convert_filters(filters)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where,
                include=["embeddings", "metadatas", "documents", "distances"]
            )
        )

        results = []
        if response and 'ids' in response and len(response['ids']) > 0:
            for i, id in enumerate(response['ids'][0]):
                # Calculate similarity score from distance
                # ChromaDB returns distance, not similarity
                # Lower distance means higher similarity
                distance = response['distances'][0][i] if 'distances' in response else 0.0
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

                results.append({
                    'id': id,
                    'score': similarity,
                    'metadata': response['metadatas'][0][i] if 'metadatas' in response else {},
                    'vector': response['embeddings'][0][i] if 'embeddings' in response else []
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
        try:
            # Convert filters to ChromaDB where format if provided
            where = None
            if filters:
                where = self._convert_filters(filters)

            # ChromaDB supports hybrid search natively
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_vector],
                    query_texts=[query_text],
                    n_results=limit,
                    where=where,
                    include=["embeddings", "metadatas", "documents", "distances"]
                )
            )

            results = []
            if response and 'ids' in response and len(response['ids']) > 0:
                for i, id in enumerate(response['ids'][0]):
                    # Calculate similarity score from distance
                    distance = response['distances'][0][i] if 'distances' in response else 0.0
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

                    results.append({
                        'id': id,
                        'score': similarity,
                        'metadata': response['metadatas'][0][i] if 'metadatas' in response else {},
                        'vector': response['embeddings'][0][i] if 'embeddings' in response else []
                    })

            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector search: {str(e)}")
            return await self.search(query_vector, limit, filters)

    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary filters to ChromaDB where format.

        Args:
            filters: Dictionary of filters

        Returns:
            ChromaDB where filter object
        """
        # ChromaDB supports filtering directly on metadata fields
        where = {}

        for key, value in filters.items():
            # Handle different value types
            if isinstance(value, (str, int, float, bool)):
                # For simple values
                where[key] = value
            elif isinstance(value, list):
                # For list values, use $in operator
                where[key] = {"$in": value}

        return where

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new collection in ChromaDB.

        Args:
            name: The name of the collection
            dimension: The dimension of the vectors (ignored for ChromaDB)
            **kwargs: Additional parameters for collection creation

        Returns:
            True if the collection was created, False otherwise
        """
        try:
            # Check if collection already exists
            try:
                self.client.get_collection(name=name)
                logger.warning(f"Collection {name} already exists in ChromaDB")
                return False
            except Exception:
                # Collection doesn't exist, create it
                embedding_function = kwargs.get('embedding_function', self.embedding_function)
                self.client.create_collection(
                    name=name,
                    embedding_function=embedding_function
                )
                logger.info(f"Created ChromaDB collection: {name}")
                return True

        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection {name}: {str(e)}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists in ChromaDB.

        Args:
            name: The name of the collection

        Returns:
            True if the collection exists, False otherwise
        """
        try:
            self.client.get_collection(name=name)
            return True
        except Exception:
            return False
