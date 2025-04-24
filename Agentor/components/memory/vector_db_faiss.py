"""
FAISS Vector Database Provider for the Agentor framework.

This module provides integration with FAISS (Facebook AI Similarity Search),
a library for efficient similarity search and clustering of dense vectors.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import asyncio
import logging
import time
import json
import os
import numpy as np
from abc import ABC, abstractmethod

from agentor.components.memory.vector_db import VectorDBProvider

logger = logging.getLogger(__name__)


class FAISSProvider(VectorDBProvider):
    """Vector database provider using FAISS."""

    def __init__(
        self,
        index_name: str,
        dimension: int = 1536,
        index_type: str = "IndexFlatL2",
        save_path: Optional[str] = None,
        load_on_init: bool = True,
        auto_save: bool = True
    ):
        """Initialize the FAISS provider.

        Args:
            index_name: Name of the FAISS index
            dimension: Dimension of the vectors
            index_type: Type of FAISS index (IndexFlatL2, IndexHNSWFlat, etc.)
            save_path: Path to save the index (optional)
            load_on_init: Whether to load the index on initialization
            auto_save: Whether to automatically save the index after modifications
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS package is required for FAISSProvider. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

        self.index_name = index_name
        self.dimension = dimension
        self.index_type = index_type
        self.save_path = save_path
        self.auto_save = auto_save

        # Import here to avoid module-level import issues
        import faiss
        self.faiss = faiss

        # Initialize storage for metadata
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.next_index = 0
        self.lock = asyncio.Lock()

        # Create or load the index
        self._create_index()
        if load_on_init and save_path and os.path.exists(self._get_index_path()):
            self._load_index()

    def _create_index(self):
        """Create a new FAISS index."""
        if self.index_type == "IndexFlatL2":
            self.index = self.faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexHNSWFlat":
            self.index = self.faiss.IndexHNSWFlat(self.dimension, 32)
        elif self.index_type == "IndexIVFFlat":
            # For IVF indexes, we need a quantizer
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            # IVF indexes need to be trained before use
            if self.index.is_trained == False and self.next_index > 0:
                # We need at least one vector to train
                vectors = np.zeros((1, self.dimension), dtype=np.float32)
                self.index.train(vectors)
        else:
            # Default to IndexFlatL2 for unknown types
            logger.warning(f"Unknown index type {self.index_type}, using IndexFlatL2")
            self.index = self.faiss.IndexFlatL2(self.dimension)

        logger.info(f"Created FAISS index: {self.index_name} ({self.index_type})")

    def _get_index_path(self) -> str:
        """Get the path to the index file.

        Returns:
            The path to the index file
        """
        if not self.save_path:
            return f"{self.index_name}.faiss"
        return os.path.join(self.save_path, f"{self.index_name}.faiss")

    def _get_metadata_path(self) -> str:
        """Get the path to the metadata file.

        Returns:
            The path to the metadata file
        """
        if not self.save_path:
            return f"{self.index_name}.metadata.json"
        return os.path.join(self.save_path, f"{self.index_name}.metadata.json")

    def _save_index(self):
        """Save the index and metadata to disk."""
        if not self.save_path:
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)

            # Save the index
            self.faiss.write_index(self.index, self._get_index_path())

            # Save the metadata
            with open(self._get_metadata_path(), 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index,
                    'next_index': self.next_index
                }, f)

            logger.debug(f"Saved FAISS index to {self._get_index_path()}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")

    def _load_index(self):
        """Load the index and metadata from disk."""
        try:
            # Load the index
            self.index = self.faiss.read_index(self._get_index_path())

            # Load the metadata
            with open(self._get_metadata_path(), 'r') as f:
                data = json.load(f)
                self.metadata = data['metadata']
                self.id_to_index = {k: int(v) for k, v in data['id_to_index'].items()}
                self.next_index = data['next_index']

            logger.debug(f"Loaded FAISS index from {self._get_index_path()}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            # Create a new index if loading fails
            self._create_index()

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """Add a vector to FAISS.

        Args:
            id: The unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        async with self.lock:
            # Convert vector to numpy array
            vector_np = np.array([vector], dtype=np.float32)

            # Check if ID already exists
            if id in self.id_to_index:
                # Update existing vector
                idx = self.id_to_index[id]
                # FAISS doesn't support updating vectors directly, so we need to remove and re-add
                # This is a limitation of FAISS, and in a production system you might want to
                # use a different approach or a different vector database
                logger.warning(f"Updating vectors in FAISS is not efficient, consider using a different vector database for frequent updates")
                # For now, we'll just add a new vector and update the mapping
                self.index.add(vector_np)
                self.id_to_index[id] = self.next_index
                self.next_index += 1
            else:
                # Add new vector
                self.index.add(vector_np)
                self.id_to_index[id] = self.next_index
                self.next_index += 1

            # Store metadata
            self.metadata[id] = metadata or {}

            # Save index if auto_save is enabled
            if self.auto_save:
                self._save_index()

            logger.debug(f"Added vector {id} to FAISS")

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from FAISS.

        Args:
            id: The unique identifier for the vector

        Returns:
            A dictionary containing the vector and metadata, or None if not found
        """
        async with self.lock:
            if id not in self.id_to_index:
                return None

            # Get the index of the vector
            idx = self.id_to_index[id]

            # FAISS doesn't provide a direct way to get a vector by index
            # We need to reconstruct it from the index
            # This is a limitation of FAISS, and in a production system you might want to
            # use a different approach or a different vector database
            # For now, we'll just return the metadata without the vector
            return {
                'id': id,
                'metadata': self.metadata.get(id, {}),
                'score': 1.0  # Perfect match for direct retrieval
            }

    async def delete(self, id: str) -> bool:
        """Delete a vector from FAISS.

        Args:
            id: The unique identifier for the vector

        Returns:
            True if the vector was deleted, False otherwise
        """
        async with self.lock:
            if id not in self.id_to_index:
                return False

            # FAISS doesn't support deleting vectors directly
            # This is a limitation of FAISS, and in a production system you might want to
            # use a different approach or a different vector database
            logger.warning(f"Deleting vectors in FAISS is not supported, marking as deleted in metadata")

            # Mark as deleted in metadata
            if id in self.metadata:
                del self.metadata[id]

            # Remove from ID to index mapping
            del self.id_to_index[id]

            # Save index if auto_save is enabled
            if self.auto_save:
                self._save_index()

            logger.debug(f"Marked vector {id} as deleted in FAISS")
            return True

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in FAISS.

        Args:
            query_vector: The query vector
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            A list of dictionaries containing the vector, metadata, and similarity score
        """
        async with self.lock:
            # Convert query vector to numpy array
            query_np = np.array([query_vector], dtype=np.float32)

            # Perform the search
            distances, indices = self.index.search(query_np, limit * 10)  # Get more results for filtering

            # Convert results to list of dictionaries
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                # Find the ID for this index
                id_found = None
                for id, index in self.id_to_index.items():
                    if index == idx:
                        id_found = id
                        break

                if id_found is None:
                    continue

                # Get metadata for this ID
                metadata = self.metadata.get(id_found, {})

                # Apply filters if provided
                if filters and not self._matches_filters(metadata, filters):
                    continue

                # Convert distance to similarity score (FAISS returns L2 distance)
                # Lower distance means higher similarity
                # We'll convert to a 0-1 scale where 1 is most similar
                # This is a simple conversion and might need adjustment based on your data
                similarity = 1.0 / (1.0 + distance)

                results.append({
                    'id': id_found,
                    'score': similarity,
                    'metadata': metadata
                })

                # Stop once we have enough results
                if len(results) >= limit:
                    break

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
        # FAISS doesn't have built-in hybrid search, so we'll implement a basic version
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

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters.

        Args:
            metadata: The metadata to check
            filters: The filters to apply

        Returns:
            True if the metadata matches the filters, False otherwise
        """
        for key, value in filters.items():
            if key not in metadata:
                return False

            if isinstance(value, list):
                # For list values, check if any item in the list matches
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False

        return True

    async def create_collection(self, name: str, dimension: int, **kwargs) -> bool:
        """Create a new FAISS index.

        Args:
            name: The name of the index
            dimension: The dimension of the vectors
            **kwargs: Additional parameters for index creation

        Returns:
            True if the index was created, False otherwise
        """
        try:
            # Get index type
            index_type = kwargs.get('index_type', 'IndexFlatL2')

            # Create a new provider with the specified parameters
            provider = FAISSProvider(
                index_name=name,
                dimension=dimension,
                index_type=index_type,
                save_path=self.save_path,
                load_on_init=False,
                auto_save=self.auto_save
            )

            # Save the index
            if self.auto_save:
                provider._save_index()

            logger.info(f"Created FAISS index: {name} ({index_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to create FAISS index {name}: {str(e)}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if a FAISS index exists.

        Args:
            name: The name of the index

        Returns:
            True if the index exists, False otherwise
        """
        if not self.save_path:
            return False

        index_path = os.path.join(self.save_path, f"{name}.faiss")
        metadata_path = os.path.join(self.save_path, f"{name}.metadata.json")

        return os.path.exists(index_path) and os.path.exists(metadata_path)
