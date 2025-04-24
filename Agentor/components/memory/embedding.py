"""
Embedding Providers for Memory Systems in the Agentor framework.

This module provides embedding providers for converting text to vector embeddings
for use in semantic search and memory operations.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import asyncio
import logging
import time
import hashlib
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple text strings.
        
        Args:
            texts: The texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's embedding API."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-ada-002",
        cache_size: int = 1000,
        batch_size: int = 16
    ):
        """Initialize the OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            model: The embedding model to use
            cache_size: Maximum number of embeddings to cache
            batch_size: Maximum number of texts to embed in a single API call
        """
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package is required for OpenAIEmbeddingProvider")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cache: Dict[str, List[float]] = {}
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.lock = asyncio.Lock()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        
        async with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Get embedding from API
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            await self._cache_embedding(cache_key, embedding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default dimension for OpenAI embeddings
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple text strings.
        
        Args:
            texts: The texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache for all texts
        cache_keys = [self._get_cache_key(text) for text in texts]
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Get cached embeddings
        async with self.lock:
            for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        
        # If all embeddings were cached, return them
        if not texts_to_embed:
            return embeddings
        
        # Process in batches
        all_new_embeddings = []
        
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch_texts = texts_to_embed[i:i+self.batch_size]
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_new_embeddings.extend(batch_embeddings)
                
                # Cache the results
                for text, embedding in zip(batch_texts, batch_embeddings):
                    cache_key = self._get_cache_key(text)
                    await self._cache_embedding(cache_key, embedding)
            
            except Exception as e:
                logger.error(f"Error getting batch embeddings: {str(e)}")
                # Return zero vectors as fallback
                zero_vector = [0.0] * 1536  # Default dimension for OpenAI embeddings
                all_new_embeddings.extend([zero_vector] * len(batch_texts))
        
        # Merge cached and new embeddings
        result = [None] * len(texts)
        for i, embedding in zip(indices_to_embed, all_new_embeddings):
            result[i] = embedding
        
        for i, embedding in enumerate(embeddings):
            if result[i] is None:
                result[i] = embedding
        
        return result
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            A cache key
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def _cache_embedding(self, key: str, embedding: List[float]):
        """Cache an embedding.
        
        Args:
            key: The cache key
            embedding: The embedding to cache
        """
        async with self.lock:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.cache_size:
                # Simple LRU: remove a random entry
                # In a real implementation, this would use a proper LRU cache
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[key] = embedding


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Hugging Face models."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = 1000,
        device: str = "cpu"
    ):
        """Initialize the Hugging Face embedding provider.
        
        Args:
            model_name: The model to use for embeddings
            cache_size: Maximum number of embeddings to cache
            device: Device to run the model on ('cpu' or 'cuda')
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package is required for HuggingFaceEmbeddingProvider")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: Dict[str, List[float]] = {}
        self.cache_size = cache_size
        self.lock = asyncio.Lock()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        
        async with self.lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Get embedding from model
        try:
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(text).tolist()
            )
            
            # Cache the result
            await self._cache_embedding(cache_key, embedding)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * self.model.get_sentence_embedding_dimension()
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple text strings.
        
        Args:
            texts: The texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache for all texts
        cache_keys = [self._get_cache_key(text) for text in texts]
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Get cached embeddings
        async with self.lock:
            for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        
        # If all embeddings were cached, return them
        if not texts_to_embed:
            return embeddings
        
        # Get embeddings from model
        try:
            # Run in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(texts_to_embed).tolist()
            )
            
            # Cache the results
            for text, embedding in zip(texts_to_embed, new_embeddings):
                cache_key = self._get_cache_key(text)
                await self._cache_embedding(cache_key, embedding)
        
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {str(e)}")
            # Return zero vectors as fallback
            dim = self.model.get_sentence_embedding_dimension()
            new_embeddings = [[0.0] * dim for _ in range(len(texts_to_embed))]
        
        # Merge cached and new embeddings
        result = [None] * len(texts)
        for i, embedding in zip(indices_to_embed, new_embeddings):
            result[i] = embedding
        
        for i, embedding in enumerate(embeddings):
            if result[i] is None:
                result[i] = embedding
        
        return result
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text string.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            A cache key
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def _cache_embedding(self, key: str, embedding: List[float]):
        """Cache an embedding.
        
        Args:
            key: The cache key
            embedding: The embedding to cache
        """
        async with self.lock:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.cache_size:
                # Simple LRU: remove a random entry
                # In a real implementation, this would use a proper LRU cache
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[key] = embedding


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        """Initialize the mock embedding provider.
        
        Args:
            dimension: The dimension of the embeddings
        """
        self.dimension = dimension
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get a deterministic mock embedding for a text string.
        
        Args:
            text: The text to embed
            
        Returns:
            A deterministic embedding vector based on the text
        """
        # Generate a deterministic embedding based on the text
        hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        random_state = np.random.RandomState(hash_value)
        
        # Generate a random vector
        vector = random_state.randn(self.dimension)
        
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get mock embeddings for multiple text strings.
        
        Args:
            texts: The texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
