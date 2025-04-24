import numpy as np
import hashlib
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio

from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse
from agentor.llm_gateway.utils.llm_metrics import (
    LLM_CACHE_HITS, LLM_CACHE_MISSES, LLM_CACHE_HIT_RATIO, 
    track_llm_cache as metrics_track_llm_cache
)
from agentor.llm_gateway.utils.cache_metrics import (
    CacheMetricsManager, time_cache_lookup, EvictionReason, CacheType
)

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache for LLM responses with semantic deduplication.
    
    This cache stores LLM responses and allows retrieving them by semantic
    similarity rather than exact matching. This enables more efficient use
    of LLM resources by avoiding redundant similar requests.
    """
    
    def __init__(
        self, 
        cache_size: int = 1000, 
        threshold: float = 0.92, 
        ttl: int = 3600,
        provider: str = "generic",
        model: str = "unknown",
        enable_metrics: bool = True
    ):
        """Initialize the semantic cache.
        
        Args:
            cache_size: Maximum number of items to store in the cache
            threshold: The similarity threshold for considering two prompts as similar (0.0-1.0)
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
            provider: The name of the LLM provider for metrics reporting
            model: The model name for metrics reporting
            enable_metrics: Whether to enable detailed metrics collection
            
        Raises:
            ValueError: If threshold is not between 0 and 1, or if cache_size is not positive
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")
        if cache_size <= 0:
            raise ValueError(f"Cache size must be positive, got {cache_size}")
            
        self.threshold = threshold
        self.ttl = ttl
        self.cache_size = cache_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.lock = asyncio.Lock()
        self.provider = provider
        self.model = model
        
        # LRU tracking
        self.access_times: Dict[str, float] = {}
        
        # Initialize metrics manager
        self.enable_metrics = enable_metrics
        if enable_metrics:
            self.metrics = CacheMetricsManager(
                cache_type=CacheType.SEMANTIC.value,
                provider=provider,
                model=model,
                auto_update_interval=60  # Update metrics every minute
            )
            self.metrics.attach_to_cache(self)
        else:
            self.metrics = None
    
    def _generate_key(self, request: LLMRequest) -> str:
        """Generate a cache key for a request.
        
        Args:
            request: The LLM request
            
        Returns:
            A cache key as a hexadecimal hash string
        """
        # Create a dictionary with the relevant fields
        key_dict = {
            "prompt": request.prompt,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stop_sequences": request.stop_sequences
        }
        
        # Convert to a stable string representation
        key_str = json.dumps(key_dict, sort_keys=True)
        
        # Hash the string to create a fixed-length key
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.
        
        Args:
            a: The first vector
            b: The second vector
            
        Returns:
            The cosine similarity (between -1.0 and 1.0)
            
        Raises:
            ValueError: If the vectors have different lengths or are empty
        """
        if not a or not b:
            raise ValueError("Vectors cannot be empty")
        if len(a) != len(b):
            raise ValueError(f"Vectors must have the same length, got {len(a)} and {len(b)}")
            
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            similarity = np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
            
            # Track similarity score if metrics are enabled
            if self.enable_metrics and self.metrics:
                self.metrics.track_similarity(similarity)
                
            return similarity
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text.
        
        This method should be implemented to use a real embedding model.
        For now, we'll use a simple hash-based approach for demonstration.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding as a normalized vector
            
        Raises:
            ValueError: If the text is empty
        """
        if not text:
            raise ValueError("Cannot embed empty text")
            
        try:
            # This is a placeholder. In a real implementation, you would use
            # a proper embedding model like OpenAI's text-embedding-ada-002
            # or a local model like sentence-transformers.
            
            # For demonstration, we'll use a simple hash-based approach
            hash_value = hashlib.md5(text.encode()).hexdigest()
            
            # Convert the hash to a list of floats
            embedding = []
            for i in range(0, len(hash_value), 2):
                if i + 2 <= len(hash_value):
                    value = int(hash_value[i:i+2], 16) / 255.0
                    embedding.append(value)
            
            # Normalize the embedding
            embedding_np = np.array(embedding)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm
            
            return embedding_np.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise ValueError(f"Failed to generate embedding: {e}") from e
    
    @time_cache_lookup(cache_type=CacheType.SEMANTIC.value)
    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get a cached response for a request.
        
        First tries an exact match, then falls back to semantic matching.
        
        Args:
            request: The LLM request
            
        Returns:
            The cached response, or None if not found
        """
        try:
            # First, try an exact match
            exact_key = self._generate_key(request)
            async with self.lock:
                if exact_key in self.cache:
                    entry = self.cache[exact_key]
                    current_time = time.time()
                    
                    if current_time < entry["expiry"]:
                        logger.info(f"Exact cache hit for key: {exact_key[:8]}...")
                        self.access_times[exact_key] = current_time  # Update LRU tracking
                        
                        # Track metrics
                        if self.enable_metrics and self.metrics:
                            ttl_remaining = entry["expiry"] - current_time
                            self.metrics.track_ttl_remaining(ttl_remaining)
                            
                        return entry["response"]
                    else:
                        # Remove expired entry
                        self._remove_entry(exact_key)
                        if self.enable_metrics and self.metrics:
                            self.metrics.track_eviction(EvictionReason.EXPIRED)
            
            # If no exact match, try semantic matching
            query_embedding = await self.get_embedding(request.prompt)
            
            async with self.lock:
                # Find the most similar cached prompt
                best_match = None
                best_similarity = 0.0
                
                for key, embedding in self.embeddings.items():
                    if key in self.cache and time.time() < self.cache[key]["expiry"]:
                        try:
                            similarity = self._cosine_similarity(query_embedding, embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = key
                        except Exception as e:
                            logger.warning(f"Error comparing embeddings: {e}")
                            continue
                
                # If we found a match above the threshold, return it
                if best_match and best_similarity >= self.threshold:
                    logger.info(f"Semantic cache hit for key: {best_match[:8]}... (similarity: {best_similarity:.4f})")
                    current_time = time.time()
                    self.access_times[best_match] = current_time  # Update LRU tracking
                    
                    # Track metrics
                    if self.enable_metrics and self.metrics:
                        ttl_remaining = self.cache[best_match]["expiry"] - current_time
                        self.metrics.track_ttl_remaining(ttl_remaining)
                        
                    return self.cache[best_match]["response"]
            
            logger.info(f"Cache miss for prompt: {request.prompt[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    async def set(self, request: LLMRequest, response: LLMResponse) -> None:
        """Cache a response for a request.
        
        Args:
            request: The LLM request
            response: The LLM response
        """
        try:
            key = self._generate_key(request)
            expiry = time.time() + self.ttl
            
            # Get the embedding for the prompt
            embedding = await self.get_embedding(request.prompt)
            
            async with self.lock:
                # Check if we need to evict entries to maintain cache size
                eviction_count = await self._enforce_cache_size()
                
                # Track evictions
                if eviction_count > 0 and self.enable_metrics and self.metrics:
                    self.metrics.track_eviction(EvictionReason.LRU, eviction_count)
                
                # Store the response and embedding
                self.cache[key] = {
                    "response": response,
                    "expiry": expiry
                }
                self.embeddings[key] = embedding
                self.access_times[key] = time.time()  # Update LRU tracking
                
                # Update metrics
                if self.enable_metrics and self.metrics:
                    self.metrics._update_size_metrics()
            
            logger.info(f"Cached response for key: {key[:8]}...")
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache.
        
        Args:
            key: The key to remove
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.embeddings:
            del self.embeddings[key]
        if key in self.access_times:
            del self.access_times[key]
    
    async def _enforce_cache_size(self) -> int:
        """Enforce the cache size limit by removing least recently used items.
        
        Returns:
            The number of entries evicted
        """
        eviction_count = 0
        while len(self.cache) >= self.cache_size:
            # Find the least recently used key
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_entry(lru_key)
            logger.debug(f"Removed LRU cache entry: {lru_key[:8]}...")
            eviction_count += 1
            
        return eviction_count
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            The number of expired entries cleared
        """
        start_time = time.time()
        cleared_count = 0
        
        try:
            now = time.time()
            async with self.lock:
                # Find expired keys
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if now >= entry["expiry"]
                ]
                
                # Remove expired entries
                for key in expired_keys:
                    self._remove_entry(key)
                
                cleared_count = len(expired_keys)
                if cleared_count > 0:
                    logger.info(f"Cleared {cleared_count} expired cache entries")
                    
                    # Track evictions
                    if self.enable_metrics and self.metrics:
                        self.metrics.track_eviction(EvictionReason.EXPIRED, cleared_count)
                        self.metrics._update_size_metrics()
            
            # Track cleaning time
            if self.enable_metrics and self.metrics:
                duration = time.time() - start_time
                self.metrics.track_cleaning_time(duration)
                
            return cleared_count
        except Exception as e:
            logger.error(f"Error clearing expired cache entries: {e}")
            return 0
            
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            # Stop metrics tracking first
            if self.enable_metrics and self.metrics and self.metrics._running:
                self.metrics.stop_auto_updates()
                
            async with self.lock:
                prev_size = len(self.cache)
                self.cache.clear()
                self.embeddings.clear()
                self.access_times.clear()
                
                # Track evictions
                if prev_size > 0 and self.enable_metrics and self.metrics:
                    self.metrics.track_eviction(EvictionReason.MANUAL, prev_size)
                    self.metrics._update_size_metrics()
                    
                logger.info(f"Cleared all cache entries ({prev_size} entries)")
                
            # Restart metrics tracking
            if self.enable_metrics and self.metrics and not self.metrics._running:
                self.metrics.start_auto_updates()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of cache metrics.
        
        Returns:
            A dictionary containing the metrics summary
        """
        if self.enable_metrics and self.metrics:
            return self.metrics.get_summary()
        else:
            return {
                "cache_type": CacheType.SEMANTIC.value,
                "provider": self.provider,
                "model": self.model,
                "size": len(self.cache),
                "capacity": self.cache_size,
                "usage_ratio": len(self.cache) / self.cache_size if self.cache_size > 0 else 0,
                "metrics_enabled": False
            }


class SemanticCachedLLM:
    """A wrapper around an LLM that adds semantic caching.
    
    This class wraps any LLM implementation and adds semantic caching capabilities,
    allowing cache hits based on semantic similarity rather than exact matching.
    This can significantly reduce API costs and response times.
    """
    
    def __init__(self, llm, cache: SemanticCache, similarity_threshold: float = 0.8,
                provider: str = "generic", model: str = "unknown"):
        """Initialize the semantically cached LLM.
        
        Args:
            llm: The LLM to wrap
            cache: The semantic cache to use
            similarity_threshold: The similarity threshold for cache hits (0.0-1.0)
            provider: The LLM provider name for metrics
            model: The model name for metrics
            
        Raises:
            ValueError: If similarity_threshold is not between 0 and 1
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {similarity_threshold}")
            
        self.llm = llm
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.provider = provider
        self.model = model
    
    @metrics_track_llm_cache("semantic")
    async def get_from_cache(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get a response from the cache.
        
        Args:
            request: The request to check in the cache
            
        Returns:
            The cached response, or None if not found
        """
        try:
            return await self.cache.get(request)
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response, using the cache if possible.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
            
        Raises:
            Exception: If the LLM generation fails
        """
        try:
            # Check the cache first
            cached_response = await self.get_from_cache(request)
            if cached_response:
                return cached_response
            
            # Generate a new response
            response = await self.llm.generate(request)
            
            # Cache the response
            await self.add_to_cache(request, response)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding as a normalized vector
            
        Raises:
            ValueError: If the text cannot be embedded
        """
        try:
            return await self.cache.get_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise ValueError(f"Failed to get embedding: {e}") from e
    
    async def find_similar(
        self, 
        embedding: List[float], 
        threshold: Optional[float] = None
    ) -> Optional[LLMResponse]:
        """Find a similar response in the cache based on embedding similarity.
        
        Args:
            embedding: The embedding to compare against
            threshold: The similarity threshold (overrides the instance default if provided)
            
        Returns:
            The most similar cached response, or None if none are similar enough
            
        Raises:
            ValueError: If threshold is provided and not between 0 and 1
        """
        if threshold is None:
            threshold = self.similarity_threshold
        elif not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")
        
        try:
            # Find the most similar cached response
            best_match = None
            best_similarity = 0.0
            
            async with self.cache.lock:
                for key, cached_embedding in self.cache.embeddings.items():
                    if key in self.cache.cache and time.time() < self.cache.cache[key]["expiry"]:
                        try:
                            similarity = self.cache._cosine_similarity(embedding, cached_embedding)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = key
                        except Exception as e:
                            logger.warning(f"Error comparing embeddings: {e}")
                            continue
                
                # If we found a match above the threshold, return it
                if best_match and best_similarity >= threshold:
                    logger.info(f"Semantic similarity match for key: {best_match[:8]}... (similarity: {best_similarity:.4f})")
                    self.cache.access_times[best_match] = time.time()  # Update LRU tracking
                    return self.cache.cache[best_match]["response"]
            
            return None
        except Exception as e:
            logger.error(f"Error finding similar response: {e}")
            return None
    
    async def clear_cache(self) -> None:
        """Clear the cache."""
        try:
            await self.cache.clear()
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def add_to_cache(self, request: LLMRequest, response: LLMResponse) -> None:
        """Add a response to the cache.
        
        Args:
            request: The request to cache
            response: The response to cache
        """
        try:
            await self.cache.set(request, response)
        except Exception as e:
            logger.error(f"Error adding to cache: {e}")
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current cache metrics.
        
        Returns:
            A dictionary containing metrics about the cache
        """
        async def _get_metrics_async():
            hit_count = LLM_CACHE_HITS._value.get()
            miss_count = LLM_CACHE_MISSES._value.get()
            total = hit_count + miss_count
            ratio = hit_count / total if total > 0 else 0
            
            return {
                "cache_size": len(self.cache.cache),
                "cache_capacity": self.cache.cache_size,
                "cache_hits": hit_count,
                "cache_misses": miss_count,
                "cache_hit_ratio": ratio,
                "threshold": self.similarity_threshold
            }
            
        # Create an event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, use a task
                return asyncio.create_task(_get_metrics_async())
            else:
                # Otherwise, run it in the loop directly
                return loop.run_until_complete(_get_metrics_async())
        except RuntimeError:
            # No event loop, create one temporarily
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(_get_metrics_async())
            loop.close()
            return result
