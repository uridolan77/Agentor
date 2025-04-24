import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse
from agentor.llm_gateway.llm.cache import LLMCache, CachedLLM
from agentor.llm_gateway.llm.semantic_cache import SemanticCache, SemanticCachedLLM


@pytest.mark.asyncio
async def test_llm_cache():
    """Test the LLMCache class."""
    # Create a cache
    cache = LLMCache(ttl=60)
    
    # Create a request
    request = LLMRequest(
        prompt="Test prompt",
        model="test-model",
        temperature=0.7
    )
    
    # Create a response
    response = LLMResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    
    # Test cache miss
    cached_response = await cache.get(request)
    assert cached_response is None
    
    # Test cache set and get
    await cache.set(request, response)
    cached_response = await cache.get(request)
    assert cached_response is not None
    assert cached_response.text == "Test response"
    
    # Test cache with different request
    different_request = LLMRequest(
        prompt="Different prompt",
        model="test-model",
        temperature=0.7
    )
    cached_response = await cache.get(different_request)
    assert cached_response is None


@pytest.mark.asyncio
async def test_cached_llm():
    """Test the CachedLLM class."""
    # Create a mock LLM
    mock_llm = AsyncMock()
    mock_response = LLMResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    mock_llm.generate.return_value = mock_response
    
    # Create a cache
    cache = LLMCache(ttl=60)
    
    # Create a cached LLM
    cached_llm = CachedLLM(llm=mock_llm, cache=cache)
    
    # Create a request
    request = LLMRequest(
        prompt="Test prompt",
        model="test-model",
        temperature=0.7
    )
    
    # Test cache miss (should call the LLM)
    response1 = await cached_llm.generate(request)
    assert response1.text == "Test response"
    mock_llm.generate.assert_called_once_with(request)
    
    # Reset the mock
    mock_llm.generate.reset_mock()
    
    # Test cache hit (should not call the LLM)
    response2 = await cached_llm.generate(request)
    assert response2.text == "Test response"
    mock_llm.generate.assert_not_called()
    
    # Test different request (should call the LLM)
    request2 = LLMRequest(
        prompt="Different prompt",
        model="test-model",
        temperature=0.7
    )
    
    response3 = await cached_llm.generate(request2)
    assert response3.text == "Test response"
    mock_llm.generate.assert_called_once_with(request2)


@pytest.mark.asyncio
async def test_semantic_cache_initialization():
    """Test the SemanticCache initialization with various parameters."""
    # Test with default parameters
    cache = SemanticCache()
    assert cache.threshold == 0.92
    assert cache.ttl == 3600
    assert cache.cache_size == 1000
    
    # Test with custom parameters
    cache = SemanticCache(cache_size=500, threshold=0.8, ttl=1800)
    assert cache.threshold == 0.8
    assert cache.ttl == 1800
    assert cache.cache_size == 500
    
    # Test with invalid threshold
    with pytest.raises(ValueError):
        SemanticCache(threshold=1.5)
    
    # Test with invalid cache size
    with pytest.raises(ValueError):
        SemanticCache(cache_size=0)


@pytest.mark.asyncio
async def test_semantic_cache_exact_match():
    """Test the SemanticCache with exact match functionality."""
    cache = SemanticCache(ttl=60)
    
    # Create a request
    request = LLMRequest(
        prompt="Test prompt",
        model="test-model",
        temperature=0.7
    )
    
    # Create a response
    response = LLMResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    
    # Test cache miss
    cached_response = await cache.get(request)
    assert cached_response is None
    
    # Test cache set and get with exact match
    await cache.set(request, response)
    cached_response = await cache.get(request)
    assert cached_response is not None
    assert cached_response.text == "Test response"


@pytest.mark.asyncio
async def test_semantic_cache_cosine_similarity():
    """Test the cosine similarity calculation."""
    cache = SemanticCache()
    
    # Test with normal vectors
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cache._cosine_similarity(vec1, vec2)
    assert similarity == 1.0
    
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = cache._cosine_similarity(vec1, vec2)
    assert similarity == 0.0
    
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    similarity = cache._cosine_similarity(vec1, vec2)
    assert similarity == -1.0
    
    # Test with empty vectors - should raise ValueError
    with pytest.raises(ValueError):
        cache._cosine_similarity([], [1.0])
    
    # Test with different length vectors - should raise ValueError
    with pytest.raises(ValueError):
        cache._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


@pytest.mark.asyncio
async def test_semantic_cache_get_embedding():
    """Test the embedding generation."""
    cache = SemanticCache()
    
    # Test with normal text
    embedding = await cache.get_embedding("Test text")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # Test with empty text - should raise ValueError
    with pytest.raises(ValueError):
        await cache.get_embedding("")


@pytest.mark.asyncio
async def test_semantic_cache_lru_eviction():
    """Test the LRU eviction policy."""
    # Create a small cache with only 2 slots
    cache = SemanticCache(cache_size=2, ttl=60)
    
    # Create 3 different requests
    request1 = LLMRequest(prompt="Test prompt 1", model="test-model")
    request2 = LLMRequest(prompt="Test prompt 2", model="test-model")
    request3 = LLMRequest(prompt="Test prompt 3", model="test-model")
    
    response = LLMResponse(text="Test response", model="test-model")
    
    # Add 2 items to fill the cache
    await cache.set(request1, response)
    await cache.set(request2, response)
    
    # Verify both are in cache
    assert await cache.get(request1) is not None
    assert await cache.get(request2) is not None
    
    # Add a third item, which should evict the LRU item (request1)
    await cache.set(request3, response)
    
    # Verify request1 is evicted, but request2 and request3 are still in cache
    assert await cache.get(request1) is None
    assert await cache.get(request2) is not None
    assert await cache.get(request3) is not None


@pytest.mark.asyncio
async def test_semantic_cache_clear():
    """Test clearing the cache."""
    cache = SemanticCache()
    
    # Add some items to the cache
    request1 = LLMRequest(prompt="Test prompt 1", model="test-model")
    request2 = LLMRequest(prompt="Test prompt 2", model="test-model")
    response = LLMResponse(text="Test response", model="test-model")
    
    await cache.set(request1, response)
    await cache.set(request2, response)
    
    # Verify items are in cache
    assert await cache.get(request1) is not None
    assert await cache.get(request2) is not None
    
    # Clear the cache
    await cache.clear()
    
    # Verify cache is empty
    assert await cache.get(request1) is None
    assert await cache.get(request2) is None


@pytest.mark.asyncio
async def test_semantic_cached_llm():
    """Test the SemanticCachedLLM class."""
    # Create a mock LLM
    mock_llm = AsyncMock()
    mock_response = LLMResponse(
        text="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    mock_llm.generate.return_value = mock_response
    
    # Mock the embedding method to return controlled vectors for testing
    # semantic similarity matches
    original_get_embedding = SemanticCache.get_embedding
    
    async def mock_get_embedding(self, text):
        if text == "What is Python?":
            return [1.0, 0.0, 0.0]
        elif text == "Tell me about Python":
            return [0.95, 0.1, 0.0]  # Very similar to "What is Python?"
        elif text == "What is JavaScript?":
            return [0.0, 1.0, 0.0]  # Different from "What is Python?"
        else:
            return await original_get_embedding(self, text)
    
    # Create a semantic cache with a controlled threshold
    semantic_cache = SemanticCache(threshold=0.9)
    
    # Patch the get_embedding method
    with patch.object(SemanticCache, 'get_embedding', mock_get_embedding):
        # Create a cached LLM
        cached_llm = SemanticCachedLLM(llm=mock_llm, cache=semantic_cache)
        
        # Create requests
        request1 = LLMRequest(
            prompt="What is Python?",
            model="test-model",
            temperature=0.7
        )
        
        # Semantic similar request
        request2 = LLMRequest(
            prompt="Tell me about Python",
            model="test-model",
            temperature=0.7
        )
        
        # Different request
        request3 = LLMRequest(
            prompt="What is JavaScript?",
            model="test-model",
            temperature=0.7
        )
        
        # Test cache miss (should call the LLM)
        response1 = await cached_llm.generate(request1)
        assert response1.text == "Test response"
        mock_llm.generate.assert_called_once_with(request1)
        
        # Reset the mock
        mock_llm.generate.reset_mock()
        
        # Test semantic cache hit with similar prompt (should not call the LLM)
        response2 = await cached_llm.generate(request2)
        assert response2.text == "Test response"
        mock_llm.generate.assert_not_called()
        
        # Test cache miss with different prompt (should call the LLM)
        response3 = await cached_llm.generate(request3)
        assert response3.text == "Test response"
        mock_llm.generate.assert_called_once_with(request3)


@pytest.mark.asyncio
async def test_semantic_cached_llm_find_similar():
    """Test the find_similar method of SemanticCachedLLM."""
    # Create a mock LLM
    mock_llm = AsyncMock()
    
    # Create a semantic cache
    semantic_cache = SemanticCache()
    
    # Create a cached LLM
    cached_llm = SemanticCachedLLM(llm=mock_llm, cache=semantic_cache, similarity_threshold=0.8)
    
    # Add a response to the cache directly
    request = LLMRequest(
        prompt="What is Python?",
        model="test-model",
        temperature=0.7
    )
    
    response = LLMResponse(
        text="Python is a programming language.",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    
    # Mock the embedding method
    with patch.object(SemanticCache, 'get_embedding') as mock_get_embedding:
        # Define embeddings
        python_embedding = [1.0, 0.0, 0.0]
        mock_get_embedding.return_value = python_embedding
        
        # Set the cache entry
        await semantic_cache.set(request, response)
        
        # Find similar with the same embedding (should find a match)
        similar_response = await cached_llm.find_similar(python_embedding)
        assert similar_response is not None
        assert similar_response.text == "Python is a programming language."
        
        # Find similar with a different embedding (should not find a match)
        javascript_embedding = [0.0, 1.0, 0.0]
        similar_response = await cached_llm.find_similar(javascript_embedding)
        assert similar_response is None
        
        # Find similar with a similar embedding but below threshold (should not find a match)
        somewhat_similar_embedding = [0.7, 0.3, 0.0]
        similar_response = await cached_llm.find_similar(somewhat_similar_embedding)
        assert similar_response is None
        
        # Find similar with a similar embedding above threshold (should find a match)
        very_similar_embedding = [0.95, 0.05, 0.0]
        similar_response = await cached_llm.find_similar(very_similar_embedding)
        assert similar_response is not None
        assert similar_response.text == "Python is a programming language."
        
        # Test with custom threshold
        similar_response = await cached_llm.find_similar(somewhat_similar_embedding, threshold=0.7)
        assert similar_response is not None
        assert similar_response.text == "Python is a programming language."
