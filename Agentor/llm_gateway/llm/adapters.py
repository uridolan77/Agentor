"""
Adapters for existing LLM providers to use the new standardized interfaces.

This module provides adapter classes that wrap existing LLM implementations
to make them compatible with the new standardized interfaces.
"""

from typing import Dict, Any, List, Optional, AsyncIterator, Union
import logging
import asyncio

from agentor.llm_gateway.llm.base import BaseLLM, LLMRequest as OldLLMRequest, LLMResponse as OldLLMResponse
from agentor.llm_gateway.llm.providers.openai import OpenAILLM
from agentor.llm_gateway.llm.semantic_cache import SemanticCache, SemanticCachedLLM
from agentor.core.interfaces.llm import (
    ILLM, IStreamingLLM, ICachedLLM, ISemanticCachedLLM,
    LLMRequest, LLMResponse, StreamingLLMResponse
)
from agentor.core.plugin import Plugin
from agentor.core.registry import get_component_registry

logger = logging.getLogger(__name__)


def convert_request(request: LLMRequest) -> OldLLMRequest:
    """Convert a new LLMRequest to an old LLMRequest.
    
    Args:
        request: The new request to convert
        
    Returns:
        The converted old request
    """
    return OldLLMRequest(
        prompt=request.prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop_sequences=request.stop_sequences,
        metadata=request.metadata
    )


def convert_response(response: OldLLMResponse) -> LLMResponse:
    """Convert an old LLMResponse to a new LLMResponse.
    
    Args:
        response: The old response to convert
        
    Returns:
        The converted new response
    """
    return LLMResponse(
        text=response.text,
        model=response.model,
        usage=response.usage,
        metadata=response.metadata
    )


class LLMAdapter(ILLM):
    """Adapter for the BaseLLM class."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize the LLM adapter.
        
        Args:
            llm: The LLM implementation to adapt
        """
        self.llm = llm
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Returns:
            The response from the LLM
        """
        old_request = convert_request(request)
        old_response = await self.llm.generate(old_request)
        return convert_response(old_response)


class StreamingLLMAdapter(LLMAdapter, IStreamingLLM):
    """Adapter for streaming LLM implementations."""
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamingLLMResponse]:
        """Generate a streaming response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Yields:
            Streaming responses from the LLM
        """
        old_request = convert_request(request)
        
        # Check if the LLM supports streaming
        if hasattr(self.llm, 'generate_stream'):
            async for chunk in self.llm.generate_stream(old_request):
                yield StreamingLLMResponse(
                    text=chunk.text,
                    model=chunk.model,
                    is_finished=chunk.is_finished,
                    metadata=chunk.metadata
                )
        else:
            # Fallback to non-streaming
            response = await self.generate(request)
            yield StreamingLLMResponse(
                text=response.text,
                model=response.model,
                is_finished=True,
                metadata=response.metadata
            )


class CachedLLMAdapter(LLMAdapter, ICachedLLM):
    """Adapter for cached LLM implementations."""
    
    async def get_from_cache(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get a response from the cache.
        
        Args:
            request: The request to check in the cache
            
        Returns:
            The cached response, or None if not found
        """
        if hasattr(self.llm, 'get_from_cache'):
            old_request = convert_request(request)
            old_response = await self.llm.get_from_cache(old_request)
            
            if old_response:
                return convert_response(old_response)
        
        return None
    
    async def add_to_cache(self, request: LLMRequest, response: LLMResponse) -> None:
        """Add a response to the cache.
        
        Args:
            request: The request to cache
            response: The response to cache
        """
        if hasattr(self.llm, 'add_to_cache'):
            old_request = convert_request(request)
            old_response = OldLLMResponse(
                text=response.text,
                model=response.model,
                usage=response.usage,
                metadata=response.metadata
            )
            
            await self.llm.add_to_cache(old_request, old_response)
    
    async def clear_cache(self) -> None:
        """Clear the cache."""
        if hasattr(self.llm, 'clear_cache'):
            await self.llm.clear_cache()


class SemanticCachedLLMAdapter(CachedLLMAdapter, ISemanticCachedLLM):
    """Adapter for semantically cached LLM implementations.
    
    This adapter extends the basic caching functionality by adding semantic
    similarity search capabilities, allowing cache hits even when prompts
    are different but semantically similar.
    """
    
    def __init__(self, llm: SemanticCachedLLM):
        """Initialize the semantic cached LLM adapter.
        
        Args:
            llm: The semantic cached LLM implementation to adapt
        """
        super().__init__(llm)
        self.semantic_cached_llm = llm
    
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
            return await self.semantic_cached_llm.get_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise ValueError(f"Failed to generate embedding: {e}") from e
    
    async def find_similar(
        self, 
        embedding: List[float], 
        threshold: float = 0.8
    ) -> Optional[LLMResponse]:
        """Find a similar response in the cache.
        
        Args:
            embedding: The embedding to compare against
            threshold: The similarity threshold (0.0-1.0, higher means more similar)
            
        Returns:
            The most similar cached response, or None if none are similar enough
            
        Raises:
            ValueError: If the threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")
            
        try:
            old_response = await self.semantic_cached_llm.find_similar(embedding, threshold)
            return convert_response(old_response) if old_response else None
        except Exception as e:
            logger.error(f"Error finding similar response: {e}")
            return None


class OpenAILLMAdapter(StreamingLLMAdapter):
    """Adapter for the OpenAILLM class."""
    
    def __init__(self, api_key: str, organization: Optional[str] = None, **kwargs):
        """Initialize the OpenAI LLM adapter.
        
        Args:
            api_key: The OpenAI API key
            organization: The OpenAI organization ID (optional)
            **kwargs: Additional arguments to pass to the OpenAILLM constructor
        """
        llm = OpenAILLM(api_key=api_key, organization=organization, **kwargs)
        super().__init__(llm)


class SemanticCachedOpenAILLMAdapter(SemanticCachedLLMAdapter):
    """Adapter for the SemanticCachedLLM with OpenAILLM.
    
    This adapter combines the OpenAI LLM implementation with semantic caching,
    providing efficient reuse of responses for semantically similar prompts.
    """
    
    def __init__(
        self,
        api_key: str,
        organization: Optional[str] = None,
        cache_size: int = 1000,
        similarity_threshold: float = 0.8,
        **kwargs
    ):
        """Initialize the semantic cached OpenAI LLM adapter.
        
        Args:
            api_key: The OpenAI API key
            organization: The OpenAI organization ID (optional)
            cache_size: The maximum number of items to store in the cache
            similarity_threshold: The similarity threshold for cache hits (0.0-1.0)
            **kwargs: Additional arguments to pass to the OpenAILLM constructor
                - timeout: Request timeout in seconds (default: 60)
                - max_retries: Maximum number of retries (default: 3)
                - model: Default model to use (default: gpt-3.5-turbo)
        """
        # Extract the default model from kwargs or use gpt-3.5-turbo as default
        model = kwargs.get('model', 'gpt-3.5-turbo')
        
        # Create the OpenAI LLM
        llm = OpenAILLM(api_key=api_key, organization=organization, **kwargs)
        
        # Create the semantic cache with the specified size
        semantic_cache = SemanticCache(cache_size=cache_size)
        
        # Create the semantically cached LLM with provider and model information for metrics
        semantic_cached_llm = SemanticCachedLLM(
            llm=llm,
            cache=semantic_cache,
            similarity_threshold=similarity_threshold,
            provider="openai",
            model=model
        )
        
        super().__init__(semantic_cached_llm)


# LLM plugins

class OpenAILLMPlugin(Plugin):
    """Plugin for the OpenAILLM class."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "openai_llm"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "OpenAI LLM provider with enhanced error handling and async capabilities"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        from agentor.core.config import get_config
        
        # Get the OpenAI API key from configuration
        config = get_config()
        api_key = config.get("openai", {}).get("api_key")
        organization = config.get("openai", {}).get("organization")
        
        if not api_key:
            logger.warning("OpenAI API key not found in configuration, skipping OpenAI LLM registration")
            return
        
        # Create the LLM adapter
        llm_adapter = OpenAILLMAdapter(
            api_key=api_key,
            organization=organization,
            timeout=60,
            max_retries=3
        )
        
        # Register the LLM provider
        component_registry = get_component_registry()
        component_registry.register_llm_provider("openai", llm_adapter)
        
        logger.info("Registered OpenAI LLM provider")


class SemanticCachedOpenAILLMPlugin(Plugin):
    """Plugin for the SemanticCachedLLM with OpenAILLM."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "semantic_cached_openai_llm"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Semantically cached OpenAI LLM provider"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        from agentor.core.config import get_config
        
        # Get the OpenAI API key from configuration
        config = get_config()
        api_key = config.get("openai", {}).get("api_key")
        organization = config.get("openai", {}).get("organization")
        
        if not api_key:
            logger.warning("OpenAI API key not found in configuration, skipping Semantic Cached OpenAI LLM registration")
            return
        
        # Create the LLM adapter
        llm_adapter = SemanticCachedOpenAILLMAdapter(
            api_key=api_key,
            organization=organization,
            cache_size=1000,
            similarity_threshold=0.8,
            timeout=60,
            max_retries=3
        )
        
        # Register the LLM provider
        component_registry = get_component_registry()
        component_registry.register_llm_provider("semantic_cached_openai", llm_adapter)
        
        logger.info("Registered Semantic Cached OpenAI LLM provider")
