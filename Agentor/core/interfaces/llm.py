"""
LLM interfaces for the Agentor framework.

This module defines the interfaces for LLM components in the Agentor framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator, Protocol, runtime_checkable
from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Request to an LLM."""
    
    prompt: str
    """The prompt to send to the LLM."""
    
    model: str
    """The model to use."""
    
    temperature: float = 0.7
    """The temperature to use for sampling."""
    
    max_tokens: Optional[int] = None
    """The maximum number of tokens to generate."""
    
    stop_sequences: Optional[List[str]] = None
    """Sequences that will stop generation."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata for the request."""


class LLMResponse(BaseModel):
    """Response from an LLM."""
    
    text: str
    """The generated text."""
    
    model: str
    """The model that generated the text."""
    
    usage: Dict[str, int]
    """Token usage information."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata for the response."""


class StreamingLLMResponse(BaseModel):
    """Streaming response from an LLM."""
    
    text: str
    """The generated text chunk."""
    
    model: str
    """The model that generated the text."""
    
    is_finished: bool = False
    """Whether this is the last chunk."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata for the response."""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Returns:
            The response from the LLM
        """
        ...


@runtime_checkable
class StreamingLLMProvider(Protocol):
    """Protocol for streaming LLM providers."""
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamingLLMResponse]:
        """Generate a streaming response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Yields:
            Streaming responses from the LLM
        """
        ...


class ILLM(ABC):
    """Interface for LLM components."""
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Returns:
            The response from the LLM
        """
        pass


class IStreamingLLM(ILLM):
    """Interface for streaming LLM components."""
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamingLLMResponse]:
        """Generate a streaming response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Yields:
            Streaming responses from the LLM
        """
        pass


class ICachedLLM(ILLM):
    """Interface for cached LLM components."""
    
    @abstractmethod
    async def get_from_cache(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get a response from the cache.
        
        Args:
            request: The request to check in the cache
            
        Returns:
            The cached response, or None if not found
        """
        pass
    
    @abstractmethod
    async def add_to_cache(self, request: LLMRequest, response: LLMResponse) -> None:
        """Add a response to the cache.
        
        Args:
            request: The request to cache
            response: The response to cache
        """
        pass
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """Clear the cache."""
        pass


class ISemanticCachedLLM(ICachedLLM):
    """Interface for semantically cached LLM components."""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding
        """
        pass
    
    @abstractmethod
    async def find_similar(
        self, 
        embedding: List[float], 
        threshold: float = 0.8
    ) -> Optional[LLMResponse]:
        """Find a similar response in the cache.
        
        Args:
            embedding: The embedding to compare against
            threshold: The similarity threshold
            
        Returns:
            The most similar cached response, or None if none are similar enough
        """
        pass
