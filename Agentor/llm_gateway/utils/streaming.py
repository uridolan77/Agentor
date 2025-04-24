from typing import AsyncIterator, Dict, Any, Optional
import logging
from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

class StreamingLLMResponse:
    """Streaming response from an LLM."""
    
    def __init__(
        self, 
        text: str = "", 
        model: str = "", 
        usage: Optional[Dict[str, int]] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the streaming response.
        
        Args:
            text: The accumulated text
            model: The model used
            usage: Token usage information
            metadata: Additional metadata
        """
        self.text = text
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}


class StreamingBaseLLM:
    """Base class for streaming LLM providers."""
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamingLLMResponse]:
        """Generate a streaming response from the LLM.
        
        Args:
            request: The request to send to the LLM
            
        Yields:
            Streaming responses from the LLM
        """
        raise NotImplementedError("Subclasses must implement generate_stream()")


class StreamingOpenAILLM(StreamingBaseLLM):
    """OpenAI LLM provider with streaming support."""
    
    def __init__(self, api_key: str, organization: Optional[str] = None):
        """Initialize the OpenAI LLM provider.

        Args:
            api_key: The OpenAI API key
            organization: The OpenAI organization ID (optional)
        """
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key, organization=organization)
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamingLLMResponse]:
        """Generate a streaming response from the OpenAI API.
        
        Args:
            request: The request to send to the OpenAI API
            
        Yields:
            Streaming responses from the OpenAI API
        """
        try:
            stream = await self.client.completions.create(
                model=request.model,
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 2000,
                stop=request.stop_sequences,
                stream=True
            )
            
            accumulated_text = ""
            response = StreamingLLMResponse(
                model=request.model,
                metadata={
                    "original_request": request.model_dump(exclude={"prompt"}),
                }
            )
            
            async for chunk in stream:
                # Extract the delta
                delta = chunk.choices[0].text
                
                # Update the accumulated text
                accumulated_text += delta
                
                # Update the response
                response.text = accumulated_text
                
                # Yield the updated response
                yield response
            
            # Update the final response with usage information
            # Note: For streaming, OpenAI doesn't provide token usage
            # We would need to estimate it ourselves
            token_estimate = len(request.prompt.split()) + len(accumulated_text.split())
            response.usage = {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(accumulated_text.split()),
                "total_tokens": token_estimate
            }
            
            # Yield the final response
            yield response
        
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class StreamingAnthropicLLM(StreamingBaseLLM):
    """Anthropic LLM provider with streaming support."""
    
    def __init__(self, api_key: str):
        """Initialize the Anthropic LLM provider.

        Args:
            api_key: The Anthropic API key
        """
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamingLLMResponse]:
        """Generate a streaming response from the Anthropic API.
        
        Args:
            request: The request to send to the Anthropic API
            
        Yields:
            Streaming responses from the Anthropic API
        """
        try:
            stream = await self.client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens or 2000,
                temperature=request.temperature,
                messages=[
                    {"role": "user", "content": request.prompt}
                ],
                stream=True
            )
            
            accumulated_text = ""
            response = StreamingLLMResponse(
                model=request.model,
                metadata={
                    "original_request": request.model_dump(exclude={"prompt"}),
                }
            )
            
            async for chunk in stream:
                if chunk.type == 'content_block_delta' and chunk.delta.type == 'text':
                    # Extract the delta
                    delta = chunk.delta.text
                    
                    # Update the accumulated text
                    accumulated_text += delta
                    
                    # Update the response
                    response.text = accumulated_text
                    
                    # Yield the updated response
                    yield response
            
            # Update the final response with usage information
            if hasattr(stream, 'usage'):
                response.usage = {
                    "input_tokens": stream.usage.input_tokens,
                    "output_tokens": stream.usage.output_tokens,
                    "total_tokens": stream.usage.input_tokens + stream.usage.output_tokens
                }
            else:
                # Estimate token usage
                token_estimate = len(request.prompt.split()) + len(accumulated_text.split())
                response.usage = {
                    "input_tokens": len(request.prompt.split()),
                    "output_tokens": len(accumulated_text.split()),
                    "total_tokens": token_estimate
                }
            
            # Yield the final response
            yield response
        
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise