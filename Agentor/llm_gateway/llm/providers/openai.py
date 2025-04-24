import openai
import asyncio
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Awaitable
import logging
from contextlib import asynccontextmanager
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log
)

from agentor.llm_gateway.llm.base import BaseLLM, LLMRequest, LLMResponse
from agentor.llm_gateway.utils.circuit_breaker import LLMCircuitBreaker, CircuitBreakerOpenError
from agentor.llm_gateway.utils.http import AsyncHTTPClient

logger = logging.getLogger(__name__)


# Define custom types for type hints
T = TypeVar('T')
AsyncFunc = Callable[..., Awaitable[T]]


class OpenAIError(Exception):
    """Base class for OpenAI-specific errors."""
    pass


class OpenAIRateLimitError(OpenAIError):
    """Error raised when OpenAI rate limits are exceeded."""
    pass


class OpenAITimeoutError(OpenAIError):
    """Error raised when an OpenAI request times out."""
    pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider with enhanced error handling and async capabilities."""

    def __init__(self,
                 api_key: str,
                 organization: Optional[str] = None,
                 timeout: int = 60,
                 max_retries: int = 3,
                 batch_size: int = 20):
        """Initialize the OpenAI LLM provider.

        Args:
            api_key: The OpenAI API key
            organization: The OpenAI organization ID (optional)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            batch_size: Maximum number of requests to batch together
        """
        self.client = openai.AsyncOpenAI(api_key=api_key, organization=organization, timeout=timeout)
        self.circuit_breaker = LLMCircuitBreaker()
        self.http_client = AsyncHTTPClient(timeout=timeout)
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.request_semaphore = asyncio.Semaphore(20)  # Limit concurrent requests

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (openai.APIError, openai.APIConnectionError, asyncio.TimeoutError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _generate_internal(self, request: LLMRequest) -> LLMResponse:
        """Internal method to generate a response from the OpenAI API.

        This method is wrapped by the circuit breaker in the generate method.

        Args:
            request: The request to send to the OpenAI API

        Returns:
            The response from the OpenAI API

        Raises:
            OpenAIRateLimitError: If the rate limit is exceeded
            OpenAITimeoutError: If the request times out
            openai.APIError: For other API errors
        """
        start_time = time.time()
        logger.info(f"Sending request to OpenAI API: {request.model}")

        # Use a semaphore to limit concurrent requests
        async with self.request_semaphore:
            try:
                response = await self.client.completions.create(
                    model=request.model,
                    prompt=request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    stop=request.stop_sequences
                )

                # Extract the response text
                text = response.choices[0].text

                # Extract usage information
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

                # Calculate latency
                latency = time.time() - start_time

                logger.info(f"Received response from OpenAI API: {len(text)} chars in {latency:.2f}s")

                return LLMResponse(
                    text=text,
                    model=request.model,
                    usage=usage,
                    metadata={
                        "request_id": response.id,
                        "original_request": request.model_dump(exclude={"prompt"}),
                        "latency": latency
                    }
                )

            except openai.RateLimitError as e:
                logger.error(f"OpenAI rate limit exceeded: {str(e)}")
                raise OpenAIRateLimitError(f"Rate limit exceeded: {str(e)}") from e

            except asyncio.TimeoutError as e:
                logger.error(f"OpenAI request timed out after {time.time() - start_time:.2f}s")
                raise OpenAITimeoutError("Request timed out") from e

            except Exception as e:
                logger.error(f"Error in OpenAI request: {type(e).__name__}: {str(e)}")
                raise

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the OpenAI API with circuit breaker protection.

        Args:
            request: The request to send to the OpenAI API

        Returns:
            The response from the OpenAI API

        Raises:
            CircuitBreakerOpenError: If the circuit breaker is open
            OpenAIRateLimitError: If the rate limit is exceeded
            OpenAITimeoutError: If the request times out
            openai.APIError: If there is an API error
            Exception: For any other unexpected error
        """
        try:
            # Use the circuit breaker to protect against service failures
            return await self.circuit_breaker.call_with_breaker(
                provider="openai",
                func=self._generate_internal,
                request=request
            )

        except CircuitBreakerOpenError as e:
            logger.error(f"Circuit breaker open: {e}")
            raise

        except OpenAIRateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise

        except OpenAITimeoutError as e:
            logger.error(f"Request timed out: {e}")
            raise

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    async def generate_batch(self, requests: List[LLMRequest]) -> List[Union[LLMResponse, Exception]]:
        """Generate responses for multiple requests in parallel.

        Args:
            requests: The requests to send to the OpenAI API

        Returns:
            A list of responses or exceptions
        """
        # Split requests into batches to avoid overwhelming the API
        batches = [requests[i:i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        all_results = []

        for batch in batches:
            # Process each batch in parallel
            tasks = [self.generate(request) for request in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(batch_results)

        return all_results

    @asynccontextmanager
    async def lifespan(self):
        """Manage the lifecycle of the LLM provider.

        This context manager initializes resources and cleans them up when done.
        """
        try:
            yield self
        finally:
            # Clean up resources
            await self.http_client.close()
