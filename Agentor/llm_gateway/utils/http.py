import aiohttp
import logging
import asyncio
import time
from typing import Dict, Any, Optional, Union, List, Tuple
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RetryOptions:
    """Options for retrying HTTP requests."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        retry_codes: List[int] = None,
        retry_exceptions: List[type] = None
    ):
        """Initialize retry options.

        Args:
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries in seconds
            max_retry_delay: Maximum delay between retries in seconds
            retry_codes: HTTP status codes to retry
            retry_exceptions: Exception types to retry
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_retry_delay = max_retry_delay
        self.retry_codes = retry_codes or [429, 500, 502, 503, 504]
        self.retry_exceptions = retry_exceptions or [
            aiohttp.ClientError,
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError
        ]


class AsyncHTTPClient:
    """Asynchronous HTTP client with connection pooling, retries, and timeouts."""

    def __init__(
        self,
        limit_per_host: int = 100,
        timeout: int = 30,
        retry_options: RetryOptions = None,
        ssl_verify: bool = True,
        trace_configs: List[aiohttp.TraceConfig] = None
    ):
        """Initialize the HTTP client.

        Args:
            limit_per_host: Maximum number of connections per host
            timeout: Timeout in seconds
            retry_options: Options for retrying requests
            ssl_verify: Whether to verify SSL certificates
            trace_configs: Trace configs for request/response tracing
        """
        self.connector = aiohttp.TCPConnector(
            limit_per_host=limit_per_host,
            ssl=ssl_verify,
            ttl_dns_cache=300  # Cache DNS results for 5 minutes
        )
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_options = retry_options or RetryOptions()
        self.trace_configs = trace_configs or []
        self.session = None
        self._create_session()

    def _create_session(self):
        """Create a new aiohttp session."""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            trace_configs=self.trace_configs
        )

    @asynccontextmanager
    async def _ensure_session(self):
        """Ensure that the session is open.

        Yields:
            The aiohttp session
        """
        if self.session is None or self.session.closed:
            self._create_session()

        try:
            yield self.session
        except Exception as e:
            # If we get a connection error, close and recreate the session
            if isinstance(e, aiohttp.ClientConnectionError):
                await self.close()
                self._create_session()
            raise

    async def close(self):
        """Close the HTTP client."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        response_format: str = 'json'
    ) -> Union[Dict[str, Any], str, bytes]:
        """Send a GET request with retries.

        Args:
            url: The URL to request
            params: The query parameters
            headers: The request headers
            timeout: Request timeout in seconds (overrides client timeout)
            response_format: Format to return ('json', 'text', or 'raw')

        Returns:
            The response in the requested format
        """
        return await self._request_with_retry(
            'GET',
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            response_format=response_format
        )

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        response_format: str = 'json'
    ) -> Union[Dict[str, Any], str, bytes]:
        """Send a POST request with retries.

        Args:
            url: The URL to request
            data: The request data (will be sent as JSON)
            params: The query parameters
            headers: The request headers
            timeout: Request timeout in seconds (overrides client timeout)
            response_format: Format to return ('json', 'text', or 'raw')

        Returns:
            The response in the requested format
        """
        return await self._request_with_retry(
            'POST',
            url,
            json=data,
            params=params,
            headers=headers,
            timeout=timeout,
            response_format=response_format
        )

    async def put(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        response_format: str = 'json'
    ) -> Union[Dict[str, Any], str, bytes]:
        """Send a PUT request with retries.

        Args:
            url: The URL to request
            data: The request data (will be sent as JSON)
            params: The query parameters
            headers: The request headers
            timeout: Request timeout in seconds (overrides client timeout)
            response_format: Format to return ('json', 'text', or 'raw')

        Returns:
            The response in the requested format
        """
        return await self._request_with_retry(
            'PUT',
            url,
            json=data,
            params=params,
            headers=headers,
            timeout=timeout,
            response_format=response_format
        )

    async def delete(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        response_format: str = 'json'
    ) -> Union[Dict[str, Any], str, bytes]:
        """Send a DELETE request with retries.

        Args:
            url: The URL to request
            params: The query parameters
            headers: The request headers
            timeout: Request timeout in seconds (overrides client timeout)
            response_format: Format to return ('json', 'text', or 'raw')

        Returns:
            The response in the requested format
        """
        return await self._request_with_retry(
            'DELETE',
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            response_format=response_format
        )

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Union[Dict[str, Any], str, bytes]:
        """Send a request with retries.

        Args:
            method: The HTTP method
            url: The URL to request
            **kwargs: Additional arguments for the request

        Returns:
            The response in the requested format

        Raises:
            aiohttp.ClientError: If the request fails after all retries
        """
        response_format = kwargs.pop('response_format', 'json')
        timeout_override = kwargs.pop('timeout', None)

        # Apply timeout override if provided
        if timeout_override is not None:
            kwargs['timeout'] = aiohttp.ClientTimeout(total=timeout_override)

        # Initialize retry state
        retry_count = 0
        retry_delay = self.retry_options.retry_delay
        last_exception = None

        # Start retry loop
        while retry_count <= self.retry_options.max_retries:
            try:
                async with self._ensure_session() as session:
                    request_method = getattr(session, method.lower())
                    async with request_method(url, **kwargs) as response:
                        # Check if we need to retry based on status code
                        if response.status in self.retry_options.retry_codes and retry_count < self.retry_options.max_retries:
                            retry_count += 1
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, self.retry_options.max_retry_delay)
                            continue

                        # Raise for non-success status codes
                        response.raise_for_status()

                        # Return the response in the requested format
                        if response_format == 'json':
                            return await response.json()
                        elif response_format == 'text':
                            return await response.text()
                        elif response_format == 'raw':
                            return await response.read()
                        else:
                            raise ValueError(f"Unknown response format: {response_format}")

            except Exception as e:
                last_exception = e

                # Check if we should retry based on exception type
                should_retry = False
                for exc_type in self.retry_options.retry_exceptions:
                    if isinstance(e, exc_type):
                        should_retry = True
                        break

                if should_retry and retry_count < self.retry_options.max_retries:
                    retry_count += 1
                    logger.warning(f"Request failed with {type(e).__name__}: {str(e)}. Retrying ({retry_count}/{self.retry_options.max_retries})...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.retry_options.max_retry_delay)
                else:
                    # We've exhausted our retries or the error is not retryable
                    logger.error(f"Request failed after {retry_count} retries: {str(e)}")
                    raise

        # This should never happen, but just in case
        if last_exception:
            raise last_exception

        raise RuntimeError("Unexpected error in request retry loop")

    async def batch_get(self, urls: List[str], **kwargs) -> List[Tuple[str, Union[Dict[str, Any], str, bytes, Exception]]]:
        """Send multiple GET requests in parallel.

        Args:
            urls: List of URLs to request
            **kwargs: Additional arguments for the requests

        Returns:
            List of (url, response) tuples, where response may be an Exception if the request failed
        """
        tasks = [self.get(url, **kwargs) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(zip(urls, results))
