"""
API interfaces for the Agentor framework.

This module provides interfaces for interacting with external APIs,
including REST APIs, GraphQL APIs, and other web services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
import asyncio
import json
import logging
import time
from enum import Enum
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientSession, ClientResponse, ClientError

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Response type


class HttpMethod(Enum):
    """HTTP methods for API requests."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ApiResponse(Generic[T]):
    """Response from an API request."""

    def __init__(
        self,
        success: bool,
        status_code: int,
        data: Optional[T] = None,
        error: Optional[Exception] = None,
        headers: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.status_code = status_code
        self.data = data
        self.error = error
        self.headers = headers or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        if self.success:
            return f"ApiResponse(success={self.success}, status_code={self.status_code})"
        else:
            return f"ApiResponse(success={self.success}, status_code={self.status_code}, error={self.error})"

    @classmethod
    def success_response(cls, status_code: int, data: T, headers: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None) -> 'ApiResponse[T]':
        """Create a successful API response."""
        return cls(success=True, status_code=status_code, data=data, headers=headers, metadata=metadata)

    @classmethod
    def error_response(cls, status_code: int, error: Union[Exception, str], headers: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None) -> 'ApiResponse[T]':
        """Create an error API response."""
        if isinstance(error, str):
            error = Exception(error)
        return cls(success=False, status_code=status_code, error=error, headers=headers, metadata=metadata)


class ApiClient(ABC):
    """Base class for API clients."""

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.headers = headers or {}

    @abstractmethod
    async def request(
        self,
        method: HttpMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a request to the API."""
        pass

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a GET request to the API."""
        return await self.request(HttpMethod.GET, endpoint, params=params, headers=headers, timeout=timeout)

    async def post(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a POST request to the API."""
        return await self.request(HttpMethod.POST, endpoint, params=params, data=data, headers=headers, timeout=timeout)

    async def put(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a PUT request to the API."""
        return await self.request(HttpMethod.PUT, endpoint, params=params, data=data, headers=headers, timeout=timeout)

    async def patch(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a PATCH request to the API."""
        return await self.request(HttpMethod.PATCH, endpoint, params=params, data=data, headers=headers, timeout=timeout)

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a DELETE request to the API."""
        return await self.request(HttpMethod.DELETE, endpoint, params=params, headers=headers, timeout=timeout)


class RestApiClient(ApiClient):
    """Client for REST APIs."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[aiohttp.BasicAuth] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True
    ):
        super().__init__(base_url=base_url, headers=headers)
        self.auth = auth
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session: Optional[ClientSession] = None

    async def _get_session(self) -> ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                auth=self.auth,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def request(
        self,
        method: HttpMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a request to the API."""
        url = urljoin(self.base_url, endpoint)
        request_headers = {**self.headers}
        if headers:
            request_headers.update(headers)

        # Convert data to JSON if it's a dict or list
        json_data = None
        if isinstance(data, (dict, list)):
            json_data = data
            data = None

        # Set timeout
        request_timeout = aiohttp.ClientTimeout(total=timeout or self.timeout)

        session = await self._get_session()

        try:
            start_time = time.time()
            async with session.request(
                method.value,
                url,
                params=params,
                data=data,
                json=json_data,
                headers=request_headers,
                timeout=request_timeout,
                ssl=None if self.verify_ssl else False
            ) as response:
                response_time = time.time() - start_time

                # Read response data
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    response_data = await response.text()

                # Create metadata
                metadata = {
                    "url": str(response.url),
                    "method": method.value,
                    "response_time": response_time,
                    "content_type": response.content_type,
                    "content_length": response.content_length
                }

                # Check if the request was successful
                if response.status < 400:
                    logger.debug(
                        f"API request successful: {method.value} {url}, "
                        f"status: {response.status}, time: {response_time:.2f}s"
                    )
                    return ApiResponse.success_response(
                        status_code=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        metadata=metadata
                    )
                else:
                    logger.warning(
                        f"API request failed: {method.value} {url}, "
                        f"status: {response.status}, time: {response_time:.2f}s"
                    )
                    return ApiResponse.error_response(
                        status_code=response.status,
                        error=f"HTTP {response.status}: {response_data}",
                        headers=dict(response.headers),
                        metadata=metadata
                    )

        except asyncio.TimeoutError as e:
            logger.error(f"API request timed out: {method.value} {url}")
            return ApiResponse.error_response(
                status_code=408,  # Request Timeout
                error=f"Request timed out: {e}",
                metadata={"url": url, "method": method.value}
            )
        except ClientError as e:
            logger.error(f"API request error: {method.value} {url}, error: {e}")
            return ApiResponse.error_response(
                status_code=500,  # Internal Server Error
                error=f"Client error: {e}",
                metadata={"url": url, "method": method.value}
            )
        except Exception as e:
            logger.error(f"Unexpected error in API request: {method.value} {url}, error: {e}")
            return ApiResponse.error_response(
                status_code=500,  # Internal Server Error
                error=f"Unexpected error: {e}",
                metadata={"url": url, "method": method.value}
            )


class GraphQLClient(ApiClient):
    """Client for GraphQL APIs."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[aiohttp.BasicAuth] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True
    ):
        super().__init__(base_url=base_url, headers=headers)
        self.auth = auth
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session: Optional[ClientSession] = None

    async def _get_session(self) -> ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                auth=self.auth,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def request(
        self,
        method: HttpMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Make a request to the GraphQL API.

        For GraphQL, this is typically a POST request with a query and variables.
        The endpoint parameter is ignored, as GraphQL typically uses a single endpoint.
        """
        url = self.base_url
        request_headers = {**self.headers}
        if headers:
            request_headers.update(headers)

        # Set timeout
        request_timeout = aiohttp.ClientTimeout(total=timeout or self.timeout)

        session = await self._get_session()

        try:
            start_time = time.time()
            async with session.post(
                url,
                json=data,
                headers=request_headers,
                timeout=request_timeout,
                ssl=None if self.verify_ssl else False
            ) as response:
                response_time = time.time() - start_time

                # Read response data
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    response_data = await response.text()

                # Create metadata
                metadata = {
                    "url": str(response.url),
                    "method": "POST",  # GraphQL always uses POST
                    "response_time": response_time,
                    "content_type": response.content_type,
                    "content_length": response.content_length
                }

                # Check if the request was successful
                if response.status < 400 and isinstance(response_data, dict) and "errors" not in response_data:
                    logger.debug(
                        f"GraphQL request successful: {url}, "
                        f"status: {response.status}, time: {response_time:.2f}s"
                    )
                    return ApiResponse.success_response(
                        status_code=response.status,
                        data=response_data.get("data"),
                        headers=dict(response.headers),
                        metadata=metadata
                    )
                else:
                    errors = response_data.get("errors") if isinstance(response_data, dict) else None
                    error_message = str(errors) if errors else f"HTTP {response.status}: {response_data}"
                    logger.warning(
                        f"GraphQL request failed: {url}, "
                        f"status: {response.status}, errors: {errors}, time: {response_time:.2f}s"
                    )
                    return ApiResponse.error_response(
                        status_code=response.status,
                        error=error_message,
                        headers=dict(response.headers),
                        metadata={
                            **metadata,
                            "errors": errors
                        }
                    )

        except asyncio.TimeoutError as e:
            logger.error(f"GraphQL request timed out: {url}")
            return ApiResponse.error_response(
                status_code=408,  # Request Timeout
                error=f"Request timed out: {e}",
                metadata={"url": url, "method": "POST"}
            )
        except ClientError as e:
            logger.error(f"GraphQL request error: {url}, error: {e}")
            return ApiResponse.error_response(
                status_code=500,  # Internal Server Error
                error=f"Client error: {e}",
                metadata={"url": url, "method": "POST"}
            )
        except Exception as e:
            logger.error(f"Unexpected error in GraphQL request: {url}, error: {e}")
            return ApiResponse.error_response(
                status_code=500,  # Internal Server Error
                error=f"Unexpected error: {e}",
                metadata={"url": url, "method": "POST"}
            )

    async def query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Execute a GraphQL query."""
        data = {
            "query": query
        }
        if variables:
            data["variables"] = variables
        if operation_name:
            data["operationName"] = operation_name

        return await self.request(
            method=HttpMethod.POST,
            endpoint="",  # Ignored for GraphQL
            data=data,
            headers=headers,
            timeout=timeout
        )

    async def mutation(
        self,
        mutation: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Execute a GraphQL mutation."""
        # GraphQL mutations use the same format as queries
        return await self.query(
            query=mutation,
            variables=variables,
            operation_name=operation_name,
            headers=headers,
            timeout=timeout
        )


class WebhookClient:
    """Client for sending webhook notifications."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        verify_ssl: bool = True
    ):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._session: Optional[ClientSession] = None

    async def _get_session(self) -> ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send(
        self,
        data: Any,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ApiResponse:
        """Send a webhook notification."""
        request_headers = {**self.headers}
        if headers:
            request_headers.update(headers)

        # Set timeout
        request_timeout = aiohttp.ClientTimeout(total=timeout or self.timeout)

        session = await self._get_session()

        try:
            start_time = time.time()
            async with session.post(
                self.url,
                json=data,
                headers=request_headers,
                timeout=request_timeout,
                ssl=None if self.verify_ssl else False
            ) as response:
                response_time = time.time() - start_time

                # Read response data
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    response_data = await response.text()

                # Create metadata
                metadata = {
                    "url": str(response.url),
                    "method": "POST",
                    "response_time": response_time,
                    "content_type": response.content_type,
                    "content_length": response.content_length
                }

                # Check if the request was successful
                if response.status < 400:
                    logger.debug(
                        f"Webhook sent successfully: {self.url}, "
                        f"status: {response.status}, time: {response_time:.2f}s"
                    )
                    return ApiResponse.success_response(
                        status_code=response.status,
                        data=response_data,
                        headers=dict(response.headers),
                        metadata=metadata
                    )
                else:
                    logger.warning(
                        f"Webhook failed: {self.url}, "
                        f"status: {response.status}, time: {response_time:.2f}s"
                    )
                    return ApiResponse.error_response(
                        status_code=response.status,
                        error=f"HTTP {response.status}: {response_data}",
                        headers=dict(response.headers),
                        metadata=metadata
                    )

        except asyncio.TimeoutError as e:
            logger.error(f"Webhook timed out: {self.url}")
            return ApiResponse.error_response(
                status_code=408,  # Request Timeout
                error=f"Request timed out: {e}",
                metadata={"url": self.url, "method": "POST"}
            )
        except ClientError as e:
            logger.error(f"Webhook error: {self.url}, error: {e}")
            return ApiResponse.error_response(
                status_code=500,  # Internal Server Error
                error=f"Client error: {e}",
                metadata={"url": self.url, "method": "POST"}
            )
        except Exception as e:
            logger.error(f"Unexpected error in webhook: {self.url}, error: {e}")
            return ApiResponse.error_response(
                status_code=500,  # Internal Server Error
                error=f"Unexpected error: {e}",
                metadata={"url": self.url, "method": "POST"}
            )


class ApiRegistry:
    """Registry for API clients."""

    def __init__(self):
        self.clients: Dict[str, ApiClient] = {}

    def register(self, name: str, client: ApiClient) -> None:
        """Register an API client."""
        if name in self.clients:
            logger.warning(f"Overwriting existing API client with name {name}")
        self.clients[name] = client
        logger.debug(f"Registered API client {name}")

    def unregister(self, name: str) -> None:
        """Unregister an API client."""
        if name in self.clients:
            del self.clients[name]
            logger.debug(f"Unregistered API client {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent API client {name}")

    def get(self, name: str) -> Optional[ApiClient]:
        """Get an API client by name."""
        return self.clients.get(name)

    def list_clients(self) -> List[str]:
        """List all registered API client names."""
        return list(self.clients.keys())

    async def close_all(self) -> None:
        """Close all API clients."""
        for name, client in self.clients.items():
            if hasattr(client, "close") and callable(client.close):
                try:
                    await client.close()
                    logger.debug(f"Closed API client {name}")
                except Exception as e:
                    logger.error(f"Error closing API client {name}: {e}")

    def clear(self) -> None:
        """Clear all registered API clients."""
        self.clients.clear()
        logger.debug("Cleared all registered API clients")


# Global API registry
default_registry = ApiRegistry()