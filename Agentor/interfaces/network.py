"""Network interfaces for the Agentor framework.

This module provides interfaces for network operations, including HTTP requests,
WebSockets, and other network protocols.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, BinaryIO, TextIO, Iterator
import time
import asyncio
import json

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type


class NetworkError(Exception):
    """Base exception for network errors."""
    pass


class ConnectionError(NetworkError):
    """Exception raised when a connection fails."""
    pass


class RequestError(NetworkError):
    """Exception raised when a request fails."""
    pass


class ResponseError(NetworkError):
    """Exception raised when a response is invalid."""
    pass


class TimeoutError(NetworkError):
    """Exception raised when a request times out."""
    pass


class HttpMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class NetworkResult(Generic[T]):
    """Result of a network operation."""

    def __init__(
        self,
        success: bool,
        data: Optional[T] = None,
        error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.status_code = status_code
        self.headers = headers or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        if self.success:
            return f"NetworkResult(success={self.success}, status_code={self.status_code})"
        else:
            return f"NetworkResult(success={self.success}, error={self.error}, status_code={self.status_code})"

    @classmethod
    def success_result(cls, data: Optional[T] = None, status_code: Optional[int] = None, headers: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None) -> 'NetworkResult[T]':
        """Create a successful network result."""
        return cls(success=True, data=data, status_code=status_code, headers=headers, metadata=metadata)

    @classmethod
    def error_result(cls, error: Union[Exception, str], status_code: Optional[int] = None, headers: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None) -> 'NetworkResult[T]':
        """Create an error network result."""
        if isinstance(error, str):
            error = NetworkError(error)
        return cls(success=False, error=error, status_code=status_code, headers=headers, metadata=metadata)


class NetworkInterface(ABC):
    """Interface for network operations."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.connected = False
        self.connection_params = kwargs
        self.last_activity = time.time()

    @abstractmethod
    async def connect(self) -> NetworkResult:
        """Connect to the network service."""
        pass

    @abstractmethod
    async def disconnect(self) -> NetworkResult:
        """Disconnect from the network service."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, connected={self.connected})"


class HttpInterface(NetworkInterface):
    """Interface for HTTP operations."""

    @abstractmethod
    async def request(
        self,
        method: HttpMethod,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP request."""
        pass

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP GET request."""
        return await self.request(
            method=HttpMethod.GET,
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )

    async def post(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP POST request."""
        return await self.request(
            method=HttpMethod.POST,
            url=url,
            params=params,
            headers=headers,
            data=data,
            json_data=json_data,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )

    async def put(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP PUT request."""
        return await self.request(
            method=HttpMethod.PUT,
            url=url,
            params=params,
            headers=headers,
            data=data,
            json_data=json_data,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )

    async def delete(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP DELETE request."""
        return await self.request(
            method=HttpMethod.DELETE,
            url=url,
            params=params,
            headers=headers,
            data=data,
            json_data=json_data,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )

    async def patch(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP PATCH request."""
        return await self.request(
            method=HttpMethod.PATCH,
            url=url,
            params=params,
            headers=headers,
            data=data,
            json_data=json_data,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )

    async def head(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP HEAD request."""
        return await self.request(
            method=HttpMethod.HEAD,
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )

    async def options(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        verify_ssl: bool = True,
        allow_redirects: bool = True,
        **kwargs
    ) -> NetworkResult:
        """Send an HTTP OPTIONS request."""
        return await self.request(
            method=HttpMethod.OPTIONS,
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify_ssl=verify_ssl,
            allow_redirects=allow_redirects,
            **kwargs
        )


class WebSocketInterface(NetworkInterface):
    """Interface for WebSocket operations."""

    @abstractmethod
    async def connect_ws(self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> NetworkResult:
        """Connect to a WebSocket server."""
        pass

    @abstractmethod
    async def disconnect_ws(self) -> NetworkResult:
        """Disconnect from the WebSocket server."""
        pass

    @abstractmethod
    async def send(self, data: Union[str, bytes, Dict[str, Any]]) -> NetworkResult:
        """Send data to the WebSocket server."""
        pass

    @abstractmethod
    async def receive(self, timeout: Optional[float] = None) -> NetworkResult:
        """Receive data from the WebSocket server."""
        pass

    @abstractmethod
    async def ping(self, data: Optional[Union[str, bytes]] = None) -> NetworkResult:
        """Send a ping to the WebSocket server."""
        pass

    @abstractmethod
    async def pong(self, data: Optional[Union[str, bytes]] = None) -> NetworkResult:
        """Send a pong to the WebSocket server."""
        pass


# Import specific implementations
from .http_client import HttpClient
from .websocket_client import WebSocketClient
from .network_registry import NetworkRegistry, default_registry

__all__ = [
    'NetworkError', 'ConnectionError', 'RequestError', 'ResponseError', 'TimeoutError',
    'HttpMethod', 'NetworkResult',
    'NetworkInterface', 'HttpInterface', 'WebSocketInterface',
    'HttpClient', 'WebSocketClient',
    'NetworkRegistry', 'default_registry'
]