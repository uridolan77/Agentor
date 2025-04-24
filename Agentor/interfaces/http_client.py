"""
HTTP client implementation for the Agentor framework.

This module provides a simple HTTP client implementation using aiohttp.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import time
import aiohttp

from .network import (
    NetworkInterface, HttpInterface, NetworkResult, HttpMethod,
    NetworkError, ConnectionError, RequestError, ResponseError, TimeoutError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class HttpClient(HttpInterface):
    """HTTP client implementation using aiohttp."""

    def __init__(
        self,
        name: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        default_timeout: float = 30.0,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.default_timeout = default_timeout
        self.session = None
        self.connected = False

    async def connect(self) -> NetworkResult:
        """Create an aiohttp session."""
        try:
            if self.connected and self.session:
                return NetworkResult.success_result()
            
            self.session = aiohttp.ClientSession(headers=self.default_headers)
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Created HTTP client session: {self.name}")
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to create HTTP client session: {self.name}, error: {e}")
            return NetworkResult.error_result(ConnectionError(f"Failed to create HTTP client session: {e}"))

    async def disconnect(self) -> NetworkResult:
        """Close the aiohttp session."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            logger.info(f"Closed HTTP client session: {self.name}")
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to close HTTP client session: {self.name}, error: {e}")
            return NetworkResult.error_result(ConnectionError(f"Failed to close HTTP client session: {e}"))

    async def _ensure_connected(self) -> NetworkResult:
        """Ensure that we have an active session."""
        if not self.connected or not self.session:
            return await self.connect()
        return NetworkResult.success_result()

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
        # Ensure we have an active session
        result = await self._ensure_connected()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
            
            # Prepare the request
            if self.base_url and not url.startswith(('http://', 'https://')):
                url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
            
            # Merge headers
            merged_headers = self.default_headers.copy()
            if headers:
                merged_headers.update(headers)
            
            # Set timeout
            timeout_obj = aiohttp.ClientTimeout(total=timeout or self.default_timeout)
            
            # Send the request
            async with self.session.request(
                method=method.value,
                url=url,
                params=params,
                headers=merged_headers,
                data=data,
                json=json_data,
                timeout=timeout_obj,
                ssl=verify_ssl,
                allow_redirects=allow_redirects,
                **kwargs
            ) as response:
                # Read the response
                try:
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                    else:
                        data = await response.text()
                except Exception as e:
                    logger.error(f"Failed to parse response: {e}")
                    data = await response.read()
                
                # Create the result
                result = NetworkResult(
                    success=response.status < 400,
                    data=data,
                    status_code=response.status,
                    headers=dict(response.headers),
                    metadata={
                        'url': str(response.url),
                        'method': method.value,
                        'reason': response.reason,
                        'content_type': content_type
                    }
                )
                
                # Add error if the request was not successful
                if not result.success:
                    result.error = RequestError(f"Request failed with status {response.status}: {response.reason}")
                
                return result
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error: {e}")
            return NetworkResult.error_result(ConnectionError(f"Connection error: {e}"))
        except aiohttp.ClientResponseError as e:
            logger.error(f"Response error: {e}")
            return NetworkResult.error_result(ResponseError(f"Response error: {e}"))
        except asyncio.TimeoutError as e:
            logger.error(f"Request timed out: {e}")
            return NetworkResult.error_result(TimeoutError(f"Request timed out: {e}"))
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return NetworkResult.error_result(NetworkError(f"Request failed: {e}"))
