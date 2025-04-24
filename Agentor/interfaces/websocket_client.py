"""
WebSocket client implementation for the Agentor framework.

This module provides a simple WebSocket client implementation using aiohttp.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import time
import aiohttp

from .network import (
    NetworkInterface, WebSocketInterface, NetworkResult,
    NetworkError, ConnectionError, RequestError, ResponseError, TimeoutError
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class WebSocketClient(WebSocketInterface):
    """WebSocket client implementation using aiohttp."""

    def __init__(
        self,
        name: str,
        default_headers: Optional[Dict[str, str]] = None,
        default_timeout: float = 30.0,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.default_headers = default_headers or {}
        self.default_timeout = default_timeout
        self.session = None
        self.ws = None
        self.connected = False

    async def connect(self) -> NetworkResult:
        """Create an aiohttp session."""
        try:
            if self.connected and self.session:
                return NetworkResult.success_result()
            
            self.session = aiohttp.ClientSession(headers=self.default_headers)
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Created WebSocket client session: {self.name}")
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to create WebSocket client session: {self.name}, error: {e}")
            return NetworkResult.error_result(ConnectionError(f"Failed to create WebSocket client session: {e}"))

    async def disconnect(self) -> NetworkResult:
        """Close the aiohttp session."""
        try:
            if self.ws:
                await self.ws.close()
                self.ws = None
            
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            logger.info(f"Closed WebSocket client session: {self.name}")
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to close WebSocket client session: {self.name}, error: {e}")
            return NetworkResult.error_result(ConnectionError(f"Failed to close WebSocket client session: {e}"))

    async def connect_ws(self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> NetworkResult:
        """Connect to a WebSocket server."""
        # Ensure we have an active session
        result = await self.connect()
        if not result.success:
            return result
        
        try:
            self.last_activity = time.time()
            
            # Merge headers
            merged_headers = self.default_headers.copy()
            if headers:
                merged_headers.update(headers)
            
            # Connect to the WebSocket server
            self.ws = await self.session.ws_connect(
                url=url,
                headers=merged_headers,
                timeout=kwargs.get('timeout', self.default_timeout),
                **kwargs
            )
            
            logger.info(f"Connected to WebSocket server: {url}")
            return NetworkResult.success_result()
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error: {e}")
            return NetworkResult.error_result(ConnectionError(f"Connection error: {e}"))
        except aiohttp.ClientResponseError as e:
            logger.error(f"Response error: {e}")
            return NetworkResult.error_result(ResponseError(f"Response error: {e}"))
        except asyncio.TimeoutError as e:
            logger.error(f"Connection timed out: {e}")
            return NetworkResult.error_result(TimeoutError(f"Connection timed out: {e}"))
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {e}")
            return NetworkResult.error_result(NetworkError(f"Failed to connect to WebSocket server: {e}"))

    async def disconnect_ws(self) -> NetworkResult:
        """Disconnect from the WebSocket server."""
        try:
            if self.ws:
                await self.ws.close()
                self.ws = None
            
            logger.info(f"Disconnected from WebSocket server")
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to disconnect from WebSocket server: {e}")
            return NetworkResult.error_result(NetworkError(f"Failed to disconnect from WebSocket server: {e}"))

    async def send(self, data: Union[str, bytes, Dict[str, Any]]) -> NetworkResult:
        """Send data to the WebSocket server."""
        if not self.ws:
            return NetworkResult.error_result(ConnectionError("Not connected to WebSocket server"))
        
        try:
            self.last_activity = time.time()
            
            # Convert data to the appropriate format
            if isinstance(data, dict):
                await self.ws.send_json(data)
            elif isinstance(data, str):
                await self.ws.send_str(data)
            elif isinstance(data, bytes):
                await self.ws.send_bytes(data)
            else:
                return NetworkResult.error_result(RequestError(f"Unsupported data type: {type(data)}"))
            
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to send data to WebSocket server: {e}")
            return NetworkResult.error_result(NetworkError(f"Failed to send data to WebSocket server: {e}"))

    async def receive(self, timeout: Optional[float] = None) -> NetworkResult:
        """Receive data from the WebSocket server."""
        if not self.ws:
            return NetworkResult.error_result(ConnectionError("Not connected to WebSocket server"))
        
        try:
            self.last_activity = time.time()
            
            # Set timeout
            if timeout is not None:
                try:
                    msg = await asyncio.wait_for(self.ws.receive(), timeout=timeout)
                except asyncio.TimeoutError:
                    return NetworkResult.error_result(TimeoutError("Receive operation timed out"))
            else:
                msg = await self.ws.receive()
            
            # Process the message based on its type
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    # Try to parse as JSON
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    # If not JSON, return as string
                    data = msg.data
                return NetworkResult.success_result(data=data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                return NetworkResult.success_result(data=msg.data)
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                return NetworkResult.error_result(ConnectionError("WebSocket connection closed"))
            elif msg.type == aiohttp.WSMsgType.ERROR:
                return NetworkResult.error_result(NetworkError(f"WebSocket error: {msg.data}"))
            else:
                return NetworkResult.error_result(NetworkError(f"Unknown WebSocket message type: {msg.type}"))
        except Exception as e:
            logger.error(f"Failed to receive data from WebSocket server: {e}")
            return NetworkResult.error_result(NetworkError(f"Failed to receive data from WebSocket server: {e}"))

    async def ping(self, data: Optional[Union[str, bytes]] = None) -> NetworkResult:
        """Send a ping to the WebSocket server."""
        if not self.ws:
            return NetworkResult.error_result(ConnectionError("Not connected to WebSocket server"))
        
        try:
            self.last_activity = time.time()
            
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            await self.ws.ping(data)
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to send ping to WebSocket server: {e}")
            return NetworkResult.error_result(NetworkError(f"Failed to send ping to WebSocket server: {e}"))

    async def pong(self, data: Optional[Union[str, bytes]] = None) -> NetworkResult:
        """Send a pong to the WebSocket server."""
        if not self.ws:
            return NetworkResult.error_result(ConnectionError("Not connected to WebSocket server"))
        
        try:
            self.last_activity = time.time()
            
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            await self.ws.pong(data)
            return NetworkResult.success_result()
        except Exception as e:
            logger.error(f"Failed to send pong to WebSocket server: {e}")
            return NetworkResult.error_result(NetworkError(f"Failed to send pong to WebSocket server: {e}"))
