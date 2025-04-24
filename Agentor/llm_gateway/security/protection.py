"""
Request and response protection for the LLM Gateway.

This module provides protection mechanisms for LLM requests and responses,
including content filtering, timing attack mitigation, and more.
"""

import re
import time
import random
import logging
import secrets
from typing import Dict, List, Optional, Set, Union, Any, Callable, Pattern
import asyncio
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class ContentFilteringConfig:
    """Configuration for content filtering."""
    
    def __init__(
        self,
        filter_pii: bool = True,
        filter_sensitive_data: bool = True,
        filter_prompt_injections: bool = True,
        pii_patterns: Optional[Dict[str, Pattern]] = None,
        sensitive_patterns: Optional[Dict[str, Pattern]] = None,
        injection_patterns: Optional[List[Pattern]] = None
    ):
        """
        Initialize content filtering configuration.
        
        Args:
            filter_pii: Whether to filter personally identifiable information
            filter_sensitive_data: Whether to filter sensitive data
            filter_prompt_injections: Whether to filter prompt injections
            pii_patterns: Dictionary mapping PII types to regex patterns
            sensitive_patterns: Dictionary mapping sensitive data types to regex patterns
            injection_patterns: List of regex patterns for prompt injections
        """
        self.filter_pii = filter_pii
        self.filter_sensitive_data = filter_sensitive_data
        self.filter_prompt_injections = filter_prompt_injections
        
        # Default PII patterns
        self.pii_patterns = pii_patterns or {
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "phone": re.compile(r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
        }
        
        # Default sensitive data patterns
        self.sensitive_patterns = sensitive_patterns or {
            "api_key": re.compile(r"\b(?:api[_-]?key|token)[=:]\s*['\"]([\w\-]{20,})['\"]\b", re.IGNORECASE),
            "password": re.compile(r"\b(?:password|passwd|pwd)[=:]\s*['\"](.*?)['\"]\b", re.IGNORECASE),
            "secret": re.compile(r"\b(?:secret|private[_-]?key)[=:]\s*['\"](.*?)['\"]\b", re.IGNORECASE)
        }
        
        # Default prompt injection patterns
        self.injection_patterns = injection_patterns or [
            re.compile(r"ignore previous instructions", re.IGNORECASE),
            re.compile(r"ignore all previous commands", re.IGNORECASE),
            re.compile(r"disregard earlier prompts", re.IGNORECASE),
            re.compile(r"forget your instructions", re.IGNORECASE),
            re.compile(r"you are now", re.IGNORECASE),
            re.compile(r"now you are", re.IGNORECASE),
            re.compile(r"you are a", re.IGNORECASE),
            re.compile(r"you're a", re.IGNORECASE)
        ]


class ContentFilter:
    """Filter for LLM content."""
    
    def __init__(self, config: Optional[ContentFilteringConfig] = None):
        """
        Initialize content filter.
        
        Args:
            config: Content filtering configuration
        """
        self.config = config or ContentFilteringConfig()
    
    def filter_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter an LLM request.
        
        Args:
            request_data: Request data
            
        Returns:
            Filtered request data
        """
        # Create a copy of the request data
        filtered_data = request_data.copy()
        
        # Filter prompt injections if enabled
        if self.config.filter_prompt_injections and "prompt" in filtered_data:
            filtered_data["prompt"] = self._filter_prompt_injections(filtered_data["prompt"])
        
        # Filter messages if present
        if "messages" in filtered_data:
            filtered_data["messages"] = self._filter_messages(filtered_data["messages"])
        
        return filtered_data
    
    def filter_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter an LLM response.
        
        Args:
            response_data: Response data
            
        Returns:
            Filtered response data
        """
        # Create a copy of the response data
        filtered_data = response_data.copy()
        
        # Filter PII and sensitive data if enabled
        if self.config.filter_pii or self.config.filter_sensitive_data:
            if "text" in filtered_data:
                filtered_data["text"] = self._filter_sensitive_content(filtered_data["text"])
            
            if "choices" in filtered_data and isinstance(filtered_data["choices"], list):
                for i, choice in enumerate(filtered_data["choices"]):
                    if isinstance(choice, dict) and "text" in choice:
                        filtered_data["choices"][i]["text"] = self._filter_sensitive_content(choice["text"])
                    elif isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                        filtered_data["choices"][i]["message"]["content"] = self._filter_sensitive_content(choice["message"]["content"])
        
        return filtered_data
    
    def _filter_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter messages in an LLM request.
        
        Args:
            messages: List of messages
            
        Returns:
            Filtered messages
        """
        filtered_messages = []
        
        for message in messages:
            filtered_message = message.copy()
            
            # Filter content if present
            if "content" in filtered_message:
                # Filter prompt injections if enabled
                if self.config.filter_prompt_injections:
                    filtered_message["content"] = self._filter_prompt_injections(filtered_message["content"])
                
                # Filter PII and sensitive data if enabled
                if (self.config.filter_pii or self.config.filter_sensitive_data) and filtered_message["role"] == "assistant":
                    filtered_message["content"] = self._filter_sensitive_content(filtered_message["content"])
            
            filtered_messages.append(filtered_message)
        
        return filtered_messages
    
    def _filter_prompt_injections(self, text: str) -> str:
        """
        Filter prompt injections from text.
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text
        """
        if not text:
            return text
        
        # Check for prompt injections
        for pattern in self.config.injection_patterns:
            if pattern.search(text):
                logger.warning(f"Detected potential prompt injection: {pattern.pattern}")
                text = pattern.sub("[FILTERED]", text)
        
        return text
    
    def _filter_sensitive_content(self, text: str) -> str:
        """
        Filter PII and sensitive data from text.
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text
        """
        if not text:
            return text
        
        # Filter PII if enabled
        if self.config.filter_pii:
            for pii_type, pattern in self.config.pii_patterns.items():
                text = pattern.sub(f"[{pii_type.upper()}]", text)
        
        # Filter sensitive data if enabled
        if self.config.filter_sensitive_data:
            for sensitive_type, pattern in self.config.sensitive_patterns.items():
                text = pattern.sub(f"[{sensitive_type.upper()}]", text)
        
        return text


class TimingAttackMitigation:
    """Mitigation for timing attacks."""
    
    def __init__(
        self,
        enabled: bool = True,
        min_delay_ms: int = 0,
        max_delay_ms: int = 50,
        constant_time_comparison: bool = True
    ):
        """
        Initialize timing attack mitigation.
        
        Args:
            enabled: Whether timing attack mitigation is enabled
            min_delay_ms: Minimum delay in milliseconds
            max_delay_ms: Maximum delay in milliseconds
            constant_time_comparison: Whether to use constant-time comparison
        """
        self.enabled = enabled
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.constant_time_comparison = constant_time_comparison
    
    async def apply_random_delay(self) -> None:
        """
        Apply a random delay to mitigate timing attacks.
        """
        if not self.enabled:
            return
        
        # Calculate random delay in seconds
        delay = random.uniform(self.min_delay_ms / 1000, self.max_delay_ms / 1000)
        
        # Apply delay
        await asyncio.sleep(delay)
    
    def constant_time_compare(self, a: str, b: str) -> bool:
        """
        Compare two strings in constant time to prevent timing attacks.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            True if strings are equal, False otherwise
        """
        if not self.constant_time_comparison:
            return a == b
        
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)
        
        return result == 0


class ContentFilteringMiddleware(BaseHTTPMiddleware):
    """Middleware for filtering LLM content."""
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[ContentFilteringConfig] = None,
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            config: Content filtering configuration
            exclude_paths: Paths to exclude from content filtering
        """
        super().__init__(app)
        self.content_filter = ContentFilter(config)
        self.exclude_paths = exclude_paths or []
        
        logger.info("Initialized content filtering middleware")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any]
    ) -> Response:
        """
        Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Check if path should be excluded
        path = request.url.path
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            return await call_next(request)
        
        # Only filter POST requests to specific endpoints
        if request.method == "POST" and path in ["/generate", "/chat/completions"]:
            # Get request body
            body = await request.body()
            
            # Parse JSON
            import json
            try:
                request_data = json.loads(body)
            except json.JSONDecodeError:
                # If not valid JSON, just pass through
                return await call_next(request)
            
            # Filter request
            filtered_data = self.content_filter.filter_request(request_data)
            
            # Create a new request with the filtered data
            new_body = json.dumps(filtered_data).encode()
            request._body = new_body
        
        # Process the request
        response = await call_next(request)
        
        # Filter response if it's JSON
        if "application/json" in response.headers.get("content-type", ""):
            # Get response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Parse JSON
            import json
            try:
                response_data = json.loads(body)
            except json.JSONDecodeError:
                # If not valid JSON, just pass through
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type
                )
            
            # Filter response
            filtered_data = self.content_filter.filter_response(response_data)
            
            # Create a new response with the filtered data
            new_body = json.dumps(filtered_data).encode()
            
            # Add header to indicate content was filtered
            headers = dict(response.headers)
            headers["X-LLM-Content-Filtered"] = "true"
            
            return Response(
                content=new_body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.media_type
            )
        
        return response


class TimingAttackMitigationMiddleware(BaseHTTPMiddleware):
    """Middleware for mitigating timing attacks."""
    
    def __init__(
        self,
        app: ASGIApp,
        mitigation: Optional[TimingAttackMitigation] = None,
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            mitigation: Timing attack mitigation
            exclude_paths: Paths to exclude from timing attack mitigation
        """
        super().__init__(app)
        self.mitigation = mitigation or TimingAttackMitigation()
        self.exclude_paths = exclude_paths or []
        
        logger.info("Initialized timing attack mitigation middleware")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any]
    ) -> Response:
        """
        Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Check if path should be excluded
        path = request.url.path
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            return await call_next(request)
        
        # Process the request
        response = await call_next(request)
        
        # Apply random delay to mitigate timing attacks
        await self.mitigation.apply_random_delay()
        
        return response


def add_protection_to_app(
    app: FastAPI,
    content_filtering_config: Optional[ContentFilteringConfig] = None,
    timing_attack_mitigation: Optional[TimingAttackMitigation] = None,
    exclude_paths: Optional[List[str]] = None
) -> None:
    """
    Add protection middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        content_filtering_config: Content filtering configuration
        timing_attack_mitigation: Timing attack mitigation
        exclude_paths: Paths to exclude from protection
    """
    # Add content filtering middleware
    app.add_middleware(
        ContentFilteringMiddleware,
        config=content_filtering_config,
        exclude_paths=exclude_paths
    )
    
    # Add timing attack mitigation middleware
    app.add_middleware(
        TimingAttackMitigationMiddleware,
        mitigation=timing_attack_mitigation,
        exclude_paths=exclude_paths
    )
    
    logger.info("Added protection middleware to FastAPI application")
