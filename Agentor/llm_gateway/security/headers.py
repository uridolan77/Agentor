"""
Security headers for the LLM Gateway.

This module provides HTTP security headers and middleware for adding them to responses.
"""

import logging
from typing import Dict, List, Optional, Set, Union, Any, Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityHeadersConfig:
    """Configuration for security headers."""
    
    def __init__(
        self,
        content_security_policy: Optional[str] = None,
        strict_transport_security: Optional[str] = None,
        x_content_type_options: str = "nosniff",
        x_frame_options: str = "DENY",
        x_xss_protection: str = "1; mode=block",
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: Optional[str] = None,
        cache_control: str = "no-store",
        pragma: str = "no-cache",
        cross_origin_opener_policy: str = "same-origin",
        cross_origin_embedder_policy: str = "require-corp",
        cross_origin_resource_policy: str = "same-origin"
    ):
        """
        Initialize security headers configuration.
        
        Args:
            content_security_policy: Content-Security-Policy header value
            strict_transport_security: Strict-Transport-Security header value
            x_content_type_options: X-Content-Type-Options header value
            x_frame_options: X-Frame-Options header value
            x_xss_protection: X-XSS-Protection header value
            referrer_policy: Referrer-Policy header value
            permissions_policy: Permissions-Policy header value
            cache_control: Cache-Control header value
            pragma: Pragma header value
            cross_origin_opener_policy: Cross-Origin-Opener-Policy header value
            cross_origin_embedder_policy: Cross-Origin-Embedder-Policy header value
            cross_origin_resource_policy: Cross-Origin-Resource-Policy header value
        """
        self.content_security_policy = content_security_policy or "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'; require-trusted-types-for 'script'; upgrade-insecure-requests;"
        self.strict_transport_security = strict_transport_security or "max-age=63072000; includeSubDomains; preload"
        self.x_content_type_options = x_content_type_options
        self.x_frame_options = x_frame_options
        self.x_xss_protection = x_xss_protection
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy or "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"
        self.cache_control = cache_control
        self.pragma = pragma
        self.cross_origin_opener_policy = cross_origin_opener_policy
        self.cross_origin_embedder_policy = cross_origin_embedder_policy
        self.cross_origin_resource_policy = cross_origin_resource_policy
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get all security headers as a dictionary.
        
        Returns:
            Dictionary of security headers
        """
        headers = {
            "Content-Security-Policy": self.content_security_policy,
            "Strict-Transport-Security": self.strict_transport_security,
            "X-Content-Type-Options": self.x_content_type_options,
            "X-Frame-Options": self.x_frame_options,
            "X-XSS-Protection": self.x_xss_protection,
            "Referrer-Policy": self.referrer_policy,
            "Permissions-Policy": self.permissions_policy,
            "Cache-Control": self.cache_control,
            "Pragma": self.pragma,
            "Cross-Origin-Opener-Policy": self.cross_origin_opener_policy,
            "Cross-Origin-Embedder-Policy": self.cross_origin_embedder_policy,
            "Cross-Origin-Resource-Policy": self.cross_origin_resource_policy
        }
        
        return headers


class EnhancedSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for adding security headers to responses."""
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[SecurityHeadersConfig] = None,
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            config: Security headers configuration
            exclude_paths: Paths to exclude from adding security headers
        """
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
        self.exclude_paths = exclude_paths or []
        self.headers = self.config.get_headers()
        
        logger.info("Initialized enhanced security headers middleware")
    
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
        # Process the request
        response = await call_next(request)
        
        # Check if path should be excluded
        path = request.url.path
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            return response
        
        # Add security headers
        for name, value in self.headers.items():
            response.headers[name] = value
        
        return response


class LLMSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding LLM-specific security headers to responses."""
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            exclude_paths: Paths to exclude from adding security headers
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or []
        
        logger.info("Initialized LLM security headers middleware")
    
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
        # Process the request
        response = await call_next(request)
        
        # Check if path should be excluded
        path = request.url.path
        if any(path.startswith(exclude_path) for exclude_path in self.exclude_paths):
            return response
        
        # Add LLM-specific security headers
        
        # X-LLM-Provider: Indicates the LLM provider used
        if "X-LLM-Provider" not in response.headers:
            response.headers["X-LLM-Provider"] = "agentor"
        
        # X-LLM-Model: Indicates the LLM model used
        if "X-LLM-Model" not in response.headers and hasattr(response, "model"):
            response.headers["X-LLM-Model"] = getattr(response, "model", "unknown")
        
        # X-LLM-Content-Filtered: Indicates if content was filtered
        if "X-LLM-Content-Filtered" not in response.headers:
            response.headers["X-LLM-Content-Filtered"] = "true"
        
        # X-LLM-Content-Policy: Indicates the content policy applied
        if "X-LLM-Content-Policy" not in response.headers:
            response.headers["X-LLM-Content-Policy"] = "standard"
        
        # X-LLM-Request-ID: Unique identifier for the request
        if "X-LLM-Request-ID" not in response.headers and hasattr(request.state, "request_id"):
            response.headers["X-LLM-Request-ID"] = request.state.request_id
        
        # X-LLM-Processing-Time: Time taken to process the request
        if "X-LLM-Processing-Time" not in response.headers and hasattr(request.state, "processing_time"):
            response.headers["X-LLM-Processing-Time"] = str(request.state.processing_time)
        
        # X-LLM-Token-Count: Number of tokens used
        if "X-LLM-Token-Count" not in response.headers and hasattr(response, "token_count"):
            response.headers["X-LLM-Token-Count"] = str(getattr(response, "token_count", 0))
        
        # X-LLM-Version: Version of the LLM Gateway
        response.headers["X-LLM-Version"] = "1.0.0"
        
        return response


def add_security_headers_to_app(
    app: FastAPI,
    config: Optional[SecurityHeadersConfig] = None,
    exclude_paths: Optional[List[str]] = None,
    add_llm_headers: bool = True
) -> None:
    """
    Add security headers middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        config: Security headers configuration
        exclude_paths: Paths to exclude from adding security headers
        add_llm_headers: Whether to add LLM-specific security headers
    """
    # Add enhanced security headers middleware
    app.add_middleware(
        EnhancedSecurityHeadersMiddleware,
        config=config,
        exclude_paths=exclude_paths
    )
    
    # Add LLM-specific security headers middleware if requested
    if add_llm_headers:
        app.add_middleware(
            LLMSecurityHeadersMiddleware,
            exclude_paths=exclude_paths
        )
    
    logger.info("Added security headers middleware to FastAPI application")
