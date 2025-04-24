"""
Security middleware for the LLM Gateway.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from agentor.llm_gateway.security.validation import validate_llm_request, InputValidationError

logger = logging.getLogger(__name__)


class EnhancedInputValidationMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for validating and sanitizing input."""
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Only validate POST requests to specific endpoints
        if request.method == "POST" and request.url.path in ["/generate", "/chat/completions"]:
            try:
                # Read request body
                body = await request.body()
                
                # Parse JSON
                try:
                    request_data = json.loads(body)
                except json.JSONDecodeError:
                    return Response(
                        content=json.dumps({
                            "error": "Invalid JSON in request body"
                        }),
                        status_code=400,
                        media_type="application/json"
                    )
                
                # Validate and sanitize request
                try:
                    validated_data = validate_llm_request(request_data)
                    
                    # Create a new request with the validated data
                    # This is a bit of a hack, but it works
                    new_body = json.dumps(validated_data).encode()
                    request._body = new_body
                    
                except InputValidationError as e:
                    return Response(
                        content=json.dumps({
                            "error": str(e)
                        }),
                        status_code=400,
                        media_type="application/json"
                    )
                
            except Exception as e:
                logger.error(f"Error in input validation middleware: {e}")
                return Response(
                    content=json.dumps({
                        "error": "Internal server error during request validation"
                    }),
                    status_code=500,
                    media_type="application/json"
                )
        
        # Process the request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers to responses."""
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware.
        
        Args:
            app: The ASGI application
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Process the request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        
        return response
