"""
Middleware for threat detection and prevention in the LLM Gateway.

This module provides FastAPI middleware for integrating threat detection
and prevention features into the LLM Gateway.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Callable
import asyncio
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .injection import (
    InjectionSeverity,
    InjectionType,
    InjectionDetection,
    PromptInjectionDetector
)
from .blocking import (
    BlockingAction,
    BlockingRule,
    BlockingPolicy,
    BlockingManager
)
from .metrics import (
    SecurityMetric,
    SecurityMetricsCollector,
    SecurityAlert,
    SecurityAlertManager
)

logger = logging.getLogger(__name__)


class ThreatDetectionMiddleware(BaseHTTPMiddleware):
    """Middleware for threat detection and prevention."""

    def __init__(
        self,
        app: ASGIApp,
        blocking_manager: Optional[BlockingManager] = None,
        metrics_collector: Optional[SecurityMetricsCollector] = None,
        alert_manager: Optional[SecurityAlertManager] = None,
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            blocking_manager: Blocking manager
            metrics_collector: Security metrics collector
            alert_manager: Security alert manager
            exclude_paths: Paths to exclude from threat detection
        """
        super().__init__(app)
        self.blocking_manager = blocking_manager or BlockingManager()
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.exclude_paths = exclude_paths or []

        logger.info("Initialized threat detection middleware")

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

        # Only process POST requests to specific endpoints
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

            # Get user ID from request
            user_id = None
            if hasattr(request.state, "user") and hasattr(request.state.user, "id"):
                user_id = request.state.user.id

            # Check request for threats
            start_time = time.time()
            modified_data, allowed, triggered_rules = await self.blocking_manager.check_request(
                request_data=request_data,
                user_id=user_id
            )
            processing_time = time.time() - start_time

            # Track metrics if collector is available
            if self.metrics_collector and triggered_rules:
                # Get highest action from triggered rules
                highest_action = None
                for rule in triggered_rules:
                    if highest_action is None or rule.action.value > highest_action.value:
                        highest_action = rule.action

                # Track action
                if highest_action:
                    self.metrics_collector.track_blocking_action(highest_action, triggered_rules)

            # Create alert if alert manager is available
            if self.alert_manager and triggered_rules:
                # Get highest severity rule
                highest_severity_rule = max(triggered_rules, key=lambda r: r.action.value)

                # Create alert
                await self.alert_manager.create_alert(
                    title=f"Security rule triggered: {highest_severity_rule.name}",
                    description=highest_severity_rule.description,
                    severity="high" if highest_severity_rule.action in [BlockingAction.REJECT, BlockingAction.TEMPORARY_BAN, BlockingAction.PERMANENT_BAN] else "medium",
                    source="threat_detection_middleware",
                    details={
                        "user_id": user_id,
                        "path": path,
                        "rules": [rule.name for rule in triggered_rules],
                        "processing_time": processing_time
                    }
                )

            # If request is not allowed, return error response
            if not allowed:
                return Response(
                    content=json.dumps({
                        "error": "Request blocked due to security violation",
                        "details": "The request was blocked by the security system due to potential security threats."
                    }),
                    status_code=403,
                    media_type="application/json"
                )

            # If request was modified, create a new request with the modified data
            if modified_data != request_data:
                # Create a new request with the modified data
                new_body = json.dumps(modified_data).encode()
                request._body = new_body

        # Process the request
        return await call_next(request)


def add_threat_detection_to_app(
    app: FastAPI,
    blocking_manager: Optional[BlockingManager] = None,
    metrics_collector: Optional[SecurityMetricsCollector] = None,
    alert_manager: Optional[SecurityAlertManager] = None,
    exclude_paths: Optional[List[str]] = None
) -> None:
    """
    Add threat detection middleware to a FastAPI application.

    Args:
        app: FastAPI application
        blocking_manager: Blocking manager
        metrics_collector: Security metrics collector
        alert_manager: Security alert manager
        exclude_paths: Paths to exclude from threat detection
    """
    # Add threat detection middleware
    app.add_middleware(
        ThreatDetectionMiddleware,
        blocking_manager=blocking_manager,
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        exclude_paths=exclude_paths
    )

    logger.info("Added threat detection middleware to FastAPI application")
