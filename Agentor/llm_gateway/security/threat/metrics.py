"""
Security metrics and alerting for the LLM Gateway.

This module provides security metrics collection and alerting capabilities.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
import asyncio
from enum import Enum
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from agentor.llm_gateway.security.threat.injection import (
    InjectionSeverity,
    InjectionType,
    InjectionDetection
)
from agentor.llm_gateway.security.threat.blocking import (
    BlockingAction,
    BlockingRule
)

logger = logging.getLogger(__name__)


class SecurityMetric:
    """Security metric for tracking security events."""
    
    def __init__(
        self,
        name: str,
        description: str,
        value: int = 0,
        tags: Optional[Dict[str, str]] = None,
        created_at: Optional[float] = None
    ):
        """
        Initialize security metric.
        
        Args:
            name: Metric name
            description: Metric description
            value: Initial value
            tags: Metric tags
            created_at: Creation timestamp
        """
        self.name = name
        self.description = description
        self.value = value
        self.tags = tags or {}
        self.created_at = created_at or time.time()
        self.updated_at = self.created_at
    
    def increment(self, amount: int = 1) -> None:
        """
        Increment metric value.
        
        Args:
            amount: Amount to increment
        """
        self.value += amount
        self.updated_at = time.time()
    
    def reset(self) -> None:
        """Reset metric value to zero."""
        self.value = 0
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metric to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "value": self.value,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class SecurityMetricsCollector:
    """Collector for security metrics."""
    
    # Default metrics
    DEFAULT_METRICS = [
        SecurityMetric(
            name="injection_attempts",
            description="Number of prompt injection attempts detected"
        ),
        SecurityMetric(
            name="blocked_requests",
            description="Number of requests blocked due to security violations"
        ),
        SecurityMetric(
            name="sanitized_requests",
            description="Number of requests sanitized due to security violations"
        ),
        SecurityMetric(
            name="warned_requests",
            description="Number of requests with security warnings"
        ),
        SecurityMetric(
            name="rate_limited_users",
            description="Number of users rate limited due to security violations"
        ),
        SecurityMetric(
            name="banned_users",
            description="Number of users banned due to security violations"
        ),
        SecurityMetric(
            name="security_alerts",
            description="Number of security alerts generated"
        )
    ]
    
    def __init__(
        self,
        metrics: Optional[List[SecurityMetric]] = None,
        enable_default_metrics: bool = True,
        enable_prometheus: bool = False,
        prometheus_prefix: str = "llm_gateway_security"
    ):
        """
        Initialize security metrics collector.
        
        Args:
            metrics: List of security metrics
            enable_default_metrics: Whether to enable default metrics
            enable_prometheus: Whether to enable Prometheus metrics
            prometheus_prefix: Prefix for Prometheus metrics
        """
        self.metrics = {}
        
        # Add default metrics if enabled
        if enable_default_metrics:
            for metric in self.DEFAULT_METRICS:
                self.metrics[metric.name] = metric
        
        # Add custom metrics
        if metrics:
            for metric in metrics:
                self.metrics[metric.name] = metric
        
        # Prometheus integration
        self.enable_prometheus = enable_prometheus
        self.prometheus_prefix = prometheus_prefix
        self.prometheus_metrics = {}
        
        # Initialize Prometheus metrics if enabled
        if self.enable_prometheus:
            try:
                from prometheus_client import Counter, Gauge
                
                # Create Prometheus metrics
                for name, metric in self.metrics.items():
                    prometheus_name = f"{self.prometheus_prefix}_{name}"
                    self.prometheus_metrics[name] = Counter(
                        prometheus_name,
                        metric.description,
                        list(metric.tags.keys())
                    )
                
                logger.info("Initialized Prometheus metrics for security metrics collector")
            except ImportError:
                logger.warning("Failed to import prometheus_client, Prometheus metrics disabled")
                self.enable_prometheus = False
    
    def get_metric(self, name: str) -> Optional[SecurityMetric]:
        """
        Get a security metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Security metric or None if not found
        """
        return self.metrics.get(name)
    
    def add_metric(self, metric: SecurityMetric) -> None:
        """
        Add a security metric.
        
        Args:
            metric: Security metric to add
        """
        self.metrics[metric.name] = metric
        
        # Add Prometheus metric if enabled
        if self.enable_prometheus:
            try:
                from prometheus_client import Counter
                
                prometheus_name = f"{self.prometheus_prefix}_{metric.name}"
                self.prometheus_metrics[metric.name] = Counter(
                    prometheus_name,
                    metric.description,
                    list(metric.tags.keys())
                )
            except ImportError:
                pass
    
    def increment_metric(self, name: str, amount: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a security metric.
        
        Args:
            name: Metric name
            amount: Amount to increment
            tags: Metric tags
        """
        # Get metric
        metric = self.get_metric(name)
        if not metric:
            logger.warning(f"Metric '{name}' not found")
            return
        
        # Increment metric
        metric.increment(amount)
        
        # Update Prometheus metric if enabled
        if self.enable_prometheus and name in self.prometheus_metrics:
            try:
                # Use tags if provided, otherwise use metric tags
                tag_values = tags or metric.tags
                self.prometheus_metrics[name].inc(amount, list(tag_values.values()))
            except Exception as e:
                logger.error(f"Failed to update Prometheus metric '{name}': {e}")
    
    def reset_metric(self, name: str) -> None:
        """
        Reset a security metric.
        
        Args:
            name: Metric name
        """
        # Get metric
        metric = self.get_metric(name)
        if not metric:
            logger.warning(f"Metric '{name}' not found")
            return
        
        # Reset metric
        metric.reset()
    
    def get_all_metrics(self) -> Dict[str, SecurityMetric]:
        """
        Get all security metrics.
        
        Returns:
            Dictionary of security metrics
        """
        return self.metrics.copy()
    
    def track_injection_detection(self, detections: List[InjectionDetection]) -> None:
        """
        Track injection detections.
        
        Args:
            detections: List of injection detections
        """
        if not detections:
            return
        
        # Increment injection attempts metric
        self.increment_metric("injection_attempts", len(detections))
        
        # Track by injection type
        for detection in detections:
            if detection.injection_type:
                # Create metric for injection type if not exists
                type_metric_name = f"injection_type_{detection.injection_type.value}"
                if type_metric_name not in self.metrics:
                    self.add_metric(SecurityMetric(
                        name=type_metric_name,
                        description=f"Number of {detection.injection_type.value} injection attempts detected"
                    ))
                
                # Increment metric
                self.increment_metric(type_metric_name)
            
            # Track by severity
            if detection.severity:
                # Create metric for severity if not exists
                severity_metric_name = f"injection_severity_{detection.severity.value}"
                if severity_metric_name not in self.metrics:
                    self.add_metric(SecurityMetric(
                        name=severity_metric_name,
                        description=f"Number of {detection.severity.value} severity injection attempts detected"
                    ))
                
                # Increment metric
                self.increment_metric(severity_metric_name)
    
    def track_blocking_action(self, action: BlockingAction, rules: List[BlockingRule]) -> None:
        """
        Track blocking actions.
        
        Args:
            action: Blocking action
            rules: Triggered rules
        """
        if not rules:
            return
        
        # Increment metric based on action
        if action == BlockingAction.REJECT:
            self.increment_metric("blocked_requests")
        elif action == BlockingAction.SANITIZE:
            self.increment_metric("sanitized_requests")
        elif action == BlockingAction.WARN:
            self.increment_metric("warned_requests")
        elif action == BlockingAction.RATE_LIMIT:
            self.increment_metric("rate_limited_users")
        elif action == BlockingAction.TEMPORARY_BAN or action == BlockingAction.PERMANENT_BAN:
            self.increment_metric("banned_users")
        
        # Track by rule
        for rule in rules:
            # Create metric for rule if not exists
            rule_metric_name = f"rule_{rule.name}"
            if rule_metric_name not in self.metrics:
                self.add_metric(SecurityMetric(
                    name=rule_metric_name,
                    description=f"Number of times rule '{rule.name}' was triggered"
                ))
            
            # Increment metric
            self.increment_metric(rule_metric_name)


class SecurityAlert:
    """Security alert for notifying about security events."""
    
    def __init__(
        self,
        title: str,
        description: str,
        severity: str,
        source: str,
        timestamp: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize security alert.
        
        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
            source: Alert source
            timestamp: Alert timestamp
            details: Additional details
            tags: Alert tags
        """
        self.title = title
        self.description = description
        self.severity = severity
        self.source = source
        self.timestamp = timestamp or time.time()
        self.details = details or {}
        self.tags = tags or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert alert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "source": self.source,
            "timestamp": self.timestamp,
            "details": self.details,
            "tags": self.tags
        }
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation
        """
        return f"[{self.severity.upper()}] {self.title}: {self.description}"


class SecurityAlertManager:
    """Manager for security alerts."""
    
    def __init__(
        self,
        metrics_collector: Optional[SecurityMetricsCollector] = None,
        alert_handlers: Optional[Dict[str, Callable[[SecurityAlert], Any]]] = None,
        email_config: Optional[Dict[str, Any]] = None,
        slack_webhook_url: Optional[str] = None,
        max_alerts: int = 1000
    ):
        """
        Initialize security alert manager.
        
        Args:
            metrics_collector: Security metrics collector
            alert_handlers: Dictionary of alert handlers
            email_config: Email configuration
            slack_webhook_url: Slack webhook URL
            max_alerts: Maximum number of alerts to store
        """
        self.metrics_collector = metrics_collector
        self.alert_handlers = alert_handlers or {}
        self.email_config = email_config
        self.slack_webhook_url = slack_webhook_url
        self.max_alerts = max_alerts
        
        # Alert storage
        self.alerts = []
        self.alert_lock = asyncio.Lock()
        
        # Add default handlers
        if email_config:
            self.alert_handlers["email"] = self._send_email_alert
        
        if slack_webhook_url:
            self.alert_handlers["slack"] = self._send_slack_alert
        
        # Always add logging handler
        self.alert_handlers["log"] = self._log_alert
    
    async def add_alert(self, alert: SecurityAlert) -> None:
        """
        Add a security alert.
        
        Args:
            alert: Security alert to add
        """
        async with self.alert_lock:
            # Add alert to storage
            self.alerts.append(alert)
            
            # Trim alerts if needed
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
        
        # Update metrics if collector is available
        if self.metrics_collector:
            self.metrics_collector.increment_metric("security_alerts")
        
        # Process alert
        await self._process_alert(alert)
    
    async def create_alert(
        self,
        title: str,
        description: str,
        severity: str,
        source: str,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> SecurityAlert:
        """
        Create and add a security alert.
        
        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
            source: Alert source
            details: Additional details
            tags: Alert tags
            
        Returns:
            Created security alert
        """
        # Create alert
        alert = SecurityAlert(
            title=title,
            description=description,
            severity=severity,
            source=source,
            details=details,
            tags=tags
        )
        
        # Add alert
        await self.add_alert(alert)
        
        return alert
    
    async def get_alerts(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[SecurityAlert]:
        """
        Get security alerts.
        
        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity
            source: Filter by source
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of security alerts
        """
        async with self.alert_lock:
            # Filter alerts
            filtered_alerts = self.alerts
            
            if severity:
                filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
            
            if source:
                filtered_alerts = [a for a in filtered_alerts if a.source == source]
            
            if start_time:
                filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]
            
            if end_time:
                filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]
            
            # Sort by timestamp (newest first)
            filtered_alerts.sort(key=lambda a: a.timestamp, reverse=True)
            
            # Limit results
            return filtered_alerts[:limit]
    
    async def _process_alert(self, alert: SecurityAlert) -> None:
        """
        Process a security alert.
        
        Args:
            alert: Security alert to process
        """
        # Call each alert handler
        for handler_name, handler_func in self.alert_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler_func):
                    await handler_func(alert)
                else:
                    handler_func(alert)
            except Exception as e:
                logger.error(f"Error in alert handler '{handler_name}': {e}")
    
    def _log_alert(self, alert: SecurityAlert) -> None:
        """
        Log a security alert.
        
        Args:
            alert: Security alert to log
        """
        log_message = f"Security Alert: {alert}"
        
        # Log based on severity
        if alert.severity.lower() in ["critical", "high"]:
            logger.error(log_message)
        elif alert.severity.lower() == "medium":
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_email_alert(self, alert: SecurityAlert) -> None:
        """
        Send a security alert via email.
        
        Args:
            alert: Security alert to send
        """
        if not self.email_config:
            logger.warning("Email configuration not provided, cannot send email alert")
            return
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg["From"] = self.email_config.get("from", "security@llmgateway.com")
            msg["To"] = self.email_config.get("to", "admin@llmgateway.com")
            msg["Subject"] = f"[{alert.severity.upper()}] Security Alert: {alert.title}"
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>Security Alert: {alert.title}</h2>
                <p><strong>Severity:</strong> {alert.severity}</p>
                <p><strong>Source:</strong> {alert.source}</p>
                <p><strong>Time:</strong> {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Description:</strong> {alert.description}</p>
                <h3>Details:</h3>
                <pre>{json.dumps(alert.details, indent=2)}</pre>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, "html"))
            
            # Send email
            smtp_server = self.email_config.get("smtp_server", "localhost")
            smtp_port = self.email_config.get("smtp_port", 25)
            smtp_user = self.email_config.get("smtp_user")
            smtp_password = self.email_config.get("smtp_password")
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Sent email alert: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_slack_alert(self, alert: SecurityAlert) -> None:
        """
        Send a security alert to Slack.
        
        Args:
            alert: Security alert to send
        """
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not provided, cannot send Slack alert")
            return
        
        try:
            # Create Slack message
            message = {
                "text": f"*Security Alert: {alert.title}*",
                "attachments": [
                    {
                        "color": self._get_slack_color(alert.severity),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity,
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": datetime.fromtimestamp(alert.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            },
                            {
                                "title": "Description",
                                "value": alert.description,
                                "short": False
                            }
                        ],
                        "footer": "LLM Gateway Security"
                    }
                ]
            }
            
            # Add details if available
            if alert.details:
                message["attachments"][0]["fields"].append({
                    "title": "Details",
                    "value": f"```{json.dumps(alert.details, indent=2)}```",
                    "short": False
                })
            
            # Send to Slack
            response = requests.post(
                self.slack_webhook_url,
                json=message,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to send Slack alert: {response.status_code} {response.text}")
            else:
                logger.info(f"Sent Slack alert: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _get_slack_color(self, severity: str) -> str:
        """
        Get Slack color for severity.
        
        Args:
            severity: Alert severity
            
        Returns:
            Slack color
        """
        severity = severity.lower()
        
        if severity == "critical":
            return "#FF0000"  # Red
        elif severity == "high":
            return "#FFA500"  # Orange
        elif severity == "medium":
            return "#FFFF00"  # Yellow
        elif severity == "low":
            return "#00FF00"  # Green
        else:
            return "#808080"  # Gray
