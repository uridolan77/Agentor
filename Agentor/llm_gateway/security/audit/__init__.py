"""
Audit and logging for the LLM Gateway.

This module provides secure logging, tamper-evident audit trails, and security
dashboards for the LLM Gateway.
"""

import logging
from typing import Dict, List, Optional, Set, Union, Any

from agentor.llm_gateway.security.audit.logging import (
    LogLevel,
    LogCategory,
    LogEvent,
    SecureLogHandler,
    AsyncLogHandler,
    SecureLogger
)

from agentor.llm_gateway.security.audit.trail import (
    AuditAction,
    AuditResource,
    AuditOutcome,
    AuditEvent,
    AuditTrail
)

from agentor.llm_gateway.security.audit.dashboard import (
    TimeRange,
    MetricAggregation,
    DashboardMetric,
    SecurityDashboard,
    DashboardManager
)

logger = logging.getLogger(__name__)


__all__ = [
    # Secure logging
    'LogLevel',
    'LogCategory',
    'LogEvent',
    'SecureLogHandler',
    'AsyncLogHandler',
    'SecureLogger',
    
    # Audit trails
    'AuditAction',
    'AuditResource',
    'AuditOutcome',
    'AuditEvent',
    'AuditTrail',
    
    # Security dashboards
    'TimeRange',
    'MetricAggregation',
    'DashboardMetric',
    'SecurityDashboard',
    'DashboardManager',
]
