"""
Threat detection and prevention for the LLM Gateway.

This module provides comprehensive threat detection and prevention features,
including prompt injection detection, automated blocking, and security metrics.
"""

import logging
from typing import Dict, List, Optional, Set, Union, Any

from .injection import (
    InjectionSeverity,
    InjectionType,
    InjectionDetection,
    PatternBasedDetector,
    HeuristicDetector,
    MLBasedDetector,
    PromptInjectionDetector
)

from .blocking import (
    BlockingPolicy,
    BlockingAction,
    BlockingRule,
    BlockingManager
)

from .metrics import (
    SecurityMetric,
    SecurityMetricsCollector,
    SecurityAlert,
    SecurityAlertManager
)

from .middleware import (
    ThreatDetectionMiddleware,
    add_threat_detection_to_app
)

logger = logging.getLogger(__name__)


__all__ = [
    # Injection detection
    'InjectionSeverity',
    'InjectionType',
    'InjectionDetection',
    'PatternBasedDetector',
    'HeuristicDetector',
    'MLBasedDetector',
    'PromptInjectionDetector',

    # Automated blocking
    'BlockingPolicy',
    'BlockingAction',
    'BlockingRule',
    'BlockingManager',

    # Security metrics and alerting
    'SecurityMetric',
    'SecurityMetricsCollector',
    'SecurityAlert',
    'SecurityAlertManager',

    # Middleware
    'ThreatDetectionMiddleware',
    'add_threat_detection_to_app',
]
