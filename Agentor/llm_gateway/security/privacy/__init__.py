"""
Privacy controls for the LLM Gateway.

This module provides privacy controls for the LLM Gateway, including PII detection
and redaction, data retention policies, and compliance with privacy regulations.
"""

import logging
from typing import Dict, List, Optional, Set, Union, Any

from .pii import (
    PIIType,
    PIIDetection,
    RedactionMethod,
    RedactionConfig,
    PatternBasedPIIDetector,
    NamedEntityPIIDetector,
    PIIDetector,
    PIIRedactor
)

from .retention import (
    RetentionPeriod,
    DataCategory,
    RetentionPolicy,
    DataRetentionManager
)

from .compliance import (
    PrivacyRegulation,
    ConsentType,
    ConsentRecord,
    DataSubjectRequest,
    DataSubjectRequestStatus,
    DataSubjectRequestRecord,
    PrivacyNotice,
    PrivacyNoticeAcceptance,
    ComplianceManager
)

logger = logging.getLogger(__name__)


__all__ = [
    # PII detection and redaction
    'PIIType',
    'PIIDetection',
    'RedactionMethod',
    'RedactionConfig',
    'PatternBasedPIIDetector',
    'NamedEntityPIIDetector',
    'PIIDetector',
    'PIIRedactor',

    # Data retention
    'RetentionPeriod',
    'DataCategory',
    'RetentionPolicy',
    'DataRetentionManager',

    # Compliance
    'PrivacyRegulation',
    'ConsentType',
    'ConsentRecord',
    'DataSubjectRequest',
    'DataSubjectRequestStatus',
    'DataSubjectRequestRecord',
    'PrivacyNotice',
    'PrivacyNoticeAcceptance',
    'ComplianceManager',
]
