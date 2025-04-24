"""
Security module for the LLM Gateway.

This module provides comprehensive security features for the LLM Gateway, including:
- Input validation and sanitization
- Secure session management
- Process isolation
- Enhanced authentication
- Transport security hardening
- Request/response protection
- Threat detection
- Data privacy controls
- Secure logging
"""

import logging
from typing import Dict, Any, Optional, List

from agentor.llm_gateway.security.validation import (
    validate_llm_request,
    sanitize_content,
    InputValidationError
)
from agentor.llm_gateway.security.session import (
    SessionSecurityManager,
    SecureSessionPool
)
from agentor.llm_gateway.security.process import (
    SecureProcessManager,
    ProcessSecurityError
)
from agentor.llm_gateway.security.middleware import (
    EnhancedInputValidationMiddleware,
    SecurityHeadersMiddleware
)
from agentor.llm_gateway.security.auth import (
    JWTAuth,
    JWTConfig,
    Role,
    Permission,
    AuthenticationError,
    AuthorizationError,
    RBACMiddleware
)
from agentor.llm_gateway.security.users import (
    User,
    UserManager,
    UserError
)
from agentor.llm_gateway.security.models import (
    ModelConfig,
    ModelPermission,
    ModelPermissionManager,
    ModelPermissionError
)
from agentor.llm_gateway.security.transport import (
    TLSConfig,
    CertificatePinningManager,
    SecureTransportManager,
    TransportSecurityError
)
from agentor.llm_gateway.security.headers import (
    SecurityHeadersConfig,
    EnhancedSecurityHeadersMiddleware,
    LLMSecurityHeadersMiddleware,
    add_security_headers_to_app
)
from agentor.llm_gateway.security.protection import (
    ContentFilteringConfig,
    ContentFilter,
    TimingAttackMitigation,
    ContentFilteringMiddleware,
    TimingAttackMitigationMiddleware,
    add_protection_to_app
)
from agentor.llm_gateway.security.threat import (
    # Injection detection
    InjectionSeverity,
    InjectionType,
    InjectionDetection,
    PatternBasedDetector,
    HeuristicDetector,
    MLBasedDetector,
    PromptInjectionDetector,

    # Automated blocking
    BlockingAction,
    BlockingRule,
    BlockingPolicy,
    BlockingManager,

    # Security metrics and alerting
    SecurityMetric,
    SecurityMetricsCollector,
    SecurityAlert,
    SecurityAlertManager,

    # Middleware
    ThreatDetectionMiddleware,
    add_threat_detection_to_app
)

from agentor.llm_gateway.security.privacy import (
    # PII detection and redaction
    PIIType,
    PIIDetection,
    RedactionMethod,
    RedactionConfig,
    PatternBasedPIIDetector,
    NamedEntityPIIDetector,
    PIIDetector,
    PIIRedactor,

    # Data retention
    RetentionPeriod,
    DataCategory,
    RetentionPolicy,
    DataRetentionManager,

    # Compliance
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

from agentor.llm_gateway.security.audit import (
    # Secure logging
    LogLevel,
    LogCategory,
    LogEvent,
    SecureLogHandler,
    AsyncLogHandler,
    SecureLogger,

    # Audit trails
    AuditAction,
    AuditResource,
    AuditOutcome,
    AuditEvent,
    AuditTrail,

    # Security dashboards
    TimeRange,
    MetricAggregation,
    DashboardMetric,
    SecurityDashboard,
    DashboardManager
)

logger = logging.getLogger(__name__)


__all__ = [
    # Validation
    'validate_llm_request',
    'sanitize_content',
    'InputValidationError',

    # Session management
    'SessionSecurityManager',
    'SecureSessionPool',

    # Process isolation
    'SecureProcessManager',
    'ProcessSecurityError',

    # Authentication and authorization
    'JWTAuth',
    'JWTConfig',
    'Role',
    'Permission',
    'AuthenticationError',
    'AuthorizationError',
    'RBACMiddleware',

    # User management
    'User',
    'UserManager',
    'UserError',

    # Model permissions
    'ModelConfig',
    'ModelPermission',
    'ModelPermissionManager',
    'ModelPermissionError',

    # Transport security
    'TLSConfig',
    'CertificatePinningManager',
    'SecureTransportManager',
    'TransportSecurityError',

    # Security headers
    'SecurityHeadersConfig',
    'EnhancedSecurityHeadersMiddleware',
    'LLMSecurityHeadersMiddleware',
    'add_security_headers_to_app',

    # Request/response protection
    'ContentFilteringConfig',
    'ContentFilter',
    'TimingAttackMitigation',
    'ContentFilteringMiddleware',
    'TimingAttackMitigationMiddleware',
    'add_protection_to_app',

    # Threat detection and prevention
    'InjectionSeverity',
    'InjectionType',
    'InjectionDetection',
    'PatternBasedDetector',
    'HeuristicDetector',
    'MLBasedDetector',
    'PromptInjectionDetector',
    'BlockingAction',
    'BlockingRule',
    'BlockingPolicy',
    'BlockingManager',
    'SecurityMetric',
    'SecurityMetricsCollector',
    'SecurityAlert',
    'SecurityAlertManager',
    'ThreatDetectionMiddleware',
    'add_threat_detection_to_app',

    # Privacy controls
    'PIIType',
    'PIIDetection',
    'RedactionMethod',
    'RedactionConfig',
    'PatternBasedPIIDetector',
    'NamedEntityPIIDetector',
    'PIIDetector',
    'PIIRedactor',
    'RetentionPeriod',
    'DataCategory',
    'RetentionPolicy',
    'DataRetentionManager',
    'PrivacyRegulation',
    'ConsentType',
    'ConsentRecord',
    'DataSubjectRequest',
    'DataSubjectRequestStatus',
    'DataSubjectRequestRecord',
    'PrivacyNotice',
    'PrivacyNoticeAcceptance',
    'ComplianceManager',

    # Audit and logging
    'LogLevel',
    'LogCategory',
    'LogEvent',
    'SecureLogHandler',
    'AsyncLogHandler',
    'SecureLogger',
    'AuditAction',
    'AuditResource',
    'AuditOutcome',
    'AuditEvent',
    'AuditTrail',
    'TimeRange',
    'MetricAggregation',
    'DashboardMetric',
    'SecurityDashboard',
    'DashboardManager',

    # Middleware
    'EnhancedInputValidationMiddleware',
    'SecurityHeadersMiddleware',
]


def setup_security(app=None):
    """
    Set up security features for the LLM Gateway.

    Args:
        app: FastAPI application (optional)

    Returns:
        Dictionary of security components
    """
    logger.info("Setting up LLM Gateway security features")

    # Initialize basic security components
    session_manager = SessionSecurityManager()
    session_pool = SecureSessionPool(security_manager=session_manager)
    process_manager = SecureProcessManager()

    # Initialize authentication and authorization components
    jwt_config = JWTConfig()
    jwt_auth = JWTAuth(config=jwt_config)
    user_manager = UserManager(jwt_auth=jwt_auth)
    model_permission_manager = ModelPermissionManager()

    # Initialize transport security components
    tls_config = TLSConfig()
    cert_pinning_manager = CertificatePinningManager()
    secure_transport_manager = SecureTransportManager(
        tls_config=tls_config,
        cert_pinning_manager=cert_pinning_manager
    )

    # Initialize security headers components
    security_headers_config = SecurityHeadersConfig()

    # Initialize request/response protection components
    content_filtering_config = ContentFilteringConfig()
    content_filter = ContentFilter(config=content_filtering_config)
    timing_attack_mitigation = TimingAttackMitigation()

    # Initialize threat detection and prevention components
    injection_detector = PromptInjectionDetector(
        pattern_detector=PatternBasedDetector(),
        heuristic_detector=HeuristicDetector(),
        ml_detector=MLBasedDetector(),
        enable_pattern=True,
        enable_heuristic=True,
        enable_ml=False  # ML detection is disabled by default
    )

    blocking_policy = BlockingPolicy()
    blocking_manager = BlockingManager(
        policy=blocking_policy,
        injection_detector=injection_detector
    )

    metrics_collector = SecurityMetricsCollector()
    alert_manager = SecurityAlertManager(
        metrics_collector=metrics_collector
    )

    # Initialize privacy controls
    pii_detector = PIIDetector(
        pattern_detector=PatternBasedPIIDetector(),
        ner_detector=NamedEntityPIIDetector(enable_spacy=False),
        enable_pattern=True,
        enable_ner=False  # NER detection is disabled by default
    )

    pii_redactor = PIIRedactor(
        detector=pii_detector,
        default_config=RedactionConfig(method=RedactionMethod.REPLACE)
    )

    retention_manager = DataRetentionManager()

    compliance_manager = ComplianceManager(
        pii_detector=pii_detector,
        pii_redactor=pii_redactor,
        retention_manager=retention_manager
    )

    # Initialize audit and logging components
    secure_logger = SecureLogger(
        handler=AsyncLogHandler(log_file="logs/security.log")
    )

    audit_trail = AuditTrail(
        secure_logger=secure_logger,
        storage_backend="memory"
    )

    dashboard_manager = DashboardManager(
        secure_logger=secure_logger,
        audit_trail=audit_trail
    )

    # Create default users and models
    try:
        # This is a synchronous context, so we can't use await directly
        # In a real implementation, you would use an async context or run these in a background task
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(user_manager.create_default_users())
        loop.run_until_complete(model_permission_manager.create_default_models())
    except Exception as e:
        logger.error(f"Error creating default users and models: {e}")

    # Add middleware if app is provided
    if app:
        logger.info("Adding security middleware to FastAPI application")

        # Add basic security middleware
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(EnhancedInputValidationMiddleware)
        app.add_middleware(RBACMiddleware, jwt_auth=jwt_auth)

        # Add enhanced security headers middleware
        app.add_middleware(
            EnhancedSecurityHeadersMiddleware,
            config=security_headers_config
        )

        # Add LLM-specific security headers middleware
        app.add_middleware(LLMSecurityHeadersMiddleware)

        # Add content filtering middleware
        app.add_middleware(
            ContentFilteringMiddleware,
            config=content_filtering_config
        )

        # Add timing attack mitigation middleware
        app.add_middleware(
            TimingAttackMitigationMiddleware,
            mitigation=timing_attack_mitigation
        )

        # Add threat detection middleware
        app.add_middleware(
            ThreatDetectionMiddleware,
            blocking_manager=blocking_manager,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager
        )

    # Return security components
    return {
        # Basic security components
        "session_manager": session_manager,
        "session_pool": session_pool,
        "process_manager": process_manager,

        # Authentication and authorization components
        "jwt_auth": jwt_auth,
        "user_manager": user_manager,
        "model_permission_manager": model_permission_manager,

        # Transport security components
        "tls_config": tls_config,
        "cert_pinning_manager": cert_pinning_manager,
        "secure_transport_manager": secure_transport_manager,

        # Security headers components
        "security_headers_config": security_headers_config,

        # Request/response protection components
        "content_filtering_config": content_filtering_config,
        "content_filter": content_filter,
        "timing_attack_mitigation": timing_attack_mitigation,

        # Threat detection and prevention components
        "injection_detector": injection_detector,
        "blocking_policy": blocking_policy,
        "blocking_manager": blocking_manager,
        "metrics_collector": metrics_collector,
        "alert_manager": alert_manager,

        # Privacy controls
        "pii_detector": pii_detector,
        "pii_redactor": pii_redactor,
        "retention_manager": retention_manager,
        "compliance_manager": compliance_manager,

        # Audit and logging components
        "secure_logger": secure_logger,
        "audit_trail": audit_trail,
        "dashboard_manager": dashboard_manager
    }


async def cleanup_security(security_components: Dict[str, Any]):
    """
    Clean up security components.

    Args:
        security_components: Dictionary of security components
    """
    logger.info("Cleaning up LLM Gateway security components")

    # Clean up session pool
    if "session_pool" in security_components:
        await security_components["session_pool"].close()

    # Clean up process manager
    if "process_manager" in security_components:
        await security_components["process_manager"].cleanup()

    # Clean up blocking manager
    if "blocking_manager" in security_components:
        await security_components["blocking_manager"].cleanup_expired_data()

    # Clean up retention manager
    if "retention_manager" in security_components:
        await security_components["retention_manager"].purge_expired_data()

    # Clean up secure logger
    if "secure_logger" in security_components:
        security_components["secure_logger"].close()

    # Clean up audit trail
    if "audit_trail" in security_components:
        security_components["audit_trail"].close()

    # Save user data (in a real implementation, this would save to a database)
    if "user_manager" in security_components:
        try:
            users_data = await security_components["user_manager"].save_users()
            logger.info(f"Saved {len(users_data)} users")
        except Exception as e:
            logger.error(f"Error saving users: {e}")

    # Save model data (in a real implementation, this would save to a database)
    if "model_permission_manager" in security_components:
        try:
            models_data = await security_components["model_permission_manager"].save_models()
            logger.info(f"Saved {len(models_data)} models")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
