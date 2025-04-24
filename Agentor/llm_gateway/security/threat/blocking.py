"""
Automated blocking for suspicious activity in the LLM Gateway.

This module provides automated blocking of suspicious activity,
including blocking policies, rules, and actions.
"""

import re
import time
import logging
import json
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Pattern, Callable
import asyncio
from enum import Enum
from datetime import datetime, timedelta

from agentor.llm_gateway.security.threat.injection import (
    InjectionSeverity,
    InjectionType,
    InjectionDetection,
    PromptInjectionDetector
)

logger = logging.getLogger(__name__)


class BlockingAction(Enum):
    """Actions to take when blocking suspicious activity."""
    LOG = "log"  # Log the activity but take no action
    WARN = "warn"  # Warn the user but allow the request
    SANITIZE = "sanitize"  # Sanitize the request
    REJECT = "reject"  # Reject the request
    RATE_LIMIT = "rate_limit"  # Rate limit the user
    TEMPORARY_BAN = "temporary_ban"  # Temporarily ban the user
    PERMANENT_BAN = "permanent_ban"  # Permanently ban the user


class BlockingRule:
    """Rule for blocking suspicious activity."""
    
    def __init__(
        self,
        name: str,
        description: str,
        action: BlockingAction,
        severity_threshold: Optional[InjectionSeverity] = None,
        injection_types: Optional[Set[InjectionType]] = None,
        confidence_threshold: float = 0.8,
        custom_condition: Optional[Callable[[List[InjectionDetection], Dict[str, Any]], bool]] = None,
        custom_action: Optional[Callable[[List[InjectionDetection], Dict[str, Any]], Dict[str, Any]]] = None,
        enabled: bool = True
    ):
        """
        Initialize blocking rule.
        
        Args:
            name: Rule name
            description: Rule description
            action: Action to take when rule is triggered
            severity_threshold: Minimum severity to trigger rule
            injection_types: Injection types to trigger rule
            confidence_threshold: Minimum confidence to trigger rule
            custom_condition: Custom condition function
            custom_action: Custom action function
            enabled: Whether the rule is enabled
        """
        self.name = name
        self.description = description
        self.action = action
        self.severity_threshold = severity_threshold
        self.injection_types = injection_types or set()
        self.confidence_threshold = confidence_threshold
        self.custom_condition = custom_condition
        self.custom_action = custom_action
        self.enabled = enabled
    
    def matches(self, detections: List[InjectionDetection], context: Dict[str, Any] = None) -> bool:
        """
        Check if rule matches detections.
        
        Args:
            detections: List of injection detections
            context: Additional context
            
        Returns:
            True if rule matches, False otherwise
        """
        if not self.enabled or not detections:
            return False
        
        # Check custom condition if provided
        if self.custom_condition:
            return self.custom_condition(detections, context or {})
        
        # Check severity threshold
        if self.severity_threshold:
            severity_values = {
                InjectionSeverity.LOW: 1,
                InjectionSeverity.MEDIUM: 2,
                InjectionSeverity.HIGH: 3,
                InjectionSeverity.CRITICAL: 4
            }
            threshold_value = severity_values.get(self.severity_threshold, 0)
            
            # Check if any detection meets the severity threshold
            for detection in detections:
                if detection.severity and severity_values.get(detection.severity, 0) >= threshold_value:
                    # Check confidence threshold
                    if detection.confidence >= self.confidence_threshold:
                        # Check injection types if specified
                        if not self.injection_types or detection.injection_type in self.injection_types:
                            return True
        
        return False
    
    def apply(self, detections: List[InjectionDetection], request_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply rule action to request.
        
        Args:
            detections: List of injection detections
            request_data: Request data
            context: Additional context
            
        Returns:
            Modified request data or None if request should be rejected
        """
        # Use custom action if provided
        if self.custom_action:
            return self.custom_action(detections, request_data, context or {})
        
        # Apply standard actions
        if self.action == BlockingAction.LOG:
            # Log the activity but take no action
            logger.warning(f"Blocking rule '{self.name}' triggered: {self.description}")
            for detection in detections:
                logger.warning(f"  Detection: {detection}")
            return request_data
        
        elif self.action == BlockingAction.WARN:
            # Warn the user but allow the request
            logger.warning(f"Blocking rule '{self.name}' triggered: {self.description}")
            for detection in detections:
                logger.warning(f"  Detection: {detection}")
            
            # Add warning to request metadata
            if "metadata" not in request_data:
                request_data["metadata"] = {}
            if "warnings" not in request_data["metadata"]:
                request_data["metadata"]["warnings"] = []
            
            request_data["metadata"]["warnings"].append({
                "rule": self.name,
                "description": self.description,
                "detections": [detection.to_dict() for detection in detections]
            })
            
            return request_data
        
        elif self.action == BlockingAction.SANITIZE:
            # Sanitize the request
            logger.warning(f"Blocking rule '{self.name}' triggered: {self.description}")
            for detection in detections:
                logger.warning(f"  Detection: {detection}")
            
            # Create a copy of the request data
            sanitized_data = request_data.copy()
            
            # Sanitize prompt if present
            if "prompt" in sanitized_data:
                for detection in detections:
                    if detection.matched_text and detection.position:
                        start, end = detection.position
                        prompt = sanitized_data["prompt"]
                        sanitized_data["prompt"] = prompt[:start] + "[FILTERED]" + prompt[end:]
            
            # Sanitize messages if present
            if "messages" in sanitized_data and isinstance(sanitized_data["messages"], list):
                for i, message in enumerate(sanitized_data["messages"]):
                    if isinstance(message, dict) and "content" in message and isinstance(message["content"], str):
                        for detection in detections:
                            if detection.matched_text and detection.position:
                                start, end = detection.position
                                content = message["content"]
                                if start < len(content) and end <= len(content):
                                    sanitized_data["messages"][i]["content"] = content[:start] + "[FILTERED]" + content[end:]
            
            # Add sanitization info to request metadata
            if "metadata" not in sanitized_data:
                sanitized_data["metadata"] = {}
            if "sanitized" not in sanitized_data["metadata"]:
                sanitized_data["metadata"]["sanitized"] = []
            
            sanitized_data["metadata"]["sanitized"].append({
                "rule": self.name,
                "description": self.description,
                "detections": [detection.to_dict() for detection in detections]
            })
            
            return sanitized_data
        
        elif self.action == BlockingAction.REJECT:
            # Reject the request
            logger.warning(f"Blocking rule '{self.name}' triggered: {self.description}")
            for detection in detections:
                logger.warning(f"  Detection: {detection}")
            
            # Return None to indicate request should be rejected
            return None
        
        elif self.action in [BlockingAction.RATE_LIMIT, BlockingAction.TEMPORARY_BAN, BlockingAction.PERMANENT_BAN]:
            # These actions require user tracking, which is handled by the BlockingManager
            logger.warning(f"Blocking rule '{self.name}' triggered: {self.description}")
            for detection in detections:
                logger.warning(f"  Detection: {detection}")
            
            # Return the original request data
            # The BlockingManager will handle the actual blocking
            return request_data
        
        # Default: return original request data
        return request_data


class BlockingPolicy:
    """Policy for blocking suspicious activity."""
    
    # Default rules
    DEFAULT_RULES = [
        BlockingRule(
            name="critical_injection",
            description="Block critical severity prompt injections",
            action=BlockingAction.REJECT,
            severity_threshold=InjectionSeverity.CRITICAL,
            confidence_threshold=0.9
        ),
        BlockingRule(
            name="high_severity_injection",
            description="Sanitize high severity prompt injections",
            action=BlockingAction.SANITIZE,
            severity_threshold=InjectionSeverity.HIGH,
            confidence_threshold=0.85
        ),
        BlockingRule(
            name="medium_severity_injection",
            description="Warn about medium severity prompt injections",
            action=BlockingAction.WARN,
            severity_threshold=InjectionSeverity.MEDIUM,
            confidence_threshold=0.8
        ),
        BlockingRule(
            name="jailbreak_attempt",
            description="Block jailbreak attempts",
            action=BlockingAction.REJECT,
            injection_types={InjectionType.JAILBREAK},
            confidence_threshold=0.85
        ),
        BlockingRule(
            name="system_prompt_leak",
            description="Block system prompt leaking attempts",
            action=BlockingAction.REJECT,
            injection_types={InjectionType.PROMPT_LEAKING},
            confidence_threshold=0.85
        )
    ]
    
    def __init__(
        self,
        rules: Optional[List[BlockingRule]] = None,
        enable_default_rules: bool = True
    ):
        """
        Initialize blocking policy.
        
        Args:
            rules: List of blocking rules
            enable_default_rules: Whether to enable default rules
        """
        self.rules = []
        
        # Add default rules if enabled
        if enable_default_rules:
            self.rules.extend(self.DEFAULT_RULES)
        
        # Add custom rules
        if rules:
            self.rules.extend(rules)
    
    def add_rule(self, rule: BlockingRule) -> None:
        """
        Add a blocking rule.
        
        Args:
            rule: Blocking rule to add
        """
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a blocking rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
    
    def get_rule(self, rule_name: str) -> Optional[BlockingRule]:
        """
        Get a blocking rule by name.
        
        Args:
            rule_name: Name of rule to get
            
        Returns:
            Blocking rule or None if not found
        """
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def enable_rule(self, rule_name: str) -> bool:
        """
        Enable a blocking rule.
        
        Args:
            rule_name: Name of rule to enable
            
        Returns:
            True if rule was enabled, False if not found
        """
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = True
            return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """
        Disable a blocking rule.
        
        Args:
            rule_name: Name of rule to disable
            
        Returns:
            True if rule was disabled, False if not found
        """
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = False
            return True
        return False
    
    def apply_rules(
        self,
        detections: List[InjectionDetection],
        request_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], List[BlockingRule]]:
        """
        Apply blocking rules to request.
        
        Args:
            detections: List of injection detections
            request_data: Request data
            context: Additional context
            
        Returns:
            Tuple of (modified request data, list of triggered rules)
        """
        if not detections:
            return request_data, []
        
        # Create a copy of the request data
        modified_data = request_data.copy()
        triggered_rules = []
        
        # Apply each rule
        for rule in self.rules:
            if rule.matches(detections, context):
                triggered_rules.append(rule)
                
                # Apply rule action
                result = rule.apply(detections, modified_data, context)
                
                # If result is None, request should be rejected
                if result is None:
                    return None, triggered_rules
                
                # Update modified data
                modified_data = result
        
        return modified_data, triggered_rules


class BlockingManager:
    """Manager for blocking suspicious activity."""
    
    def __init__(
        self,
        policy: Optional[BlockingPolicy] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
        rate_limit_window_seconds: int = 60,
        rate_limit_max_requests: int = 5,
        temporary_ban_duration_seconds: int = 3600,
        enable_user_tracking: bool = True
    ):
        """
        Initialize blocking manager.
        
        Args:
            policy: Blocking policy
            injection_detector: Prompt injection detector
            rate_limit_window_seconds: Window for rate limiting in seconds
            rate_limit_max_requests: Maximum requests in rate limit window
            temporary_ban_duration_seconds: Duration of temporary bans in seconds
            enable_user_tracking: Whether to enable user tracking
        """
        self.policy = policy or BlockingPolicy()
        self.injection_detector = injection_detector or PromptInjectionDetector()
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.rate_limit_max_requests = rate_limit_max_requests
        self.temporary_ban_duration_seconds = temporary_ban_duration_seconds
        self.enable_user_tracking = enable_user_tracking
        
        # User tracking data
        self.user_requests = {}  # user_id -> list of request timestamps
        self.user_violations = {}  # user_id -> list of violation timestamps
        self.user_bans = {}  # user_id -> ban expiration timestamp
        self.user_lock = asyncio.Lock()
    
    async def check_request(
        self,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], bool, List[BlockingRule]]:
        """
        Check a request for suspicious activity.
        
        Args:
            request_data: Request data
            user_id: User ID
            context: Additional context
            
        Returns:
            Tuple of (modified request data, allowed flag, triggered rules)
        """
        # Track user request if enabled
        if self.enable_user_tracking and user_id:
            await self._track_user_request(user_id)
            
            # Check if user is banned
            if await self._is_user_banned(user_id):
                logger.warning(f"User {user_id} is banned, rejecting request")
                return None, False, []
            
            # Check if user is rate limited
            if await self._is_user_rate_limited(user_id):
                logger.warning(f"User {user_id} is rate limited, rejecting request")
                return None, False, []
        
        # Detect prompt injections
        detections = await self.injection_detector.detect_request(request_data)
        
        # If no detections, allow request
        if not detections:
            return request_data, True, []
        
        # Apply blocking policy
        modified_data, triggered_rules = self.policy.apply_rules(detections, request_data, context)
        
        # If modified data is None, request should be rejected
        if modified_data is None:
            # Track violation if enabled
            if self.enable_user_tracking and user_id:
                await self._track_user_violation(user_id, triggered_rules)
            
            return None, False, triggered_rules
        
        # Check if any triggered rules require user tracking
        if self.enable_user_tracking and user_id:
            for rule in triggered_rules:
                if rule.action in [BlockingAction.RATE_LIMIT, BlockingAction.TEMPORARY_BAN, BlockingAction.PERMANENT_BAN]:
                    await self._track_user_violation(user_id, [rule])
                    
                    # Apply user tracking actions
                    if rule.action == BlockingAction.RATE_LIMIT:
                        # Rate limiting is handled by _is_user_rate_limited
                        pass
                    elif rule.action == BlockingAction.TEMPORARY_BAN:
                        await self._ban_user(user_id, temporary=True)
                    elif rule.action == BlockingAction.PERMANENT_BAN:
                        await self._ban_user(user_id, temporary=False)
        
        return modified_data, True, triggered_rules
    
    async def _track_user_request(self, user_id: str) -> None:
        """
        Track a user request.
        
        Args:
            user_id: User ID
        """
        async with self.user_lock:
            # Initialize user requests if not exists
            if user_id not in self.user_requests:
                self.user_requests[user_id] = []
            
            # Add request timestamp
            self.user_requests[user_id].append(time.time())
            
            # Remove old requests
            cutoff = time.time() - self.rate_limit_window_seconds
            self.user_requests[user_id] = [t for t in self.user_requests[user_id] if t >= cutoff]
    
    async def _track_user_violation(self, user_id: str, rules: List[BlockingRule]) -> None:
        """
        Track a user violation.
        
        Args:
            user_id: User ID
            rules: Triggered rules
        """
        async with self.user_lock:
            # Initialize user violations if not exists
            if user_id not in self.user_violations:
                self.user_violations[user_id] = []
            
            # Add violation timestamp and rules
            self.user_violations[user_id].append({
                "timestamp": time.time(),
                "rules": [rule.name for rule in rules]
            })
    
    async def _is_user_rate_limited(self, user_id: str) -> bool:
        """
        Check if a user is rate limited.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user is rate limited, False otherwise
        """
        async with self.user_lock:
            # Check if user has requests
            if user_id not in self.user_requests:
                return False
            
            # Count recent requests
            cutoff = time.time() - self.rate_limit_window_seconds
            recent_requests = [t for t in self.user_requests[user_id] if t >= cutoff]
            
            # Check if user has too many recent requests
            return len(recent_requests) > self.rate_limit_max_requests
    
    async def _ban_user(self, user_id: str, temporary: bool = True) -> None:
        """
        Ban a user.
        
        Args:
            user_id: User ID
            temporary: Whether the ban is temporary
        """
        async with self.user_lock:
            if temporary:
                # Set ban expiration timestamp
                self.user_bans[user_id] = time.time() + self.temporary_ban_duration_seconds
                logger.warning(f"User {user_id} temporarily banned for {self.temporary_ban_duration_seconds} seconds")
            else:
                # Set ban expiration to far future (effectively permanent)
                self.user_bans[user_id] = time.time() + 10 * 365 * 24 * 60 * 60  # 10 years
                logger.warning(f"User {user_id} permanently banned")
    
    async def _is_user_banned(self, user_id: str) -> bool:
        """
        Check if a user is banned.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user is banned, False otherwise
        """
        async with self.user_lock:
            # Check if user is banned
            if user_id not in self.user_bans:
                return False
            
            # Check if ban has expired
            if self.user_bans[user_id] <= time.time():
                # Remove expired ban
                del self.user_bans[user_id]
                return False
            
            return True
    
    async def unban_user(self, user_id: str) -> bool:
        """
        Unban a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user was unbanned, False if not banned
        """
        async with self.user_lock:
            if user_id in self.user_bans:
                del self.user_bans[user_id]
                logger.info(f"User {user_id} unbanned")
                return True
            return False
    
    async def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get user status.
        
        Args:
            user_id: User ID
            
        Returns:
            User status
        """
        async with self.user_lock:
            status = {
                "user_id": user_id,
                "banned": False,
                "ban_expiration": None,
                "rate_limited": False,
                "request_count": 0,
                "violation_count": 0
            }
            
            # Check if user is banned
            if user_id in self.user_bans:
                if self.user_bans[user_id] > time.time():
                    status["banned"] = True
                    status["ban_expiration"] = self.user_bans[user_id]
            
            # Check if user is rate limited
            if user_id in self.user_requests:
                cutoff = time.time() - self.rate_limit_window_seconds
                recent_requests = [t for t in self.user_requests[user_id] if t >= cutoff]
                status["request_count"] = len(recent_requests)
                status["rate_limited"] = len(recent_requests) > self.rate_limit_max_requests
            
            # Count violations
            if user_id in self.user_violations:
                status["violation_count"] = len(self.user_violations[user_id])
            
            return status
    
    async def cleanup_expired_data(self) -> None:
        """Clean up expired user data."""
        async with self.user_lock:
            # Clean up expired bans
            now = time.time()
            expired_bans = [user_id for user_id, expiration in self.user_bans.items() if expiration <= now]
            for user_id in expired_bans:
                del self.user_bans[user_id]
            
            # Clean up old requests
            cutoff = now - self.rate_limit_window_seconds
            for user_id in list(self.user_requests.keys()):
                self.user_requests[user_id] = [t for t in self.user_requests[user_id] if t >= cutoff]
                if not self.user_requests[user_id]:
                    del self.user_requests[user_id]
            
            # Clean up old violations (keep for 30 days)
            violation_cutoff = now - 30 * 24 * 60 * 60
            for user_id in list(self.user_violations.keys()):
                self.user_violations[user_id] = [v for v in self.user_violations[user_id] if v["timestamp"] >= violation_cutoff]
                if not self.user_violations[user_id]:
                    del self.user_violations[user_id]
