"""
Tamper-evident audit trails for the LLM Gateway.

This module provides tamper-evident audit trails for tracking security-relevant
events and actions in the LLM Gateway.
"""

import time
import logging
import json
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import threading
import asyncio
import uuid

from agentor.llm_gateway.security.audit.logging import (
    LogLevel,
    LogCategory,
    LogEvent,
    SecureLogger
)

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Actions for audit events."""
    CREATE = "create"  # Create a resource
    READ = "read"  # Read a resource
    UPDATE = "update"  # Update a resource
    DELETE = "delete"  # Delete a resource
    LOGIN = "login"  # User login
    LOGOUT = "logout"  # User logout
    AUTHENTICATE = "authenticate"  # User authentication
    AUTHORIZE = "authorize"  # User authorization
    GENERATE = "generate"  # Generate content
    MODERATE = "moderate"  # Moderate content
    BLOCK = "block"  # Block content or user
    ALLOW = "allow"  # Allow content or user
    CONFIGURE = "configure"  # Configure system
    ADMIN = "admin"  # Administrative action
    CUSTOM = "custom"  # Custom action


class AuditResource(Enum):
    """Resources for audit events."""
    USER = "user"  # User resource
    SESSION = "session"  # Session resource
    TOKEN = "token"  # Token resource
    PROMPT = "prompt"  # Prompt resource
    RESPONSE = "response"  # Response resource
    MODEL = "model"  # Model resource
    SYSTEM = "system"  # System resource
    SETTING = "setting"  # Setting resource
    PERMISSION = "permission"  # Permission resource
    ROLE = "role"  # Role resource
    LOG = "log"  # Log resource
    AUDIT = "audit"  # Audit resource
    CUSTOM = "custom"  # Custom resource


class AuditOutcome(Enum):
    """Outcomes for audit events."""
    SUCCESS = "success"  # Action succeeded
    FAILURE = "failure"  # Action failed
    ERROR = "error"  # Action resulted in error
    DENIED = "denied"  # Action was denied
    PENDING = "pending"  # Action is pending
    UNKNOWN = "unknown"  # Outcome is unknown


class AuditEvent:
    """Event for audit trails."""
    
    def __init__(
        self,
        action: AuditAction,
        resource: AuditResource,
        outcome: AuditOutcome,
        timestamp: Optional[float] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audit event.
        
        Args:
            action: Audit action
            resource: Audit resource
            outcome: Audit outcome
            timestamp: Event timestamp
            user_id: User ID
            resource_id: Resource ID
            request_id: Request ID
            source: Event source
            details: Event details
            metadata: Additional metadata
        """
        self.action = action
        self.resource = resource
        self.outcome = outcome
        self.timestamp = timestamp or time.time()
        self.user_id = user_id
        self.resource_id = resource_id
        self.request_id = request_id
        self.source = source or "llm_gateway"
        self.details = details or {}
        self.metadata = metadata or {}
        self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "event_id": self.event_id,
            "action": self.action.value,
            "resource": self.resource.value,
            "outcome": self.outcome.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "request_id": self.request_id,
            "source": self.source,
            "details": self.details,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert event to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """
        Create event from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Audit event
        """
        event = cls(
            action=AuditAction(data["action"]),
            resource=AuditResource(data["resource"]),
            outcome=AuditOutcome(data["outcome"]),
            timestamp=data["timestamp"],
            user_id=data.get("user_id"),
            resource_id=data.get("resource_id"),
            request_id=data.get("request_id"),
            source=data.get("source"),
            details=data.get("details", {}),
            metadata=data.get("metadata", {})
        )
        
        if "event_id" in data:
            event.event_id = data["event_id"]
        
        return event
    
    @classmethod
    def from_json(cls, json_str: str) -> "AuditEvent":
        """
        Create event from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Audit event
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class AuditTrail:
    """Tamper-evident audit trail."""
    
    def __init__(
        self,
        secure_logger: Optional[SecureLogger] = None,
        hmac_key: Optional[bytes] = None,
        storage_backend: Optional[str] = None,
        storage_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize audit trail.
        
        Args:
            secure_logger: Secure logger
            hmac_key: HMAC key for tamper-evident audit trail
            storage_backend: Storage backend ("file", "database", "memory")
            storage_config: Storage configuration
        """
        self.secure_logger = secure_logger
        self.hmac_key = hmac_key or os.urandom(32)
        self.storage_backend = storage_backend or "memory"
        self.storage_config = storage_config or {}
        
        # Previous event hash
        self.previous_hash = None
        
        # Storage for audit events
        self.events = []
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    def _calculate_hash(self, event_json: str) -> str:
        """
        Calculate HMAC hash for tamper-evident audit trail.
        
        Args:
            event_json: JSON string of audit event
            
        Returns:
            HMAC hash
        """
        # Create HMAC
        h = hmac.new(self.hmac_key, digestmod=hashlib.sha256)
        
        # Add previous hash if available
        if self.previous_hash:
            h.update(self.previous_hash.encode())
        
        # Add event data
        h.update(event_json.encode())
        
        # Return base64-encoded hash
        return base64.b64encode(h.digest()).decode()
    
    async def add_event(self, event: AuditEvent) -> str:
        """
        Add an event to the audit trail.
        
        Args:
            event: Audit event
            
        Returns:
            Event hash
        """
        # Convert event to JSON
        event_json = event.to_json()
        
        # Calculate hash
        async with self.lock:
            event_hash = self._calculate_hash(event_json)
            
            # Update previous hash
            self.previous_hash = event_hash
            
            # Add hash to event metadata
            event.metadata["hash"] = event_hash
            
            # Store event
            if self.storage_backend == "memory":
                self.events.append(event)
            elif self.storage_backend == "file":
                # Implement file storage
                pass
            elif self.storage_backend == "database":
                # Implement database storage
                pass
        
        # Log event if secure logger is available
        if self.secure_logger:
            self.secure_logger.audit(
                message=f"{event.action.value} {event.resource.value} ({event.outcome.value})",
                user_id=event.user_id,
                request_id=event.request_id,
                source=event.source,
                metadata={
                    "audit_event": event.to_dict(),
                    "hash": event_hash
                }
            )
        
        return event_hash
    
    async def get_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[AuditResource] = None,
        outcome: Optional[AuditOutcome] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Get events from the audit trail.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            user_id: User ID filter
            action: Action filter
            resource: Resource filter
            outcome: Outcome filter
            limit: Maximum number of events to return
            
        Returns:
            List of audit events
        """
        async with self.lock:
            # Get all events
            all_events = self.events.copy()
        
        # Filter events
        filtered_events = all_events
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if action:
            filtered_events = [e for e in filtered_events if e.action == action]
        
        if resource:
            filtered_events = [e for e in filtered_events if e.resource == resource]
        
        if outcome:
            filtered_events = [e for e in filtered_events if e.outcome == outcome]
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Limit results
        return filtered_events[:limit]
    
    async def verify_integrity(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify integrity of the audit trail.
        
        Returns:
            Tuple of (integrity status, list of invalid entries)
        """
        async with self.lock:
            # Get all events
            all_events = self.events.copy()
        
        if not all_events:
            return True, []
        
        # Verify each event
        previous_hash = None
        invalid_entries = []
        
        for i, event in enumerate(all_events):
            try:
                # Get hash from metadata
                if "hash" not in event.metadata:
                    invalid_entries.append({
                        "index": i,
                        "event": event.to_dict(),
                        "reason": "No hash"
                    })
                    continue
                
                event_hash = event.metadata.pop("hash")
                
                # Convert to JSON
                event_json = event.to_json()
                
                # Calculate expected hash
                h = hmac.new(self.hmac_key, digestmod=hashlib.sha256)
                
                # Add previous hash if available
                if previous_hash:
                    h.update(previous_hash.encode())
                
                # Add event data
                h.update(event_json.encode())
                
                # Get expected hash
                expected_hash = base64.b64encode(h.digest()).decode()
                
                # Compare hashes
                if event_hash != expected_hash:
                    invalid_entries.append({
                        "index": i,
                        "event": event.to_dict(),
                        "reason": "Invalid hash",
                        "expected": expected_hash,
                        "actual": event_hash
                    })
                
                # Update previous hash
                previous_hash = event_hash
                
                # Restore hash to metadata
                event.metadata["hash"] = event_hash
            
            except Exception as e:
                invalid_entries.append({
                    "index": i,
                    "event": event.to_dict() if hasattr(event, "to_dict") else str(event),
                    "reason": f"Error: {str(e)}"
                })
        
        # Return integrity status
        return len(invalid_entries) == 0, invalid_entries
    
    async def export_events(
        self,
        format: str = "json",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource: Optional[AuditResource] = None,
        outcome: Optional[AuditOutcome] = None
    ) -> str:
        """
        Export events from the audit trail.
        
        Args:
            format: Export format ("json", "csv")
            start_time: Start time filter
            end_time: End time filter
            user_id: User ID filter
            action: Action filter
            resource: Resource filter
            outcome: Outcome filter
            
        Returns:
            Exported events
        """
        # Get filtered events
        events = await self.get_events(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            action=action,
            resource=resource,
            outcome=outcome,
            limit=10000  # Large limit for export
        )
        
        if format == "json":
            # Export as JSON
            return json.dumps([e.to_dict() for e in events], indent=2)
        
        elif format == "csv":
            # Export as CSV
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "event_id",
                "timestamp",
                "action",
                "resource",
                "outcome",
                "user_id",
                "resource_id",
                "request_id",
                "source",
                "details",
                "metadata"
            ])
            
            # Write events
            for event in events:
                writer.writerow([
                    event.event_id,
                    event.timestamp,
                    event.action.value,
                    event.resource.value,
                    event.outcome.value,
                    event.user_id or "",
                    event.resource_id or "",
                    event.request_id or "",
                    event.source or "",
                    json.dumps(event.details),
                    json.dumps(event.metadata)
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def close(self) -> None:
        """Close audit trail."""
        pass
