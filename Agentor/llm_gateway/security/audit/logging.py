"""
Secure logging for the LLM Gateway.

This module provides secure logging capabilities, including tamper-evident logs,
structured logging, and log filtering.
"""

import os
import time
import logging
import json
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Set, Union, Any, Tuple, Callable, TextIO
from enum import Enum
from datetime import datetime, timedelta
import threading
import queue
import uuid

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for secure logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"  # Special level for security events


class LogCategory(Enum):
    """Categories of log events."""
    SYSTEM = "system"  # System events
    SECURITY = "security"  # Security events
    USER = "user"  # User events
    REQUEST = "request"  # Request events
    RESPONSE = "response"  # Response events
    PERFORMANCE = "performance"  # Performance events
    ERROR = "error"  # Error events
    AUDIT = "audit"  # Audit events
    CUSTOM = "custom"  # Custom events


class LogEvent:
    """Event for secure logging."""
    
    def __init__(
        self,
        message: str,
        level: LogLevel,
        category: LogCategory,
        timestamp: Optional[float] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize log event.
        
        Args:
            message: Log message
            level: Log level
            category: Log category
            timestamp: Event timestamp
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
        """
        self.message = message
        self.level = level
        self.category = category
        self.timestamp = timestamp or time.time()
        self.user_id = user_id
        self.request_id = request_id
        self.source = source or "llm_gateway"
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
            "message": self.message,
            "level": self.level.value,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "source": self.source,
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
    def from_dict(cls, data: Dict[str, Any]) -> "LogEvent":
        """
        Create event from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Log event
        """
        event = cls(
            message=data["message"],
            level=LogLevel(data["level"]),
            category=LogCategory(data["category"]),
            timestamp=data["timestamp"],
            user_id=data.get("user_id"),
            request_id=data.get("request_id"),
            source=data.get("source"),
            metadata=data.get("metadata", {})
        )
        
        if "event_id" in data:
            event.event_id = data["event_id"]
        
        return event
    
    @classmethod
    def from_json(cls, json_str: str) -> "LogEvent":
        """
        Create event from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Log event
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class SecureLogHandler:
    """Handler for secure logs."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        hmac_key: Optional[bytes] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        filter_levels: Optional[Set[LogLevel]] = None,
        filter_categories: Optional[Set[LogCategory]] = None,
        include_metadata: bool = True
    ):
        """
        Initialize secure log handler.
        
        Args:
            log_file: Path to log file
            hmac_key: HMAC key for tamper-evident logs
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
            filter_levels: Set of log levels to include
            filter_categories: Set of log categories to include
            include_metadata: Whether to include metadata in logs
        """
        self.log_file = log_file
        self.hmac_key = hmac_key or os.urandom(32)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.filter_levels = filter_levels
        self.filter_categories = filter_categories
        self.include_metadata = include_metadata
        
        # File handle
        self.file_handle = None
        self.file_size = 0
        
        # Previous log hash
        self.previous_hash = None
        
        # Open log file if specified
        if self.log_file:
            self._open_log_file()
    
    def _open_log_file(self) -> None:
        """Open log file."""
        # Check if file exists
        file_exists = os.path.exists(self.log_file)
        
        # Open file
        self.file_handle = open(self.log_file, "a")
        
        # Get file size
        self.file_size = os.path.getsize(self.log_file) if file_exists else 0
        
        # Get previous hash if file exists
        if file_exists and self.file_size > 0:
            # Read last line to get previous hash
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    try:
                        log_data = json.loads(last_line)
                        if "hash" in log_data:
                            self.previous_hash = log_data["hash"]
                    except json.JSONDecodeError:
                        pass
    
    def _rotate_log_file(self) -> None:
        """Rotate log file if it exceeds maximum size."""
        if not self.log_file or not self.file_handle:
            return
        
        # Check if file size exceeds maximum
        if self.file_size > self.max_file_size:
            # Close current file
            self.file_handle.close()
            
            # Rotate files
            for i in range(self.backup_count - 1, 0, -1):
                src = f"{self.log_file}.{i}"
                dst = f"{self.log_file}.{i + 1}"
                
                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)
            
            # Rename current file
            if os.path.exists(self.log_file):
                if os.path.exists(f"{self.log_file}.1"):
                    os.remove(f"{self.log_file}.1")
                os.rename(self.log_file, f"{self.log_file}.1")
            
            # Open new file
            self._open_log_file()
    
    def _calculate_hash(self, event_json: str) -> str:
        """
        Calculate HMAC hash for tamper-evident logs.
        
        Args:
            event_json: JSON string of log event
            
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
    
    def handle(self, event: LogEvent) -> None:
        """
        Handle log event.
        
        Args:
            event: Log event
        """
        # Filter by level
        if self.filter_levels and event.level not in self.filter_levels:
            return
        
        # Filter by category
        if self.filter_categories and event.category not in self.filter_categories:
            return
        
        # Convert event to dictionary
        event_dict = event.to_dict()
        
        # Remove metadata if not included
        if not self.include_metadata:
            event_dict.pop("metadata", None)
        
        # Convert to JSON
        event_json = json.dumps(event_dict)
        
        # Calculate hash
        event_hash = self._calculate_hash(event_json)
        
        # Add hash to event
        event_dict["hash"] = event_hash
        
        # Update previous hash
        self.previous_hash = event_hash
        
        # Convert to JSON with hash
        event_json_with_hash = json.dumps(event_dict)
        
        # Write to log file
        if self.file_handle:
            self.file_handle.write(event_json_with_hash + "\n")
            self.file_handle.flush()
            
            # Update file size
            self.file_size += len(event_json_with_hash) + 1
            
            # Rotate log file if needed
            self._rotate_log_file()
        
        # Also log to standard logger
        log_message = f"{event.level.value.upper()}: {event.message}"
        if event.user_id:
            log_message += f" [User: {event.user_id}]"
        if event.request_id:
            log_message += f" [Request: {event.request_id}]"
        
        # Map log level to standard logger level
        if event.level == LogLevel.DEBUG:
            logger.debug(log_message)
        elif event.level == LogLevel.INFO:
            logger.info(log_message)
        elif event.level == LogLevel.WARNING:
            logger.warning(log_message)
        elif event.level == LogLevel.ERROR:
            logger.error(log_message)
        elif event.level == LogLevel.CRITICAL:
            logger.critical(log_message)
        elif event.level == LogLevel.SECURITY:
            logger.warning(f"SECURITY: {log_message}")
    
    def close(self) -> None:
        """Close log handler."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def verify_log_integrity(self, log_file: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify integrity of log file.
        
        Args:
            log_file: Path to log file (defaults to handler's log file)
            
        Returns:
            Tuple of (integrity status, list of invalid entries)
        """
        log_file = log_file or self.log_file
        if not log_file or not os.path.exists(log_file):
            return False, []
        
        # Read log file
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        if not lines:
            return True, []
        
        # Verify each entry
        previous_hash = None
        invalid_entries = []
        
        for i, line in enumerate(lines):
            try:
                # Parse JSON
                entry = json.loads(line.strip())
                
                # Skip entries without hash
                if "hash" not in entry:
                    invalid_entries.append({
                        "line": i + 1,
                        "entry": entry,
                        "reason": "No hash"
                    })
                    continue
                
                # Get hash
                entry_hash = entry.pop("hash")
                
                # Convert to JSON
                entry_json = json.dumps(entry)
                
                # Calculate expected hash
                h = hmac.new(self.hmac_key, digestmod=hashlib.sha256)
                
                # Add previous hash if available
                if previous_hash:
                    h.update(previous_hash.encode())
                
                # Add entry data
                h.update(entry_json.encode())
                
                # Get expected hash
                expected_hash = base64.b64encode(h.digest()).decode()
                
                # Compare hashes
                if entry_hash != expected_hash:
                    invalid_entries.append({
                        "line": i + 1,
                        "entry": entry,
                        "reason": "Invalid hash",
                        "expected": expected_hash,
                        "actual": entry_hash
                    })
                
                # Update previous hash
                previous_hash = entry_hash
            
            except json.JSONDecodeError:
                invalid_entries.append({
                    "line": i + 1,
                    "entry": line.strip(),
                    "reason": "Invalid JSON"
                })
        
        # Return integrity status
        return len(invalid_entries) == 0, invalid_entries


class AsyncLogHandler(SecureLogHandler):
    """Asynchronous handler for secure logs."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        hmac_key: Optional[bytes] = None,
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        filter_levels: Optional[Set[LogLevel]] = None,
        filter_categories: Optional[Set[LogCategory]] = None,
        include_metadata: bool = True,
        queue_size: int = 1000
    ):
        """
        Initialize asynchronous log handler.
        
        Args:
            log_file: Path to log file
            hmac_key: HMAC key for tamper-evident logs
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
            filter_levels: Set of log levels to include
            filter_categories: Set of log categories to include
            include_metadata: Whether to include metadata in logs
            queue_size: Size of event queue
        """
        super().__init__(
            log_file=log_file,
            hmac_key=hmac_key,
            max_file_size=max_file_size,
            backup_count=backup_count,
            filter_levels=filter_levels,
            filter_categories=filter_categories,
            include_metadata=include_metadata
        )
        
        # Event queue
        self.queue = queue.Queue(maxsize=queue_size)
        
        # Worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.running = True
        self.worker_thread.start()
    
    def _worker(self) -> None:
        """Worker thread for processing log events."""
        while self.running:
            try:
                # Get event from queue
                event = self.queue.get(timeout=1.0)
                
                # Process event
                super().handle(event)
                
                # Mark task as done
                self.queue.task_done()
            
            except queue.Empty:
                # Queue is empty, continue
                continue
            
            except Exception as e:
                # Log error
                logger.error(f"Error processing log event: {e}")
    
    def handle(self, event: LogEvent) -> None:
        """
        Handle log event asynchronously.
        
        Args:
            event: Log event
        """
        try:
            # Add event to queue
            self.queue.put(event, block=False)
        except queue.Full:
            # Queue is full, log warning
            logger.warning("Log event queue is full, dropping event")
    
    def close(self) -> None:
        """Close log handler."""
        # Stop worker thread
        self.running = False
        
        # Wait for worker thread to finish
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Close file handle
        super().close()


class SecureLogger:
    """Secure logger for the LLM Gateway."""
    
    def __init__(
        self,
        handler: Optional[SecureLogHandler] = None,
        default_source: str = "llm_gateway",
        include_thread_id: bool = True
    ):
        """
        Initialize secure logger.
        
        Args:
            handler: Log handler
            default_source: Default event source
            include_thread_id: Whether to include thread ID in metadata
        """
        self.handler = handler or AsyncLogHandler()
        self.default_source = default_source
        self.include_thread_id = include_thread_id
    
    def log(
        self,
        message: str,
        level: LogLevel,
        category: LogCategory,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log an event.
        
        Args:
            message: Log message
            level: Log level
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        # Create metadata
        event_metadata = metadata or {}
        
        # Add thread ID if enabled
        if self.include_thread_id:
            event_metadata["thread_id"] = threading.get_ident()
        
        # Create event
        event = LogEvent(
            message=message,
            level=level,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source or self.default_source,
            metadata=event_metadata
        )
        
        # Handle event
        self.handler.handle(event)
        
        return event
    
    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log a debug event.
        
        Args:
            message: Log message
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.DEBUG,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log an info event.
        
        Args:
            message: Log message
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.INFO,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log a warning event.
        
        Args:
            message: Log message
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.WARNING,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.ERROR,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log an error event.
        
        Args:
            message: Log message
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.ERROR,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def critical(
        self,
        message: str,
        category: LogCategory = LogCategory.ERROR,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log a critical event.
        
        Args:
            message: Log message
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.CRITICAL,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def security(
        self,
        message: str,
        category: LogCategory = LogCategory.SECURITY,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log a security event.
        
        Args:
            message: Log message
            category: Log category
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.SECURITY,
            category=category,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def audit(
        self,
        message: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LogEvent:
        """
        Log an audit event.
        
        Args:
            message: Log message
            user_id: User ID
            request_id: Request ID
            source: Event source
            metadata: Additional metadata
            
        Returns:
            Log event
        """
        return self.log(
            message=message,
            level=LogLevel.INFO,
            category=LogCategory.AUDIT,
            user_id=user_id,
            request_id=request_id,
            source=source,
            metadata=metadata
        )
    
    def close(self) -> None:
        """Close logger."""
        if self.handler:
            self.handler.close()
    
    def verify_log_integrity(self, log_file: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify integrity of log file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            Tuple of (integrity status, list of invalid entries)
        """
        if self.handler:
            return self.handler.verify_log_integrity(log_file)
        
        return False, []
