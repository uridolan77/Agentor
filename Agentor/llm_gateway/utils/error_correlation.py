"""
Error correlation system for the Agentor framework.

This module provides mechanisms for correlating errors across the system,
identifying patterns, and detecting error storms.
"""

from typing import Dict, Any, List, Optional, Set, Tuple, Callable
import time
import logging
import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib

from agentor.llm_gateway.utils.error_handler import ErrorContext, ErrorSeverity, ErrorCategory

logger = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """An error event with context information."""
    
    error_type: str
    """The type of the error."""
    
    message: str
    """The error message."""
    
    component: str
    """The component where the error occurred."""
    
    operation: str
    """The operation that was being performed."""
    
    severity: ErrorSeverity
    """The severity of the error."""
    
    category: ErrorCategory
    """The category of the error."""
    
    timestamp: float
    """When the error occurred."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the error."""
    
    error_id: str = field(init=False)
    """Unique identifier for this error event."""
    
    def __post_init__(self):
        """Generate a unique error ID based on the error details."""
        # Create a string representation of the error
        error_str = f"{self.error_type}:{self.message}:{self.component}:{self.operation}"
        
        # Generate a hash of the error string
        self.error_id = hashlib.md5(error_str.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error event to a dictionary.
        
        Returns:
            Dictionary representation of the error event
        """
        return {
            'error_id': self.error_id,
            'error_type': self.error_type,
            'message': self.message,
            'component': self.component,
            'operation': self.operation,
            'severity': self.severity.value,
            'category': self.category.value,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_exception(cls, error: Exception, context: ErrorContext) -> 'ErrorEvent':
        """Create an error event from an exception and context.
        
        Args:
            error: The exception
            context: The error context
            
        Returns:
            An ErrorEvent object
        """
        return cls(
            error_type=type(error).__name__,
            message=str(error),
            component=context.component,
            operation=context.operation,
            severity=context.severity,
            category=context.category,
            timestamp=context.timestamp,
            metadata=context.metadata
        )


class ErrorPattern:
    """A pattern of related errors."""
    
    def __init__(self, pattern_id: str, description: str):
        """Initialize the error pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            description: Description of the pattern
        """
        self.pattern_id = pattern_id
        self.description = description
        self.error_ids: Set[str] = set()
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.count = 0
    
    def add_error(self, error_event: ErrorEvent):
        """Add an error to this pattern.
        
        Args:
            error_event: The error event to add
        """
        self.error_ids.add(error_event.error_id)
        self.last_seen = error_event.timestamp
        self.count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error pattern to a dictionary.
        
        Returns:
            Dictionary representation of the error pattern
        """
        return {
            'pattern_id': self.pattern_id,
            'description': self.description,
            'error_ids': list(self.error_ids),
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'count': self.count
        }


class ErrorCorrelator:
    """System for correlating errors and identifying patterns."""
    
    def __init__(
        self,
        window_size: int = 1000,
        time_window: float = 3600.0,
        pattern_threshold: int = 3,
        storm_threshold: int = 10,
        storm_window: float = 60.0
    ):
        """Initialize the error correlator.
        
        Args:
            window_size: Maximum number of error events to keep
            time_window: Time window for error correlation in seconds
            pattern_threshold: Minimum number of similar errors to form a pattern
            storm_threshold: Minimum number of errors in a short time to detect a storm
            storm_window: Time window for storm detection in seconds
        """
        self.errors: Dict[str, ErrorEvent] = {}
        self.error_queue: deque = deque(maxlen=window_size)
        self.patterns: Dict[str, ErrorPattern] = {}
        
        self.window_size = window_size
        self.time_window = time_window
        self.pattern_threshold = pattern_threshold
        self.storm_threshold = storm_threshold
        self.storm_window = storm_window
        
        self.component_errors: Dict[str, List[str]] = defaultdict(list)
        self.operation_errors: Dict[str, List[str]] = defaultdict(list)
        self.category_errors: Dict[ErrorCategory, List[str]] = defaultdict(list)
        
        self.storm_callbacks: List[Callable[[str, List[ErrorEvent]], None]] = []
        self.pattern_callbacks: List[Callable[[ErrorPattern], None]] = []
        
        self.lock = asyncio.Lock()
    
    async def add_error(self, error: Exception, context: ErrorContext):
        """Add an error event to the correlator.
        
        Args:
            error: The exception
            context: The error context
        """
        # Create an error event
        error_event = ErrorEvent.from_exception(error, context)
        
        async with self.lock:
            # Store the error
            self.errors[error_event.error_id] = error_event
            self.error_queue.append(error_event.error_id)
            
            # Update indices
            self.component_errors[error_event.component].append(error_event.error_id)
            self.operation_errors[error_event.operation].append(error_event.error_id)
            self.category_errors[error_event.category].append(error_event.error_id)
            
            # Check for patterns and storms
            await self._check_for_patterns(error_event)
            await self._check_for_storms(error_event)
            
            # Prune old errors
            await self._prune_old_errors()
    
    async def _check_for_patterns(self, error_event: ErrorEvent):
        """Check for error patterns.
        
        Args:
            error_event: The new error event
        """
        # Look for similar errors in the same component
        component_pattern_id = f"component:{error_event.component}:{error_event.error_type}"
        component_similar = [
            err_id for err_id in self.component_errors[error_event.component]
            if self.errors[err_id].error_type == error_event.error_type
        ]
        
        if len(component_similar) >= self.pattern_threshold:
            # Create or update the pattern
            if component_pattern_id not in self.patterns:
                pattern = ErrorPattern(
                    pattern_id=component_pattern_id,
                    description=f"Multiple {error_event.error_type} errors in component {error_event.component}"
                )
                self.patterns[component_pattern_id] = pattern
                
                # Notify pattern callbacks
                for callback in self.pattern_callbacks:
                    callback(pattern)
            
            # Add the error to the pattern
            self.patterns[component_pattern_id].add_error(error_event)
        
        # Look for similar errors in the same operation
        operation_pattern_id = f"operation:{error_event.operation}:{error_event.error_type}"
        operation_similar = [
            err_id for err_id in self.operation_errors[error_event.operation]
            if self.errors[err_id].error_type == error_event.error_type
        ]
        
        if len(operation_similar) >= self.pattern_threshold:
            # Create or update the pattern
            if operation_pattern_id not in self.patterns:
                pattern = ErrorPattern(
                    pattern_id=operation_pattern_id,
                    description=f"Multiple {error_event.error_type} errors in operation {error_event.operation}"
                )
                self.patterns[operation_pattern_id] = pattern
                
                # Notify pattern callbacks
                for callback in self.pattern_callbacks:
                    callback(pattern)
            
            # Add the error to the pattern
            self.patterns[operation_pattern_id].add_error(error_event)
        
        # Look for similar errors in the same category
        category_pattern_id = f"category:{error_event.category.value}:{error_event.error_type}"
        category_similar = [
            err_id for err_id in self.category_errors[error_event.category]
            if self.errors[err_id].error_type == error_event.error_type
        ]
        
        if len(category_similar) >= self.pattern_threshold:
            # Create or update the pattern
            if category_pattern_id not in self.patterns:
                pattern = ErrorPattern(
                    pattern_id=category_pattern_id,
                    description=f"Multiple {error_event.error_type} errors in category {error_event.category.value}"
                )
                self.patterns[category_pattern_id] = pattern
                
                # Notify pattern callbacks
                for callback in self.pattern_callbacks:
                    callback(pattern)
            
            # Add the error to the pattern
            self.patterns[category_pattern_id].add_error(error_event)
    
    async def _check_for_storms(self, error_event: ErrorEvent):
        """Check for error storms.
        
        Args:
            error_event: The new error event
        """
        # Check for component storms
        component_storm_id = f"storm:component:{error_event.component}"
        component_recent = [
            err_id for err_id in self.component_errors[error_event.component]
            if time.time() - self.errors[err_id].timestamp <= self.storm_window
        ]
        
        if len(component_recent) >= self.storm_threshold:
            # Get the recent errors
            recent_errors = [self.errors[err_id] for err_id in component_recent]
            
            # Notify storm callbacks
            for callback in self.storm_callbacks:
                callback(component_storm_id, recent_errors)
        
        # Check for operation storms
        operation_storm_id = f"storm:operation:{error_event.operation}"
        operation_recent = [
            err_id for err_id in self.operation_errors[error_event.operation]
            if time.time() - self.errors[err_id].timestamp <= self.storm_window
        ]
        
        if len(operation_recent) >= self.storm_threshold:
            # Get the recent errors
            recent_errors = [self.errors[err_id] for err_id in operation_recent]
            
            # Notify storm callbacks
            for callback in self.storm_callbacks:
                callback(operation_storm_id, recent_errors)
        
        # Check for category storms
        category_storm_id = f"storm:category:{error_event.category.value}"
        category_recent = [
            err_id for err_id in self.category_errors[error_event.category]
            if time.time() - self.errors[err_id].timestamp <= self.storm_window
        ]
        
        if len(category_recent) >= self.storm_threshold:
            # Get the recent errors
            recent_errors = [self.errors[err_id] for err_id in category_recent]
            
            # Notify storm callbacks
            for callback in self.storm_callbacks:
                callback(category_storm_id, recent_errors)
    
    async def _prune_old_errors(self):
        """Remove errors that are outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        # Find errors to remove
        to_remove = []
        for error_id, error in self.errors.items():
            if error.timestamp < cutoff_time:
                to_remove.append(error_id)
        
        # Remove the errors
        for error_id in to_remove:
            error = self.errors.pop(error_id, None)
            if error:
                # Remove from indices
                if error_id in self.component_errors[error.component]:
                    self.component_errors[error.component].remove(error_id)
                if error_id in self.operation_errors[error.operation]:
                    self.operation_errors[error.operation].remove(error_id)
                if error_id in self.category_errors[error.category]:
                    self.category_errors[error.category].remove(error_id)
    
    def register_storm_callback(self, callback: Callable[[str, List[ErrorEvent]], None]):
        """Register a callback for error storms.
        
        Args:
            callback: Function to call when an error storm is detected
        """
        self.storm_callbacks.append(callback)
    
    def register_pattern_callback(self, callback: Callable[[ErrorPattern], None]):
        """Register a callback for error patterns.
        
        Args:
            callback: Function to call when an error pattern is detected
        """
        self.pattern_callbacks.append(callback)
    
    def get_error(self, error_id: str) -> Optional[ErrorEvent]:
        """Get an error by ID.
        
        Args:
            error_id: The ID of the error
            
        Returns:
            The error event, or None if not found
        """
        return self.errors.get(error_id)
    
    def get_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        """Get a pattern by ID.
        
        Args:
            pattern_id: The ID of the pattern
            
        Returns:
            The error pattern, or None if not found
        """
        return self.patterns.get(pattern_id)
    
    def get_errors_by_component(self, component: str) -> List[ErrorEvent]:
        """Get all errors for a component.
        
        Args:
            component: The component name
            
        Returns:
            List of error events
        """
        return [self.errors[err_id] for err_id in self.component_errors[component]]
    
    def get_errors_by_operation(self, operation: str) -> List[ErrorEvent]:
        """Get all errors for an operation.
        
        Args:
            operation: The operation name
            
        Returns:
            List of error events
        """
        return [self.errors[err_id] for err_id in self.operation_errors[operation]]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorEvent]:
        """Get all errors for a category.
        
        Args:
            category: The error category
            
        Returns:
            List of error events
        """
        return [self.errors[err_id] for err_id in self.category_errors[category]]
    
    def get_all_patterns(self) -> List[ErrorPattern]:
        """Get all error patterns.
        
        Returns:
            List of error patterns
        """
        return list(self.patterns.values())
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorEvent]:
        """Get the most recent errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error events
        """
        recent_ids = list(self.error_queue)[-limit:]
        return [self.errors[err_id] for err_id in recent_ids if err_id in self.errors]


# Create a global error correlator instance
error_correlator = ErrorCorrelator()


def log_storm_callback(storm_id: str, errors: List[ErrorEvent]):
    """Log error storms.
    
    Args:
        storm_id: The ID of the storm
        errors: The errors in the storm
    """
    logger.warning(
        f"Error storm detected: {storm_id} with {len(errors)} errors in the last minute. "
        f"First error: {errors[0].message}"
    )


def log_pattern_callback(pattern: ErrorPattern):
    """Log error patterns.
    
    Args:
        pattern: The error pattern
    """
    logger.warning(
        f"Error pattern detected: {pattern.pattern_id} - {pattern.description}. "
        f"Count: {pattern.count}"
    )


async def setup_error_correlation():
    """Set up the error correlation system."""
    # Register callbacks
    error_correlator.register_storm_callback(log_storm_callback)
    error_correlator.register_pattern_callback(log_pattern_callback)
    
    logger.info("Error correlation system initialized")
