import logging
import traceback
import sys
import asyncio
from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Union, Awaitable
from enum import Enum
from functools import wraps
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Define type variables for type hints
T = TypeVar('T')
E = TypeVar('E', bound=Exception)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RESOURCE = "resource"
    INTERNAL = "internal"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"


class ErrorContext:
    """Context information for an error."""
    
    def __init__(
        self,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the error context.
        
        Args:
            component: The component where the error occurred
            operation: The operation that was being performed
            severity: The severity of the error
            category: The category of the error
            metadata: Additional metadata about the error
        """
        self.component = component
        self.operation = operation
        self.severity = severity
        self.category = category
        self.metadata = metadata or {}
        self.timestamp = asyncio.get_event_loop().time()


class EnhancedError(Exception):
    """Base class for enhanced errors with context information."""
    
    def __init__(
        self,
        message: str,
        context: ErrorContext,
        cause: Optional[Exception] = None
    ):
        """Initialize the enhanced error.
        
        Args:
            message: The error message
            context: The error context
            cause: The original exception that caused this error
        """
        self.message = message
        self.context = context
        self.cause = cause
        
        # Construct the full error message
        full_message = f"{message} [component={context.component}, operation={context.operation}]"
        if cause:
            full_message += f" Caused by: {type(cause).__name__}: {str(cause)}"
        
        super().__init__(full_message)


class ErrorHandler:
    """Centralized error handler for the application."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_handlers: Dict[Type[Exception], List[Callable]] = {}
        self.global_handlers: List[Callable] = []
    
    def register_handler(self, exception_type: Type[E], handler: Callable[[E, ErrorContext], None]):
        """Register a handler for a specific exception type.
        
        Args:
            exception_type: The type of exception to handle
            handler: The handler function
        """
        if exception_type not in self.error_handlers:
            self.error_handlers[exception_type] = []
        self.error_handlers[exception_type].append(handler)
    
    def register_global_handler(self, handler: Callable[[Exception, ErrorContext], None]):
        """Register a global handler for all exceptions.
        
        Args:
            handler: The handler function
        """
        self.global_handlers.append(handler)
    
    def handle_error(self, error: Exception, context: ErrorContext):
        """Handle an error.
        
        Args:
            error: The error to handle
            context: The error context
        """
        # Log the error
        log_level = getattr(logging, context.severity.name)
        logger.log(log_level, f"Error in {context.component}.{context.operation}: {str(error)}")
        
        # Call specific handlers for this error type
        handled = False
        for exc_type, handlers in self.error_handlers.items():
            if isinstance(error, exc_type):
                for handler in handlers:
                    handler(error, context)
                handled = True
        
        # Call global handlers
        for handler in self.global_handlers:
            handler(error, context)
        
        # If no specific handlers were found, log the full traceback
        if not handled and context.severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL):
            logger.error(f"Unhandled error: {traceback.format_exc()}")
    
    def wrap_function(
        self,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        reraise: bool = True
    ):
        """Decorator to wrap a function with error handling.
        
        Args:
            component: The component name
            operation: The operation name
            severity: The error severity
            category: The error category
            reraise: Whether to re-raise the error after handling
            
        Returns:
            A decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        severity=severity,
                        category=category,
                        metadata={"args": args, "kwargs": kwargs}
                    )
                    self.handle_error(e, context)
                    if reraise:
                        raise EnhancedError(str(e), context, e) from e
            return wrapper
        return decorator
    
    def wrap_async_function(
        self,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        reraise: bool = True
    ):
        """Decorator to wrap an async function with error handling.
        
        Args:
            component: The component name
            operation: The operation name
            severity: The error severity
            category: The error category
            reraise: Whether to re-raise the error after handling
            
        Returns:
            A decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        component=component,
                        operation=operation,
                        severity=severity,
                        category=category,
                        metadata={"args": args, "kwargs": kwargs}
                    )
                    self.handle_error(e, context)
                    if reraise:
                        raise EnhancedError(str(e), context, e) from e
            return wrapper
        return decorator
    
    @asynccontextmanager
    async def error_boundary(
        self,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        reraise: bool = True
    ):
        """Context manager for error handling.
        
        Args:
            component: The component name
            operation: The operation name
            severity: The error severity
            category: The error category
            reraise: Whether to re-raise the error after handling
            
        Yields:
            None
        """
        try:
            yield
        except Exception as e:
            context = ErrorContext(
                component=component,
                operation=operation,
                severity=severity,
                category=category
            )
            self.handle_error(e, context)
            if reraise:
                raise EnhancedError(str(e), context, e) from e


# Create a global error handler instance
error_handler = ErrorHandler()


def handle_error(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    reraise: bool = True
):
    """Decorator for error handling.
    
    Args:
        component: The component name
        operation: The operation name
        severity: The error severity
        category: The error category
        reraise: Whether to re-raise the error after handling
        
    Returns:
        A decorator function
    """
    return error_handler.wrap_function(component, operation, severity, category, reraise)


def handle_async_error(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    reraise: bool = True
):
    """Decorator for async error handling.
    
    Args:
        component: The component name
        operation: The operation name
        severity: The error severity
        category: The error category
        reraise: Whether to re-raise the error after handling
        
    Returns:
        A decorator function
    """
    return error_handler.wrap_async_function(component, operation, severity, category, reraise)


async def setup_error_handlers():
    """Set up default error handlers."""
    # Register a handler for critical errors
    def critical_error_handler(error: Exception, context: ErrorContext):
        if context.severity == ErrorSeverity.CRITICAL:
            # In a real application, this might send alerts or notifications
            logger.critical(f"CRITICAL ERROR: {str(error)}")
    
    error_handler.register_global_handler(critical_error_handler)
