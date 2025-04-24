"""
Enhanced error handling for tools in the Agentor framework.

This module provides classes and utilities for better error handling and recovery
in tool execution, including retry mechanisms, fallbacks, and error reporting.
"""

from typing import Dict, Any, List, Optional, Callable, Type, Union, Tuple
import logging
import asyncio
import time
import traceback
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps

from agentor.core.interfaces.tool import ITool, ToolResult, IToolRegistry
from agentor.agents.enhanced_tools import EnhancedTool

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories for errors."""
    
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    DEPENDENCY = "dependency"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class ToolErrorContext:
    """Context information for tool errors."""
    
    def __init__(
        self,
        tool_name: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the error context.
        
        Args:
            tool_name: The name of the tool
            operation: The operation being performed
            severity: The severity of the error
            category: The category of the error
            metadata: Additional metadata about the error
        """
        self.tool_name = tool_name
        self.operation = operation
        self.severity = severity
        self.category = category
        self.metadata = metadata or {}
        self.timestamp = time.time()


class ToolError(Exception):
    """Enhanced error class for tool errors."""
    
    def __init__(
        self,
        message: str,
        context: ToolErrorContext,
        original_error: Optional[Exception] = None
    ):
        """Initialize the error.
        
        Args:
            message: The error message
            context: The error context
            original_error: The original exception
        """
        super().__init__(message)
        self.context = context
        self.original_error = original_error
        self.traceback = traceback.format_exc() if original_error else None


class ErrorHandler(ABC):
    """Base class for error handlers."""
    
    @abstractmethod
    async def handle_error(self, error: Exception, context: ToolErrorContext) -> Optional[ToolResult]:
        """Handle an error.
        
        Args:
            error: The error to handle
            context: The error context
            
        Returns:
            An optional tool result if the error was handled
        """
        pass


class RetryHandler(ErrorHandler):
    """Error handler that retries the operation."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_categories: Optional[List[ErrorCategory]] = None,
        retry_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """Initialize the retry handler.
        
        Args:
            max_retries: Maximum number of retries
            base_delay: Base delay between retries (in seconds)
            max_delay: Maximum delay between retries (in seconds)
            retry_categories: Categories of errors to retry
            retry_exceptions: Types of exceptions to retry
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_categories = retry_categories or [
            ErrorCategory.CONNECTION,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT
        ]
        self.retry_exceptions = retry_exceptions or [
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError
        ]
        
        # Retry state
        self.retry_counts: Dict[str, int] = {}
    
    def _get_retry_key(self, context: ToolErrorContext) -> str:
        """Get a key for tracking retries.
        
        Args:
            context: The error context
            
        Returns:
            A key for tracking retries
        """
        return f"{context.tool_name}:{context.operation}"
    
    def _should_retry(self, error: Exception, context: ToolErrorContext) -> bool:
        """Check if the operation should be retried.
        
        Args:
            error: The error
            context: The error context
            
        Returns:
            True if the operation should be retried
        """
        # Check retry count
        retry_key = self._get_retry_key(context)
        retry_count = self.retry_counts.get(retry_key, 0)
        
        if retry_count >= self.max_retries:
            return False
        
        # Check error category
        if context.category in self.retry_categories:
            return True
        
        # Check exception type
        for exc_type in self.retry_exceptions:
            if isinstance(error, exc_type):
                return True
        
        return False
    
    def _get_delay(self, retry_count: int) -> float:
        """Get the delay for a retry.
        
        Args:
            retry_count: The current retry count
            
        Returns:
            The delay in seconds
        """
        # Exponential backoff with jitter
        delay = min(self.base_delay * (2 ** retry_count), self.max_delay)
        jitter = delay * 0.1  # 10% jitter
        return delay + (jitter * (2 * (0.5 - (time.time() % 1))))
    
    async def handle_error(self, error: Exception, context: ToolErrorContext) -> Optional[ToolResult]:
        """Handle an error by retrying the operation.
        
        Args:
            error: The error to handle
            context: The error context
            
        Returns:
            None, as the operation will be retried elsewhere
        """
        if not self._should_retry(error, context):
            return None
        
        # Increment retry count
        retry_key = self._get_retry_key(context)
        retry_count = self.retry_counts.get(retry_key, 0)
        self.retry_counts[retry_key] = retry_count + 1
        
        # Log the retry
        logger.info(
            f"Retrying {context.tool_name}.{context.operation} "
            f"(attempt {retry_count + 1}/{self.max_retries}) "
            f"after error: {str(error)}"
        )
        
        # Return a result indicating a retry is needed
        return ToolResult(
            success=False,
            error=f"Operation will be retried (attempt {retry_count + 1}/{self.max_retries})",
            data={"retry": True, "retry_count": retry_count + 1, "max_retries": self.max_retries}
        )


class FallbackHandler(ErrorHandler):
    """Error handler that provides a fallback result."""
    
    def __init__(
        self,
        fallback_function: Callable[[Exception, ToolErrorContext], Dict[str, Any]],
        handle_categories: Optional[List[ErrorCategory]] = None,
        handle_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """Initialize the fallback handler.
        
        Args:
            fallback_function: Function that returns fallback data
            handle_categories: Categories of errors to handle
            handle_exceptions: Types of exceptions to handle
        """
        self.fallback_function = fallback_function
        self.handle_categories = handle_categories or [
            ErrorCategory.VALIDATION,
            ErrorCategory.INTERNAL,
            ErrorCategory.UNKNOWN
        ]
        self.handle_exceptions = handle_exceptions or [Exception]
    
    def _should_handle(self, error: Exception, context: ToolErrorContext) -> bool:
        """Check if the error should be handled.
        
        Args:
            error: The error
            context: The error context
            
        Returns:
            True if the error should be handled
        """
        # Check error category
        if context.category in self.handle_categories:
            return True
        
        # Check exception type
        for exc_type in self.handle_exceptions:
            if isinstance(error, exc_type):
                return True
        
        return False
    
    async def handle_error(self, error: Exception, context: ToolErrorContext) -> Optional[ToolResult]:
        """Handle an error by providing a fallback result.
        
        Args:
            error: The error to handle
            context: The error context
            
        Returns:
            A fallback tool result
        """
        if not self._should_handle(error, context):
            return None
        
        try:
            # Get fallback data
            fallback_data = self.fallback_function(error, context)
            
            # Log the fallback
            logger.info(
                f"Using fallback for {context.tool_name}.{context.operation} "
                f"after error: {str(error)}"
            )
            
            # Return a fallback result
            return ToolResult(
                success=True,
                data=fallback_data,
                error=f"Using fallback data due to error: {str(error)}"
            )
        except Exception as e:
            logger.error(f"Error in fallback handler: {str(e)}")
            return None


class ErrorReportingHandler(ErrorHandler):
    """Error handler that reports errors to a monitoring system."""
    
    def __init__(
        self,
        report_function: Callable[[Exception, ToolErrorContext], None],
        report_categories: Optional[List[ErrorCategory]] = None,
        report_severities: Optional[List[ErrorSeverity]] = None
    ):
        """Initialize the error reporting handler.
        
        Args:
            report_function: Function that reports the error
            report_categories: Categories of errors to report
            report_severities: Severities of errors to report
        """
        self.report_function = report_function
        self.report_categories = report_categories or list(ErrorCategory)
        self.report_severities = report_severities or [
            ErrorSeverity.ERROR,
            ErrorSeverity.CRITICAL
        ]
    
    def _should_report(self, error: Exception, context: ToolErrorContext) -> bool:
        """Check if the error should be reported.
        
        Args:
            error: The error
            context: The error context
            
        Returns:
            True if the error should be reported
        """
        # Check error category
        if context.category not in self.report_categories:
            return False
        
        # Check error severity
        if context.severity not in self.report_severities:
            return False
        
        return True
    
    async def handle_error(self, error: Exception, context: ToolErrorContext) -> Optional[ToolResult]:
        """Handle an error by reporting it.
        
        Args:
            error: The error to handle
            context: The error context
            
        Returns:
            None, as this handler doesn't modify the result
        """
        if not self._should_report(error, context):
            return None
        
        try:
            # Report the error
            self.report_function(error, context)
            
            # Log the reporting
            logger.debug(
                f"Reported error in {context.tool_name}.{context.operation}: {str(error)}"
            )
        except Exception as e:
            logger.error(f"Error in error reporting handler: {str(e)}")
        
        return None


class ToolErrorManager:
    """Manager for tool error handling."""
    
    def __init__(self):
        """Initialize the error manager."""
        self.handlers: List[ErrorHandler] = []
        self.error_counts: Dict[str, int] = {}
        self.error_timestamps: Dict[str, float] = {}
    
    def add_handler(self, handler: ErrorHandler) -> None:
        """Add an error handler.
        
        Args:
            handler: The error handler to add
        """
        self.handlers.append(handler)
    
    def _get_error_key(self, context: ToolErrorContext) -> str:
        """Get a key for tracking errors.
        
        Args:
            context: The error context
            
        Returns:
            A key for tracking errors
        """
        return f"{context.tool_name}:{context.operation}:{context.category.value}"
    
    def _record_error(self, context: ToolErrorContext) -> None:
        """Record an error occurrence.
        
        Args:
            context: The error context
        """
        error_key = self._get_error_key(context)
        
        # Update count
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Update timestamp
        self.error_timestamps[error_key] = time.time()
    
    async def handle_error(
        self,
        error: Exception,
        context: ToolErrorContext
    ) -> Tuple[bool, Optional[ToolResult]]:
        """Handle an error using registered handlers.
        
        Args:
            error: The error to handle
            context: The error context
            
        Returns:
            A tuple of (handled, result)
        """
        # Record the error
        self._record_error(context)
        
        # Log the error
        log_level = getattr(logging, context.severity.value.upper())
        logger.log(
            log_level,
            f"Error in {context.tool_name}.{context.operation} "
            f"({context.category.value}): {str(error)}"
        )
        
        # Try each handler
        for handler in self.handlers:
            try:
                result = await handler.handle_error(error, context)
                if result is not None:
                    return True, result
            except Exception as e:
                logger.error(f"Error in error handler: {str(e)}")
        
        # No handler handled the error
        return False, None


class ErrorHandlingTool(EnhancedTool):
    """Wrapper for tools that adds enhanced error handling."""
    
    def __init__(
        self,
        tool: ITool,
        error_manager: ToolErrorManager,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None
    ):
        """Initialize the error handling tool.
        
        Args:
            tool: The tool to wrap
            error_manager: The error manager to use
            name: Optional override for the tool name
            description: Optional override for the tool description
            version: Optional override for the tool version
        """
        super().__init__(
            name=name or tool.name,
            description=description or tool.description,
            version=version or getattr(tool, "version", "1.0.0")
        )
        self.tool = tool
        self.error_manager = error_manager
    
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with error handling.
        
        Args:
            **kwargs: The parameters for the tool
            
        Returns:
            The result of running the tool
        """
        try:
            # Run the original tool
            return await self.tool.run(**kwargs)
        except Exception as e:
            # Create error context
            context = ToolErrorContext(
                tool_name=self.name,
                operation="run",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.UNKNOWN,
                metadata={"params": kwargs}
            )
            
            # Handle the error
            handled, result = await self.error_manager.handle_error(e, context)
            
            if handled and result is not None:
                return result
            
            # If not handled, return a generic error result
            return ToolResult(
                success=False,
                error=f"Error executing tool {self.name}: {str(e)}"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        return self.tool.get_schema()


class ErrorHandlingToolRegistry(IToolRegistry):
    """Tool registry that adds error handling to all tools."""
    
    def __init__(
        self,
        base_registry: IToolRegistry,
        error_manager: Optional[ToolErrorManager] = None,
        add_default_handlers: bool = True
    ):
        """Initialize the error handling tool registry.
        
        Args:
            base_registry: The base tool registry to wrap
            error_manager: Optional error manager to use
            add_default_handlers: Whether to add default error handlers
        """
        self.base_registry = base_registry
        self.error_manager = error_manager or ToolErrorManager()
        
        # Add default handlers if requested
        if add_default_handlers:
            self._add_default_handlers()
    
    def _add_default_handlers(self) -> None:
        """Add default error handlers."""
        # Add retry handler
        self.error_manager.add_handler(RetryHandler())
        
        # Add fallback handler for calculator tool
        def calculator_fallback(error: Exception, context: ToolErrorContext) -> Dict[str, Any]:
            if context.tool_name == "calculator":
                return {"result": 0, "fallback": True}
            return {"fallback": True}
        
        self.error_manager.add_handler(FallbackHandler(calculator_fallback))
        
        # Add error reporting handler
        def report_error(error: Exception, context: ToolErrorContext) -> None:
            # In a real implementation, this would send to a monitoring system
            logger.error(
                f"REPORTED ERROR: {context.tool_name}.{context.operation} "
                f"({context.category.value}): {str(error)}"
            )
        
        self.error_manager.add_handler(ErrorReportingHandler(report_error))
    
    def register_tool(self, tool: ITool) -> None:
        """Register a tool.
        
        Args:
            tool: The tool to register
        """
        # Register the original tool with the base registry
        self.base_registry.register_tool(tool)
    
    def get_tool(self, name: str, version: Optional[str] = None) -> Optional[ITool]:
        """Get a tool by name and optional version.
        
        Args:
            name: The name of the tool
            version: Optional version of the tool
            
        Returns:
            The tool, or None if not found
        """
        # Get the tool from the base registry
        tool = self.base_registry.get_tool(name, version)
        
        if tool is None:
            return None
        
        # Wrap the tool with error handling
        return ErrorHandlingTool(tool, self.error_manager)
    
    def get_tools(self) -> Dict[str, ITool]:
        """Get all tools.
        
        Returns:
            A dictionary of tool names to tools
        """
        # Get all tools from the base registry
        tools = self.base_registry.get_tools()
        
        # Wrap each tool with error handling
        return {
            name: ErrorHandlingTool(tool, self.error_manager)
            for name, tool in tools.items()
        }


def create_error_handling_registry(
    base_registry: IToolRegistry,
    add_default_handlers: bool = True
) -> ErrorHandlingToolRegistry:
    """Create an error handling tool registry.
    
    Args:
        base_registry: The base tool registry to wrap
        add_default_handlers: Whether to add default error handlers
        
    Returns:
        An error handling tool registry
    """
    return ErrorHandlingToolRegistry(base_registry, add_default_handlers=add_default_handlers)
