"""
Structured logging for the Agentor framework.

This module provides utilities for structured logging, including:
- JSON logging formatter
- Context-aware logging
- Log enrichment with metadata
"""

import logging
import json
import sys
import time
import uuid
import traceback
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import threading

# Thread-local storage for context
_context = threading.local()


def get_context() -> Dict[str, Any]:
    """Get the current logging context.
    
    Returns:
        The current logging context
    """
    if not hasattr(_context, "data"):
        _context.data = {}
    return _context.data


def set_context(key: str, value: Any) -> None:
    """Set a value in the logging context.
    
    Args:
        key: The key to set
        value: The value to set
    """
    context = get_context()
    context[key] = value


def clear_context() -> None:
    """Clear the logging context."""
    if hasattr(_context, "data"):
        _context.data = {}


@contextmanager
def logging_context(**kwargs):
    """Context manager for adding context to logs.
    
    Args:
        **kwargs: Key-value pairs to add to the context
    """
    # Save the original context
    original_context = get_context().copy()
    
    try:
        # Update the context with the new values
        for key, value in kwargs.items():
            set_context(key, value)
        
        # Yield control back to the caller
        yield
    finally:
        # Restore the original context
        _context.data = original_context


class StructuredLogRecord(logging.LogRecord):
    """Log record that includes structured data."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the log record."""
        super().__init__(*args, **kwargs)
        
        # Add the context to the record
        self.context = get_context().copy()


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""
    
    def __init__(self, include_context: bool = True, include_stack_info: bool = True):
        """Initialize the formatter.
        
        Args:
            include_context: Whether to include the context in the output
            include_stack_info: Whether to include stack info in the output
        """
        super().__init__()
        self.include_context = include_context
        self.include_stack_info = include_stack_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            The formatted log record
        """
        # Create the base log data
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.thread,
        }
        
        # Add the context if available and requested
        if self.include_context and hasattr(record, "context"):
            log_data["context"] = record.context
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        
        # Add stack info if available and requested
        if self.include_stack_info and record.stack_info:
            log_data["stack_info"] = record.stack_info
        
        # Convert to JSON
        return json.dumps(log_data)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    include_context: bool = True,
    include_stack_info: bool = True,
    log_file: Optional[str] = None
) -> None:
    """Configure logging for the Agentor framework.
    
    Args:
        level: The logging level
        json_format: Whether to use JSON formatting
        include_context: Whether to include context in JSON output
        include_stack_info: Whether to include stack info in JSON output
        log_file: Optional file to log to
    """
    # Set the log record factory
    logging.setLogRecordFactory(StructuredLogRecord)
    
    # Create the handler
    handlers = []
    
    # Add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)
    
    # Add a file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    # Set the formatter
    if json_format:
        formatter = JsonFormatter(
            include_context=include_context,
            include_stack_info=include_stack_info
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Add the handlers
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: The name of the logger
        
    Returns:
        The logger
    """
    return logging.getLogger(name)


class LoggingPlugin:
    """Plugin for configuring logging."""
    
    def __init__(
        self,
        level: int = logging.INFO,
        json_format: bool = False,
        include_context: bool = True,
        include_stack_info: bool = True,
        log_file: Optional[str] = None
    ):
        """Initialize the plugin.
        
        Args:
            level: The logging level
            json_format: Whether to use JSON formatting
            include_context: Whether to include context in JSON output
            include_stack_info: Whether to include stack info in JSON output
            log_file: Optional file to log to
        """
        self.level = level
        self.json_format = json_format
        self.include_context = include_context
        self.include_stack_info = include_stack_info
        self.log_file = log_file
    
    @property
    def name(self) -> str:
        """Get the name of the plugin.
        
        Returns:
            The name of the plugin
        """
        return "logging"
    
    @property
    def version(self) -> str:
        """Get the version of the plugin.
        
        Returns:
            The version of the plugin
        """
        return "0.1.0"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin.
        
        Returns:
            The description of the plugin
        """
        return "Plugin for configuring logging"
    
    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies of the plugin.
        
        Returns:
            The dependencies of the plugin
        """
        return []
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        configure_logging(
            level=self.level,
            json_format=self.json_format,
            include_context=self.include_context,
            include_stack_info=self.include_stack_info,
            log_file=self.log_file
        )
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass
