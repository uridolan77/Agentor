"""
Structured logging system for the Agentor framework.

This module provides a comprehensive structured logging system with JSON formatting
and context support. It includes thread-local logging context to enrich logs with
contextual information.
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

# Thread-local storage for logging context
_log_context = threading.local()


class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord that includes structured data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structured_data = getattr(_log_context, "data", {}).copy()


class StructuredLogger(logging.Logger):
    """Logger that creates StructuredLogRecord instances."""

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Create a StructuredLogRecord instance."""
        record = StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key in ["message", "asctime", "levelname", "levelno"]:
                    raise KeyError(f"Attempt to overwrite {key} in LogRecord")
                record.__dict__[key] = extra[key]
        return record


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""

    def __init__(self, include_timestamp: bool = True, include_level: bool = True, include_name: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name

    def format(self, record: logging.LogRecord) -> str:
        """Format the record as JSON."""
        log_data = {}

        # Include standard fields
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if self.include_level:
            log_data["level"] = record.levelname
        if self.include_name:
            log_data["logger"] = record.name

        # Include message
        log_data["message"] = record.getMessage()

        # Include exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include structured data
        if hasattr(record, "structured_data"):
            log_data.update(record.structured_data)

        # Include extra attributes
        for key, value in record.__dict__.items():
            if key not in ["msg", "args", "exc_info", "exc_text", "structured_data",
                          "message", "levelname", "levelno", "pathname", "filename",
                          "module", "name", "lineno", "funcName", "created",
                          "asctime", "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "stack_info"]:
                log_data[key] = value

        return json.dumps(log_data)


class LoggingContext:
    """Context manager for adding context to logs."""

    def __init__(self, **kwargs):
        self.old_data = {}
        self.new_data = kwargs

    def __enter__(self):
        if not hasattr(_log_context, "data"):
            _log_context.data = {}

        self.old_data = _log_context.data.copy()
        _log_context.data.update(self.new_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _log_context.data = self.old_data


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger with the given name."""
    logging.setLoggerClass(StructuredLogger)
    return logging.getLogger(name)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    json_format: bool = True,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_name: bool = True,
    handlers: Optional[list] = None,
) -> None:
    """Configure the logging system.

    Args:
        level: The logging level to use
        json_format: Whether to use JSON formatting
        include_timestamp: Whether to include timestamps in logs
        include_level: Whether to include log levels in logs
        include_name: Whether to include logger names in logs
        handlers: Additional handlers to add to the root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Set formatter
    if json_format:
        formatter = JSONFormatter(
            include_timestamp=include_timestamp,
            include_level=include_level,
            include_name=include_name,
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add additional handlers
    if handlers:
        for handler in handlers:
            root_logger.addHandler(handler)

    # Set logger class
    logging.setLoggerClass(StructuredLogger)


def add_request_id_to_context():
    """Add a unique request ID to the logging context."""
    request_id = str(uuid.uuid4())
    if not hasattr(_log_context, "data"):
        _log_context.data = {}
    _log_context.data["request_id"] = request_id
    return request_id


def get_context_data() -> Dict[str, Any]:
    """Get the current logging context data."""
    if not hasattr(_log_context, "data"):
        _log_context.data = {}
    return _log_context.data.copy()


def clear_context():
    """Clear the current logging context."""
    if hasattr(_log_context, "data"):
        _log_context.data = {}