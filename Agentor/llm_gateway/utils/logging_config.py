import logging
import logging.config
import json
import os
import sys
import time
import traceback
from typing import Dict, Any, Optional, List, Union
import socket
import uuid
from datetime import datetime
from enum import Enum
import threading
from functools import wraps

# Define log levels
class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the log record."""
    
    def __init__(
        self,
        fmt_keys: Optional[Dict[str, str]] = None,
        time_format: str = "%Y-%m-%dT%H:%M:%S.%fZ",
        msec_format: str = "%s.%03dZ",
        include_stack_info: bool = False
    ):
        """Initialize the formatter.
        
        Args:
            fmt_keys: Map of log record attributes to JSON keys
            time_format: Format for timestamps
            msec_format: Format for milliseconds
            include_stack_info: Whether to include stack info in the log
        """
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}
        self.time_format = time_format
        self.msec_format = msec_format
        self.include_stack_info = include_stack_info
        self.hostname = socket.gethostname()
        self.default_keys = {
            "timestamp": "timestamp",
            "level": "level",
            "name": "logger",
            "message": "message"
        }
        
        # Merge default keys with user-provided keys
        for key, value in self.default_keys.items():
            if key not in self.fmt_keys:
                self.fmt_keys[key] = value
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.
        
        Args:
            record: The log record
            
        Returns:
            The formatted log record
        """
        log_data = {}
        
        # Add basic log record attributes
        for key, json_key in self.fmt_keys.items():
            if hasattr(record, key):
                log_data[json_key] = getattr(record, key)
        
        # Add timestamp
        log_data["timestamp"] = datetime.utcfromtimestamp(record.created).strftime(self.time_format)
        
        # Add log level
        log_data["level"] = record.levelname
        
        # Add logger name
        log_data["logger"] = record.name
        
        # Add message
        log_data["message"] = record.getMessage()
        
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add stack info if available and requested
        if self.include_stack_info and record.stack_info:
            log_data["stack_info"] = record.stack_info
        
        # Add thread info
        log_data["thread"] = {
            "id": record.thread,
            "name": record.threadName
        }
        
        # Add process info
        log_data["process"] = {
            "id": record.process,
            "name": record.processName
        }
        
        # Add hostname
        log_data["hostname"] = self.hostname
        
        # Add extra attributes from the record
        if hasattr(record, "extra") and record.extra:
            for key, value in record.extra.items():
                log_data[key] = value
        
        # Convert to JSON
        return json.dumps(log_data)


class StructuredLogger(logging.Logger):
    """Logger that supports structured logging."""
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize the logger.
        
        Args:
            name: The logger name
            level: The logger level
        """
        super().__init__(name, level)
        self.request_id = None
        self.context = {}
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """Bind context values to the logger.
        
        Args:
            **kwargs: Context values to bind
            
        Returns:
            The logger with bound context
        """
        logger = self.getChild("")
        logger.context = {**self.context, **kwargs}
        return logger
    
    def unbind(self, *keys) -> "StructuredLogger":
        """Unbind context values from the logger.
        
        Args:
            *keys: Keys to unbind
            
        Returns:
            The logger with unbound context
        """
        logger = self.getChild("")
        logger.context = {k: v for k, v in self.context.items() if k not in keys}
        return logger
    
    def with_request_id(self, request_id: Optional[str] = None) -> "StructuredLogger":
        """Set a request ID for the logger.
        
        Args:
            request_id: The request ID, or None to generate a new one
            
        Returns:
            The logger with the request ID
        """
        logger = self.getChild("")
        logger.request_id = request_id or str(uuid.uuid4())
        return logger
    
    def _log(
        self,
        level: int,
        msg: str,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        **kwargs
    ):
        """Log a message with the specified level and optional extra data.
        
        Args:
            level: The log level
            msg: The log message
            args: Arguments for the message
            exc_info: Exception info
            extra: Extra data
            stack_info: Whether to include stack info
            stacklevel: Stack level
            **kwargs: Additional context values
        """
        # Merge context with extra
        if extra is None:
            extra = {}
        
        if self.context:
            extra.update(self.context)
        
        if kwargs:
            extra.update(kwargs)
        
        if self.request_id:
            extra["request_id"] = self.request_id
        
        # Add timestamp
        extra["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Call the parent logger
        super()._log(level, msg, args, exc_info, {"extra": extra}, stack_info, stacklevel)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger.
    
    Args:
        name: The logger name
        
    Returns:
        A structured logger
    """
    return logging.getLogger(name)


def configure_logging(
    level: Union[int, str] = logging.INFO,
    json_format: bool = False,
    log_file: Optional[str] = None,
    include_stack_info: bool = False
):
    """Configure logging.
    
    Args:
        level: The log level
        json_format: Whether to use JSON format
        log_file: Path to the log file, or None to log to stderr
        include_stack_info: Whether to include stack info in the log
    """
    # Set the logger class
    logging.setLoggerClass(StructuredLogger)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    # Set formatters
    if json_format:
        formatter = JsonFormatter(include_stack_info=include_stack_info)
        for handler in handlers:
            handler.setFormatter(formatter)
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        for handler in handlers:
            handler.setFormatter(formatter)
    
    # Configure the root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    # Set the logger class for future loggers
    logging.setLoggerClass(StructuredLogger)


def log_execution_time(logger=None, level=logging.INFO):
    """Decorator to log the execution time of a function.
    
    Args:
        logger: The logger to use, or None to use the module logger
        level: The log level
        
    Returns:
        A decorator function
    """
    def decorator(func):
        # Get the logger
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.log(level, f"{func.__name__} executed in {elapsed_time:.3f}s")
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.log(level, f"{func.__name__} executed in {elapsed_time:.3f}s")
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# Import asyncio at the end to avoid circular imports
import asyncio
