"""
Tests for the structured logging module.
"""

import pytest
import logging
import json
import io
import sys
from contextlib import redirect_stdout

from agentor.core.logging import (
    get_context, set_context, clear_context, logging_context,
    JsonFormatter, StructuredLogRecord, configure_logging, get_logger
)


def test_logging_context():
    """Test the logging context functions."""
    # Clear the context
    clear_context()
    
    # Check that the context is empty
    assert get_context() == {}
    
    # Set a value in the context
    set_context("test_key", "test_value")
    
    # Check that the value was set
    assert get_context() == {"test_key": "test_value"}
    
    # Use the context manager
    with logging_context(another_key="another_value"):
        # Check that both values are in the context
        assert get_context() == {
            "test_key": "test_value",
            "another_key": "another_value"
        }
    
    # Check that the context was restored
    assert get_context() == {"test_key": "test_value"}
    
    # Clear the context
    clear_context()
    
    # Check that the context is empty
    assert get_context() == {}


def test_structured_log_record():
    """Test the StructuredLogRecord class."""
    # Set up the context
    clear_context()
    set_context("test_key", "test_value")
    
    # Create a log record
    record = StructuredLogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    # Check that the context was added to the record
    assert hasattr(record, "context")
    assert record.context == {"test_key": "test_value"}
    
    # Clear the context
    clear_context()


def test_json_formatter():
    """Test the JsonFormatter class."""
    # Create a formatter
    formatter = JsonFormatter()
    
    # Create a log record
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    # Format the record
    formatted = formatter.format(record)
    
    # Check that the result is valid JSON
    data = json.loads(formatted)
    
    # Check the fields
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["message"] == "Test message"
    
    # Create a record with context
    record = StructuredLogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    # Set the context
    record.context = {"test_key": "test_value"}
    
    # Format the record
    formatted = formatter.format(record)
    
    # Check that the result is valid JSON
    data = json.loads(formatted)
    
    # Check the fields
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["message"] == "Test message"
    assert data["context"] == {"test_key": "test_value"}


def test_configure_logging():
    """Test the configure_logging function."""
    # Capture stdout
    stdout = io.StringIO()
    
    with redirect_stdout(stdout):
        # Configure logging
        configure_logging(level=logging.INFO)
        
        # Get a logger
        logger = get_logger("test_logger")
        
        # Log a message
        logger.info("Test message")
    
    # Check that the message was logged
    output = stdout.getvalue()
    assert "Test message" in output
    
    # Configure JSON logging
    stdout = io.StringIO()
    
    with redirect_stdout(stdout):
        # Configure logging
        configure_logging(level=logging.INFO, json_format=True)
        
        # Get a logger
        logger = get_logger("test_logger")
        
        # Log a message
        logger.info("Test message")
    
    # Check that the message was logged as JSON
    output = stdout.getvalue()
    data = json.loads(output)
    
    # Check the fields
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["message"] == "Test message"


def test_logging_with_context():
    """Test logging with context."""
    # Configure logging
    configure_logging(level=logging.INFO, json_format=True)
    
    # Get a logger
    logger = get_logger("test_logger")
    
    # Capture stdout
    stdout = io.StringIO()
    
    with redirect_stdout(stdout):
        # Log with context
        with logging_context(user_id="123", request_id="456"):
            logger.info("Test message with context")
    
    # Check that the message was logged with context
    output = stdout.getvalue()
    data = json.loads(output)
    
    # Check the fields
    assert data["level"] == "INFO"
    assert data["name"] == "test_logger"
    assert data["message"] == "Test message with context"
    assert data["context"] == {"user_id": "123", "request_id": "456"}
    
    # Clear the context
    clear_context()
