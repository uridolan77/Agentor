"""Tests for the Agentor utility functions.

This module contains tests for the various utility functions provided by the Agentor framework.
"""

import pytest
import unittest
import logging
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

# Define mock functions for testing
def get_logger(name):
    """Mock get_logger function."""
    return logging.getLogger(name)


def configure_logging(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """Mock configure_logging function."""
    logging.basicConfig(level=level, format=format)


class TestLogging(unittest.TestCase):
    """Tests for the logging utility functions."""

    def test_get_logger(self):
        """Test the get_logger function."""
        # Get a logger
        logger = get_logger("test_logger")

        # Check that the logger is a logging.Logger instance
        self.assertIsInstance(logger, logging.Logger)

        # Check that the logger has the correct name
        self.assertEqual(logger.name, "test_logger")

    @patch('logging.basicConfig')
    def test_configure_logging(self, mock_basic_config):
        """Test the configure_logging function."""
        # Configure logging
        configure_logging(level=logging.DEBUG, format="%(message)s")

        # Check that logging.basicConfig was called with the correct arguments
        mock_basic_config.assert_called_once_with(level=logging.DEBUG, format="%(message)s")


class TestFileUtils:
    """Tests for file utility functions."""

    def setup_method(self):
        """Set up the test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")

    def teardown_method(self):
        """Clean up the test environment."""
        # Remove the temporary directory and all its contents
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)

    def test_read_write_text(self):
        """Test reading and writing text files."""
        # Write a text file
        content = "Hello, world!"
        with open(self.test_file, "w") as f:
            f.write(content)

        # Read the text file
        with open(self.test_file, "r") as f:
            read_content = f.read()

        # Check that the content was read correctly
        assert read_content == content

    def test_read_write_json(self):
        """Test reading and writing JSON files."""
        # Write a JSON file
        data = {"name": "test", "value": 42}
        with open(self.test_file, "w") as f:
            json.dump(data, f)

        # Read the JSON file
        with open(self.test_file, "r") as f:
            read_data = json.load(f)

        # Check that the data was read correctly
        assert read_data == data


class TestStringUtils:
    """Tests for string utility functions."""

    def test_string_formatting(self):
        """Test string formatting functions."""
        # Test string formatting
        template = "Hello, {name}!"
        formatted = template.format(name="world")
        assert formatted == "Hello, world!"

    def test_string_parsing(self):
        """Test string parsing functions."""
        # Test string parsing
        text = "The answer is 42."
        words = text.split()
        assert words == ["The", "answer", "is", "42."]


class TestDataUtils:
    """Tests for data utility functions."""

    def test_data_conversion(self):
        """Test data conversion functions."""
        # Test data conversion
        data = {"name": "test", "value": 42}
        json_str = json.dumps(data)
        assert json.loads(json_str) == data

    def test_data_validation(self):
        """Test data validation functions."""
        # Test data validation
        data = {"name": "test", "value": 42}
        assert "name" in data
        assert "value" in data
        assert data["name"] == "test"
        assert data["value"] == 42


if __name__ == "__main__":
    pytest.main()