"""
Base tool classes for the Agentor framework.

This module provides the base classes for tools in the Agentor framework.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import logging

# Import the core ToolResult for compatibility
from agentor.core.interfaces.tool import ToolResult as CoreToolResult

logger = logging.getLogger(__name__)


class ToolResult:
    """Result of a tool execution.

    This is a wrapper around the core ToolResult class for backward compatibility.
    New code should use the core ToolResult directly.
    """

    def __init__(
        self,
        success: bool = True,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a tool result.

        Args:
            success: Whether the tool execution was successful
            data: The data returned by the tool
            error: Error message if the tool execution failed
            metadata: Additional metadata about the tool execution
        """
        self.success = success
        self.data = data or {}
        self.error = error
        self.metadata = metadata or {}

        # Create a core ToolResult for compatibility
        self._core_result = CoreToolResult(
            success=success,
            data=data,
            error=error
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary.

        Returns:
            A dictionary representation of the result
        """
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """Create a ToolResult from a dictionary.

        Args:
            data: The dictionary to convert

        Returns:
            A ToolResult instance
        """
        return cls(
            success=data.get("success", True),
            data=data.get("data", {}),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def success_result(cls, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> 'ToolResult':
        """Create a successful result.

        Args:
            data: The data returned by the tool
            metadata: Additional metadata about the tool execution

        Returns:
            A successful ToolResult
        """
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> 'ToolResult':
        """Create an error result.

        Args:
            error: The error message
            metadata: Additional metadata about the tool execution

        Returns:
            An error ToolResult
        """
        return cls(success=False, error=error, metadata=metadata)

    def to_core_result(self) -> CoreToolResult:
        """Convert to a core ToolResult.

        Returns:
            A core ToolResult instance
        """
        return self._core_result


class BaseTool:
    """Base class for tools in the Agentor framework.

    This is a simplified version of EnhancedTool for backward compatibility.
    New code should use EnhancedTool directly.
    """

    def __init__(self, name: str, description: str):
        """Initialize a tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
        self.version = "1.0.0"  # Default version for backward compatibility

    async def run(self, **kwargs) -> Union[Dict[str, Any], ToolResult]:
        """Run the tool with the given arguments.

        Args:
            **kwargs: Arguments for the tool

        Returns:
            The result of running the tool
        """
        raise NotImplementedError("Subclasses must implement run()")

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool's parameters.

        Returns:
            A dictionary describing the tool's parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {}
        }

    def __str__(self) -> str:
        """Get a string representation of the tool.

        Returns:
            A string representation
        """
        return f"{self.name}: {self.description}"
