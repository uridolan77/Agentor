"""
Tool interfaces for the Agentor framework.

This module defines the interfaces for tool components in the Agentor framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable
from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    success: bool
    """Whether the tool execution was successful."""
    
    data: Optional[Dict[str, Any]] = None
    """The data returned by the tool."""
    
    error: Optional[str] = None
    """The error message if the tool execution failed."""


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for tool providers."""
    
    @property
    def name(self) -> str:
        """Get the name of the tool.
        
        Returns:
            The name of the tool
        """
        ...
    
    @property
    def description(self) -> str:
        """Get the description of the tool.
        
        Returns:
            The description of the tool
        """
        ...
    
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with the given parameters.
        
        Args:
            **kwargs: The parameters for the tool
            
        Returns:
            The result of running the tool
        """
        ...


class ITool(ABC):
    """Interface for tool components."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the tool.
        
        Returns:
            The name of the tool
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get the description of the tool.
        
        Returns:
            The description of the tool
        """
        pass
    
    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with the given parameters.
        
        Args:
            **kwargs: The parameters for the tool
            
        Returns:
            The result of running the tool
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        pass


class IToolRegistry(ABC):
    """Interface for tool registries."""
    
    @abstractmethod
    def register_tool(self, tool: ITool) -> None:
        """Register a tool.
        
        Args:
            tool: The tool to register
        """
        pass
    
    @abstractmethod
    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool.
        
        Args:
            tool_name: The name of the tool to unregister
        """
        pass
    
    @abstractmethod
    def get_tool(self, tool_name: str) -> Optional[ITool]:
        """Get a tool by name.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            The tool, or None if not found
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> Dict[str, ITool]:
        """Get all registered tools.
        
        Returns:
            A dictionary of tool names to tools
        """
        pass
