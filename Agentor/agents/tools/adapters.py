"""
Adapters for existing tools to use the new standardized interfaces.

This module provides adapter classes that wrap existing tool implementations
to make them compatible with the new standardized interfaces.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import inspect
import asyncio

# Import the legacy tool classes
from agentor.agents.tools.base import BaseTool as OldBaseTool, ToolResult as OldToolResult

# Import the standardized interfaces
from agentor.core.interfaces.tool import ITool, ToolResult as CoreToolResult, IToolRegistry
from agentor.core.plugin import Plugin
from agentor.core.registry import get_component_registry

logger = logging.getLogger(__name__)


def convert_result(result: OldToolResult) -> CoreToolResult:
    """Convert an old ToolResult to a new ToolResult.

    Args:
        result: The old result to convert

    Returns:
        The converted new result
    """
    return CoreToolResult(
        success=result.success,
        data=result.data,
        error=result.error
    )


class ToolAdapter(ITool):
    """Adapter for the BaseTool class."""

    def __init__(self, tool: OldBaseTool):
        """Initialize the tool adapter.

        Args:
            tool: The tool implementation to adapt
        """
        self.tool = tool

    @property
    def name(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool
        """
        return self.tool.name

    @property
    def description(self) -> str:
        """Get the description of the tool.

        Returns:
            The description of the tool
        """
        return self.tool.description

    async def run(self, **kwargs) -> CoreToolResult:
        """Run the tool with the given parameters.

        Args:
            **kwargs: The parameters for the tool

        Returns:
            The result of running the tool
        """
        old_result = await self.tool.run(**kwargs)
        return convert_result(old_result)

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.

        Returns:
            A dictionary describing the parameters for the tool
        """
        # Try to get the schema from the tool
        if hasattr(self.tool, 'get_schema'):
            return self.tool.get_schema()

        # If the tool doesn't have a schema, try to infer it from the run method
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Get the signature of the run method
        sig = inspect.signature(self.tool.run)

        # Add parameters to the schema
        for name, param in sig.parameters.items():
            if name == 'self':
                continue

            # Add the parameter to the schema
            schema["properties"][name] = {
                "type": "string",  # Default to string
                "description": f"Parameter: {name}"
            }

            # If the parameter has no default value, it's required
            if param.default == inspect.Parameter.empty:
                schema["required"].append(name)

        return schema


class ToolRegistryAdapter(IToolRegistry):
    """Adapter for a dictionary of tools."""

    def __init__(self, tools: Optional[Dict[str, OldBaseTool]] = None):
        """Initialize the tool registry adapter.

        Args:
            tools: The tools to adapt, or None to start with an empty registry
        """
        self._tools: Dict[str, ITool] = {}

        # Adapt the tools if provided
        if tools:
            for name, tool in tools.items():
                self.register_tool(ToolAdapter(tool))

    def register_tool(self, tool: ITool) -> None:
        """Register a tool.

        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool

    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool.

        Args:
            tool_name: The name of the tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[ITool]:
        """Get a tool by name.

        Args:
            tool_name: The name of the tool

        Returns:
            The tool, or None if not found
        """
        return self._tools.get(tool_name)

    def get_tools(self) -> Dict[str, ITool]:
        """Get all registered tools.

        Returns:
            A dictionary of tool names to tools
        """
        return self._tools.copy()


# Common tools

class WeatherToolAdapter(ToolAdapter):
    """Adapter for the WeatherTool class."""

    def __init__(self, tool: Optional[OldBaseTool] = None):
        """Initialize the weather tool adapter.

        Args:
            tool: The weather tool implementation to adapt, or None to create a new one
        """
        if tool is None:
            from agentor.agents.enhanced_tools import WeatherTool
            tool = WeatherTool()

        super().__init__(tool)


class NewsToolAdapter(ToolAdapter):
    """Adapter for the NewsTool class."""

    def __init__(self, tool: Optional[OldBaseTool] = None):
        """Initialize the news tool adapter.

        Args:
            tool: The news tool implementation to adapt, or None to create a new one
        """
        if tool is None:
            from agentor.agents.enhanced_tools import NewsTool
            tool = NewsTool()

        super().__init__(tool)


class CalculatorToolAdapter(ToolAdapter):
    """Adapter for the CalculatorTool class."""

    def __init__(self, tool: Optional[OldBaseTool] = None):
        """Initialize the calculator tool adapter.

        Args:
            tool: The calculator tool implementation to adapt, or None to create a new one
        """
        if tool is None:
            from agentor.agents.enhanced_tools import CalculatorTool
            tool = CalculatorTool()

        super().__init__(tool)


# Tool plugins

class WeatherToolPlugin(Plugin):
    """Plugin for the WeatherTool class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "weather_tool"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Tool for getting weather information"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Create the tool adapter
        tool_adapter = WeatherToolAdapter()

        # Register the tool provider
        component_registry = get_component_registry()
        component_registry.register_tool_provider("weather", tool_adapter)

        logger.info("Registered weather tool provider")


class NewsToolPlugin(Plugin):
    """Plugin for the NewsTool class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "news_tool"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Tool for getting news headlines"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Create the tool adapter
        tool_adapter = NewsToolAdapter()

        # Register the tool provider
        component_registry = get_component_registry()
        component_registry.register_tool_provider("news", tool_adapter)

        logger.info("Registered news tool provider")


class CalculatorToolPlugin(Plugin):
    """Plugin for the CalculatorTool class."""

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "calculator_tool"

    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Tool for performing calculations"

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        Args:
            registry: The plugin registry
        """
        # Create the tool adapter
        tool_adapter = CalculatorToolAdapter()

        # Register the tool provider
        component_registry = get_component_registry()
        component_registry.register_tool_provider("calculator", tool_adapter)

        logger.info("Registered calculator tool provider")
