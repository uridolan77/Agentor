"""
Enhanced tools implementation for the Agentor framework.

This module provides enhanced tool implementations that use the standardized
interfaces and dependency injection system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Type, Set, TypeVar, Generic, get_type_hints, get_origin, get_args, Tuple, Union
import logging
import inspect
import json
import asyncio
import ast
import operator
import importlib
import pkgutil
import sys
from pathlib import Path
from pydantic import BaseModel, create_model, Field

from agentor.core.interfaces.tool import ITool, ToolResult, IToolRegistry
from agentor.core.plugin import Plugin
from agentor.core.di import inject, get_container
from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema, model_to_json_schema, create_input_schema_from_signature, enhance_schema_with_docstring
from agentor.agents.versioning import SemanticVersion, VersionConstraint, VersionRange

# Import API server lazily to avoid circular imports
_tool_api_server = None

def get_tool_api_server():
    """Get the tool API server.

    Returns:
        The tool API server
    """
    global _tool_api_server
    if _tool_api_server is None:
        # Import here to avoid circular imports
        from agentor.agents.api.server import create_tool_api_server
        _tool_api_server = create_tool_api_server(get_tool_registry())
    return _tool_api_server

logger = logging.getLogger(__name__)


class ToolDependency:
    """A dependency on another tool."""
    
    def __init__(self, tool_name: str, version_constraint: Union[str, VersionConstraint] = None):
        """Initialize the tool dependency.
        
        Args:
            tool_name: The name of the dependent tool
            version_constraint: Optional version constraint
        """
        self.tool_name = tool_name
        
        if version_constraint is None:
            # No constraint means any version is acceptable
            self.version_constraint = None
        elif isinstance(version_constraint, str):
            # Parse the constraint string
            constraints = []
            for spec in version_constraint.split(","):
                if spec.startswith("=="):
                    constraints.append(VersionConstraint(exact_version=spec[2:]))
                elif spec.startswith(">="):
                    constraints.append(VersionConstraint(min_version=spec[2:]))
                elif spec.startswith("<"):
                    constraints.append(VersionConstraint(max_version=spec[1:]))
                elif spec.startswith("~"):
                    constraints.append(VersionConstraint(compatible_with=spec[1:]))
                else:
                    # Default to compatibility with specified version
                    constraints.append(VersionConstraint(compatible_with=spec))
            
            self.version_constraint = constraints
        else:
            # Use the provided constraint directly
            self.version_constraint = [version_constraint] if isinstance(version_constraint, VersionConstraint) else version_constraint
    
    def is_satisfied_by(self, tool: 'EnhancedTool') -> bool:
        """Check if a tool satisfies this dependency.
        
        Args:
            tool: The tool to check
            
        Returns:
            True if the tool satisfies this dependency, False otherwise
        """
        if tool.name != self.tool_name:
            return False
        
        if self.version_constraint is None:
            return True
            
        # Check if the tool's version satisfies all constraints
        for constraint in self.version_constraint:
            if not constraint.is_satisfied_by(tool.version):
                return False
                
        return True
    
    def __str__(self) -> str:
        """Get the string representation of this dependency.
        
        Returns:
            The dependency as a string
        """
        if self.version_constraint is None:
            return self.tool_name
            
        constraints = []
        for constraint in self.version_constraint:
            constraints.append(str(constraint))
            
        return f"{self.tool_name} ({', '.join(constraints)})"


class EnhancedTool(ITool, Plugin):
    """Enhanced base class for all tools."""

    # Type variables for input and output schemas
    InputType = TypeVar('InputType', bound=ToolInputSchema)
    OutputType = TypeVar('OutputType', bound=ToolOutputSchema)

    def __init__(
        self, 
        name: str, 
        description: str, 
        version: str = "1.0.0",
        input_schema: Optional[Type[InputType]] = None, 
        output_schema: Optional[Type[OutputType]] = None,
        tool_dependencies: Optional[List[Union[str, ToolDependency]]] = None
    ):
        """Initialize the tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            version: The version of the tool (semantic versioning)
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
            tool_dependencies: Optional list of tool dependencies
        """
        self._name = name
        self._description = description
        self._version = version
        self._input_schema = input_schema
        self._output_schema = output_schema
        
        # Process tool dependencies
        self._tool_dependencies = []
        if tool_dependencies:
            for dep in tool_dependencies:
                if isinstance(dep, str):
                    # Format: "tool_name" or "tool_name:version_constraint"
                    parts = dep.split(":", 1)
                    if len(parts) == 1:
                        self._tool_dependencies.append(ToolDependency(parts[0]))
                    else:
                        self._tool_dependencies.append(ToolDependency(parts[0], parts[1]))
                else:
                    # Already a ToolDependency object
                    self._tool_dependencies.append(dep)

    @property
    def name(self) -> str:
        """Get the name of the tool.

        Returns:
            The name of the tool
        """
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the tool.

        Returns:
            The description of the tool
        """
        return self._description

    @property
    def version(self) -> str:
        """Get the version of the tool.

        Returns:
            The version of the tool
        """
        return self._version
        
    @property
    def tool_dependencies(self) -> List[ToolDependency]:
        """Get the tool dependencies.
        
        Returns:
            The list of tool dependencies
        """
        return self._tool_dependencies

    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies of the tool.

        Returns:
            A list of plugin names that this tool depends on
        """
        # Include framework plugin dependencies here if needed
        return []

    @property
    def input_schema(self) -> Optional[Type[InputType]]:
        """Get the input schema for the tool.

        Returns:
            The input schema, or None if not defined
        """
        return self._input_schema

    @property
    def output_schema(self) -> Optional[Type[OutputType]]:
        """Get the output schema for the tool.

        Returns:
            The output schema, or None if not defined
        """
        return self._output_schema

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with the given parameters.

        Args:
            **kwargs: The parameters for the tool

        Returns:
            The result of running the tool
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.

        Returns:
            A dictionary describing the parameters for the tool
        """
        # If an input schema is defined, use it
        if self._input_schema is not None:
            return model_to_json_schema(self._input_schema)

        # Otherwise, try to infer the schema from the run method
        # Create a dynamic input schema from the run method's signature
        dynamic_schema = create_input_schema_from_signature(self.run)

        # Convert the schema to JSON Schema
        schema = model_to_json_schema(dynamic_schema)

        # Enhance the schema with docstring information
        schema = enhance_schema_with_docstring(schema, self.run)

        return schema
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata for this tool.
        
        Returns:
            A dictionary of metadata
        """
        meta = {
            "name": self.name,
            "description": self.description, 
            "version": self.version,
            "schema": self.get_schema()
        }
        
        # Add dependencies if present
        if self.tool_dependencies:
            meta["tool_dependencies"] = [str(dep) for dep in self.tool_dependencies]
        
        return meta

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        This method is called when the plugin is registered.

        Args:
            registry: The plugin registry
        """
        logger.info(f"Initializing tool: {self.name} v{self.version}")

    def shutdown(self) -> None:
        """Shutdown the plugin.

        This method is called when the plugin is unregistered.
        """
        logger.info(f"Shutting down tool: {self.name} v{self.version}")


class EnhancedToolRegistry(IToolRegistry, Plugin):
    """Registry for tools in the Agentor framework."""

    def __init__(self, auto_discover: bool = False):
        """Initialize the tool registry.

        Args:
            auto_discover: Whether to automatically discover tools
        """
        # Store tools by name and version: {name: {version: tool}}
        self._tools: Dict[str, Dict[str, ITool]] = {}
        self._discovered_modules: Set[str] = set()

        # Auto-discover tools if requested
        if auto_discover:
            self.discover_tools()

    @property
    def name(self) -> str:
        """Get the name of the plugin.

        Returns:
            The name of the plugin
        """
        return "tool_registry"

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
        return "Registry for tools in the Agentor framework"

    def register_tool(self, tool: ITool) -> None:
        """Register a tool.

        Args:
            tool: The tool to register

        Raises:
            ValueError: If a tool with the same name and version is already registered
        """
        name, version = tool.name, tool.version
        
        # Ensure the name entry exists
        if name not in self._tools:
            self._tools[name] = {}
        
        # Check if this exact version is already registered
        if version in self._tools[name]:
            raise ValueError(f"Tool '{name}' version '{version}' is already registered")

        self._tools[name][version] = tool
        logger.info(f"Registered tool: {name} v{version}")

    def unregister_tool(self, tool_name: str, version: str = None) -> None:
        """Unregister a tool.

        Args:
            tool_name: The name of the tool to unregister
            version: The version to unregister, or None for all versions

        Raises:
            ValueError: If the tool or version is not registered
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        if version is None:
            # Unregister all versions
            del self._tools[tool_name]
            logger.info(f"Unregistered all versions of tool: {tool_name}")
        else:
            # Unregister specific version
            if version not in self._tools[tool_name]:
                raise ValueError(f"Tool '{tool_name}' version '{version}' is not registered")
            
            del self._tools[tool_name][version]
            logger.info(f"Unregistered tool: {tool_name} v{version}")
            
            # Remove the tool name if no versions remain
            if not self._tools[tool_name]:
                del self._tools[tool_name]

    def get_tool(self, tool_name: str, version: str = None, version_constraint: str = None) -> Optional[ITool]:
        """Get a tool by name and optional version.

        Args:
            tool_name: The name of the tool
            version: The specific version to get, or None for latest
            version_constraint: A version constraint string (e.g., ">=1.0.0,<2.0.0")

        Returns:
            The tool, or None if not found or no version satisfies the constraint
        """
        if tool_name not in self._tools or not self._tools[tool_name]:
            return None
            
        versions = self._tools[tool_name]
            
        # Case 1: Specific version requested
        if version is not None:
            return versions.get(version)
        
        # Case 2: Version constraint provided
        if version_constraint is not None:
            version_range = VersionRange.parse(version_constraint)
            best_match = version_range.get_best_matching_version(list(versions.keys()))
            if best_match:
                return versions[best_match]
            return None
        
        # Case 3: No version specified, return latest
        latest_version = max(versions.keys(), key=lambda v: SemanticVersion(v))
        return versions[latest_version]

    def get_tools(self, include_all_versions: bool = False) -> Dict[str, Union[ITool, Dict[str, ITool]]]:
        """Get all registered tools.

        Args:
            include_all_versions: Whether to include all versions of each tool

        Returns:
            If include_all_versions is True, returns a dictionary of tool names to
            dictionaries of versions to tools. Otherwise, returns a dictionary of
            tool names to the latest version of each tool.
        """
        if include_all_versions:
            return {name: versions.copy() for name, versions in self._tools.items()}
        
        # Return only the latest version of each tool
        result = {}
        for name, versions in self._tools.items():
            if versions:
                latest_version = max(versions.keys(), key=lambda v: SemanticVersion(v))
                result[name] = versions[latest_version]
                
        return result

    def discover_tools(self, package_name: str = 'agentor') -> None:
        """Discover and register tools from a package.

        Args:
            package_name: The name of the package to discover tools from
        """
        logger.info(f"Discovering tools from package: {package_name}")

        try:
            # Import the package
            package = importlib.import_module(package_name)
            package_path = getattr(package, '__path__', [None])[0]

            if not package_path:
                logger.warning(f"Could not find path for package: {package_name}")
                return

            # Track discovered modules to avoid duplicates
            if package_name in self._discovered_modules:
                logger.debug(f"Package already discovered: {package_name}")
                return

            self._discovered_modules.add(package_name)

            # Walk through the package and discover tools
            for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
                full_module_name = f"{package_name}.{module_name}"

                # If it's a package, recursively discover tools
                if is_pkg:
                    self.discover_tools(full_module_name)
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(full_module_name)

                    # Find all classes in the module
                    for name in dir(module):
                        obj = getattr(module, name)

                        # Check if it's a class and a subclass of EnhancedTool
                        if (inspect.isclass(obj) and
                            obj is not EnhancedTool and
                            issubclass(obj, EnhancedTool)):

                            try:
                                # Create an instance of the tool
                                tool = obj()

                                # Register the tool
                                self.register_tool(tool)
                                logger.info(f"Discovered and registered tool: {tool.name} v{tool.version} from {full_module_name}")
                            except Exception as e:
                                logger.warning(f"Failed to instantiate tool {name} from {full_module_name}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Failed to import module {full_module_name}: {str(e)}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from package {package_name}: {str(e)}")

    def discover_tools_from_entry_points(self, group: str = 'agentor.tools') -> None:
        """Discover and register tools from entry points.

        Args:
            group: The entry point group to discover tools from
        """
        try:
            import importlib.metadata as metadata
        except ImportError:
            # Python < 3.8
            import importlib_metadata as metadata

        logger.info(f"Discovering tools from entry points: {group}")

        try:
            # Get all entry points in the specified group
            for entry_point in metadata.entry_points(group=group):
                try:
                    # Load the entry point
                    tool_class = entry_point.load()

                    # Check if it's a class and a subclass of EnhancedTool
                    if (inspect.isclass(tool_class) and
                        tool_class is not EnhancedTool and
                        issubclass(tool_class, EnhancedTool)):

                        try:
                            # Create an instance of the tool
                            tool = tool_class()

                            # Register the tool
                            self.register_tool(tool)
                            logger.info(f"Discovered and registered tool: {tool.name} v{tool.version} from entry point {entry_point.name}")
                        except Exception as e:
                            logger.warning(f"Failed to instantiate tool {entry_point.name}: {str(e)}")

                except Exception as e:
                    logger.warning(f"Failed to load entry point {entry_point.name}: {str(e)}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from entry points: {str(e)}")
            
    def resolve_dependencies(self, tool: EnhancedTool) -> Dict[str, ITool]:
        """Resolve dependencies for a tool.
        
        Args:
            tool: The tool to resolve dependencies for
            
        Returns:
            A dictionary of dependency names to dependency tools
            
        Raises:
            ValueError: If any dependency cannot be resolved
        """
        if not hasattr(tool, 'tool_dependencies') or not tool.tool_dependencies:
            return {}
            
        resolved = {}
        for dep in tool.tool_dependencies:
            dependency = None
            
            # Try to find a tool that satisfies this dependency
            if dep.version_constraint:
                # Get the best matching version
                constraint_str = ", ".join(str(c) for c in dep.version_constraint)
                dependency = self.get_tool(dep.tool_name, version_constraint=constraint_str)
            else:
                # Get the latest version
                dependency = self.get_tool(dep.tool_name)
            
            if dependency is None:
                raise ValueError(f"Cannot resolve dependency: {dep}")
                
            resolved[dep.tool_name] = dependency
            
        return resolved

    def initialize(self, registry) -> None:
        """Initialize the plugin.

        This method is called when the plugin is registered.

        Args:
            registry: The plugin registry
        """
        logger.info("Initializing tool registry")

    def shutdown(self) -> None:
        """Shutdown the plugin.

        This method is called when the plugin is unregistered.
        """
        logger.info("Shutting down tool registry")

    def create_api_server(self, settings=None):
        """Create an API server for the tools in this registry.

        Args:
            settings: Optional settings for the API server

        Returns:
            The API server
        """
        # Import here to avoid circular imports
        from agentor.agents.api.server import create_tool_api_server
        return create_tool_api_server(self, settings)

    def generate_openapi_schema(self, title="Agentor Tool API", description="API for accessing Agentor tools", version="0.1.0"):
        """Generate an OpenAPI schema for the tools in this registry.

        Args:
            title: The API title
            description: The API description
            version: The API version

        Returns:
            The OpenAPI schema
        """
        # Import here to avoid circular imports
        from agentor.agents.api.openapi import generate_openapi_schema
        
        # Get the latest version of each tool
        tools = {name: tool for name, tool in self.get_tools(include_all_versions=False).items()}
        
        return generate_openapi_schema(
            tools=tools,
            title=title,
            description=description,
            version=version
        )

    def save_openapi_schema(self, file_path, title="Agentor Tool API", description="API for accessing Agentor tools", version="0.1.0"):
        """Save an OpenAPI schema for the tools in this registry to a file.

        Args:
            file_path: The file path to save to
            title: The API title
            description: The API description
            version: The API version
        """
        # Import here to avoid circular imports
        from agentor.agents.api.openapi import save_openapi_schema
        schema = self.generate_openapi_schema(title, description, version)
        save_openapi_schema(schema, file_path)


# Create some example enhanced tools with Pydantic schemas

# Weather tool schemas
class WeatherToolInput(ToolInputSchema):
    """Input schema for the weather tool."""
    location: str = Field(..., description="The location to get weather for")


class WeatherToolOutput(ToolOutputSchema):
    """Output schema for the weather tool."""
    location: str = Field(..., description="The location")
    temperature: float = Field(..., description="The temperature in Fahrenheit")
    conditions: str = Field(..., description="The weather conditions")
    humidity: int = Field(..., description="The humidity percentage")
    wind_speed: float = Field(..., description="The wind speed in mph")


class WeatherTool(EnhancedTool):
    """Tool for getting weather information."""

    def __init__(self):
        """Initialize the weather tool."""
        super().__init__(
            name="weather",
            description="Get weather information for a location",
            version="1.0.0",
            input_schema=WeatherToolInput,
            output_schema=WeatherToolOutput
        )

    async def run(self, location: str) -> ToolResult:
        """Get weather information for a location.

        Args:
            location: The location to get weather for

        Returns:
            The weather information
        """
        # Validate input using the schema
        input_data = WeatherToolInput(location=location)

        # In a real implementation, this would call a weather API
        # For now, we'll just return some mock data
        output_data = WeatherToolOutput(
            location=input_data.location,
            temperature=72.5,
            conditions="sunny",
            humidity=45,
            wind_speed=5.2
        )

        return ToolResult(
            success=True,
            data=output_data.dict()
        )


# News tool schemas
class NewsToolInput(ToolInputSchema):
    """Input schema for the news tool."""
    topic: str = Field(..., description="The topic to get news for")
    count: int = Field(3, description="The number of headlines to return", ge=1, le=10)


class NewsToolOutput(ToolOutputSchema):
    """Output schema for the news tool."""
    topic: str = Field(..., description="The topic")
    headlines: List[str] = Field(..., description="The news headlines")


class NewsTool(EnhancedTool):
    """Tool for getting news headlines."""

    def __init__(self):
        """Initialize the news tool."""
        super().__init__(
            name="news",
            description="Get news headlines for a topic",
            version="1.0.0",
            input_schema=NewsToolInput,
            output_schema=NewsToolOutput
        )

    async def run(self, topic: str, count: int = 3) -> ToolResult:
        """Get news headlines for a topic.

        Args:
            topic: The topic to get news for
            count: The number of headlines to return

        Returns:
            The news headlines
        """
        # Validate input using the schema
        input_data = NewsToolInput(topic=topic, count=count)

        # In a real implementation, this would call a news API
        # For now, we'll just return some mock data
        headlines = [
            f"Latest {input_data.topic} news story",
            f"Breaking: New developments in {input_data.topic}",
            f"{input_data.topic} experts weigh in on recent events",
            f"Analysis: What the {input_data.topic} trends mean for you",
            f"Opinion: The future of {input_data.topic}"
        ]

        output_data = NewsToolOutput(
            topic=input_data.topic,
            headlines=headlines[:input_data.count]
        )

        return ToolResult(
            success=True,
            data=output_data.dict()
        )


# Calculator tool schemas
class CalculatorToolInput(ToolInputSchema):
    """Input schema for the calculator tool."""
    expression: str = Field(..., description="The expression to evaluate")


class CalculatorToolOutput(ToolOutputSchema):
    """Output schema for the calculator tool."""
    expression: str = Field(..., description="The expression that was evaluated")
    result: Any = Field(..., description="The result of the calculation")


class CalculatorTool(EnhancedTool):
    """Tool for performing calculations."""

    def __init__(self):
        """Initialize the calculator tool."""
        super().__init__(
            name="calculator",
            description="Perform a calculation",
            version="1.0.0", 
            input_schema=CalculatorToolInput,
            output_schema=CalculatorToolOutput
        )

    async def run(self, expression: str) -> ToolResult:
        """Perform a calculation.

        Args:
            expression: The expression to evaluate

        Returns:
            The result of the calculation
        """
        # Validate input using the schema
        input_data = CalculatorToolInput(expression=expression)

        try:
            # Use ast.literal_eval for safe evaluation
            # This only allows literals like numbers, strings, lists, dicts, etc.
            # For more complex expressions, we use a custom safe evaluator
            try:
                # First try with literal_eval for simple expressions
                result = ast.literal_eval(input_data.expression)
            except (ValueError, SyntaxError):
                # If that fails, try our custom safe evaluator for arithmetic
                result = self._safe_eval(input_data.expression)

            output_data = CalculatorToolOutput(
                expression=input_data.expression,
                result=result
            )

            return ToolResult(
                success=True,
                data=output_data.dict()
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Error evaluating expression: {str(e)}"
            )

    def _safe_eval(self, expression: str):
        """Safely evaluate a mathematical expression.

        This only allows basic arithmetic operations (+, -, *, /, **, %) and numbers.

        Args:
            expression: The expression to evaluate

        Returns:
            The result of the calculation

        Raises:
            ValueError: If the expression contains unsupported operations
        """
        # Define allowed operators
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,  # Unary minus
        }

        def _eval(node):
            # Handle numbers
            if isinstance(node, ast.Num):
                return node.n
            # Handle unary operations (like -5)
            elif isinstance(node, ast.UnaryOp):
                if isinstance(node.op, tuple(allowed_operators.keys())):
                    return allowed_operators[type(node.op)](_eval(node.operand))
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            # Handle binary operations (like 2+3)
            elif isinstance(node, ast.BinOp):
                if isinstance(node.op, tuple(allowed_operators.keys())):
                    return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            # Handle parentheses
            elif isinstance(node, ast.Expr):
                return _eval(node.value)
            else:
                raise ValueError(f"Unsupported node type: {type(node).__name__}")

        # Parse the expression into an AST
        parsed = ast.parse(expression, mode='eval')

        # Evaluate the expression
        return _eval(parsed.body)


# Create an advanced calculator tool that depends on the base calculator
class AdvancedCalculatorToolInput(CalculatorToolInput):
    """Input schema for the advanced calculator tool."""
    pass


class AdvancedCalculatorToolOutput(CalculatorToolOutput):
    """Output schema for the advanced calculator tool."""
    steps: List[str] = Field([], description="Steps taken to calculate the result")


class AdvancedCalculatorTool(EnhancedTool):
    """Advanced calculator with step-by-step explanation."""
    
    def __init__(self):
        """Initialize the advanced calculator tool."""
        super().__init__(
            name="advanced_calculator",
            description="Advanced calculator with step-by-step explanation",
            version="1.0.0",
            input_schema=AdvancedCalculatorToolInput,
            output_schema=AdvancedCalculatorToolOutput,
            tool_dependencies=[ToolDependency("calculator", ">=1.0.0")]
        )
        self._calculator = None
        
    async def run(self, expression: str) -> ToolResult:
        """Perform a calculation with step-by-step explanation.

        Args:
            expression: The expression to evaluate

        Returns:
            The result of the calculation with steps
        """
        # First, make sure we have our dependency
        if self._calculator is None:
            registry = get_tool_registry()
            try:
                dependencies = registry.resolve_dependencies(self)
                self._calculator = dependencies.get("calculator")
                if self._calculator is None:
                    return ToolResult(
                        success=False,
                        error="Required dependency 'calculator' not found"
                    )
            except ValueError as e:
                return ToolResult(
                    success=False,
                    error=f"Failed to resolve dependencies: {str(e)}"
                )
                
        # Validate input
        input_data = AdvancedCalculatorToolInput(expression=expression)
        
        # Use the calculator to evaluate the expression
        result = await self._calculator.run(expression=input_data.expression)
        
        if not result.success:
            return result
            
        # Generate steps (in a real implementation, this would be more sophisticated)
        steps = [
            f"Parsing expression: {input_data.expression}",
            "Converting to abstract syntax tree",
            "Evaluating operations according to precedence",
            f"Final result: {result.data['result']}"
        ]
        
        # Create output
        output_data = AdvancedCalculatorToolOutput(
            expression=input_data.expression,
            result=result.data["result"],
            steps=steps
        )
        
        return ToolResult(
            success=True,
            data=output_data.dict()
        )


# Global tool registry
tool_registry = EnhancedToolRegistry()


def get_tool_registry() -> EnhancedToolRegistry:
    """Get the global tool registry.

    Returns:
        The global tool registry
    """
    return tool_registry


# Register the example tools
weather_tool = WeatherTool()
news_tool = NewsTool()
calculator_tool = CalculatorTool()
advanced_calculator_tool = AdvancedCalculatorTool()

tool_registry.register_tool(weather_tool)
tool_registry.register_tool(news_tool)
tool_registry.register_tool(calculator_tool)
tool_registry.register_tool(advanced_calculator_tool)
