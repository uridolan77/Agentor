"""
Tool composition for the Agentor framework.

This module provides classes for composing multiple tools together into pipelines,
parallel execution flows, conditional branches, and dynamic composition patterns.
These composition patterns enable building complex workflows from simple tools
while maintaining proper validation, error handling, and dependency management.
"""

from typing import Dict, Any, List, Optional, Callable, Union, TypeVar, Generic, Type, Set, Tuple, cast
import logging
import asyncio
import inspect
import json
import traceback
from datetime import datetime
from pydantic import BaseModel, Field, create_model, ValidationError
from abc import ABC, abstractmethod

from agentor.core.interfaces.tool import ITool, ToolResult, IToolRegistry
from agentor.agents.enhanced_tools import EnhancedTool
from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema, model_to_json_schema
from agentor.agents.versioning import ComponentVersion, versioned

logger = logging.getLogger(__name__)

# Type variables for input and output
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')
IntermediateType = TypeVar('IntermediateType')


class CompositionError(Exception):
    """Base exception for tool composition errors."""
    pass


class SchemaValidationError(CompositionError):
    """Exception raised when schema validation fails."""
    pass


class ToolExecutionError(CompositionError):
    """Exception raised when tool execution fails."""
    
    def __init__(self, tool_name: str, message: str, node_index: Optional[int] = None):
        """Initialize the tool execution error.

        Args:
            tool_name: The name of the tool that failed
            message: The error message
            node_index: Optional index of the node in a pipeline
        """
        self.tool_name = tool_name
        self.node_index = node_index
        super().__init__(f"Error executing tool '{tool_name}'{f' (node {node_index})' if node_index is not None else ''}: {message}")


class PipelineConfigurationError(CompositionError):
    """Exception raised when pipeline configuration is invalid."""
    pass


class CircularDependencyError(CompositionError):
    """Exception raised when circular dependencies are detected."""
    pass


class ToolTransformer(Generic[InputType, OutputType]):
    """A transformer that modifies data between tools in a pipeline."""

    def __init__(
        self, 
        transform_function: Callable[[InputType], OutputType],
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize the transformer.

        Args:
            transform_function: The function that transforms the data
            name: Optional name for the transformer (defaults to the function name)
            description: Optional description of what the transformer does
        """
        self.transform_function = transform_function
        self.name = name or getattr(transform_function, "__name__", "transformer")
        self.description = description or getattr(
            transform_function, "__doc__", "Transforms data between tools"
        ).strip()

    async def transform(self, data: InputType) -> OutputType:
        """Transform the data.

        Args:
            data: The data to transform

        Returns:
            The transformed data
            
        Raises:
            Exception: If the transformation function raises an exception
        """
        try:
            if asyncio.iscoroutinefunction(self.transform_function):
                return await self.transform_function(data)
            else:
                return self.transform_function(data)
        except Exception as e:
            logger.error(f"Error in transformer '{self.name}': {str(e)}")
            raise CompositionError(f"Transformer '{self.name}' failed: {str(e)}") from e


class ToolCondition:
    """A condition used for branching in tool pipelines."""

    def __init__(
        self, 
        condition_function: Callable[[Any], bool],
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """Initialize the condition.

        Args:
            condition_function: The function that evaluates the condition
            name: Optional name for the condition (defaults to the function name)
            description: Optional description of what the condition checks
        """
        self.condition_function = condition_function
        self.name = name or getattr(condition_function, "__name__", "condition")
        self.description = description or getattr(
            condition_function, "__doc__", "Evaluates a condition for branching"
        ).strip()

    async def evaluate(self, data: Any) -> bool:
        """Evaluate the condition.

        Args:
            data: The data to evaluate the condition on

        Returns:
            The result of the condition
            
        Raises:
            CompositionError: If the condition function raises an exception
        """
        try:
            if asyncio.iscoroutinefunction(self.condition_function):
                return await self.condition_function(data)
            else:
                return self.condition_function(data)
        except Exception as e:
            logger.error(f"Error in condition '{self.name}': {str(e)}")
            raise CompositionError(f"Condition '{self.name}' evaluation failed: {str(e)}") from e


class SchemaValidator:
    """Validator for tool input and output schemas."""

    @staticmethod
    def get_tool_input_schema(tool: ITool) -> Optional[Dict[str, Any]]:
        """Get the input schema for a tool.

        Args:
            tool: The tool to get the schema for

        Returns:
            The input schema, or None if not available
        """
        if hasattr(tool, 'get_schema') and callable(tool.get_schema):
            schema = tool.get_schema()
            if schema and 'properties' in schema:
                return schema

        if hasattr(tool, 'input_schema') and tool.input_schema:
            if hasattr(tool.input_schema, 'schema') and callable(tool.input_schema.schema):
                return tool.input_schema.schema()

        return None

    @staticmethod
    def get_tool_output_schema(tool: ITool) -> Optional[Dict[str, Any]]:
        """Get the output schema for a tool.

        Args:
            tool: The tool to get the schema for

        Returns:
            The output schema, or None if not available
        """
        if hasattr(tool, 'output_schema') and tool.output_schema:
            if hasattr(tool.output_schema, 'schema') and callable(tool.output_schema.schema):
                return tool.output_schema.schema()

        return None

    @staticmethod
    def validate_mapping(
        source_schema: Dict[str, Any], 
        target_schema: Dict[str, Any],
        mapping: Dict[str, str]
    ) -> List[str]:
        """Validate that a mapping between two schemas is compatible.

        Args:
            source_schema: The source schema
            target_schema: The target schema
            mapping: The mapping from source to target

        Returns:
            A list of validation errors, or an empty list if valid
        """
        errors = []

        # Get the properties from both schemas
        source_props = source_schema.get('properties', {})
        target_props = target_schema.get('properties', {})
        
        # Check for required target properties that aren't mapped
        target_required = target_schema.get('required', [])
        mapped_target_keys = set(mapping.values())
        missing_required = [key for key in target_required if key not in mapped_target_keys]
        
        if missing_required:
            errors.append(f"Missing required target properties: {', '.join(missing_required)}")

        # Check each mapping
        for source_key, target_key in mapping.items():
            # Check if the source key exists
            if source_key not in source_props:
                errors.append(f"Source key '{source_key}' not found in source schema")
                continue

            # Check if the target key exists
            if target_key not in target_props:
                errors.append(f"Target key '{target_key}' not found in target schema")
                continue

            # Check type compatibility
            source_type = source_props[source_key].get('type')
            target_type = target_props[target_key].get('type')

            if source_type and target_type and source_type != target_type:
                # Some basic type compatibility checks
                if not SchemaValidator._are_types_compatible(source_type, target_type):
                    errors.append(f"Type mismatch: '{source_key}' is {source_type}, but '{target_key}' is {target_type}")

        return errors

    @staticmethod
    def _are_types_compatible(source_type: str, target_type: str) -> bool:
        """Check if two types are compatible.

        Args:
            source_type: The source type
            target_type: The target type

        Returns:
            True if the types are compatible, False otherwise
        """
        # Same types are always compatible
        if source_type == target_type:
            return True

        # Number types can be compatible
        if source_type in ('integer', 'number') and target_type in ('integer', 'number'):
            return True

        # String can be converted to many types
        if source_type == 'string' and target_type in ('integer', 'number', 'boolean'):
            return True
            
        # Arrays and objects can often be represented as strings (JSON)
        if source_type in ('array', 'object') and target_type == 'string':
            return True

        # Any type can be converted to string with a suitable transformer
        if target_type == 'string':
            return True

        return False
        
    @staticmethod
    def validate_data_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate data against a JSON schema.
        
        Args:
            data: The data to validate
            schema: The schema to validate against
            
        Returns:
            A list of validation errors, or an empty list if valid
        """
        errors = []
        
        # Check required properties
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                
        # Check property types
        properties = schema.get('properties', {})
        for field_name, field_value in data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                expected_type = field_schema.get('type')
                
                if expected_type:
                    # Basic type checking
                    valid = SchemaValidator._check_value_type(field_value, expected_type)
                    if not valid:
                        errors.append(f"Field '{field_name}' has invalid type: expected {expected_type}")
                        
        return errors
    
    @staticmethod
    def _check_value_type(value: Any, expected_type: str) -> bool:
        """Check if a value matches an expected type.
        
        Args:
            value: The value to check
            expected_type: The expected type
            
        Returns:
            True if the value matches the expected type, False otherwise
        """
        if expected_type == 'null':
            return value is None
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'integer':
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == 'number':
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        return True  # Unknown type, assume valid


class ToolExecutionStats:
    """Statistics about tool execution."""
    
    def __init__(self):
        """Initialize the tool execution statistics."""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration_ms: Optional[float] = None
        self.input_size: Optional[int] = None
        self.output_size: Optional[int] = None
        self.success: bool = False
        self.error: Optional[str] = None
        
    @property
    def elapsed_time_ms(self) -> Optional[float]:
        """Get the elapsed time in milliseconds.
        
        Returns:
            The elapsed time in milliseconds, or None if not available
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
        
    def start(self, input_data: Dict[str, Any]):
        """Start timing the execution.
        
        Args:
            input_data: The input data
        """
        self.start_time = datetime.now()
        try:
            self.input_size = len(json.dumps(input_data))
        except:
            self.input_size = None
            
    def end(self, success: bool, output_data: Optional[Dict[str, Any]], error: Optional[str] = None):
        """End timing the execution.
        
        Args:
            success: Whether the execution was successful
            output_data: The output data
            error: The error message, if any
        """
        self.end_time = datetime.now()
        self.success = success
        self.error = error
        
        if self.start_time:
            self.duration_ms = self.elapsed_time_ms
            
        if output_data:
            try:
                self.output_size = len(json.dumps(output_data))
            except:
                self.output_size = None
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "success": self.success,
            "error": self.error
        }


class ToolNode:
    """A node in a tool pipeline.
    
    Each node represents a tool execution step within a pipeline, including
    input/output mapping and validation logic.
    """

    def __init__(
        self,
        tool: ITool,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        validate_schemas: bool = True,
        required_inputs: Optional[List[str]] = None,
        pre_processor: Optional[ToolTransformer] = None,
        post_processor: Optional[ToolTransformer] = None
    ):
        """Initialize the tool node.

        Args:
            tool: The tool to execute
            input_mapping: Optional mapping of pipeline data keys to tool parameter names
            output_mapping: Optional mapping of tool output keys to pipeline data keys
            validate_schemas: Whether to validate schemas between connected tools
            required_inputs: Optional list of required inputs from the pipeline
            pre_processor: Optional transformer to apply to inputs before tool execution
            post_processor: Optional transformer to apply to outputs after tool execution
        """
        self.tool = tool
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self.validate_schemas = validate_schemas
        self.required_inputs = required_inputs or []
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        
        # Get schemas
        self.input_schema = SchemaValidator.get_tool_input_schema(tool)
        self.output_schema = SchemaValidator.get_tool_output_schema(tool)
        
        # Stats tracking
        self.stats = ToolExecutionStats()

    def validate_input_mapping(self, pipeline_schema: Dict[str, Any]) -> List[str]:
        """Validate the input mapping against the pipeline schema.

        Args:
            pipeline_schema: The schema of the pipeline data

        Returns:
            A list of validation errors, or an empty list if valid
        """
        if not self.input_schema or not pipeline_schema:
            return []

        # Build a reverse mapping from tool param to pipeline key
        reverse_mapping = {tool_param: pipeline_key for pipeline_key, tool_param in self.input_mapping.items()}
        
        return SchemaValidator.validate_mapping(
            pipeline_schema, self.input_schema, reverse_mapping
        )

    def validate_output_mapping(self, pipeline_schema: Dict[str, Any]) -> List[str]:
        """Validate the output mapping against the pipeline schema.

        Args:
            pipeline_schema: The schema of the pipeline data

        Returns:
            A list of validation errors, or an empty list if valid
        """
        if not self.output_schema or not pipeline_schema:
            return []

        # For output mapping, the source is the tool's output and the target is the pipeline
        return SchemaValidator.validate_mapping(
            self.output_schema, pipeline_schema, self.output_mapping
        )

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with the given data.

        Args:
            data: The data to pass to the tool

        Returns:
            The updated data after executing the tool
            
        Raises:
            ToolExecutionError: If tool execution fails
            SchemaValidationError: If schema validation fails
        """
        # Start tracking execution stats
        self.stats.start(data)
        
        try:
            # Check for required inputs
            for required_key in self.required_inputs:
                if required_key not in data:
                    raise SchemaValidationError(f"Missing required input: {required_key}")

            # Map input data to tool parameters
            params = {}
            for pipeline_key, param_name in self.input_mapping.items():
                if pipeline_key in data:
                    params[param_name] = data[pipeline_key]

            # Use direct keys if no mapping specified
            if not self.input_mapping:
                params = data

            # Apply pre-processor if available
            if self.pre_processor:
                try:
                    params = await self.pre_processor.transform(params)
                except Exception as e:
                    raise CompositionError(f"Pre-processor for tool '{self.tool.name}' failed: {str(e)}") from e

            # Validate input parameters against the tool's schema if available
            if self.validate_schemas and self.input_schema:
                errors = SchemaValidator.validate_data_against_schema(params, self.input_schema)
                if errors:
                    error_msg = f"Tool '{self.tool.name}' input validation failed: {', '.join(errors)}"
                    logger.error(error_msg)
                    raise SchemaValidationError(error_msg)

            # Run the tool
            result = await self.tool.run(**params)

            # Handle tool failure
            if not result.success:
                raise ToolExecutionError(self.tool.name, result.error or "Unknown error")

            # Apply post-processor if available
            if self.post_processor:
                try:
                    result.data = await self.post_processor.transform(result.data)
                except Exception as e:
                    raise CompositionError(f"Post-processor for tool '{self.tool.name}' failed: {str(e)}") from e

            # Copy the output data
            result_data = data.copy()

            # Map tool output to pipeline data
            if self.output_mapping:
                for output_key, pipeline_key in self.output_mapping.items():
                    if output_key in result.data:
                        result_data[pipeline_key] = result.data[output_key]
            else:
                # No mapping, merge all output into the pipeline data
                result_data.update(result.data)
                
            # Update stats with success
            self.stats.end(True, result_data)

            return result_data
            
        except Exception as e:
            # Update stats with failure
            error_msg = str(e)
            self.stats.end(False, None, error_msg)
            
            # Re-raise appropriate exception type
            if isinstance(e, (SchemaValidationError, ToolExecutionError, CompositionError)):
                logger.error(f"Error in tool node {self.tool.name}: {error_msg}")
                raise
            else:
                logger.error(f"Unexpected error in tool node {self.tool.name}: {error_msg}")
                raise ToolExecutionError(self.tool.name, error_msg)


class PipelineSchema:
    """Schema for a tool pipeline.
    
    This class manages the combined schema of all tools in a pipeline,
    ensuring type compatibility between connected nodes.
    """

    def __init__(self):
        """Initialize the pipeline schema."""
        self.properties: Dict[str, Dict[str, Any]] = {}
        self.required: List[str] = []

    def add_property(self, name: str, schema: Dict[str, Any], required: bool = False):
        """Add a property to the schema.

        Args:
            name: The name of the property
            schema: The schema of the property
            required: Whether the property is required
        """
        self.properties[name] = schema
        if required and name not in self.required:
            self.required.append(name)

    def merge(self, other: 'PipelineSchema'):
        """Merge another schema into this one.

        Args:
            other: The schema to merge
        """
        for name, schema in other.properties.items():
            self.properties[name] = schema

        for name in other.required:
            if name not in self.required:
                self.required.append(name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary.

        Returns:
            The schema as a dictionary
        """
        return {
            "type": "object",
            "properties": self.properties,
            "required": self.required
        }

    @staticmethod
    def from_tool(tool: ITool) -> 'PipelineSchema':
        """Create a schema from a tool's input schema.

        Args:
            tool: The tool to create the schema from

        Returns:
            The created schema
        """
        schema = PipelineSchema()

        # Get the tool's input schema
        input_schema = SchemaValidator.get_tool_input_schema(tool)
        if not input_schema:
            return schema

        # Add properties from the input schema
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])

        for name, prop_schema in properties.items():
            schema.add_property(name, prop_schema, name in required)

        return schema
        
    @staticmethod
    def from_tools(tools: List[ITool]) -> 'PipelineSchema':
        """Create a combined schema from multiple tools.
        
        Args:
            tools: The tools to create the schema from
            
        Returns:
            The combined schema
        """
        schema = PipelineSchema()
        
        for tool in tools:
            tool_schema = PipelineSchema.from_tool(tool)
            schema.merge(tool_schema)
            
        return schema


@versioned(
    component_type="composition",
    component_id="ToolPipeline",
    version="1.0.0"
)
class ToolPipeline:
    """A pipeline of tools that are executed in sequence.
    
    Tool pipelines enable the composition of multiple tools into a single workflow,
    with automatic data passing between steps, validation, and error handling.
    
    Example:
        ```python
        # Create a pipeline that processes text
        pipeline = ToolPipeline("text_processor", "Process text with multiple tools")
        pipeline.add_tool(
            text_analyzer,
            input_mapping={"input_text": "text"},
            output_mapping={"sentiment": "text_sentiment"}
        )
        pipeline.add_tool(
            summarizer,
            input_mapping={"text": "input_text"},
            output_mapping={"summary": "text_summary"}
        )
        
        # Execute the pipeline
        result = await pipeline.execute({"input_text": "Some text to process"})
        # Result contains text_sentiment and text_summary
        ```
    """

    def __init__(self, name: str, description: str, validate_schemas: bool = True):
        """Initialize the tool pipeline.

        Args:
            name: The name of the pipeline
            description: A description of what the pipeline does
            validate_schemas: Whether to validate schemas between connected tools
        """
        self.name = name
        self.description = description
        self.nodes: List[ToolNode] = []
        self.validate_schemas = validate_schemas
        self.schema = PipelineSchema()
        self.current_index = 0
        self.last_execution_stats: List[Dict[str, Any]] = []

    def add_tool(
        self,
        tool: ITool,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        required_inputs: Optional[List[str]] = None,
        pre_processor: Optional[ToolTransformer] = None,
        post_processor: Optional[ToolTransformer] = None
    ) -> 'ToolPipeline':
        """Add a tool to the pipeline.

        Args:
            tool: The tool to add
            input_mapping: Optional mapping of pipeline data keys to tool parameter names
            output_mapping: Optional mapping of tool output keys to pipeline data keys
            required_inputs: Optional list of required inputs from the pipeline
            pre_processor: Optional transformer to apply to inputs before tool execution
            post_processor: Optional transformer to apply to outputs after tool execution

        Returns:
            The pipeline itself for method chaining
            
        Raises:
            PipelineConfigurationError: If there are schema validation errors and validate_schemas is True
        """
        node = ToolNode(
            tool, 
            input_mapping, 
            output_mapping, 
            self.validate_schemas,
            required_inputs,
            pre_processor,
            post_processor
        )

        # Validate the node's input mapping against the current pipeline schema
        if self.validate_schemas:
            errors = node.validate_input_mapping(self.schema.to_dict())
            if errors:
                error_msg = f"Schema validation errors when adding tool '{tool.name}' to pipeline '{self.name}':\n"
                error_msg += "\n".join([f"- {error}" for error in errors])
                logger.warning(error_msg)
                if any(error.startswith("Missing required") for error in errors):
                    raise PipelineConfigurationError(error_msg)

        # Add the node to the pipeline
        self.nodes.append(node)

        # Update the pipeline schema with the tool's output schema
        if node.output_schema:
            output_schema = PipelineSchema()
            properties = node.output_schema.get('properties', {})
            required = node.output_schema.get('required', [])

            for name, prop_schema in properties.items():
                # If there's an output mapping, use the mapped name
                if node.output_mapping and name in node.output_mapping:
                    mapped_name = node.output_mapping[name]
                    output_schema.add_property(mapped_name, prop_schema, name in required)
                else:
                    output_schema.add_property(name, prop_schema, name in required)

            self.schema.merge(output_schema)

        return self
        
    def add_transformer(self, transformer: ToolTransformer) -> 'ToolPipeline':
        """Add a standalone transformer to the pipeline.
        
        This is useful for data manipulation steps that don't require a full tool.
        
        Args:
            transformer: The transformer to add
            
        Returns:
            The pipeline itself for method chaining
        """
        # Create a lightweight "transformer tool" that just applies the transformer
        class TransformerTool(ITool):
            def __init__(self, transformer: ToolTransformer):
                self.transformer = transformer
                self.name = f"transformer_{transformer.name}"
                self.description = transformer.description
                
            async def run(self, **kwargs) -> ToolResult:
                try:
                    result = await self.transformer.transform(kwargs)
                    return ToolResult(success=True, data=result if isinstance(result, dict) else {"result": result})
                except Exception as e:
                    return ToolResult(success=False, error=str(e))
        
        # Add the transformer tool to the pipeline
        return self.add_tool(TransformerTool(transformer))

    def get_schema(self) -> Dict[str, Any]:
        """Get the schema of the pipeline.

        Returns:
            The pipeline schema
        """
        return self.schema.to_dict()
        
    def get_execution_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about the last execution.
        
        Returns:
            List of execution statistics for each node
        """
        return self.last_execution_stats

    async def execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with the given initial data.

        Args:
            initial_data: The initial data to pass to the pipeline

        Returns:
            The final data after executing all tools in the pipeline
            
        Raises:
            SchemaValidationError: If initial data validation fails
            ToolExecutionError: If a tool execution fails
            CircularDependencyError: If circular dependencies are detected
            PipelineConfigurationError: If pipeline configuration is invalid
        """
        current_data = initial_data.copy()
        self.last_execution_stats = []
        self.current_index = 0
        
        # Check if the pipeline has any nodes
        if not self.nodes:
            logger.warning(f"Pipeline '{self.name}' has no nodes")
            return current_data

        # Validate the initial data against the pipeline schema
        if self.validate_schemas:
            pipeline_schema = self.get_schema()
            try:
                # Create a dynamic model from the schema
                properties = pipeline_schema.get('properties', {})
                field_type_map = {}
                
                for field_name, field_schema in properties.items():
                    # Use Any type for all fields to avoid type validation issues
                    field_type_map[field_name] = (Any, ... if field_name in pipeline_schema.get('required', []) else None)
                    
                model = create_model('PipelineInput', **field_type_map)
                
                # Validate the data
                validated_input = model(**current_data)
                current_data = validated_input.dict()
            except ValidationError as e:
                logger.warning(f"Initial data validation warning in pipeline '{self.name}': {str(e)}")
                # Don't raise an error for missing fields, as they might be produced by the pipeline

        # Execute each node in sequence
        exec_depth = 0
        max_depth = len(self.nodes) * 2  # Safety limit to prevent infinite loops
        
        for i, node in enumerate(self.nodes):
            self.current_index = i
            
            # Safety check for circular dependencies
            exec_depth += 1
            if exec_depth > max_depth:
                raise CircularDependencyError(f"Possible circular dependency detected in pipeline '{self.name}' after {max_depth} steps")
                
            try:
                current_data = await node.execute(current_data)
                self.last_execution_stats.append(node.stats.to_dict())
            except ToolExecutionError as e:
                # Add node index to the error
                raise ToolExecutionError(e.tool_name, str(e), i)
            except Exception as e:
                # Unexpected error
                logger.error(f"Error executing tool {node.tool.name} (node {i+1}/{len(self.nodes)}) in pipeline {self.name}: {str(e)}")
                logger.error(traceback.format_exc())
                raise ToolExecutionError(node.tool.name, str(e), i)

        return current_data


class ConditionalBranch:
    """A conditional branch in a tool pipeline.
    
    Conditional branches allow for dynamic flow control within pipelines,
    executing different tools based on data conditions.
    """

    def __init__(
        self,
        condition: ToolCondition,
        true_branch: Union[ToolNode, ToolPipeline, 'ConditionalBranch'],
        false_branch: Optional[Union[ToolNode, ToolPipeline, 'ConditionalBranch']] = None,
        name: str = "conditional_branch"
    ):
        """Initialize the conditional branch.

        Args:
            condition: The condition to evaluate
            true_branch: The branch to execute if the condition is true
            false_branch: The branch to execute if the condition is false, or None
            name: Optional name for the branch
        """
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.name = name
        self.last_branch_taken: Optional[str] = None
        
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the branch with the given data.

        Args:
            data: The data to pass to the branch

        Returns:
            The updated data after executing the appropriate branch
            
        Raises:
            CompositionError: If condition evaluation or branch execution fails
        """
        try:
            # Evaluate the condition
            result = await self.condition.evaluate(data)
            
            # Record which branch was taken
            self.last_branch_taken = "true" if result else "false"

            # Execute the appropriate branch
            if result:
                logger.debug(f"Conditional branch '{self.name}': taking TRUE branch")
                return await self.true_branch.execute(data)
            elif self.false_branch is not None:
                logger.debug(f"Conditional branch '{self.name}': taking FALSE branch")
                return await self.false_branch.execute(data)
            else:
                # No false branch, return the data unchanged
                logger.debug(f"Conditional branch '{self.name}': condition is FALSE, no branch to take")
                return data
                
        except Exception as e:
            logger.error(f"Error in conditional branch '{self.name}': {str(e)}")
            raise CompositionError(f"Conditional branch '{self.name}' failed: {str(e)}") from e


@versioned(
    component_type="composition",
    component_id="ComposableTool",
    version="1.0.0"
)
class ComposableTool(EnhancedTool):
    """A tool that composes multiple tools into a pipeline.
    
    This allows creating new tools by combining existing tools into a pipeline,
    which can then be used as a single tool in other agents or pipelines.
    """

    InputType = TypeVar('InputType', bound=ToolInputSchema)
    OutputType = TypeVar('OutputType', bound=ToolOutputSchema)

    def __init__(
        self,
        name: str,
        description: str,
        pipeline: ToolPipeline,
        version: str = "1.0.0",
        input_schema: Optional[Type[InputType]] = None,
        output_schema: Optional[Type[OutputType]] = None,
        tool_dependencies: Optional[List[str]] = None
    ):
        """Initialize the composable tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            pipeline: The pipeline of tools to execute
            version: The version of the tool
            input_schema: Optional Pydantic model for input validation
            output_schema: Optional Pydantic model for output validation
            tool_dependencies: Optional list of tool dependencies
        """
        # Extract dependencies from the pipeline if not provided
        if tool_dependencies is None:
            tool_dependencies = []
            for node in pipeline.nodes:
                tool_name = node.tool.name
                if tool_name not in tool_dependencies:
                    tool_dependencies.append(f"{tool_name}:>=1.0.0")

        super().__init__(
            name=name,
            description=description,
            version=version,
            input_schema=input_schema,
            output_schema=output_schema,
            tool_dependencies=tool_dependencies
        )

        self.pipeline = pipeline
        self.pipeline_stats: List[Dict[str, Any]] = []

    async def run(self, **kwargs) -> ToolResult:
        """Run the pipeline with the given parameters.

        Args:
            **kwargs: The parameters for the pipeline

        Returns:
            The result of running the pipeline
        """
        try:
            # Reset stats
            self.pipeline_stats = []
            
            # Validate input if schema exists
            if self.input_schema:
                try:
                    input_data = self.input_schema(**kwargs)
                    kwargs = input_data.dict()
                except ValidationError as e:
                    logger.error(f"Input validation failed for composable tool '{self.name}': {str(e)}")
                    return ToolResult(success=False, error=f"Input validation failed: {str(e)}")

            # Execute the pipeline
            result_data = await self.pipeline.execute(kwargs)
            
            # Save execution stats
            self.pipeline_stats = self.pipeline.get_execution_stats()

            # Validate output if schema exists
            if self.output_schema:
                try:
                    output_data = self.output_schema(**result_data)
                    result_data = output_data.dict()
                except ValidationError as e:
                    logger.error(f"Output validation failed for composable tool '{self.name}': {str(e)}")
                    return ToolResult(success=False, error=f"Output validation failed: {str(e)}")

            return ToolResult(success=True, data=result_data)
        except Exception as e:
            logger.error(f"Error executing pipeline in tool {self.name}: {str(e)}")
            return ToolResult(success=False, error=str(e))
            
    def get_execution_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about the last execution.
        
        Returns:
            List of execution statistics for each node in the pipeline
        """
        return self.pipeline_stats


@versioned(
    component_type="composition",
    component_id="ParallelToolPipeline",
    version="1.0.0"
)
class ParallelToolPipeline:
    """A pipeline that executes multiple tools in parallel.
    
    This class allows executing multiple tools simultaneously for improved
    performance when tools don't depend on each other's outputs.
    """

    def __init__(self, name: str, description: str):
        """Initialize the parallel tool pipeline.

        Args:
            name: The name of the pipeline
            description: A description of what the pipeline does
        """
        self.name = name
        self.description = description
        self.nodes: List[ToolNode] = []
        self.last_execution_stats: List[Dict[str, Any]] = []

    def add_tool(
        self,
        tool: ITool,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        pre_processor: Optional[ToolTransformer] = None,
        post_processor: Optional[ToolTransformer] = None
    ) -> 'ParallelToolPipeline':
        """Add a tool to the pipeline.

        Args:
            tool: The tool to add
            input_mapping: Optional mapping of pipeline data keys to tool parameter names
            output_mapping: Optional mapping of tool output keys to pipeline data keys
            pre_processor: Optional transformer to apply to inputs before tool execution
            post_processor: Optional transformer to apply to outputs after tool execution

        Returns:
            The pipeline itself for method chaining
        """
        self.nodes.append(
            ToolNode(
                tool, 
                input_mapping, 
                output_mapping,
                pre_processor=pre_processor,
                post_processor=post_processor
            )
        )
        return self
        
    def get_execution_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about the last execution.
        
        Returns:
            List of execution statistics for each node
        """
        return self.last_execution_stats

    async def execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all tools in parallel with the given initial data.

        Args:
            initial_data: The initial data to pass to the pipeline

        Returns:
            The merged data from all parallel executions
            
        Raises:
            CompositionError: If any parallel execution fails
        """
        # Reset stats
        self.last_execution_stats = []
        
        # Create tasks for all nodes
        tasks = []
        for node in self.nodes:
            tasks.append(node.execute(initial_data.copy()))

        # Run all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect stats from all nodes
            for node in self.nodes:
                self.last_execution_stats.append(node.stats.to_dict())
                
            # Check for exceptions
            errors = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    node_name = self.nodes[i].tool.name
                    error_msg = f"Tool '{node_name}' failed: {str(result)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    
            if errors:
                raise CompositionError(f"Errors in parallel execution: {'; '.join(errors)}")
                
            # Filter out exceptions
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            # Merge results (last node's outputs have priority for key conflicts)
            merged_data = initial_data.copy()
            for result_data in valid_results:
                merged_data.update(result_data)

            return merged_data
            
        except asyncio.CancelledError:
            logger.warning(f"Parallel pipeline '{self.name}' execution was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in parallel pipeline {self.name}: {str(e)}")
            raise CompositionError(f"Error in parallel pipeline {self.name}: {str(e)}") from e


class FanOutProcessor:
    """A processor that fans out to multiple tools and combines their results.
    
    This is useful for processing multiple items in parallel using the same tool,
    or applying multiple tools to the same input and merging the results.
    """
    
    def __init__(
        self, 
        name: str,
        item_extractor: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        result_combiner: Callable[[Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]
    ):
        """Initialize the fan-out processor.
        
        Args:
            name: The name of the processor
            item_extractor: Function to extract items to process in parallel
            result_combiner: Function to combine the results back into a single output
        """
        self.name = name
        self.item_extractor = item_extractor
        self.result_combiner = result_combiner
        self.tool: Optional[ITool] = None
        self.pipeline: Optional[ToolPipeline] = None
        
    def with_tool(self, tool: ITool) -> 'FanOutProcessor':
        """Set the tool to use for processing items.
        
        Args:
            tool: The tool to apply to each item
            
        Returns:
            Self for method chaining
        """
        self.tool = tool
        self.pipeline = None
        return self
        
    def with_pipeline(self, pipeline: ToolPipeline) -> 'FanOutProcessor':
        """Set the pipeline to use for processing items.
        
        Args:
            pipeline: The pipeline to apply to each item
            
        Returns:
            Self for method chaining
        """
        self.pipeline = pipeline
        self.tool = None
        return self
        
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data by fanning out and then combining results.
        
        Args:
            data: The input data
            
        Returns:
            The combined results
            
        Raises:
            CompositionError: If no tool or pipeline is set or if processing fails
        """
        if not self.tool and not self.pipeline:
            raise CompositionError(f"FanOutProcessor '{self.name}' has no tool or pipeline set")
            
        try:
            # Extract items to process
            items = self.item_extractor(data)
            
            if not items:
                logger.warning(f"FanOutProcessor '{self.name}' extracted 0 items to process")
                return data
                
            # Create tasks for parallel processing
            tasks = []
            for item in items:
                if self.tool:
                    tasks.append(self.tool.run(**item))
                else:
                    tasks.append(self.pipeline.execute(item))  # type: ignore
                    
            # Run tasks in parallel
            results = await asyncio.gather(*tasks)
            
            # Extract data from tool results if needed
            processed_items = []
            for i, result in enumerate(results):
                if self.tool:
                    # For tools, we get ToolResult objects
                    if isinstance(result, ToolResult):
                        if result.success:
                            processed_items.append(result.data)
                        else:
                            raise CompositionError(f"Tool execution failed for item {i}: {result.error}")
                    else:
                        processed_items.append(result)
                else:
                    # For pipelines, we get the data directly
                    processed_items.append(result)
                    
            # Combine results
            return self.result_combiner(data, processed_items)
            
        except Exception as e:
            logger.error(f"Error in FanOutProcessor '{self.name}': {str(e)}")
            raise CompositionError(f"FanOutProcessor '{self.name}' failed: {str(e)}") from e


@versioned(
    component_type="composition",
    component_id="DynamicToolPipeline",
    version="1.0.0"
)
class DynamicToolPipeline(ToolPipeline):
    """A pipeline that can be modified at runtime based on the execution data.
    
    This class allows for dynamic tool selection and execution paths based on
    the data flowing through the pipeline.
    """

    def __init__(
        self, 
        name: str, 
        description: str, 
        decision_logic: Callable[[Dict[str, Any]], List[ITool]]
    ):
        """Initialize the dynamic tool pipeline.

        Args:
            name: The name of the pipeline
            description: A description of what the pipeline does
            decision_logic: A function that decides which tools to use based on the current data
        """
        super().__init__(name, description)
        self.decision_logic = decision_logic
        self.dynamic_tools_used: List[str] = []

    async def execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with dynamic tool selection.

        Args:
            initial_data: The initial data to pass to the pipeline

        Returns:
            The final data after executing the dynamically selected tools
            
        Raises:
            CompositionError: If decision logic or tool execution fails
        """
        self.dynamic_tools_used = []
        current_data = initial_data.copy()

        # First run the predefined nodes
        try:
            current_data = await super().execute(current_data)
        except Exception as e:
            logger.error(f"Error executing static nodes in dynamic pipeline {self.name}: {str(e)}")
            raise

        # Then decide which additional tools to run
        try:
            tools = self.decision_logic(current_data)
            self.dynamic_tools_used = [tool.name for tool in tools]

            if not tools:
                logger.info(f"Dynamic pipeline '{self.name}' decision logic returned no tools to run")
                return current_data

            # Run the dynamically selected tools
            for i, tool in enumerate(tools):
                try:
                    node = ToolNode(tool)
                    current_data = await node.execute(current_data)
                    self.last_execution_stats.append(node.stats.to_dict())
                except Exception as e:
                    logger.error(f"Error executing dynamic tool {tool.name} ({i+1}/{len(tools)}) in pipeline {self.name}: {str(e)}")
                    raise ToolExecutionError(tool.name, str(e))

            return current_data
            
        except Exception as e:
            logger.error(f"Error in dynamic pipeline {self.name}: {str(e)}")
            raise CompositionError(f"Dynamic pipeline '{self.name}' failed: {str(e)}") from e
            
    def get_dynamic_tools_used(self) -> List[str]:
        """Get the names of dynamically selected tools from the last execution.
        
        Returns:
            List of tool names that were dynamically selected
        """
        return self.dynamic_tools_used