"""
Pydantic schemas for tools in the Agentor framework.

This module provides Pydantic models for tool input and output schemas,
which provide better structure, validation, and documentation compared to
plain dictionaries. These schemas ensure consistent validation and documentation
of tool inputs and outputs across the framework.
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, Generic, get_type_hints, get_origin, get_args, Union, Callable
from pydantic import BaseModel, create_model, Field, validator, ValidationError
import inspect
import logging
import json
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)
U = TypeVar('U', bound=BaseModel)


class SchemaValidationError(Exception):
    """Exception raised for schema validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the error.
        
        Args:
            message: The error message
            details: Additional details about the error
        """
        self.details = details or {}
        super().__init__(message)


class ToolSchemaError(Exception):
    """Exception raised for errors in tool schema creation or usage."""
    pass


class ToolInputSchema(BaseModel):
    """Base class for tool input schemas.
    
    This class should be extended by all tool input schemas to ensure
    consistent validation and documentation. It enforces strict validation
    by default, rejecting extra fields.
    
    Examples:
        >>> class MyToolInput(ToolInputSchema):
        ...     name: str
        ...     count: int = 1
        >>> input_data = MyToolInput(name="test", count=5)
        >>> input_data.dict()
        {'name': 'test', 'count': 5}
    """
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Forbid extra fields to ensure strict validation
        validate_assignment = True  # Validate attribute assignment
        json_encoders = {
            # Custom JSON encoders can be added here
        }


class ToolOutputSchema(BaseModel):
    """Base class for tool output schemas.
    
    This class should be extended by all tool output schemas to ensure
    consistent validation and documentation. It enforces strict validation
    by default, rejecting extra fields.
    
    Examples:
        >>> class MyToolOutput(ToolOutputSchema):
        ...     result: str
        ...     status: int
        >>> output_data = MyToolOutput(result="success", status=200)
        >>> output_data.dict()
        {'result': 'success', 'status': 200}
    """
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Forbid extra fields to ensure strict validation
        validate_assignment = True  # Validate attribute assignment
        json_encoders = {
            # Custom JSON encoders can be added here
        }


def create_input_schema_from_signature(func: Callable, schema_name: Optional[str] = None) -> Type[ToolInputSchema]:
    """Create a Pydantic model from a function signature.
    
    This function analyzes the signature of the provided function and creates
    a Pydantic model that can be used to validate inputs to the function.
    The model will have fields for each parameter in the function signature,
    with appropriate types and default values.
    
    Args:
        func: The function to create a model from
        schema_name: Optional name for the schema, defaults to DynamicToolInputSchema
        
    Returns:
        A Pydantic model class for input validation
        
    Raises:
        ToolSchemaError: If the function signature cannot be analyzed
    """
    try:
        # Generate a name for the schema if not provided
        if schema_name is None:
            schema_name = f"{func.__name__.title()}InputSchema"
        
        # Get the function signature
        sig = inspect.signature(func)
        
        # Create field definitions
        fields = {}
        annotations = {}
        
        for name, param in sig.parameters.items():
            # Skip 'self' parameter
            if name == 'self':
                continue
            
            # Get the annotation
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                annotation = Any
            
            # Get the default value
            default = param.default
            if default is inspect.Parameter.empty:
                default = ...  # Required field
            
            # Add the field
            fields[name] = (annotation, default)
            annotations[name] = annotation
        
        # Create the model
        model = create_model(
            schema_name,
            __base__=ToolInputSchema,
            **fields
        )
        
        # Add docstrings from the function
        if func.__doc__:
            model.__doc__ = func.__doc__
            
        # Extract parameter descriptions from docstring and add them as field descriptions
        param_descriptions = extract_input_schema_from_docstring(func)
        for param_name, description in param_descriptions.items():
            if param_name in model.__fields__:
                model.__fields__[param_name].field_info.description = description
        
        return model
    except Exception as e:
        error_msg = f"Failed to create input schema from function signature: {str(e)}"
        logger.error(error_msg)
        raise ToolSchemaError(error_msg) from e


def create_output_schema_from_result(result_type: Type, schema_name: Optional[str] = None) -> Type[ToolOutputSchema]:
    """Create a Pydantic model from a result type.
    
    This function creates a Pydantic model that can be used to validate outputs
    from a function. If the result type is a Pydantic model, it will be used;
    otherwise, a simple model with a 'result' field will be created.
    
    Args:
        result_type: The type to create a model from
        schema_name: Optional name for the schema
        
    Returns:
        A Pydantic model class for output validation
        
    Raises:
        ToolSchemaError: If the result type cannot be used to create a schema
    """
    try:
        # Generate a name for the schema if not provided
        if schema_name is None:
            if hasattr(result_type, "__name__"):
                schema_name = f"{result_type.__name__}OutputSchema"
            else:
                schema_name = "DynamicToolOutputSchema"
        
        # If the result type is already a Pydantic model, use it
        if inspect.isclass(result_type) and issubclass(result_type, BaseModel):
            # Create a new model that inherits from both the result type and ToolOutputSchema
            model = create_model(
                schema_name,
                __base__=(result_type, ToolOutputSchema),
            )
            return model
        
        # Otherwise, create a simple model with a 'result' field
        model = create_model(
            schema_name,
            __base__=ToolOutputSchema,
            result=(result_type, ...),
        )
        
        return model
    except Exception as e:
        error_msg = f"Failed to create output schema from result type: {str(e)}"
        logger.error(error_msg)
        raise ToolSchemaError(error_msg) from e


def model_to_json_schema(model: Type[BaseModel], include_examples: bool = True) -> Dict[str, Any]:
    """Convert a Pydantic model to a JSON schema.
    
    This function converts a Pydantic model to a JSON schema that can be used
    for documentation and validation. The schema includes type information,
    required fields, and descriptions.
    
    Args:
        model: The Pydantic model to convert
        include_examples: Whether to include example values in the schema
        
    Returns:
        A JSON schema as a dictionary
        
    Raises:
        ValueError: If the model is not a Pydantic model
    """
    if not inspect.isclass(model) or not issubclass(model, BaseModel):
        raise ValueError("Model must be a Pydantic model")
    
    try:
        schema = model.schema()
        
        # Remove Pydantic-specific fields
        if 'title' in schema:
            del schema['title']
        
        # Add examples if available and requested
        if include_examples:
            for field_name, field in model.__fields__.items():
                if hasattr(field.field_info, 'example') and field.field_info.example is not None:
                    if 'properties' not in schema:
                        schema['properties'] = {}
                    if field_name not in schema['properties']:
                        schema['properties'][field_name] = {}
                    schema['properties'][field_name]['example'] = field.field_info.example
        
        return schema
    except Exception as e:
        logger.error(f"Error converting model to JSON schema: {e}")
        # Return a simplified schema
        return {
            "type": "object",
            "properties": {
                field_name: {"type": "any"} 
                for field_name in model.__fields__
            },
            "required": [
                field_name for field_name, field in model.__fields__.items()
                if field.required
            ]
        }


def extract_input_schema_from_docstring(func: Callable) -> Dict[str, str]:
    """Extract parameter descriptions from a function's docstring.
    
    This function parses a function's docstring and extracts descriptions for
    each parameter. It supports both Google-style and ReStructuredText-style
    docstrings.
    
    Args:
        func: The function to extract parameter descriptions from
        
    Returns:
        A dictionary mapping parameter names to descriptions
    """
    if not func.__doc__:
        return {}
    
    # Parse the docstring
    docstring = func.__doc__
    
    # Extract parameter descriptions
    param_descriptions = {}
    in_args_section = False
    current_param = None
    
    # Check for Google-style docstrings first
    for line in docstring.split('\n'):
        line = line.strip()
        
        # Check if we're in the Args section
        if line.lower() == 'args:':
            in_args_section = True
            continue
        
        # Check if we've left the Args section
        if in_args_section and (not line or (line and line.endswith(':') and not line.startswith(' '))):
            in_args_section = False
            continue
        
        # Parse parameter descriptions
        if in_args_section:
            # Check if this is a new parameter
            if line and not line.startswith(' '):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    param_descriptions[param_name] = param_desc
                    current_param = param_name
            # Check if this is a continuation of the previous parameter
            elif current_param and line:
                param_descriptions[current_param] += ' ' + line
    
    # If no parameters were found, check for RST-style docstrings
    if not param_descriptions:
        in_params_section = False
        param_pattern = ":param "
        
        for line in docstring.split('\n'):
            line = line.strip()
            
            if line.startswith(param_pattern):
                # Extract parameter name and description
                param_text = line[len(param_pattern):]
                parts = param_text.split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    param_descriptions[param_name] = param_desc
            elif line.startswith(':') and ' ' in line:
                # Skip other RST directives
                continue
            elif any(param_name in line for param_name in param_descriptions.keys()):
                # Add continuation lines
                for param_name in param_descriptions.keys():
                    if param_name in line:
                        param_descriptions[param_name] += ' ' + line
    
    return param_descriptions


def enhance_schema_with_docstring(schema: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    """Enhance a JSON schema with descriptions from a function's docstring.
    
    This function adds parameter descriptions from a function's docstring to
    a JSON schema, making the schema more informative and useful for documentation.
    
    Args:
        schema: The JSON schema to enhance
        func: The function to extract parameter descriptions from
        
    Returns:
        The enhanced JSON schema
    """
    # Extract parameter descriptions from the docstring
    param_descriptions = extract_input_schema_from_docstring(func)
    
    # Add descriptions to the schema
    if 'properties' in schema:
        for param_name, param_schema in schema['properties'].items():
            if param_name in param_descriptions:
                param_schema['description'] = param_descriptions[param_name]
    
    return schema


def validate_schema_compatibility(source_schema: Dict[str, Any], target_schema: Dict[str, Any]) -> List[str]:
    """Validate that two schemas are compatible.
    
    This function checks if data conforming to the source schema can be used
    with the target schema. It checks for compatible types, required fields,
    and other constraints.
    
    Args:
        source_schema: The source schema
        target_schema: The target schema
        
    Returns:
        A list of validation errors, or an empty list if the schemas are compatible
    """
    errors = []
    
    # Check if source schema has all required fields from target schema
    target_required = target_schema.get('required', [])
    source_properties = source_schema.get('properties', {})
    
    for field_name in target_required:
        if field_name not in source_properties:
            errors.append(f"Required field '{field_name}' not found in source schema")
    
    # Check if field types are compatible
    target_properties = target_schema.get('properties', {})
    
    for field_name, target_prop in target_properties.items():
        if field_name in source_properties:
            source_prop = source_properties[field_name]
            
            # Check type compatibility
            source_type = source_prop.get('type')
            target_type = target_prop.get('type')
            
            if source_type and target_type and source_type != target_type:
                # Check if types are compatible
                if not _are_types_compatible(source_type, target_type):
                    errors.append(f"Field '{field_name}' has incompatible type: {source_type} vs {target_type}")
    
    return errors


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
        
    # Arrays can be compatible if target is array or object
    if source_type == 'array' and target_type in ('array', 'object'):
        return True
        
    # Objects can be compatible if target is object
    if source_type == 'object' and target_type == 'object':
        return True

    return False


def schema_decorator(input_schema: Optional[Type[ToolInputSchema]] = None, 
                    output_schema: Optional[Type[ToolOutputSchema]] = None):
    """Decorator for adding schema validation to a function.
    
    This decorator adds input and output schema validation to a function.
    It uses the provided schemas to validate inputs and outputs, or creates
    schemas from the function signature if none are provided.
    
    Args:
        input_schema: Optional input schema to use for validation
        output_schema: Optional output schema to use for validation
        
    Returns:
        A decorator function
        
    Examples:
        >>> @schema_decorator(MyInputSchema, MyOutputSchema)
        ... def my_function(name: str, count: int) -> str:
        ...     return f"{name} called {count} times"
    """
    def decorator(func):
        # Create input schema from function signature if not provided
        func_input_schema = input_schema
        if func_input_schema is None:
            func_input_schema = create_input_schema_from_signature(func)
            
        # Store schemas in the function for later use
        func._input_schema = func_input_schema
        func._output_schema = output_schema
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate input
            if func_input_schema:
                try:
                    # Handle 'self' parameter in methods
                    if args and len(args) > 0 and hasattr(args[0], func.__name__):
                        # This is a method call, skip the 'self' parameter
                        validated_input = func_input_schema(**kwargs)
                        validated_kwargs = validated_input.dict()
                        return func(*args, **validated_kwargs)
                    else:
                        # This is a regular function call
                        validated_input = func_input_schema(**kwargs)
                        validated_kwargs = validated_input.dict()
                        return func(*args, **validated_kwargs)
                except ValidationError as e:
                    raise SchemaValidationError(f"Input validation failed: {e}", e.errors())
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate output
            if output_schema and result is not None:
                try:
                    if isinstance(result, dict):
                        validated_output = output_schema(**result)
                        return validated_output.dict()
                    else:
                        validated_output = output_schema(result=result)
                        return validated_output.dict()
                except ValidationError as e:
                    raise SchemaValidationError(f"Output validation failed: {e}", e.errors())
            
            return result
        
        # Copy schemas to the wrapper for introspection
        wrapper._input_schema = func_input_schema
        wrapper._output_schema = output_schema
        
        return wrapper
    
    return decorator


def get_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Get the JSON schema for a function.
    
    This function extracts schema information from a function, either from
    explicitly attached schemas or by analyzing the function signature.
    
    Args:
        func: The function to get the schema for
        
    Returns:
        A JSON schema for the function
    """
    # Check if the function has schemas attached
    if hasattr(func, '_input_schema'):
        input_schema = func._input_schema
        schema = model_to_json_schema(input_schema)
        return enhance_schema_with_docstring(schema, func)
    
    # Otherwise, create a schema from the function signature
    input_schema = create_input_schema_from_signature(func)
    schema = model_to_json_schema(input_schema)
    return enhance_schema_with_docstring(schema, func)
