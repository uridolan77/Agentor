"""
OpenAPI utilities for the Agentor framework.

This module provides utilities for converting tool schemas to OpenAPI schemas
and generating OpenAPI documentation from tool schemas.
"""

import logging
import inspect
import json
from typing import Dict, Any, List, Optional, Type, Union, get_type_hints
from pydantic import BaseModel, Field, create_model

from agentor.core.interfaces.tool import ITool, ToolResult
from agentor.agents.enhanced_tools import EnhancedTool
from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema, model_to_json_schema

logger = logging.getLogger(__name__)


def tool_to_openapi_path(tool: EnhancedTool) -> Dict[str, Any]:
    """Convert a tool to an OpenAPI path object.
    
    Args:
        tool: The tool to convert
        
    Returns:
        An OpenAPI path object
    """
    # Get the tool schema
    schema = tool.get_schema()
    
    # Create the request body schema
    request_body = {
        "content": {
            "application/json": {
                "schema": schema
            }
        },
        "required": True
    }
    
    # Create the response schema
    response_schema = {}
    if tool.output_schema:
        response_schema = model_to_json_schema(tool.output_schema)
    else:
        response_schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "description": "The result of the tool execution"
                }
            },
            "required": ["result"]
        }
    
    # Create the responses object
    responses = {
        "200": {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "schema": response_schema
                }
            }
        },
        "400": {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "description": "Error message"
                            }
                        },
                        "required": ["detail"]
                    }
                }
            }
        },
        "401": {
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "description": "Error message"
                            }
                        },
                        "required": ["detail"]
                    }
                }
            }
        },
        "500": {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "description": "Error message"
                            }
                        },
                        "required": ["detail"]
                    }
                }
            }
        }
    }
    
    # Create the path object
    path = {
        "post": {
            "summary": tool.description,
            "description": tool.description,
            "operationId": f"execute_{tool.name}",
            "tags": ["Tools"],
            "requestBody": request_body,
            "responses": responses
        }
    }
    
    return path


def tools_to_openapi_paths(tools: Dict[str, EnhancedTool]) -> Dict[str, Any]:
    """Convert a dictionary of tools to OpenAPI paths.
    
    Args:
        tools: A dictionary of tool names to tools
        
    Returns:
        A dictionary of OpenAPI paths
    """
    paths = {}
    
    for tool_name, tool in tools.items():
        if isinstance(tool, EnhancedTool):
            path = tool_to_openapi_path(tool)
            paths[f"/tools/{tool_name}"] = path
    
    return paths


def generate_openapi_schema(
    tools: Dict[str, EnhancedTool],
    title: str = "Agentor Tool API",
    description: str = "API for accessing Agentor tools",
    version: str = "0.1.0",
    servers: Optional[List[Dict[str, str]]] = None,
    security_schemes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate an OpenAPI schema from a dictionary of tools.
    
    Args:
        tools: A dictionary of tool names to tools
        title: The API title
        description: The API description
        version: The API version
        servers: A list of server objects
        security_schemes: A dictionary of security schemes
        
    Returns:
        An OpenAPI schema
    """
    # Create the paths
    paths = tools_to_openapi_paths(tools)
    
    # Add the health check endpoint
    paths["/health"] = {
        "get": {
            "summary": "Health check",
            "description": "Check if the API is running",
            "operationId": "health",
            "tags": ["System"],
            "responses": {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {
                                        "type": "string",
                                        "description": "The API status"
                                    }
                                },
                                "required": ["status"]
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Create the components
    components = {}
    
    # Add security schemes if provided
    if security_schemes:
        components["securitySchemes"] = security_schemes
    
    # Create the OpenAPI schema
    openapi_schema = {
        "openapi": "3.0.2",
        "info": {
            "title": title,
            "description": description,
            "version": version
        },
        "paths": paths
    }
    
    # Add servers if provided
    if servers:
        openapi_schema["servers"] = servers
    
    # Add components if not empty
    if components:
        openapi_schema["components"] = components
    
    # Add security if security schemes are provided
    if security_schemes:
        openapi_schema["security"] = [
            {scheme_name: []} for scheme_name in security_schemes.keys()
        ]
    
    return openapi_schema


def save_openapi_schema(
    schema: Dict[str, Any],
    file_path: str
) -> None:
    """Save an OpenAPI schema to a file.
    
    Args:
        schema: The OpenAPI schema
        file_path: The file path to save to
    """
    with open(file_path, "w") as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Saved OpenAPI schema to {file_path}")


def load_openapi_schema(file_path: str) -> Dict[str, Any]:
    """Load an OpenAPI schema from a file.
    
    Args:
        file_path: The file path to load from
        
    Returns:
        The OpenAPI schema
    """
    with open(file_path, "r") as f:
        schema = json.load(f)
    
    logger.info(f"Loaded OpenAPI schema from {file_path}")
    
    return schema
