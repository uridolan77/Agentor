"""
Command-line interface for the Tool API Server.

This module provides a command-line interface for starting the Tool API Server
and generating OpenAPI documentation.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, List, Optional

from agentor.agents.enhanced_tools import EnhancedToolRegistry, get_tool_registry
from agentor.agents.api.server import ToolAPIServer, ToolAPISettings
from agentor.agents.api.openapi import generate_openapi_schema, save_openapi_schema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Tool API Server")
    
    # Server options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    parser.add_argument("--allowed-origins", type=str, default="*", help="Comma-separated list of allowed origins")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # OpenAPI options
    parser.add_argument("--title", type=str, default="Agentor Tool API", help="API title")
    parser.add_argument("--description", type=str, default="API for accessing Agentor tools", help="API description")
    parser.add_argument("--version", type=str, default="0.1.0", help="API version")
    parser.add_argument("--openapi-url", type=str, default="/openapi.json", help="OpenAPI URL")
    parser.add_argument("--docs-url", type=str, default="/docs", help="Swagger UI URL")
    parser.add_argument("--redoc-url", type=str, default="/redoc", help="ReDoc URL")
    
    # Tool discovery options
    parser.add_argument("--discover", action="store_true", help="Discover tools")
    parser.add_argument("--package", type=str, default="agentor", help="Package to discover tools from")
    parser.add_argument("--entry-points", action="store_true", help="Discover tools from entry points")
    parser.add_argument("--entry-point-group", type=str, default="agentor.tools", help="Entry point group to discover tools from")
    
    # OpenAPI schema generation options
    parser.add_argument("--generate-schema", action="store_true", help="Generate OpenAPI schema")
    parser.add_argument("--schema-file", type=str, default="openapi.json", help="File to save OpenAPI schema to")
    
    # Server mode
    parser.add_argument("--schema-only", action="store_true", help="Generate schema only, don't start server")
    
    return parser.parse_args()


def main():
    """Run the Tool API Server CLI."""
    args = parse_args()
    
    # Create a tool registry
    registry = get_tool_registry()
    
    # Discover tools if requested
    if args.discover:
        registry.discover_tools(args.package)
    
    # Discover tools from entry points if requested
    if args.entry_points:
        registry.discover_tools_from_entry_points(args.entry_point_group)
    
    # Print information about the tools
    tools = registry.get_tools()
    logger.info(f"Found {len(tools)} tools:")
    for tool_name in tools:
        logger.info(f"  - {tool_name}")
    
    # Generate OpenAPI schema if requested
    if args.generate_schema:
        schema = registry.generate_openapi_schema(
            title=args.title,
            description=args.description,
            version=args.version
        )
        
        # Save the schema to a file
        os.makedirs(os.path.dirname(os.path.abspath(args.schema_file)), exist_ok=True)
        registry.save_openapi_schema(
            args.schema_file,
            title=args.title,
            description=args.description,
            version=args.version
        )
        
        logger.info(f"Generated OpenAPI schema and saved to {args.schema_file}")
    
    # Exit if schema-only mode
    if args.schema_only:
        return
    
    # Create API server settings
    settings = ToolAPISettings(
        api_key=args.api_key,
        allowed_origins=args.allowed_origins.split(","),
        debug=args.debug,
        title=args.title,
        description=args.description,
        version=args.version,
        openapi_url=args.openapi_url,
        docs_url=args.docs_url,
        redoc_url=args.redoc_url
    )
    
    # Create the API server
    server = registry.create_api_server(settings)
    
    # Print information about the server
    logger.info(f"Starting Tool API Server with {len(tools)} tools")
    logger.info(f"Swagger UI: http://{args.host}:{args.port}{args.docs_url}")
    logger.info(f"ReDoc: http://{args.host}:{args.port}{args.redoc_url}")
    logger.info(f"OpenAPI Schema: http://{args.host}:{args.port}{args.openapi_url}")
    
    # Run the server
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
