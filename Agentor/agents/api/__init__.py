"""
API module for the Agentor framework.

This module provides API functionality for the Agentor framework,
including a FastAPI server for exposing tools as API endpoints
with OpenAPI/Swagger UI documentation.
"""

from agentor.agents.api.server import (
    ToolAPIServer,
    ToolAPISettings,
    create_tool_api_server,
    get_settings,
    validate_api_key
)
from agentor.agents.api.openapi import (
    tool_to_openapi_path,
    tools_to_openapi_paths,
    generate_openapi_schema,
    save_openapi_schema,
    load_openapi_schema
)
from agentor.agents.api.cli import main as cli_main

__all__ = [
    # Server
    "ToolAPIServer",
    "ToolAPISettings",
    "create_tool_api_server",
    "get_settings",
    "validate_api_key",

    # OpenAPI
    "tool_to_openapi_path",
    "tools_to_openapi_paths",
    "generate_openapi_schema",
    "save_openapi_schema",
    "load_openapi_schema",

    # CLI
    "cli_main"
]
