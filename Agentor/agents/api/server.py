"""
Tool API Server for the Agentor framework.

This module provides a FastAPI application for exposing tools as API endpoints
with OpenAPI/Swagger UI documentation.
"""

import logging
import inspect
import asyncio
from typing import Dict, Any, List, Optional, Type, Callable, Union, get_type_hints
from contextlib import asynccontextmanager
from pydantic import BaseModel, create_model, Field

from fastapi import FastAPI, Depends, HTTPException, Security, Body, Request, Response
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

from agentor.core.interfaces.tool import ITool, ToolResult, IToolRegistry
from agentor.agents.enhanced_tools import EnhancedTool, EnhancedToolRegistry
from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema

logger = logging.getLogger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


class ToolAPISettings(BaseModel):
    """Settings for the Tool API Server."""
    
    api_key: Optional[str] = None
    allowed_origins: List[str] = ["*"]
    debug: bool = False
    title: str = "Agentor Tool API"
    description: str = "API for accessing Agentor tools"
    version: str = "0.1.0"
    openapi_url: str = "/openapi.json"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


def get_settings() -> ToolAPISettings:
    """Get the Tool API settings.
    
    This would typically load from environment variables or a config file.
    For this example, we'll just return a hardcoded Settings object.
    
    Returns:
        The Tool API settings
    """
    return ToolAPISettings()


async def validate_api_key(
    api_key: Optional[str] = Security(api_key_header),
    settings: ToolAPISettings = Depends(get_settings)
):
    """Validate the API key.
    
    Args:
        api_key: The API key from the request header
        settings: The application settings
        
    Raises:
        HTTPException: If the API key is invalid
        
    Returns:
        The validated API key
    """
    # If no API key is configured, allow all requests
    if not settings.api_key:
        return None
    
    # If an API key is configured, validate it
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    
    return api_key


class ToolAPIServer:
    """API server for exposing tools as API endpoints."""
    
    def __init__(
        self,
        tool_registry: Optional[IToolRegistry] = None,
        settings: Optional[ToolAPISettings] = None
    ):
        """Initialize the Tool API server.
        
        Args:
            tool_registry: The tool registry to use
            settings: The server settings
        """
        self.tool_registry = tool_registry or EnhancedToolRegistry()
        self.settings = settings or get_settings()
        self.app = self._create_app()
        self._register_tools()
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application.
        
        Returns:
            The FastAPI application
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Configure the application on startup and shutdown."""
            # Startup
            logger.info("Starting Tool API server")
            
            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.settings.allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            yield
            
            # Shutdown
            logger.info("Shutting down Tool API server")
        
        # Create the app with lifespan
        app = FastAPI(
            title=self.settings.title,
            description=self.settings.description,
            version=self.settings.version,
            openapi_url=self.settings.openapi_url,
            docs_url=self.settings.docs_url,
            redoc_url=self.settings.redoc_url,
            lifespan=lifespan
        )
        
        # Add health check endpoint
        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok"}
        
        # Add custom OpenAPI schema
        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema
            
            openapi_schema = get_openapi(
                title=self.settings.title,
                version=self.settings.version,
                description=self.settings.description,
                routes=app.routes,
            )
            
            # Add custom components and security schemes
            if self.settings.api_key:
                openapi_schema["components"] = openapi_schema.get("components", {})
                openapi_schema["components"]["securitySchemes"] = {
                    "APIKeyHeader": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-KEY"
                    }
                }
                openapi_schema["security"] = [{"APIKeyHeader": []}]
            
            app.openapi_schema = openapi_schema
            return app.openapi_schema
        
        app.openapi = custom_openapi
        
        return app
    
    def _register_tools(self) -> None:
        """Register all tools as API endpoints."""
        # Get all tools from the registry
        tools = self.tool_registry.get_tools()
        
        # Register each tool as an API endpoint
        for tool_name, tool in tools.items():
            self._register_tool(tool)
    
    def _register_tool(self, tool: ITool) -> None:
        """Register a tool as an API endpoint.
        
        Args:
            tool: The tool to register
        """
        # Skip if not an EnhancedTool
        if not isinstance(tool, EnhancedTool):
            logger.warning(f"Skipping non-EnhancedTool: {tool.name}")
            return
        
        # Get the tool schema
        schema = tool.get_schema()
        
        # Create a Pydantic model for the request body
        if tool.input_schema:
            # Use the existing input schema
            request_model = tool.input_schema
        else:
            # Create a dynamic model from the schema
            fields = {}
            for name, prop in schema.get("properties", {}).items():
                field_type = self._get_field_type(prop)
                default = ... if name in schema.get("required", []) else None
                description = prop.get("description", f"Parameter: {name}")
                fields[name] = (field_type, Field(default, description=description))
            
            request_model = create_model(
                f"{tool.name.title()}Request",
                __base__=ToolInputSchema,
                **fields
            )
        
        # Create a Pydantic model for the response body
        if tool.output_schema:
            # Use the existing output schema
            response_model = tool.output_schema
        else:
            # Create a simple response model
            response_model = create_model(
                f"{tool.name.title()}Response",
                __base__=ToolOutputSchema,
                result=(Dict[str, Any], Field(..., description="The result of the tool execution"))
            )
        
        # Create the endpoint handler
        async def endpoint_handler(
            request: request_model,  # type: ignore
            api_key: Optional[str] = Depends(validate_api_key)
        ) -> response_model:  # type: ignore
            """Handle the API request for the tool.
            
            Args:
                request: The request body
                api_key: The validated API key
                
            Returns:
                The tool response
            """
            try:
                # Run the tool with the request parameters
                result = await tool.run(**request.dict())
                
                # Check if the tool execution was successful
                if not result.success:
                    raise HTTPException(
                        status_code=500,
                        detail=result.error or "Tool execution failed"
                    )
                
                # Return the result
                if issubclass(response_model, ToolOutputSchema):
                    return response_model(**result.data)
                else:
                    return {"result": result.data}
            
            except Exception as e:
                logger.exception(f"Error executing tool {tool.name}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error executing tool: {str(e)}"
                )
        
        # Set the endpoint metadata
        endpoint_handler.__name__ = f"execute_{tool.name}"
        endpoint_handler.__doc__ = tool.description
        
        # Register the endpoint
        self.app.post(
            f"/tools/{tool.name}",
            response_model=response_model,
            summary=tool.description,
            tags=["Tools"]
        )(endpoint_handler)
        
        logger.info(f"Registered tool as API endpoint: {tool.name}")
    
    def _get_field_type(self, prop: Dict[str, Any]) -> Type:
        """Get the Python type for a JSON Schema property.
        
        Args:
            prop: The JSON Schema property
            
        Returns:
            The Python type
        """
        prop_type = prop.get("type", "string")
        
        if prop_type == "string":
            return str
        elif prop_type == "integer":
            return int
        elif prop_type == "number":
            return float
        elif prop_type == "boolean":
            return bool
        elif prop_type == "array":
            items = prop.get("items", {})
            item_type = self._get_field_type(items)
            return List[item_type]
        elif prop_type == "object":
            return Dict[str, Any]
        else:
            return Any
    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the API server.
        
        Args:
            host: The host to bind to
            port: The port to bind to
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application.
        
        Returns:
            The FastAPI application
        """
        return self.app


def create_tool_api_server(
    tool_registry: Optional[IToolRegistry] = None,
    settings: Optional[ToolAPISettings] = None
) -> ToolAPIServer:
    """Create a Tool API server.
    
    Args:
        tool_registry: The tool registry to use
        settings: The server settings
        
    Returns:
        The Tool API server
    """
    return ToolAPIServer(tool_registry, settings)
