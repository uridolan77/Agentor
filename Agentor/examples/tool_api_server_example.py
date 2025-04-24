"""
Example demonstrating the Tool API Server in Agentor.

This example shows how to use the Tool API Server to expose tools as API endpoints
with OpenAPI/Swagger UI documentation.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from pydantic import Field

from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema
from agentor.agents.enhanced_tools import EnhancedTool, EnhancedToolRegistry, ToolResult
from agentor.agents.api import (
    ToolAPIServer,
    ToolAPISettings,
    generate_openapi_schema,
    save_openapi_schema
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define a custom tool with Pydantic schemas
class TranslateToolInput(ToolInputSchema):
    """Input schema for the translate tool."""
    text: str = Field(..., description="The text to translate")
    source_language: str = Field("auto", description="The source language (or 'auto' for auto-detection)")
    target_language: str = Field(..., description="The target language")


class TranslateToolOutput(ToolOutputSchema):
    """Output schema for the translate tool."""
    original_text: str = Field(..., description="The original text")
    translated_text: str = Field(..., description="The translated text")
    source_language: str = Field(..., description="The detected or specified source language")
    target_language: str = Field(..., description="The target language")


class TranslateTool(EnhancedTool):
    """Tool for translating text between languages."""

    def __init__(self):
        """Initialize the translate tool."""
        super().__init__(
            name="translate",
            description="Translate text between languages",
            input_schema=TranslateToolInput,
            output_schema=TranslateToolOutput
        )

    async def run(self, text: str, source_language: str = "auto", target_language: str = "en") -> ToolResult:
        """Translate text between languages.

        Args:
            text: The text to translate
            source_language: The source language (or 'auto' for auto-detection)
            target_language: The target language

        Returns:
            The translated text
        """
        # Validate input using the schema
        input_data = TranslateToolInput(
            text=text,
            source_language=source_language,
            target_language=target_language
        )

        # In a real implementation, this would call a translation API
        # For now, we'll just return some mock data
        if input_data.source_language == "auto":
            # Pretend we detected the language
            detected_language = "en"
        else:
            detected_language = input_data.source_language

        # Mock translation (just append the target language)
        if detected_language == input_data.target_language:
            translated_text = input_data.text
        else:
            translated_text = f"{input_data.text} [{input_data.target_language}]"

        output_data = TranslateToolOutput(
            original_text=input_data.text,
            translated_text=translated_text,
            source_language=detected_language,
            target_language=input_data.target_language
        )

        return ToolResult(
            success=True,
            data=output_data.dict()
        )


# Define another custom tool with Pydantic schemas
class ImageGenerationToolInput(ToolInputSchema):
    """Input schema for the image generation tool."""
    prompt: str = Field(..., description="The text prompt for image generation")
    width: int = Field(512, description="The width of the image in pixels", ge=64, le=1024)
    height: int = Field(512, description="The height of the image in pixels", ge=64, le=1024)
    num_images: int = Field(1, description="The number of images to generate", ge=1, le=4)


class ImageGenerationToolOutput(ToolOutputSchema):
    """Output schema for the image generation tool."""
    prompt: str = Field(..., description="The prompt used for generation")
    images: List[str] = Field(..., description="List of image URLs or base64-encoded images")
    width: int = Field(..., description="The width of the generated images")
    height: int = Field(..., description="The height of the generated images")


class ImageGenerationTool(EnhancedTool):
    """Tool for generating images from text prompts."""

    def __init__(self):
        """Initialize the image generation tool."""
        super().__init__(
            name="image_generation",
            description="Generate images from text prompts",
            input_schema=ImageGenerationToolInput,
            output_schema=ImageGenerationToolOutput
        )

    async def run(self, prompt: str, width: int = 512, height: int = 512, num_images: int = 1) -> ToolResult:
        """Generate images from a text prompt.

        Args:
            prompt: The text prompt for image generation
            width: The width of the image in pixels
            height: The height of the image in pixels
            num_images: The number of images to generate

        Returns:
            The generated images
        """
        # Validate input using the schema
        input_data = ImageGenerationToolInput(
            prompt=prompt,
            width=width,
            height=height,
            num_images=num_images
        )

        # In a real implementation, this would call an image generation API
        # For now, we'll just return some mock data
        images = [
            f"https://example.com/image_{i}.jpg" for i in range(input_data.num_images)
        ]

        output_data = ImageGenerationToolOutput(
            prompt=input_data.prompt,
            images=images,
            width=input_data.width,
            height=input_data.height
        )

        return ToolResult(
            success=True,
            data=output_data.dict()
        )


def main():
    """Run the Tool API Server example."""
    # Create a tool registry
    registry = EnhancedToolRegistry()

    # Register our custom tools
    registry.register_tool(TranslateTool())
    registry.register_tool(ImageGenerationTool())

    # Discover tools from the agentor package
    registry.discover_tools()

    # Create API server settings
    settings = ToolAPISettings(
        api_key=os.environ.get("TOOL_API_KEY"),  # Set this environment variable to enable API key authentication
        allowed_origins=["*"],
        debug=True,
        title="Agentor Tool API",
        description="API for accessing Agentor tools with OpenAPI/Swagger UI documentation",
        version="0.1.0"
    )

    # Create the API server
    server = ToolAPIServer(registry, settings)

    # Generate and save OpenAPI schema (optional)
    schema = generate_openapi_schema(
        tools=registry.get_tools(),
        title=settings.title,
        description=settings.description,
        version=settings.version,
        servers=[{"url": "http://localhost:8000", "description": "Local server"}],
        security_schemes={
            "APIKeyHeader": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-KEY"
            }
        } if settings.api_key else None
    )
    
    # Save the schema to a file (optional)
    os.makedirs("docs", exist_ok=True)
    save_openapi_schema(schema, "docs/openapi.json")

    # Print information about the server
    logger.info(f"Starting Tool API Server with {len(registry.get_tools())} tools")
    logger.info(f"Swagger UI: http://localhost:8000{settings.docs_url}")
    logger.info(f"ReDoc: http://localhost:8000{settings.redoc_url}")
    logger.info(f"OpenAPI Schema: http://localhost:8000{settings.openapi_url}")

    # Run the server
    server.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
