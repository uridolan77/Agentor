"""
Schema validators for LLM Gateway messages to ensure integrity and security.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List, Union

from pydantic import BaseModel, Field, ValidationError, validator

logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    """Raised when input validation fails."""
    pass


class MessageContent(BaseModel):
    """Content of a message, which can be text or structured data."""
    type: str = Field(default="text")
    text: Optional[str] = None
    parts: Optional[List[Dict[str, Any]]] = None

    @validator("text")
    def validate_text(cls, v, values):
        """Validate text content."""
        if values.get("type") == "text" and v is None:
            raise ValueError("Text content is required for text type messages")
        return v

    @validator("parts")
    def validate_parts(cls, v, values):
        """Validate structured content parts."""
        if values.get("type") == "structured" and (v is None or len(v) == 0):
            raise ValueError("Parts are required for structured type messages")
        return v


class Message(BaseModel):
    """A message in a conversation."""
    role: str = Field(..., description="Role of the message sender")
    content: Union[str, List[Dict[str, Any]], MessageContent] = Field(
        ..., description="Content of the message"
    )

    @validator("role")
    def validate_role(cls, v):
        """Validate message role."""
        allowed_roles = ["system", "user", "assistant", "tool", "function"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

    @validator("content")
    def validate_content(cls, v):
        """Convert string or list content to MessageContent."""
        if isinstance(v, str):
            return MessageContent(type="text", text=v)
        elif isinstance(v, list):
            return MessageContent(type="structured", parts=v)
        return v


class Tool(BaseModel):
    """A tool definition."""
    type: str = Field(..., description="Type of tool")
    function: Dict[str, Any] = Field(..., description="Function definition")

    @validator("type")
    def validate_type(cls, v):
        """Validate tool type."""
        if v != "function":
            raise ValueError("Currently only function tools are supported")
        return v

    @validator("function")
    def validate_function(cls, v):
        """Validate function definition."""
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Function definition must include {field}")
        return v


class LLMRequestSchema(BaseModel):
    """Schema for LLM request validation."""
    messages: List[Message] = Field(..., description="Messages in the conversation")
    model: str = Field(..., description="Model identifier")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter")
    stream: Optional[bool] = Field(None, description="Whether to stream the response")
    tools: Optional[List[Tool]] = Field(None, description="Available tools")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")

    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        """Validate max_tokens."""
        if v is not None and (v < 1 or v > 32768):
            raise ValueError("max_tokens must be between 1 and 32768")
        return v

    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature."""
        if v is not None and (v < 0 or v > 2.0):
            raise ValueError("temperature must be between 0 and 2.0")
        return v

    @validator("top_p")
    def validate_top_p(cls, v):
        """Validate top_p."""
        if v is not None and (v < 0 or v > 1.0):
            raise ValueError("top_p must be between 0 and 1.0")
        return v


def sanitize_string(content: str) -> str:
    """
    Sanitize string content to prevent injection attacks.
    
    Args:
        content: String content to sanitize
        
    Returns:
        Sanitized string content
    """
    # Remove control characters except for whitespace
    sanitized = "".join(c for c in content if ord(c) >= 32 or c in "\n\r\t")
    
    # Prevent common injection patterns
    # This is a basic implementation - in production, use a proper HTML sanitizer
    sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized


def sanitize_content(content: Any) -> Any:
    """
    Recursively sanitize content to prevent injection attacks.
    
    Args:
        content: Content to sanitize
        
    Returns:
        Sanitized content
    """
    if isinstance(content, str):
        return sanitize_string(content)
    elif isinstance(content, list):
        return [sanitize_content(item) for item in content]
    elif isinstance(content, dict):
        return {k: sanitize_content(v) for k, v in content.items()}
    else:
        return content


def validate_llm_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an LLM request against its schema.
    
    Args:
        request_data: Request data to validate
        
    Returns:
        Validated and sanitized request data
        
    Raises:
        InputValidationError: If validation fails
    """
    try:
        # First sanitize the input
        sanitized_data = sanitize_content(request_data)
        
        # Then validate against schema
        validated_data = LLMRequestSchema(**sanitized_data)
        
        # Return as dict
        return validated_data.dict(exclude_none=True)
    except ValidationError as e:
        logger.error(f"Request validation failed: {e}")
        raise InputValidationError(f"Invalid request format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise InputValidationError(f"Validation error: {str(e)}")
