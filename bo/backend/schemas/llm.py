"""LLM connection schemas for the Agentor BackOffice API."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import re

class LLMConnectionBase(BaseModel):
    """Base LLM connection model."""
    name: str
    provider: str  # OpenAI, Anthropic, etc.
    model: str  # gpt-4, claude-2, etc.
    api_key: str
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

    @validator('api_key')
    def api_key_not_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v

class LLMConnectionCreate(LLMConnectionBase):
    """LLM connection creation model."""
    pass

class LLMConnectionUpdate(BaseModel):
    """LLM connection update model."""
    name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

    @validator('api_key')
    def api_key_not_empty(cls, v):
        if v is not None and not v:
            raise ValueError('API key cannot be empty')
        return v

class LLMConnection(LLMConnectionBase):
    """LLM connection model with ID."""
    id: int
    creator_id: int
    created_at: datetime
    updated_at: datetime

    # Hide the API key in responses
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "name": "OpenAI GPT-4",
                "provider": "OpenAI",
                "model": "gpt-4",
                "api_key": "sk-***",
                "configuration": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "is_active": True,
                "creator_id": 1,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }

class LLMConnectionWithStats(LLMConnection):
    """LLM connection model with statistics."""
    stats: Dict[str, Any] = Field(default_factory=dict)

class LLMModel(BaseModel):
    """LLM model information."""
    id: str
    name: str
    provider: str
    capabilities: List[str] = Field(default_factory=list)
    max_tokens: int
    pricing: Dict[str, float] = Field(default_factory=dict)

class LLMProvider(BaseModel):
    """LLM provider information."""
    id: str
    name: str
    models: List[LLMModel] = Field(default_factory=list)
    documentation_url: Optional[str] = None
