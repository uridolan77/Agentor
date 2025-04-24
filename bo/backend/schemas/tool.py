"""Tool schemas for the Agentor BackOffice API."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ToolBase(BaseModel):
    """Base tool model."""
    name: str
    description: Optional[str] = None
    tool_type: str  # BaseTool, EnhancedTool, ComposableTool, etc.
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class ToolCreate(ToolBase):
    """Tool creation model."""
    team_id: Optional[int] = None

class ToolUpdate(BaseModel):
    """Tool update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    tool_type: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    team_id: Optional[int] = None

class Tool(ToolBase):
    """Tool model with ID."""
    id: int
    creator_id: int
    team_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ToolWithAgents(Tool):
    """Tool model with agents."""
    agents: List[Dict[str, Any]] = Field(default_factory=list)

class ToolWithStats(Tool):
    """Tool model with statistics."""
    stats: Dict[str, Any] = Field(default_factory=dict)

class ToolSchema(BaseModel):
    """Tool schema model."""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
