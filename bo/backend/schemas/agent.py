"""Agent schemas for the Agentor BackOffice API."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AgentBase(BaseModel):
    """Base agent model."""
    name: str
    description: Optional[str] = None
    agent_type: str  # ReactiveAgent, MemoryEnhancedAgent, etc.
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class AgentCreate(AgentBase):
    """Agent creation model."""
    team_id: Optional[int] = None

class AgentUpdate(BaseModel):
    """Agent update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    agent_type: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    team_id: Optional[int] = None

class Agent(AgentBase):
    """Agent model with ID."""
    id: int
    creator_id: int
    team_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class AgentWithTools(Agent):
    """Agent model with tools."""
    tools: List[Dict[str, Any]] = Field(default_factory=list)

class AgentWithStats(Agent):
    """Agent model with statistics."""
    stats: Dict[str, Any] = Field(default_factory=dict)
