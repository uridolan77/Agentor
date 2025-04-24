"""Workflow schemas for the Agentor BackOffice API."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class WorkflowBase(BaseModel):
    """Base workflow model."""
    name: str
    description: Optional[str] = None
    workflow_type: str  # ToolPipeline, ParallelToolPipeline, etc.
    configuration: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True

class WorkflowCreate(WorkflowBase):
    """Workflow creation model."""
    team_id: Optional[int] = None

class WorkflowUpdate(BaseModel):
    """Workflow update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_type: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    team_id: Optional[int] = None

class Workflow(WorkflowBase):
    """Workflow model with ID."""
    id: int
    creator_id: int
    team_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class WorkflowWithAgents(Workflow):
    """Workflow model with agents."""
    agents: List[Dict[str, Any]] = Field(default_factory=list)

class WorkflowWithTools(Workflow):
    """Workflow model with tools."""
    tools: List[Dict[str, Any]] = Field(default_factory=list)

class WorkflowWithStats(Workflow):
    """Workflow model with statistics."""
    stats: Dict[str, Any] = Field(default_factory=dict)

class WorkflowNode(BaseModel):
    """Workflow node model."""
    id: int
    node_type: str  # agent, tool
    position: Dict[str, float]  # x, y coordinates
    configuration: Dict[str, Any] = Field(default_factory=dict)

class WorkflowEdge(BaseModel):
    """Workflow edge model."""
    id: int
    source_id: int
    target_id: int
    configuration: Dict[str, Any] = Field(default_factory=dict)

class WorkflowExecution(BaseModel):
    """Workflow execution model."""
    id: int
    workflow_id: int
    status: str  # pending, running, completed, failed
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
