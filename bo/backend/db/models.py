from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

# Association table for many-to-many relationship between users and teams
user_team = Table(
    "user_team",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.id"), primary_key=True),
)

class User(Base):
    """User database model."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user")  # admin, manager, user
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    teams = relationship("Team", secondary=user_team, back_populates="members")
    created_agents = relationship("Agent", back_populates="creator")
    created_tools = relationship("Tool", back_populates="creator")
    created_workflows = relationship("Workflow", back_populates="creator")

class Team(Base):
    """Team database model."""
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    manager_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    manager = relationship("User", foreign_keys=[manager_id])
    members = relationship("User", secondary=user_team, back_populates="teams")
    agents = relationship("Agent", back_populates="team")
    tools = relationship("Tool", back_populates="team")
    workflows = relationship("Workflow", back_populates="team")

class Agent(Base):
    """Agent database model."""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    agent_type = Column(String)  # ReactiveAgent, MemoryEnhancedAgent, etc.
    configuration = Column(String)  # JSON configuration
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    
    # Relationships
    creator = relationship("User", back_populates="created_agents")
    team = relationship("Team", back_populates="agents")
    tools = relationship("AgentTool", back_populates="agent")
    workflows = relationship("WorkflowAgent", back_populates="agent")

class Tool(Base):
    """Tool database model."""
    __tablename__ = "tools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    tool_type = Column(String)  # BaseTool, EnhancedTool, ComposableTool, etc.
    configuration = Column(String)  # JSON configuration
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    
    # Relationships
    creator = relationship("User", back_populates="created_tools")
    team = relationship("Team", back_populates="tools")
    agents = relationship("AgentTool", back_populates="tool")
    workflows = relationship("WorkflowTool", back_populates="tool")

class AgentTool(Base):
    """Association table for many-to-many relationship between agents and tools."""
    __tablename__ = "agent_tools"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    tool_id = Column(Integer, ForeignKey("tools.id"))
    configuration = Column(String, nullable=True)  # JSON configuration for tool usage
    
    # Relationships
    agent = relationship("Agent", back_populates="tools")
    tool = relationship("Tool", back_populates="agents")

class Workflow(Base):
    """Workflow database model."""
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    workflow_type = Column(String)  # ToolPipeline, ParallelToolPipeline, etc.
    configuration = Column(String)  # JSON configuration
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    
    # Relationships
    creator = relationship("User", back_populates="created_workflows")
    team = relationship("Team", back_populates="workflows")
    agents = relationship("WorkflowAgent", back_populates="workflow")
    tools = relationship("WorkflowTool", back_populates="workflow")

class WorkflowAgent(Base):
    """Association table for many-to-many relationship between workflows and agents."""
    __tablename__ = "workflow_agents"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    position = Column(Integer)  # Position in the workflow
    configuration = Column(String, nullable=True)  # JSON configuration
    
    # Relationships
    workflow = relationship("Workflow", back_populates="agents")
    agent = relationship("Agent", back_populates="workflows")

class WorkflowTool(Base):
    """Association table for many-to-many relationship between workflows and tools."""
    __tablename__ = "workflow_tools"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"))
    tool_id = Column(Integer, ForeignKey("tools.id"))
    position = Column(Integer)  # Position in the workflow
    configuration = Column(String, nullable=True)  # JSON configuration
    
    # Relationships
    workflow = relationship("Workflow", back_populates="tools")
    tool = relationship("Tool", back_populates="workflows")

class LLMConnection(Base):
    """LLM Connection database model."""
    __tablename__ = "llm_connections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    provider = Column(String)  # OpenAI, Anthropic, etc.
    model = Column(String)  # gpt-4, claude-2, etc.
    api_key = Column(String)
    configuration = Column(String, nullable=True)  # JSON configuration
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    creator = relationship("User")


# Training-related models
class TrainingDataset(Base):
    """Dataset database model for training."""
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    format = Column(String)  # csv, json, etc.
    size = Column(Integer)  # Size in bytes
    dataset_metadata = Column(String, nullable=True)  # JSON metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    creator = relationship("User", backref="datasets")


class TrainingSession(Base):
    """Training Session database model."""
    __tablename__ = "training_sessions"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    dataset_id = Column(String, ForeignKey("datasets.id"))
    config = Column(String)  # JSON configuration
    metrics = Column(String, nullable=True)  # JSON metrics
    status = Column(String)  # idle, running, completed, failed, stopped
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    agent = relationship("Agent", backref="training_sessions")
    dataset = relationship("TrainingDataset", backref="training_sessions")


class TrainingModel(Base):
    """ML Model database model for training."""
    __tablename__ = "models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    training_session_id = Column(String, ForeignKey("training_sessions.id"), nullable=True)
    config = Column(String)  # JSON configuration
    metrics = Column(String, nullable=True)  # JSON metrics
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", backref="models")
    training_session = relationship("TrainingSession", backref="models")
