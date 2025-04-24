"""Schema module for the Agentor BackOffice."""

from .agent import (
    AgentBase, 
    AgentCreate, 
    AgentUpdate, 
    Agent, 
    AgentWithTools, 
    AgentWithStats
)

from .tool import (
    ToolBase, 
    ToolCreate, 
    ToolUpdate, 
    Tool, 
    ToolWithAgents, 
    ToolWithStats, 
    ToolSchema
)

from .workflow import (
    WorkflowBase, 
    WorkflowCreate, 
    WorkflowUpdate, 
    Workflow, 
    WorkflowWithAgents, 
    WorkflowWithTools, 
    WorkflowWithStats,
    WorkflowNode,
    WorkflowEdge,
    WorkflowExecution
)

from .llm import (
    LLMConnectionBase, 
    LLMConnectionCreate, 
    LLMConnectionUpdate, 
    LLMConnection, 
    LLMConnectionWithStats,
    LLMModel,
    LLMProvider
)
