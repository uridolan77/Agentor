"""
Agents module for the Agentor framework.

This module provides various agent implementations, including:
- Base agent class
- Enhanced agent with standardized interfaces
- Learning agents
- Adapters for using agent components with the standardized interfaces
"""

from agentor.agents.enhanced_base import Agent, EnhancedAgent
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.agents.rule_based import RuleBasedAgent, rule
from agentor.agents.utility_based import UtilityBasedAgent, utility
from agentor.agents.reactive import ReactiveAgent, behavior
from agentor.agents.memory_enhanced_agent import MemoryEnhancedAgent, MemoryEntry
from agentor.agents.tools.recommender import ToolRecommender
from agentor.agents.tools.error_handling import (
    ErrorHandlingToolRegistry,
    ErrorHandlingTool,
    ToolErrorManager
)
from agentor.agents.learning import (
    EnhancedDeepQLearningAgent, EnhancedPPOAgent,
    MultiAgentRLAgent, IMPALAAgent, ModelBasedRLAgent, OfflineRLAgent,
    StableBaselinesAgent, StableBaselinesFactory
)
from agentor.agents.adapters import (
    AgentAdapter,
    DeepQLearningAgentAdapter,
    PPOAgentAdapter,
    SemanticRouterAdapter,
    RuleBasedRouterAdapter,
    HierarchicalRouterAdapter,
    BaseAgentPlugin,
    DeepQLearningAgentPlugin,
    PPOAgentPlugin,
    SemanticRouterPlugin,
    RuleBasedRouterPlugin,
    HierarchicalRouterPlugin
)

__all__ = [
    # Base agent classes
    'Agent',
    'AgentInput',
    'AgentOutput',
    'EnhancedAgent',

    # Rule-based, utility-based, reactive, and memory-enhanced agents
    'RuleBasedAgent',
    'rule',
    'UtilityBasedAgent',
    'utility',
    'ReactiveAgent',
    'behavior',
    'MemoryEnhancedAgent',
    'MemoryEntry',

    # Learning agents
    'EnhancedDeepQLearningAgent',
    'EnhancedPPOAgent',

    # Enhanced learning mechanisms
    'MultiAgentRLAgent',
    'IMPALAAgent',
    'ModelBasedRLAgent',
    'OfflineRLAgent',
    'StableBaselinesAgent',
    'StableBaselinesFactory',

    # Agent adapters
    'AgentAdapter',
    'DeepQLearningAgentAdapter',
    'PPOAgentAdapter',
    'SemanticRouterAdapter',
    'RuleBasedRouterAdapter',
    'HierarchicalRouterAdapter',

    # Agent plugins
    'BaseAgentPlugin',
    'DeepQLearningAgentPlugin',
    'PPOAgentPlugin',
    'SemanticRouterPlugin',
    'RuleBasedRouterPlugin',
    'HierarchicalRouterPlugin',

    # Enhanced tool components
    'ToolRecommender',
    'ErrorHandlingToolRegistry',
    'ErrorHandlingTool',
    'ToolErrorManager',
]