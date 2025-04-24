"""
Learning agents for the Agentor framework.

This module provides enhanced learning agent implementations that use
the standardized interfaces, dependency injection system, and environment interface.
"""

from agentor.agents.learning.deep_q_learning import EnhancedDeepQLearningAgent
from agentor.agents.learning.ppo_agent import EnhancedPPOAgent
from agentor.agents.learning.multi_agent_rl import MultiAgentRLAgent
from agentor.agents.learning.impala import IMPALAAgent
from agentor.agents.learning.model_based_rl import ModelBasedRLAgent
from agentor.agents.learning.offline_rl import OfflineRLAgent
from agentor.agents.learning.stable_baselines_integration import StableBaselinesAgent, StableBaselinesFactory

__all__ = [
    # Basic RL algorithms
    "EnhancedDeepQLearningAgent",
    "EnhancedPPOAgent",

    # Enhanced learning mechanisms
    "MultiAgentRLAgent",
    "IMPALAAgent",
    "ModelBasedRLAgent",
    "OfflineRLAgent",
    "StableBaselinesAgent",
    "StableBaselinesFactory"
]
