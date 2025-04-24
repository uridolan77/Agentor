"""
State models for agents in the Agentor framework.

This module provides Pydantic models for agent state, which provide better
structure and validation compared to plain dictionaries.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Generic, TypeVar
from pydantic import BaseModel, Field
import numpy as np

T = TypeVar('T')


class BaseAgentState(BaseModel):
    """Base model for agent state."""

    current_query: Optional[str] = None
    current_context: Dict[str, Any] = Field(default_factory=dict)
    last_perception: Dict[str, Any] = Field(default_factory=dict)
    start_time: Optional[float] = None

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class LearningAgentState(BaseAgentState):
    """Model for learning agent state."""

    current_state: Dict[str, Any] = Field(default_factory=dict)
    current_episode_reward: float = 0.0
    current_episode_length: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    episodes_completed: int = 0
    exploration_rate: Optional[float] = None
    last_action: Optional[str] = None
    last_reward: Optional[float] = None
    training_metrics: Dict[str, Any] = Field(default_factory=dict)


class DeepQLearningState(LearningAgentState):
    """Model for Deep Q-Learning agent state."""

    memory_size: int = 0
    batch_size: int = 64
    target_update_counter: int = 0
    last_q_values: Optional[List[float]] = None
    last_target_q_values: Optional[List[float]] = None
    last_loss: Optional[float] = None


class PPOState(LearningAgentState):
    """Model for PPO agent state."""

    last_action_prob: Optional[float] = None
    last_value: Optional[float] = None
    last_log_prob: Optional[float] = None
    last_entropy: Optional[float] = None
    last_advantage: Optional[float] = None
    last_return: Optional[float] = None
    last_policy_loss: Optional[float] = None
    last_value_loss: Optional[float] = None
    last_total_loss: Optional[float] = None


class RuleBasedAgentState(BaseAgentState):
    """Model for rule-based agent state."""

    current_state: Dict[str, Any] = Field(default_factory=dict)
    applicable_rules: List[str] = Field(default_factory=list)
    selected_rule: Optional[str] = None
    rule_priorities: Dict[str, int] = Field(default_factory=dict)


class UtilityBasedAgentState(BaseAgentState):
    """Model for utility-based agent state."""

    current_state: Dict[str, Any] = Field(default_factory=dict)
    action_utilities: Dict[str, float] = Field(default_factory=dict)
    selected_action: Optional[str] = None
    utility_threshold: float = 0.0
    last_utility: Optional[float] = None


class ReactiveAgentState(BaseAgentState):
    """Model for reactive agent state."""

    current_state: Dict[str, Any] = Field(default_factory=dict)
    applicable_behaviors: List[str] = Field(default_factory=list)
    selected_behavior: Optional[str] = None
    behavior_priorities: Dict[str, int] = Field(default_factory=dict)


class MemoryEnhancedAgentState(BaseAgentState):
    """Model for memory-enhanced agent state."""

    current_state: Dict[str, Any] = Field(default_factory=dict)
    memory_type: str = "simple"  # "simple" or "semantic"
    memory_count: int = 0
    last_memory: Optional[Dict[str, Any]] = None
    last_recall_query: Optional[str] = None
    last_recall_count: int = 0
