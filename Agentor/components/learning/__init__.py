"""
Learning module for the Agentor framework.

This module provides various learning algorithms and mechanisms for agents, including:
- Deep Q-Learning for reinforcement learning with neural networks
- Proximal Policy Optimization (PPO) for policy-based reinforcement learning
- Transfer learning for sharing knowledge between agents
- Advanced learning mechanisms with feedback loops
"""

# Import transfer learning components
from agentor.components.learning.transfer_learning import (
    KnowledgeTransfer,
    ModelTransfer,
    ExperienceTransfer,
    PolicyDistillation,
    TransferLearningManager
)

# Import advanced learning components
from agentor.components.learning.advanced_learning import (
    FeedbackEnhancedLearning,
    AdaptiveLearningRateScheduler,
    FeedbackCollector,
    FeedbackEntry,
    PrioritizedExperienceReplay,
    CurriculumLearning,
    MetaLearningOptimizer
)

__all__ = [
    # Transfer learning components
    'KnowledgeTransfer',
    'ModelTransfer',
    'ExperienceTransfer',
    'PolicyDistillation',
    'TransferLearningManager',
    
    # Advanced learning components
    'FeedbackEnhancedLearning',
    'AdaptiveLearningRateScheduler',
    'FeedbackCollector',
    'FeedbackEntry',
    'PrioritizedExperienceReplay',
    'CurriculumLearning',
    'MetaLearningOptimizer',
]
