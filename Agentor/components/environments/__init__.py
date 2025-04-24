"""
Environment components for the Agentor framework.

This module provides environment components for the Agentor framework.
"""

from agentor.components.environments.base import (
    BaseEnvironment, TimeLimit, Monitor
)
from agentor.components.environments.grid_world import GridWorldEnv
from agentor.components.environments.cart_pole import CartPoleEnv
from agentor.components.environments.mountain_car import MountainCarEnv
from agentor.components.environments.gymnasium_adapter import GymnasiumAdapter
from agentor.components.environments.training import AgentEnvironmentLoop
from agentor.components.environments.plugin import EnvironmentPlugin

# New environment components
from agentor.components.environments.streaming import (
    StreamingEnvironment, DataStream, CallbackDataStream,
    GeneratorDataStream, MarketDataStream, NewsStream
)
from agentor.components.environments.hierarchical import (
    HierarchicalEnvironment, SubEnvironment, TaskHierarchicalEnvironment,
    OptionsHierarchicalEnvironment
)
from agentor.components.environments.wrappers import (
    EnvironmentWrapper, NormalizeObservation, ClipReward, FrameStack,
    VideoRecorder, ActionRepeat, ActionNoise, RewardScaling,
    TransformObservation, TransformAction, VisualizeEnvironment
)
from agentor.components.environments.enhanced_multi_agent import (
    CommunicativeMultiAgentEnv, TeamBasedMultiAgentEnv,
    DynamicMultiAgentEnv, CompetitiveMultiAgentEnv,
    Message, CommunicationChannel
)
from agentor.components.environments.visualization import (
    EnvironmentRenderer, MatplotlibRenderer, VideoRenderer,
    DashboardRenderer, EnvironmentMonitor
)

__all__ = [
    # Base environment components
    "BaseEnvironment",
    "TimeLimit",
    "Monitor",

    # Standard environments
    "GridWorldEnv",
    "CartPoleEnv",
    "MountainCarEnv",
    "GymnasiumAdapter",
    "AgentEnvironmentLoop",
    "EnvironmentPlugin",

    # Streaming environment
    "StreamingEnvironment",
    "DataStream",
    "CallbackDataStream",
    "GeneratorDataStream",
    "MarketDataStream",
    "NewsStream",

    # Hierarchical environments
    "HierarchicalEnvironment",
    "SubEnvironment",
    "TaskHierarchicalEnvironment",
    "OptionsHierarchicalEnvironment",

    # Environment wrappers
    "EnvironmentWrapper",
    "NormalizeObservation",
    "ClipReward",
    "FrameStack",
    "VideoRecorder",
    "ActionRepeat",
    "ActionNoise",
    "RewardScaling",
    "TransformObservation",
    "TransformAction",
    "VisualizeEnvironment",

    # Enhanced multi-agent environments
    "CommunicativeMultiAgentEnv",
    "TeamBasedMultiAgentEnv",
    "DynamicMultiAgentEnv",
    "CompetitiveMultiAgentEnv",
    "Message",
    "CommunicationChannel",

    # Visualization tools
    "EnvironmentRenderer",
    "MatplotlibRenderer",
    "VideoRenderer",
    "DashboardRenderer",
    "EnvironmentMonitor"
]
