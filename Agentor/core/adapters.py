"""
Module for registering all adapters with the plugin system.

This module provides functions for registering all the adapters for existing
components with the plugin system.
"""

import logging
from typing import List

from agentor.core.plugin import get_plugin_registry, Plugin
from agentor.components.memory.adapters import (
    SimpleMemoryPlugin,
    VectorMemoryPlugin,
    EpisodicMemoryPlugin,
    SemanticMemoryPlugin,
    ProceduralMemoryPlugin,
    UnifiedMemoryPlugin
)
from agentor.llm_gateway.llm.adapters import (
    OpenAILLMPlugin,
    SemanticCachedOpenAILLMPlugin
)
from agentor.agents.adapters import (
    BaseAgentPlugin,
    DeepQLearningAgentPlugin,
    PPOAgentPlugin,
    SemanticRouterPlugin,
    RuleBasedRouterPlugin,
    HierarchicalRouterPlugin
)
from agentor.agents.tools.adapters import (
    WeatherToolPlugin,
    NewsToolPlugin,
    CalculatorToolPlugin
)

logger = logging.getLogger(__name__)


def get_adapter_plugins() -> List[Plugin]:
    """Get all adapter plugins.
    
    Returns:
        A list of adapter plugins
    """
    return [
        # Memory adapters
        SimpleMemoryPlugin(),
        VectorMemoryPlugin(),
        EpisodicMemoryPlugin(),
        SemanticMemoryPlugin(),
        ProceduralMemoryPlugin(),
        UnifiedMemoryPlugin(),
        
        # LLM adapters
        OpenAILLMPlugin(),
        SemanticCachedOpenAILLMPlugin(),
        
        # Agent adapters
        BaseAgentPlugin(),
        DeepQLearningAgentPlugin(),
        PPOAgentPlugin(),
        SemanticRouterPlugin(),
        RuleBasedRouterPlugin(),
        HierarchicalRouterPlugin(),
        
        # Tool adapters
        WeatherToolPlugin(),
        NewsToolPlugin(),
        CalculatorToolPlugin()
    ]


def register_adapters() -> None:
    """Register all adapters with the plugin system."""
    registry = get_plugin_registry()
    
    # Get all adapter plugins
    plugins = get_adapter_plugins()
    
    # Register each plugin
    for plugin in plugins:
        registry.register_plugin(plugin)
    
    logger.info(f"Registered {len(plugins)} adapter plugins")
