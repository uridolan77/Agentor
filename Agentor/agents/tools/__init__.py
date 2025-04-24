"""
Tools module for the Agentor framework.

This module provides various tool implementations, including:
- Base tool class
- Enhanced tools with standardized interfaces
- Adapters for using tool components with the standardized interfaces
"""

# Base tool classes (legacy, use enhanced versions for new code)
from agentor.agents.tools.base import BaseTool, ToolResult

# Enhanced tool classes (preferred for new code)
from agentor.agents.enhanced_tools import EnhancedTool, EnhancedToolRegistry
from agentor.agents.tools.adapters import (
    ToolAdapter,
    ToolRegistryAdapter,
    WeatherToolAdapter,
    NewsToolAdapter,
    CalculatorToolAdapter,
    WeatherToolPlugin,
    NewsToolPlugin,
    CalculatorToolPlugin
)
from agentor.agents.tools.recommender import ToolRecommender
from agentor.agents.tools.error_handling import (
    ErrorHandlingToolRegistry,
    ErrorHandlingTool,
    ToolErrorManager
)
from agentor.agents.tools.cached_registry import (
    CachedToolRegistry,
    create_cached_registry
)

__all__ = [
    # Legacy tool classes (for backward compatibility)
    'BaseTool',
    'ToolResult',

    # Enhanced tool classes (preferred for new code)
    'EnhancedTool',
    'EnhancedToolRegistry',

    # Tool adapters
    'ToolAdapter',
    'ToolRegistryAdapter',
    'WeatherToolAdapter',
    'NewsToolAdapter',
    'CalculatorToolAdapter',

    # Tool plugins
    'WeatherToolPlugin',
    'NewsToolPlugin',
    'CalculatorToolPlugin',

    # Tool recommender
    'ToolRecommender',

    # Error handling
    'ErrorHandlingToolRegistry',
    'ErrorHandlingTool',
    'ToolErrorManager',

    # Caching
    'CachedToolRegistry',
    'create_cached_registry',
]
