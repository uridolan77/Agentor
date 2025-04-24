"""
Environment plugin for the Agentor framework.

This module provides a plugin for registering environment components.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from agentor.core.plugin import Plugin
from agentor.core.registry import get_component_registry
from agentor.components.environments.grid_world import GridWorldEnv

logger = logging.getLogger(__name__)


class EnvironmentPlugin(Plugin):
    """Plugin for environment components."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin.
        
        Returns:
            The name of the plugin
        """
        return "environment"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin.
        
        Returns:
            The description of the plugin
        """
        return "Plugin for environment components"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin.
        
        Args:
            registry: The plugin registry
        """
        # Get the component registry
        component_registry = get_component_registry()
        
        # Register environment providers
        component_registry.register_environment_provider("grid_world", GridWorldEnv)
        
        logger.info("Registered environment providers")
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        logger.info("Shutting down environment plugin")
