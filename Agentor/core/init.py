"""
Initialization module for the Agentor framework.

This module provides functions for initializing the Agentor framework, including
registering plugins, setting up configuration, and initializing the dependency
injection system.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from agentor.core.plugin import get_plugin_registry, PluginRegistry
from agentor.core.config import get_config_manager, ConfigManager, ConfigSource, DictConfigSource
from agentor.core.di import get_container, Container
from agentor.core.registry import get_component_registry, ComponentRegistry
from agentor.core.adapters import register_adapters

logger = logging.getLogger(__name__)


def initialize_framework(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize the Agentor framework.
    
    This function initializes the framework by setting up the configuration,
    registering adapters, and initializing the plugin system.
    
    Args:
        config: Optional configuration to use
    """
    logger.info("Initializing Agentor framework")
    
    # Get the plugin registry
    registry = get_plugin_registry()
    
    # Get the configuration manager
    config_manager = get_config_manager()
    
    # Get the container
    container = get_container()
    
    # Get the component registry
    component_registry = get_component_registry()
    
    # Add the configuration if provided
    if config:
        config_source = DictConfigSource(config)
        config_manager.add_source(config_source, priority=200)  # Higher priority than default sources
    
    # Register the adapters
    register_adapters()
    
    # Initialize the plugin registry
    registry.initialize()
    
    logger.info("Agentor framework initialized")


def shutdown_framework() -> None:
    """Shutdown the Agentor framework.
    
    This function shuts down the framework by shutting down the plugin system.
    """
    logger.info("Shutting down Agentor framework")
    
    # Get the plugin registry
    registry = get_plugin_registry()
    
    # Shutdown the plugin registry
    registry.shutdown()
    
    logger.info("Agentor framework shut down")


@asynccontextmanager
async def framework_lifespan():
    """Manage the lifecycle of the Agentor framework.
    
    This context manager initializes the framework and shuts it down when done.
    """
    try:
        # Initialize the framework
        initialize_framework()
        yield
    finally:
        # Shutdown the framework
        shutdown_framework()


async def run_with_framework(func, *args, **kwargs):
    """Run a function with the Agentor framework.
    
    This function initializes the framework, runs the provided function,
    and then shuts down the framework.
    
    Args:
        func: The function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
    """
    async with framework_lifespan():
        return await func(*args, **kwargs)
