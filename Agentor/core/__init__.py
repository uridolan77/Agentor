"""
Core module for the Agentor framework.

This module provides the core infrastructure for the Agentor framework, including:
- Plugin system for extensibility
- Configuration management
- Dependency injection
- Component registry
"""

from agentor.core.plugin import Plugin, PluginRegistry, PluginLoader, get_plugin_registry
from agentor.core.config import ConfigSource, EnvConfigSource, FileConfigSource, DictConfigSource, ConfigManager, get_config_manager, get_config, get_config_section, get_typed_config
from agentor.core.di import Container, Lifetime, inject, get_container
from agentor.core.registry import ComponentRegistry, get_component_registry

__all__ = [
    # Plugin system
    'Plugin',
    'PluginRegistry',
    'PluginLoader',
    'get_plugin_registry',
    
    # Configuration
    'ConfigSource',
    'EnvConfigSource',
    'FileConfigSource',
    'DictConfigSource',
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'get_config_section',
    'get_typed_config',
    
    # Dependency injection
    'Container',
    'Lifetime',
    'inject',
    'get_container',
    
    # Component registry
    'ComponentRegistry',
    'get_component_registry',
]
