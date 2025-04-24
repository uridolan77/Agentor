"""
Unit tests for the plugin system.
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional

from agentor.core.plugin import Plugin, PluginRegistry
from agentor.core.di import Container, Lifetime
from agentor.core.config import ConfigManager, DictConfigSource


class TestPlugin(Plugin):
    """Test plugin for unit tests."""
    
    def __init__(self, name="test_plugin"):
        """Initialize the test plugin.
        
        Args:
            name: The name of the plugin
        """
        self._name = name
        self.initialized = False
        self.shutdown_called = False
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Test plugin for unit tests"
    
    def test_method(self, value: str) -> str:
        """Test method.
        
        Args:
            value: The value to process
            
        Returns:
            The processed value
        """
        return f"Processed: {value}"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin."""
        self.initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.shutdown_called = True


class DependentPlugin(Plugin):
    """Plugin that depends on another plugin."""
    
    def __init__(self, name="dependent_plugin"):
        """Initialize the dependent plugin.
        
        Args:
            name: The name of the plugin
        """
        self._name = name
        self.initialized = False
        self.shutdown_called = False
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Plugin that depends on another plugin"
    
    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies of the plugin."""
        return ["test_plugin"]
    
    def initialize(self, registry) -> None:
        """Initialize the plugin."""
        self.initialized = True
        
        # Get the dependency
        self.dependency = registry.get_plugin("test_plugin")
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.shutdown_called = True
    
    def use_dependency(self, value: str) -> str:
        """Use the dependency.
        
        Args:
            value: The value to process
            
        Returns:
            The processed value
        """
        return self.dependency.test_method(value)


@pytest.fixture
def registry():
    """Create a plugin registry for testing."""
    return PluginRegistry()


def test_register_plugin(registry):
    """Test registering a plugin."""
    # Create a plugin
    plugin = TestPlugin()
    
    # Register the plugin
    registry.register_plugin(plugin)
    
    # Check that the plugin is registered
    assert plugin.name in registry.get_plugins()
    assert registry.get_plugin(plugin.name) == plugin


def test_unregister_plugin(registry):
    """Test unregistering a plugin."""
    # Create a plugin
    plugin = TestPlugin()
    
    # Register the plugin
    registry.register_plugin(plugin)
    
    # Unregister the plugin
    registry.unregister_plugin(plugin.name)
    
    # Check that the plugin is unregistered
    assert plugin.name not in registry.get_plugins()
    assert registry.get_plugin(plugin.name) is None
    assert plugin.shutdown_called


def test_initialize_plugins(registry):
    """Test initializing plugins."""
    # Create plugins
    plugin1 = TestPlugin(name="plugin1")
    plugin2 = TestPlugin(name="plugin2")
    
    # Register the plugins
    registry.register_plugin(plugin1)
    registry.register_plugin(plugin2)
    
    # Initialize the plugins
    registry.initialize()
    
    # Check that the plugins are initialized
    assert plugin1.initialized
    assert plugin2.initialized


def test_plugin_dependencies(registry):
    """Test plugin dependencies."""
    # Create plugins
    plugin1 = TestPlugin(name="test_plugin")
    plugin2 = DependentPlugin()
    
    # Register the plugins in reverse order
    registry.register_plugin(plugin2)
    registry.register_plugin(plugin1)
    
    # Initialize the plugins
    registry.initialize()
    
    # Check that the plugins are initialized
    assert plugin1.initialized
    assert plugin2.initialized
    
    # Check that the dependency is resolved
    assert plugin2.dependency == plugin1
    assert plugin2.use_dependency("test") == "Processed: test"


def test_plugin_hooks(registry):
    """Test plugin hooks."""
    # Create a plugin
    plugin = TestPlugin()
    
    # Register the plugin
    registry.register_plugin(plugin)
    
    # Register a hook
    hook_called = False
    hook_args = None
    hook_kwargs = None
    
    def test_hook(*args, **kwargs):
        nonlocal hook_called, hook_args, hook_kwargs
        hook_called = True
        hook_args = args
        hook_kwargs = kwargs
        return "hook_result"
    
    registry.register_hook("test_hook", test_hook)
    
    # Call the hook
    results = registry.call_hook("test_hook", "arg1", "arg2", kwarg1="value1")
    
    # Check that the hook was called
    assert hook_called
    assert hook_args == ("arg1", "arg2")
    assert hook_kwargs == {"kwarg1": "value1"}
    assert results == ["hook_result"]


def test_plugin_types(registry):
    """Test getting plugins by type."""
    # Create plugins
    plugin1 = TestPlugin(name="plugin1")
    plugin2 = TestPlugin(name="plugin2")
    plugin3 = DependentPlugin()
    
    # Register the plugins
    registry.register_plugin(plugin1)
    registry.register_plugin(plugin2)
    registry.register_plugin(plugin3)
    
    # Get plugins by type
    test_plugins = registry.get_plugins_by_type(TestPlugin)
    dependent_plugins = registry.get_plugins_by_type(DependentPlugin)
    
    # Check the results
    assert len(test_plugins) == 2
    assert "plugin1" in test_plugins
    assert "plugin2" in test_plugins
    
    assert len(dependent_plugins) == 1
    assert "dependent_plugin" in dependent_plugins


def test_container_integration():
    """Test integration with the dependency injection container."""
    # Create a container
    container = Container()
    
    # Register a service
    class TestService:
        def get_value(self):
            return "test_value"
    
    container.register(TestService, TestService, Lifetime.SINGLETON)
    
    # Create a plugin that uses the service
    class ServiceUsingPlugin(Plugin):
        def __init__(self, service: TestService = None):
            self._service = service
        
        @property
        def name(self):
            return "service_using_plugin"
        
        def get_service_value(self):
            return self._service.get_value()
    
    # Create a factory function that uses the container
    def create_plugin(container):
        service = container.resolve(TestService)
        return ServiceUsingPlugin(service)
    
    # Register the plugin factory
    container.register_factory(Plugin, create_plugin, Lifetime.SINGLETON, name="service_using_plugin")
    
    # Resolve the plugin
    plugin = container.resolve("service_using_plugin")
    
    # Check that the plugin works
    assert plugin.name == "service_using_plugin"
    assert plugin.get_service_value() == "test_value"


def test_config_integration():
    """Test integration with the configuration system."""
    # Create a config manager
    config_manager = ConfigManager()
    
    # Add a config source
    config_manager.add_source(DictConfigSource({
        "plugin": {
            "name": "configured_plugin",
            "value": "configured_value"
        }
    }))
    
    # Create a plugin that uses the configuration
    class ConfiguredPlugin(Plugin):
        def __init__(self, name=None, value=None):
            self._name = name
            self._value = value
        
        @property
        def name(self):
            return self._name
        
        def get_value(self):
            return self._value
    
    # Create a factory function that uses the configuration
    def create_plugin(config_manager):
        config = config_manager.get_config_section("plugin")
        return ConfiguredPlugin(
            name=config.get("name"),
            value=config.get("value")
        )
    
    # Create the plugin
    plugin = create_plugin(config_manager)
    
    # Check that the plugin is configured correctly
    assert plugin.name == "configured_plugin"
    assert plugin.get_value() == "configured_value"
