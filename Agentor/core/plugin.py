"""
Plugin system for the Agentor framework.

This module provides the core plugin infrastructure, including:
- Plugin interface for creating new plugins
- Plugin registry for registering and discovering plugins
- Plugin loader for loading plugins from various sources
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable, TypeVar, Generic, Set
import importlib
import inspect
import logging
import pkgutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Plugin(ABC):
    """Base class for all plugins in the Agentor framework."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the plugin.

        Returns:
            The name of the plugin
        """
        pass

    @property
    def version(self) -> str:
        """Get the version of the plugin.

        Returns:
            The version of the plugin
        """
        return "0.1.0"

    @property
    def description(self) -> str:
        """Get the description of the plugin.

        Returns:
            The description of the plugin
        """
        return ""

    @property
    def dependencies(self) -> List[str]:
        """Get the dependencies of the plugin.

        Returns:
            A list of plugin names that this plugin depends on
        """
        return []

    def initialize(self, registry: 'PluginRegistry') -> None:
        """Initialize the plugin.

        This method is called when the plugin is registered.

        Args:
            registry: The plugin registry
        """
        pass

    def shutdown(self) -> None:
        """Shutdown the plugin.

        This method is called when the plugin is unregistered.
        """
        pass


class PluginRegistry:
    """Registry for plugins in the Agentor framework."""

    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_types: Dict[Type[Plugin], Dict[str, Plugin]] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._cleanup_tasks: List[Callable] = []
        self._initialized = False

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin.

        Args:
            plugin: The plugin to register

        Raises:
            ValueError: If a plugin with the same name is already registered
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' is already registered")

        # Register the plugin
        self._plugins[plugin.name] = plugin

        # Register the plugin by type
        for base in inspect.getmro(plugin.__class__):
            if base is Plugin or issubclass(base, Plugin):
                if base not in self._plugin_types:
                    self._plugin_types[base] = {}
                self._plugin_types[base][plugin.name] = plugin

        # Initialize the plugin if the registry is initialized
        if self._initialized:
            plugin.initialize(self)

        logger.info(f"Registered plugin: {plugin.name} (version {plugin.version})")

    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin.

        Args:
            name: The name of the plugin to unregister

        Raises:
            ValueError: If the plugin is not registered
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered")

        # Get the plugin
        plugin = self._plugins[name]

        # Shutdown the plugin
        plugin.shutdown()

        # Unregister the plugin
        del self._plugins[name]

        # Unregister the plugin by type
        for base in inspect.getmro(plugin.__class__):
            if base is Plugin or issubclass(base, Plugin):
                if base in self._plugin_types and name in self._plugin_types[base]:
                    del self._plugin_types[base][name]

        logger.info(f"Unregistered plugin: {name}")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name.

        Args:
            name: The name of the plugin

        Returns:
            The plugin, or None if not found
        """
        return self._plugins.get(name)

    def get_plugins(self) -> Dict[str, Plugin]:
        """Get all registered plugins.

        Returns:
            A dictionary of plugin names to plugins
        """
        return self._plugins.copy()

    def get_plugins_by_type(self, plugin_type: Type[T]) -> Dict[str, T]:
        """Get all plugins of a specific type.

        Args:
            plugin_type: The plugin type

        Returns:
            A dictionary of plugin names to plugins of the specified type
        """
        if plugin_type not in self._plugin_types:
            return {}

        return self._plugin_types[plugin_type].copy()

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback.

        Args:
            hook_name: The name of the hook
            callback: The callback function
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        self._hooks[hook_name].append(callback)
        logger.debug(f"Registered hook: {hook_name}")

    def unregister_hook(self, hook_name: str, callback: Callable) -> None:
        """Unregister a hook callback.

        Args:
            hook_name: The name of the hook
            callback: The callback function

        Raises:
            ValueError: If the hook or callback is not registered
        """
        if hook_name not in self._hooks:
            raise ValueError(f"Hook '{hook_name}' is not registered")

        if callback not in self._hooks[hook_name]:
            raise ValueError(f"Callback is not registered for hook '{hook_name}'")

        self._hooks[hook_name].remove(callback)
        logger.debug(f"Unregistered hook: {hook_name}")

    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all callbacks for a hook.

        Args:
            hook_name: The name of the hook
            *args: Positional arguments to pass to the callbacks
            **kwargs: Keyword arguments to pass to the callbacks

        Returns:
            A list of results from the callbacks
        """
        if hook_name not in self._hooks:
            return []

        results = []
        for callback in self._hooks[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error calling hook '{hook_name}': {str(e)}")

        return results

    def initialize(self) -> None:
        """Initialize all registered plugins."""
        if self._initialized:
            return

        # Initialize plugins in dependency order
        remaining_plugins = set(self._plugins.keys())
        initialized_plugins = set()

        while remaining_plugins:
            progress = False

            for name in list(remaining_plugins):
                plugin = self._plugins[name]
                dependencies = set(plugin.dependencies)

                # Check if all dependencies are initialized
                if dependencies.issubset(initialized_plugins):
                    # Initialize the plugin
                    try:
                        plugin.initialize(self)
                        initialized_plugins.add(name)
                        remaining_plugins.remove(name)
                        progress = True
                        logger.info(f"Initialized plugin: {name}")
                    except Exception as e:
                        logger.error(f"Error initializing plugin '{name}': {str(e)}")
                        remaining_plugins.remove(name)
                        progress = True

            # If no progress was made, there might be a circular dependency
            if not progress:
                logger.error(f"Circular dependency detected in plugins: {remaining_plugins}")
                break

        self._initialized = True

    def register_cleanup_task(self, task: Callable) -> None:
        """Register a cleanup task to be executed during shutdown.

        Args:
            task: The cleanup task to register. Can be a coroutine function.
        """
        self._cleanup_tasks.append(task)
        logger.debug(f"Registered cleanup task: {task.__name__ if hasattr(task, '__name__') else 'anonymous'}")

    async def execute_cleanup_tasks(self) -> None:
        """Execute all registered cleanup tasks."""
        import asyncio

        for task in self._cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
                logger.debug(f"Executed cleanup task: {task.__name__ if hasattr(task, '__name__') else 'anonymous'}")
            except Exception as e:
                logger.error(f"Error executing cleanup task: {str(e)}")

    def shutdown(self) -> None:
        """Shutdown all registered plugins."""
        if not self._initialized:
            return

        # Shutdown plugins in reverse dependency order
        for name, plugin in reversed(list(self._plugins.items())):
            try:
                plugin.shutdown()
                logger.info(f"Shutdown plugin: {name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin '{name}': {str(e)}")

        # Execute cleanup tasks
        import asyncio
        try:
            # Create a new event loop if needed
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # No event loop exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the cleanup tasks
            loop.run_until_complete(self.execute_cleanup_tasks())
        except Exception as e:
            logger.error(f"Error executing cleanup tasks: {str(e)}")

        self._initialized = False


class PluginLoader:
    """Loader for plugins in the Agentor framework."""

    @staticmethod
    def load_plugins_from_module(module_name: str, registry: PluginRegistry) -> None:
        """Load plugins from a module.

        Args:
            module_name: The name of the module
            registry: The plugin registry
        """
        try:
            module = importlib.import_module(module_name)

            # Find all Plugin subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, Plugin) and
                    obj is not Plugin and
                    not inspect.isabstract(obj)):
                    try:
                        plugin = obj()
                        registry.register_plugin(plugin)
                    except Exception as e:
                        logger.error(f"Error creating plugin '{name}' from module '{module_name}': {str(e)}")

        except ImportError as e:
            logger.error(f"Error importing module '{module_name}': {str(e)}")

    @staticmethod
    def load_plugins_from_directory(directory: str, registry: PluginRegistry) -> None:
        """Load plugins from a directory.

        Args:
            directory: The directory to load plugins from
            registry: The plugin registry
        """
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            logger.error(f"Plugin directory '{directory}' does not exist or is not a directory")
            return

        # Add the directory to the Python path
        sys.path.insert(0, str(path.parent))

        try:
            # Load all Python modules in the directory
            for _, name, is_pkg in pkgutil.iter_modules([str(path)]):
                if not name.startswith('_'):  # Skip private modules
                    module_name = f"{path.name}.{name}"
                    PluginLoader.load_plugins_from_module(module_name, registry)

        finally:
            # Remove the directory from the Python path
            sys.path.remove(str(path.parent))

    @staticmethod
    def load_plugins_from_entry_points(group: str, registry: PluginRegistry) -> None:
        """Load plugins from entry points.

        Args:
            group: The entry point group
            registry: The plugin registry
        """
        try:
            import pkg_resources

            for entry_point in pkg_resources.iter_entry_points(group):
                try:
                    plugin_class = entry_point.load()
                    if (inspect.isclass(plugin_class) and
                        issubclass(plugin_class, Plugin) and
                        not inspect.isabstract(plugin_class)):
                        plugin = plugin_class()
                        registry.register_plugin(plugin)
                except Exception as e:
                    logger.error(f"Error loading plugin from entry point '{entry_point.name}': {str(e)}")

        except ImportError:
            logger.warning("pkg_resources not available, skipping entry point loading")


# Global plugin registry
plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry.

    Returns:
        The global plugin registry
    """
    return plugin_registry
