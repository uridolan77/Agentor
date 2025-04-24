"""
Component registry for the Agentor framework.

This module provides a registry for components in the Agentor framework, including:
- Memory providers
- LLM providers
- Agent providers
- Tool providers
- Environment providers
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, Generic, Protocol, runtime_checkable
import logging

from agentor.core.interfaces.memory import MemoryProvider
from agentor.core.interfaces.llm import LLMProvider, StreamingLLMProvider
from agentor.core.interfaces.agent import AgentProvider
from agentor.core.interfaces.tool import ToolProvider
from agentor.core.interfaces.environment import IEnvironment

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ComponentRegistry:
    """Registry for components in the Agentor framework."""

    def __init__(self):
        """Initialize the component registry."""
        self._memory_providers: Dict[str, MemoryProvider] = {}
        self._llm_providers: Dict[str, LLMProvider] = {}
        self._streaming_llm_providers: Dict[str, StreamingLLMProvider] = {}
        self._agent_providers: Dict[str, AgentProvider] = {}
        self._tool_providers: Dict[str, ToolProvider] = {}
        self._environment_providers: Dict[str, Type[IEnvironment]] = {}

    def register_memory_provider(self, name: str, provider: MemoryProvider) -> None:
        """Register a memory provider.

        Args:
            name: The name of the provider
            provider: The memory provider

        Raises:
            ValueError: If a provider with the same name is already registered
        """
        if name in self._memory_providers:
            raise ValueError(f"Memory provider '{name}' is already registered")

        self._memory_providers[name] = provider
        logger.info(f"Registered memory provider: {name}")

    def unregister_memory_provider(self, name: str) -> None:
        """Unregister a memory provider.

        Args:
            name: The name of the provider to unregister

        Raises:
            ValueError: If the provider is not registered
        """
        if name not in self._memory_providers:
            raise ValueError(f"Memory provider '{name}' is not registered")

        del self._memory_providers[name]
        logger.info(f"Unregistered memory provider: {name}")

    def get_memory_provider(self, name: str) -> Optional[MemoryProvider]:
        """Get a memory provider by name.

        Args:
            name: The name of the provider

        Returns:
            The memory provider, or None if not found
        """
        return self._memory_providers.get(name)

    def get_memory_providers(self) -> Dict[str, MemoryProvider]:
        """Get all registered memory providers.

        Returns:
            A dictionary of provider names to memory providers
        """
        return self._memory_providers.copy()

    def register_llm_provider(self, name: str, provider: LLMProvider) -> None:
        """Register an LLM provider.

        Args:
            name: The name of the provider
            provider: The LLM provider

        Raises:
            ValueError: If a provider with the same name is already registered
        """
        if name in self._llm_providers:
            raise ValueError(f"LLM provider '{name}' is already registered")

        self._llm_providers[name] = provider

        # If the provider also supports streaming, register it as a streaming provider
        if isinstance(provider, StreamingLLMProvider):
            self._streaming_llm_providers[name] = provider

        logger.info(f"Registered LLM provider: {name}")

    def unregister_llm_provider(self, name: str) -> None:
        """Unregister an LLM provider.

        Args:
            name: The name of the provider to unregister

        Raises:
            ValueError: If the provider is not registered
        """
        if name not in self._llm_providers:
            raise ValueError(f"LLM provider '{name}' is not registered")

        del self._llm_providers[name]

        # Also unregister from streaming providers if present
        if name in self._streaming_llm_providers:
            del self._streaming_llm_providers[name]

        logger.info(f"Unregistered LLM provider: {name}")

    def get_llm_provider(self, name: str) -> Optional[LLMProvider]:
        """Get an LLM provider by name.

        Args:
            name: The name of the provider

        Returns:
            The LLM provider, or None if not found
        """
        return self._llm_providers.get(name)

    def get_llm_providers(self) -> Dict[str, LLMProvider]:
        """Get all registered LLM providers.

        Returns:
            A dictionary of provider names to LLM providers
        """
        return self._llm_providers.copy()

    def get_streaming_llm_provider(self, name: str) -> Optional[StreamingLLMProvider]:
        """Get a streaming LLM provider by name.

        Args:
            name: The name of the provider

        Returns:
            The streaming LLM provider, or None if not found
        """
        return self._streaming_llm_providers.get(name)

    def get_streaming_llm_providers(self) -> Dict[str, StreamingLLMProvider]:
        """Get all registered streaming LLM providers.

        Returns:
            A dictionary of provider names to streaming LLM providers
        """
        return self._streaming_llm_providers.copy()

    def register_agent_provider(self, name: str, provider: AgentProvider) -> None:
        """Register an agent provider.

        Args:
            name: The name of the provider
            provider: The agent provider

        Raises:
            ValueError: If a provider with the same name is already registered
        """
        if name in self._agent_providers:
            raise ValueError(f"Agent provider '{name}' is already registered")

        self._agent_providers[name] = provider
        logger.info(f"Registered agent provider: {name}")

    def unregister_agent_provider(self, name: str) -> None:
        """Unregister an agent provider.

        Args:
            name: The name of the provider to unregister

        Raises:
            ValueError: If the provider is not registered
        """
        if name not in self._agent_providers:
            raise ValueError(f"Agent provider '{name}' is not registered")

        del self._agent_providers[name]
        logger.info(f"Unregistered agent provider: {name}")

    def get_agent_provider(self, name: str) -> Optional[AgentProvider]:
        """Get an agent provider by name.

        Args:
            name: The name of the provider

        Returns:
            The agent provider, or None if not found
        """
        return self._agent_providers.get(name)

    def get_agent_providers(self) -> Dict[str, AgentProvider]:
        """Get all registered agent providers.

        Returns:
            A dictionary of provider names to agent providers
        """
        return self._agent_providers.copy()

    def register_tool_provider(self, provider: ToolProvider) -> None:
        """Register a tool provider.

        Args:
            provider: The tool provider

        Raises:
            ValueError: If a provider with the same name is already registered
        """
        name = provider.name

        if name in self._tool_providers:
            raise ValueError(f"Tool provider '{name}' is already registered")

        self._tool_providers[name] = provider
        logger.info(f"Registered tool provider: {name}")

    def unregister_tool_provider(self, name: str) -> None:
        """Unregister a tool provider.

        Args:
            name: The name of the provider to unregister

        Raises:
            ValueError: If the provider is not registered
        """
        if name not in self._tool_providers:
            raise ValueError(f"Tool provider '{name}' is not registered")

        del self._tool_providers[name]
        logger.info(f"Unregistered tool provider: {name}")

    def get_tool_provider(self, name: str) -> Optional[ToolProvider]:
        """Get a tool provider by name.

        Args:
            name: The name of the provider

        Returns:
            The tool provider, or None if not found
        """
        return self._tool_providers.get(name)

    def get_tool_providers(self) -> Dict[str, ToolProvider]:
        """Get all registered tool providers.

        Returns:
            A dictionary of provider names to tool providers
        """
        return self._tool_providers.copy()

    def register_environment_provider(self, name: str, provider: Type[IEnvironment]) -> None:
        """Register an environment provider.

        Args:
            name: The name of the provider
            provider: The environment provider class

        Raises:
            ValueError: If a provider with the same name is already registered
        """
        if name in self._environment_providers:
            raise ValueError(f"Environment provider '{name}' is already registered")

        self._environment_providers[name] = provider
        logger.info(f"Registered environment provider: {name}")

    def unregister_environment_provider(self, name: str) -> None:
        """Unregister an environment provider.

        Args:
            name: The name of the provider to unregister

        Raises:
            ValueError: If the provider is not registered
        """
        if name not in self._environment_providers:
            raise ValueError(f"Environment provider '{name}' is not registered")

        del self._environment_providers[name]
        logger.info(f"Unregistered environment provider: {name}")

    def get_environment_provider(self, name: str) -> Optional[Type[IEnvironment]]:
        """Get an environment provider by name.

        Args:
            name: The name of the provider

        Returns:
            The environment provider class, or None if not found
        """
        return self._environment_providers.get(name)

    def get_environment_providers(self) -> Dict[str, Type[IEnvironment]]:
        """Get all registered environment providers.

        Returns:
            A dictionary of provider names to environment provider classes
        """
        return self._environment_providers.copy()


# Global component registry
component_registry = ComponentRegistry()


def get_component_registry() -> ComponentRegistry:
    """Get the global component registry.

    Returns:
        The global component registry
    """
    return component_registry
