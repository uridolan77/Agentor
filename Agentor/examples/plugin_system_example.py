"""
Example demonstrating the plugin system and dependency injection in the Agentor framework.

This example shows how to:
- Create and register plugins
- Use dependency injection
- Configure the framework using the unified configuration system
- Create agents and tools using the standardized interfaces
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from agentor.core.plugin import Plugin, get_plugin_registry
from agentor.core.config import get_config, get_typed_config
from agentor.core.di import inject, get_container, Lifetime
from agentor.core.registry import get_component_registry
from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents.enhanced_tools import EnhancedTool, ToolResult, get_tool_registry
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.interfaces.agent import AgentOutput

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define a custom plugin
class GreetingPlugin(Plugin):
    """Plugin that provides greeting functionality."""
    
    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return "greeting"
    
    @property
    def description(self) -> str:
        """Get the description of the plugin."""
        return "Provides greeting functionality"
    
    def get_greeting(self, name: str) -> str:
        """Get a greeting for a name.
        
        Args:
            name: The name to greet
            
        Returns:
            The greeting
        """
        return f"Hello, {name}!"
    
    def initialize(self, registry) -> None:
        """Initialize the plugin."""
        logger.info("Initializing greeting plugin")
    
    def shutdown(self) -> None:
        """Shutdown the plugin."""
        logger.info("Shutting down greeting plugin")


# Define a custom tool
class GreetingTool(EnhancedTool):
    """Tool for generating greetings."""
    
    def __init__(self, greeting_plugin: Optional[GreetingPlugin] = None):
        """Initialize the greeting tool.
        
        Args:
            greeting_plugin: The greeting plugin to use
        """
        super().__init__(
            name="greeting",
            description="Generate a greeting for a name"
        )
        self.greeting_plugin = greeting_plugin
    
    async def run(self, name: str) -> ToolResult:
        """Generate a greeting for a name.
        
        Args:
            name: The name to greet
            
        Returns:
            The greeting
        """
        if self.greeting_plugin:
            greeting = self.greeting_plugin.get_greeting(name)
        else:
            greeting = f"Hello, {name}!"
        
        return ToolResult(
            success=True,
            data={"greeting": greeting}
        )


# Define a custom agent
class GreetingAgent(EnhancedAgent):
    """Agent that generates greetings."""
    
    def __init__(self, name=None, tool_registry: Optional[IToolRegistry] = None):
        """Initialize the greeting agent.
        
        Args:
            name: The name of the agent
            tool_registry: The tool registry to use
        """
        super().__init__(name=name or "GreetingAgent", tool_registry=tool_registry)
    
    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return "Agent that generates greetings"
    
    def decide(self) -> str:
        """Decide what action to take."""
        # Always use the greeting action
        return "generate_greeting"
    
    async def preprocess(self, input_data):
        """Preprocess the input data."""
        # Extract the name from the query
        query = input_data.query.strip()
        
        # If the query starts with "Hello", extract the name
        if query.lower().startswith("hello"):
            name = query[5:].strip().rstrip("!.,;:")
        else:
            name = query
        
        # Store the name in the state
        self.state["name"] = name
        
        return input_data
    
    async def run_once(self):
        """Run one cycle of the agent."""
        # Get the name from the state
        name = self.state.get("name", "World")
        
        # Check if we have the greeting tool
        if "greeting" in self.tools:
            # Use the greeting tool
            result = await self.tools["greeting"].run(name=name)
            
            if result.success:
                return result.data["greeting"]
            else:
                return f"Error generating greeting: {result.error}"
        else:
            # Generate a greeting directly
            return f"Hello, {name}!"


# Function that uses dependency injection
@inject
async def run_example(greeting_agent: GreetingAgent, tool_registry: IToolRegistry):
    """Run the example.
    
    Args:
        greeting_agent: The greeting agent (injected)
        tool_registry: The tool registry (injected)
    """
    logger.info("Running plugin system example")
    
    # Print the registered tools
    logger.info("Registered tools:")
    for name, tool in tool_registry.get_tools().items():
        logger.info(f"  - {name}: {tool.description}")
    
    # Run the agent with different inputs
    inputs = [
        "World",
        "Hello Alice",
        "Bob"
    ]
    
    for input_text in inputs:
        logger.info(f"\nProcessing input: {input_text}")
        
        # Run the agent
        output = await greeting_agent.run(input_text)
        
        # Print the output
        logger.info(f"Agent response: {output.response}")
        logger.info(f"Execution time: {output.metadata['execution_time']:.3f} seconds")


async def main():
    """Main function."""
    # Get the plugin registry
    registry = get_plugin_registry()
    
    # Get the container
    container = get_container()
    
    # Get the component registry
    component_registry = get_component_registry()
    
    # Get the tool registry
    tool_registry = get_tool_registry()
    
    # Register the greeting plugin
    greeting_plugin = GreetingPlugin()
    registry.register_plugin(greeting_plugin)
    
    # Register the greeting tool
    greeting_tool = GreetingTool(greeting_plugin=greeting_plugin)
    tool_registry.register_tool(greeting_tool)
    
    # Register the tool registry in the container
    container.register_instance(IToolRegistry, tool_registry)
    
    # Create and register the greeting agent
    greeting_agent = GreetingAgent(tool_registry=tool_registry)
    container.register_instance(GreetingAgent, greeting_agent)
    
    # Register the agent in the component registry
    component_registry.register_agent_provider(greeting_agent.name, greeting_agent)
    
    # Initialize the plugin registry
    registry.initialize()
    
    try:
        # Run the example
        await run_example()
    finally:
        # Shutdown the plugin registry
        registry.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
