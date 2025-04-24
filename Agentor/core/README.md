# Agentor Core Framework

The Agentor Core Framework provides a standardized, extensible foundation for building agent-based systems. It includes a plugin system, dependency injection, configuration management, and standardized interfaces for components.

## Key Features

- **Plugin System**: Easily extend the framework with new components
- **Dependency Injection**: Reduce coupling between components
- **Configuration Management**: Unified configuration from multiple sources
- **Standardized Interfaces**: Consistent APIs for all components
- **Component Registry**: Discover and use registered components

## Plugin System

The plugin system allows you to extend the framework with new components:

```python
from agentor.core.plugin import Plugin, get_plugin_registry

class MyPlugin(Plugin):
    @property
    def name(self) -> str:
        return "my_plugin"
    
    @property
    def description(self) -> str:
        return "My custom plugin"
    
    def initialize(self, registry) -> None:
        print("Initializing my plugin")
    
    def shutdown(self) -> None:
        print("Shutting down my plugin")
    
    def my_function(self, param: str) -> str:
        return f"Processed: {param}"

# Register the plugin
registry = get_plugin_registry()
registry.register_plugin(MyPlugin())
registry.initialize()

# Use the plugin
plugin = registry.get_plugin("my_plugin")
result = plugin.my_function("hello")
print(result)  # Output: "Processed: hello"
```

## Dependency Injection

The dependency injection system reduces coupling between components:

```python
from agentor.core.di import inject, get_container, Lifetime

# Define a service
class DatabaseService:
    def get_data(self, id: str) -> dict:
        return {"id": id, "name": "Example"}

# Register the service
container = get_container()
container.register(DatabaseService, DatabaseService, Lifetime.SINGLETON)

# Use dependency injection
@inject
def process_data(db_service: DatabaseService):
    data = db_service.get_data("123")
    print(f"Processing data: {data}")

# Call the function
process_data()  # The DatabaseService is automatically injected
```

## Configuration Management

The configuration system provides a unified way to manage configuration:

```python
from agentor.core.config import get_config, get_config_section, get_typed_config
from pydantic import BaseModel

# Define a configuration model
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    username: str
    password: str
    database: str

# Get configuration
config = get_config()
db_config = get_config_section("database")
typed_config = get_typed_config(DatabaseConfig)

# Use the configuration
print(f"Database host: {typed_config.host}")
print(f"Database port: {typed_config.port}")
```

## Standardized Interfaces

The framework provides standardized interfaces for all components:

```python
from agentor.core.interfaces.memory import ISemanticMemory
from agentor.core.interfaces.llm import ILLM, LLMRequest, LLMResponse
from agentor.core.interfaces.agent import IAgent
from agentor.core.interfaces.tool import ITool, ToolResult

# Implement a tool
class CalculatorTool(ITool):
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform calculations"
    
    async def run(self, expression: str) -> ToolResult:
        try:
            result = eval(expression)
            return ToolResult(success=True, data={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The expression to evaluate"
                }
            },
            "required": ["expression"]
        }
```

## Component Registry

The component registry allows you to discover and use registered components:

```python
from agentor.core.registry import get_component_registry

# Get the component registry
registry = get_component_registry()

# Register components
registry.register_memory_provider("semantic", my_semantic_memory)
registry.register_llm_provider("openai", my_openai_llm)
registry.register_agent_provider("assistant", my_assistant_agent)
registry.register_tool_provider(my_calculator_tool)

# Get components
memory = registry.get_memory_provider("semantic")
llm = registry.get_llm_provider("openai")
agent = registry.get_agent_provider("assistant")
tool = registry.get_tool_provider("calculator")

# Get all components of a type
all_tools = registry.get_tool_providers()
```

## Enhanced Base Classes

The framework provides enhanced base classes that implement the standardized interfaces:

```python
from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents.enhanced_tools import EnhancedTool, ToolResult

# Create an enhanced agent
class MyAgent(EnhancedAgent):
    def decide(self) -> str:
        return "my_action"
    
    async def run_once(self):
        await self.perceive()
        action = self.decide()
        return await self.act(action)

# Create an enhanced tool
class MyTool(EnhancedTool):
    def __init__(self):
        super().__init__(name="my_tool", description="My custom tool")
    
    async def run(self, param: str) -> ToolResult:
        return ToolResult(success=True, data={"result": f"Processed: {param}"})
```

## Integration with Existing Components

The framework is designed to work with existing components through adapters:

```python
from agentor.core.interfaces.memory import MemoryProvider
from agentor.core.interfaces.llm import LLMProvider
from agentor.core.registry import get_component_registry

# Adapt an existing memory component
class LegacyMemoryAdapter(MemoryProvider):
    def __init__(self, legacy_memory):
        self.legacy_memory = legacy_memory
    
    async def add(self, item):
        return await self.legacy_memory.add_item(item)
    
    async def get(self, query, limit=10):
        return await self.legacy_memory.search(query, max_results=limit)
    
    async def clear(self):
        return await self.legacy_memory.clear_all()

# Register the adapter
registry = get_component_registry()
registry.register_memory_provider("legacy", LegacyMemoryAdapter(my_legacy_memory))
```

## Example: Creating a Plugin-Based Agent

Here's a complete example of creating a plugin-based agent:

```python
import asyncio
from agentor.core.plugin import Plugin, get_plugin_registry
from agentor.core.di import inject, get_container
from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents.enhanced_tools import EnhancedTool, ToolResult
from agentor.core.interfaces.tool import IToolRegistry

# Define a plugin
class GreetingPlugin(Plugin):
    @property
    def name(self) -> str:
        return "greeting"
    
    def get_greeting(self, name: str) -> str:
        return f"Hello, {name}!"

# Define a tool
class GreetingTool(EnhancedTool):
    def __init__(self, greeting_plugin):
        super().__init__(name="greeting", description="Generate a greeting")
        self.greeting_plugin = greeting_plugin
    
    async def run(self, name: str) -> ToolResult:
        greeting = self.greeting_plugin.get_greeting(name)
        return ToolResult(success=True, data={"greeting": greeting})

# Define an agent
class GreetingAgent(EnhancedAgent):
    def decide(self) -> str:
        return "generate_greeting"
    
    async def run_once(self):
        name = self.state.get("name", "World")
        if "greeting" in self.tools:
            result = await self.tools["greeting"].run(name=name)
            return result.data["greeting"]
        else:
            return f"Hello, {name}!"

# Main function
async def main():
    # Set up the framework
    registry = get_plugin_registry()
    container = get_container()
    
    # Register components
    greeting_plugin = GreetingPlugin()
    registry.register_plugin(greeting_plugin)
    
    tool_registry = get_container().resolve(IToolRegistry)
    greeting_tool = GreetingTool(greeting_plugin)
    tool_registry.register_tool(greeting_tool)
    
    greeting_agent = GreetingAgent(tool_registry=tool_registry)
    
    # Initialize the framework
    registry.initialize()
    
    # Run the agent
    result = await greeting_agent.run("Alice")
    print(result.response)  # Output: "Hello, Alice!"
    
    # Shutdown the framework
    registry.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```
