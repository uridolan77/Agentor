"""
Example demonstrating the improved tool system in Agentor.

This example shows how to use the new tool system improvements:
- Tool recommender for discovering appropriate tools
- Enhanced error handling with retry, fallback, and reporting
- Improved tool composition with better error recovery
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional

from agentor.agents.enhanced_tools import (
    EnhancedTool, 
    EnhancedToolRegistry, 
    ToolResult, 
    WeatherTool, 
    NewsTool, 
    CalculatorTool
)
from agentor.agents.tools.recommender import ToolRecommender
from agentor.agents.tools.error_handling import (
    ErrorHandlingToolRegistry,
    ToolErrorManager,
    RetryHandler,
    FallbackHandler,
    ErrorReportingHandler,
    ErrorCategory,
    ErrorSeverity,
    ToolErrorContext
)
from agentor.agents.composition import (
    ToolPipeline, 
    ComposableTool, 
    ParallelToolPipeline,
    ConditionalBranch, 
    ToolCondition, 
    ToolNode
)
from agentor.core.interfaces.tool import ITool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FailingTool(EnhancedTool):
    """A tool that fails in different ways for testing error handling."""
    
    def __init__(self):
        """Initialize the failing tool."""
        super().__init__(
            name="failing_tool",
            description="A tool that fails in different ways for testing",
            version="1.0.0"
        )
    
    async def run(self, failure_type: str = "random") -> ToolResult:
        """Run the tool with a specified failure type.
        
        Args:
            failure_type: The type of failure to simulate
            
        Returns:
            The result of running the tool
        """
        # Simulate different types of failures
        if failure_type == "random":
            failure_type = random.choice([
                "timeout", "connection", "validation", "internal", "none"
            ])
        
        logger.info(f"Running failing tool with failure type: {failure_type}")
        
        if failure_type == "timeout":
            # Simulate a timeout
            await asyncio.sleep(0.5)
            raise TimeoutError("Operation timed out")
        elif failure_type == "connection":
            # Simulate a connection error
            raise ConnectionError("Failed to connect to service")
        elif failure_type == "validation":
            # Simulate a validation error
            return ToolResult(
                success=False,
                error="Invalid input parameters"
            )
        elif failure_type == "internal":
            # Simulate an internal error
            raise RuntimeError("Internal server error")
        else:
            # No failure
            return ToolResult(
                success=True,
                data={"message": "Operation completed successfully"}
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        return {
            "type": "object",
            "properties": {
                "failure_type": {
                    "type": "string",
                    "enum": ["random", "timeout", "connection", "validation", "internal", "none"],
                    "description": "The type of failure to simulate"
                }
            }
        }


class EnhancedCalculatorTool(CalculatorTool):
    """Enhanced calculator tool with better error handling."""
    
    async def run(self, expression: str) -> ToolResult:
        """Perform a calculation with better error handling.
        
        Args:
            expression: The expression to evaluate
            
        Returns:
            The result of the calculation
        """
        try:
            # Try to evaluate the expression
            result = await super().run(expression)
            return result
        except Exception as e:
            # Try to provide a helpful error message
            if "division by zero" in str(e):
                return ToolResult(
                    success=False,
                    error="Division by zero is not allowed",
                    data={"expression": expression, "error_type": "division_by_zero"}
                )
            elif "invalid syntax" in str(e):
                return ToolResult(
                    success=False,
                    error="The expression has invalid syntax",
                    data={"expression": expression, "error_type": "syntax_error"}
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"Error evaluating expression: {str(e)}",
                    data={"expression": expression, "error_type": "unknown_error"}
                )


class ResilientPipeline(ToolPipeline):
    """A pipeline with enhanced error handling and recovery."""
    
    async def execute(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with enhanced error handling.
        
        Args:
            initial_data: The initial data to pass to the pipeline
            
        Returns:
            The final data after executing all tools in the pipeline
        """
        current_data = initial_data.copy()
        errors = []
        
        for i, node in enumerate(self.nodes):
            try:
                # Execute the node
                current_data = await node.execute(current_data)
            except Exception as e:
                # Log the error
                logger.error(f"Error executing node {i} ({node.tool.name}) in pipeline {self.name}: {str(e)}")
                
                # Record the error
                errors.append({
                    "node_index": i,
                    "tool_name": node.tool.name,
                    "error": str(e)
                })
                
                # Try to continue with the next node
                continue
        
        # Add error information to the result
        if errors:
            current_data["_pipeline_errors"] = errors
        
        return current_data


async def run_tool_recommender_demo():
    """Demonstrate the tool recommender."""
    logger.info("\n=== Tool Recommender Demo ===")
    
    # Create a tool registry
    registry = EnhancedToolRegistry()
    
    # Register some tools
    registry.register_tool(WeatherTool())
    registry.register_tool(NewsTool())
    registry.register_tool(CalculatorTool())
    registry.register_tool(FailingTool())
    
    # Create a tool recommender
    recommender = ToolRecommender(
        tool_registry=registry,
        use_semantic_matching=True,
        use_keyword_matching=True
    )
    
    # Test the recommender with different queries
    queries = [
        "What's the weather like in New York?",
        "Calculate 2 + 2 * 3",
        "Get the latest news about technology",
        "I need to test error handling",
        "Show me the temperature in London"
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        
        # Get recommendations
        result = await recommender.run(task=query)
        
        if result.success:
            logger.info(f"Recommended tools: {result.data['recommended_tools']}")
            
            # Show details for the top recommendation
            if result.data['recommendations']:
                top_rec = result.data['recommendations'][0]
                logger.info(f"Top recommendation: {top_rec['name']} (score: {top_rec['score']})")
                logger.info(f"Description: {top_rec['description']}")
        else:
            logger.error(f"Error getting recommendations: {result.error}")


async def run_error_handling_demo():
    """Demonstrate the enhanced error handling."""
    logger.info("\n=== Error Handling Demo ===")
    
    # Create a base tool registry
    base_registry = EnhancedToolRegistry()
    
    # Register tools
    base_registry.register_tool(CalculatorTool())
    base_registry.register_tool(FailingTool())
    
    # Create an error handling registry
    error_registry = ErrorHandlingToolRegistry(base_registry)
    
    # Test with different failure types
    failure_types = ["timeout", "connection", "validation", "internal", "none"]
    
    for failure_type in failure_types:
        logger.info(f"\nTesting with failure type: {failure_type}")
        
        # Get the failing tool
        failing_tool = error_registry.get_tool("failing_tool")
        
        # Run the tool
        start_time = time.time()
        result = await failing_tool.run(failure_type=failure_type)
        duration = time.time() - start_time
        
        # Log the result
        if result.success:
            logger.info(f"Success: {result.data}")
        else:
            logger.info(f"Error: {result.error}")
        
        logger.info(f"Duration: {duration:.2f} seconds")
    
    # Test calculator with error handling
    logger.info("\nTesting calculator with error handling")
    
    # Get the calculator tool
    calculator = error_registry.get_tool("calculator")
    
    # Test with valid and invalid expressions
    expressions = ["2 + 2", "10 / 0", "5 * (2 + 3)", "invalid expression"]
    
    for expression in expressions:
        logger.info(f"\nCalculating: {expression}")
        
        # Run the calculator
        result = await calculator.run(expression=expression)
        
        # Log the result
        if result.success:
            logger.info(f"Result: {result.data}")
        else:
            logger.info(f"Error: {result.error}")


async def run_resilient_pipeline_demo():
    """Demonstrate the resilient pipeline."""
    logger.info("\n=== Resilient Pipeline Demo ===")
    
    # Create tools
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()
    failing_tool = FailingTool()
    
    # Create a resilient pipeline
    pipeline = ResilientPipeline("resilient_pipeline", "Pipeline that continues despite errors")
    
    # Add tools to the pipeline
    pipeline.add_tool(
        weather_tool,
        input_mapping={"location": "city"},
        output_mapping={"temperature": "temp_f"}
    )
    
    pipeline.add_tool(
        failing_tool,
        input_mapping={"failure_type": "failure_type"}
    )
    
    pipeline.add_tool(
        calculator_tool,
        input_mapping={"expression": "calc_expression"},
        output_mapping={"result": "calc_result"}
    )
    
    # Test the pipeline with different configurations
    test_cases = [
        {
            "name": "All tools succeed",
            "data": {
                "city": "New York",
                "failure_type": "none",
                "calc_expression": "2 + 2"
            }
        },
        {
            "name": "Middle tool fails",
            "data": {
                "city": "London",
                "failure_type": "timeout",
                "calc_expression": "3 * 4"
            }
        },
        {
            "name": "Last tool fails",
            "data": {
                "city": "Paris",
                "failure_type": "none",
                "calc_expression": "10 / 0"
            }
        }
    ]
    
    for test_case in test_cases:
        logger.info(f"\nTest case: {test_case['name']}")
        
        # Execute the pipeline
        try:
            result = await pipeline.execute(test_case['data'])
            
            # Check for errors
            if "_pipeline_errors" in result:
                logger.info(f"Pipeline completed with errors: {result['_pipeline_errors']}")
                logger.info(f"Partial result: {result}")
            else:
                logger.info(f"Pipeline completed successfully: {result}")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")


async def run_enhanced_calculator_demo():
    """Demonstrate the enhanced calculator tool."""
    logger.info("\n=== Enhanced Calculator Demo ===")
    
    # Create the enhanced calculator
    calculator = EnhancedCalculatorTool()
    
    # Test with different expressions
    expressions = [
        "2 + 2",
        "10 / 0",
        "5 * (2 + 3)",
        "invalid expression",
        "10 ** 1000"  # Very large number
    ]
    
    for expression in expressions:
        logger.info(f"\nCalculating: {expression}")
        
        # Run the calculator
        result = await calculator.run(expression=expression)
        
        # Log the result
        if result.success:
            logger.info(f"Result: {result.data}")
        else:
            logger.info(f"Error: {result.error}")
            if result.data:
                logger.info(f"Error data: {result.data}")


async def main():
    """Run all the demos."""
    # Run the tool recommender demo
    await run_tool_recommender_demo()
    
    # Run the error handling demo
    await run_error_handling_demo()
    
    # Run the resilient pipeline demo
    await run_resilient_pipeline_demo()
    
    # Run the enhanced calculator demo
    await run_enhanced_calculator_demo()


if __name__ == "__main__":
    asyncio.run(main())
