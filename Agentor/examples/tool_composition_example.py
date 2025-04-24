"""
Example demonstrating tool composition in Agentor.

This example shows how to compose tools into pipelines:
- Creating a sequential pipeline of tools
- Using conditional branching in pipelines
- Creating parallel tool pipelines
- Dynamic tool selection based on runtime data
"""

import asyncio
import logging
from typing import Dict, Any, List

from agentor.agents.enhanced_tools import (
    EnhancedTool, 
    EnhancedToolRegistry, 
    ToolResult, 
    WeatherTool, 
    NewsTool, 
    CalculatorTool
)
from agentor.agents.composition import (
    ToolPipeline, 
    ComposableTool, 
    ParallelToolPipeline,
    ConditionalBranch, 
    ToolCondition, 
    ToolNode, 
    DynamicToolPipeline
)
from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema
from agentor.core.interfaces.tool import ITool
from pydantic import Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define schemas for our composite tool
class WeatherNewsInput(ToolInputSchema):
    """Input schema for the weather and news combo tool."""
    location: str = Field(..., description="The location to get weather for")
    topics: List[str] = Field(..., description="Topics to get news for")


class WeatherNewsOutput(ToolOutputSchema):
    """Output schema for the weather and news combo tool."""
    location: str = Field(..., description="The location")
    temperature: float = Field(..., description="The temperature in Fahrenheit")
    conditions: str = Field(..., description="The weather conditions")
    topics: List[str] = Field(..., description="The topics")
    headlines: Dict[str, List[str]] = Field(..., description="News headlines by topic")


class WeatherNewsComboTool(ComposableTool):
    """Tool that combines weather and news information."""
    
    def __init__(self, weather_tool: WeatherTool, news_tool: NewsTool):
        """Initialize the weather news combo tool.
        
        Args:
            weather_tool: The weather tool
            news_tool: The news tool
        """
        # Create a pipeline that runs the weather and news tools for each topic
        pipeline = ParallelToolPipeline(
            name="weather_news_pipeline",
            description="Pipeline that gets weather and news in parallel"
        )
        
        # Add the weather tool to the pipeline
        pipeline.add_tool(
            weather_tool,
            input_mapping={"location": "location"}
        )
        
        super().__init__(
            name="weather_news_combo",
            description="Get weather and news information together",
            pipeline=pipeline,
            version="1.0.0",
            input_schema=WeatherNewsInput,
            output_schema=WeatherNewsOutput
        )
        
        # Store the tools for processing in the run method
        self.weather_tool = weather_tool
        self.news_tool = news_tool
    
    async def run(self, location: str, topics: List[str]) -> ToolResult:
        """Get weather and news information.
        
        Args:
            location: The location to get weather for
            topics: Topics to get news for
            
        Returns:
            Combined weather and news information
        """
        # Get weather information
        weather_result = await self.weather_tool.run(location=location)
        if not weather_result.success:
            return weather_result
        
        # Get news for each topic
        headlines = {}
        for topic in topics:
            news_result = await self.news_tool.run(topic=topic, count=3)
            if news_result.success:
                headlines[topic] = news_result.data["headlines"]
        
        # Build the output
        output_data = {
            "location": weather_result.data["location"],
            "temperature": weather_result.data["temperature"],
            "conditions": weather_result.data["conditions"],
            "topics": topics,
            "headlines": headlines
        }
        
        return ToolResult(success=True, data=output_data)


# Define schemas for a more complex composite tool
class TravelAdvisorInput(ToolInputSchema):
    """Input schema for the travel advisor tool."""
    location: str = Field(..., description="The travel destination")
    include_temperature_conversion: bool = Field(False, description="Whether to include temperature conversion")


class TravelAdvisorOutput(ToolOutputSchema):
    """Output schema for the travel advisor tool."""
    location: str = Field(..., description="The location")
    temperature_f: float = Field(..., description="The temperature in Fahrenheit")
    temperature_c: float = Field(None, description="The temperature in Celsius")
    conditions: str = Field(..., description="The weather conditions")
    headlines: List[str] = Field(..., description="Local news headlines")
    travel_advisory: str = Field(..., description="Travel advisory message")


class TravelAdvisorTool(ComposableTool):
    """Tool that provides travel advisories based on weather and news."""
    
    def __init__(self, weather_tool: WeatherTool, news_tool: NewsTool, calculator_tool: CalculatorTool):
        """Initialize the travel advisor tool.
        
        Args:
            weather_tool: The weather tool
            news_tool: The news tool
            calculator_tool: The calculator tool
        """
        # Create a sequential pipeline
        pipeline = ToolPipeline(
            name="travel_advisor_pipeline",
            description="Pipeline that gathers travel information"
        )
        
        # Add the weather tool to get weather information
        pipeline.add_tool(
            weather_tool,
            input_mapping={"location": "location"}
        )
        
        # Add the news tool to get local news
        pipeline.add_tool(
            news_tool,
            input_mapping={"topic": "location", "count": "count"},
            output_mapping={"headlines": "headlines"}
        )
        
        super().__init__(
            name="travel_advisor",
            description="Get travel advisories based on weather and news",
            pipeline=pipeline,
            version="1.0.0",
            input_schema=TravelAdvisorInput,
            output_schema=TravelAdvisorOutput
        )
        
        # Store the tools for use in the run method
        self.weather_tool = weather_tool
        self.news_tool = news_tool
        self.calculator_tool = calculator_tool
    
    async def run(self, location: str, include_temperature_conversion: bool = False) -> ToolResult:
        """Get travel advisory information.
        
        Args:
            location: The travel destination
            include_temperature_conversion: Whether to include temperature conversion
            
        Returns:
            Travel advisory information
        """
        # Get weather information
        weather_result = await self.weather_tool.run(location=location)
        if not weather_result.success:
            return ToolResult(
                success=False,
                error=f"Failed to get weather information: {weather_result.error}"
            )
        
        # Get local news
        news_result = await self.news_tool.run(topic=location, count=3)
        if not news_result.success:
            # We can continue without news
            headlines = [f"No news available for {location}"]
        else:
            headlines = news_result.data["headlines"]
        
        # Build output data
        output_data = {
            "location": weather_result.data["location"],
            "temperature_f": weather_result.data["temperature"],
            "conditions": weather_result.data["conditions"],
            "headlines": headlines
        }
        
        # Add temperature conversion if requested
        if include_temperature_conversion:
            try:
                # Convert Fahrenheit to Celsius: (F - 32) * 5/9
                calc_result = await self.calculator_tool.run(
                    expression=f"({weather_result.data['temperature']} - 32) * 5/9"
                )
                if calc_result.success:
                    output_data["temperature_c"] = round(calc_result.data["result"], 1)
            except Exception as e:
                logger.warning(f"Temperature conversion failed: {str(e)}")
        
        # Generate a travel advisory message based on weather and news
        advisory = "No specific advisories."
        temp = weather_result.data["temperature"]
        conditions = weather_result.data["conditions"].lower()
        
        if temp > 90:
            advisory = "Very hot conditions expected. Stay hydrated and seek shade."
        elif temp < 32:
            advisory = "Freezing conditions expected. Dress warmly and be cautious of ice."
        
        if "rain" in conditions or "storm" in conditions:
            advisory += " Expect precipitation; bring appropriate gear."
        elif "snow" in conditions:
            advisory += " Snow expected; check road conditions before traveling."
        
        # Add news-based advisories
        for headline in headlines:
            headline_lower = headline.lower()
            if any(word in headline_lower for word in ["warning", "alert", "emergency", "caution", "danger"]):
                advisory += f" Possible local concerns: {headline}"
                break
        
        output_data["travel_advisory"] = advisory
        
        return ToolResult(success=True, data=output_data)


async def run_tool_demo(tool: ITool, **kwargs):
    """Run a tool and display the results.
    
    Args:
        tool: The tool to run
        **kwargs: The parameters for the tool
    """
    logger.info(f"\n=== Running {tool.name} ===")
    logger.info(f"Input: {kwargs}")
    
    result = await tool.run(**kwargs)
    
    if result.success:
        logger.info(f"Success: {result.data}")
    else:
        logger.error(f"Error: {result.error}")
    
    logger.info("")


async def run_pipeline_demo():
    """Demonstrate using a manually created pipeline."""
    logger.info("\n=== Simple Pipeline Demo ===")
    
    # Create tools
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()
    
    # Create a pipeline that:
    # 1. Gets the weather
    # 2. Converts F to C
    pipeline = ToolPipeline("temp_converter", "Gets weather and converts temperature")
    
    # Add the weather tool
    pipeline.add_tool(
        weather_tool,
        input_mapping={"location_name": "location"},
        output_mapping={"temperature": "temp_f"}
    )
    
    # Add the calculator tool
    pipeline.add_tool(
        calculator_tool,
        input_mapping={"calc_input": "temp_f"},
        output_mapping={"result": "temp_c"}
    )
    
    # Execute the pipeline
    try:
        initial_data = {
            "location": "New York",
            "calc_input": "(temp_f - 32) * 5/9"
        }
        
        result = await pipeline.execute(initial_data)
        logger.info(f"Pipeline result: {result}")
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")


async def run_conditional_pipeline_demo():
    """Demonstrate using a conditional pipeline."""
    logger.info("\n=== Conditional Pipeline Demo ===")
    
    # Create tools
    weather_tool = WeatherTool()
    news_tool = NewsTool()
    
    # Create pipeline nodes
    weather_node = ToolNode(weather_tool, {"location": "location"})
    
    # Create a condition: if temperature is above 80F, get news about "heat wave"
    # otherwise get news about "weather"
    def check_temperature(data: Dict[str, Any]) -> bool:
        return data.get("temperature", 0) > 80
    
    condition = ToolCondition(check_temperature)
    
    # Create branches
    hot_news_node = ToolNode(
        news_tool, 
        {"heat_topic": "topic", "count": "count"},
        {"headlines": "hot_headlines"}
    )
    
    normal_news_node = ToolNode(
        news_tool, 
        {"normal_topic": "topic", "count": "count"},
        {"headlines": "normal_headlines"}
    )
    
    # Create a conditional branch
    branch = ConditionalBranch(condition, hot_news_node, normal_news_node)
    
    # Execute the nodes manually to demonstrate the concept
    try:
        # Initial data
        data = {
            "location": "Miami",
            "heat_topic": "heat wave",
            "normal_topic": "weather", 
            "count": 2
        }
        
        # First step: get weather
        data = await weather_node.execute(data)
        logger.info(f"Weather data: {data}")
        
        # Second step: choose appropriate news based on temperature
        data = await branch.execute(data)
        logger.info(f"Final data after conditional: {data}")
    except Exception as e:
        logger.error(f"Conditional pipeline error: {str(e)}")


async def main():
    """Run the tool composition example."""
    
    # Create instance of each tool we need
    weather_tool = WeatherTool()
    news_tool = NewsTool()
    calculator_tool = CalculatorTool()
    
    # Create our composite tools
    weather_news_tool = WeatherNewsComboTool(weather_tool, news_tool)
    travel_advisor_tool = TravelAdvisorTool(weather_tool, news_tool, calculator_tool)
    
    # Register our tools with the registry
    registry = EnhancedToolRegistry()
    registry.register_tool(weather_tool)
    registry.register_tool(news_tool)
    registry.register_tool(calculator_tool)
    registry.register_tool(weather_news_tool)
    registry.register_tool(travel_advisor_tool)
    
    # Demo 1: Run the weather news combo tool
    await run_tool_demo(
        weather_news_tool,
        location="Seattle",
        topics=["technology", "weather"]
    )
    
    # Demo 2: Run the travel advisor tool
    await run_tool_demo(
        travel_advisor_tool,
        location="Boston",
        include_temperature_conversion=True
    )
    
    # Demo 3: Run a manually created pipeline
    await run_pipeline_demo()
    
    # Demo 4: Run a conditional pipeline
    await run_conditional_pipeline_demo()


if __name__ == "__main__":
    asyncio.run(main())