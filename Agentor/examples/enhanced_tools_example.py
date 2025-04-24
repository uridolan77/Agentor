"""
Complete example demonstrating all tool enhancements in Agentor.

This example shows how to use all of the tool enhancements together:
- Tool versioning and compatibility
- Tool composition with pipelines
- Authentication and authorization
"""

import asyncio
import logging
import os
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
    ConditionalBranch,
    ToolCondition,
    ToolNode
)
from agentor.agents.versioning import (
    SemanticVersion,
    VersionConstraint,
    VersionRange
)
from agentor.agents.tools.auth import (
    UserRole,
    User,
    ToolPermission,
    ToolAuditLogger,
    RateLimiter,
    AuthenticatedTool,
    AuthenticatedToolRegistry,
    generate_api_key
)
from agentor.agents.tool_schemas import ToolInputSchema, ToolOutputSchema
from pydantic import Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a newer version of the Weather tool with more features
class WeatherToolOutputV2(ToolOutputSchema):
    """Output schema for the weather tool v2."""
    location: str = Field(..., description="The location")
    temperature_f: float = Field(..., description="The temperature in Fahrenheit")
    temperature_c: float = Field(..., description="The temperature in Celsius")
    conditions: str = Field(..., description="The weather conditions")
    humidity: int = Field(..., description="The humidity percentage")
    wind_speed: float = Field(..., description="The wind speed in mph")
    forecast: List[Dict[str, Any]] = Field(..., description="Forecast for next 3 days")


class WeatherToolV2(WeatherTool):
    """Enhanced weather tool with more features."""

    def __init__(self):
        """Initialize the enhanced weather tool."""
        super().__init__()
        self._name = "weather"
        self._description = "Get detailed weather information for a location"
        self._version = "2.0.0"
        # Output schema is different, but input schema remains the same
        self._output_schema = WeatherToolOutputV2

    async def run(self, location: str) -> ToolResult:
        """Get weather information for a location.

        Args:
            location: The location to get weather for

        Returns:
            The weather information
        """
        # Get basic weather info from parent class
        base_result = await super().run(location)
        
        if not base_result.success:
            return base_result
            
        # Add the enhanced data
        base_data = base_result.data
        
        # Convert F to C
        temp_f = base_data["temperature"]
        temp_c = round((temp_f - 32) * 5/9, 1)
        
        # Create a mock forecast
        forecast = [
            {
                "day": "Tomorrow",
                "temp_high_f": temp_f + 2,
                "temp_low_f": temp_f - 5,
                "conditions": base_data["conditions"]
            },
            {
                "day": "Day after tomorrow",
                "temp_high_f": temp_f + 1,
                "temp_low_f": temp_f - 6,
                "conditions": "Partly cloudy"
            },
            {
                "day": "In two days",
                "temp_high_f": temp_f - 2,
                "temp_low_f": temp_f - 8,
                "conditions": "Mostly cloudy"
            }
        ]
        
        # Return enhanced data
        return ToolResult(
            success=True,
            data={
                "location": base_data["location"],
                "temperature_f": temp_f,
                "temperature_c": temp_c,
                "conditions": base_data["conditions"],
                "humidity": base_data["humidity"],
                "wind_speed": base_data["wind_speed"],
                "forecast": forecast
            }
        )


# Create a travel advisor tool that demonstrates tool composition with versioned dependencies
class TravelAdvisorToolV2(ComposableTool):
    """Enhanced travel advisor tool with versioned dependencies."""
    
    def __init__(
        self,
        weather_tool_v1: WeatherTool,
        weather_tool_v2: WeatherToolV2,
        news_tool: NewsTool,
        calculator_tool: CalculatorTool
    ):
        """Initialize the enhanced travel advisor."""
        # Create a pipeline that can conditionally use different weather tool versions
        pipeline = ToolPipeline(
            name="travel_advisor_pipeline_v2",
            description="Pipeline that gathers travel information with version selection"
        )
        
        # Create a condition that chooses weather v2 if detailed forecast is requested
        def should_use_detailed_forecast(data: Dict[str, Any]) -> bool:
            return data.get("detailed_forecast", False)
            
        condition = ToolCondition(should_use_detailed_forecast)
        
        # Create branches for different weather tools
        weather_v2_node = ToolNode(weather_tool_v2, {"location": "location"})
        weather_v1_node = ToolNode(weather_tool_v1, {"location": "location"})
        
        # Add conditional weather node to pipeline
        pipeline.add_tool(
            ConditionalBranch(condition, weather_v2_node, weather_v1_node),
        )
        
        # Add news tool
        pipeline.add_tool(
            news_tool,
            input_mapping={"topic": "location", "count": "news_count"},
            output_mapping={"headlines": "headlines"}
        )
        
        # Define tool dependencies based on versions
        dependencies = [
            f"weather:>=1.0.0,<3.0.0",  # Accept any weather tool v1.x or v2.x
            f"news:>=1.0.0",
            f"calculator:>=1.0.0"
        ]
        
        super().__init__(
            name="travel_advisor",
            description="Get travel advisories with optional detailed forecast",
            pipeline=pipeline,
            version="2.0.0",  # This is v2 of the travel advisor
            tool_dependencies=dependencies
        )
        
        self.weather_tool_v1 = weather_tool_v1
        self.weather_tool_v2 = weather_tool_v2
        self.news_tool = news_tool
        self.calculator_tool = calculator_tool
    
    async def run(self, location: str, detailed_forecast: bool = False, news_count: int = 3) -> ToolResult:
        """Get travel advisory with optional detailed forecast.
        
        Args:
            location: The travel destination
            detailed_forecast: Whether to include detailed forecast
            news_count: Number of news headlines to retrieve
            
        Returns:
            Travel advisory information
        """
        try:
            # Execute the pipeline
            result_data = await self.pipeline.execute({
                "location": location,
                "detailed_forecast": detailed_forecast,
                "news_count": news_count
            })
            
            # Generate a travel advisory message based on weather and news
            advisory = "No specific advisories."
            
            # Access temperature (handling both v1 and v2 formats)
            temp_f = result_data.get("temperature_f", result_data.get("temperature", 72))
            conditions = result_data.get("conditions", "").lower()
            
            # Generate advisory based on conditions
            if temp_f > 90:
                advisory = "Very hot conditions expected. Stay hydrated and seek shade."
            elif temp_f < 32:
                advisory = "Freezing conditions expected. Dress warmly and be cautious of ice."
            
            if "rain" in conditions or "storm" in conditions:
                advisory += " Expect precipitation; bring appropriate gear."
            elif "snow" in conditions:
                advisory += " Snow expected; check road conditions before traveling."
            
            # Add forecast summary if available
            if "forecast" in result_data:
                forecast_summary = []
                for day in result_data["forecast"]:
                    forecast_summary.append(f"{day['day']}: {day['conditions']}, High: {day['temp_high_f']}°F")
                advisory += f" Forecast: {' | '.join(forecast_summary)}"
                
            # Add news-based advisories
            headlines = result_data.get("headlines", [])
            for headline in headlines:
                headline_lower = headline.lower()
                if any(word in headline_lower for word in ["warning", "alert", "emergency"]):
                    advisory += f" Possible local concerns: {headline}"
                    break
                    
            result_data["travel_advisory"] = advisory
            
            return ToolResult(success=True, data=result_data)
        except Exception as e:
            return ToolResult(success=False, error=f"Travel advisor error: {str(e)}")


async def demo_versioning():
    """Demonstrate tool versioning features."""
    logger.info("\n=== Tool Versioning Demo ===")
    
    # Create tools with different versions
    weather_v1 = WeatherTool()  # v1.0.0
    weather_v2 = WeatherToolV2()  # v2.0.0
    
    # Create registry and register both versions
    registry = EnhancedToolRegistry()
    registry.register_tool(weather_v1)
    registry.register_tool(weather_v2)
    
    # Get tool by name (returns latest version)
    latest_weather = registry.get_tool("weather")
    logger.info(f"Latest weather tool version: {latest_weather.version}")
    
    # Get specific version
    v1_weather = registry.get_tool("weather", version="1.0.0")
    logger.info(f"v1 weather tool version: {v1_weather.version}")
    
    # Get tool with version constraint
    v1_x_weather = registry.get_tool("weather", version_constraint=">=1.0.0,<2.0.0")
    logger.info(f"v1.x weather tool version: {v1_x_weather.version}")
    
    v2_x_weather = registry.get_tool("weather", version_constraint=">=2.0.0,<3.0.0")
    logger.info(f"v2.x weather tool version: {v2_x_weather.version}")
    
    # Try to get non-existent version
    v3_weather = registry.get_tool("weather", version="3.0.0")
    logger.info(f"v3 weather tool exists: {v3_weather is not None}")
    
    # Run both versions
    result_v1 = await weather_v1.run(location="New York")
    result_v2 = await weather_v2.run(location="New York")
    
    logger.info(f"v1 output keys: {list(result_v1.data.keys())}")
    logger.info(f"v2 output keys: {list(result_v2.data.keys())}")


async def demo_auth():
    """Demonstrate authentication and authorization features."""
    logger.info("\n=== Authentication and Authorization Demo ===")
    
    # Create tools
    weather_tool = WeatherTool()
    news_tool = NewsTool()
    
    # Create users with different roles
    admin_api_key = generate_api_key()
    user_api_key = generate_api_key()
    guest_api_key = generate_api_key()
    
    admin_user = User(
        id="admin123",
        username="admin",
        roles=[UserRole.ADMIN],
        api_keys=[admin_api_key]
    )
    
    regular_user = User(
        id="user456",
        username="user",
        roles=[UserRole.USER],
        api_keys=[user_api_key]
    )
    
    guest_user = User(
        id="guest789",
        username="guest",
        roles=[UserRole.GUEST],
        api_keys=[guest_api_key]
    )
    
    # Create authenticated registry
    auth_registry = AuthenticatedToolRegistry()
    
    # Register users
    auth_registry.register_user(admin_user)
    auth_registry.register_user(regular_user)
    auth_registry.register_user(guest_user)
    
    # Register tools with permissions
    auth_registry.register_tool(
        weather_tool,
        ToolPermission(
            tool_name="weather",
            allowed_roles=[UserRole.ADMIN, UserRole.USER, UserRole.GUEST],
            requires_authentication=True,
            rate_limit=10
        )
    )
    
    auth_registry.register_tool(
        news_tool,
        ToolPermission(
            tool_name="news",
            allowed_roles=[UserRole.ADMIN, UserRole.USER],  # Not accessible to guests
            requires_authentication=True,
            rate_limit=5
        )
    )
    
    # Get user by API key
    user = auth_registry.get_user_by_api_key(user_api_key)
    logger.info(f"Found user: {user.username}")
    
    # Get accessible tools for user
    user_tools = auth_registry.get_tools(user)
    logger.info(f"User tools: {list(user_tools.keys())}")
    
    guest = auth_registry.get_user_by_api_key(guest_api_key)
    guest_tools = auth_registry.get_tools(guest)
    logger.info(f"Guest tools: {list(guest_tools.keys())}")
    
    # Run a tool with authentication
    weather_tool = auth_registry.get_tool("weather")
    result = await weather_tool.run(
        user=user,
        request_metadata={"client_ip": "127.0.0.1", "user_agent": "Example/1.0"},
        location="Seattle"
    )
    
    logger.info(f"Weather tool result success: {result.success}")
    
    # Try to access a tool the guest doesn't have permission for
    news_tool = auth_registry.get_tool("news")
    result = await news_tool.run(
        user=guest,
        request_metadata={"client_ip": "127.0.0.1", "user_agent": "Example/1.0"},
        topic="technology"
    )
    
    logger.info(f"News tool result success for guest: {result.success}")
    if not result.success:
        logger.info(f"Error message: {result.error}")
    
    # Get audit logs
    logs = auth_registry.get_audit_logs(limit=5)
    logger.info(f"Audit logs count: {len(logs)}")
    for log in logs:
        logger.info(f"Log: {log.tool_name} - {log.username} - {log.success}")


async def demo_full_integration():
    """Demonstrate all features working together."""
    logger.info("\n=== Full Integration Demo ===")
    
    # Create tools with different versions
    weather_v1 = WeatherTool()  # v1.0.0
    weather_v2 = WeatherToolV2()  # v2.0.0
    news_tool = NewsTool()
    calculator_tool = CalculatorTool()
    
    # Create composite tool that uses different versions
    travel_advisor = TravelAdvisorToolV2(
        weather_tool_v1=weather_v1,
        weather_tool_v2=weather_v2,
        news_tool=news_tool,
        calculator_tool=calculator_tool
    )
    
    # Create authenticated registry
    auth_registry = AuthenticatedToolRegistry()
    
    # Create users
    admin_api_key = generate_api_key()
    admin_user = User(
        id="admin123",
        username="admin",
        roles=[UserRole.ADMIN],
        api_keys=[admin_api_key]
    )
    
    # Register user
    auth_registry.register_user(admin_user)
    
    # Register all tools
    auth_registry.register_tool(weather_v1)
    auth_registry.register_tool(weather_v2)
    auth_registry.register_tool(news_tool)
    auth_registry.register_tool(calculator_tool)
    auth_registry.register_tool(
        travel_advisor,
        ToolPermission(
            tool_name="travel_advisor",
            allowed_roles=[UserRole.ADMIN],
            requires_authentication=True,
            rate_limit=5
        )
    )
    
    # Get the authenticated travel advisor tool
    auth_travel_advisor = auth_registry.get_tool("travel_advisor")
    
    # Run the tool with authentication
    logger.info("Running travel advisor with basic forecast...")
    result1 = await auth_travel_advisor.run(
        user=admin_user,
        request_metadata={"client_ip": "127.0.0.1", "user_agent": "Example/1.0"},
        location="Miami",
        detailed_forecast=False,
        news_count=2
    )
    
    if result1.success:
        logger.info(f"Weather: {result1.data.get('conditions', 'Unknown')}")
        logger.info(f"Advisory: {result1.data.get('travel_advisory', 'None')}")
        if "temperature_f" in result1.data:
            logger.info(f"Temperature: {result1.data['temperature_f']}°F")
        else:
            logger.info(f"Temperature: {result1.data.get('temperature', 'Unknown')}°F")
    else:
        logger.info(f"Error: {result1.error}")
    
    # Run with detailed forecast (uses weather v2)
    logger.info("\nRunning travel advisor with detailed forecast...")
    result2 = await auth_travel_advisor.run(
        user=admin_user,
        request_metadata={"client_ip": "127.0.0.1", "user_agent": "Example/1.0"},
        location="Miami",
        detailed_forecast=True,
        news_count=2
    )
    
    if result2.success:
        logger.info(f"Weather: {result2.data.get('conditions', 'Unknown')}")
        logger.info(f"Temperature: {result2.data.get('temperature_f', 'Unknown')}°F / {result2.data.get('temperature_c', 'Unknown')}°C")
        logger.info(f"Advisory: {result2.data.get('travel_advisory', 'None')}")
        if "forecast" in result2.data:
            logger.info(f"Number of forecast days: {len(result2.data['forecast'])}")
    else:
        logger.info(f"Error: {result2.error}")
    
    # Get audit logs
    logs = auth_registry.get_audit_logs(tool_name="travel_advisor", limit=2)
    logger.info(f"\nAudit logs for travel advisor: {len(logs)}")
    for log in logs:
        logger.info(f"Log: {log.timestamp} - User: {log.username} - Success: {log.success}")


async def main():
    """Run all the demonstration functions."""
    await demo_versioning()
    await demo_auth()
    await demo_full_integration()


if __name__ == "__main__":
    asyncio.run(main())
