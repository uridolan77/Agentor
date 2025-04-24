"""
Example demonstrating the reactive agent in Agentor.

This example shows how to use the reactive agent pattern:
- Creating a reactive agent with behaviors
- Using the @behavior decorator for defining behaviors
- Handling environment changes with reactive behaviors
- Prioritizing behaviors based on importance
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional

from agentor.agents.reactive import ReactiveAgent, behavior
from agentor.core.interfaces.tool import ITool, ToolResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartHomeAgent(ReactiveAgent):
    """A reactive agent for managing a smart home."""

    def __init__(self, name=None):
        super().__init__(name=name or "SmartHomeAgent")
        self._setup_sensors()
        self._setup_actions()

    def _setup_sensors(self):
        """Set up the sensors for this agent."""
        # Temperature sensor
        self.register_sensor('temperature', lambda a: random.randint(15, 35))
        
        # Light level sensor (0-100)
        self.register_sensor('light_level', lambda a: random.randint(0, 100))
        
        # Motion sensor (True/False)
        self.register_sensor('motion_detected', lambda a: random.choice([True, False]))
        
        # Door sensor (open/closed)
        self.register_sensor('door_status', lambda a: random.choice(['open', 'closed']))
        
        # Time of day (morning, afternoon, evening, night)
        self.register_sensor('time_of_day', lambda a: random.choice(['morning', 'afternoon', 'evening', 'night']))

    def _setup_actions(self):
        """Set up the actions for this agent."""
        self.register_action('adjust_temperature', self._adjust_temperature)
        self.register_action('adjust_lighting', self._adjust_lighting)
        self.register_action('secure_home', self._secure_home)
        self.register_action('welcome_home', self._welcome_home)
        self.register_action('energy_saving', self._energy_saving)
        self.register_action('do_nothing', lambda a: "No action needed")

    async def _adjust_temperature(self, target_temp=22):
        """Adjust the temperature in the home."""
        current_temp = self.state.get('last_perception', {}).get('temperature', 22)
        if current_temp > target_temp:
            return f"Cooling the home from {current_temp}°C to {target_temp}°C"
        elif current_temp < target_temp:
            return f"Heating the home from {current_temp}°C to {target_temp}°C"
        else:
            return f"Maintaining temperature at {target_temp}°C"

    async def _adjust_lighting(self):
        """Adjust the lighting in the home."""
        light_level = self.state.get('last_perception', {}).get('light_level', 50)
        time_of_day = self.state.get('last_perception', {}).get('time_of_day', 'day')
        
        if time_of_day in ['evening', 'night'] and light_level < 30:
            return "Turning on ambient lighting for evening/night"
        elif time_of_day in ['morning', 'afternoon'] and light_level < 20:
            return "Turning on lights for daytime use"
        else:
            return "Adjusting lighting to optimal levels"

    async def _secure_home(self):
        """Secure the home."""
        door_status = self.state.get('last_perception', {}).get('door_status', 'closed')
        time_of_day = self.state.get('last_perception', {}).get('time_of_day', 'day')
        
        actions = []
        if door_status == 'open':
            actions.append("Closing the door")
        
        if time_of_day == 'night':
            actions.append("Activating night security mode")
            actions.append("Turning on exterior lights")
        
        if not actions:
            return "Home is already secure"
        
        return ", ".join(actions)

    async def _welcome_home(self):
        """Welcome someone home."""
        return "Welcome home! Adjusting environment to your preferences"

    async def _energy_saving(self):
        """Activate energy saving mode."""
        return "Activating energy saving mode: reducing heating/cooling, dimming lights"

    # Define behaviors using the decorator
    @behavior(name="high_temperature", action="adjust_temperature", priority=3)
    def check_high_temperature(self):
        """Check if the temperature is too high."""
        temp = self.state.get('last_perception', {}).get('temperature', 22)
        return temp > 28

    @behavior(name="low_temperature", action="adjust_temperature", priority=3)
    def check_low_temperature(self):
        """Check if the temperature is too low."""
        temp = self.state.get('last_perception', {}).get('temperature', 22)
        return temp < 18

    @behavior(name="low_light", action="adjust_lighting", priority=2)
    def check_low_light(self):
        """Check if the light level is too low."""
        light = self.state.get('last_perception', {}).get('light_level', 50)
        time = self.state.get('last_perception', {}).get('time_of_day', 'day')
        return light < 30 and time in ['evening', 'night']

    @behavior(name="motion_detected", action="welcome_home", priority=4)
    def check_motion(self):
        """Check if motion is detected."""
        motion = self.state.get('last_perception', {}).get('motion_detected', False)
        door = self.state.get('last_perception', {}).get('door_status', 'closed')
        return motion and door == 'open'

    @behavior(name="night_security", action="secure_home", priority=5)
    def check_night_security(self):
        """Check if night security should be activated."""
        time = self.state.get('last_perception', {}).get('time_of_day', 'day')
        door = self.state.get('last_perception', {}).get('door_status', 'closed')
        return time == 'night' and door == 'open'

    @behavior(name="energy_saving", action="energy_saving", priority=1)
    def check_energy_saving(self):
        """Check if energy saving mode should be activated."""
        motion = self.state.get('last_perception', {}).get('motion_detected', False)
        time = self.state.get('last_perception', {}).get('time_of_day', 'day')
        return not motion and time in ['morning', 'afternoon']


class WeatherTool(ITool):
    """A tool for getting weather information."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get weather information for a location"

    async def run(self, location: str) -> ToolResult:
        """Get weather information for a location.
        
        Args:
            location: The location to get weather for
            
        Returns:
            Weather information
        """
        # Simulate getting weather data
        weather_data = {
            "temperature": random.randint(0, 35),
            "conditions": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
            "humidity": random.randint(30, 90),
            "wind_speed": random.randint(0, 30)
        }
        
        return ToolResult(
            success=True,
            data={
                "location": location,
                **weather_data
            }
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            },
            "required": ["location"]
        }


class WeatherAwareHomeAgent(SmartHomeAgent):
    """A reactive agent that uses weather information to manage a smart home."""

    def __init__(self, name=None):
        super().__init__(name=name or "WeatherAwareHomeAgent")
        
        # Register the weather tool
        self.weather_tool = WeatherTool()
        self.register_tool(self.weather_tool)
        
        # Add location to state
        self.state['location'] = "New York"
        
        # Add weather-aware behaviors
        self.add_behavior(
            name="rainy_weather",
            condition=self._check_rainy_weather,
            action="secure_home",
            priority=4
        )
        
        self.add_behavior(
            name="extreme_temperature",
            condition=self._check_extreme_weather,
            action="adjust_temperature",
            priority=5
        )

    async def _check_rainy_weather(self):
        """Check if the weather is rainy."""
        try:
            # Get the weather for the current location
            location = self.state.get('location', "New York")
            weather_result = await self.weather_tool.run(location=location)
            
            if weather_result.success:
                # Store the weather data in the state
                self.state['weather'] = weather_result.data
                
                # Check if it's rainy
                return weather_result.data.get('conditions') == 'rainy'
            
            return False
        except Exception as e:
            logger.error(f"Error checking weather: {str(e)}")
            return False

    async def _check_extreme_weather(self):
        """Check if the weather has extreme temperatures."""
        try:
            # Get the weather for the current location
            location = self.state.get('location', "New York")
            weather_result = await self.weather_tool.run(location=location)
            
            if weather_result.success:
                # Store the weather data in the state
                self.state['weather'] = weather_result.data
                
                # Check if temperature is extreme
                temp = weather_result.data.get('temperature', 22)
                return temp > 30 or temp < 5
            
            return False
        except Exception as e:
            logger.error(f"Error checking weather: {str(e)}")
            return False

    async def _adjust_temperature(self):
        """Adjust the temperature based on weather and indoor conditions."""
        indoor_temp = self.state.get('last_perception', {}).get('temperature', 22)
        
        # Check if we have weather data
        if 'weather' in self.state:
            outdoor_temp = self.state['weather'].get('temperature', 22)
            conditions = self.state['weather'].get('conditions', 'sunny')
            
            if outdoor_temp > 30:
                return f"Activating cooling: Indoor {indoor_temp}°C, Outdoor {outdoor_temp}°C"
            elif outdoor_temp < 5:
                return f"Activating heating: Indoor {indoor_temp}°C, Outdoor {outdoor_temp}°C"
            elif conditions == 'rainy':
                return f"Adjusting for rainy weather: Indoor {indoor_temp}°C, Outdoor {outdoor_temp}°C"
        
        # Fall back to basic temperature adjustment
        return await super()._adjust_temperature()


async def run_smart_home_agent():
    """Run the smart home agent example."""
    logger.info("=== Smart Home Agent Example ===")
    
    # Create the agent
    agent = SmartHomeAgent()
    
    # Run the agent for a few cycles
    for i in range(5):
        logger.info(f"\nCycle {i+1}:")
        
        # Perceive the environment
        perception = await agent.perceive()
        logger.info(f"Perception: {perception}")
        
        # Run the agent
        result = await agent.run_once()
        logger.info(f"Action: {result}")
        
        # Wait a bit
        await asyncio.sleep(1)


async def run_weather_aware_home_agent():
    """Run the weather-aware home agent example."""
    logger.info("\n=== Weather-Aware Home Agent Example ===")
    
    # Create the agent
    agent = WeatherAwareHomeAgent()
    
    # Run the agent for a few cycles
    for i in range(5):
        logger.info(f"\nCycle {i+1}:")
        
        # Perceive the environment
        perception = await agent.perceive()
        logger.info(f"Perception: {perception}")
        
        # Run the agent
        result = await agent.run_once()
        logger.info(f"Action: {result}")
        
        # Wait a bit
        await asyncio.sleep(1)


async def main():
    """Run the reactive agent examples."""
    # Run the smart home agent example
    await run_smart_home_agent()
    
    # Run the weather-aware home agent example
    await run_weather_aware_home_agent()


if __name__ == "__main__":
    asyncio.run(main())
