"""
Example of using the environment interface in the Agentor framework.

This example demonstrates how to use the environment interface to train an agent
in a grid world environment.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from agentor.core.plugin import PluginManager
from agentor.core.registry import get_component_registry
from agentor.components.environments import GridWorldEnv, AgentEnvironmentLoop
from agentor.components.agents import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGridWorldAgent(BaseAgent):
    """Simple agent for the grid world environment."""
    
    def __init__(self, name: str = "simple_grid_world_agent"):
        """Initialize the simple grid world agent.
        
        Args:
            name: The name of the agent
        """
        super().__init__(name=name)
        self.state = {}
    
    def decide(self) -> str:
        """Decide what action to take.
        
        Returns:
            The name of the action to take
        """
        # Get the current state
        state = self.state.get("current_state", {})
        
        if not state:
            return "0"  # Default to moving up
        
        # Get the agent and goal positions
        agent_pos = state.get("agent_pos", np.array([0, 0]))
        goal_pos = state.get("goal_pos", np.array([4, 4]))
        
        # Calculate the direction to move
        dx = goal_pos[0] - agent_pos[0]
        dy = goal_pos[1] - agent_pos[1]
        
        # Choose the action based on the direction
        if abs(dx) > abs(dy):
            # Move horizontally
            if dx > 0:
                return "3"  # Right
            else:
                return "2"  # Left
        else:
            # Move vertically
            if dy > 0:
                return "1"  # Down
            else:
                return "0"  # Up
    
    async def act(self, action_name: str) -> Any:
        """Execute the specified action.
        
        Args:
            action_name: The name of the action to execute
            
        Returns:
            The result of the action
        """
        # In this simple agent, we don't need to do anything here
        return None


async def main():
    """Run the environment example."""
    # Initialize the plugin manager
    plugin_manager = PluginManager()
    await plugin_manager.initialize()
    
    # Create a grid world environment
    env = GridWorldEnv(width=5, height=5, obstacle_density=0.2)
    
    # Create a simple agent
    agent = SimpleGridWorldAgent()
    
    # Create the agent-environment loop
    loop = AgentEnvironmentLoop(
        env=env,
        agent=agent,
        max_episodes=10,
        render_interval=1
    )
    
    # Train the agent
    logger.info("Training the agent...")
    stats = await loop.train()
    
    # Print the training statistics
    logger.info(f"Training complete. Statistics: {stats}")
    
    # Evaluate the agent
    logger.info("Evaluating the agent...")
    eval_stats = await loop.evaluate(num_episodes=5, render=True)
    
    # Print the evaluation statistics
    logger.info(f"Evaluation complete. Statistics: {eval_stats}")
    
    # Clean up
    env.close()
    await plugin_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
