"""
Grid World environment for the Agentor framework.

This module provides a simple grid world environment for testing agents.
"""

import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class GridWorldEnv(BaseEnvironment):
    """Simple grid world environment for testing agents."""
    
    def __init__(
        self, 
        width: int = 5, 
        height: int = 5,
        max_episode_steps: Optional[int] = 100,
        auto_reset: bool = True,
        obstacle_density: float = 0.1
    ):
        """Initialize the grid world environment.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
            obstacle_density: Density of obstacles in the grid
        """
        # Define the observation and action spaces
        observation_space = DictSpace({
            "agent_pos": BoxSpace(low=0, high=max(width, height), shape=(2,), dtype=np.int32),
            "goal_pos": BoxSpace(low=0, high=max(width, height), shape=(2,), dtype=np.int32),
            "grid": BoxSpace(low=0, high=1, shape=(height, width), dtype=np.int32)
        })
        
        # Define the action space (0: up, 1: down, 2: left, 3: right)
        action_space = DiscreteSpace(4)
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        
        # Initialize the grid
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.agent_pos = (0, 0)
        self.goal_pos = (width - 1, height - 1)
        self.obstacles = []
        
        # Add metadata
        self.metadata = {
            "render_modes": ["human", "rgb_array", "ansi"],
            "render_fps": 4
        }
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset the agent position
        self.agent_pos = (0, 0)
        
        # Reset the goal position
        if options.get("random_goal", False):
            while True:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if (x, y) != self.agent_pos:
                    self.goal_pos = (x, y)
                    break
        else:
            self.goal_pos = (self.width - 1, self.height - 1)
        
        # Reset the grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Reset obstacles
        self.obstacles = []
        
        # Add random obstacles
        num_obstacles = int(self.width * self.height * self.obstacle_density)
        for _ in range(num_obstacles):
            while True:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                
                # Don't put obstacles at the start or goal
                if (x, y) != self.agent_pos and (x, y) != self.goal_pos:
                    self.obstacles.append((x, y))
                    self.grid[y, x] = 1  # Mark as obstacle
                    break
        
        # Get the observation
        observation = self._get_observation()
        
        # Return the observation and info
        return observation, {}
    
    def _step_impl(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The action to take (0: up, 1: down, 2: left, 3: right)
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        x, y = self.agent_pos
        
        # Move the agent
        if action == 0 and y > 0:  # Up
            y -= 1
        elif action == 1 and y < self.height - 1:  # Down
            y += 1
        elif action == 2 and x > 0:  # Left
            x -= 1
        elif action == 3 and x < self.width - 1:  # Right
            x += 1
        
        # Check if the new position is valid
        if (x, y) not in self.obstacles:
            self.agent_pos = (x, y)
        
        # Check if the agent reached the goal
        terminated = self.agent_pos == self.goal_pos
        
        # Calculate the reward
        if terminated:
            reward = 1.0
        else:
            # Small negative reward for each step to encourage shorter paths
            reward = -0.01
            
            # Add a distance-based reward component
            goal_x, goal_y = self.goal_pos
            distance = abs(x - goal_x) + abs(y - goal_y)
            max_distance = self.width + self.height
            reward += 0.01 * (1 - distance / max_distance)
        
        # Get the observation
        observation = self._get_observation()
        
        # Return the observation, reward, terminated, and info
        return observation, reward, terminated, {}
    
    def _render_impl(self) -> str:
        """Implementation of the render method.
        
        Returns:
            The rendered environment as a string
        """
        # Create a grid for rendering
        grid = np.zeros((self.height, self.width), dtype=str)
        grid.fill('.')
        
        # Add obstacles
        for x, y in self.obstacles:
            grid[y, x] = '#'
        
        # Add the goal
        goal_x, goal_y = self.goal_pos
        grid[goal_y, goal_x] = 'G'
        
        # Add the agent
        agent_x, agent_y = self.agent_pos
        grid[agent_y, agent_x] = 'A'
        
        # Convert the grid to a string
        result = ""
        for row in grid:
            result += ' '.join(row) + '\n'
        
        return result
    
    def _close_impl(self) -> None:
        """Implementation of the close method."""
        pass
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation.
        
        Returns:
            The current observation
        """
        return {
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "goal_pos": np.array(self.goal_pos, dtype=np.int32),
            "grid": self.grid.copy()
        }
