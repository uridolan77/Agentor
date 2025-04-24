"""
Multi-Agent Environment interfaces and implementations for the Agentor framework.

This module provides interfaces and implementations for multi-agent environments,
allowing multiple agents to interact in the same environment.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, TypeVar, Generic, Callable
from abc import ABC, abstractmethod

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class IMultiAgentEnvironment(IEnvironment):
    """Interface for multi-agent environments."""
    
    @property
    @abstractmethod
    def num_agents(self) -> int:
        """Get the number of agents in the environment.
        
        Returns:
            The number of agents
        """
        pass
    
    @property
    @abstractmethod
    def agent_ids(self) -> List[str]:
        """Get the IDs of all agents in the environment.
        
        Returns:
            List of agent IDs
        """
        pass
    
    @property
    @abstractmethod
    def observation_spaces(self) -> Dict[str, Space]:
        """Get the observation spaces for all agents.
        
        Returns:
            Dictionary mapping agent IDs to observation spaces
        """
        pass
    
    @property
    @abstractmethod
    def action_spaces(self) -> Dict[str, Space]:
        """Get the action spaces for all agents.
        
        Returns:
            Dictionary mapping agent IDs to action spaces
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observations, info)
            - observations: Dictionary mapping agent IDs to observations
            - info: Dictionary of additional information
        """
        pass
    
    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Take a step in the environment with actions from all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
            - observations: Dictionary mapping agent IDs to observations
            - rewards: Dictionary mapping agent IDs to rewards
            - terminated: Dictionary mapping agent IDs to termination flags
            - truncated: Dictionary mapping agent IDs to truncation flags
            - info: Dictionary of additional information
        """
        pass


class MultiAgentEnv(BaseEnvironment, IMultiAgentEnvironment):
    """Base class for multi-agent environment implementations."""
    
    def __init__(
        self,
        agent_ids: List[str],
        observation_spaces: Dict[str, Space],
        action_spaces: Dict[str, Space],
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        render_mode: Optional[str] = None
    ):
        """Initialize the multi-agent environment.
        
        Args:
            agent_ids: List of agent IDs
            observation_spaces: Dictionary mapping agent IDs to observation spaces
            action_spaces: Dictionary mapping agent IDs to action spaces
            max_episode_steps: Maximum number of steps per episode
            auto_reset: Whether to automatically reset the environment when done
            render_mode: Rendering mode (None, 'human', 'rgb_array')
        """
        super().__init__(
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset,
            render_mode=render_mode
        )
        
        self._agent_ids = agent_ids
        self._observation_spaces = observation_spaces
        self._action_spaces = action_spaces
        
        # For compatibility with the IEnvironment interface
        # Use the first agent's spaces as the default
        self._observation_space = observation_spaces[agent_ids[0]]
        self._action_space = action_spaces[agent_ids[0]]
        
        # Initialize state
        self._observations = {agent_id: None for agent_id in agent_ids}
        self._rewards = {agent_id: 0.0 for agent_id in agent_ids}
        self._terminated = {agent_id: False for agent_id in agent_ids}
        self._truncated = {agent_id: False for agent_id in agent_ids}
        self._info = {}
    
    @property
    def num_agents(self) -> int:
        """Get the number of agents in the environment.
        
        Returns:
            The number of agents
        """
        return len(self._agent_ids)
    
    @property
    def agent_ids(self) -> List[str]:
        """Get the IDs of all agents in the environment.
        
        Returns:
            List of agent IDs
        """
        return self._agent_ids
    
    @property
    def observation_spaces(self) -> Dict[str, Space]:
        """Get the observation spaces for all agents.
        
        Returns:
            Dictionary mapping agent IDs to observation spaces
        """
        return self._observation_spaces
    
    @property
    def action_spaces(self) -> Dict[str, Space]:
        """Get the action spaces for all agents.
        
        Returns:
            Dictionary mapping agent IDs to action spaces
        """
        return self._action_spaces
    
    @property
    def observation_space(self) -> Space:
        """Get the observation space of the environment.
        
        Returns:
            The observation space
        """
        return self._observation_space
    
    @property
    def action_space(self) -> Space:
        """Get the action space of the environment.
        
        Returns:
            The action space
        """
        return self._action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observations, info)
            - observations: Dictionary mapping agent IDs to observations
            - info: Dictionary of additional information
        """
        # Reset the step counter
        self._step_count = 0
        
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the environment
        observations, info = self._reset_impl(seed, options)
        
        # Store the observations and info
        self._observations = observations
        self._info = info
        
        # Reset rewards and done flags
        self._rewards = {agent_id: 0.0 for agent_id in self._agent_ids}
        self._terminated = {agent_id: False for agent_id in self._agent_ids}
        self._truncated = {agent_id: False for agent_id in self._agent_ids}
        
        return observations, info
    
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Take a step in the environment with actions from all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
            - observations: Dictionary mapping agent IDs to observations
            - rewards: Dictionary mapping agent IDs to rewards
            - terminated: Dictionary mapping agent IDs to termination flags
            - truncated: Dictionary mapping agent IDs to truncation flags
            - info: Dictionary of additional information
        """
        # Check if the environment needs to be reset
        if self._needs_reset:
            logger.warning("Environment needs to be reset. Calling reset().")
            self.reset()
        
        # Increment the step counter
        self._step_count += 1
        
        # Take a step in the environment
        observations, rewards, terminated, truncated, info = self._step_impl(actions)
        
        # Store the observations, rewards, done flags, and info
        self._observations = observations
        self._rewards = rewards
        self._terminated = terminated
        self._truncated = truncated
        self._info = info
        
        # Check if the episode is done
        all_terminated = all(terminated.values())
        any_truncated = any(truncated.values())
        
        # Check if the episode is truncated due to max steps
        if self._max_episode_steps is not None and self._step_count >= self._max_episode_steps:
            truncated = {agent_id: True for agent_id in self._agent_ids}
            any_truncated = True
        
        # Set the needs_reset flag if the episode is done
        self._needs_reset = all_terminated or any_truncated
        
        # Auto-reset if needed
        if self._needs_reset and self._auto_reset:
            self.reset()
        
        return observations, rewards, terminated, truncated, info
    
    @abstractmethod
    def _reset_impl(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observations, info)
        """
        pass
    
    @abstractmethod
    def _step_impl(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        pass


class SimpleGridWorldMultiAgentEnv(MultiAgentEnv):
    """Simple grid world environment for multi-agent reinforcement learning."""
    
    def __init__(
        self,
        num_agents: int = 2,
        grid_size: int = 10,
        max_episode_steps: Optional[int] = 100,
        auto_reset: bool = True,
        render_mode: Optional[str] = None,
        competitive: bool = False,
        shared_reward: bool = False
    ):
        """Initialize the simple grid world environment.
        
        Args:
            num_agents: Number of agents in the environment
            grid_size: Size of the grid (grid_size x grid_size)
            max_episode_steps: Maximum number of steps per episode
            auto_reset: Whether to automatically reset the environment when done
            render_mode: Rendering mode (None, 'human', 'rgb_array')
            competitive: Whether agents compete for rewards
            shared_reward: Whether agents share rewards
        """
        # Create agent IDs
        agent_ids = [f"agent_{i}" for i in range(num_agents)]
        
        # Create observation and action spaces
        observation_spaces = {}
        action_spaces = {}
        
        for agent_id in agent_ids:
            # Observation space: (x, y, goal_x, goal_y)
            observation_spaces[agent_id] = BoxSpace(
                low=np.array([0, 0, 0, 0]),
                high=np.array([grid_size-1, grid_size-1, grid_size-1, grid_size-1]),
                shape=(4,),
                dtype=np.float32
            )
            
            # Action space: 0=up, 1=right, 2=down, 3=left
            action_spaces[agent_id] = DiscreteSpace(4)
        
        super().__init__(
            agent_ids=agent_ids,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset,
            render_mode=render_mode
        )
        
        self.grid_size = grid_size
        self.competitive = competitive
        self.shared_reward = shared_reward
        
        # Initialize positions and goals
        self.agent_positions = {}
        self.agent_goals = {}
        self.agent_reached_goal = {}
    
    def _reset_impl(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observations, info)
        """
        # Initialize agent positions randomly
        self.agent_positions = {}
        self.agent_goals = {}
        self.agent_reached_goal = {}
        
        # Place agents and goals randomly
        available_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        
        for agent_id in self._agent_ids:
            # Place agent
            if available_positions:
                pos_idx = np.random.randint(0, len(available_positions))
                self.agent_positions[agent_id] = available_positions.pop(pos_idx)
            else:
                # If no positions left, place randomly (may overlap)
                self.agent_positions[agent_id] = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
            
            # Place goal
            if available_positions:
                goal_idx = np.random.randint(0, len(available_positions))
                self.agent_goals[agent_id] = available_positions.pop(goal_idx)
            else:
                # If no positions left, place randomly (may overlap)
                self.agent_goals[agent_id] = (
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size)
                )
            
            # Initialize reached_goal flag
            self.agent_reached_goal[agent_id] = False
        
        # Create observations
        observations = {}
        for agent_id in self._agent_ids:
            pos = self.agent_positions[agent_id]
            goal = self.agent_goals[agent_id]
            observations[agent_id] = np.array([pos[0], pos[1], goal[0], goal[1]], dtype=np.float32)
        
        return observations, {}
    
    def _step_impl(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Process actions for each agent
        for agent_id, action in actions.items():
            if agent_id not in self._agent_ids:
                continue
            
            # Skip if agent has already reached its goal
            if self.agent_reached_goal[agent_id]:
                continue
            
            # Get current position
            x, y = self.agent_positions[agent_id]
            
            # Process action
            if action == 0:  # Up
                y = max(0, y - 1)
            elif action == 1:  # Right
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # Down
                y = min(self.grid_size - 1, y + 1)
            elif action == 3:  # Left
                x = max(0, x - 1)
            
            # Update position
            self.agent_positions[agent_id] = (x, y)
            
            # Check if agent reached its goal
            if self.agent_positions[agent_id] == self.agent_goals[agent_id]:
                self.agent_reached_goal[agent_id] = True
        
        # Calculate rewards
        rewards = {}
        for agent_id in self._agent_ids:
            # Calculate distance to goal
            pos = self.agent_positions[agent_id]
            goal = self.agent_goals[agent_id]
            distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            
            # Base reward is negative distance to goal
            reward = -distance / (2 * self.grid_size)
            
            # Bonus for reaching goal
            if self.agent_reached_goal[agent_id]:
                reward += 1.0
            
            # In competitive mode, penalize agents for others reaching goals
            if self.competitive:
                for other_id in self._agent_ids:
                    if other_id != agent_id and self.agent_reached_goal[other_id]:
                        reward -= 0.5
            
            rewards[agent_id] = reward
        
        # In shared reward mode, all agents get the average reward
        if self.shared_reward:
            avg_reward = sum(rewards.values()) / len(rewards)
            rewards = {agent_id: avg_reward for agent_id in self._agent_ids}
        
        # Create observations
        observations = {}
        for agent_id in self._agent_ids:
            pos = self.agent_positions[agent_id]
            goal = self.agent_goals[agent_id]
            observations[agent_id] = np.array([pos[0], pos[1], goal[0], goal[1]], dtype=np.float32)
        
        # Check if all agents have reached their goals
        terminated = {agent_id: self.agent_reached_goal[agent_id] for agent_id in self._agent_ids}
        truncated = {agent_id: False for agent_id in self._agent_ids}
        
        return observations, rewards, terminated, truncated, {}
    
    def _render_impl(self) -> Optional[Union[np.ndarray, str]]:
        """Implementation of the render method.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        if self._render_mode is None:
            return None
        
        # Create a grid for rendering
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Fill the grid with a background color
        grid.fill(200)  # Light gray
        
        # Draw goals
        for agent_id, goal in self.agent_goals.items():
            agent_idx = int(agent_id.split('_')[1])
            color = self._get_agent_color(agent_idx)
            grid[goal[1], goal[0]] = color
        
        # Draw agents
        for agent_id, pos in self.agent_positions.items():
            agent_idx = int(agent_id.split('_')[1])
            color = self._get_agent_color(agent_idx)
            # Make agent brighter than goal
            color = np.minimum(color + 100, 255)
            grid[pos[1], pos[0]] = color
        
        if self._render_mode == 'human':
            # Display the grid using matplotlib
            try:
                import matplotlib.pyplot as plt
                plt.imshow(grid)
                plt.show(block=False)
                plt.pause(0.1)
                return None
            except ImportError:
                logger.warning("Matplotlib not available. Cannot render in 'human' mode.")
                return None
        
        return grid
    
    def _get_agent_color(self, agent_idx: int) -> np.ndarray:
        """Get a color for an agent based on its index.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            RGB color array
        """
        colors = [
            [255, 0, 0],    # Red
            [0, 0, 255],    # Blue
            [0, 255, 0],    # Green
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]
        
        return np.array(colors[agent_idx % len(colors)], dtype=np.uint8)
