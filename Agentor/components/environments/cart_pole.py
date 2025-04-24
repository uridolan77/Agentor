"""
CartPole environment for the Agentor framework.

This module provides a CartPole environment implementation based on the Gymnasium CartPole environment.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    logger.warning("Gymnasium not available. CartPoleEnv will not work.")
    GYM_AVAILABLE = False


class CartPoleEnv(BaseEnvironment):
    """CartPole environment based on the Gymnasium CartPole environment."""
    
    def __init__(
        self,
        max_episode_steps: Optional[int] = 500,
        auto_reset: bool = True,
        render_mode: Optional[str] = None
    ):
        """Initialize the CartPole environment.
        
        Args:
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
            render_mode: The render mode to use, or None for no rendering
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium is required for CartPoleEnv")
        
        # Create the Gymnasium environment
        self.gym_env = gym.make("CartPole-v1", render_mode=render_mode)
        
        # Define the observation and action spaces
        observation_space = BoxSpace(
            low=self.gym_env.observation_space.low,
            high=self.gym_env.observation_space.high,
            shape=self.gym_env.observation_space.shape,
            dtype=np.float32
        )
        
        action_space = DiscreteSpace(self.gym_env.action_space.n)
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        # Add metadata
        self.metadata = self.gym_env.metadata
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.gym_env.reset(**options)
        return observation, info
    
    def _step_impl(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        observation, reward, terminated, truncated, info = self.gym_env.step(action)
        return observation, reward, terminated, info
    
    def _render_impl(self) -> Optional[Union[np.ndarray, str]]:
        """Implementation of the render method.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self.gym_env.render()
    
    def _close_impl(self) -> None:
        """Implementation of the close method."""
        self.gym_env.close()
