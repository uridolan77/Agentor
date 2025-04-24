"""
Gymnasium adapter for the Agentor framework.

This module provides an adapter for using Gymnasium environments with the Agentor framework.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium.spaces import Discrete, Box, Dict as GymDict, Tuple as GymTuple
    GYM_AVAILABLE = True
except ImportError:
    logger.warning("Gymnasium not available. GymnasiumAdapter will not work.")
    GYM_AVAILABLE = False


class GymnasiumAdapter(BaseEnvironment):
    """Adapter for using Gymnasium environments with the Agentor framework."""
    
    def __init__(
        self,
        env_id: str,
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Gymnasium adapter.
        
        Args:
            env_id: The ID of the Gymnasium environment to create
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
            render_mode: The render mode to use, or None for no rendering
            **kwargs: Additional arguments to pass to the Gymnasium environment
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gymnasium is required for GymnasiumAdapter")
        
        # Create the Gymnasium environment
        self.gym_env = gym.make(env_id, render_mode=render_mode, **kwargs)
        
        # Convert the observation and action spaces
        observation_space = self._convert_space(self.gym_env.observation_space)
        action_space = self._convert_space(self.gym_env.action_space)
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        # Add metadata
        self.metadata = self.gym_env.metadata
    
    def _convert_space(self, gym_space: gym.Space) -> Space:
        """Convert a Gymnasium space to an Agentor space.
        
        Args:
            gym_space: The Gymnasium space to convert
            
        Returns:
            The converted Agentor space
        """
        if isinstance(gym_space, Discrete):
            return DiscreteSpace(gym_space.n)
        elif isinstance(gym_space, Box):
            return BoxSpace(
                low=gym_space.low,
                high=gym_space.high,
                shape=gym_space.shape,
                dtype=gym_space.dtype
            )
        elif isinstance(gym_space, GymDict):
            return DictSpace({
                key: self._convert_space(space)
                for key, space in gym_space.spaces.items()
            })
        elif isinstance(gym_space, GymTuple):
            return TupleSpace([
                self._convert_space(space)
                for space in gym_space.spaces
            ])
        else:
            raise ValueError(f"Unsupported Gymnasium space: {type(gym_space)}")
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.gym_env.reset(**options)
        return observation, info
    
    def _step_impl(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
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
