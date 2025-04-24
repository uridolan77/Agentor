"""
Base environment implementation for the Agentor framework.

This module provides a base implementation of the IEnvironment interface
and utility classes for working with environments.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, TypeVar, Generic, Callable
from abc import ABC, abstractmethod

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace, TextSpace
)

logger = logging.getLogger(__name__)


class BaseEnvironment(IEnvironment):
    """Base class for environment implementations."""
    
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True
    ):
        """Initialize the base environment.
        
        Args:
            observation_space: The observation space
            action_space: The action space
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self.max_episode_steps = max_episode_steps
        self.auto_reset = auto_reset
        
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.last_observation = None
        self.metadata = {}
    
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.steps = 0
        self.episodes += 1
        self.episode_reward = 0.0
        
        observation, info = self._reset_impl(options or {})
        self.last_observation = observation
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.last_observation is None:
            logger.warning("Environment not reset, calling reset() automatically")
            self.reset()
        
        # Check if the action is valid
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action: {action}, using a random action instead")
            action = self.action_space.sample()
        
        # Take a step in the environment
        observation, reward, terminated, info = self._step_impl(action)
        self.last_observation = observation
        
        # Update statistics
        self.steps += 1
        self.episode_reward += reward
        self.total_reward += reward
        
        # Check if the episode is truncated (reached max steps)
        truncated = False
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True
        
        # Auto-reset if needed
        if (terminated or truncated) and self.auto_reset:
            self.last_observation, reset_info = self.reset()
            info.update({"reset_info": reset_info})
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self._render_impl()
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        self._close_impl()
    
    @abstractmethod
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    @abstractmethod
    def _step_impl(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        pass
    
    def _render_impl(self) -> Optional[Union[np.ndarray, str]]:
        """Implementation of the render method.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return None
    
    def _close_impl(self) -> None:
        """Implementation of the close method."""
        pass


class TimeLimit:
    """Wrapper that limits the number of steps in an episode."""
    
    def __init__(self, env: IEnvironment, max_episode_steps: int):
        """Initialize the time limit wrapper.
        
        Args:
            env: The environment to wrap
            max_episode_steps: Maximum number of steps per episode
        """
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.steps = 0
    
    @property
    def observation_space(self) -> Space:
        """Get the observation space of the environment.
        
        Returns:
            The observation space
        """
        return self.env.observation_space
    
    @property
    def action_space(self) -> Space:
        """Get the action space of the environment.
        
        Returns:
            The action space
        """
        return self.env.action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        self.steps = 0
        return self.env.reset(seed, options)
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        
        # Check if the episode is truncated (reached max steps)
        if self.steps >= self.max_episode_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        self.env.close()


class Monitor:
    """Wrapper that monitors the environment and collects statistics."""
    
    def __init__(
        self, 
        env: IEnvironment, 
        log_interval: int = 10,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize the monitor wrapper.
        
        Args:
            env: The environment to wrap
            log_interval: Number of episodes between logging
            log_callback: Callback function for logging, or None to use the default logger
        """
        self.env = env
        self.log_interval = log_interval
        self.log_callback = log_callback or self._default_log_callback
        
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_start_time = None
    
    @property
    def observation_space(self) -> Space:
        """Get the observation space of the environment.
        
        Returns:
            The observation space
        """
        return self.env.observation_space
    
    @property
    def action_space(self) -> Space:
        """Get the action space of the environment.
        
        Returns:
            The action space
        """
        return self.env.action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        self.episode_start_time = time.time()
        self.episode_length = 0
        self.episode_reward = 0.0
        
        return self.env.reset(seed, options)
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_length += 1
        self.episode_reward += reward
        self.step_count += 1
        
        if terminated or truncated:
            self.episode_count += 1
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            
            # Calculate episode duration
            episode_duration = time.time() - self.episode_start_time
            
            # Log statistics
            if self.episode_count % self.log_interval == 0:
                stats = {
                    "episode": self.episode_count,
                    "steps": self.step_count,
                    "episode_reward": self.episode_reward,
                    "episode_length": self.episode_length,
                    "episode_duration": episode_duration,
                    "mean_reward": np.mean(self.episode_rewards[-self.log_interval:]),
                    "mean_length": np.mean(self.episode_lengths[-self.log_interval:])
                }
                
                self.log_callback(stats)
            
            # Add statistics to info
            info.update({
                "episode": {
                    "r": self.episode_reward,
                    "l": self.episode_length,
                    "t": episode_duration
                }
            })
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        self.env.close()
    
    def _default_log_callback(self, stats: Dict[str, Any]) -> None:
        """Default callback for logging statistics.
        
        Args:
            stats: Statistics to log
        """
        logger.info(
            f"Episode {stats['episode']} - "
            f"Steps: {stats['steps']}, "
            f"Reward: {stats['episode_reward']:.2f}, "
            f"Length: {stats['episode_length']}, "
            f"Duration: {stats['episode_duration']:.2f}s, "
            f"Mean Reward: {stats['mean_reward']:.2f}, "
            f"Mean Length: {stats['mean_length']:.2f}"
        )
