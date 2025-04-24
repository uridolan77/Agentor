"""
Hierarchical environments for the Agentor framework.

This module provides hierarchical environment implementations that support
nested environments, allowing for complex scenarios with multiple levels
of abstraction and decision-making.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import numpy as np

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class SubEnvironment:
    """A sub-environment within a hierarchical environment."""
    
    def __init__(
        self,
        env: IEnvironment,
        name: str,
        parent_to_sub_action: Optional[Callable[[Any], Any]] = None,
        sub_to_parent_observation: Optional[Callable[[Any], Any]] = None,
        reward_scale: float = 1.0,
        auto_reset: bool = True
    ):
        """Initialize the sub-environment.
        
        Args:
            env: The environment to wrap
            name: The name of the sub-environment
            parent_to_sub_action: Function to convert parent actions to sub-environment actions
            sub_to_parent_observation: Function to convert sub-environment observations to parent observations
            reward_scale: Scale factor for rewards from this sub-environment
            auto_reset: Whether to automatically reset the sub-environment when done
        """
        self.env = env
        self.name = name
        self.parent_to_sub_action = parent_to_sub_action or (lambda x: x)
        self.sub_to_parent_observation = sub_to_parent_observation or (lambda x: x)
        self.reward_scale = reward_scale
        self.auto_reset = auto_reset
        
        self.last_observation = None
        self.last_info = {}
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
    
    async def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the sub-environment.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(seed, options)
        self.last_observation = observation
        self.last_info = info
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
        
        return self.sub_to_parent_observation(observation), info
    
    async def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the sub-environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert the action
        sub_action = self.parent_to_sub_action(action)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(sub_action)
        
        # Update state
        self.last_observation = observation
        self.last_info = info
        self.terminated = terminated
        self.truncated = truncated
        self.total_reward += reward
        
        # Scale the reward
        scaled_reward = reward * self.reward_scale
        
        # Auto-reset if needed
        if (terminated or truncated) and self.auto_reset:
            self.last_observation, reset_info = self.env.reset()
            info.update({"reset_info": reset_info})
        
        # Convert the observation
        parent_observation = self.sub_to_parent_observation(observation)
        
        return parent_observation, scaled_reward, terminated, truncated, info
    
    @property
    def observation_space(self) -> Space:
        """Get the observation space of the sub-environment.
        
        Returns:
            The observation space
        """
        return self.env.observation_space
    
    @property
    def action_space(self) -> Space:
        """Get the action space of the sub-environment.
        
        Returns:
            The action space
        """
        return self.env.action_space
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the sub-environment.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the sub-environment and clean up resources."""
        self.env.close()


class HierarchicalEnvironment(BaseEnvironment):
    """Hierarchical environment that contains multiple sub-environments.
    
    This environment type is useful for complex scenarios with multiple levels
    of abstraction and decision-making, such as hierarchical reinforcement learning
    or multi-task learning.
    
    Examples:
        >>> # Create sub-environments
        >>> sub_env1 = SubEnvironment(GridWorldEnv(), "grid_world")
        >>> sub_env2 = SubEnvironment(CartPoleEnv(), "cart_pole")
        >>> 
        >>> # Create a hierarchical environment
        >>> env = HierarchicalEnvironment()
        >>> env.add_sub_environment(sub_env1)
        >>> env.add_sub_environment(sub_env2)
        >>> 
        >>> # Use the environment
        >>> obs, info = env.reset()
        >>> action = {"grid_world": 0, "cart_pole": 1}
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        termination_mode: str = "any"
    ):
        """Initialize the hierarchical environment.
        
        Args:
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
            termination_mode: How to determine termination ('any', 'all', or 'none')
        """
        # Initialize with empty spaces
        super().__init__(
            observation_space=DictSpace({}),
            action_space=DictSpace({}),
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        self.sub_environments: Dict[str, SubEnvironment] = {}
        self.termination_mode = termination_mode
        self.active_sub_environments: List[str] = []
    
    def add_sub_environment(self, sub_env: SubEnvironment) -> None:
        """Add a sub-environment.
        
        Args:
            sub_env: The sub-environment to add
        """
        if sub_env.name in self.sub_environments:
            logger.warning(f"Sub-environment {sub_env.name} already exists, replacing")
        
        self.sub_environments[sub_env.name] = sub_env
        self.active_sub_environments.append(sub_env.name)
        
        # Update the observation and action spaces
        self._update_spaces()
    
    def remove_sub_environment(self, name: str) -> None:
        """Remove a sub-environment.
        
        Args:
            name: The name of the sub-environment to remove
        """
        if name not in self.sub_environments:
            logger.warning(f"Sub-environment {name} does not exist")
            return
        
        # Close the sub-environment
        self.sub_environments[name].close()
        
        # Remove the sub-environment
        del self.sub_environments[name]
        
        if name in self.active_sub_environments:
            self.active_sub_environments.remove(name)
        
        # Update the observation and action spaces
        self._update_spaces()
    
    def activate_sub_environment(self, name: str) -> None:
        """Activate a sub-environment.
        
        Args:
            name: The name of the sub-environment to activate
        """
        if name not in self.sub_environments:
            logger.warning(f"Sub-environment {name} does not exist")
            return
        
        if name not in self.active_sub_environments:
            self.active_sub_environments.append(name)
            
            # Update the observation and action spaces
            self._update_spaces()
    
    def deactivate_sub_environment(self, name: str) -> None:
        """Deactivate a sub-environment.
        
        Args:
            name: The name of the sub-environment to deactivate
        """
        if name not in self.sub_environments:
            logger.warning(f"Sub-environment {name} does not exist")
            return
        
        if name in self.active_sub_environments:
            self.active_sub_environments.remove(name)
            
            # Update the observation and action spaces
            self._update_spaces()
    
    def _update_spaces(self) -> None:
        """Update the observation and action spaces."""
        # Create observation space
        observation_spaces = {}
        for name in self.active_sub_environments:
            sub_env = self.sub_environments[name]
            observation_spaces[name] = sub_env.observation_space
        
        self._observation_space = DictSpace(observation_spaces)
        
        # Create action space
        action_spaces = {}
        for name in self.active_sub_environments:
            sub_env = self.sub_environments[name]
            action_spaces[name] = sub_env.action_space
        
        self._action_space = DictSpace(action_spaces)
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observations = {}
        info = {}
        
        # Reset all active sub-environments
        for name in self.active_sub_environments:
            sub_env = self.sub_environments[name]
            sub_options = options.get(name, {})
            sub_observation, sub_info = sub_env.reset(options=sub_options)
            observations[name] = sub_observation
            info[name] = sub_info
        
        return observations, info
    
    def _step_impl(self, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        observations = {}
        rewards = {}
        terminated_flags = {}
        truncated_flags = {}
        infos = {}
        
        # Take a step in each active sub-environment
        for name in self.active_sub_environments:
            sub_env = self.sub_environments[name]
            
            # Get the action for this sub-environment
            sub_action = action.get(name)
            
            if sub_action is None:
                # If no action is provided, use a no-op action
                sub_action = sub_env.action_space.sample()
                logger.warning(f"No action provided for sub-environment {name}, using random action")
            
            # Take a step in the sub-environment
            sub_observation, sub_reward, sub_terminated, sub_truncated, sub_info = sub_env.step(sub_action)
            
            # Store the results
            observations[name] = sub_observation
            rewards[name] = sub_reward
            terminated_flags[name] = sub_terminated
            truncated_flags[name] = sub_truncated
            infos[name] = sub_info
        
        # Calculate the total reward
        total_reward = sum(rewards.values())
        
        # Determine termination based on the termination mode
        if self.termination_mode == "any":
            # Terminate if any sub-environment is terminated
            terminated = any(terminated_flags.values())
        elif self.termination_mode == "all":
            # Terminate if all sub-environments are terminated
            terminated = all(terminated_flags.values())
        else:  # "none"
            # Never terminate based on sub-environments
            terminated = False
        
        # Add aggregated info
        info = {
            "sub_environments": infos,
            "sub_rewards": rewards,
            "sub_terminated": terminated_flags,
            "sub_truncated": truncated_flags
        }
        
        return observations, total_reward, terminated, info
    
    def _render_impl(self) -> Optional[Union[np.ndarray, str]]:
        """Implementation of the render method.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        # Create a string representation of all sub-environments
        result = "Hierarchical Environment:\n"
        
        for name in self.active_sub_environments:
            sub_env = self.sub_environments[name]
            sub_render = sub_env.render()
            
            result += f"=== {name} ===\n"
            if isinstance(sub_render, str):
                result += sub_render
            elif sub_render is not None:
                result += f"<Rendered as non-string type: {type(sub_render)}>\n"
            else:
                result += "<Rendering not supported>\n"
            
            result += "\n"
        
        return result
    
    def _close_impl(self) -> None:
        """Implementation of the close method."""
        # Close all sub-environments
        for name, sub_env in self.sub_environments.items():
            sub_env.close()
        
        # Clear the sub-environments
        self.sub_environments.clear()
        self.active_sub_environments.clear()


class TaskHierarchicalEnvironment(HierarchicalEnvironment):
    """Hierarchical environment with task-based progression.
    
    This environment type is useful for scenarios where tasks need to be completed
    in a specific order or with dependencies between them.
    
    Examples:
        >>> # Create sub-environments for different tasks
        >>> task1 = SubEnvironment(GridWorldEnv(), "task1")
        >>> task2 = SubEnvironment(CartPoleEnv(), "task2")
        >>> 
        >>> # Create a task hierarchical environment
        >>> env = TaskHierarchicalEnvironment()
        >>> env.add_task(task1, dependencies=[])
        >>> env.add_task(task2, dependencies=["task1"])
        >>> 
        >>> # Use the environment
        >>> obs, info = env.reset()
        >>> action = {"task1": 0}  # Only task1 is active initially
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True
    ):
        """Initialize the task hierarchical environment.
        
        Args:
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
        """
        super().__init__(
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset,
            termination_mode="none"  # We'll handle termination ourselves
        )
        
        self.task_dependencies: Dict[str, List[str]] = {}
        self.completed_tasks: List[str] = []
        self.active_tasks: List[str] = []
    
    def add_task(self, sub_env: SubEnvironment, dependencies: List[str] = None) -> None:
        """Add a task (sub-environment) with dependencies.
        
        Args:
            sub_env: The sub-environment representing the task
            dependencies: List of task names that must be completed before this task
        """
        # Add the sub-environment
        self.add_sub_environment(sub_env)
        
        # Set dependencies
        self.task_dependencies[sub_env.name] = dependencies or []
        
        # Deactivate the task initially
        self.deactivate_sub_environment(sub_env.name)
        
        # Update active tasks
        self._update_active_tasks()
    
    def _update_active_tasks(self) -> None:
        """Update the list of active tasks based on dependencies."""
        self.active_tasks = []
        
        for task_name, dependencies in self.task_dependencies.items():
            # Skip completed tasks
            if task_name in self.completed_tasks:
                continue
            
            # Check if all dependencies are completed
            if all(dep in self.completed_tasks for dep in dependencies):
                self.active_tasks.append(task_name)
        
        # Activate/deactivate sub-environments based on active tasks
        for name in self.sub_environments:
            if name in self.active_tasks:
                self.activate_sub_environment(name)
            else:
                self.deactivate_sub_environment(name)
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset completed tasks
        self.completed_tasks = []
        
        # Update active tasks
        self._update_active_tasks()
        
        # Reset all sub-environments
        return super()._reset_impl(options)
    
    def _step_impl(self, action: Dict[str, Any]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        # Take a step in the active sub-environments
        observations, reward, _, info = super()._step_impl(action)
        
        # Check for completed tasks
        for name in self.active_tasks:
            if name not in self.completed_tasks:
                sub_env = self.sub_environments[name]
                if sub_env.terminated:
                    self.completed_tasks.append(name)
        
        # Update active tasks
        self._update_active_tasks()
        
        # Determine termination
        # The environment is terminated when all tasks are completed
        all_tasks_completed = len(self.completed_tasks) == len(self.task_dependencies)
        
        # Add task-specific info
        info.update({
            "completed_tasks": self.completed_tasks,
            "active_tasks": self.active_tasks,
            "all_tasks_completed": all_tasks_completed
        })
        
        return observations, reward, all_tasks_completed, info


class OptionsHierarchicalEnvironment(HierarchicalEnvironment):
    """Hierarchical environment with options (temporally extended actions).
    
    This environment type is useful for scenarios where agents can choose
    between different options (temporally extended actions) to achieve goals.
    
    Examples:
        >>> # Create sub-environments for different options
        >>> option1 = SubEnvironment(GridWorldEnv(), "option1")
        >>> option2 = SubEnvironment(CartPoleEnv(), "option2")
        >>> 
        >>> # Create an options hierarchical environment
        >>> env = OptionsHierarchicalEnvironment()
        >>> env.add_option(option1)
        >>> env.add_option(option2)
        >>> 
        >>> # Use the environment
        >>> obs, info = env.reset()
        >>> action = 0  # Select option1
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    def __init__(
        self,
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        option_termination_mode: str = "any"
    ):
        """Initialize the options hierarchical environment.
        
        Args:
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
            option_termination_mode: How to determine option termination ('any', 'all', or 'none')
        """
        super().__init__(
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset,
            termination_mode="none"  # We'll handle termination ourselves
        )
        
        self.option_termination_mode = option_termination_mode
        self.options: List[str] = []
        self.current_option: Optional[str] = None
        self.option_steps: int = 0
        
        # Override the action space to be a discrete space for option selection
        self._action_space = DiscreteSpace(0)  # Will be updated when options are added
    
    def add_option(self, sub_env: SubEnvironment) -> None:
        """Add an option (sub-environment).
        
        Args:
            sub_env: The sub-environment representing the option
        """
        # Add the sub-environment
        self.add_sub_environment(sub_env)
        
        # Add to options list
        self.options.append(sub_env.name)
        
        # Deactivate all options initially
        self.deactivate_sub_environment(sub_env.name)
        
        # Update the action space
        self._action_space = DiscreteSpace(len(self.options))
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset the current option
        self.current_option = None
        self.option_steps = 0
        
        # Reset all sub-environments
        observations = {}
        info = {}
        
        for name, sub_env in self.sub_environments.items():
            sub_options = options.get(name, {})
            sub_observation, sub_info = sub_env.reset(options=sub_options)
            observations[name] = sub_observation
            info[name] = sub_info
        
        return observations, info
    
    def _step_impl(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The option to select (index into self.options)
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        # Check if we need to select a new option
        if self.current_option is None or self._should_terminate_option():
            # Select a new option
            if action < 0 or action >= len(self.options):
                logger.warning(f"Invalid option index: {action}, using random option")
                action = np.random.randint(0, len(self.options))
            
            self.current_option = self.options[action]
            self.option_steps = 0
            
            # Activate the selected option and deactivate others
            for name in self.options:
                if name == self.current_option:
                    self.activate_sub_environment(name)
                else:
                    self.deactivate_sub_environment(name)
        
        # Increment option steps
        self.option_steps += 1
        
        # Take a step in the current option
        sub_env = self.sub_environments[self.current_option]
        
        # Use the action space of the current option to sample an action
        sub_action = sub_env.action_space.sample()
        
        # Take a step in the sub-environment
        sub_observation, sub_reward, sub_terminated, sub_truncated, sub_info = sub_env.step(sub_action)
        
        # Create observations and info for all options
        observations = {name: None for name in self.options}
        observations[self.current_option] = sub_observation
        
        info = {
            "current_option": self.current_option,
            "option_steps": self.option_steps,
            "option_terminated": sub_terminated,
            "option_truncated": sub_truncated,
            "option_info": sub_info
        }
        
        return observations, sub_reward, sub_terminated, info
    
    def _should_terminate_option(self) -> bool:
        """Check if the current option should be terminated.
        
        Returns:
            True if the option should be terminated
        """
        if self.current_option is None:
            return True
        
        sub_env = self.sub_environments[self.current_option]
        
        # Check termination based on the option termination mode
        if self.option_termination_mode == "any":
            # Terminate if the option is terminated or truncated
            return sub_env.terminated or sub_env.truncated
        elif self.option_termination_mode == "all":
            # Terminate if the option is both terminated and truncated
            return sub_env.terminated and sub_env.truncated
        else:  # "none"
            # Never terminate based on the option
            return False
