"""
Training utilities for environments in the Agentor framework.

This module provides utilities for training agents in environments.
"""

import logging
import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from agentor.core.interfaces.environment import IEnvironment
from agentor.core.interfaces.agent import IAgent, AgentInput, AgentOutput
from agentor.components.environments.base import Monitor

logger = logging.getLogger(__name__)


class AgentEnvironmentLoop:
    """Loop for training agents in environments."""
    
    def __init__(
        self,
        env: IEnvironment,
        agent: IAgent,
        max_episodes: int = 100,
        max_steps_per_episode: Optional[int] = None,
        render_interval: Optional[int] = None,
        log_interval: int = 10,
        log_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        state_adapter: Optional[Callable[[Any], Dict[str, Any]]] = None,
        action_adapter: Optional[Callable[[str], Any]] = None,
        reward_adapter: Optional[Callable[[float], float]] = None
    ):
        """Initialize the agent-environment loop.
        
        Args:
            env: The environment to train in
            agent: The agent to train
            max_episodes: Maximum number of episodes to train for
            max_steps_per_episode: Maximum number of steps per episode, or None to use the environment's limit
            render_interval: Number of episodes between rendering, or None to disable rendering
            log_interval: Number of episodes between logging
            log_callback: Callback function for logging, or None to use the default logger
            state_adapter: Function to convert environment observations to agent states, or None to use the default
            action_adapter: Function to convert agent actions to environment actions, or None to use the default
            reward_adapter: Function to convert environment rewards to agent rewards, or None to use the default
        """
        self.env = Monitor(env, log_interval, log_callback)
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.render_interval = render_interval
        
        # Set up adapters
        self.state_adapter = state_adapter or self._default_state_adapter
        self.action_adapter = action_adapter or self._default_action_adapter
        self.reward_adapter = reward_adapter or self._default_reward_adapter
        
        # Initialize statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
    
    async def train(self) -> Dict[str, Any]:
        """Train the agent in the environment.
        
        Returns:
            Training statistics
        """
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            # Reset the environment
            observation, info = self.env.reset()
            
            # Convert the observation to a state
            state = self.state_adapter(observation)
            
            # Set the agent's state
            self.agent.state['current_state'] = state
            
            # Initialize episode statistics
            episode_reward = 0.0
            episode_length = 0
            
            # Run the episode
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Check if we've reached the maximum steps for this episode
                if self.max_steps_per_episode is not None and episode_length >= self.max_steps_per_episode:
                    break
                
                # Choose an action
                action_name = self.agent.decide()
                
                # Convert the action to an environment action
                action = self.action_adapter(action_name)
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Convert the reward
                adjusted_reward = self.reward_adapter(reward)
                
                # Convert the observation to a state
                next_state = self.state_adapter(observation)
                
                # Update the agent's state
                self.agent.state['current_state'] = next_state
                
                # Store the transition
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(state, action_name, adjusted_reward, next_state, terminated or truncated)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
                
                # Update the state
                state = next_state
                
                # Train the agent if it has a train method
                if hasattr(self.agent, 'train') and episode_length % 10 == 0:
                    await self.agent.train()
                
                # Render the environment if needed
                if self.render_interval is not None and episode % self.render_interval == 0:
                    self.env.render()
            
            # End the episode
            if hasattr(self.agent, 'end_episode'):
                self.agent.end_episode()
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
        
        # Calculate training duration
        duration = time.time() - start_time
        
        # Return training statistics
        return {
            "episodes": self.max_episodes,
            "total_steps": self.total_steps,
            "mean_reward": np.mean(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "duration": duration
        }
    
    async def evaluate(self, num_episodes: int = 10, render: bool = True) -> Dict[str, Any]:
        """Evaluate the agent in the environment.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            
        Returns:
            Evaluation statistics
        """
        # Initialize statistics
        rewards = []
        lengths = []
        
        for episode in range(num_episodes):
            # Reset the environment
            observation, info = self.env.reset()
            
            # Convert the observation to a state
            state = self.state_adapter(observation)
            
            # Set the agent's state
            self.agent.state['current_state'] = state
            
            # Initialize episode statistics
            episode_reward = 0.0
            episode_length = 0
            
            # Run the episode
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Choose an action
                action_name = self.agent.decide()
                
                # Convert the action to an environment action
                action = self.action_adapter(action_name)
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Convert the observation to a state
                next_state = self.state_adapter(observation)
                
                # Update the agent's state
                self.agent.state['current_state'] = next_state
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                
                # Update the state
                state = next_state
                
                # Render the environment if needed
                if render:
                    self.env.render()
            
            # Store episode statistics
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
            logger.info(
                f"Evaluation Episode {episode+1}/{num_episodes}: "
                f"Reward: {episode_reward:.2f}, "
                f"Length: {episode_length}"
            )
        
        # Return evaluation statistics
        return {
            "episodes": num_episodes,
            "mean_reward": np.mean(rewards),
            "mean_length": np.mean(lengths),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "rewards": rewards,
            "lengths": lengths
        }
    
    def _default_state_adapter(self, observation: Any) -> Dict[str, Any]:
        """Default adapter for converting observations to states.
        
        Args:
            observation: The observation from the environment
            
        Returns:
            The state for the agent
        """
        if isinstance(observation, dict):
            return observation
        else:
            return {"observation": observation}
    
    def _default_action_adapter(self, action_name: str) -> Any:
        """Default adapter for converting agent actions to environment actions.
        
        Args:
            action_name: The action name from the agent
            
        Returns:
            The action for the environment
        """
        # Try to convert the action name to an integer
        try:
            return int(action_name)
        except ValueError:
            # If that fails, return the action name as is
            return action_name
    
    def _default_reward_adapter(self, reward: float) -> float:
        """Default adapter for converting environment rewards to agent rewards.
        
        Args:
            reward: The reward from the environment
            
        Returns:
            The reward for the agent
        """
        return reward
