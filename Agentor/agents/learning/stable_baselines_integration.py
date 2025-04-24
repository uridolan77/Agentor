"""
Stable-Baselines3 integration for the Agentor framework.

This module provides integration with the Stable-Baselines3 library,
allowing the use of a wide range of reinforcement learning algorithms.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
import numpy as np
import random
import time
import logging
import asyncio
import os
import json
from collections import deque

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents.state_models import AgentStateModel
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.core.interfaces.environment import IEnvironment, Space

logger = logging.getLogger(__name__)

try:
    import gym
    import stable_baselines3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
    SB3_AVAILABLE = True
except ImportError:
    logger.warning("Stable-Baselines3 not available. StableBaselinesAgent will not work.")
    SB3_AVAILABLE = False


class GymEnvWrapper(gym.Env):
    """Wrapper to convert Agentor environments to Gym environments."""
    
    def __init__(self, env: IEnvironment):
        """Initialize the wrapper.
        
        Args:
            env: The Agentor environment to wrap
        """
        super().__init__()
        self.env = env
        
        # Set up observation and action spaces
        self.observation_space = self._convert_space(env.observation_space)
        self.action_space = self._convert_space(env.action_space)
    
    def _convert_space(self, space: Space) -> gym.Space:
        """Convert an Agentor space to a Gym space.
        
        Args:
            space: The Agentor space to convert
            
        Returns:
            The equivalent Gym space
        """
        from agentor.core.interfaces.environment import (
            DiscreteSpace, BoxSpace, DictSpace, TupleSpace
        )
        
        if isinstance(space, DiscreteSpace):
            return gym.spaces.Discrete(space.n)
        elif isinstance(space, BoxSpace):
            return gym.spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype
            )
        elif isinstance(space, DictSpace):
            return gym.spaces.Dict({
                k: self._convert_space(v) for k, v in space.spaces.items()
            })
        elif isinstance(space, TupleSpace):
            return gym.spaces.Tuple([
                self._convert_space(s) for s in space.spaces
            ])
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    
    def reset(self) -> np.ndarray:
        """Reset the environment.
        
        Returns:
            The initial observation
        """
        observation, _ = self.env.reset()
        return observation
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()


class StableBaselinesCallback(BaseCallback):
    """Callback for Stable-Baselines3 training."""
    
    def __init__(self, verbose: int = 0):
        """Initialize the callback.
        
        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        """Called after each step in the environment.
        
        Returns:
            Whether to continue training
        """
        # Update episode statistics
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Check if the episode is done
        if self.locals["dones"][0]:
            # Store episode statistics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log episode statistics
            if self.verbose > 0:
                logger.info(
                    f"Episode {len(self.episode_rewards)}: "
                    f"reward={self.current_episode_reward:.2f}, "
                    f"length={self.current_episode_length}"
                )
            
            # Reset episode statistics
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


class StableBaselinesAgent(EnhancedAgent):
    """Agent implementation using Stable-Baselines3 algorithms."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        environment: Optional[IEnvironment] = None,
        tool_registry: Optional[IToolRegistry] = None
    ):
        """Initialize the Stable-Baselines3 agent.
        
        Args:
            name: Name of the agent
            algorithm: Name of the Stable-Baselines3 algorithm to use
            policy: Name of the policy to use
            algorithm_kwargs: Additional arguments for the algorithm
            environment: The environment
            tool_registry: Tool registry for the agent
        """
        super().__init__(name=name, tool_registry=tool_registry)
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 is required for StableBaselinesAgent")
        
        self.algorithm_name = algorithm
        self.policy_name = policy
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.environment = environment
        
        # Initialize state model
        self.state_model = AgentStateModel()
        self.state_model.current_state = None
        self.state_model.episode_reward = 0.0
        self.state_model.episode_length = 0
        self.state_model.total_reward = 0.0
        self.state_model.total_steps = 0
        self.state_model.episode_rewards = []
        self.state_model.episode_lengths = []
        self.state_model.training_metrics = {}
        
        # Create the algorithm
        self.model = None
        if environment is not None:
            self._create_model()
        
        # Register actions
        if environment is not None:
            action_space = environment.action_space
            from agentor.core.interfaces.environment import DiscreteSpace
            
            if isinstance(action_space, DiscreteSpace):
                for i in range(action_space.n):
                    self.register_action(str(i), lambda i=i: self._take_action(i))
    
    def _create_model(self) -> None:
        """Create the Stable-Baselines3 model."""
        # Wrap the environment
        env = GymEnvWrapper(self.environment)
        vec_env = DummyVecEnv([lambda: env])
        
        # Get the algorithm class
        algorithm_class = getattr(stable_baselines3, self.algorithm_name)
        
        # Create the model
        self.model = algorithm_class(
            policy=self.policy_name,
            env=vec_env,
            **self.algorithm_kwargs
        )
    
    def _take_action(self, action: Any) -> Dict[str, Any]:
        """Take an action in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Result of the action
        """
        return {"action": action}
    
    def decide(self) -> str:
        """Choose an action using the Stable-Baselines3 model.
        
        Returns:
            The name of the action to take
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Get the current state
        state = self.state_model.current_state
        
        # Get observation from state
        if isinstance(state, dict) and "observation" in state:
            observation = state["observation"]
        else:
            observation = state
        
        # Predict action
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Convert to string
        if isinstance(action, np.ndarray):
            action = action.item()
        
        action_str = str(action)
        
        # Store the action in state model
        self.state_model.last_action = action_str
        
        return action_str
    
    async def perceive(self) -> Dict[str, Any]:
        """Perceive the environment.
        
        Returns:
            The current state
        """
        # If no environment is provided, return an empty state
        if self.environment is None:
            return {}
        
        # If the environment has not been reset, reset it
        if not hasattr(self, "_observation") or self._observation is None:
            observation, info = self.environment.reset()
            self._observation = observation
            self._info = info
        
        return {"observation": self._observation, "info": self._info, "done": False}
    
    async def act(self, action_name: str) -> Any:
        """Execute the specified action.
        
        Args:
            action_name: The name of the action to execute
            
        Returns:
            The result of the action
        """
        # If no environment is provided, use the registered action
        if self.environment is None:
            return await super().act(action_name)
        
        # Convert action name to appropriate type
        try:
            action = int(action_name)
        except ValueError:
            action = action_name
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.environment.step(action)
        
        # Update state model
        self.state_model.episode_reward += reward
        self.state_model.episode_length += 1
        self.state_model.total_reward += reward
        self.state_model.total_steps += 1
        
        # Store the observation
        self._observation = observation
        self._info = info
        
        # Check if the episode is done
        if terminated or truncated:
            # Store episode statistics
            self.state_model.episode_rewards.append(self.state_model.episode_reward)
            self.state_model.episode_lengths.append(self.state_model.episode_length)
            
            # Reset episode statistics
            self.state_model.episode_reward = 0.0
            self.state_model.episode_length = 0
            
            # Reset the environment
            observation, info = self.environment.reset()
            self._observation = observation
            self._info = info
        
        return {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    
    async def run_once(self) -> Any:
        """Run one perception-decision-action cycle.
        
        Returns:
            The result of the action
        """
        # Get the current state
        state = await self.perceive()
        self.state_model.current_state = state
        
        # Choose an action
        action = self.decide()
        
        # Execute the action
        result = await self.act(action)
        
        return result
    
    async def train(self, total_timesteps: int = 10000, progress_bar: bool = True) -> Dict[str, Any]:
        """Train the agent using the Stable-Baselines3 algorithm.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            progress_bar: Whether to display a progress bar
            
        Returns:
            Training statistics
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Create a callback
        callback = StableBaselinesCallback(verbose=1)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar
        )
        
        # Store statistics
        self.state_model.episode_rewards = callback.episode_rewards
        self.state_model.episode_lengths = callback.episode_lengths
        
        # Calculate mean rewards and lengths
        mean_reward = np.mean(callback.episode_rewards) if callback.episode_rewards else 0.0
        mean_length = np.mean(callback.episode_lengths) if callback.episode_lengths else 0.0
        
        return {
            "episode_rewards": callback.episode_rewards,
            "episode_lengths": callback.episode_lengths,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "total_timesteps": total_timesteps
        }
    
    async def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """Evaluate the agent in the environment.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Evaluation statistics
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        if self.environment is None:
            raise ValueError("No environment provided for evaluation")
        
        # Initialize statistics
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            # Reset the environment
            observation, info = self.environment.reset()
            
            # Initialize episode statistics
            episode_reward = 0.0
            episode_length = 0
            
            # Run the episode
            done = False
            while not done:
                # Predict action
                action, _ = self.model.predict(observation, deterministic=True)
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                
                # Render the environment if needed
                if render:
                    self.environment.render()
            
            # Store episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Log episode statistics
            logger.info(
                f"Evaluation Episode {episode+1}/{num_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}"
            )
        
        # Calculate mean rewards and lengths
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": mean_reward,
            "mean_length": mean_length
        }
    
    async def save_model(self, path: str) -> None:
        """Save the agent's model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.save(path)
    
    async def load_model(self, path: str) -> None:
        """Load the agent's model.
        
        Args:
            path: Path to load the model from
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Get the algorithm class
        algorithm_class = getattr(stable_baselines3, self.algorithm_name)
        
        # Load the model
        self.model = algorithm_class.load(path)
        
        # Update the environment
        if self.environment is not None:
            env = GymEnvWrapper(self.environment)
            vec_env = DummyVecEnv([lambda: env])
            self.model.set_env(vec_env)


class StableBaselinesFactory:
    """Factory for creating Stable-Baselines3 agents."""
    
    @staticmethod
    def create_agent(
        algorithm: str,
        environment: IEnvironment,
        policy: str = "MlpPolicy",
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> StableBaselinesAgent:
        """Create a Stable-Baselines3 agent.
        
        Args:
            algorithm: Name of the Stable-Baselines3 algorithm to use
            environment: The environment
            policy: Name of the policy to use
            algorithm_kwargs: Additional arguments for the algorithm
            name: Name of the agent
            
        Returns:
            The created agent
        """
        return StableBaselinesAgent(
            name=name or f"SB3_{algorithm}",
            algorithm=algorithm,
            policy=policy,
            algorithm_kwargs=algorithm_kwargs,
            environment=environment
        )
    
    @staticmethod
    def list_available_algorithms() -> List[str]:
        """List the available Stable-Baselines3 algorithms.
        
        Returns:
            List of available algorithm names
        """
        if not SB3_AVAILABLE:
            return []
        
        return [
            "A2C",
            "DDPG",
            "DQN",
            "PPO",
            "SAC",
            "TD3"
        ]
