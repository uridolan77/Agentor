"""
Example demonstrating the enhanced learning agents and environments in Agentor.

This example shows how to use the enhanced learning agents with the new environment interface:
- EnhancedDeepQLearningAgent for reinforcement learning with neural networks
- EnhancedPPOAgent for policy-based reinforcement learning
- CartPoleEnv and MountainCarEnv environments
- GymnasiumAdapter for using any Gymnasium environment
"""

import asyncio
import logging
import time
import random
import numpy as np
import os
from typing import Dict, Any, List, Optional

from agentor.agents.learning import EnhancedDeepQLearningAgent, EnhancedPPOAgent
from agentor.components.environments import CartPoleEnv, MountainCarEnv, GymnasiumAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CartPoleDeepQLearningAgent(EnhancedDeepQLearningAgent):
    """Deep Q-Learning agent for the CartPole environment."""
    
    def __init__(self, **kwargs):
        """Initialize the agent.
        
        Args:
            **kwargs: Additional arguments for the base class
        """
        # CartPole has 4 state dimensions and 2 actions
        super().__init__(
            name="CartPoleDQL",
            state_dim=4,
            action_dim=2,
            **kwargs
        )
    
    def get_reward(
        self, 
        state: Dict[str, Any], 
        action: str, 
        next_state: Dict[str, Any],
        result: Any
    ) -> float:
        """Get the reward for an action.
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            result: The result of the action
            
        Returns:
            The reward
        """
        # Use the reward from the environment
        if isinstance(result, dict) and 'reward' in result:
            return result['reward']
        else:
            return 1.0  # Default reward for CartPole is 1.0 for each step
    
    def is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done.
        
        Args:
            state: The current state
            
        Returns:
            True if the episode is done, False otherwise
        """
        # Check if the episode is done
        if isinstance(state, dict) and 'done' in state:
            return state['done']
        else:
            return False


class CartPolePPOAgent(EnhancedPPOAgent):
    """PPO agent for the CartPole environment."""
    
    def __init__(self, **kwargs):
        """Initialize the agent.
        
        Args:
            **kwargs: Additional arguments for the base class
        """
        # CartPole has 4 state dimensions and 2 actions
        super().__init__(
            name="CartPolePPO",
            state_dim=4,
            action_dim=2,
            **kwargs
        )
    
    def get_reward(
        self, 
        state: Dict[str, Any], 
        action: str, 
        next_state: Dict[str, Any],
        result: Any
    ) -> float:
        """Get the reward for an action.
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            result: The result of the action
            
        Returns:
            The reward
        """
        # Use the reward from the environment
        if isinstance(result, dict) and 'reward' in result:
            return result['reward']
        else:
            return 1.0  # Default reward for CartPole is 1.0 for each step
    
    def is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done.
        
        Args:
            state: The current state
            
        Returns:
            True if the episode is done, False otherwise
        """
        # Check if the episode is done
        if isinstance(state, dict) and 'done' in state:
            return state['done']
        else:
            return False


class MountainCarDeepQLearningAgent(EnhancedDeepQLearningAgent):
    """Deep Q-Learning agent for the MountainCar environment."""
    
    def __init__(self, **kwargs):
        """Initialize the agent.
        
        Args:
            **kwargs: Additional arguments for the base class
        """
        # MountainCar has 2 state dimensions and 3 actions
        super().__init__(
            name="MountainCarDQL",
            state_dim=2,
            action_dim=3,
            **kwargs
        )
    
    def get_reward(
        self, 
        state: Dict[str, Any], 
        action: str, 
        next_state: Dict[str, Any],
        result: Any
    ) -> float:
        """Get the reward for an action.
        
        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action
            result: The result of the action
            
        Returns:
            The reward
        """
        # Use the reward from the environment
        if isinstance(result, dict) and 'reward' in result:
            return result['reward']
        else:
            # Default reward for MountainCar is -1.0 for each step
            # We'll add a bonus for reaching the goal
            if isinstance(next_state, dict) and 'observation' in next_state:
                position = next_state['observation'][0]
                if position >= 0.5:  # Goal position
                    return 0.0  # No penalty for reaching the goal
            
            return -1.0
    
    def is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done.
        
        Args:
            state: The current state
            
        Returns:
            True if the episode is done, False otherwise
        """
        # Check if the episode is done
        if isinstance(state, dict) and 'done' in state:
            return state['done']
        else:
            return False


async def train_cartpole_dql():
    """Train a Deep Q-Learning agent on the CartPole environment."""
    logger.info("\n=== Training Deep Q-Learning Agent on CartPole ===")
    
    # Create the environment
    env = CartPoleEnv(render_mode=None)
    
    # Create the agent
    agent = CartPoleDeepQLearningAgent(
        environment=env,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=0.1,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        batch_size=64,
        memory_size=10000,
        target_update_frequency=10,
        hidden_dim=128
    )
    
    # Train the agent
    stats = await agent.train_with_environment(
        num_episodes=100,
        max_steps=500,
        render_interval=None
    )
    
    logger.info(f"Training complete. Mean reward: {stats['mean_reward']:.2f}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    await agent.save_model("models/cartpole_dql.pt")
    
    return agent


async def train_cartpole_ppo():
    """Train a PPO agent on the CartPole environment."""
    logger.info("\n=== Training PPO Agent on CartPole ===")
    
    # Create the environment
    env = CartPoleEnv(render_mode=None)
    
    # Create the agent
    agent = CartPolePPOAgent(
        environment=env,
        learning_rate=0.0003,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=64,
        epochs=10,
        hidden_dim=128
    )
    
    # Train the agent
    stats = await agent.train_with_environment(
        num_episodes=100,
        max_steps=500,
        render_interval=None
    )
    
    logger.info(f"Training complete. Mean reward: {stats['mean_reward']:.2f}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    await agent.save_model("models/cartpole_ppo.pt")
    
    return agent


async def train_mountaincar_dql():
    """Train a Deep Q-Learning agent on the MountainCar environment."""
    logger.info("\n=== Training Deep Q-Learning Agent on MountainCar ===")
    
    # Create the environment
    env = MountainCarEnv(render_mode=None)
    
    # Create the agent
    agent = MountainCarDeepQLearningAgent(
        environment=env,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=0.1,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        batch_size=64,
        memory_size=10000,
        target_update_frequency=10,
        hidden_dim=128
    )
    
    # Train the agent
    stats = await agent.train_with_environment(
        num_episodes=200,
        max_steps=1000,
        render_interval=None
    )
    
    logger.info(f"Training complete. Mean reward: {stats['mean_reward']:.2f}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    await agent.save_model("models/mountaincar_dql.pt")
    
    return agent


async def train_custom_env():
    """Train an agent on a custom Gymnasium environment."""
    logger.info("\n=== Training Agent on Custom Gymnasium Environment ===")
    
    # Create the environment using the adapter
    env = GymnasiumAdapter("Acrobot-v1", render_mode=None)
    
    # Create the agent
    agent = EnhancedDeepQLearningAgent(
        name="AcrobotDQL",
        state_dim=6,  # Acrobot has 6 state dimensions
        action_dim=3,  # Acrobot has 3 actions
        environment=env,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=0.1,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        batch_size=64,
        memory_size=10000,
        target_update_frequency=10,
        hidden_dim=128
    )
    
    # Train the agent
    stats = await agent.train_with_environment(
        num_episodes=100,
        max_steps=500,
        render_interval=None
    )
    
    logger.info(f"Training complete. Mean reward: {stats['mean_reward']:.2f}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    await agent.save_model("models/acrobot_dql.pt")
    
    return agent


async def main():
    """Run the enhanced learning example."""
    # Train agents on different environments
    await train_cartpole_dql()
    await train_cartpole_ppo()
    await train_mountaincar_dql()
    await train_custom_env()


if __name__ == "__main__":
    asyncio.run(main())
