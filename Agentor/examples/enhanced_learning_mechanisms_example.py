"""
Example demonstrating the enhanced learning mechanisms in the Agentor framework.

This example shows how to use the various enhanced learning mechanisms:
- Multi-Agent Reinforcement Learning
- IMPALA (Importance Weighted Actor-Learner Architecture)
- Model-Based Reinforcement Learning
- Offline Reinforcement Learning
- Integration with Stable-Baselines3
"""

import asyncio
import logging
import numpy as np
import os
import time
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Agentor components
from agentor.components.environments.multi_agent import SimpleGridWorldMultiAgentEnv
from agentor.components.environments.gym_adapter import GymEnvironment
from agentor.agents.learning.multi_agent_rl import MultiAgentRLAgent
from agentor.agents.learning.impala import IMPALAAgent
from agentor.agents.learning.model_based_rl import ModelBasedRLAgent
from agentor.agents.learning.offline_rl import OfflineRLAgent
from agentor.agents.learning.stable_baselines_integration import StableBaselinesFactory


async def multi_agent_rl_example():
    """Example demonstrating Multi-Agent Reinforcement Learning."""
    logger.info("\n=== Multi-Agent Reinforcement Learning Example ===")
    
    # Create a multi-agent environment
    env = SimpleGridWorldMultiAgentEnv(
        num_agents=2,
        grid_size=5,
        max_episode_steps=50,
        competitive=True  # Agents compete for rewards
    )
    
    # Create a multi-agent RL agent
    agent = MultiAgentRLAgent(
        name="MARL_Agent",
        num_agents=2,
        state_dim=4,  # (x, y, goal_x, goal_y)
        action_dim=4,  # up, right, down, left
        hidden_dim=64,
        learning_rate=0.001,
        discount_factor=0.99,
        memory_size=10000,
        batch_size=64,
        update_frequency=4,
        environment=env,
        cooperative=False  # Agents compete rather than cooperate
    )
    
    # Train the agent
    logger.info("Training the multi-agent RL agent...")
    start_time = time.time()
    stats = await agent.train_with_environment(
        num_episodes=100,
        max_steps=50,
        render_interval=None
    )
    end_time = time.time()
    
    # Log training statistics
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Mean rewards: {stats['mean_rewards']}")
    logger.info(f"Mean episode length: {stats['mean_length']:.2f}")
    
    # Save the models
    os.makedirs("./models", exist_ok=True)
    await agent.save_models("./models/marl")
    logger.info("Models saved to ./models/marl")


async def impala_example():
    """Example demonstrating IMPALA (Importance Weighted Actor-Learner Architecture)."""
    logger.info("\n=== IMPALA Example ===")
    
    try:
        # Create a Gym environment
        env = GymEnvironment("CartPole-v1")
        
        # Create an IMPALA agent
        agent = IMPALAAgent(
            name="IMPALA_Agent",
            state_dim=4,  # CartPole has 4 state dimensions
            action_dim=2,  # CartPole has 2 actions
            hidden_dim=128,
            learning_rate=0.0003,
            discount_factor=0.99,
            entropy_coef=0.01,
            value_coef=0.5,
            num_actors=4,
            trajectory_length=20,
            batch_size=64,
            environment=env
        )
        
        # Train the agent
        logger.info("Training the IMPALA agent...")
        start_time = time.time()
        stats = await agent.train_with_environment(
            num_episodes=100,
            max_steps=500,
            render_interval=None
        )
        end_time = time.time()
        
        # Log training statistics
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Mean reward: {stats['mean_reward']:.2f}")
        logger.info(f"Mean episode length: {stats['mean_length']:.2f}")
        
        # Save the model
        os.makedirs("./models", exist_ok=True)
        await agent.save_model("./models/impala.pt")
        logger.info("Model saved to ./models/impala.pt")
    
    except ImportError as e:
        logger.error(f"Failed to run IMPALA example: {str(e)}")
        logger.error("Make sure PyTorch is installed")


async def model_based_rl_example():
    """Example demonstrating Model-Based Reinforcement Learning."""
    logger.info("\n=== Model-Based Reinforcement Learning Example ===")
    
    try:
        # Create a Gym environment
        env = GymEnvironment("CartPole-v1")
        
        # Create a Model-Based RL agent
        agent = ModelBasedRLAgent(
            name="MBRL_Agent",
            state_dim=4,  # CartPole has 4 state dimensions
            action_dim=2,  # CartPole has 2 actions
            hidden_dim=128,
            learning_rate=0.001,
            model_learning_rate=0.001,
            discount_factor=0.99,
            exploration_rate=0.1,
            exploration_decay=0.995,
            min_exploration_rate=0.01,
            batch_size=64,
            memory_size=10000,
            model_batch_size=128,
            model_epochs=10,
            planning_horizon=5,
            num_simulations=10,
            environment=env
        )
        
        # Train the agent
        logger.info("Training the Model-Based RL agent...")
        start_time = time.time()
        stats = await agent.train_with_environment(
            num_episodes=100,
            max_steps=500,
            render_interval=None
        )
        end_time = time.time()
        
        # Log training statistics
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Mean reward: {stats['mean_reward']:.2f}")
        logger.info(f"Mean episode length: {stats['mean_length']:.2f}")
        
        # Save the model
        os.makedirs("./models", exist_ok=True)
        await agent.save_model("./models/mbrl.pt")
        logger.info("Model saved to ./models/mbrl.pt")
    
    except ImportError as e:
        logger.error(f"Failed to run Model-Based RL example: {str(e)}")
        logger.error("Make sure PyTorch is installed")


async def offline_rl_example():
    """Example demonstrating Offline Reinforcement Learning."""
    logger.info("\n=== Offline Reinforcement Learning Example ===")
    
    try:
        # Create a Gym environment
        env = GymEnvironment("CartPole-v1")
        
        # Create an Offline RL agent
        agent = OfflineRLAgent(
            name="Offline_RL_Agent",
            state_dim=4,  # CartPole has 4 state dimensions
            action_dim=2,  # CartPole has 2 actions
            hidden_dim=128,
            learning_rate=0.0003,
            discount_factor=0.99,
            cql_alpha=1.0,
            batch_size=64,
            target_update_frequency=100,
            environment=env
        )
        
        # Collect data from the environment using a random policy
        logger.info("Collecting data from the environment...")
        await agent.collect_data_from_environment(
            num_episodes=50,
            policy=None,  # Use random actions
            exploration_rate=1.0
        )
        
        # Save the dataset
        os.makedirs("./data", exist_ok=True)
        await agent.save_dataset_to_file("./data/offline_rl_dataset.json")
        logger.info("Dataset saved to ./data/offline_rl_dataset.json")
        
        # Train the agent on the collected data
        logger.info("Training the Offline RL agent...")
        start_time = time.time()
        stats = await agent.train(num_iterations=1000)
        end_time = time.time()
        
        # Log training statistics
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Final loss: {stats['loss']:.6f}")
        logger.info(f"Final Q-loss: {stats['q_loss']:.6f}")
        logger.info(f"Final CQL-loss: {stats['cql_loss']:.6f}")
        
        # Evaluate the agent
        logger.info("Evaluating the Offline RL agent...")
        eval_stats = await agent.evaluate(num_episodes=10, render=False)
        
        # Log evaluation statistics
        logger.info(f"Evaluation mean reward: {eval_stats['mean_reward']:.2f}")
        logger.info(f"Evaluation mean episode length: {eval_stats['mean_length']:.2f}")
        
        # Save the model
        os.makedirs("./models", exist_ok=True)
        await agent.save_model("./models/offline_rl.pt")
        logger.info("Model saved to ./models/offline_rl.pt")
    
    except ImportError as e:
        logger.error(f"Failed to run Offline RL example: {str(e)}")
        logger.error("Make sure PyTorch is installed")


async def stable_baselines_example():
    """Example demonstrating integration with Stable-Baselines3."""
    logger.info("\n=== Stable-Baselines3 Integration Example ===")
    
    try:
        # List available algorithms
        algorithms = StableBaselinesFactory.list_available_algorithms()
        logger.info(f"Available algorithms: {algorithms}")
        
        # Create a Gym environment
        env = GymEnvironment("CartPole-v1")
        
        # Create a PPO agent
        agent = StableBaselinesFactory.create_agent(
            algorithm="PPO",
            environment=env,
            policy="MlpPolicy",
            algorithm_kwargs={
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            }
        )
        
        # Train the agent
        logger.info("Training the PPO agent...")
        start_time = time.time()
        stats = await agent.train(
            total_timesteps=50000,
            progress_bar=True
        )
        end_time = time.time()
        
        # Log training statistics
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Mean reward: {stats['mean_reward']:.2f}")
        logger.info(f"Mean episode length: {stats['mean_length']:.2f}")
        
        # Evaluate the agent
        logger.info("Evaluating the PPO agent...")
        eval_stats = await agent.evaluate(num_episodes=10, render=False)
        
        # Log evaluation statistics
        logger.info(f"Evaluation mean reward: {eval_stats['mean_reward']:.2f}")
        logger.info(f"Evaluation mean episode length: {eval_stats['mean_length']:.2f}")
        
        # Save the model
        os.makedirs("./models", exist_ok=True)
        await agent.save_model("./models/sb3_ppo")
        logger.info("Model saved to ./models/sb3_ppo")
    
    except ImportError as e:
        logger.error(f"Failed to run Stable-Baselines3 example: {str(e)}")
        logger.error("Make sure Stable-Baselines3 is installed")


async def main():
    """Run all examples."""
    # Run the Multi-Agent RL example
    await multi_agent_rl_example()
    
    # Run the IMPALA example
    await impala_example()
    
    # Run the Model-Based RL example
    await model_based_rl_example()
    
    # Run the Offline RL example
    await offline_rl_example()
    
    # Run the Stable-Baselines3 example
    await stable_baselines_example()


if __name__ == "__main__":
    asyncio.run(main())
