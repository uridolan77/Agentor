"""
Example demonstrating the enhanced learning capabilities in Agentor.

This example shows how to use the different learning algorithms:
- Deep Q-Learning for reinforcement learning with neural networks
- Proximal Policy Optimization (PPO) for policy-based reinforcement learning
- Transfer learning for sharing knowledge between agents
"""

import asyncio
import logging
import time
import random
import numpy as np
import os
from typing import Dict, Any, List, Optional

# Import from the new location
from agentor.agents.learning.deep_q_learning import EnhancedDeepQLearningAgent as DeepQLearningAgent
from agentor.agents.learning.ppo_agent import EnhancedPPOAgent as PPOAgent
from agentor.components.learning.transfer_learning import TransferLearningManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridWorldEnv:
    """Simple grid world environment for testing learning agents."""

    def __init__(self, width: int = 5, height: int = 5):
        """Initialize the grid world.

        Args:
            width: Width of the grid
            height: Height of the grid
        """
        self.width = width
        self.height = height
        self.agent_pos = (0, 0)
        self.goal_pos = (width - 1, height - 1)
        self.obstacles = []

        # Add some random obstacles
        num_obstacles = (width * height) // 10
        for _ in range(num_obstacles):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            # Don't put obstacles at the start or goal
            if (x, y) != self.agent_pos and (x, y) != self.goal_pos:
                self.obstacles.append((x, y))

    def reset(self) -> Dict[str, Any]:
        """Reset the environment.

        Returns:
            The initial state
        """
        self.agent_pos = (0, 0)
        return self.get_state()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool]:
        """Take a step in the environment.

        Args:
            action: The action to take ('up', 'down', 'left', 'right')

        Returns:
            Tuple of (next_state, reward, done)
        """
        x, y = self.agent_pos

        # Move the agent
        if action == 'up' and y > 0:
            y -= 1
        elif action == 'down' and y < self.height - 1:
            y += 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < self.width - 1:
            x += 1

        # Check if the new position is valid
        if (x, y) not in self.obstacles:
            self.agent_pos = (x, y)

        # Check if the agent reached the goal
        done = self.agent_pos == self.goal_pos

        # Calculate reward
        if done:
            reward = 10.0  # High reward for reaching the goal
        else:
            # Small negative reward for each step to encourage efficiency
            reward = -0.1

            # Add a distance-based reward component
            goal_x, goal_y = self.goal_pos
            distance = abs(x - goal_x) + abs(y - goal_y)
            max_distance = self.width + self.height

            # Normalize distance to [0, 1] and invert
            distance_reward = 1.0 - (distance / max_distance)

            # Scale the distance reward
            reward += distance_reward * 0.5

        return self.get_state(), reward, done

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment.

        Returns:
            The current state
        """
        return {
            'agent_pos': self.agent_pos,
            'goal_pos': self.goal_pos,
            'obstacles': self.obstacles,
            'width': self.width,
            'height': self.height
        }

    def render(self):
        """Render the grid world."""
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == self.agent_pos:
                    print('A', end=' ')
                elif (x, y) == self.goal_pos:
                    print('G', end=' ')
                elif (x, y) in self.obstacles:
                    print('X', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()


class GridWorldDQNAgent(DeepQLearningAgent):
    """Deep Q-Learning agent for the grid world environment."""

    def __init__(self, env: GridWorldEnv, **kwargs):
        """Initialize the agent.

        Args:
            env: The grid world environment
            **kwargs: Additional arguments for the base class
        """
        # Set up the state and action dimensions
        state_dim = 4  # Agent x, y, goal x, y
        action_dim = 4  # up, down, left, right

        super().__init__(
            name="GridWorldDQN",
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )

        self.env = env

        # Register actions
        self.register_action('up', self.move_up)
        self.register_action('down', self.move_down)
        self.register_action('left', self.move_left)
        self.register_action('right', self.move_right)

        # Set up the state encoder
        self.set_state_encoder(self.encode_grid_state)

    def encode_grid_state(self, state: Dict[str, Any]) -> np.ndarray:
        """Encode the grid world state as a vector.

        Args:
            state: The grid world state

        Returns:
            A vector representation of the state
        """
        agent_x, agent_y = state['agent_pos']
        goal_x, goal_y = state['goal_pos']

        # Normalize positions to [0, 1]
        agent_x_norm = agent_x / (self.env.width - 1)
        agent_y_norm = agent_y / (self.env.height - 1)
        goal_x_norm = goal_x / (self.env.width - 1)
        goal_y_norm = goal_y / (self.env.height - 1)

        return np.array([agent_x_norm, agent_y_norm, goal_x_norm, goal_y_norm], dtype=np.float32)

    async def perceive(self) -> Dict[str, Any]:
        """Perceive the environment.

        Returns:
            The current state
        """
        return self.env.get_state()

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
        # Unpack the result
        _, reward, _ = result
        return reward

    def is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done.

        Args:
            state: The current state

        Returns:
            True if the episode is done, False otherwise
        """
        return state['agent_pos'] == state['goal_pos']

    async def move_up(self, agent):
        """Move the agent up.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('up')

    async def move_down(self, agent):
        """Move the agent down.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('down')

    async def move_left(self, agent):
        """Move the agent left.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('left')

    async def move_right(self, agent):
        """Move the agent right.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('right')


class GridWorldPPOAgent(PPOAgent):
    """PPO agent for the grid world environment."""

    def __init__(self, env: GridWorldEnv, **kwargs):
        """Initialize the agent.

        Args:
            env: The grid world environment
            **kwargs: Additional arguments for the base class
        """
        # Set up the state and action dimensions
        state_dim = 4  # Agent x, y, goal x, y
        action_dim = 4  # up, down, left, right

        super().__init__(
            name="GridWorldPPO",
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )

        self.env = env

        # Register actions
        self.register_action('up', self.move_up)
        self.register_action('down', self.move_down)
        self.register_action('left', self.move_left)
        self.register_action('right', self.move_right)

        # Set up the state encoder
        self.set_state_encoder(self.encode_grid_state)

    def encode_grid_state(self, state: Dict[str, Any]) -> np.ndarray:
        """Encode the grid world state as a vector.

        Args:
            state: The grid world state

        Returns:
            A vector representation of the state
        """
        agent_x, agent_y = state['agent_pos']
        goal_x, goal_y = state['goal_pos']

        # Normalize positions to [0, 1]
        agent_x_norm = agent_x / (self.env.width - 1)
        agent_y_norm = agent_y / (self.env.height - 1)
        goal_x_norm = goal_x / (self.env.width - 1)
        goal_y_norm = goal_y / (self.env.height - 1)

        return np.array([agent_x_norm, agent_y_norm, goal_x_norm, goal_y_norm], dtype=np.float32)

    async def perceive(self) -> Dict[str, Any]:
        """Perceive the environment.

        Returns:
            The current state
        """
        return self.env.get_state()

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
        # Unpack the result
        _, reward, _ = result
        return reward

    def is_done(self, state: Dict[str, Any]) -> bool:
        """Check if the episode is done.

        Args:
            state: The current state

        Returns:
            True if the episode is done, False otherwise
        """
        return state['agent_pos'] == state['goal_pos']

    async def move_up(self, agent):
        """Move the agent up.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('up')

    async def move_down(self, agent):
        """Move the agent down.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('down')

    async def move_left(self, agent):
        """Move the agent left.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('left')

    async def move_right(self, agent):
        """Move the agent right.

        Args:
            agent: The agent

        Returns:
            The result of the action
        """
        return self.env.step('right')


async def train_dqn_agent():
    """Train a DQN agent on the grid world environment."""
    logger.info("=== Training DQN Agent ===")

    # Create the environment
    env = GridWorldEnv(width=5, height=5)

    # Create the agent
    agent = GridWorldDQNAgent(
        env=env,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        batch_size=32,
        memory_size=10000,
        target_update_frequency=10
    )

    # Train the agent
    num_episodes = 100
    max_steps = 100

    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        agent.state['current_state'] = state

        total_reward = 0

        for step in range(max_steps):
            # Run one step
            result = await agent.run_once()

            # Unpack the result
            _, reward, done = result
            total_reward += reward

            if done:
                break

        # End the episode
        agent.end_episode()

        # Log progress
        logger.info(
            f"Episode {episode+1}/{num_episodes}: "
            f"Steps: {step+1}, "
            f"Total Reward: {total_reward:.2f}, "
            f"Exploration Rate: {agent.exploration_rate:.4f}"
        )

        # Render the environment occasionally
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode+1} final state:")
            env.render()

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/dqn_gridworld.pt")

    return agent


async def train_ppo_agent():
    """Train a PPO agent on the grid world environment."""
    logger.info("\n=== Training PPO Agent ===")

    # Create the environment
    env = GridWorldEnv(width=5, height=5)

    # Create the agent
    agent = GridWorldPPOAgent(
        env=env,
        learning_rate=0.0003,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        batch_size=32,
        epochs=10
    )

    # Train the agent
    num_episodes = 100
    max_steps = 100

    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        agent.state['current_state'] = state

        total_reward = 0

        for step in range(max_steps):
            # Run one step
            result = await agent.run_once()

            # Unpack the result
            _, reward, done = result
            total_reward += reward

            if done:
                break

        # End the episode
        agent.end_episode()

        # Train if we have enough data
        if len(agent.states) >= agent.batch_size:
            losses = await agent.train()
            logger.info(f"Training losses: {losses}")

        # Log progress
        logger.info(
            f"Episode {episode+1}/{num_episodes}: "
            f"Steps: {step+1}, "
            f"Total Reward: {total_reward:.2f}"
        )

        # Render the environment occasionally
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode+1} final state:")
            env.render()

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/ppo_gridworld.pt")

    return agent


async def transfer_learning_example(source_agent, target_agent):
    """Demonstrate transfer learning between agents."""
    logger.info("\n=== Transfer Learning Example ===")

    # Create a transfer learning manager
    transfer_manager = TransferLearningManager()

    # Transfer model parameters
    logger.info("Transferring model parameters...")
    success = await transfer_manager.transfer(
        source_agent=source_agent,
        target_agent=target_agent,
        method="model",
        transfer_type="full"
    )

    logger.info(f"Model transfer {'succeeded' if success else 'failed'}")

    # Create a new environment for testing
    test_env = GridWorldEnv(width=5, height=5)
    target_agent.env = test_env

    # Test the target agent
    logger.info("\nTesting target agent after transfer...")

    # Reset the environment
    state = test_env.reset()
    target_agent.state['current_state'] = state

    # Run for a few steps
    max_steps = 20
    total_reward = 0

    for step in range(max_steps):
        # Run one step
        result = await target_agent.run_once()

        # Unpack the result
        _, reward, done = result
        total_reward += reward

        # Render the environment
        logger.info(f"Step {step+1}, Reward: {reward:.2f}")
        test_env.render()

        if done:
            logger.info("Goal reached!")
            break

    logger.info(f"Test completed: Steps: {step+1}, Total Reward: {total_reward:.2f}")


async def main():
    """Run the advanced learning examples."""
    # Train a DQN agent
    dqn_agent = await train_dqn_agent()

    # Train a PPO agent
    ppo_agent = await train_ppo_agent()

    # Demonstrate transfer learning
    await transfer_learning_example(dqn_agent, ppo_agent)


if __name__ == "__main__":
    asyncio.run(main())
