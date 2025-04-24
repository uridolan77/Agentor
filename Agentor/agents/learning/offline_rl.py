"""
Offline Reinforcement Learning implementation for the Agentor framework.

This module provides an Offline Reinforcement Learning agent implementation
that can learn from historical data without interacting with the environment.
It implements Conservative Q-Learning (CQL) for offline RL.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
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
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. OfflineRLAgent will not work.")
    TORCH_AVAILABLE = False


class QNetwork(nn.Module):
    """Neural network for Q-value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize the Q-network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OfflineRLAgent(EnhancedAgent):
    """Offline Reinforcement Learning agent implementation using Conservative Q-Learning (CQL)."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        cql_alpha: float = 1.0,
        batch_size: int = 64,
        target_update_frequency: int = 100,
        device: str = "cpu",
        environment: Optional[IEnvironment] = None,
        tool_registry: Optional[IToolRegistry] = None,
        dataset: Optional[List[Tuple[Dict[str, Any], str, float, Dict[str, Any], bool]]] = None
    ):
        """Initialize the offline reinforcement learning agent.
        
        Args:
            name: Name of the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            discount_factor: Discount factor for future rewards
            cql_alpha: Weight for the CQL regularization term
            batch_size: Batch size for training
            target_update_frequency: How often to update the target network
            device: Device to use for computation
            environment: The environment (for evaluation only)
            tool_registry: Tool registry for the agent
            dataset: Initial dataset of transitions (state, action, reward, next_state, done)
        """
        super().__init__(name=name, tool_registry=tool_registry)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for OfflineRLAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.cql_alpha = cql_alpha
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device
        self.environment = environment
        
        # Create the Q-network
        self.q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Create the target Q-network
        self.target_q_network = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Create the optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize dataset
        self.dataset = dataset or []
        
        # Initialize state model
        self.state_model = AgentStateModel()
        self.state_model.current_state = None
        self.state_model.update_count = 0
        self.state_model.episode_reward = 0.0
        self.state_model.episode_length = 0
        self.state_model.total_reward = 0.0
        self.state_model.total_steps = 0
        self.state_model.episode_rewards = []
        self.state_model.episode_lengths = []
        self.state_model.training_metrics = {}
        
        # Create action mapping
        self.actions_list = [str(i) for i in range(action_dim)]
        self.action_to_index = {action: i for i, action in enumerate(self.actions_list)}
        self.index_to_action = {i: action for i, action in enumerate(self.actions_list)}
        
        # Register actions
        for action in self.actions_list:
            self.register_action(action, lambda action=action: self._take_action(action))
    
    def _take_action(self, action: str) -> Dict[str, Any]:
        """Take an action in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Result of the action
        """
        return {"action": action}
    
    def encode_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """Encode a state as a tensor.
        
        Args:
            state: The state to encode
            
        Returns:
            Encoded state tensor
        """
        if isinstance(state, dict) and "observation" in state:
            return torch.FloatTensor(state["observation"]).to(self.device)
        elif isinstance(state, np.ndarray):
            return torch.FloatTensor(state).to(self.device)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
    
    def add_to_dataset(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        done: bool
    ) -> None:
        """Add a transition to the dataset.
        
        Args:
            state: The state before the action
            action: The action taken
            reward: The reward received
            next_state: The state after the action
            done: Whether the episode is done
        """
        self.dataset.append((state, action, reward, next_state, done))
    
    def decide(self) -> str:
        """Choose the best action according to the Q-network.
        
        Returns:
            The name of the action to take
        """
        # Get the current state
        state = self.state_model.current_state
        
        # Encode the state
        state_tensor = self.encode_state(state)
        
        # Get Q-values from the network
        with torch.no_grad():
            q_values = self.q_network(state_tensor.unsqueeze(0))
            action_index = q_values.max(1)[1].item()
        
        # Get the action name
        action = self.actions_list[action_index]
        
        # Store the action in state model
        self.state_model.last_action = action
        
        return action
    
    async def train(self, num_iterations: int = 1000) -> Dict[str, float]:
        """Train the agent on the offline dataset.
        
        Args:
            num_iterations: Number of training iterations
            
        Returns:
            Dictionary of loss values
        """
        if len(self.dataset) < self.batch_size:
            return {"loss": 0.0, "q_loss": 0.0, "cql_loss": 0.0}
        
        total_loss = 0.0
        total_q_loss = 0.0
        total_cql_loss = 0.0
        
        for _ in range(num_iterations):
            # Sample a batch of transitions
            batch = random.sample(self.dataset, self.batch_size)
            
            # Separate the batch into components
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            state_tensors = torch.cat([self.encode_state(s).unsqueeze(0) for s in states])
            action_indices = torch.LongTensor([self.action_to_index[a] for a in actions]).to(self.device)
            reward_tensors = torch.FloatTensor(rewards).to(self.device)
            next_state_tensors = torch.cat([self.encode_state(s).unsqueeze(0) for s in next_states])
            done_tensors = torch.FloatTensor(dones).to(self.device)
            
            # Get current Q-values
            q_values = self.q_network(state_tensors)
            q_values_selected = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            
            # Get next Q-values from target network
            with torch.no_grad():
                next_q_values = self.target_q_network(next_state_tensors).max(1)[0]
            
            # Calculate target Q-values
            target_q_values = reward_tensors + (1 - done_tensors) * self.discount_factor * next_q_values
            
            # Calculate standard Q-learning loss
            q_loss = F.smooth_l1_loss(q_values_selected, target_q_values)
            
            # Calculate CQL loss (to prevent overestimation of unseen actions)
            # This is the key component of Conservative Q-Learning
            cql_loss = torch.logsumexp(q_values, dim=1).mean() - q_values_selected.mean()
            
            # Total loss
            loss = q_loss + self.cql_alpha * cql_loss
            
            # Optimize the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network if needed
            self.state_model.update_count += 1
            if self.state_model.update_count % self.target_update_frequency == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            # Accumulate losses
            total_loss += loss.item()
            total_q_loss += q_loss.item()
            total_cql_loss += cql_loss.item()
        
        # Calculate average losses
        avg_loss = total_loss / num_iterations
        avg_q_loss = total_q_loss / num_iterations
        avg_cql_loss = total_cql_loss / num_iterations
        
        # Update state model
        self.state_model.training_metrics = {
            "loss": avg_loss,
            "q_loss": avg_q_loss,
            "cql_loss": avg_cql_loss
        }
        
        return self.state_model.training_metrics
    
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
        
        # Convert action name to integer
        action = int(action_name)
        
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
        """Run one perception-decision-action cycle (for evaluation only).
        
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
    
    async def load_dataset_from_file(self, path: str) -> None:
        """Load a dataset from a file.
        
        Args:
            path: Path to the dataset file
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.dataset = []
        for item in data:
            state = item["state"]
            action = item["action"]
            reward = item["reward"]
            next_state = item["next_state"]
            done = item["done"]
            
            self.dataset.append((state, action, reward, next_state, done))
        
        logger.info(f"Loaded dataset with {len(self.dataset)} transitions from {path}")
    
    async def save_dataset_to_file(self, path: str) -> None:
        """Save the dataset to a file.
        
        Args:
            path: Path to save the dataset
        """
        data = []
        for state, action, reward, next_state, done in self.dataset:
            data.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved dataset with {len(self.dataset)} transitions to {path}")
    
    async def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """Evaluate the agent in the environment.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Evaluation statistics
        """
        if self.environment is None:
            raise ValueError("No environment provided for evaluation")
        
        # Initialize statistics
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            # Reset the environment
            observation, info = self.environment.reset()
            state = {"observation": observation, "info": info, "done": False}
            self.state_model.current_state = state
            
            # Initialize episode statistics
            episode_reward = 0.0
            episode_length = 0
            
            # Run the episode
            done = False
            while not done:
                # Choose an action
                action = self.decide()
                
                # Convert action name to integer
                action_int = int(action)
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.environment.step(action_int)
                done = terminated or truncated
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                
                # Update the state
                state = {"observation": observation, "info": info, "done": done}
                self.state_model.current_state = state
                
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
    
    async def collect_data_from_environment(
        self,
        num_episodes: int,
        policy: Optional[Callable[[Dict[str, Any]], str]] = None,
        exploration_rate: float = 0.1
    ) -> None:
        """Collect data from the environment using a specified policy.
        
        Args:
            num_episodes: Number of episodes to collect
            policy: Policy function that maps states to actions, or None to use random actions
            exploration_rate: Probability of taking a random action
        """
        if self.environment is None:
            raise ValueError("No environment provided for data collection")
        
        for episode in range(num_episodes):
            # Reset the environment
            observation, info = self.environment.reset()
            state = {"observation": observation, "info": info, "done": False}
            
            # Initialize episode statistics
            episode_reward = 0.0
            episode_length = 0
            
            # Run the episode
            done = False
            while not done:
                # Choose an action
                if policy is None or random.random() < exploration_rate:
                    # Random action
                    action = random.choice(self.actions_list)
                else:
                    # Use the provided policy
                    action = policy(state)
                
                # Convert action name to integer
                action_int = int(action)
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.environment.step(action_int)
                done = terminated or truncated
                
                # Create the next state
                next_state = {"observation": observation, "info": info, "done": done}
                
                # Add the transition to the dataset
                self.add_to_dataset(state, action, reward, next_state, done)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                
                # Update the state
                state = next_state
            
            # Log episode statistics
            logger.info(
                f"Data Collection Episode {episode+1}/{num_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}"
            )
        
        logger.info(f"Collected dataset with {len(self.dataset)} transitions")
    
    async def save_model(self, path: str) -> None:
        """Save the agent's model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
    
    async def load_model(self, path: str) -> None:
        """Load the agent's model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
