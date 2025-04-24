"""
Model-Based Reinforcement Learning implementation for the Agentor framework.

This module provides a Model-Based Reinforcement Learning agent implementation
that learns a model of the environment dynamics to improve sample efficiency.
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
    logger.warning("PyTorch not available. ModelBasedRLAgent will not work.")
    TORCH_AVAILABLE = False


class DynamicsModel(nn.Module):
    """Neural network model for predicting environment dynamics."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize the dynamics model.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super().__init__()
        
        # Input: state and action
        self.input_dim = state_dim + action_dim
        
        # Output: next state and reward
        self.output_dim = state_dim + 1
        
        # Network architecture
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor (one-hot encoded)
            
        Returns:
            Tuple of (next_state, reward)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Split output into next state and reward
        next_state = x[:, :-1]
        reward = x[:, -1:]
        
        return next_state, reward


class QNetwork(nn.Module):
    """Neural network for Q-value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
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


class ModelBasedRLAgent(EnhancedAgent):
    """Model-Based Reinforcement Learning agent implementation."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        model_learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        model_batch_size: int = 128,
        model_epochs: int = 10,
        planning_horizon: int = 5,
        num_simulations: int = 10,
        device: str = "cpu",
        environment: Optional[IEnvironment] = None,
        tool_registry: Optional[IToolRegistry] = None
    ):
        """Initialize the model-based reinforcement learning agent.
        
        Args:
            name: Name of the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the Q-network
            model_learning_rate: Learning rate for the dynamics model
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            exploration_decay: Decay rate for exploration
            min_exploration_rate: Minimum exploration rate
            batch_size: Batch size for Q-network training
            memory_size: Size of the replay memory
            model_batch_size: Batch size for dynamics model training
            model_epochs: Number of epochs to train the dynamics model
            planning_horizon: Horizon for planning with the dynamics model
            num_simulations: Number of simulations for planning
            device: Device to use for computation
            environment: The environment
            tool_registry: Tool registry for the agent
        """
        super().__init__(name=name, tool_registry=tool_registry)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModelBasedRLAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model_learning_rate = model_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.model_epochs = model_epochs
        self.planning_horizon = planning_horizon
        self.num_simulations = num_simulations
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
        
        # Create the dynamics model
        self.dynamics_model = DynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Create the optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.model_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=model_learning_rate)
        
        # Create replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize state model
        self.state_model = AgentStateModel()
        self.state_model.current_state = None
        self.state_model.exploration_rate = exploration_rate
        self.state_model.update_count = 0
        self.state_model.model_update_count = 0
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
    
    def encode_action(self, action: str) -> torch.Tensor:
        """Encode an action as a one-hot tensor.
        
        Args:
            action: The action to encode
            
        Returns:
            One-hot encoded action tensor
        """
        action_idx = self.action_to_index[action]
        action_tensor = torch.zeros(self.action_dim).to(self.device)
        action_tensor[action_idx] = 1.0
        return action_tensor
    
    def store_transition(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        done: bool
    ) -> None:
        """Store a transition in the replay memory.
        
        Args:
            state: The state before the action
            action: The action taken
            reward: The reward received
            next_state: The state after the action
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def decide(self) -> str:
        """Choose an action using epsilon-greedy policy with model-based planning.
        
        Returns:
            The name of the action to take
        """
        # Get the current state
        state = self.state_model.current_state
        
        # Encode the state
        state_tensor = self.encode_state(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.state_model.exploration_rate:
            # Explore: choose a random action
            action_index = random.randrange(len(self.actions_list))
            action = self.actions_list[action_index]
        else:
            # Exploit: choose the best action using model-based planning
            action = self._plan(state_tensor)
        
        # Store the action in state model
        self.state_model.last_action = action
        
        return action
    
    def _plan(self, state_tensor: torch.Tensor) -> str:
        """Plan the best action using the dynamics model.
        
        Args:
            state_tensor: The current state tensor
            
        Returns:
            The best action
        """
        # If the dynamics model hasn't been trained enough, use the Q-network directly
        if self.state_model.model_update_count < 100:
            with torch.no_grad():
                q_values = self.q_network(state_tensor.unsqueeze(0))
                action_index = q_values.max(1)[1].item()
                return self.actions_list[action_index]
        
        # Initialize best action and value
        best_action = None
        best_value = float('-inf')
        
        # Evaluate each action
        for action_idx in range(self.action_dim):
            action = self.actions_list[action_idx]
            action_tensor = torch.zeros(self.action_dim).to(self.device)
            action_tensor[action_idx] = 1.0
            
            # Simulate trajectories
            total_value = 0.0
            
            for _ in range(self.num_simulations):
                # Start from the current state
                sim_state = state_tensor.clone()
                sim_value = 0.0
                
                # Simulate for planning_horizon steps
                for step in range(self.planning_horizon):
                    # Choose action for the first step, then use Q-network
                    if step == 0:
                        sim_action_tensor = action_tensor
                    else:
                        with torch.no_grad():
                            q_values = self.q_network(sim_state.unsqueeze(0))
                            sim_action_idx = q_values.max(1)[1].item()
                            sim_action_tensor = torch.zeros(self.action_dim).to(self.device)
                            sim_action_tensor[sim_action_idx] = 1.0
                    
                    # Predict next state and reward
                    with torch.no_grad():
                        next_sim_state, sim_reward = self.dynamics_model(
                            sim_state.unsqueeze(0),
                            sim_action_tensor.unsqueeze(0)
                        )
                    
                    # Update simulated state and value
                    sim_state = next_sim_state.squeeze(0)
                    sim_value += sim_reward.item() * (self.discount_factor ** step)
                
                # Add the value of the final state
                with torch.no_grad():
                    final_q_values = self.q_network(sim_state.unsqueeze(0))
                    final_value = final_q_values.max(1)[0].item()
                    sim_value += final_value * (self.discount_factor ** self.planning_horizon)
                
                total_value += sim_value
            
            # Average value across simulations
            avg_value = total_value / self.num_simulations
            
            # Update best action if needed
            if avg_value > best_value:
                best_value = avg_value
                best_action = action
        
        return best_action
    
    async def train_model(self) -> Dict[str, float]:
        """Train the dynamics model on a batch of transitions.
        
        Returns:
            Dictionary of loss values
        """
        if len(self.memory) < self.model_batch_size:
            return {"model_loss": 0.0}
        
        # Sample a batch of transitions
        batch = random.sample(self.memory, self.model_batch_size)
        
        # Separate the batch into components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_tensors = torch.cat([self.encode_state(s).unsqueeze(0) for s in states])
        action_tensors = torch.cat([self.encode_action(a).unsqueeze(0) for a in actions])
        reward_tensors = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_tensors = torch.cat([self.encode_state(s).unsqueeze(0) for s in next_states])
        
        # Train the dynamics model for multiple epochs
        total_loss = 0.0
        
        for _ in range(self.model_epochs):
            # Predict next states and rewards
            pred_next_states, pred_rewards = self.dynamics_model(state_tensors, action_tensors)
            
            # Calculate loss
            state_loss = F.mse_loss(pred_next_states, next_state_tensors)
            reward_loss = F.mse_loss(pred_rewards, reward_tensors)
            loss = state_loss + reward_loss
            
            # Optimize the model
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()
            
            total_loss += loss.item()
        
        # Update state model
        self.state_model.model_update_count += 1
        self.state_model.training_metrics["model_loss"] = total_loss / self.model_epochs
        
        return {"model_loss": total_loss / self.model_epochs}
    
    async def train_q_network(self) -> Dict[str, float]:
        """Train the Q-network on a batch of transitions.
        
        Returns:
            Dictionary of loss values
        """
        if len(self.memory) < self.batch_size:
            return {"q_loss": 0.0}
        
        # Sample a batch of transitions
        batch = random.sample(self.memory, self.batch_size)
        
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
        q_values = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_state_tensors).max(1)[0]
        
        # Calculate target Q-values
        target_q_values = reward_tensors + (1 - done_tensors) * self.discount_factor * next_q_values
        
        # Calculate loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize the Q-network
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        # Update state model
        self.state_model.update_count += 1
        self.state_model.training_metrics["q_loss"] = loss.item()
        
        # Update target network if needed
        if self.state_model.update_count % 10 == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Decay exploration rate
        self.state_model.exploration_rate = max(
            self.min_exploration_rate,
            self.state_model.exploration_rate * self.exploration_decay
        )
        
        return {"q_loss": loss.item()}
    
    async def train(self) -> Dict[str, float]:
        """Train both the dynamics model and the Q-network.
        
        Returns:
            Dictionary of loss values
        """
        # Train the dynamics model
        model_metrics = await self.train_model()
        
        # Train the Q-network
        q_metrics = await self.train_q_network()
        
        # Combine metrics
        metrics = {**model_metrics, **q_metrics}
        self.state_model.training_metrics = metrics
        
        return metrics
    
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
        """Run one perception-decision-action-learning cycle.
        
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
        
        # Get the new state
        next_state = await self.perceive()
        
        # Get the reward and done flag
        reward = result.get("reward", 0.0)
        done = result.get("terminated", False) or result.get("truncated", False)
        
        # Store the transition
        self.store_transition(state, action, reward, next_state, done)
        
        # Train the agent
        if len(self.memory) >= self.batch_size:
            await self.train()
        
        return result
    
    async def train_with_environment(
        self,
        num_episodes: int,
        max_steps: int = 100,
        render_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the agent in the environment.
        
        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode
            render_interval: Number of episodes between rendering, or None to disable rendering
            
        Returns:
            Training statistics
        """
        if self.environment is None:
            raise ValueError("No environment provided")
        
        # Initialize statistics
        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        
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
            while not done and episode_length < max_steps:
                # Choose an action
                action = self.decide()
                
                # Convert action name to integer
                action_int = int(action)
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = self.environment.step(action_int)
                done = terminated or truncated
                
                # Create the next state
                next_state = {"observation": observation, "info": info, "done": done}
                
                # Store the transition
                self.store_transition(state, action, reward, next_state, done)
                
                # Update statistics
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Update state model
                self.state_model.episode_reward += reward
                self.state_model.episode_length += 1
                self.state_model.total_reward += reward
                self.state_model.total_steps += 1
                
                # Train the agent
                if len(self.memory) >= self.batch_size:
                    await self.train()
                
                # Update the state
                state = next_state
                self.state_model.current_state = state
                
                # Render the environment if needed
                if render_interval is not None and episode % render_interval == 0:
                    self.environment.render()
            
            # Store episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            self.state_model.episode_rewards.append(episode_reward)
            self.state_model.episode_lengths.append(episode_length)
            
            # Reset episode statistics
            self.state_model.episode_reward = 0.0
            self.state_model.episode_length = 0
            
            # Log episode statistics
            logger.info(
                f"Episode {episode+1}/{num_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}, "
                f"exploration_rate={self.state_model.exploration_rate:.3f}"
            )
        
        # Calculate mean rewards and lengths
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "total_steps": total_steps
        }
    
    async def save_model(self, path: str) -> None:
        """Save the agent's models.
        
        Args:
            path: Path to save the models
        """
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "dynamics_model": self.dynamics_model.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "exploration_rate": self.state_model.exploration_rate
        }, path)
    
    async def load_model(self, path: str) -> None:
        """Load the agent's models.
        
        Args:
            path: Path to load the models from
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.dynamics_model.load_state_dict(checkpoint["dynamics_model"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.state_model.exploration_rate = checkpoint["exploration_rate"]
