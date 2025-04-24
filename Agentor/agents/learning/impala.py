"""
IMPALA (Importance Weighted Actor-Learner Architecture) implementation for the Agentor framework.

This module provides an implementation of the IMPALA algorithm for scalable distributed
reinforcement learning, as described in the paper:
"IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
by Espeholt et al. (2018).
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Deque
import numpy as np
import random
import time
import logging
import asyncio
import os
import json
from collections import deque, namedtuple

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
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. IMPALAAgent will not work.")
    TORCH_AVAILABLE = False


# Define a named tuple for storing trajectories
Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rewards', 'dones', 'log_probs', 'values'])


class IMPALANetwork(nn.Module):
    """Neural network for the IMPALA agent."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize the network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Linear(hidden_dim, action_dim)
        
        # Value head (critic)
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        # Shared layers
        shared_features = self.shared(x)
        
        # Policy head
        action_logits = self.policy(shared_features)
        
        # Value head
        value = self.value(shared_features)
        
        return action_logits, value


class ActorState:
    """State for an actor in the IMPALA architecture."""
    
    def __init__(self, actor_id: str):
        """Initialize the actor state.
        
        Args:
            actor_id: ID of the actor
        """
        self.actor_id = actor_id
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_value = 0.0
        self.last_log_prob = 0.0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.total_reward = 0.0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add_transition(self, state, action, reward, done, log_prob, value):
        """Add a transition to the trajectory buffer.
        
        Args:
            state: The state
            action: The action
            reward: The reward
            done: Whether the episode is done
            log_prob: Log probability of the action
            value: Value estimate
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_trajectory(self) -> Trajectory:
        """Get the current trajectory.
        
        Returns:
            The trajectory
        """
        return Trajectory(
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            log_probs=self.log_probs,
            values=self.values
        )
    
    def clear_trajectory(self):
        """Clear the trajectory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []


class IMPALAAgent(EnhancedAgent):
    """IMPALA (Importance Weighted Actor-Learner Architecture) agent implementation."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        num_actors: int = 4,
        trajectory_length: int = 20,
        batch_size: int = 64,
        device: str = "cpu",
        environment: Optional[IEnvironment] = None,
        tool_registry: Optional[IToolRegistry] = None
    ):
        """Initialize the IMPALA agent.
        
        Args:
            name: Name of the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            discount_factor: Discount factor for future rewards
            entropy_coef: Coefficient for the entropy loss
            value_coef: Coefficient for the value loss
            max_grad_norm: Maximum gradient norm for clipping
            num_actors: Number of actors to use
            trajectory_length: Length of trajectories before sending to learner
            batch_size: Batch size for training
            device: Device to use for computation
            environment: The environment
            tool_registry: Tool registry for the agent
        """
        super().__init__(name=name, tool_registry=tool_registry)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for IMPALAAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.num_actors = num_actors
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.device = device
        self.environment = environment
        
        # Create the learner network
        self.learner_network = IMPALANetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Create the optimizer
        self.optimizer = optim.Adam(self.learner_network.parameters(), lr=learning_rate)
        
        # Create actor networks and states
        self.actor_networks = []
        self.actor_states = []
        
        for i in range(num_actors):
            # Create actor network (copy of learner network)
            actor_network = IMPALANetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(device)
            actor_network.load_state_dict(self.learner_network.state_dict())
            self.actor_networks.append(actor_network)
            
            # Create actor state
            actor_state = ActorState(actor_id=f"actor_{i}")
            self.actor_states.append(actor_state)
        
        # Create trajectory queue
        self.trajectory_queue = asyncio.Queue()
        
        # Initialize state model
        self.state_model = AgentStateModel()
        self.state_model.current_state = None
        self.state_model.current_actor = 0
        self.state_model.update_count = 0
        self.state_model.training_metrics = {}
        
        # Register actions
        for i in range(action_dim):
            self.register_action(str(i), lambda i=i: self._take_action(i))
    
    def _take_action(self, action: int) -> Dict[str, Any]:
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
    
    def decide(self) -> str:
        """Choose an action using the current actor network.
        
        Returns:
            The name of the action to take
        """
        # Get the current state
        state = self.state_model.current_state
        
        # Get the current actor
        actor_idx = self.state_model.current_actor
        actor_network = self.actor_networks[actor_idx]
        actor_state = self.actor_states[actor_idx]
        
        # Encode the state
        state_tensor = self.encode_state(state)
        
        # Get action logits and value from the actor network
        with torch.no_grad():
            action_logits, value = actor_network(state_tensor)
            
            # Convert logits to probabilities
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample an action
            action_dist = Categorical(action_probs)
            action = action_dist.sample().item()
            
            # Get log probability of the action
            log_prob = action_dist.log_prob(torch.tensor([action])).item()
        
        # Store the action, log probability, and value in the actor state
        actor_state.last_action = action
        actor_state.last_log_prob = log_prob
        actor_state.last_value = value.item()
        
        # Store the action in the state model
        self.state_model.last_action = action
        
        return str(action)
    
    async def act(self, action_name: str) -> Any:
        """Execute the specified action.
        
        Args:
            action_name: The name of the action to execute
            
        Returns:
            The result of the action
        """
        # If no environment is provided, use the registered action
        if self.environment is None:
            action = int(action_name)
            return await super().act(action_name)
        
        # Convert action name to integer
        action = int(action_name)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.environment.step(action)
        
        # Get the current actor
        actor_idx = self.state_model.current_actor
        actor_state = self.actor_states[actor_idx]
        
        # Store the reward and update episode statistics
        actor_state.last_reward = reward
        actor_state.current_episode_reward += reward
        actor_state.current_episode_length += 1
        actor_state.total_reward += reward
        actor_state.total_steps += 1
        
        # Store the transition in the actor's trajectory
        actor_state.add_transition(
            state=self.state_model.current_state,
            action=action,
            reward=reward,
            done=terminated or truncated,
            log_prob=actor_state.last_log_prob,
            value=actor_state.last_value
        )
        
        # Check if the trajectory is complete
        if len(actor_state.states) >= self.trajectory_length or terminated or truncated:
            # Get the trajectory
            trajectory = actor_state.get_trajectory()
            
            # Add the trajectory to the queue
            await self.trajectory_queue.put(trajectory)
            
            # Clear the trajectory
            actor_state.clear_trajectory()
        
        # Check if the episode is done
        if terminated or truncated:
            # Store episode statistics
            actor_state.episode_rewards.append(actor_state.current_episode_reward)
            actor_state.episode_lengths.append(actor_state.current_episode_length)
            
            # Reset episode statistics
            actor_state.current_episode_reward = 0.0
            actor_state.current_episode_length = 0
            
            # Reset the environment
            observation, info = self.environment.reset()
        
        # Update the state model
        self.state_model.current_state = {"observation": observation, "info": info, "done": terminated or truncated}
        
        return {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    
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
        
        # Train the learner if there are trajectories in the queue
        if not self.trajectory_queue.empty():
            await self.train()
        
        return result
    
    async def train(self) -> Dict[str, float]:
        """Train the learner on trajectories from the queue.
        
        Returns:
            Dictionary of loss values
        """
        # Get trajectories from the queue (up to batch_size)
        trajectories = []
        for _ in range(min(self.batch_size, self.trajectory_queue.qsize())):
            try:
                trajectory = self.trajectory_queue.get_nowait()
                trajectories.append(trajectory)
                self.trajectory_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        if not trajectories:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        
        # Process trajectories
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_values = []
        
        for trajectory in trajectories:
            all_states.extend(trajectory.states)
            all_actions.extend(trajectory.actions)
            all_rewards.extend(trajectory.rewards)
            all_dones.extend(trajectory.dones)
            all_log_probs.extend(trajectory.log_probs)
            all_values.extend(trajectory.values)
        
        # Convert to tensors
        states_tensor = torch.cat([self.encode_state(s) for s in all_states])
        actions_tensor = torch.LongTensor(all_actions).to(self.device)
        rewards_tensor = torch.FloatTensor(all_rewards).to(self.device)
        dones_tensor = torch.FloatTensor(all_dones).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_log_probs).to(self.device)
        old_values_tensor = torch.FloatTensor(all_values).to(self.device)
        
        # Calculate returns and advantages
        returns = self._compute_returns(rewards_tensor, dones_tensor)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Get current action logits and values from the learner network
        action_logits, values = self.learner_network(states_tensor)
        values = values.squeeze(-1)
        
        # Calculate advantages
        advantages = returns_tensor - old_values_tensor
        
        # Convert logits to probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Create a distribution from the probabilities
        dist = Categorical(action_probs)
        
        # Calculate log probabilities of the actions
        log_probs = dist.log_prob(actions_tensor)
        
        # Calculate importance weights
        rhos = torch.exp(log_probs - old_log_probs_tensor)
        
        # Clipped importance weights
        rhos_clipped = torch.clamp(rhos, 0.0, 1.0)
        
        # Calculate policy loss (importance weighted)
        policy_loss = -torch.mean(rhos_clipped * advantages)
        
        # Calculate value loss
        value_loss = F.mse_loss(values, returns_tensor)
        
        # Calculate entropy
        entropy = torch.mean(dist.entropy())
        
        # Calculate total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimize the learner network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        nn.utils.clip_grad_norm_(self.learner_network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Update actor networks
        self._update_actor_networks()
        
        # Update state model
        self.state_model.update_count += 1
        self.state_model.training_metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }
        
        return self.state_model.training_metrics
    
    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> List[float]:
        """Compute returns for a batch of episodes.
        
        Args:
            rewards: Tensor of rewards
            dones: Tensor of done flags
            
        Returns:
            List of returns
        """
        returns = []
        R = 0
        
        # Compute returns in reverse order
        for i in range(len(rewards) - 1, -1, -1):
            # If done, reset the return
            if dones[i]:
                R = 0
            
            # Update the return
            R = rewards[i] + self.discount_factor * R * (1 - dones[i])
            returns.insert(0, R)
        
        return returns
    
    def _update_actor_networks(self):
        """Update the actor networks with the learner network's parameters."""
        learner_state_dict = self.learner_network.state_dict()
        
        for actor_network in self.actor_networks:
            actor_network.load_state_dict(learner_state_dict)
    
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
        
        # Start actor tasks
        actor_tasks = []
        for i in range(self.num_actors):
            task = asyncio.create_task(self._run_actor(i, num_episodes // self.num_actors, max_steps))
            actor_tasks.append(task)
        
        # Start learner task
        learner_task = asyncio.create_task(self._run_learner())
        
        # Wait for all actors to finish
        await asyncio.gather(*actor_tasks)
        
        # Stop the learner
        learner_task.cancel()
        
        # Collect statistics from all actors
        for actor_state in self.actor_states:
            episode_rewards.extend(actor_state.episode_rewards)
            episode_lengths.extend(actor_state.episode_lengths)
            total_steps += actor_state.total_steps
        
        # Calculate mean rewards and lengths
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "total_steps": total_steps
        }
    
    async def _run_actor(self, actor_idx: int, num_episodes: int, max_steps: int) -> None:
        """Run an actor for a number of episodes.
        
        Args:
            actor_idx: Index of the actor
            num_episodes: Number of episodes to run
            max_steps: Maximum number of steps per episode
        """
        # Get the actor network and state
        actor_network = self.actor_networks[actor_idx]
        actor_state = self.actor_states[actor_idx]
        
        # Create a copy of the environment
        if hasattr(self.environment, "copy"):
            env = self.environment.copy()
        else:
            env = self.environment
        
        for episode in range(num_episodes):
            # Reset the environment
            observation, info = env.reset()
            actor_state.current_state = {"observation": observation, "info": info, "done": False}
            
            # Initialize episode statistics
            episode_reward = 0.0
            episode_length = 0
            
            # Run the episode
            done = False
            while not done and episode_length < max_steps:
                # Encode the state
                state_tensor = self.encode_state(actor_state.current_state)
                
                # Get action logits and value from the actor network
                with torch.no_grad():
                    action_logits, value = actor_network(state_tensor)
                    
                    # Convert logits to probabilities
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    # Sample an action
                    action_dist = Categorical(action_probs)
                    action = action_dist.sample().item()
                    
                    # Get log probability of the action
                    log_prob = action_dist.log_prob(torch.tensor([action])).item()
                
                # Take a step in the environment
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update episode statistics
                episode_reward += reward
                episode_length += 1
                
                # Store the transition in the actor's trajectory
                actor_state.add_transition(
                    state=actor_state.current_state,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value.item()
                )
                
                # Update the actor state
                actor_state.current_state = {"observation": observation, "info": info, "done": done}
                actor_state.last_action = action
                actor_state.last_reward = reward
                actor_state.last_log_prob = log_prob
                actor_state.last_value = value.item()
                actor_state.current_episode_reward += reward
                actor_state.current_episode_length += 1
                actor_state.total_reward += reward
                actor_state.total_steps += 1
                
                # Check if the trajectory is complete
                if len(actor_state.states) >= self.trajectory_length or done:
                    # Get the trajectory
                    trajectory = actor_state.get_trajectory()
                    
                    # Add the trajectory to the queue
                    await self.trajectory_queue.put(trajectory)
                    
                    # Clear the trajectory
                    actor_state.clear_trajectory()
            
            # Store episode statistics
            actor_state.episode_rewards.append(actor_state.current_episode_reward)
            actor_state.episode_lengths.append(actor_state.current_episode_length)
            
            # Reset episode statistics
            actor_state.current_episode_reward = 0.0
            actor_state.current_episode_length = 0
            
            # Log episode statistics
            logger.info(
                f"Actor {actor_idx}, Episode {episode+1}/{num_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}"
            )
    
    async def _run_learner(self) -> None:
        """Run the learner to train on trajectories from the queue."""
        try:
            while True:
                # Wait for trajectories
                if self.trajectory_queue.empty():
                    await asyncio.sleep(0.01)
                    continue
                
                # Train on trajectories
                await self.train()
                
                # Update actor networks
                self._update_actor_networks()
        except asyncio.CancelledError:
            logger.info("Learner task cancelled")
    
    async def save_model(self, path: str) -> None:
        """Save the learner model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "learner_network": self.learner_network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
    
    async def load_model(self, path: str) -> None:
        """Load the learner model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.learner_network.load_state_dict(checkpoint["learner_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Update actor networks
        self._update_actor_networks()
