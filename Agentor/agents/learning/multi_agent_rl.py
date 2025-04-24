"""
Multi-Agent Reinforcement Learning agent implementation for the Agentor framework.

This module provides a Multi-Agent Reinforcement Learning agent implementation
that supports both cooperative and competitive scenarios.
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
from agentor.core.interfaces.agent import AgentInput, AgentOutput, IAgent
from agentor.core.interfaces.environment import IEnvironment, Space
from agentor.components.environments.multi_agent import IMultiAgentEnvironment

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. MultiAgentRLAgent will not work.")
    TORCH_AVAILABLE = False


class MAAgentStateModel(AgentStateModel):
    """State model for multi-agent reinforcement learning agents."""
    
    def __init__(self, agent_id: str):
        """Initialize the state model.
        
        Args:
            agent_id: ID of the agent
        """
        super().__init__()
        self.agent_id = agent_id
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_value = 0.0
        self.last_action_prob = 0.0
        self.last_log_prob = 0.0
        self.last_entropy = 0.0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.total_reward = 0.0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {}


class MultiAgentRLEnvironment:
    """Environment wrapper for multi-agent reinforcement learning."""
    
    def __init__(
        self,
        environment: IMultiAgentEnvironment,
        num_agents: int,
        observation_space: Space,
        action_space: Space
    ):
        """Initialize the multi-agent RL environment.
        
        Args:
            environment: The multi-agent environment
            num_agents: Number of agents in the environment
            observation_space: Observation space for each agent
            action_space: Action space for each agent
        """
        self.environment = environment
        self.num_agents = num_agents
        self.observation_space = observation_space
        self.action_space = action_space
        self.agents = []
        self.rewards = [0] * num_agents
    
    def add_agent(self, agent: IAgent) -> None:
        """Add an agent to the environment.
        
        Args:
            agent: The agent to add
        """
        self.agents.append(agent)
    
    async def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Take a step in the environment with actions from all agents.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        return self.environment.step(actions)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observations, info)
        """
        return self.environment.reset(seed, options)


class MAAgent:
    """Individual agent in a multi-agent reinforcement learning system."""
    
    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        """Initialize the agent.
        
        Args:
            agent_id: ID of the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            device: Device to use for computation
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.device = device
        
        # Create the network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)
        
        # Create the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize state model
        self.state_model = MAAgentStateModel(agent_id)
    
    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """Choose an action based on the current state.
        
        Args:
            state: The current state
            
        Returns:
            Tuple of (action, action_prob)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            q_values = self.network(state_tensor)
            action_probs = F.softmax(q_values, dim=1)
        
        # Sample an action
        action_dist = Categorical(action_probs)
        action = action_dist.sample().item()
        
        # Get the probability of the selected action
        action_prob = action_probs[0, action].item()
        
        return action, action_prob
    
    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """Learn from a transition.
        
        Args:
            state: The state before the action
            action: The action taken
            reward: The reward received
            next_state: The state after the action
            done: Whether the episode is done
            
        Returns:
            Dictionary of loss values
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Get current Q-values
        q_values = self.network(state_tensor)
        
        # Get Q-value for the taken action
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Get next Q-values
        with torch.no_grad():
            next_q_values = self.network(next_state_tensor)
            next_q_value = next_q_values.max(1)[0]
        
        # Calculate target Q-value
        target_q_value = reward_tensor + (1 - done_tensor) * 0.99 * next_q_value
        
        # Calculate loss
        loss = F.smooth_l1_loss(q_value, target_q_value)
        
        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    def save(self, path: str) -> None:
        """Save the agent's model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """Load the agent's model.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class MultiAgentRLAgent(EnhancedAgent):
    """Multi-Agent Reinforcement Learning agent implementation."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        num_agents: int = 2,
        state_dim: int = 4,
        action_dim: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        memory_size: int = 10000,
        batch_size: int = 64,
        update_frequency: int = 4,
        device: str = "cpu",
        environment: Optional[IMultiAgentEnvironment] = None,
        tool_registry: Optional[IToolRegistry] = None,
        cooperative: bool = True
    ):
        """Initialize the multi-agent reinforcement learning agent.
        
        Args:
            name: Name of the agent
            num_agents: Number of agents in the environment
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
            learning_rate: Learning rate for the optimizer
            discount_factor: Discount factor for future rewards
            memory_size: Size of the replay memory
            batch_size: Batch size for training
            update_frequency: How often to update the network
            device: Device to use for computation
            environment: The multi-agent environment
            tool_registry: Tool registry for the agent
            cooperative: Whether agents cooperate or compete
        """
        super().__init__(name=name, tool_registry=tool_registry)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MultiAgentRLAgent")
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.device = device
        self.environment = environment
        self.cooperative = cooperative
        
        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            agent = MAAgent(
                agent_id=agent_id,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                device=device
            )
            self.agents.append(agent)
        
        # Create replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize state model
        self.state_model = AgentStateModel()
        self.state_model.current_state = None
        self.state_model.update_count = 0
        
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
    
    def store_transition(
        self,
        states: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_states: Dict[str, np.ndarray],
        dones: Dict[str, bool]
    ) -> None:
        """Store a transition in the replay memory.
        
        Args:
            states: Dictionary mapping agent IDs to states
            actions: Dictionary mapping agent IDs to actions
            rewards: Dictionary mapping agent IDs to rewards
            next_states: Dictionary mapping agent IDs to next states
            dones: Dictionary mapping agent IDs to done flags
        """
        self.memory.append((states, actions, rewards, next_states, dones))
    
    async def train(self) -> Dict[str, float]:
        """Train the agents on a batch of transitions.
        
        Returns:
            Dictionary of loss values
        """
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample a batch of transitions
        batch = random.sample(self.memory, self.batch_size)
        
        # Separate the batch into components
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)
        
        # Train each agent
        losses = {}
        for i, agent in enumerate(self.agents):
            agent_id = agent.agent_id
            
            # Collect agent-specific data
            agent_states = [states[agent_id] for states in states_batch]
            agent_actions = [actions[agent_id] for actions in actions_batch]
            agent_rewards = [rewards[agent_id] for rewards in rewards_batch]
            agent_next_states = [next_states[agent_id] for next_states in next_states_batch]
            agent_dones = [dones[agent_id] for dones in dones_batch]
            
            # Train the agent
            for j in range(len(agent_states)):
                loss = agent.learn(
                    state=agent_states[j],
                    action=agent_actions[j],
                    reward=agent_rewards[j],
                    next_state=agent_next_states[j],
                    done=agent_dones[j]
                )
                
                # Store the loss
                if agent_id not in losses:
                    losses[agent_id] = loss["loss"]
                else:
                    losses[agent_id] += loss["loss"]
            
            # Average the loss
            losses[agent_id] /= len(agent_states)
        
        # Update the state model
        self.state_model.update_count += 1
        self.state_model.training_metrics = losses
        
        return losses
    
    async def perceive(self) -> Dict[str, Any]:
        """Perceive the environment.
        
        Returns:
            The current state
        """
        # If no environment is provided, return an empty state
        if self.environment is None:
            return {}
        
        # If the environment has not been reset, reset it
        if not hasattr(self, "_observations") or self._observations is None:
            self._observations, _ = self.environment.reset()
        
        return self._observations
    
    def decide(self) -> Dict[str, str]:
        """Choose actions for all agents.
        
        Returns:
            Dictionary mapping agent IDs to action names
        """
        # Get the current state
        state = self.state_model.current_state
        
        # Choose actions for each agent
        actions = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            
            # Get the agent's state
            if isinstance(state, dict) and agent_id in state:
                agent_state = state[agent_id]
            else:
                # If no state is available, use a default state
                agent_state = np.zeros(self.state_dim, dtype=np.float32)
            
            # Choose an action
            action, action_prob = agent.act(agent_state)
            
            # Store the action and probability
            actions[agent_id] = str(action)
            
            # Update the agent's state model
            agent.state_model.last_action = action
            agent.state_model.last_action_prob = action_prob
        
        return actions
    
    async def act(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """Execute actions in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to action names
            
        Returns:
            Result of the actions
        """
        # If no environment is provided, return an empty result
        if self.environment is None:
            return {}
        
        # Convert action names to integers
        int_actions = {agent_id: int(action) for agent_id, action in actions.items()}
        
        # Take a step in the environment
        observations, rewards, terminated, truncated, info = await self.environment.step(int_actions)
        
        # Store the results
        self._observations = observations
        self._rewards = rewards
        self._terminated = terminated
        self._truncated = truncated
        self._info = info
        
        # Update agent state models
        for agent in self.agents:
            agent_id = agent.agent_id
            if agent_id in rewards:
                agent.state_model.last_reward = rewards[agent_id]
                agent.state_model.current_episode_reward += rewards[agent_id]
                agent.state_model.current_episode_length += 1
                agent.state_model.total_reward += rewards[agent_id]
                agent.state_model.total_steps += 1
        
        return {
            "observations": observations,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    
    async def run_once(self) -> Dict[str, Any]:
        """Run one perception-decision-action-learning cycle.
        
        Returns:
            Result of the actions
        """
        # Get the current state
        states = await self.perceive()
        self.state_model.current_state = states
        
        # Choose actions
        actions = self.decide()
        
        # Execute actions
        result = await self.act(actions)
        
        # Get the new state
        next_states = await self.perceive()
        
        # Store the transition
        self.store_transition(
            states=states,
            actions={agent_id: int(action) for agent_id, action in actions.items()},
            rewards=result["rewards"],
            next_states=next_states,
            dones={agent_id: result["terminated"][agent_id] or result["truncated"][agent_id] for agent_id in result["terminated"]}
        )
        
        # Train the agents
        if len(self.memory) >= self.batch_size and self.state_model.update_count % self.update_frequency == 0:
            await self.train()
        
        # Check if the episode is done
        all_done = all(result["terminated"].values()) or all(result["truncated"].values())
        if all_done:
            self._end_episode()
        
        return result
    
    def _end_episode(self) -> None:
        """End the current episode and update statistics."""
        for agent in self.agents:
            # Store episode statistics
            agent.state_model.episode_rewards.append(agent.state_model.current_episode_reward)
            agent.state_model.episode_lengths.append(agent.state_model.current_episode_length)
            
            # Reset episode statistics
            agent.state_model.current_episode_reward = 0.0
            agent.state_model.current_episode_length = 0
    
    async def train_with_environment(
        self,
        num_episodes: int,
        max_steps: int = 100,
        render_interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the agents in the environment.
        
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
        episode_rewards = {agent.agent_id: [] for agent in self.agents}
        episode_lengths = []
        total_steps = 0
        
        for episode in range(num_episodes):
            # Reset the environment
            observations, _ = self.environment.reset()
            self._observations = observations
            
            # Initialize episode statistics
            episode_reward = {agent.agent_id: 0.0 for agent in self.agents}
            episode_length = 0
            
            # Reset agent state models
            for agent in self.agents:
                agent.state_model.current_episode_reward = 0.0
                agent.state_model.current_episode_length = 0
            
            # Run the episode
            done = False
            while not done and episode_length < max_steps:
                # Choose actions
                actions = self.decide()
                
                # Convert action names to integers
                int_actions = {agent_id: int(action) for agent_id, action in actions.items()}
                
                # Take a step in the environment
                observations, rewards, terminated, truncated, info = await self.environment.step(int_actions)
                
                # Store the results
                self._observations = observations
                self._rewards = rewards
                self._terminated = terminated
                self._truncated = truncated
                self._info = info
                
                # Update statistics
                for agent_id, reward in rewards.items():
                    episode_reward[agent_id] += reward
                episode_length += 1
                total_steps += 1
                
                # Update agent state models
                for agent in self.agents:
                    agent_id = agent.agent_id
                    if agent_id in rewards:
                        agent.state_model.last_reward = rewards[agent_id]
                        agent.state_model.current_episode_reward += rewards[agent_id]
                        agent.state_model.current_episode_length += 1
                        agent.state_model.total_reward += rewards[agent_id]
                        agent.state_model.total_steps += 1
                
                # Store the transition
                self.store_transition(
                    states=self.state_model.current_state,
                    actions={agent_id: int(action) for agent_id, action in actions.items()},
                    rewards=rewards,
                    next_states=observations,
                    dones={agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in terminated}
                )
                
                # Train the agents
                if len(self.memory) >= self.batch_size and self.state_model.update_count % self.update_frequency == 0:
                    await self.train()
                
                # Update the state
                self.state_model.current_state = observations
                
                # Check if the episode is done
                done = all(terminated.values()) or all(truncated.values())
                
                # Render the environment if needed
                if render_interval is not None and episode % render_interval == 0:
                    self.environment.render()
            
            # End the episode
            self._end_episode()
            
            # Store episode statistics
            for agent_id, reward in episode_reward.items():
                episode_rewards[agent_id].append(reward)
            episode_lengths.append(episode_length)
            
            # Log episode statistics
            logger.info(
                f"Episode {episode+1}/{num_episodes}: "
                f"length={episode_length}, "
                f"rewards={episode_reward}"
            )
        
        # Calculate mean rewards
        mean_rewards = {agent_id: np.mean(rewards) for agent_id, rewards in episode_rewards.items()}
        mean_length = np.mean(episode_lengths)
        
        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_rewards": mean_rewards,
            "mean_length": mean_length,
            "total_steps": total_steps
        }
    
    async def save_models(self, directory: str) -> None:
        """Save the agents' models.
        
        Args:
            directory: Directory to save the models
        """
        os.makedirs(directory, exist_ok=True)
        
        for agent in self.agents:
            path = os.path.join(directory, f"{agent.agent_id}.pt")
            agent.save(path)
    
    async def load_models(self, directory: str) -> None:
        """Load the agents' models.
        
        Args:
            directory: Directory to load the models from
        """
        for agent in self.agents:
            path = os.path.join(directory, f"{agent.agent_id}.pt")
            if os.path.exists(path):
                agent.load(path)
