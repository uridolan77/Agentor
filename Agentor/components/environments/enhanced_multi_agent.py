"""
Enhanced multi-agent environments for the Agentor framework.

This module provides enhanced multi-agent environment implementations with
additional features such as communication channels, team-based rewards,
and dynamic agent populations.
"""

import logging
import asyncio
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from abc import ABC, abstractmethod
import numpy as np

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace, TextSpace
)
from agentor.components.environments.base import BaseEnvironment
from agentor.components.environments.multi_agent import IMultiAgentEnvironment, MultiAgentEnv

logger = logging.getLogger(__name__)


class Message:
    """A message between agents in a multi-agent environment."""
    
    def __init__(
        self,
        sender: str,
        content: Any,
        recipients: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a message.
        
        Args:
            sender: The ID of the agent sending the message
            content: The content of the message
            recipients: List of agent IDs to receive the message, or None for broadcast
            metadata: Additional metadata for the message
        """
        self.sender = sender
        self.content = content
        self.recipients = recipients
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def is_recipient(self, agent_id: str) -> bool:
        """Check if an agent is a recipient of this message.
        
        Args:
            agent_id: The agent ID to check
            
        Returns:
            True if the agent is a recipient
        """
        return self.recipients is None or agent_id in self.recipients
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "sender": self.sender,
            "content": self.content,
            "recipients": self.recipients,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary.
        
        Args:
            data: Dictionary representation of the message
            
        Returns:
            A new message
        """
        return cls(
            sender=data["sender"],
            content=data["content"],
            recipients=data["recipients"],
            metadata=data["metadata"]
        )


class CommunicationChannel:
    """A communication channel for agents in a multi-agent environment."""
    
    def __init__(
        self,
        name: str,
        message_limit: Optional[int] = None,
        delivery_delay: Optional[float] = None,
        delivery_probability: float = 1.0,
        content_filter: Optional[Callable[[Any], Any]] = None
    ):
        """Initialize a communication channel.
        
        Args:
            name: The name of the channel
            message_limit: Maximum number of messages to store, or None for unlimited
            delivery_delay: Delay in seconds before messages are delivered, or None for immediate
            delivery_probability: Probability of message delivery (0.0 to 1.0)
            content_filter: Function to filter or transform message content
        """
        self.name = name
        self.message_limit = message_limit
        self.delivery_delay = delivery_delay
        self.delivery_probability = delivery_probability
        self.content_filter = content_filter
        
        self.messages: List[Message] = []
        self.pending_messages: List[Tuple[Message, float]] = []  # (message, delivery_time)
    
    def send_message(self, message: Message) -> bool:
        """Send a message through the channel.
        
        Args:
            message: The message to send
            
        Returns:
            True if the message was accepted
        """
        # Check if the message will be delivered
        if random.random() > self.delivery_probability:
            logger.debug(f"Message from {message.sender} dropped due to delivery probability")
            return False
        
        # Apply content filter if provided
        if self.content_filter is not None:
            message.content = self.content_filter(message.content)
        
        # Add delivery delay if specified
        if self.delivery_delay is not None:
            delivery_time = time.time() + self.delivery_delay
            self.pending_messages.append((message, delivery_time))
        else:
            # Add to messages immediately
            self._add_message(message)
        
        return True
    
    def _add_message(self, message: Message) -> None:
        """Add a message to the channel.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        
        # Enforce message limit if specified
        if self.message_limit is not None and len(self.messages) > self.message_limit:
            self.messages.pop(0)  # Remove oldest message
    
    def get_messages(self, agent_id: Optional[str] = None) -> List[Message]:
        """Get messages from the channel.
        
        Args:
            agent_id: The agent ID to filter messages for, or None for all messages
            
        Returns:
            List of messages
        """
        # Process pending messages
        current_time = time.time()
        remaining_pending = []
        
        for message, delivery_time in self.pending_messages:
            if current_time >= delivery_time:
                self._add_message(message)
            else:
                remaining_pending.append((message, delivery_time))
        
        self.pending_messages = remaining_pending
        
        # Filter messages for the agent if specified
        if agent_id is not None:
            return [msg for msg in self.messages if msg.is_recipient(agent_id)]
        
        return self.messages
    
    def clear(self) -> None:
        """Clear all messages from the channel."""
        self.messages.clear()
        self.pending_messages.clear()


class CommunicativeMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment with communication capabilities."""
    
    def __init__(
        self,
        agent_ids: List[str],
        observation_spaces: Dict[str, Space],
        action_spaces: Dict[str, Space],
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        communication_enabled: bool = True,
        message_space: Optional[Space] = None
    ):
        """Initialize the communicative multi-agent environment.
        
        Args:
            agent_ids: List of agent IDs
            observation_spaces: Dictionary mapping agent IDs to observation spaces
            action_spaces: Dictionary mapping agent IDs to action spaces
            max_episode_steps: Maximum number of steps per episode
            auto_reset: Whether to automatically reset the environment when done
            communication_enabled: Whether communication between agents is enabled
            message_space: Space for message content, or None for text messages
        """
        super().__init__(
            agent_ids=agent_ids,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        self.communication_enabled = communication_enabled
        
        # Set up message space
        self.message_space = message_space or TextSpace()
        
        # Set up communication channels
        self.channels: Dict[str, CommunicationChannel] = {
            "global": CommunicationChannel(name="global")
        }
        
        # Update observation spaces to include messages
        if communication_enabled:
            for agent_id in agent_ids:
                # Create a space for messages
                message_list_space = DictSpace({
                    "sender": DiscreteSpace(len(agent_ids)),
                    "content": self.message_space,
                    "timestamp": BoxSpace(low=0, high=np.inf, shape=(1,), dtype=np.float32)
                })
                
                # Add to the agent's observation space
                if isinstance(observation_spaces[agent_id], DictSpace):
                    spaces = observation_spaces[agent_id].spaces.copy()
                    spaces["messages"] = message_list_space
                    observation_spaces[agent_id] = DictSpace(spaces)
                else:
                    # If not a dict space, create a new dict space
                    observation_spaces[agent_id] = DictSpace({
                        "observation": observation_spaces[agent_id],
                        "messages": message_list_space
                    })
    
    def add_channel(self, channel: CommunicationChannel) -> None:
        """Add a communication channel.
        
        Args:
            channel: The channel to add
        """
        if channel.name in self.channels:
            logger.warning(f"Channel {channel.name} already exists, replacing")
        
        self.channels[channel.name] = channel
    
    def remove_channel(self, name: str) -> None:
        """Remove a communication channel.
        
        Args:
            name: The name of the channel to remove
        """
        if name not in self.channels:
            logger.warning(f"Channel {name} does not exist")
            return
        
        if name == "global":
            logger.warning("Cannot remove the global channel")
            return
        
        del self.channels[name]
    
    def send_message(
        self,
        sender: str,
        content: Any,
        recipients: Optional[List[str]] = None,
        channel: str = "global",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a message from an agent.
        
        Args:
            sender: The ID of the agent sending the message
            content: The content of the message
            recipients: List of agent IDs to receive the message, or None for broadcast
            channel: The name of the channel to send the message on
            metadata: Additional metadata for the message
            
        Returns:
            True if the message was sent successfully
        """
        if not self.communication_enabled:
            logger.warning("Communication is disabled")
            return False
        
        if sender not in self._agent_ids:
            logger.warning(f"Unknown sender: {sender}")
            return False
        
        if channel not in self.channels:
            logger.warning(f"Unknown channel: {channel}")
            return False
        
        # Validate recipients
        if recipients is not None:
            for recipient in recipients:
                if recipient not in self._agent_ids:
                    logger.warning(f"Unknown recipient: {recipient}")
                    return False
        
        # Validate content against the message space
        if not self.message_space.contains(content):
            logger.warning(f"Invalid message content: {content}")
            return False
        
        # Create and send the message
        message = Message(
            sender=sender,
            content=content,
            recipients=recipients,
            metadata=metadata
        )
        
        return self.channels[channel].send_message(message)
    
    def get_messages(self, agent_id: str, channel: str = "global") -> List[Message]:
        """Get messages for an agent.
        
        Args:
            agent_id: The ID of the agent
            channel: The name of the channel to get messages from
            
        Returns:
            List of messages
        """
        if not self.communication_enabled:
            return []
        
        if agent_id not in self._agent_ids:
            logger.warning(f"Unknown agent: {agent_id}")
            return []
        
        if channel not in self.channels:
            logger.warning(f"Unknown channel: {channel}")
            return []
        
        return self.channels[channel].get_messages(agent_id)
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observations, info)
        """
        # Reset all communication channels
        for channel in self.channels.values():
            channel.clear()
        
        # Call the parent implementation
        observations, info = super()._reset_impl(options)
        
        # Add empty message lists to observations
        if self.communication_enabled:
            for agent_id in self._agent_ids:
                if isinstance(observations[agent_id], dict):
                    observations[agent_id]["messages"] = []
                else:
                    observations[agent_id] = {
                        "observation": observations[agent_id],
                        "messages": []
                    }
        
        return observations, info
    
    def _step_impl(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Process communication actions if included
        if self.communication_enabled:
            for agent_id, action in actions.items():
                if isinstance(action, dict) and "message" in action:
                    message_action = action["message"]
                    
                    if isinstance(message_action, dict):
                        # Extract message parameters
                        content = message_action.get("content")
                        recipients = message_action.get("recipients")
                        channel = message_action.get("channel", "global")
                        metadata = message_action.get("metadata")
                        
                        if content is not None:
                            self.send_message(
                                sender=agent_id,
                                content=content,
                                recipients=recipients,
                                channel=channel,
                                metadata=metadata
                            )
        
        # Call the parent implementation
        observations, rewards, terminated, truncated, info = super()._step_impl(actions)
        
        # Add messages to observations
        if self.communication_enabled:
            for agent_id in self._agent_ids:
                messages = self.get_messages(agent_id)
                
                if isinstance(observations[agent_id], dict):
                    observations[agent_id]["messages"] = messages
                else:
                    observations[agent_id] = {
                        "observation": observations[agent_id],
                        "messages": messages
                    }
        
        return observations, rewards, terminated, truncated, info


class TeamBasedMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment with team-based dynamics."""
    
    def __init__(
        self,
        agent_ids: List[str],
        team_memberships: Dict[str, str],
        observation_spaces: Dict[str, Space],
        action_spaces: Dict[str, Space],
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        shared_team_reward: bool = True,
        team_reward_scale: float = 1.0,
        individual_reward_scale: float = 1.0
    ):
        """Initialize the team-based multi-agent environment.
        
        Args:
            agent_ids: List of agent IDs
            team_memberships: Dictionary mapping agent IDs to team IDs
            observation_spaces: Dictionary mapping agent IDs to observation spaces
            action_spaces: Dictionary mapping agent IDs to action spaces
            max_episode_steps: Maximum number of steps per episode
            auto_reset: Whether to automatically reset the environment when done
            shared_team_reward: Whether team members share rewards
            team_reward_scale: Scale factor for team rewards
            individual_reward_scale: Scale factor for individual rewards
        """
        super().__init__(
            agent_ids=agent_ids,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        self.team_memberships = team_memberships
        self.shared_team_reward = shared_team_reward
        self.team_reward_scale = team_reward_scale
        self.individual_reward_scale = individual_reward_scale
        
        # Validate team memberships
        for agent_id in agent_ids:
            if agent_id not in team_memberships:
                logger.warning(f"Agent {agent_id} has no team membership, assigning to 'default'")
                team_memberships[agent_id] = "default"
        
        # Get the set of teams
        self.teams = set(team_memberships.values())
        
        # Create a mapping from teams to agents
        self.team_agents: Dict[str, List[str]] = {}
        for agent_id, team_id in team_memberships.items():
            if team_id not in self.team_agents:
                self.team_agents[team_id] = []
            self.team_agents[team_id].append(agent_id)
    
    def get_team(self, agent_id: str) -> str:
        """Get the team ID for an agent.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            The team ID
        """
        return self.team_memberships.get(agent_id, "default")
    
    def get_team_members(self, team_id: str) -> List[str]:
        """Get the members of a team.
        
        Args:
            team_id: The team ID
            
        Returns:
            List of agent IDs in the team
        """
        return self.team_agents.get(team_id, [])
    
    def get_team_reward(self, team_id: str, individual_rewards: Dict[str, float]) -> float:
        """Calculate the reward for a team.
        
        Args:
            team_id: The team ID
            individual_rewards: Dictionary mapping agent IDs to individual rewards
            
        Returns:
            The team reward
        """
        team_members = self.get_team_members(team_id)
        
        if not team_members:
            return 0.0
        
        # Sum the rewards of all team members
        team_reward = sum(individual_rewards.get(agent_id, 0.0) for agent_id in team_members)
        
        # Scale by the number of team members if sharing rewards
        if self.shared_team_reward:
            team_reward /= len(team_members)
        
        return team_reward * self.team_reward_scale
    
    def _step_impl(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Call the parent implementation
        observations, individual_rewards, terminated, truncated, info = super()._step_impl(actions)
        
        # Calculate team rewards
        team_rewards: Dict[str, float] = {}
        for team_id in self.teams:
            team_rewards[team_id] = self.get_team_reward(team_id, individual_rewards)
        
        # Combine individual and team rewards
        final_rewards: Dict[str, float] = {}
        for agent_id in self._agent_ids:
            team_id = self.get_team(agent_id)
            individual_reward = individual_rewards.get(agent_id, 0.0) * self.individual_reward_scale
            team_reward = team_rewards.get(team_id, 0.0)
            
            final_rewards[agent_id] = individual_reward + team_reward
        
        # Add team information to info
        info["teams"] = {
            "memberships": self.team_memberships,
            "team_rewards": team_rewards
        }
        
        return observations, final_rewards, terminated, truncated, info


class DynamicMultiAgentEnv(MultiAgentEnv):
    """Multi-agent environment with dynamic agent population."""
    
    def __init__(
        self,
        initial_agent_ids: List[str],
        observation_spaces: Dict[str, Space],
        action_spaces: Dict[str, Space],
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        max_agents: Optional[int] = None
    ):
        """Initialize the dynamic multi-agent environment.
        
        Args:
            initial_agent_ids: Initial list of agent IDs
            observation_spaces: Dictionary mapping agent IDs to observation spaces
            action_spaces: Dictionary mapping agent IDs to action spaces
            max_episode_steps: Maximum number of steps per episode
            auto_reset: Whether to automatically reset the environment when done
            max_agents: Maximum number of agents allowed, or None for unlimited
        """
        super().__init__(
            agent_ids=initial_agent_ids,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        self.max_agents = max_agents
        self.active_agents: Set[str] = set(initial_agent_ids)
        self.inactive_agents: Set[str] = set()
    
    def add_agent(
        self,
        agent_id: str,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None
    ) -> bool:
        """Add a new agent to the environment.
        
        Args:
            agent_id: The ID of the agent to add
            observation_space: The observation space for the agent, or None to use default
            action_space: The action space for the agent, or None to use default
            
        Returns:
            True if the agent was added successfully
        """
        # Check if the agent already exists
        if agent_id in self._agent_ids:
            logger.warning(f"Agent {agent_id} already exists")
            
            # If the agent is inactive, reactivate it
            if agent_id in self.inactive_agents:
                self.inactive_agents.remove(agent_id)
                self.active_agents.add(agent_id)
                logger.info(f"Reactivated agent {agent_id}")
                return True
            
            return False
        
        # Check if we've reached the maximum number of agents
        if self.max_agents is not None and len(self._agent_ids) >= self.max_agents:
            logger.warning(f"Cannot add agent {agent_id}, maximum number of agents reached")
            return False
        
        # Use default spaces if not provided
        if observation_space is None:
            # Use the first agent's observation space as default
            default_agent_id = next(iter(self._observation_spaces.keys()))
            observation_space = self._observation_spaces[default_agent_id]
        
        if action_space is None:
            # Use the first agent's action space as default
            default_agent_id = next(iter(self._action_spaces.keys()))
            action_space = self._action_spaces[default_agent_id]
        
        # Add the agent
        self._agent_ids.append(agent_id)
        self._observation_spaces[agent_id] = observation_space
        self._action_spaces[agent_id] = action_space
        self.active_agents.add(agent_id)
        
        logger.info(f"Added agent {agent_id}")
        return True
    
    def remove_agent(self, agent_id: str, permanent: bool = False) -> bool:
        """Remove an agent from the environment.
        
        Args:
            agent_id: The ID of the agent to remove
            permanent: Whether to permanently remove the agent
            
        Returns:
            True if the agent was removed successfully
        """
        # Check if the agent exists
        if agent_id not in self._agent_ids:
            logger.warning(f"Agent {agent_id} does not exist")
            return False
        
        # Remove the agent
        if permanent:
            self._agent_ids.remove(agent_id)
            del self._observation_spaces[agent_id]
            del self._action_spaces[agent_id]
            
            if agent_id in self.active_agents:
                self.active_agents.remove(agent_id)
            
            if agent_id in self.inactive_agents:
                self.inactive_agents.remove(agent_id)
            
            logger.info(f"Permanently removed agent {agent_id}")
        else:
            # Just mark the agent as inactive
            if agent_id in self.active_agents:
                self.active_agents.remove(agent_id)
                self.inactive_agents.add(agent_id)
                logger.info(f"Deactivated agent {agent_id}")
            else:
                logger.warning(f"Agent {agent_id} is already inactive")
                return False
        
        return True
    
    def is_active(self, agent_id: str) -> bool:
        """Check if an agent is active.
        
        Args:
            agent_id: The agent ID to check
            
        Returns:
            True if the agent is active
        """
        return agent_id in self.active_agents
    
    def _step_impl(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Filter actions to only include active agents
        active_actions = {agent_id: action for agent_id, action in actions.items() if agent_id in self.active_agents}
        
        # Call the parent implementation with active agents only
        observations, rewards, terminated, truncated, info = super()._step_impl(active_actions)
        
        # Add information about agent status
        info["agents"] = {
            "active": list(self.active_agents),
            "inactive": list(self.inactive_agents),
            "total": len(self._agent_ids)
        }
        
        return observations, rewards, terminated, truncated, info


class CompetitiveMultiAgentEnv(TeamBasedMultiAgentEnv):
    """Multi-agent environment with competitive dynamics."""
    
    def __init__(
        self,
        agent_ids: List[str],
        team_memberships: Dict[str, str],
        observation_spaces: Dict[str, Space],
        action_spaces: Dict[str, Space],
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True,
        zero_sum: bool = True,
        competitive_reward_scale: float = 1.0
    ):
        """Initialize the competitive multi-agent environment.
        
        Args:
            agent_ids: List of agent IDs
            team_memberships: Dictionary mapping agent IDs to team IDs
            observation_spaces: Dictionary mapping agent IDs to observation spaces
            action_spaces: Dictionary mapping agent IDs to action spaces
            max_episode_steps: Maximum number of steps per episode
            auto_reset: Whether to automatically reset the environment when done
            zero_sum: Whether the environment is zero-sum
            competitive_reward_scale: Scale factor for competitive rewards
        """
        super().__init__(
            agent_ids=agent_ids,
            team_memberships=team_memberships,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset,
            shared_team_reward=True,
            team_reward_scale=1.0,
            individual_reward_scale=1.0
        )
        
        self.zero_sum = zero_sum
        self.competitive_reward_scale = competitive_reward_scale
    
    def _step_impl(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        # Call the parent implementation
        observations, rewards, terminated, truncated, info = super()._step_impl(actions)
        
        # Get team rewards
        team_rewards = info["teams"]["team_rewards"]
        
        # Apply competitive dynamics
        if self.zero_sum:
            # In a zero-sum game, one team's gain is another team's loss
            total_reward = sum(team_rewards.values())
            
            # Adjust rewards to make the total zero
            for team_id in team_rewards:
                team_members = self.get_team_members(team_id)
                team_reward = team_rewards[team_id]
                
                # Calculate the competitive reward
                competitive_reward = team_reward - (total_reward - team_reward) / (len(self.teams) - 1)
                competitive_reward *= self.competitive_reward_scale
                
                # Apply to all team members
                for agent_id in team_members:
                    rewards[agent_id] = competitive_reward
        else:
            # In a non-zero-sum game, teams compete but the total reward can vary
            for team_id in team_rewards:
                team_members = self.get_team_members(team_id)
                team_reward = team_rewards[team_id]
                
                # Calculate the competitive reward
                other_teams_reward = sum(r for tid, r in team_rewards.items() if tid != team_id)
                competitive_reward = team_reward - other_teams_reward * 0.5
                competitive_reward *= self.competitive_reward_scale
                
                # Apply to all team members
                for agent_id in team_members:
                    rewards[agent_id] = competitive_reward
        
        # Update info with competitive rewards
        info["competitive_rewards"] = {
            agent_id: rewards[agent_id] for agent_id in self._agent_ids
        }
        
        return observations, rewards, terminated, truncated, info
