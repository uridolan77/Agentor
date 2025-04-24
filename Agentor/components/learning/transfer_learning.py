"""
Transfer Learning implementation for the Agentor framework.

This module provides mechanisms for transferring knowledge between agents,
allowing agents to leverage knowledge and skills learned by other agents.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type
import numpy as np
import random
import time
import logging
import asyncio
import os
import json
import copy

from agentor.agents import Agent
from agentor.agents.learning import EnhancedDeepQLearningAgent as DeepQLearningAgent
from agentor.agents.learning import EnhancedPPOAgent as PPOAgent

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Transfer learning will have limited functionality.")
    TORCH_AVAILABLE = False


class KnowledgeTransfer:
    """Base class for knowledge transfer between agents."""

    @staticmethod
    async def transfer(source_agent: Agent, target_agent: Agent, **kwargs) -> bool:
        """Transfer knowledge from source agent to target agent.

        Args:
            source_agent: The agent to transfer knowledge from
            target_agent: The agent to transfer knowledge to
            **kwargs: Additional parameters for the transfer

        Returns:
            True if the transfer was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement transfer()")


class ModelTransfer(KnowledgeTransfer):
    """Transfer neural network models between agents."""

    @staticmethod
    async def transfer(
        source_agent: Agent,
        target_agent: Agent,
        transfer_type: str = "full",
        layer_mapping: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> bool:
        """Transfer model parameters from source agent to target agent.

        Args:
            source_agent: The agent to transfer knowledge from
            target_agent: The agent to transfer knowledge to
            transfer_type: Type of transfer ('full', 'partial', or 'feature_extractor')
            layer_mapping: Mapping of layer names from source to target
            **kwargs: Additional parameters for the transfer

        Returns:
            True if the transfer was successful, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for ModelTransfer")
            return False

        # Check if both agents have models
        if not hasattr(source_agent, 'policy_net') and not hasattr(source_agent, 'network'):
            logger.error("Source agent does not have a neural network model")
            return False

        if not hasattr(target_agent, 'policy_net') and not hasattr(target_agent, 'network'):
            logger.error("Target agent does not have a neural network model")
            return False

        try:
            # Get the source model
            if hasattr(source_agent, 'policy_net'):
                source_model = source_agent.policy_net
            else:
                source_model = source_agent.network

            # Get the target model
            if hasattr(target_agent, 'policy_net'):
                target_model = target_agent.policy_net
            else:
                target_model = target_agent.network

            # Get source and target state dictionaries
            source_state_dict = source_model.state_dict()
            target_state_dict = target_model.state_dict()

            # Perform the transfer based on the transfer type
            if transfer_type == "full":
                # Full transfer: copy all matching parameters
                return await ModelTransfer._full_transfer(
                    source_state_dict,
                    target_state_dict,
                    target_model,
                    layer_mapping
                )

            elif transfer_type == "partial":
                # Partial transfer: copy only specified layers
                layers_to_transfer = kwargs.get('layers_to_transfer', [])
                return await ModelTransfer._partial_transfer(
                    source_state_dict,
                    target_state_dict,
                    target_model,
                    layers_to_transfer,
                    layer_mapping
                )

            elif transfer_type == "feature_extractor":
                # Feature extractor transfer: copy only the feature extraction layers
                return await ModelTransfer._feature_extractor_transfer(
                    source_state_dict,
                    target_state_dict,
                    target_model,
                    layer_mapping
                )

            else:
                logger.error(f"Unknown transfer type: {transfer_type}")
                return False

        except Exception as e:
            logger.error(f"Error during model transfer: {str(e)}")
            return False

    @staticmethod
    async def _full_transfer(
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        target_model: nn.Module,
        layer_mapping: Optional[Dict[str, str]] = None
    ) -> bool:
        """Perform a full transfer of parameters.

        Args:
            source_state_dict: Source model state dictionary
            target_state_dict: Target model state dictionary
            target_model: Target model
            layer_mapping: Mapping of layer names from source to target

        Returns:
            True if the transfer was successful, False otherwise
        """
        # Create a new state dictionary for the target
        new_state_dict = {}

        # Track which layers were transferred
        transferred_layers = []

        # Apply layer mapping if provided
        if layer_mapping is None:
            layer_mapping = {}

        # Copy parameters from source to target
        for target_key in target_state_dict.keys():
            # Check if there's a mapping for this key
            source_key = layer_mapping.get(target_key, target_key)

            # Check if the source has this key
            if source_key in source_state_dict:
                source_param = source_state_dict[source_key]
                target_param = target_state_dict[target_key]

                # Check if the shapes match
                if source_param.shape == target_param.shape:
                    new_state_dict[target_key] = source_param
                    transferred_layers.append(target_key)
                else:
                    # Keep the original parameter
                    new_state_dict[target_key] = target_param
                    logger.warning(
                        f"Shape mismatch for layer {target_key}: "
                        f"source {source_param.shape} vs target {target_param.shape}"
                    )
            else:
                # Keep the original parameter
                new_state_dict[target_key] = target_state_dict[target_key]

        # Load the new state dictionary
        target_model.load_state_dict(new_state_dict)

        logger.info(f"Transferred {len(transferred_layers)} layers: {', '.join(transferred_layers)}")

        return len(transferred_layers) > 0

    @staticmethod
    async def _partial_transfer(
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        target_model: nn.Module,
        layers_to_transfer: List[str],
        layer_mapping: Optional[Dict[str, str]] = None
    ) -> bool:
        """Perform a partial transfer of parameters.

        Args:
            source_state_dict: Source model state dictionary
            target_state_dict: Target model state dictionary
            target_model: Target model
            layers_to_transfer: List of layer names to transfer
            layer_mapping: Mapping of layer names from source to target

        Returns:
            True if the transfer was successful, False otherwise
        """
        # Create a new state dictionary for the target
        new_state_dict = target_state_dict.copy()

        # Track which layers were transferred
        transferred_layers = []

        # Apply layer mapping if provided
        if layer_mapping is None:
            layer_mapping = {}

        # Copy parameters for specified layers
        for target_key in layers_to_transfer:
            # Check if there's a mapping for this key
            source_key = layer_mapping.get(target_key, target_key)

            # Check if the source and target have this key
            if source_key in source_state_dict and target_key in target_state_dict:
                source_param = source_state_dict[source_key]
                target_param = target_state_dict[target_key]

                # Check if the shapes match
                if source_param.shape == target_param.shape:
                    new_state_dict[target_key] = source_param
                    transferred_layers.append(target_key)
                else:
                    logger.warning(
                        f"Shape mismatch for layer {target_key}: "
                        f"source {source_param.shape} vs target {target_param.shape}"
                    )
            else:
                if source_key not in source_state_dict:
                    logger.warning(f"Source model does not have layer {source_key}")
                if target_key not in target_state_dict:
                    logger.warning(f"Target model does not have layer {target_key}")

        # Load the new state dictionary
        target_model.load_state_dict(new_state_dict)

        logger.info(f"Transferred {len(transferred_layers)} layers: {', '.join(transferred_layers)}")

        return len(transferred_layers) > 0

    @staticmethod
    async def _feature_extractor_transfer(
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        target_model: nn.Module,
        layer_mapping: Optional[Dict[str, str]] = None
    ) -> bool:
        """Transfer feature extraction layers.

        Args:
            source_state_dict: Source model state dictionary
            target_state_dict: Target model state dictionary
            target_model: Target model
            layer_mapping: Mapping of layer names from source to target

        Returns:
            True if the transfer was successful, False otherwise
        """
        # Create a new state dictionary for the target
        new_state_dict = target_state_dict.copy()

        # Track which layers were transferred
        transferred_layers = []

        # Apply layer mapping if provided
        if layer_mapping is None:
            layer_mapping = {}

        # Identify feature extraction layers (typically early layers)
        # This is a heuristic and may need to be adjusted for different models
        feature_extraction_layers = []

        for key in target_state_dict.keys():
            # Typically, feature extraction layers are the first few convolutional or linear layers
            if 'fc1' in key or 'conv1' in key or 'features' in key:
                feature_extraction_layers.append(key)

        # Copy parameters for feature extraction layers
        for target_key in feature_extraction_layers:
            # Check if there's a mapping for this key
            source_key = layer_mapping.get(target_key, target_key)

            # Check if the source has this key
            if source_key in source_state_dict:
                source_param = source_state_dict[source_key]
                target_param = target_state_dict[target_key]

                # Check if the shapes match
                if source_param.shape == target_param.shape:
                    new_state_dict[target_key] = source_param
                    transferred_layers.append(target_key)
                else:
                    logger.warning(
                        f"Shape mismatch for layer {target_key}: "
                        f"source {source_param.shape} vs target {target_param.shape}"
                    )

        # Load the new state dictionary
        target_model.load_state_dict(new_state_dict)

        logger.info(f"Transferred {len(transferred_layers)} feature extraction layers: {', '.join(transferred_layers)}")

        return len(transferred_layers) > 0


class ExperienceTransfer(KnowledgeTransfer):
    """Transfer experiences between agents."""

    @staticmethod
    async def transfer(
        source_agent: Agent,
        target_agent: Agent,
        num_experiences: int = 1000,
        **kwargs
    ) -> bool:
        """Transfer experiences from source agent to target agent.

        Args:
            source_agent: The agent to transfer experiences from
            target_agent: The agent to transfer experiences to
            num_experiences: Number of experiences to transfer
            **kwargs: Additional parameters for the transfer

        Returns:
            True if the transfer was successful, False otherwise
        """
        # Check if both agents have memory buffers
        if not hasattr(source_agent, 'memory'):
            logger.error("Source agent does not have a memory buffer")
            return False

        if not hasattr(target_agent, 'memory'):
            logger.error("Target agent does not have a memory buffer")
            return False

        try:
            # Get the source memory
            source_memory = source_agent.memory

            # Check if the source has enough experiences
            if len(source_memory) < num_experiences:
                logger.warning(
                    f"Source agent has only {len(source_memory)} experiences, "
                    f"transferring all of them instead of {num_experiences}"
                )
                num_experiences = len(source_memory)

            # Sample experiences from the source
            if hasattr(random, 'sample') and isinstance(source_memory, list):
                experiences = random.sample(source_memory, num_experiences)
            else:
                # If the memory is not a list or random.sample is not available,
                # try to get the most recent experiences
                experiences = list(source_memory)[-num_experiences:]

            # Add experiences to the target
            for experience in experiences:
                target_agent.memory.append(copy.deepcopy(experience))

            logger.info(f"Transferred {len(experiences)} experiences")

            return True

        except Exception as e:
            logger.error(f"Error during experience transfer: {str(e)}")
            return False


class PolicyDistillation(KnowledgeTransfer):
    """Transfer knowledge through policy distillation."""

    @staticmethod
    async def transfer(
        source_agent: Agent,
        target_agent: Agent,
        num_samples: int = 1000,
        temperature: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 10,
        **kwargs
    ) -> bool:
        """Transfer knowledge through policy distillation.

        Args:
            source_agent: The teacher agent
            target_agent: The student agent
            num_samples: Number of state samples to use
            temperature: Temperature for softening the teacher's policy
            learning_rate: Learning rate for the student's optimizer
            batch_size: Batch size for training
            epochs: Number of epochs to train
            **kwargs: Additional parameters for the transfer

        Returns:
            True if the transfer was successful, False otherwise
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch is required for PolicyDistillation")
            return False

        # Check if both agents have the necessary attributes
        if not (hasattr(source_agent, 'policy_net') or hasattr(source_agent, 'network')):
            logger.error("Source agent does not have a policy network")
            return False

        if not (hasattr(target_agent, 'policy_net') or hasattr(target_agent, 'network')):
            logger.error("Target agent does not have a policy network")
            return False

        try:
            # Get the source model
            if hasattr(source_agent, 'policy_net'):
                source_model = source_agent.policy_net
            else:
                source_model = source_agent.network

            # Get the target model
            if hasattr(target_agent, 'policy_net'):
                target_model = target_agent.policy_net
            else:
                target_model = target_agent.network

            # Set up optimizer for the target model
            optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate)

            # Generate state samples
            # This is a simplified implementation - in a real system, we would use
            # states from the source agent's experience
            states = []

            # If the source agent has a memory buffer, sample states from it
            if hasattr(source_agent, 'memory') and len(source_agent.memory) > 0:
                for _ in range(min(num_samples, len(source_agent.memory))):
                    # Sample a random experience
                    experience = random.choice(source_agent.memory)

                    # Extract the state
                    if isinstance(experience, tuple) and len(experience) >= 1:
                        states.append(experience[0])
            else:
                # Generate more meaningful states based on the agent type and domain
                logger.info("Source agent has no memory buffer, generating domain-specific states")

                # Check if the agent has a state_space or observation_space attribute
                if hasattr(source_agent, 'state_space'):
                    states = PolicyDistillation._generate_states_from_space(source_agent.state_space, num_samples)
                elif hasattr(source_agent, 'observation_space'):
                    states = PolicyDistillation._generate_states_from_space(source_agent.observation_space, num_samples)
                # Check if the agent has a get_state_template method
                elif hasattr(source_agent, 'get_state_template'):
                    template = source_agent.get_state_template()
                    states = PolicyDistillation._generate_states_from_template(template, num_samples)
                # Check if the agent has a sample_state method
                elif hasattr(source_agent, 'sample_state'):
                    states = [source_agent.sample_state() for _ in range(num_samples)]
                # Check if the agent has a default_state attribute
                elif hasattr(source_agent, 'default_state'):
                    # Create variations of the default state
                    states = PolicyDistillation._generate_states_from_template(source_agent.default_state, num_samples)
                else:
                    # Fallback to a simple state structure with random values
                    logger.warning("Using generic fallback state generation - consider implementing a state template")
                    for _ in range(num_samples):
                        state = {}
                        for i in range(10):  # Assuming 10 state features
                            state[f"feature_{i}"] = random.random()
                        states.append(state)

            # Train the target model to mimic the source model
            total_loss = 0.0

            for epoch in range(epochs):
                epoch_loss = 0.0

                # Process in batches
                for i in range(0, len(states), batch_size):
                    batch_states = states[i:i+batch_size]

                    # Encode states
                    if hasattr(source_agent, 'encode_state'):
                        source_state_tensors = torch.cat([source_agent.encode_state(s) for s in batch_states])
                    else:
                        # Fallback encoding
                        source_state_tensors = torch.tensor(
                            [[float(v) for v in s.values()] for s in batch_states],
                            dtype=torch.float32
                        )

                    if hasattr(target_agent, 'encode_state'):
                        target_state_tensors = torch.cat([target_agent.encode_state(s) for s in batch_states])
                    else:
                        # Fallback encoding
                        target_state_tensors = torch.tensor(
                            [[float(v) for v in s.values()] for s in batch_states],
                            dtype=torch.float32
                        )

                    # Get teacher's outputs
                    with torch.no_grad():
                        if isinstance(source_agent, DeepQLearningAgent):
                            teacher_q_values = source_model(source_state_tensors)
                            # Convert Q-values to probabilities using softmax with temperature
                            teacher_probs = F.softmax(teacher_q_values / temperature, dim=1)
                        elif isinstance(source_agent, PPOAgent):
                            teacher_probs, _ = source_model(source_state_tensors)
                        else:
                            # Fallback: assume the model outputs action probabilities
                            teacher_outputs = source_model(source_state_tensors)
                            if isinstance(teacher_outputs, tuple):
                                teacher_probs = teacher_outputs[0]
                            else:
                                teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)

                    # Get student's outputs
                    if isinstance(target_agent, DeepQLearningAgent):
                        student_q_values = target_model(target_state_tensors)
                        # Convert Q-values to probabilities using softmax with temperature
                        student_probs = F.softmax(student_q_values / temperature, dim=1)
                    elif isinstance(target_agent, PPOAgent):
                        student_probs, _ = target_model(target_state_tensors)
                    else:
                        # Fallback: assume the model outputs action probabilities
                        student_outputs = target_model(target_state_tensors)
                        if isinstance(student_outputs, tuple):
                            student_probs = student_outputs[0]
                        else:
                            student_probs = F.softmax(student_outputs / temperature, dim=1)

                    # Compute KL divergence loss
                    loss = F.kl_div(
                        F.log_softmax(student_probs, dim=1),
                        teacher_probs,
                        reduction='batchmean'
                    )

                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # Log progress
                avg_epoch_loss = epoch_loss / ((len(states) + batch_size - 1) // batch_size)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

                total_loss += avg_epoch_loss

            avg_loss = total_loss / epochs
            logger.info(f"Policy distillation completed with average loss: {avg_loss:.6f}")

            return True

        except Exception as e:
            logger.error(f"Error during policy distillation: {str(e)}")
            return False


    @staticmethod
    def _generate_states_from_space(space, num_samples):
        """Generate states from a state/observation space.

        Args:
            space: The state or observation space
            num_samples: Number of states to generate

        Returns:
            A list of generated states
        """
        states = []

        # Check if the space has a sample method (like gym spaces)
        if hasattr(space, 'sample'):
            states = [space.sample() for _ in range(num_samples)]
        # Check if the space is a dict-like object
        elif hasattr(space, 'items'):
            for _ in range(num_samples):
                state = {}
                for key, subspace in space.items():
                    if hasattr(subspace, 'sample'):
                        state[key] = subspace.sample()
                    elif hasattr(subspace, 'low') and hasattr(subspace, 'high'):
                        # Assume it's a bounded space
                        if hasattr(subspace, 'shape'):
                            # Multi-dimensional space
                            state[key] = np.random.uniform(
                                subspace.low, subspace.high, subspace.shape
                            )
                        else:
                            # Scalar space
                            state[key] = np.random.uniform(subspace.low, subspace.high)
                    else:
                        # Default to random value
                        state[key] = random.random()
                states.append(state)
        # Check if the space has bounds
        elif hasattr(space, 'low') and hasattr(space, 'high'):
            if hasattr(space, 'shape'):
                # Multi-dimensional space
                states = [
                    np.random.uniform(space.low, space.high, space.shape)
                    for _ in range(num_samples)
                ]
            else:
                # Scalar space
                states = [
                    np.random.uniform(space.low, space.high)
                    for _ in range(num_samples)
                ]
        else:
            # Default to random values
            states = [random.random() for _ in range(num_samples)]

        return states

    @staticmethod
    def _generate_states_from_template(template, num_samples):
        """Generate states from a template state.

        Args:
            template: The template state
            num_samples: Number of states to generate

        Returns:
            A list of generated states
        """
        states = []

        for _ in range(num_samples):
            if isinstance(template, dict):
                # For dict templates, create variations of each value
                state = {}
                for key, value in template.items():
                    if isinstance(value, (int, float)):
                        # Add some noise to numeric values
                        state[key] = value * (0.8 + 0.4 * random.random())  # ±20% variation
                    elif isinstance(value, bool):
                        # Randomly flip boolean values with 20% probability
                        state[key] = value if random.random() > 0.2 else not value
                    elif isinstance(value, str):
                        # Keep strings as is
                        state[key] = value
                    elif isinstance(value, list) and len(value) > 0:
                        # For lists, randomly select an element or keep as is
                        if random.random() > 0.5 and all(isinstance(x, (int, float)) for x in value):
                            # If all elements are numeric, add some noise
                            state[key] = [x * (0.8 + 0.4 * random.random()) for x in value]
                        else:
                            # Otherwise keep as is
                            state[key] = value.copy()
                    else:
                        # For other types, keep as is
                        state[key] = copy.deepcopy(value)
            elif isinstance(template, np.ndarray):
                # For numpy arrays, add some noise
                state = template * (0.8 + 0.4 * np.random.random(template.shape))  # ±20% variation
            elif isinstance(template, list):
                # For lists, add some noise if numeric
                if all(isinstance(x, (int, float)) for x in template):
                    state = [x * (0.8 + 0.4 * random.random()) for x in template]  # ±20% variation
                else:
                    state = copy.deepcopy(template)
            else:
                # For other types, use as is
                state = copy.deepcopy(template)

            states.append(state)

        return states


class TransferLearningManager:
    """Manager for transfer learning operations."""

    def __init__(self):
        """Initialize the transfer learning manager."""
        self.transfer_methods = {
            "model": ModelTransfer,
            "experience": ExperienceTransfer,
            "policy_distillation": PolicyDistillation
        }

    async def transfer(
        self,
        source_agent: Agent,
        target_agent: Agent,
        method: str = "model",
        **kwargs
    ) -> bool:
        """Transfer knowledge from source agent to target agent.

        Args:
            source_agent: The agent to transfer knowledge from
            target_agent: The agent to transfer knowledge to
            method: The transfer method to use
            **kwargs: Additional parameters for the transfer

        Returns:
            True if the transfer was successful, False otherwise
        """
        if method not in self.transfer_methods:
            logger.error(f"Unknown transfer method: {method}")
            return False

        transfer_class = self.transfer_methods[method]
        return await transfer_class.transfer(source_agent, target_agent, **kwargs)

    def register_transfer_method(self, name: str, method: Type[KnowledgeTransfer]):
        """Register a new transfer method.

        Args:
            name: The name of the method
            method: The transfer method class
        """
        self.transfer_methods[name] = method
        logger.info(f"Registered transfer method: {name}")

    async def multi_agent_transfer(
        self,
        source_agents: List[Agent],
        target_agent: Agent,
        method: str = "model",
        **kwargs
    ) -> bool:
        """Transfer knowledge from multiple source agents to a target agent.

        Args:
            source_agents: The agents to transfer knowledge from
            target_agent: The agent to transfer knowledge to
            method: The transfer method to use
            **kwargs: Additional parameters for the transfer

        Returns:
            True if the transfer was successful, False otherwise
        """
        success = True

        for i, source_agent in enumerate(source_agents):
            logger.info(f"Transferring from agent {i+1}/{len(source_agents)}")

            # Transfer from this source agent
            result = await self.transfer(source_agent, target_agent, method, **kwargs)

            # Update overall success
            success = success and result

        return success
