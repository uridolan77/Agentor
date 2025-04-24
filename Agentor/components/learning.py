"""
Learning components for the Agentor framework.

This module provides interfaces and implementations for agent learning components,
including reinforcement learning, supervised learning, and transfer learning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Generic
import numpy as np
import random
import logging
from collections import deque

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
S = TypeVar('S')  # State type
A = TypeVar('A')  # Action type
R = TypeVar('R')  # Reward type


class Experience:
    """A single experience tuple (state, action, reward, next_state, done)."""

    def __init__(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}

    def __str__(self) -> str:
        return f"Experience(state={self.state}, action={self.action}, reward={self.reward}, done={self.done})"


class ReplayBuffer:
    """A buffer for storing experiences."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()


class LearningAlgorithm(Generic[S, A, R], ABC):
    """Base class for learning algorithms."""

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Learning algorithm: {name}"

    @abstractmethod
    def update(self, experience: Experience) -> Dict[str, Any]:
        """Update the algorithm with a new experience."""
        pass

    @abstractmethod
    def select_action(self, state: S) -> A:
        """Select an action based on the current state."""
        pass

    def batch_update(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Update the algorithm with a batch of experiences."""
        metrics = {}
        for experience in experiences:
            update_metrics = self.update(experience)
            for key, value in update_metrics.items():
                if key in metrics:
                    metrics[key].append(value)
                else:
                    metrics[key] = [value]

        # Average the metrics
        return {key: np.mean(values) for key, values in metrics.items()}

    def save(self, path: str) -> None:
        """Save the algorithm to a file."""
        raise NotImplementedError("Subclasses must implement save()")

    def load(self, path: str) -> None:
        """Load the algorithm from a file."""
        raise NotImplementedError("Subclasses must implement load()")


class QLearning(LearningAlgorithm[S, A, float]):
    """Q-learning algorithm for discrete state and action spaces."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        name: str = "QLearning"
    ):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table: Dict[S, Dict[A, float]] = {}

    def update(self, experience: Experience) -> Dict[str, Any]:
        """Update the Q-table with a new experience."""
        state = experience.state
        action = experience.action
        reward = experience.reward
        next_state = experience.next_state
        done = experience.done

        # Initialize Q-values if not already present
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        # Get the current Q-value
        current_q = self.q_table[state][action]

        # Calculate the target Q-value
        if done:
            target_q = reward
        else:
            # Initialize next state Q-values if not already present
            if next_state not in self.q_table:
                self.q_table[next_state] = {}

            # Get the maximum Q-value for the next state
            if self.q_table[next_state]:
                max_next_q = max(self.q_table[next_state].values())
            else:
                max_next_q = 0.0

            target_q = reward + self.discount_factor * max_next_q

        # Update the Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

        return {
            "q_value": new_q,
            "target_q": target_q,
            "td_error": target_q - current_q,
            "exploration_rate": self.exploration_rate
        }

    def select_action(self, state: S) -> A:
        """Select an action based on the current state using epsilon-greedy policy."""
        # Initialize state Q-values if not already present
        if state not in self.q_table:
            self.q_table[state] = {}

        # Explore: choose a random action
        if random.random() < self.exploration_rate or not self.q_table[state]:
            # If we don't have any actions for this state, we need to get them from somewhere
            # This is environment-dependent, so we'll just return None and let the caller handle it
            return None

        # Exploit: choose the best action
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]

    def save(self, path: str) -> None:
        """Save the Q-table to a file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate
            }, f)

    def load(self, path: str) -> None:
        """Load the Q-table from a file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.exploration_rate = data['exploration_rate']
            self.exploration_decay = data['exploration_decay']
            self.min_exploration_rate = data['min_exploration_rate']


class DeepQLearning(LearningAlgorithm[np.ndarray, int, float]):
    """Deep Q-learning algorithm for continuous state spaces and discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        batch_size: int = 32,
        target_update_freq: int = 100,
        name: str = "DeepQLearning"
    ):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        # Initialize neural networks
        try:
            import tensorflow as tf
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.tf_available = True
        except ImportError:
            logger.warning("TensorFlow not available. DeepQLearning will not work.")
            self.tf_available = False

    def _build_model(self):
        """Build a neural network model for Q-learning."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        return model

    def update(self, experience: Experience) -> Dict[str, Any]:
        """Update the model with a new experience."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot update model.")
            return {}

        # Add experience to replay buffer
        self.replay_buffer.add(experience)

        # Only update if we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "exploration_rate": self.exploration_rate}

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)

        # Extract batch data
        import tensorflow as tf
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])

        # Compute target Q-values
        next_q_values = self.target_model.predict(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.discount_factor * max_next_q

        # Compute current Q-values and update
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            one_hot_actions = tf.one_hot(actions, self.action_dim)
            current_q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = self.loss_fn(target_q_values, current_q_values)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

        return {
            "loss": loss.numpy(),
            "exploration_rate": self.exploration_rate
        }

    def select_action(self, state: np.ndarray) -> int:
        """Select an action based on the current state using epsilon-greedy policy."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot select action.")
            return 0

        # Explore: choose a random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_dim - 1)

        # Exploit: choose the best action
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)

    def save(self, path: str) -> None:
        """Save the model to a file."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot save model.")
            return

        self.model.save(path)

    def load(self, path: str) -> None:
        """Load the model from a file."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot load model.")
            return

        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)


class PPOAgent(LearningAlgorithm[np.ndarray, np.ndarray, float]):
    """Proximal Policy Optimization (PPO) agent for continuous state and action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        name: str = "PPOAgent"
    ):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()

        # Initialize neural networks
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp

            # Actor network (policy)
            self.actor = self._build_actor()
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Critic network (value function)
            self.critic = self._build_critic()
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            self.tf_available = True
        except ImportError:
            logger.warning("TensorFlow or TensorFlow Probability not available. PPOAgent will not work.")
            self.tf_available = False

    def _build_actor(self):
        """Build the actor network (policy)."""
        import tensorflow as tf
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        mu = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)  # Mean of the Gaussian
        log_std = tf.keras.layers.Dense(self.action_dim, activation='linear')(x)  # Log standard deviation
        model = tf.keras.Model(inputs=inputs, outputs=[mu, log_std])
        return model

    def _build_critic(self):
        """Build the critic network (value function)."""
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model

    def update(self, experience: Experience) -> Dict[str, Any]:
        """Update the model with a new experience."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot update model.")
            return {}

        # Add experience to replay buffer
        self.replay_buffer.add(experience)

        # We'll implement the actual PPO update in batch_update
        return {}

    def batch_update(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Update the model with a batch of experiences."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot update model.")
            return {}

        import tensorflow as tf
        import tensorflow_probability as tfp

        # Extract batch data
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        # Compute advantages and returns
        values = self.critic(states).numpy().flatten()
        next_values = self.critic(next_states).numpy().flatten()

        # Compute TD errors
        deltas = rewards + (1 - dones) * self.discount_factor * next_values - values

        # Compute GAE (Generalized Advantage Estimation)
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Compute returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Get old action probabilities
        mu, log_std = self.actor(states)
        std = tf.exp(log_std)
        old_dist = tfp.distributions.Normal(mu, std)
        old_log_probs = old_dist.log_prob(actions).numpy()

        # PPO update
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # Multiple epochs of optimization
        for _ in range(5):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                # Actor loss
                mu, log_std = self.actor(states)
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(mu, std)
                new_log_probs = dist.log_prob(actions)

                # Compute ratio and clipped ratio
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                # Compute surrogate losses
                surrogate1 = ratio * advantages
                surrogate2 = clipped_ratio * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

                # Entropy loss
                entropy = dist.entropy()
                entropy_loss = -tf.reduce_mean(entropy)

                # Total actor loss
                total_actor_loss = actor_loss + self.entropy_coef * entropy_loss

                # Critic loss
                value_pred = self.critic(states)
                critic_loss = tf.reduce_mean(tf.square(returns - value_pred))

            # Backpropagation for actor
            actor_grads = actor_tape.gradient(total_actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Backpropagation for critic
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss.numpy())
            entropy_losses.append(entropy_loss.numpy())

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy_loss": np.mean(entropy_losses)
        }

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action based on the current state."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot select action.")
            return np.zeros(self.action_dim)

        import tensorflow as tf
        import tensorflow_probability as tfp

        state = np.expand_dims(state, axis=0)  # Add batch dimension
        mu, log_std = self.actor(state)
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mu, std)
        action = dist.sample()
        return action.numpy()[0]

    def save(self, path: str) -> None:
        """Save the model to a file."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot save model.")
            return

        self.actor.save(f"{path}_actor")
        self.critic.save(f"{path}_critic")

    def load(self, path: str) -> None:
        """Load the model from a file."""
        if not self.tf_available:
            logger.error("TensorFlow not available. Cannot load model.")
            return

        import tensorflow as tf
        self.actor = tf.keras.models.load_model(f"{path}_actor")
        self.critic = tf.keras.models.load_model(f"{path}_critic")