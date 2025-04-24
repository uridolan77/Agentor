"""
Simple environments for the Agentor framework.

This module provides simple environments for testing and demonstration purposes.
These environments are intentionally simple to make it easy to understand how
the environment system works.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import random
import time
import math

from agentor.environments.custom import (
    Environment, EnvironmentState, ActionResult, EnvironmentRegistry
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class BanditAction(Enum):
    """Actions for the multi-armed bandit environment."""
    ARM_1 = 0
    ARM_2 = 1
    ARM_3 = 2
    ARM_4 = 3
    ARM_5 = 4


class BanditState:
    """State of the multi-armed bandit environment."""

    def __init__(
        self,
        arm_values: Dict[BanditAction, float],
        arm_counts: Dict[BanditAction, int],
        step_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.arm_values = arm_values
        self.arm_counts = arm_counts
        self.step_count = step_count
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return f"BanditState(step_count={self.step_count}, arm_counts={self.arm_counts})"


class MultiArmedBandit(Environment[BanditState, BanditAction, float, Dict[str, Any]]):
    """A simple multi-armed bandit environment.

    This environment simulates a multi-armed bandit problem where the agent must
    choose between different arms, each with a different reward distribution.
    """

    def __init__(
        self,
        num_arms: int = 5,
        mean_range: Tuple[float, float] = (0.0, 1.0),
        std_dev: float = 0.1,
        max_steps: Optional[int] = 100,
        seed: Optional[int] = None,
        name: str = "MultiArmedBandit",
        description: Optional[str] = None
    ):
        description = description or f"Multi-armed bandit environment with {num_arms} arms"
        super().__init__(name=name, description=description)

        self.num_arms = min(num_arms, len(BanditAction))
        self.mean_range = mean_range
        self.std_dev = std_dev
        self.max_steps = max_steps

        # Set the random seed
        self.seed(seed)

        # Initialize the arm values
        self.arm_means = {}
        self._initialize_arms()

        # Initialize the state
        self.state = None

    def _initialize_arms(self) -> None:
        """Initialize the arm values."""
        self.arm_means = {}
        for i in range(self.num_arms):
            action = BanditAction(i)
            mean = random.uniform(self.mean_range[0], self.mean_range[1])
            self.arm_means[action] = mean

        logger.debug(f"Initialized arm means: {self.arm_means}")

    def reset(self) -> EnvironmentState[BanditState]:
        """Reset the environment to its initial state."""
        # Reset episode counters
        self.episode_count += 1
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_start_time = time.time()

        # Initialize the arm values
        self._initialize_arms()

        # Initialize the arm counts
        arm_counts = {action: 0 for action in BanditAction if action.value < self.num_arms}

        # Create the initial state
        bandit_state = BanditState(
            arm_values=self.arm_means.copy(),
            arm_counts=arm_counts,
            step_count=0
        )

        # Create the environment state
        self.state = EnvironmentState(
            data=bandit_state,
            metadata={
                "episode": self.episode_count,
                "max_steps": self.max_steps
            }
        )

        logger.debug(f"Reset multi-armed bandit {self.name}")
        return self.state

    def step(self, action: BanditAction) -> ActionResult[float]:
        """Take a step in the environment by performing an action."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Check if the action is valid
        if action.value >= self.num_arms:
            raise ValueError(f"Invalid action: {action}. Only {self.num_arms} arms are available.")

        # Get the current state
        bandit_state = self.state.data

        # Get the arm mean
        arm_mean = self.arm_means[action]

        # Generate a reward from a normal distribution
        reward = random.normalvariate(arm_mean, self.std_dev)

        # Update the arm counts
        bandit_state.arm_counts[action] += 1

        # Check if the maximum number of steps has been reached
        self.step_count += 1
        done = self.max_steps is not None and self.step_count >= self.max_steps

        # Update the state
        bandit_state.step_count = self.step_count

        # Update the environment state
        self.state = EnvironmentState(
            data=bandit_state,
            metadata={
                **self.state.metadata,
                "step": self.step_count
            }
        )

        # Update rewards
        self.total_reward += reward
        self.episode_reward += reward

        # Create the action result
        result = ActionResult(
            success=True,
            reward=reward,
            observation=bandit_state,
            done=done,
            info={
                "arm_mean": arm_mean,
                "arm_counts": bandit_state.arm_counts.copy()
            }
        )

        logger.debug(
            f"Step {self.step_count} in multi-armed bandit {self.name}, "
            f"action: {action.name}, reward: {reward}, done: {done}"
        )

        return result

    def render(self, mode: str = 'human') -> Optional[Dict[str, Any]]:
        """Render the environment."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Get the current state
        bandit_state = self.state.data

        if mode == 'human':
            # Print the state to the console
            print(f"Step: {self.step_count}")
            print("Arm counts:")
            for action, count in bandit_state.arm_counts.items():
                print(f"  {action.name}: {count}")
            print()
            return None
        elif mode == 'dict':
            # Return the state as a dictionary
            return {
                "step": self.step_count,
                "arm_counts": {action.name: count for action, count in bandit_state.arm_counts.items()},
                "arm_means": {action.name: mean for action, mean in self.arm_means.items()}
            }
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed for the environment."""
        if seed is not None:
            random.seed(seed)
        return [seed] if seed is not None else []

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass


class CartPoleState:
    """State of the cart-pole environment."""

    def __init__(
        self,
        position: float,
        velocity: float,
        angle: float,
        angular_velocity: float,
        step_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.step_count = step_count
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return (
            f"CartPoleState(position={self.position:.2f}, velocity={self.velocity:.2f}, "
            f"angle={self.angle:.2f}, angular_velocity={self.angular_velocity:.2f})"
        )

    def to_array(self) -> List[float]:
        """Convert the state to a list of floats."""
        return [self.position, self.velocity, self.angle, self.angular_velocity]


class CartPoleAction(Enum):
    """Actions for the cart-pole environment."""
    LEFT = 0
    RIGHT = 1


class CartPole(Environment[CartPoleState, CartPoleAction, float, List[float]]):
    """A simple cart-pole environment.

    This environment simulates a cart-pole system where the agent must balance a
    pole on a cart by moving the cart left or right.
    """

    def __init__(
        self,
        gravity: float = 9.8,
        cart_mass: float = 1.0,
        pole_mass: float = 0.1,
        pole_length: float = 0.5,
        force_magnitude: float = 10.0,
        time_step: float = 0.02,
        max_steps: Optional[int] = 200,
        position_threshold: float = 2.4,
        angle_threshold: float = 12.0 * math.pi / 180.0,  # 12 degrees in radians
        seed: Optional[int] = None,
        name: str = "CartPole",
        description: Optional[str] = None
    ):
        description = description or "Cart-pole balancing environment"
        super().__init__(name=name, description=description)

        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.force_magnitude = force_magnitude
        self.time_step = time_step
        self.max_steps = max_steps
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold

        # Set the random seed
        self.seed(seed)

        # Initialize the state
        self.state = None

    def reset(self) -> EnvironmentState[CartPoleState]:
        """Reset the environment to its initial state."""
        # Reset episode counters
        self.episode_count += 1
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_start_time = time.time()

        # Initialize the state with small random values
        cart_pole_state = CartPoleState(
            position=random.uniform(-0.05, 0.05),
            velocity=random.uniform(-0.05, 0.05),
            angle=random.uniform(-0.05, 0.05),
            angular_velocity=random.uniform(-0.05, 0.05),
            step_count=0
        )

        # Create the environment state
        self.state = EnvironmentState(
            data=cart_pole_state,
            metadata={
                "episode": self.episode_count,
                "max_steps": self.max_steps
            }
        )

        logger.debug(f"Reset cart-pole {self.name}")
        return self.state

    def step(self, action: CartPoleAction) -> ActionResult[float]:
        """Take a step in the environment by performing an action."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Get the current state
        cart_pole_state = self.state.data

        # Extract state variables
        x = cart_pole_state.position
        x_dot = cart_pole_state.velocity
        theta = cart_pole_state.angle
        theta_dot = cart_pole_state.angular_velocity

        # Apply force based on action
        force = self.force_magnitude if action == CartPoleAction.RIGHT else -self.force_magnitude

        # Calculate the acceleration of the cart and pole
        # These equations are based on the dynamics of the cart-pole system
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.pole_length

        # Calculate the acceleration of the pole
        temp = (force + pole_mass_length * theta_dot**2 * math.sin(theta)) / total_mass
        theta_acc = (self.gravity * math.sin(theta) - math.cos(theta) * temp) / \
            (self.pole_length * (4.0/3.0 - self.pole_mass * math.cos(theta)**2 / total_mass))

        # Calculate the acceleration of the cart
        x_acc = temp - pole_mass_length * theta_acc * math.cos(theta) / total_mass

        # Update the state using Euler integration
        x = x + self.time_step * x_dot
        x_dot = x_dot + self.time_step * x_acc
        theta = theta + self.time_step * theta_dot
        theta_dot = theta_dot + self.time_step * theta_acc

        # Create the new state
        new_cart_pole_state = CartPoleState(
            position=x,
            velocity=x_dot,
            angle=theta,
            angular_velocity=theta_dot,
            step_count=self.step_count + 1
        )

        # Check if the episode is done
        done = False
        info = {}

        # Check if the cart has gone out of bounds
        if abs(x) > self.position_threshold:
            done = True
            info["position_out_of_bounds"] = True

        # Check if the pole has fallen over
        if abs(theta) > self.angle_threshold:
            done = True
            info["pole_fallen"] = True

        # Check if the maximum number of steps has been reached
        self.step_count += 1
        if self.max_steps is not None and self.step_count >= self.max_steps:
            done = True
            info["max_steps_reached"] = True

        # Update the environment state
        self.state = EnvironmentState(
            data=new_cart_pole_state,
            metadata={
                **self.state.metadata,
                "step": self.step_count
            }
        )

        # Calculate the reward (1 for each step the pole is balanced)
        reward = 1.0 if not done else 0.0

        # Update rewards
        self.total_reward += reward
        self.episode_reward += reward

        # Create the action result
        result = ActionResult(
            success=True,
            reward=reward,
            observation=new_cart_pole_state,
            done=done,
            info=info
        )

        logger.debug(
            f"Step {self.step_count} in cart-pole {self.name}, "
            f"action: {action.name}, reward: {reward}, done: {done}"
        )

        return result

    def render(self, mode: str = 'human') -> Optional[List[float]]:
        """Render the environment."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Get the current state
        cart_pole_state = self.state.data

        if mode == 'human':
            # Print the state to the console
            print(f"Step: {self.step_count}")
            print(f"Position: {cart_pole_state.position:.2f}")
            print(f"Velocity: {cart_pole_state.velocity:.2f}")
            print(f"Angle: {cart_pole_state.angle:.2f}")
            print(f"Angular Velocity: {cart_pole_state.angular_velocity:.2f}")
            print()
            return None
        elif mode == 'array':
            # Return the state as a list of floats
            return cart_pole_state.to_array()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed for the environment."""
        if seed is not None:
            random.seed(seed)
        return [seed] if seed is not None else []

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass


# Register the environments
registry = EnvironmentRegistry()
registry.register(MultiArmedBandit())
registry.register(CartPole())