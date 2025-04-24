"""
Custom environment module for the Agentor framework.

This module provides a framework for creating custom environments for agents.
Environments define the world in which agents operate and the rules that govern it.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic
import logging
import time
from enum import Enum

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
S = TypeVar('S')  # State type
A = TypeVar('A')  # Action type
R = TypeVar('R')  # Reward type
O = TypeVar('O')  # Observation type


class EnvironmentState(Generic[S]):
    """Represents the state of an environment."""

    def __init__(self, data: S, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        return f"EnvironmentState(data={self.data}, metadata={self.metadata})"


class ActionResult(Generic[R]):
    """Result of an action in an environment."""

    def __init__(
        self,
        success: bool,
        reward: R,
        observation: Any = None,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.reward = reward
        self.observation = observation
        self.done = done
        self.info = info or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        return f"ActionResult(success={self.success}, reward={self.reward}, done={self.done})"


class Environment(Generic[S, A, R, O], ABC):
    """Base class for all environments."""

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Environment: {name}"
        self.state: Optional[EnvironmentState[S]] = None
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.start_time = time.time()
        self.episode_start_time = time.time()

    @abstractmethod
    def reset(self) -> EnvironmentState[S]:
        """Reset the environment to its initial state."""
        pass

    @abstractmethod
    def step(self, action: A) -> ActionResult[R]:
        """Take a step in the environment by performing an action."""
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[O]:
        """Render the environment."""
        pass

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed for the environment."""
        return [seed] if seed is not None else []

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class CustomEnvironment(Environment[S, A, R, O]):
    """A customizable environment that can be configured with user-defined functions."""

    def __init__(
        self,
        name: str,
        reset_func: callable,
        step_func: callable,
        render_func: Optional[callable] = None,
        close_func: Optional[callable] = None,
        seed_func: Optional[callable] = None,
        description: Optional[str] = None
    ):
        super().__init__(name=name, description=description)
        self.reset_func = reset_func
        self.step_func = step_func
        self.render_func = render_func or (lambda mode='human': None)
        self.close_func = close_func or (lambda: None)
        self.seed_func = seed_func or (lambda seed=None: [seed] if seed is not None else [])

    def reset(self) -> EnvironmentState[S]:
        """Reset the environment using the provided reset function."""
        try:
            # Reset episode counters
            self.episode_count += 1
            self.episode_reward = 0.0
            self.step_count = 0
            self.episode_start_time = time.time()

            # Call the reset function
            result = self.reset_func()

            # Handle different return types
            if isinstance(result, EnvironmentState):
                self.state = result
            elif isinstance(result, tuple) and len(result) == 2:
                # Assume (state_data, info) format
                state_data, info = result
                self.state = EnvironmentState(data=state_data, metadata=info)
            else:
                # Assume just state data
                self.state = EnvironmentState(data=result)

            logger.debug(f"Reset environment {self.name}, new state: {self.state}")
            return self.state
        except Exception as e:
            logger.error(f"Error resetting environment {self.name}: {e}")
            raise

    def step(self, action: A) -> ActionResult[R]:
        """Take a step using the provided step function."""
        try:
            # Call the step function
            result = self.step_func(action)

            # Handle different return types
            if isinstance(result, ActionResult):
                action_result = result
            elif isinstance(result, tuple):
                if len(result) == 4:
                    # Assume (observation, reward, done, info) format (OpenAI Gym style)
                    observation, reward, done, info = result
                    action_result = ActionResult(
                        success=True,
                        reward=reward,
                        observation=observation,
                        done=done,
                        info=info
                    )
                elif len(result) == 5:
                    # Assume (success, reward, observation, done, info) format
                    success, reward, observation, done, info = result
                    action_result = ActionResult(
                        success=success,
                        reward=reward,
                        observation=observation,
                        done=done,
                        info=info
                    )
                else:
                    raise ValueError(f"Unexpected result format from step function: {result}")
            else:
                raise ValueError(f"Unexpected result type from step function: {type(result)}")

            # Update counters
            self.step_count += 1
            self.total_reward += action_result.reward
            self.episode_reward += action_result.reward

            # Update state if observation is provided
            if action_result.observation is not None:
                if self.state is None:
                    self.state = EnvironmentState(data=action_result.observation)
                else:
                    self.state.data = action_result.observation
                    self.state.timestamp = time.time()
                    self.state.metadata.update(action_result.info)

            logger.debug(f"Step {self.step_count} in environment {self.name}, action: {action}, result: {action_result}")
            return action_result
        except Exception as e:
            logger.error(f"Error stepping environment {self.name} with action {action}: {e}")
            raise

    def render(self, mode: str = 'human') -> Optional[O]:
        """Render the environment using the provided render function."""
        try:
            return self.render_func(mode)
        except Exception as e:
            logger.error(f"Error rendering environment {self.name}: {e}")
            raise

    def close(self) -> None:
        """Clean up resources using the provided close function."""
        try:
            self.close_func()
            logger.debug(f"Closed environment {self.name}")
        except Exception as e:
            logger.error(f"Error closing environment {self.name}: {e}")
            raise

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed using the provided seed function."""
        try:
            return self.seed_func(seed)
        except Exception as e:
            logger.error(f"Error seeding environment {self.name}: {e}")
            raise


class EnvironmentWrapper(Environment[S, A, R, O]):
    """A wrapper for environments that allows modifying their behavior."""

    def __init__(self, env: Environment[S, A, R, O], name: Optional[str] = None, description: Optional[str] = None):
        name = name or f"Wrapped{env.name}"
        description = description or f"Wrapper for {env.name}"
        super().__init__(name=name, description=description)
        self.env = env

    def reset(self) -> EnvironmentState[S]:
        """Reset the wrapped environment."""
        return self.env.reset()

    def step(self, action: A) -> ActionResult[R]:
        """Take a step in the wrapped environment."""
        return self.env.step(action)

    def render(self, mode: str = 'human') -> Optional[O]:
        """Render the wrapped environment."""
        return self.env.render(mode)

    def close(self) -> None:
        """Clean up resources used by the wrapped environment."""
        return self.env.close()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed for the wrapped environment."""
        return self.env.seed(seed)


class RewardWrapper(EnvironmentWrapper[S, A, R, O]):
    """A wrapper that modifies the reward returned by an environment."""

    def __init__(
        self,
        env: Environment[S, A, R, O],
        reward_func: callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        name = name or f"RewardWrapped{env.name}"
        description = description or f"Reward wrapper for {env.name}"
        super().__init__(env=env, name=name, description=description)
        self.reward_func = reward_func

    def step(self, action: A) -> ActionResult[R]:
        """Take a step and modify the reward."""
        result = self.env.step(action)

        # Modify the reward
        modified_reward = self.reward_func(result.reward, result.observation, result.done, result.info)

        # Create a new result with the modified reward
        return ActionResult(
            success=result.success,
            reward=modified_reward,
            observation=result.observation,
            done=result.done,
            info={**result.info, "original_reward": result.reward}
        )


class ObservationWrapper(EnvironmentWrapper[S, A, R, O]):
    """A wrapper that modifies the observations returned by an environment."""

    def __init__(
        self,
        env: Environment[S, A, R, O],
        observation_func: callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        name = name or f"ObservationWrapped{env.name}"
        description = description or f"Observation wrapper for {env.name}"
        super().__init__(env=env, name=name, description=description)
        self.observation_func = observation_func

    def reset(self) -> EnvironmentState[S]:
        """Reset and modify the initial observation."""
        state = self.env.reset()

        # Modify the observation
        modified_data = self.observation_func(state.data)

        # Create a new state with the modified data
        return EnvironmentState(
            data=modified_data,
            metadata={**state.metadata, "original_observation": state.data}
        )

    def step(self, action: A) -> ActionResult[R]:
        """Take a step and modify the observation."""
        result = self.env.step(action)

        # Modify the observation
        modified_observation = self.observation_func(result.observation)

        # Create a new result with the modified observation
        return ActionResult(
            success=result.success,
            reward=result.reward,
            observation=modified_observation,
            done=result.done,
            info={**result.info, "original_observation": result.observation}
        )


class ActionWrapper(EnvironmentWrapper[S, A, R, O]):
    """A wrapper that modifies the actions before they are passed to an environment."""

    def __init__(
        self,
        env: Environment[S, A, R, O],
        action_func: callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        name = name or f"ActionWrapped{env.name}"
        description = description or f"Action wrapper for {env.name}"
        super().__init__(env=env, name=name, description=description)
        self.action_func = action_func

    def step(self, action: A) -> ActionResult[R]:
        """Modify the action and then take a step."""
        # Modify the action
        modified_action = self.action_func(action)

        # Take a step with the modified action
        result = self.env.step(modified_action)

        # Add the original action to the info
        result.info["original_action"] = action

        return result


class MonitoringWrapper(EnvironmentWrapper[S, A, R, O]):
    """A wrapper that monitors and logs environment interactions."""

    def __init__(
        self,
        env: Environment[S, A, R, O],
        log_level: int = logging.INFO,
        log_frequency: int = 1,  # Log every N steps
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        name = name or f"Monitored{env.name}"
        description = description or f"Monitoring wrapper for {env.name}"
        super().__init__(env=env, name=name, description=description)
        self.log_level = log_level
        self.log_frequency = log_frequency
        self.episode_history: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []

    def reset(self) -> EnvironmentState[S]:
        """Reset the environment and log the event."""
        # Reset the environment
        state = self.env.reset()

        # Log the reset
        logger.log(self.log_level, f"Reset environment {self.name} (episode {self.episode_count + 1})")

        # Record episode start
        episode_info = {
            "episode": self.episode_count + 1,
            "start_time": time.time(),
            "initial_state": state.data
        }
        self.episode_history.append(episode_info)
        self.action_history = []

        return state

    def step(self, action: A) -> ActionResult[R]:
        """Take a step and log the action and result."""
        # Take the step
        result = self.env.step(action)

        # Record the action and result
        action_info = {
            "step": self.step_count + 1,
            "time": time.time(),
            "action": action,
            "reward": result.reward,
            "observation": result.observation,
            "done": result.done,
            "info": result.info
        }
        self.action_history.append(action_info)

        # Log the step if it's a logging step
        if (self.step_count + 1) % self.log_frequency == 0:
            logger.log(
                self.log_level,
                f"Step {self.step_count + 1} in environment {self.name}: "
                f"action={action}, reward={result.reward}, done={result.done}"
            )

        # Log episode completion
        if result.done:
            episode_duration = time.time() - self.episode_history[-1]["start_time"]
            logger.log(
                self.log_level,
                f"Episode {self.episode_count + 1} completed after {self.step_count + 1} steps. "
                f"Total reward: {self.episode_reward + result.reward:.2f}, duration: {episode_duration:.2f}s"
            )

            # Update episode info
            self.episode_history[-1].update({
                "steps": self.step_count + 1,
                "total_reward": self.episode_reward + result.reward,
                "duration": episode_duration,
                "end_time": time.time(),
                "final_state": result.observation
            })

        return result

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Get the history of all episodes."""
        return self.episode_history

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get the history of actions in the current episode."""
        return self.action_history


class TimeLimit(EnvironmentWrapper[S, A, R, O]):
    """A wrapper that limits the number of steps in an episode."""

    def __init__(
        self,
        env: Environment[S, A, R, O],
        max_steps: int,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        name = name or f"TimeLimited{env.name}"
        description = description or f"Time limit wrapper for {env.name} ({max_steps} steps)"
        super().__init__(env=env, name=name, description=description)
        self.max_steps = max_steps
        self.steps_taken = 0

    def reset(self) -> EnvironmentState[S]:
        """Reset the environment and the step counter."""
        self.steps_taken = 0
        return self.env.reset()

    def step(self, action: A) -> ActionResult[R]:
        """Take a step and check if the time limit has been reached."""
        # Take the step
        result = self.env.step(action)

        # Increment step counter
        self.steps_taken += 1

        # Check if time limit is reached
        if self.steps_taken >= self.max_steps and not result.done:
            logger.debug(f"Time limit reached in environment {self.name} after {self.steps_taken} steps")
            return ActionResult(
                success=result.success,
                reward=result.reward,
                observation=result.observation,
                done=True,  # Force done to be True
                info={**result.info, "TimeLimit.truncated": True}
            )

        return result


class EnvironmentRegistry:
    """Registry for environments."""

    def __init__(self):
        self.environments: Dict[str, Environment] = {}

    def register(self, env: Environment) -> None:
        """Register an environment."""
        if env.name in self.environments:
            logger.warning(f"Overwriting existing environment with name {env.name}")
        self.environments[env.name] = env
        logger.debug(f"Registered environment {env.name}")

    def unregister(self, name: str) -> None:
        """Unregister an environment."""
        if name in self.environments:
            del self.environments[name]
            logger.debug(f"Unregistered environment {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent environment {name}")

    def get(self, name: str) -> Optional[Environment]:
        """Get an environment by name."""
        return self.environments.get(name)

    def list_environments(self) -> List[str]:
        """List all registered environment names."""
        return list(self.environments.keys())

    def clear(self) -> None:
        """Clear all registered environments."""
        self.environments.clear()
        logger.debug("Cleared all registered environments")


# Global environment registry
default_registry = EnvironmentRegistry()