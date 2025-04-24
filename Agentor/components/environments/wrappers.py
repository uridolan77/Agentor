"""
Environment wrappers for the Agentor framework.

This module provides wrappers for environments that add functionality or
transform the environment in some way, such as normalizing observations,
clipping rewards, or recording videos.
"""

import logging
import time
import os
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, TypeVar, Generic
import numpy as np
from collections import deque

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Observation type
A = TypeVar('A')  # Action type


class EnvironmentWrapper(IEnvironment):
    """Base class for environment wrappers."""
    
    def __init__(self, env: IEnvironment):
        """Initialize the environment wrapper.
        
        Args:
            env: The environment to wrap
        """
        self.env = env
    
    @property
    def observation_space(self) -> Space:
        """Get the observation space of the environment.
        
        Returns:
            The observation space
        """
        return self.env.observation_space
    
    @property
    def action_space(self) -> Space:
        """Get the action space of the environment.
        
        Returns:
            The action space
        """
        return self.env.action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        return self.env.reset(seed, options)
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        return self.env.step(action)
    
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        self.env.close()


class NormalizeObservation(EnvironmentWrapper):
    """Wrapper that normalizes observations."""
    
    def __init__(
        self,
        env: IEnvironment,
        mean: Optional[Union[float, np.ndarray]] = None,
        std: Optional[Union[float, np.ndarray]] = None,
        epsilon: float = 1e-8,
        clip_range: Optional[Tuple[float, float]] = None
    ):
        """Initialize the normalize observation wrapper.
        
        Args:
            env: The environment to wrap
            mean: Mean for normalization, or None to estimate from data
            std: Standard deviation for normalization, or None to estimate from data
            epsilon: Small constant to avoid division by zero
            clip_range: Range to clip normalized observations to, or None for no clipping
        """
        super().__init__(env)
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.clip_range = clip_range
        
        # Initialize running statistics if needed
        self.running_mean = None
        self.running_var = None
        self.count = 0
        
        if mean is None or std is None:
            # Get the shape of the observation space
            if isinstance(env.observation_space, BoxSpace):
                shape = env.observation_space.shape
                self.running_mean = np.zeros(shape, dtype=np.float32)
                self.running_var = np.ones(shape, dtype=np.float32)
            else:
                logger.warning("NormalizeObservation only works with BoxSpace, using scalar statistics")
                self.running_mean = 0.0
                self.running_var = 1.0
    
    def _update_stats(self, observation: np.ndarray) -> None:
        """Update running statistics with a new observation.
        
        Args:
            observation: The observation to update with
        """
        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        delta2 = observation - self.running_mean
        self.running_var += delta * delta2
    
    def _normalize(self, observation: np.ndarray) -> np.ndarray:
        """Normalize an observation.
        
        Args:
            observation: The observation to normalize
            
        Returns:
            The normalized observation
        """
        # Update statistics if needed
        if self.mean is None or self.std is None:
            self._update_stats(observation)
        
        # Get mean and std
        mean = self.mean if self.mean is not None else self.running_mean
        std = self.std if self.std is not None else np.sqrt(self.running_var / max(1, self.count))
        
        # Normalize
        normalized = (observation - mean) / (std + self.epsilon)
        
        # Clip if needed
        if self.clip_range is not None:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
        
        return normalized
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(seed, options)
        
        # Normalize the observation
        if isinstance(observation, np.ndarray):
            observation = self._normalize(observation)
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Normalize the observation
        if isinstance(observation, np.ndarray):
            observation = self._normalize(observation)
        
        return observation, reward, terminated, truncated, info


class ClipReward(EnvironmentWrapper):
    """Wrapper that clips rewards to a specified range."""
    
    def __init__(
        self,
        env: IEnvironment,
        min_reward: float = -1.0,
        max_reward: float = 1.0
    ):
        """Initialize the clip reward wrapper.
        
        Args:
            env: The environment to wrap
            min_reward: Minimum reward value
            max_reward: Maximum reward value
        """
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Clip the reward
        clipped_reward = np.clip(reward, self.min_reward, self.max_reward)
        
        # Add the original reward to info
        info["original_reward"] = reward
        
        return observation, clipped_reward, terminated, truncated, info


class FrameStack(EnvironmentWrapper):
    """Wrapper that stacks multiple frames together."""
    
    def __init__(
        self,
        env: IEnvironment,
        num_frames: int = 4,
        channel_order: str = "first"
    ):
        """Initialize the frame stack wrapper.
        
        Args:
            env: The environment to wrap
            num_frames: Number of frames to stack
            channel_order: Whether to stack frames along the first or last dimension
        """
        super().__init__(env)
        self.num_frames = num_frames
        self.channel_order = channel_order
        self.frames = deque(maxlen=num_frames)
        
        # Update the observation space
        if isinstance(env.observation_space, BoxSpace):
            shape = list(env.observation_space.shape)
            
            if channel_order == "first":
                if len(shape) == 1:
                    # For 1D observations, add a channel dimension
                    new_shape = [num_frames] + shape
                else:
                    # For multi-dimensional observations, stack along the first dimension
                    new_shape = [num_frames * shape[0]] + shape[1:]
            else:  # "last"
                if len(shape) == 1:
                    # For 1D observations, add a channel dimension
                    new_shape = shape + [num_frames]
                else:
                    # For multi-dimensional observations, stack along the last dimension
                    new_shape = shape[:-1] + [shape[-1] * num_frames]
            
            self._observation_space = BoxSpace(
                low=np.min(env.observation_space.low),
                high=np.max(env.observation_space.high),
                shape=tuple(new_shape),
                dtype=env.observation_space.dtype
            )
        else:
            logger.warning("FrameStack only works with BoxSpace, observation space will not be updated")
            self._observation_space = env.observation_space
    
    @property
    def observation_space(self) -> Space:
        """Get the observation space of the environment.
        
        Returns:
            The observation space
        """
        return self._observation_space
    
    def _stack_frames(self) -> np.ndarray:
        """Stack the frames together.
        
        Returns:
            The stacked frames
        """
        # Convert frames to numpy arrays
        frames = [np.array(frame) for frame in self.frames]
        
        # Stack the frames
        if self.channel_order == "first":
            if len(frames[0].shape) == 1:
                # For 1D observations, stack along a new first dimension
                return np.stack(frames)
            else:
                # For multi-dimensional observations, concatenate along the first dimension
                return np.concatenate(frames, axis=0)
        else:  # "last"
            if len(frames[0].shape) == 1:
                # For 1D observations, stack along a new last dimension
                return np.stack(frames, axis=-1)
            else:
                # For multi-dimensional observations, concatenate along the last dimension
                return np.concatenate(frames, axis=-1)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(seed, options)
        
        # Clear the frames
        self.frames.clear()
        
        # Add the initial observation multiple times
        for _ in range(self.num_frames):
            self.frames.append(observation)
        
        # Stack the frames
        stacked_frames = self._stack_frames()
        
        return stacked_frames, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Add the new observation
        self.frames.append(observation)
        
        # Stack the frames
        stacked_frames = self._stack_frames()
        
        return stacked_frames, reward, terminated, truncated, info


class VideoRecorder(EnvironmentWrapper):
    """Wrapper that records videos of the environment."""
    
    def __init__(
        self,
        env: IEnvironment,
        video_dir: str = "videos",
        episode_trigger: Optional[Callable[[int], bool]] = None,
        step_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 1000,
        name_prefix: str = "video"
    ):
        """Initialize the video recorder wrapper.
        
        Args:
            env: The environment to wrap
            video_dir: Directory to save videos
            episode_trigger: Function that takes the episode number and returns whether to record
            step_trigger: Function that takes the step number and returns whether to record
            video_length: Maximum length of the video in steps
            name_prefix: Prefix for video filenames
        """
        super().__init__(env)
        self.video_dir = video_dir
        self.episode_trigger = episode_trigger or (lambda x: x % 10 == 0)  # Default: record every 10th episode
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.name_prefix = name_prefix
        
        # Create the video directory if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)
        
        # Initialize recording state
        self.episode_count = 0
        self.step_count = 0
        self.recording = False
        self.frames = []
        self.current_video_path = None
        
        # Check if we have the necessary libraries
        try:
            import imageio
            self.imageio = imageio
        except ImportError:
            logger.warning("imageio not available. VideoRecorder will not work.")
            self.imageio = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(seed, options)
        
        # Increment episode count
        self.episode_count += 1
        self.step_count = 0
        
        # Check if we should start recording
        if self.episode_trigger(self.episode_count):
            self._start_recording()
        
        # Record the first frame if recording
        if self.recording:
            self._record_frame()
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Increment step count
        self.step_count += 1
        
        # Check if we should start recording based on step trigger
        if not self.recording and self.step_trigger is not None and self.step_trigger(self.step_count):
            self._start_recording()
        
        # Record the frame if recording
        if self.recording:
            self._record_frame()
        
        # Check if we should stop recording
        if self.recording and (terminated or truncated or len(self.frames) >= self.video_length):
            self._stop_recording()
        
        return observation, reward, terminated, truncated, info
    
    def _start_recording(self) -> None:
        """Start recording a video."""
        if self.imageio is None:
            return
        
        self.recording = True
        self.frames = []
        
        # Create a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_path = os.path.join(
            self.video_dir,
            f"{self.name_prefix}_ep{self.episode_count}_step{self.step_count}_{timestamp}.mp4"
        )
        
        logger.info(f"Started recording video: {self.current_video_path}")
    
    def _record_frame(self) -> None:
        """Record a frame."""
        if self.imageio is None:
            return
        
        # Render the environment
        frame = self.env.render()
        
        # Skip if rendering is not supported or doesn't return an image
        if frame is None or not isinstance(frame, np.ndarray):
            return
        
        # Ensure the frame is in RGB format
        if len(frame.shape) == 2:
            # Convert grayscale to RGB
            frame = np.stack([frame] * 3, axis=2)
        elif frame.shape[2] == 1:
            # Convert single-channel to RGB
            frame = np.repeat(frame, 3, axis=2)
        elif frame.shape[2] > 3:
            # Use only the first 3 channels
            frame = frame[:, :, :3]
        
        # Add the frame to the list
        self.frames.append(frame)
    
    def _stop_recording(self) -> None:
        """Stop recording and save the video."""
        if self.imageio is None or not self.frames:
            return
        
        logger.info(f"Saving video with {len(self.frames)} frames: {self.current_video_path}")
        
        try:
            # Save the video
            self.imageio.mimsave(
                self.current_video_path,
                self.frames,
                fps=30
            )
            
            logger.info(f"Saved video: {self.current_video_path}")
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
        
        # Reset recording state
        self.recording = False
        self.frames = []
        self.current_video_path = None
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        # Stop recording if needed
        if self.recording:
            self._stop_recording()
        
        # Close the environment
        self.env.close()


class ActionRepeat(EnvironmentWrapper):
    """Wrapper that repeats actions multiple times."""
    
    def __init__(
        self,
        env: IEnvironment,
        repeat: int = 4
    ):
        """Initialize the action repeat wrapper.
        
        Args:
            env: The environment to wrap
            repeat: Number of times to repeat each action
        """
        super().__init__(env)
        self.repeat = repeat
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        total_reward = 0.0
        
        for i in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return observation, total_reward, terminated, truncated, info


class ActionNoise(EnvironmentWrapper):
    """Wrapper that adds noise to actions."""
    
    def __init__(
        self,
        env: IEnvironment,
        noise_scale: float = 0.1,
        noise_type: str = "gaussian"
    ):
        """Initialize the action noise wrapper.
        
        Args:
            env: The environment to wrap
            noise_scale: Scale of the noise
            noise_type: Type of noise ('gaussian' or 'uniform')
        """
        super().__init__(env)
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        
        # Check if the action space is compatible
        if not isinstance(env.action_space, BoxSpace):
            logger.warning("ActionNoise only works with BoxSpace, noise will not be added")
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Add noise to the action if the action space is compatible
        if isinstance(self.env.action_space, BoxSpace) and isinstance(action, np.ndarray):
            # Generate noise
            if self.noise_type == "gaussian":
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
            else:  # "uniform"
                noise = np.random.uniform(-self.noise_scale, self.noise_scale, size=action.shape)
            
            # Add noise to the action
            noisy_action = action + noise
            
            # Clip to the action space
            noisy_action = np.clip(
                noisy_action,
                self.env.action_space.low,
                self.env.action_space.high
            )
            
            # Take a step with the noisy action
            return self.env.step(noisy_action)
        
        # If the action space is not compatible, just take a step with the original action
        return self.env.step(action)


class RewardScaling(EnvironmentWrapper):
    """Wrapper that scales rewards by a constant factor."""
    
    def __init__(
        self,
        env: IEnvironment,
        scale: float = 0.1
    ):
        """Initialize the reward scaling wrapper.
        
        Args:
            env: The environment to wrap
            scale: Scale factor for rewards
        """
        super().__init__(env)
        self.scale = scale
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Scale the reward
        scaled_reward = reward * self.scale
        
        # Add the original reward to info
        info["original_reward"] = reward
        
        return observation, scaled_reward, terminated, truncated, info


class TransformObservation(EnvironmentWrapper):
    """Wrapper that applies a transformation to observations."""
    
    def __init__(
        self,
        env: IEnvironment,
        transform: Callable[[Any], Any]
    ):
        """Initialize the transform observation wrapper.
        
        Args:
            env: The environment to wrap
            transform: Function to transform observations
        """
        super().__init__(env)
        self.transform = transform
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(seed, options)
        
        # Transform the observation
        transformed_observation = self.transform(observation)
        
        return transformed_observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Transform the observation
        transformed_observation = self.transform(observation)
        
        return transformed_observation, reward, terminated, truncated, info


class TransformAction(EnvironmentWrapper):
    """Wrapper that applies a transformation to actions."""
    
    def __init__(
        self,
        env: IEnvironment,
        transform: Callable[[Any], Any]
    ):
        """Initialize the transform action wrapper.
        
        Args:
            env: The environment to wrap
            transform: Function to transform actions
        """
        super().__init__(env)
        self.transform = transform
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Transform the action
        transformed_action = self.transform(action)
        
        # Take a step with the transformed action
        return self.env.step(transformed_action)


class VisualizeEnvironment(EnvironmentWrapper):
    """Wrapper that visualizes the environment state."""
    
    def __init__(
        self,
        env: IEnvironment,
        visualize_fn: Optional[Callable[[IEnvironment], None]] = None,
        visualize_interval: int = 1
    ):
        """Initialize the visualize environment wrapper.
        
        Args:
            env: The environment to wrap
            visualize_fn: Function to visualize the environment, or None to use matplotlib
            visualize_interval: Interval between visualizations in steps
        """
        super().__init__(env)
        self.visualize_fn = visualize_fn
        self.visualize_interval = visualize_interval
        self.step_count = 0
        
        # Check if we have matplotlib if no custom function is provided
        if visualize_fn is None:
            try:
                import matplotlib.pyplot as plt
                self.plt = plt
            except ImportError:
                logger.warning("matplotlib not available. VisualizeEnvironment will not work.")
                self.plt = None
    
    def _default_visualize(self) -> None:
        """Default visualization using matplotlib."""
        if self.plt is None:
            return
        
        # Render the environment
        frame = self.env.render()
        
        # Skip if rendering is not supported or doesn't return an image
        if frame is None or not isinstance(frame, np.ndarray):
            return
        
        # Display the frame
        plt = self.plt
        plt.figure(figsize=(8, 8))
        
        if len(frame.shape) == 3 and frame.shape[2] in [1, 3, 4]:
            # RGB or RGBA image
            plt.imshow(frame)
        else:
            # Grayscale or other format
            plt.imshow(frame, cmap='gray')
        
        plt.axis('off')
        plt.title(f"Step {self.step_count}")
        plt.show()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        observation, info = self.env.reset(seed, options)
        
        # Reset step count
        self.step_count = 0
        
        # Visualize the initial state
        if self.visualize_fn is not None:
            self.visualize_fn(self.env)
        else:
            self._default_visualize()
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Increment step count
        self.step_count += 1
        
        # Visualize if it's time
        if self.step_count % self.visualize_interval == 0:
            if self.visualize_fn is not None:
                self.visualize_fn(self.env)
            else:
                self._default_visualize()
        
        return observation, reward, terminated, truncated, info
