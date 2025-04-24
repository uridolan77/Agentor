"""
Visualization tools for environments in the Agentor framework.

This module provides tools for visualizing environment states, transitions,
and agent behaviors to help with debugging and understanding agent performance.
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import numpy as np
from collections import deque

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TupleSpace
)

logger = logging.getLogger(__name__)


class EnvironmentRenderer:
    """Base class for environment renderers."""
    
    def __init__(self, env: IEnvironment):
        """Initialize the environment renderer.
        
        Args:
            env: The environment to render
        """
        self.env = env
    
    def render(self) -> Any:
        """Render the environment.
        
        Returns:
            The rendered environment
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the renderer and clean up resources."""
        pass


class MatplotlibRenderer(EnvironmentRenderer):
    """Environment renderer using matplotlib."""
    
    def __init__(
        self,
        env: IEnvironment,
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 100,
        title: Optional[str] = None,
        update_interval: float = 0.1,
        block: bool = False
    ):
        """Initialize the matplotlib renderer.
        
        Args:
            env: The environment to render
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch
            title: Figure title, or None to use the environment class name
            update_interval: Interval between updates in seconds
            block: Whether to block when showing the figure
        """
        super().__init__(env)
        self.figsize = figsize
        self.dpi = dpi
        self.title = title or env.__class__.__name__
        self.update_interval = update_interval
        self.block = block
        
        # Initialize matplotlib
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig = None
            self.ax = None
            self.image = None
            self.last_update_time = 0
        except ImportError:
            logger.warning("matplotlib not available. MatplotlibRenderer will not work.")
            self.plt = None
    
    def render(self) -> Any:
        """Render the environment using matplotlib.
        
        Returns:
            The matplotlib figure
        """
        if self.plt is None:
            return None
        
        # Check if we need to throttle updates
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return self.fig
        
        self.last_update_time = current_time
        
        # Get the rendered environment
        frame = self.env.render()
        
        # Skip if rendering is not supported or doesn't return an image
        if frame is None:
            return None
        
        # Convert string to text image if needed
        if isinstance(frame, str):
            frame = self._text_to_image(frame)
        
        # Initialize the figure if needed
        if self.fig is None or self.ax is None:
            self.fig, self.ax = self.plt.subplots(figsize=self.figsize, dpi=self.dpi)
            self.plt.ion()  # Turn on interactive mode
        
        # Clear the axis
        self.ax.clear()
        
        # Display the frame
        if len(frame.shape) == 3 and frame.shape[2] in [1, 3, 4]:
            # RGB or RGBA image
            self.image = self.ax.imshow(frame)
        else:
            # Grayscale or other format
            self.image = self.ax.imshow(frame, cmap='gray')
        
        # Set title and turn off axis
        self.ax.set_title(self.title)
        self.ax.axis('off')
        
        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Show the figure if not already shown
        self.plt.show(block=self.block)
        
        return self.fig
    
    def _text_to_image(self, text: str) -> np.ndarray:
        """Convert text to an image.
        
        Args:
            text: The text to convert
            
        Returns:
            An image representation of the text
        """
        # Split the text into lines
        lines = text.strip().split('\n')
        
        # Create a binary image
        height = len(lines)
        width = max(len(line) for line in lines)
        
        # Create a grayscale image
        image = np.ones((height, width), dtype=np.uint8) * 255
        
        # Draw the text
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char != ' ':
                    image[i, j] = 0
        
        return image
    
    def close(self) -> None:
        """Close the renderer and clean up resources."""
        if self.plt is not None and self.fig is not None:
            self.plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.image = None


class VideoRenderer(EnvironmentRenderer):
    """Environment renderer that saves videos."""
    
    def __init__(
        self,
        env: IEnvironment,
        video_dir: str = "videos",
        fps: int = 30,
        filename: Optional[str] = None,
        max_frames: int = 1000
    ):
        """Initialize the video renderer.
        
        Args:
            env: The environment to render
            video_dir: Directory to save videos
            fps: Frames per second
            filename: Video filename, or None to generate automatically
            max_frames: Maximum number of frames to record
        """
        super().__init__(env)
        self.video_dir = video_dir
        self.fps = fps
        self.filename = filename
        self.max_frames = max_frames
        
        # Create the video directory if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)
        
        # Initialize recording state
        self.frames = []
        self.recording = False
        self.writer = None
        
        # Check if we have the necessary libraries
        try:
            import imageio
            self.imageio = imageio
        except ImportError:
            logger.warning("imageio not available. VideoRenderer will not work.")
            self.imageio = None
    
    def start_recording(self) -> None:
        """Start recording a video."""
        if self.imageio is None:
            logger.warning("imageio not available. Cannot record video.")
            return
        
        self.recording = True
        self.frames = []
        
        logger.info("Started recording video")
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save the video.
        
        Returns:
            The path to the saved video, or None if no video was saved
        """
        if not self.recording or self.imageio is None or not self.frames:
            return None
        
        # Generate a filename if not provided
        if self.filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"env_{timestamp}.mp4"
        else:
            filename = self.filename
        
        # Ensure the filename has the correct extension
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        
        # Create the full path
        filepath = os.path.join(self.video_dir, filename)
        
        try:
            # Save the video
            self.imageio.mimsave(
                filepath,
                self.frames,
                fps=self.fps
            )
            
            logger.info(f"Saved video with {len(self.frames)} frames: {filepath}")
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            filepath = None
        
        # Reset recording state
        self.recording = False
        self.frames = []
        
        return filepath
    
    def render(self) -> Any:
        """Render the environment and record a frame if recording.
        
        Returns:
            The rendered frame
        """
        # Render the environment
        frame = self.env.render()
        
        # Skip if rendering is not supported or doesn't return an image
        if frame is None or not isinstance(frame, np.ndarray):
            return frame
        
        # Record the frame if recording
        if self.recording and self.imageio is not None:
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
            
            # Check if we've reached the maximum number of frames
            if len(self.frames) >= self.max_frames:
                self.stop_recording()
        
        return frame
    
    def close(self) -> None:
        """Close the renderer and clean up resources."""
        if self.recording:
            self.stop_recording()


class DashboardRenderer:
    """Renderer that creates a dashboard with multiple visualizations."""
    
    def __init__(
        self,
        title: str = "Environment Dashboard",
        figsize: Tuple[int, int] = (12, 8),
        update_interval: float = 0.5
    ):
        """Initialize the dashboard renderer.
        
        Args:
            title: Dashboard title
            figsize: Figure size (width, height) in inches
            update_interval: Interval between updates in seconds
        """
        self.title = title
        self.figsize = figsize
        self.update_interval = update_interval
        
        # Initialize matplotlib
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            self.plt = plt
            self.GridSpec = GridSpec
            self.fig = None
            self.axes = {}
            self.last_update_time = 0
        except ImportError:
            logger.warning("matplotlib not available. DashboardRenderer will not work.")
            self.plt = None
            self.GridSpec = None
    
    def create_dashboard(self, layout: Tuple[int, int]) -> None:
        """Create the dashboard with the specified layout.
        
        Args:
            layout: Layout of the dashboard (rows, columns)
        """
        if self.plt is None:
            return
        
        # Create the figure
        self.fig = self.plt.figure(figsize=self.figsize)
        self.fig.suptitle(self.title)
        
        # Create the grid
        gs = self.GridSpec(*layout, figure=self.fig)
        
        # Create axes
        self.axes = {}
        for i in range(layout[0]):
            for j in range(layout[1]):
                idx = i * layout[1] + j
                self.axes[idx] = self.fig.add_subplot(gs[i, j])
        
        # Turn on interactive mode
        self.plt.ion()
    
    def add_visualization(
        self,
        idx: int,
        title: str,
        visualization_fn: Callable[[], Any],
        visualization_type: str = "image"
    ) -> None:
        """Add a visualization to the dashboard.
        
        Args:
            idx: Index of the visualization in the dashboard
            title: Title of the visualization
            visualization_fn: Function that returns the visualization data
            visualization_type: Type of visualization ('image', 'plot', 'text')
        """
        if self.plt is None or idx not in self.axes:
            return
        
        # Set the title
        self.axes[idx].set_title(title)
        
        # Store the visualization function and type
        self.axes[idx].visualization_fn = visualization_fn
        self.axes[idx].visualization_type = visualization_type
    
    def update(self) -> None:
        """Update all visualizations in the dashboard."""
        if self.plt is None or self.fig is None:
            return
        
        # Check if we need to throttle updates
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Update each visualization
        for idx, ax in self.axes.items():
            if hasattr(ax, 'visualization_fn'):
                # Clear the axis
                ax.clear()
                ax.set_title(ax.get_title())
                
                # Get the visualization data
                data = ax.visualization_fn()
                
                # Display the data based on the visualization type
                if ax.visualization_type == "image":
                    if data is not None and isinstance(data, np.ndarray):
                        if len(data.shape) == 3 and data.shape[2] in [1, 3, 4]:
                            # RGB or RGBA image
                            ax.imshow(data)
                        else:
                            # Grayscale or other format
                            ax.imshow(data, cmap='gray')
                        ax.axis('off')
                elif ax.visualization_type == "plot":
                    if data is not None and isinstance(data, tuple) and len(data) == 2:
                        x, y = data
                        ax.plot(x, y)
                        ax.grid(True)
                elif ax.visualization_type == "text":
                    if data is not None:
                        ax.text(0.5, 0.5, str(data), ha='center', va='center', fontsize=10)
                        ax.axis('off')
        
        # Update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Show the figure if not already shown
        self.plt.show(block=False)
    
    def close(self) -> None:
        """Close the dashboard and clean up resources."""
        if self.plt is not None and self.fig is not None:
            self.plt.close(self.fig)
            self.fig = None
            self.axes = {}


class EnvironmentMonitor:
    """Monitor that tracks and visualizes environment metrics."""
    
    def __init__(
        self,
        env: IEnvironment,
        log_interval: int = 10,
        window_size: int = 100
    ):
        """Initialize the environment monitor.
        
        Args:
            env: The environment to monitor
            log_interval: Number of episodes between logging
            window_size: Window size for moving averages
        """
        self.env = env
        self.log_interval = log_interval
        self.window_size = window_size
        
        # Initialize metrics
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_times = deque(maxlen=window_size)
        
        # Initialize visualization
        self.dashboard = None
    
    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self.episode_start_time = time.time()
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def record_step(self, reward: float) -> None:
        """Record a step in the current episode.
        
        Args:
            reward: The reward received
        """
        self.step_count += 1
        self.episode_length += 1
        self.episode_reward += reward
    
    def end_episode(self) -> Dict[str, Any]:
        """End the current episode and return metrics.
        
        Returns:
            Dictionary of episode metrics
        """
        episode_time = time.time() - self.episode_start_time
        
        # Record metrics
        self.episode_count += 1
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_length)
        self.episode_times.append(episode_time)
        
        # Log metrics
        if self.episode_count % self.log_interval == 0:
            self._log_metrics()
        
        # Return metrics
        return {
            "episode": self.episode_count,
            "reward": self.episode_reward,
            "length": self.episode_length,
            "time": episode_time
        }
    
    def _log_metrics(self) -> None:
        """Log current metrics."""
        if not self.episode_rewards:
            return
        
        # Calculate statistics
        mean_reward = np.mean(self.episode_rewards)
        mean_length = np.mean(self.episode_lengths)
        mean_time = np.mean(self.episode_times)
        
        logger.info(
            f"Episode {self.episode_count} - "
            f"Mean reward: {mean_reward:.2f}, "
            f"Mean length: {mean_length:.2f}, "
            f"Mean time: {mean_time:.2f}s"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.episode_rewards:
            return {
                "episodes": 0,
                "steps": 0
            }
        
        # Calculate statistics
        mean_reward = np.mean(self.episode_rewards)
        std_reward = np.std(self.episode_rewards)
        min_reward = np.min(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)
        
        mean_length = np.mean(self.episode_lengths)
        std_length = np.std(self.episode_lengths)
        
        mean_time = np.mean(self.episode_times)
        
        return {
            "episodes": self.episode_count,
            "steps": self.step_count,
            "reward": {
                "mean": mean_reward,
                "std": std_reward,
                "min": min_reward,
                "max": max_reward,
                "last": self.episode_rewards[-1]
            },
            "length": {
                "mean": mean_length,
                "std": std_length,
                "last": self.episode_lengths[-1]
            },
            "time": {
                "mean": mean_time,
                "last": self.episode_times[-1]
            }
        }
    
    def create_dashboard(self) -> None:
        """Create a dashboard for visualizing metrics."""
        self.dashboard = DashboardRenderer(title="Environment Monitor")
        self.dashboard.create_dashboard((2, 2))
        
        # Add visualizations
        self.dashboard.add_visualization(
            0,
            "Environment",
            lambda: self.env.render(),
            "image"
        )
        
        self.dashboard.add_visualization(
            1,
            "Episode Rewards",
            lambda: (range(len(self.episode_rewards)), list(self.episode_rewards)),
            "plot"
        )
        
        self.dashboard.add_visualization(
            2,
            "Episode Lengths",
            lambda: (range(len(self.episode_lengths)), list(self.episode_lengths)),
            "plot"
        )
        
        self.dashboard.add_visualization(
            3,
            "Metrics",
            lambda: self._format_metrics_text(),
            "text"
        )
    
    def _format_metrics_text(self) -> str:
        """Format metrics as text for display.
        
        Returns:
            Formatted metrics text
        """
        metrics = self.get_metrics()
        
        if metrics["episodes"] == 0:
            return "No episodes completed yet"
        
        text = (
            f"Episodes: {metrics['episodes']}\n"
            f"Steps: {metrics['steps']}\n\n"
            f"Reward:\n"
            f"  Mean: {metrics['reward']['mean']:.2f}\n"
            f"  Std: {metrics['reward']['std']:.2f}\n"
            f"  Min: {metrics['reward']['min']:.2f}\n"
            f"  Max: {metrics['reward']['max']:.2f}\n"
            f"  Last: {metrics['reward']['last']:.2f}\n\n"
            f"Length:\n"
            f"  Mean: {metrics['length']['mean']:.2f}\n"
            f"  Std: {metrics['length']['std']:.2f}\n"
            f"  Last: {metrics['length']['last']}\n\n"
            f"Time:\n"
            f"  Mean: {metrics['time']['mean']:.2f}s\n"
            f"  Last: {metrics['time']['last']:.2f}s"
        )
        
        return text
    
    def update_dashboard(self) -> None:
        """Update the dashboard."""
        if self.dashboard is not None:
            self.dashboard.update()
    
    def close(self) -> None:
        """Close the monitor and clean up resources."""
        if self.dashboard is not None:
            self.dashboard.close()
            self.dashboard = None
