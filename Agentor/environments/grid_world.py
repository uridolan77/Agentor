"""
Grid World environment for the Agentor framework.

This module provides a simple grid world environment for reinforcement learning.
The grid world consists of a 2D grid with various cell types (empty, wall, goal, etc.)
and an agent that can move in four directions (up, down, left, right).
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import random
import time

from agentor.environments.custom import (
    Environment, EnvironmentState, ActionResult, EnvironmentRegistry
)
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


class CellType(Enum):
    """Types of cells in the grid world."""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    TRAP = 3
    START = 4


class Action(Enum):
    """Actions that can be taken in the grid world."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridWorldState:
    """State of the grid world environment."""

    def __init__(
        self,
        grid: np.ndarray,
        agent_position: Tuple[int, int],
        step_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.grid = grid
        self.agent_position = agent_position
        self.step_count = step_count
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return f"GridWorldState(agent_position={self.agent_position}, step_count={self.step_count})"

    def copy(self) -> 'GridWorldState':
        """Create a copy of the state."""
        return GridWorldState(
            grid=self.grid.copy(),
            agent_position=self.agent_position,
            step_count=self.step_count,
            metadata=self.metadata.copy()
        )


class GridWorld(Environment[GridWorldState, Action, float, np.ndarray]):
    """A simple grid world environment."""

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        wall_density: float = 0.2,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        step_reward: float = -0.01,
        max_steps: Optional[int] = 100,
        random_start: bool = True,
        seed: Optional[int] = None,
        name: str = "GridWorld",
        description: Optional[str] = None
    ):
        description = description or f"Grid world environment ({width}x{height})"
        super().__init__(name=name, description=description)

        self.width = width
        self.height = height
        self.wall_density = wall_density
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.step_reward = step_reward
        self.max_steps = max_steps
        self.random_start = random_start

        # Set the random seed
        self.seed(seed)

        # Initialize the grid
        self.grid = np.zeros((height, width), dtype=int)
        self.start_position = (0, 0)
        self.goal_positions = []
        self.trap_positions = []

        # Generate the grid
        self._generate_grid()

        # Initialize the state
        self.state = None

    def _generate_grid(self) -> None:
        """Generate a random grid."""
        # Clear the grid
        self.grid.fill(CellType.EMPTY.value)

        # Add walls
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.wall_density:
                    self.grid[y, x] = CellType.WALL.value

        # Ensure the borders are walls
        self.grid[0, :] = CellType.WALL.value
        self.grid[-1, :] = CellType.WALL.value
        self.grid[:, 0] = CellType.WALL.value
        self.grid[:, -1] = CellType.WALL.value

        # Add a goal
        goal_added = False
        while not goal_added:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.GOAL.value
                self.goal_positions = [(y, x)]
                goal_added = True

        # Add some traps
        num_traps = max(1, int((self.width * self.height) * 0.05))  # 5% of cells are traps
        traps_added = 0
        self.trap_positions = []
        while traps_added < num_traps:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.TRAP.value
                self.trap_positions.append((y, x))
                traps_added += 1

        # Set the start position
        start_added = False
        while not start_added:
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid[y, x] == CellType.EMPTY.value:
                self.grid[y, x] = CellType.START.value
                self.start_position = (y, x)
                start_added = True

    def reset(self) -> EnvironmentState[GridWorldState]:
        """Reset the environment to its initial state."""
        # Reset episode counters
        self.episode_count += 1
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_start_time = time.time()

        # Generate a new grid
        self._generate_grid()

        # Set the agent position
        if self.random_start:
            # Find a random empty cell
            empty_cells = []
            for y in range(self.height):
                for x in range(self.width):
                    if self.grid[y, x] == CellType.EMPTY.value:
                        empty_cells.append((y, x))
            if empty_cells:
                self.start_position = random.choice(empty_cells)
            else:
                # If no empty cells, use the start position
                pass

        # Create the initial state
        grid_state = GridWorldState(
            grid=self.grid.copy(),
            agent_position=self.start_position,
            step_count=0
        )

        # Create the environment state
        self.state = EnvironmentState(
            data=grid_state,
            metadata={
                "episode": self.episode_count,
                "max_steps": self.max_steps
            }
        )

        logger.debug(f"Reset grid world {self.name}, agent position: {self.start_position}")
        return self.state

    def step(self, action: Action) -> ActionResult[float]:
        """Take a step in the environment by performing an action."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Get the current state
        grid_state = self.state.data

        # Get the agent position
        y, x = grid_state.agent_position

        # Calculate the new position based on the action
        if action == Action.UP:
            new_y, new_x = y - 1, x
        elif action == Action.RIGHT:
            new_y, new_x = y, x + 1
        elif action == Action.DOWN:
            new_y, new_x = y + 1, x
        elif action == Action.LEFT:
            new_y, new_x = y, x - 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if the new position is valid
        if (
            new_y < 0 or new_y >= self.height or
            new_x < 0 or new_x >= self.width or
            self.grid[new_y, new_x] == CellType.WALL.value
        ):
            # Agent hit a wall, stay in the same position
            new_y, new_x = y, x

        # Check if the agent reached a goal or trap
        done = False
        reward = self.step_reward
        info = {}

        if self.grid[new_y, new_x] == CellType.GOAL.value:
            reward = self.goal_reward
            done = True
            info["goal_reached"] = True
        elif self.grid[new_y, new_x] == CellType.TRAP.value:
            reward = self.trap_reward
            done = True
            info["trap_reached"] = True

        # Check if the maximum number of steps has been reached
        self.step_count += 1
        if self.max_steps is not None and self.step_count >= self.max_steps:
            done = True
            info["max_steps_reached"] = True

        # Update the state
        new_grid_state = grid_state.copy()
        new_grid_state.agent_position = (new_y, new_x)
        new_grid_state.step_count = self.step_count

        # Update the environment state
        self.state = EnvironmentState(
            data=new_grid_state,
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
            observation=new_grid_state,
            done=done,
            info=info
        )

        logger.debug(
            f"Step {self.step_count} in grid world {self.name}, "
            f"action: {action.name}, new position: {(new_y, new_x)}, "
            f"reward: {reward}, done: {done}"
        )

        return result

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        # Get the current state
        grid_state = self.state.data

        # Create a copy of the grid for rendering
        render_grid = self.grid.copy()

        # Add the agent to the grid
        y, x = grid_state.agent_position
        render_grid[y, x] = 5  # Agent is represented by 5

        if mode == 'human':
            # Print the grid to the console
            print(f"Step: {self.step_count}, Position: {grid_state.agent_position}")
            for y in range(self.height):
                row = ""
                for x in range(self.width):
                    cell = render_grid[y, x]
                    if cell == CellType.EMPTY.value:
                        row += ".  "
                    elif cell == CellType.WALL.value:
                        row += "#  "
                    elif cell == CellType.GOAL.value:
                        row += "G  "
                    elif cell == CellType.TRAP.value:
                        row += "T  "
                    elif cell == CellType.START.value:
                        row += "S  "
                    elif cell == 5:  # Agent
                        row += "A  "
                    else:
                        row += "?  "
                print(row)
            print()
            return None
        elif mode == 'rgb_array':
            # Return the grid as a numpy array
            return render_grid
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the random seed for the environment."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed] if seed is not None else []

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass


# Register the environment
EnvironmentRegistry().register(GridWorld())