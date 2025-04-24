"""
Environment interfaces for the Agentor framework.

This module defines the interfaces for environment components in the Agentor framework.
These interfaces are inspired by the Gymnasium API but adapted for the Agentor framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, TypeVar, Generic
import numpy as np
from enum import Enum

# Type variable for generic space types
T = TypeVar('T')


class SpaceType(Enum):
    """Types of spaces for observations and actions."""
    DISCRETE = "discrete"
    BOX = "box"
    DICT = "dict"
    TUPLE = "tuple"
    MULTI_DISCRETE = "multi_discrete"
    MULTI_BINARY = "multi_binary"
    TEXT = "text"
    CUSTOM = "custom"


class Space(Generic[T], ABC):
    """Base class for observation and action spaces."""
    
    @property
    @abstractmethod
    def shape(self) -> Tuple:
        """Get the shape of the space.
        
        Returns:
            The shape of the space
        """
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> Any:
        """Get the data type of the space.
        
        Returns:
            The data type of the space
        """
        pass
    
    @property
    @abstractmethod
    def space_type(self) -> SpaceType:
        """Get the type of the space.
        
        Returns:
            The type of the space
        """
        pass
    
    @abstractmethod
    def sample(self) -> T:
        """Sample a random value from the space.
        
        Returns:
            A random value from the space
        """
        pass
    
    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if a value is contained in the space.
        
        Args:
            x: The value to check
            
        Returns:
            True if the value is contained in the space, False otherwise
        """
        pass


class DiscreteSpace(Space[int]):
    """Space with a discrete number of possible values."""
    
    def __init__(self, n: int):
        """Initialize the discrete space.
        
        Args:
            n: The number of possible values
        """
        self.n = n
    
    @property
    def shape(self) -> Tuple:
        """Get the shape of the space.
        
        Returns:
            The shape of the space
        """
        return ()
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the space.
        
        Returns:
            The data type of the space
        """
        return np.int64
    
    @property
    def space_type(self) -> SpaceType:
        """Get the type of the space.
        
        Returns:
            The type of the space
        """
        return SpaceType.DISCRETE
    
    def sample(self) -> int:
        """Sample a random value from the space.
        
        Returns:
            A random value from the space
        """
        return np.random.randint(self.n)
    
    def contains(self, x: Any) -> bool:
        """Check if a value is contained in the space.
        
        Args:
            x: The value to check
            
        Returns:
            True if the value is contained in the space, False otherwise
        """
        if isinstance(x, int):
            return 0 <= x < self.n
        return False
    
    def __repr__(self) -> str:
        """Get a string representation of the space.
        
        Returns:
            A string representation of the space
        """
        return f"DiscreteSpace({self.n})"


class BoxSpace(Space[np.ndarray]):
    """Continuous space with a range of possible values."""
    
    def __init__(
        self, 
        low: Union[float, np.ndarray], 
        high: Union[float, np.ndarray], 
        shape: Optional[Tuple] = None, 
        dtype: Any = np.float32
    ):
        """Initialize the box space.
        
        Args:
            low: Lower bound of the space
            high: Upper bound of the space
            shape: Shape of the space
            dtype: Data type of the space
        """
        if shape is None:
            if isinstance(low, np.ndarray):
                shape = low.shape
            elif isinstance(high, np.ndarray):
                shape = high.shape
            else:
                raise ValueError("Shape must be provided if low and high are scalars")
        
        self._shape = shape
        self._dtype = dtype
        
        # Convert scalars to arrays
        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)
        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)
        
        self.low = low
        self.high = high
    
    @property
    def shape(self) -> Tuple:
        """Get the shape of the space.
        
        Returns:
            The shape of the space
        """
        return self._shape
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the space.
        
        Returns:
            The data type of the space
        """
        return self._dtype
    
    @property
    def space_type(self) -> SpaceType:
        """Get the type of the space.
        
        Returns:
            The type of the space
        """
        return SpaceType.BOX
    
    def sample(self) -> np.ndarray:
        """Sample a random value from the space.
        
        Returns:
            A random value from the space
        """
        return np.random.uniform(
            low=self.low, 
            high=self.high, 
            size=self._shape
        ).astype(self._dtype)
    
    def contains(self, x: Any) -> bool:
        """Check if a value is contained in the space.
        
        Args:
            x: The value to check
            
        Returns:
            True if the value is contained in the space, False otherwise
        """
        if not isinstance(x, np.ndarray):
            return False
        
        if x.shape != self._shape:
            return False
        
        return np.all(x >= self.low) and np.all(x <= self.high)
    
    def __repr__(self) -> str:
        """Get a string representation of the space.
        
        Returns:
            A string representation of the space
        """
        return f"BoxSpace(low={self.low}, high={self.high}, shape={self._shape}, dtype={self._dtype})"


class DictSpace(Space[Dict[str, Any]]):
    """Space with a dictionary of values."""
    
    def __init__(self, spaces: Dict[str, Space]):
        """Initialize the dict space.
        
        Args:
            spaces: Dictionary of spaces
        """
        self.spaces = spaces
    
    @property
    def shape(self) -> Tuple:
        """Get the shape of the space.
        
        Returns:
            The shape of the space
        """
        return ()
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the space.
        
        Returns:
            The data type of the space
        """
        return dict
    
    @property
    def space_type(self) -> SpaceType:
        """Get the type of the space.
        
        Returns:
            The type of the space
        """
        return SpaceType.DICT
    
    def sample(self) -> Dict[str, Any]:
        """Sample a random value from the space.
        
        Returns:
            A random value from the space
        """
        return {key: space.sample() for key, space in self.spaces.items()}
    
    def contains(self, x: Any) -> bool:
        """Check if a value is contained in the space.
        
        Args:
            x: The value to check
            
        Returns:
            True if the value is contained in the space, False otherwise
        """
        if not isinstance(x, dict):
            return False
        
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        
        return all(space.contains(x[key]) for key, space in self.spaces.items())
    
    def __repr__(self) -> str:
        """Get a string representation of the space.
        
        Returns:
            A string representation of the space
        """
        return f"DictSpace({self.spaces})"


class TupleSpace(Space[Tuple]):
    """Space with a tuple of values."""
    
    def __init__(self, spaces: List[Space]):
        """Initialize the tuple space.
        
        Args:
            spaces: List of spaces
        """
        self.spaces = spaces
    
    @property
    def shape(self) -> Tuple:
        """Get the shape of the space.
        
        Returns:
            The shape of the space
        """
        return (len(self.spaces),)
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the space.
        
        Returns:
            The data type of the space
        """
        return tuple
    
    @property
    def space_type(self) -> SpaceType:
        """Get the type of the space.
        
        Returns:
            The type of the space
        """
        return SpaceType.TUPLE
    
    def sample(self) -> Tuple:
        """Sample a random value from the space.
        
        Returns:
            A random value from the space
        """
        return tuple(space.sample() for space in self.spaces)
    
    def contains(self, x: Any) -> bool:
        """Check if a value is contained in the space.
        
        Args:
            x: The value to check
            
        Returns:
            True if the value is contained in the space, False otherwise
        """
        if not isinstance(x, tuple):
            return False
        
        if len(x) != len(self.spaces):
            return False
        
        return all(space.contains(x_i) for space, x_i in zip(self.spaces, x))
    
    def __repr__(self) -> str:
        """Get a string representation of the space.
        
        Returns:
            A string representation of the space
        """
        return f"TupleSpace({self.spaces})"


class TextSpace(Space[str]):
    """Space for text values."""
    
    def __init__(
        self, 
        min_length: int = 0, 
        max_length: Optional[int] = None, 
        charset: Optional[str] = None
    ):
        """Initialize the text space.
        
        Args:
            min_length: Minimum length of the text
            max_length: Maximum length of the text, or None for unlimited
            charset: Set of characters to use, or None for all printable ASCII
        """
        self.min_length = min_length
        self.max_length = max_length
        
        if charset is None:
            import string
            self.charset = string.printable
        else:
            self.charset = charset
    
    @property
    def shape(self) -> Tuple:
        """Get the shape of the space.
        
        Returns:
            The shape of the space
        """
        return ()
    
    @property
    def dtype(self) -> Any:
        """Get the data type of the space.
        
        Returns:
            The data type of the space
        """
        return str
    
    @property
    def space_type(self) -> SpaceType:
        """Get the type of the space.
        
        Returns:
            The type of the space
        """
        return SpaceType.TEXT
    
    def sample(self) -> str:
        """Sample a random value from the space.
        
        Returns:
            A random value from the space
        """
        import random
        
        if self.max_length is None:
            # If no max length, use a reasonable default
            length = self.min_length + random.randint(0, 20)
        else:
            length = random.randint(self.min_length, self.max_length)
        
        return ''.join(random.choice(self.charset) for _ in range(length))
    
    def contains(self, x: Any) -> bool:
        """Check if a value is contained in the space.
        
        Args:
            x: The value to check
            
        Returns:
            True if the value is contained in the space, False otherwise
        """
        if not isinstance(x, str):
            return False
        
        if len(x) < self.min_length:
            return False
        
        if self.max_length is not None and len(x) > self.max_length:
            return False
        
        return all(c in self.charset for c in x)
    
    def __repr__(self) -> str:
        """Get a string representation of the space.
        
        Returns:
            A string representation of the space
        """
        return f"TextSpace(min_length={self.min_length}, max_length={self.max_length})"


class IEnvironment(ABC):
    """Interface for environment components."""
    
    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Get the observation space of the environment.
        
        Returns:
            The observation space
        """
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Get the action space of the environment.
        
        Returns:
            The action space
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def render(self) -> Optional[Union[np.ndarray, str]]:
        """Render the environment.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass
