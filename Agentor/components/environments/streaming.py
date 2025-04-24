"""
Streaming environment for the Agentor framework.

This module provides a streaming environment that supports real-time data streams,
which is useful for scenarios where observations arrive continuously rather than
in discrete steps, such as real-time data feeds or event streams.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union, AsyncIterable, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
import numpy as np

from agentor.core.interfaces.environment import (
    IEnvironment, Space, DiscreteSpace, BoxSpace, DictSpace, TextSpace
)
from agentor.components.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Stream data type


class DataStream(Generic[T], AsyncIterable[T]):
    """Base class for data streams."""
    
    def __init__(
        self,
        name: str,
        space: Space,
        buffer_size: int = 100,
        max_frequency: Optional[float] = None
    ):
        """Initialize the data stream.
        
        Args:
            name: The name of the stream
            space: The space of the stream data
            buffer_size: Maximum number of items to buffer
            max_frequency: Maximum frequency of data in Hz, or None for unlimited
        """
        self.name = name
        self.space = space
        self.buffer_size = buffer_size
        self.max_frequency = max_frequency
        
        self.buffer: List[T] = []
        self.subscribers: List[asyncio.Queue] = []
        self.running = False
        self.task = None
        self.last_emit_time = 0
    
    async def start(self) -> None:
        """Start the data stream."""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._stream_loop())
        logger.info(f"Started data stream: {self.name}")
    
    async def stop(self) -> None:
        """Stop the data stream."""
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        
        logger.info(f"Stopped data stream: {self.name}")
    
    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to the data stream.
        
        Returns:
            A queue that will receive data from the stream
        """
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue
    
    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from the data stream.
        
        Args:
            queue: The queue to unsubscribe
        """
        if queue in self.subscribers:
            self.subscribers.remove(queue)
    
    async def emit(self, data: T) -> None:
        """Emit data to the stream.
        
        Args:
            data: The data to emit
        """
        # Check if we need to throttle
        if self.max_frequency is not None:
            min_interval = 1.0 / self.max_frequency
            elapsed = time.time() - self.last_emit_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        
        self.last_emit_time = time.time()
        
        # Add to buffer
        self.buffer.append(data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Send to subscribers
        for queue in self.subscribers:
            await queue.put(data)
    
    @abstractmethod
    async def _stream_loop(self) -> None:
        """Main loop for the data stream."""
        pass
    
    def __aiter__(self):
        """Return an async iterator for the stream."""
        return self
    
    async def __anext__(self) -> T:
        """Get the next item from the stream."""
        queue = await self.subscribe()
        try:
            return await queue.get()
        finally:
            await self.unsubscribe(queue)


class CallbackDataStream(DataStream[T]):
    """Data stream that gets data from a callback function."""
    
    def __init__(
        self,
        name: str,
        space: Space,
        callback: Callable[[], Awaitable[T]],
        interval: float = 1.0,
        **kwargs
    ):
        """Initialize the callback data stream.
        
        Args:
            name: The name of the stream
            space: The space of the stream data
            callback: Async function that returns data
            interval: Interval between data points in seconds
            **kwargs: Additional arguments for the data stream
        """
        super().__init__(name, space, **kwargs)
        self.callback = callback
        self.interval = interval
    
    async def _stream_loop(self) -> None:
        """Main loop for the data stream."""
        while self.running:
            try:
                # Get data from the callback
                data = await self.callback()
                
                # Emit the data
                await self.emit(data)
                
                # Wait for the next interval
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data stream {self.name}: {str(e)}")
                await asyncio.sleep(1.0)  # Wait a bit before retrying


class GeneratorDataStream(DataStream[T]):
    """Data stream that gets data from an async generator."""
    
    def __init__(
        self,
        name: str,
        space: Space,
        generator: AsyncIterable[T],
        **kwargs
    ):
        """Initialize the generator data stream.
        
        Args:
            name: The name of the stream
            space: The space of the stream data
            generator: Async generator that yields data
            **kwargs: Additional arguments for the data stream
        """
        super().__init__(name, space, **kwargs)
        self.generator = generator
    
    async def _stream_loop(self) -> None:
        """Main loop for the data stream."""
        try:
            async for data in self.generator:
                if not self.running:
                    break
                
                # Emit the data
                await self.emit(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in data stream {self.name}: {str(e)}")


class StreamingEnvironment(BaseEnvironment):
    """Environment that supports streaming observations.
    
    This environment type is useful for scenarios where observations arrive
    continuously rather than in discrete steps, such as real-time data feeds
    or event streams.
    
    Examples:
        >>> env = StreamingEnvironment()
        >>> env.register_stream("market_data", market_data_stream)
        >>> async for observation in env.stream("market_data"):
        ...     print(observation)
    """
    
    def __init__(
        self,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        max_episode_steps: Optional[int] = None,
        auto_reset: bool = True
    ):
        """Initialize the streaming environment.
        
        Args:
            observation_space: The observation space, or None to use a dict space
            action_space: The action space, or None to use a dict space
            max_episode_steps: Maximum number of steps per episode, or None for unlimited
            auto_reset: Whether to automatically reset the environment when an episode ends
        """
        # Default to dict spaces if not provided
        if observation_space is None:
            observation_space = DictSpace({})
        
        if action_space is None:
            action_space = DictSpace({})
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            max_episode_steps=max_episode_steps,
            auto_reset=auto_reset
        )
        
        self.streams: Dict[str, DataStream] = {}
        self.current_observation: Dict[str, Any] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
    
    def register_stream(self, name: str, stream: DataStream) -> None:
        """Register a data stream.
        
        Args:
            name: The name of the stream
            stream: The data stream
        """
        if name in self.streams:
            logger.warning(f"Stream {name} already registered, replacing")
        
        self.streams[name] = stream
        
        # Update the observation space
        if isinstance(self._observation_space, DictSpace):
            spaces = self._observation_space.spaces.copy()
            spaces[name] = stream.space
            self._observation_space = DictSpace(spaces)
    
    def unregister_stream(self, name: str) -> None:
        """Unregister a data stream.
        
        Args:
            name: The name of the stream
        """
        if name not in self.streams:
            logger.warning(f"Stream {name} not registered")
            return
        
        # Stop the stream task if running
        if name in self.stream_tasks and self.stream_tasks[name] is not None:
            self.stream_tasks[name].cancel()
            self.stream_tasks[name] = None
        
        # Remove the stream
        del self.streams[name]
        
        # Update the observation space
        if isinstance(self._observation_space, DictSpace):
            spaces = self._observation_space.spaces.copy()
            if name in spaces:
                del spaces[name]
            self._observation_space = DictSpace(spaces)
    
    async def stream(self, name: str) -> AsyncIterable:
        """Get a stream of observations.
        
        Args:
            name: The name of the stream
            
        Returns:
            An async iterable of observations
            
        Raises:
            KeyError: If the stream is not registered
        """
        if name not in self.streams:
            raise KeyError(f"Stream {name} not registered")
        
        # Start the stream if not already running
        if not self.streams[name].running:
            await self.streams[name].start()
        
        # Subscribe to the stream
        queue = await self.streams[name].subscribe()
        
        try:
            while True:
                # Get the next observation
                observation = await queue.get()
                
                # Update the current observation
                self.current_observation[name] = observation
                
                # Yield the observation
                yield observation
        finally:
            # Unsubscribe from the stream
            await self.streams[name].unsubscribe(queue)
    
    async def start_all_streams(self) -> None:
        """Start all registered streams."""
        for name, stream in self.streams.items():
            if not stream.running:
                await stream.start()
    
    async def stop_all_streams(self) -> None:
        """Stop all registered streams."""
        for name, stream in self.streams.items():
            if stream.running:
                await stream.stop()
    
    def _reset_impl(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Implementation of the reset method.
        
        Args:
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset the current observation
        self.current_observation = {}
        
        # Start all streams asynchronously
        for name, stream in self.streams.items():
            if not stream.running:
                self.stream_tasks[name] = asyncio.create_task(stream.start())
        
        return self.current_observation, {}
    
    def _step_impl(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Implementation of the step method.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, terminated, info)
        """
        # In a streaming environment, the step method doesn't do much
        # since observations arrive asynchronously
        
        # Return the current observation
        return self.current_observation, 0.0, False, {}
    
    def _render_impl(self) -> Optional[Union[np.ndarray, str]]:
        """Implementation of the render method.
        
        Returns:
            The rendered environment, or None if rendering is not supported
        """
        # Create a string representation of the current observation
        result = "Streaming Environment:\n"
        for name, value in self.current_observation.items():
            result += f"  {name}: {value}\n"
        
        return result
    
    def _close_impl(self) -> None:
        """Implementation of the close method."""
        # Stop all streams
        for name, stream in self.streams.items():
            if stream.running:
                self.stream_tasks[name] = asyncio.create_task(stream.stop())
        
        # Clear the current observation
        self.current_observation = {}


class MarketDataStream(DataStream[Dict[str, float]]):
    """Example data stream for market data."""
    
    def __init__(
        self,
        symbols: List[str],
        update_interval: float = 1.0,
        volatility: float = 0.01,
        **kwargs
    ):
        """Initialize the market data stream.
        
        Args:
            symbols: List of symbols to track
            update_interval: Interval between updates in seconds
            volatility: Volatility of price changes
            **kwargs: Additional arguments for the data stream
        """
        # Create a space for the market data
        space = DictSpace({
            symbol: BoxSpace(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            for symbol in symbols
        })
        
        super().__init__(
            name="market_data",
            space=space,
            **kwargs
        )
        
        self.symbols = symbols
        self.update_interval = update_interval
        self.volatility = volatility
        
        # Initialize prices
        self.prices = {symbol: 100.0 for symbol in symbols}
    
    async def _stream_loop(self) -> None:
        """Main loop for the data stream."""
        while self.running:
            try:
                # Update prices
                for symbol in self.symbols:
                    # Random walk with drift
                    change = np.random.normal(0.0001, self.volatility)
                    self.prices[symbol] *= (1.0 + change)
                
                # Emit the data
                await self.emit(self.prices.copy())
                
                # Wait for the next update
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data stream: {str(e)}")
                await asyncio.sleep(1.0)  # Wait a bit before retrying


class NewsStream(DataStream[Dict[str, str]]):
    """Example data stream for news data."""
    
    def __init__(
        self,
        topics: List[str],
        update_interval: float = 5.0,
        **kwargs
    ):
        """Initialize the news data stream.
        
        Args:
            topics: List of topics to track
            update_interval: Interval between updates in seconds
            **kwargs: Additional arguments for the data stream
        """
        # Create a space for the news data
        space = DictSpace({
            topic: TextSpace()
            for topic in topics
        })
        
        super().__init__(
            name="news_data",
            space=space,
            **kwargs
        )
        
        self.topics = topics
        self.update_interval = update_interval
        
        # Sample news headlines
        self.headlines = {
            "technology": [
                "New AI breakthrough announced by research lab",
                "Tech giant releases latest smartphone model",
                "Startup raises $100M in funding round",
                "Security vulnerability discovered in popular software",
                "Quantum computing milestone achieved"
            ],
            "finance": [
                "Stock market reaches all-time high",
                "Central bank announces interest rate decision",
                "Cryptocurrency prices surge overnight",
                "Major merger announced between industry leaders",
                "Economic indicators show mixed signals"
            ],
            "health": [
                "New medical treatment shows promising results",
                "Health officials issue updated guidelines",
                "Research study reveals breakthrough in disease prevention",
                "Healthcare costs continue to rise",
                "Wellness trend gains popularity"
            ],
            "politics": [
                "Election results announced in key race",
                "New policy initiative unveiled by government",
                "International treaty signed by world leaders",
                "Political tensions rise in disputed region",
                "Legislative body passes controversial bill"
            ]
        }
    
    async def _stream_loop(self) -> None:
        """Main loop for the data stream."""
        while self.running:
            try:
                # Generate news for each topic
                news = {}
                for topic in self.topics:
                    # Get headlines for the topic or use generic headlines
                    topic_headlines = self.headlines.get(topic, [
                        f"Breaking news about {topic}",
                        f"Latest developments in {topic}",
                        f"New report on {topic} released",
                        f"Experts discuss {topic} trends",
                        f"Analysis of recent {topic} events"
                    ])
                    
                    # Select a random headline
                    news[topic] = np.random.choice(topic_headlines)
                
                # Emit the data
                await self.emit(news)
                
                # Wait for the next update
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in news stream: {str(e)}")
                await asyncio.sleep(1.0)  # Wait a bit before retrying
