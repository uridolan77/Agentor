"""
MySQL stream processor for the Agentor framework.

This module provides a specialized stream processor for MySQL databases.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set, Deque, AsyncIterator
from collections import deque
import weakref

from .config import StreamingConfig, StreamStrategy

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Row type


class MySqlStreamProcessor(Generic[T]):
    """MySQL stream processor with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: StreamingConfig,
        execute_query_func: Callable[[str, Dict[str, Any], int], AsyncIterator[List[T]]]
    ):
        """Initialize the MySQL stream processor.
        
        Args:
            name: The name of the stream processor
            config: The streaming configuration
            execute_query_func: Function to execute a query and return a stream of chunks
        """
        self.name = name
        self.config = config
        self.execute_query_func = execute_query_func
        
        # Stream metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_queries": 0,
            "total_chunks": 0,
            "total_rows": 0,
            "total_query_time": 0.0,
            "total_processing_time": 0.0,
            "failed_queries": 0,
            "timed_out_queries": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the stream processor."""
        logger.info(f"Initialized MySQL stream processor {self.name}")
    
    async def close(self) -> None:
        """Close the stream processor."""
        logger.info(f"Closed MySQL stream processor {self.name}")
    
    async def stream_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        buffer_size: Optional[int] = None
    ) -> AsyncIterator[Union[T, List[T]]]:
        """Stream the results of a query.
        
        Args:
            query: The query to execute
            params: The query parameters
            chunk_size: The chunk size, or None to use the default
            buffer_size: The buffer size, or None to use the default
            
        Returns:
            An async iterator of rows or chunks
        """
        # Check if the query should be streamed
        if not self.config.should_stream_query(query):
            raise ValueError(f"Query should not be streamed: {query}")
        
        # Get the chunk size and buffer size
        if chunk_size is None:
            chunk_size = self.config.get_chunk_size(query)
        
        if buffer_size is None:
            buffer_size = self.config.get_buffer_size(query)
        
        # Update metrics
        self.metrics["total_queries"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Execute the query
        start_time = time.time()
        
        try:
            # Create a buffer for chunks
            buffer: Deque[List[T]] = deque(maxlen=buffer_size)
            buffer_event = asyncio.Event()
            buffer_done = False
            
            # Create a task to fill the buffer
            async def fill_buffer():
                nonlocal buffer_done
                
                try:
                    async for chunk in self.execute_query_func(query, params or {}, chunk_size):
                        # Add the chunk to the buffer
                        buffer.append(chunk)
                        
                        # Update metrics
                        self.metrics["total_chunks"] += 1
                        self.metrics["total_rows"] += len(chunk)
                        
                        # Set the buffer event
                        buffer_event.set()
                        
                        # Wait for the buffer to have space
                        while len(buffer) >= buffer_size:
                            await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error filling buffer for query: {e}")
                    
                    # Update metrics
                    self.metrics["failed_queries"] += 1
                    
                    # Re-raise the exception
                    raise
                finally:
                    # Set the buffer done flag
                    buffer_done = True
                    
                    # Set the buffer event
                    buffer_event.set()
            
            # Start the buffer filling task
            buffer_task = asyncio.create_task(fill_buffer())
            
            try:
                # Stream the results
                if self.config.strategy == StreamStrategy.CHUNK:
                    # Stream by chunks
                    while not buffer_done or buffer:
                        # Wait for the buffer to have data
                        if not buffer:
                            # Wait for the buffer event
                            try:
                                await asyncio.wait_for(buffer_event.wait(), self.config.chunk_timeout)
                            except asyncio.TimeoutError:
                                # Check if the buffer is done
                                if buffer_done:
                                    break
                                
                                # Update metrics
                                self.metrics["timed_out_queries"] += 1
                                
                                # Raise a timeout exception
                                raise asyncio.TimeoutError(f"Timeout waiting for chunk after {self.config.chunk_timeout}s")
                            
                            # Clear the buffer event
                            buffer_event.clear()
                            
                            # Check if the buffer is still empty
                            if not buffer:
                                continue
                        
                        # Get a chunk from the buffer
                        chunk = buffer.popleft()
                        
                        # Yield the chunk
                        yield chunk
                elif self.config.strategy == StreamStrategy.ROW:
                    # Stream by rows
                    while not buffer_done or buffer:
                        # Wait for the buffer to have data
                        if not buffer:
                            # Wait for the buffer event
                            try:
                                await asyncio.wait_for(buffer_event.wait(), self.config.chunk_timeout)
                            except asyncio.TimeoutError:
                                # Check if the buffer is done
                                if buffer_done:
                                    break
                                
                                # Update metrics
                                self.metrics["timed_out_queries"] += 1
                                
                                # Raise a timeout exception
                                raise asyncio.TimeoutError(f"Timeout waiting for chunk after {self.config.chunk_timeout}s")
                            
                            # Clear the buffer event
                            buffer_event.clear()
                            
                            # Check if the buffer is still empty
                            if not buffer:
                                continue
                        
                        # Get a chunk from the buffer
                        chunk = buffer.popleft()
                        
                        # Yield each row in the chunk
                        for row in chunk:
                            yield row
                else:
                    # Custom streaming strategy
                    custom_func = self.config.additional_settings.get("stream_func")
                    if custom_func:
                        async for item in custom_func(buffer, buffer_event, buffer_done, self.config.chunk_timeout):
                            yield item
                    else:
                        # Default to streaming by chunks
                        while not buffer_done or buffer:
                            # Wait for the buffer to have data
                            if not buffer:
                                # Wait for the buffer event
                                try:
                                    await asyncio.wait_for(buffer_event.wait(), self.config.chunk_timeout)
                                except asyncio.TimeoutError:
                                    # Check if the buffer is done
                                    if buffer_done:
                                        break
                                    
                                    # Update metrics
                                    self.metrics["timed_out_queries"] += 1
                                    
                                    # Raise a timeout exception
                                    raise asyncio.TimeoutError(f"Timeout waiting for chunk after {self.config.chunk_timeout}s")
                                
                                # Clear the buffer event
                                buffer_event.clear()
                                
                                # Check if the buffer is still empty
                                if not buffer:
                                    continue
                            
                            # Get a chunk from the buffer
                            chunk = buffer.popleft()
                            
                            # Yield the chunk
                            yield chunk
            finally:
                # Cancel the buffer filling task
                buffer_task.cancel()
                try:
                    await buffer_task
                except asyncio.CancelledError:
                    pass
            
            # Update metrics
            query_time = time.time() - start_time
            self.metrics["total_query_time"] += query_time
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            
            # Update metrics
            self.metrics["failed_queries"] += 1
            
            # Re-raise the exception
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get stream processor metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
