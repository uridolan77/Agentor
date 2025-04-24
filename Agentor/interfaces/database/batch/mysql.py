"""
MySQL batch processor for the Agentor framework.

This module provides a specialized batch processor for MySQL databases.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic, Tuple, Set, Deque
from collections import deque
import weakref

from .config import BatchProcessingConfig, BatchStrategy

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Operation type
R = TypeVar('R')  # Result type


class BatchOperation(Generic[T, R]):
    """Batch operation for MySQL."""
    
    def __init__(self, operation: T, future: asyncio.Future):
        """Initialize the batch operation.
        
        Args:
            operation: The operation
            future: The future to resolve with the result
        """
        self.operation = operation
        self.future = future
        self.created_at = time.time()


class MySqlBatchProcessor(Generic[T, R]):
    """MySQL batch processor with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: BatchProcessingConfig,
        execute_batch_func: Callable[[List[T]], List[R]]
    ):
        """Initialize the MySQL batch processor.
        
        Args:
            name: The name of the batch processor
            config: The batch processing configuration
            execute_batch_func: Function to execute a batch of operations
        """
        self.name = name
        self.config = config
        self.execute_batch_func = execute_batch_func
        
        # Batch queues
        self.batch_queues: Dict[str, Deque[BatchOperation[T, R]]] = {}
        
        # Batch locks
        self.batch_locks: Dict[str, asyncio.Lock] = {}
        
        # Batch tasks
        self.batch_tasks: Dict[str, asyncio.Task] = {}
        
        # Batch metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_operations": 0,
            "total_batches": 0,
            "total_batch_size": 0,
            "total_batch_time": 0.0,
            "total_execution_time": 0.0,
            "total_wait_time": 0.0,
            "failed_batches": 0,
            "retried_batches": 0
        }
        
        # Set up the event loop
        self.loop = asyncio.get_event_loop()
    
    async def initialize(self) -> None:
        """Initialize the batch processor."""
        logger.info(f"Initialized MySQL batch processor {self.name}")
    
    async def close(self) -> None:
        """Close the batch processor."""
        # Cancel all batch tasks
        for operation_type, task in list(self.batch_tasks.items()):
            # Cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Clear the batch tasks
        self.batch_tasks.clear()
        
        # Clear the batch queues
        for operation_type, queue in list(self.batch_queues.items()):
            # Reject all pending operations
            while queue:
                operation = queue.popleft()
                operation.future.set_exception(Exception("Batch processor closed"))
        
        # Clear the batch queues
        self.batch_queues.clear()
        
        logger.info(f"Closed MySQL batch processor {self.name}")
    
    async def add_operation(self, operation_type: str, operation: T) -> R:
        """Add an operation to the batch processor.
        
        Args:
            operation_type: The type of operation
            operation: The operation
            
        Returns:
            The result of the operation
        """
        # Check if the operation should be batched
        if not self.config.should_batch_operation(operation_type):
            # Execute the operation immediately
            return await self._execute_single_operation(operation)
        
        # Create a future for the operation
        future = self.loop.create_future()
        
        # Create a batch operation
        batch_op = BatchOperation(operation, future)
        
        # Add the operation to the batch queue
        await self._add_to_batch_queue(operation_type, batch_op)
        
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Wait for the operation to complete
        return await future
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get batch processor metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _add_to_batch_queue(self, operation_type: str, operation: BatchOperation[T, R]) -> None:
        """Add an operation to the batch queue.
        
        Args:
            operation_type: The type of operation
            operation: The batch operation
        """
        # Create a lock for the operation type if it doesn't exist
        if operation_type not in self.batch_locks:
            self.batch_locks[operation_type] = asyncio.Lock()
        
        # Acquire the lock
        async with self.batch_locks[operation_type]:
            # Create a queue for the operation type if it doesn't exist
            if operation_type not in self.batch_queues:
                self.batch_queues[operation_type] = deque()
            
            # Add the operation to the queue
            self.batch_queues[operation_type].append(operation)
            
            # Create a batch task for the operation type if it doesn't exist
            if operation_type not in self.batch_tasks or self.batch_tasks[operation_type].done():
                self.batch_tasks[operation_type] = self.loop.create_task(
                    self._process_batch_queue(operation_type)
                )
    
    async def _process_batch_queue(self, operation_type: str) -> None:
        """Process the batch queue for an operation type.
        
        Args:
            operation_type: The type of operation
        """
        try:
            # Get the batch size and time
            batch_size = self.config.get_batch_size(operation_type)
            batch_time = self.config.get_batch_time(operation_type)
            
            # Wait for the batch to fill or the batch time to elapse
            start_time = time.time()
            
            while True:
                # Check if the queue exists
                if operation_type not in self.batch_queues:
                    break
                
                # Get the queue
                queue = self.batch_queues[operation_type]
                
                # Check if the queue is empty
                if not queue:
                    break
                
                # Check if the batch is full
                if len(queue) >= batch_size:
                    # Process the batch
                    await self._execute_batch(operation_type, batch_size)
                    
                    # Reset the start time
                    start_time = time.time()
                    continue
                
                # Check if the batch time has elapsed
                elapsed_time = time.time() - start_time
                if elapsed_time >= batch_time:
                    # Process the batch
                    await self._execute_batch(operation_type, len(queue))
                    
                    # Reset the start time
                    start_time = time.time()
                    continue
                
                # Wait for more operations or the batch time to elapse
                remaining_time = batch_time - elapsed_time
                try:
                    await asyncio.sleep(remaining_time)
                except asyncio.CancelledError:
                    # Task was cancelled
                    break
        except Exception as e:
            logger.error(f"Error processing batch queue for {operation_type}: {e}")
        finally:
            # Remove the batch task
            if operation_type in self.batch_tasks:
                del self.batch_tasks[operation_type]
    
    async def _execute_batch(self, operation_type: str, batch_size: int) -> None:
        """Execute a batch of operations.
        
        Args:
            operation_type: The type of operation
            batch_size: The size of the batch
        """
        # Check if the queue exists
        if operation_type not in self.batch_queues:
            return
        
        # Get the queue
        queue = self.batch_queues[operation_type]
        
        # Check if the queue is empty
        if not queue:
            return
        
        # Get the operations from the queue
        operations = []
        batch_ops = []
        
        for _ in range(min(batch_size, len(queue))):
            batch_op = queue.popleft()
            operations.append(batch_op.operation)
            batch_ops.append(batch_op)
        
        # Update metrics
        self.metrics["total_batches"] += 1
        self.metrics["total_batch_size"] += len(operations)
        self.metrics["last_activity"] = time.time()
        
        # Execute the batch
        start_time = time.time()
        
        try:
            # Execute the batch
            results = await self.execute_batch_func(operations)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            
            # Check if the number of results matches the number of operations
            if len(results) != len(operations):
                # Reject all operations
                for batch_op in batch_ops:
                    batch_op.future.set_exception(Exception("Batch execution returned incorrect number of results"))
                
                # Update metrics
                self.metrics["failed_batches"] += 1
                
                return
            
            # Resolve the futures
            for i, batch_op in enumerate(batch_ops):
                batch_op.future.set_result(results[i])
        except Exception as e:
            logger.error(f"Error executing batch for {operation_type}: {e}")
            
            # Retry the batch if configured
            if self.config.max_retries > 0:
                # Update metrics
                self.metrics["retried_batches"] += 1
                
                # Retry the batch
                await self._retry_batch(operation_type, batch_ops, 1)
            else:
                # Reject all operations
                for batch_op in batch_ops:
                    batch_op.future.set_exception(e)
                
                # Update metrics
                self.metrics["failed_batches"] += 1
    
    async def _retry_batch(self, operation_type: str, batch_ops: List[BatchOperation[T, R]], retry_count: int) -> None:
        """Retry a batch of operations.
        
        Args:
            operation_type: The type of operation
            batch_ops: The batch operations
            retry_count: The retry count
        """
        # Check if we've exceeded the maximum retries
        if retry_count > self.config.max_retries:
            # Reject all operations
            for batch_op in batch_ops:
                batch_op.future.set_exception(Exception(f"Batch execution failed after {retry_count} retries"))
            
            # Update metrics
            self.metrics["failed_batches"] += 1
            
            return
        
        # Wait before retrying
        await asyncio.sleep(self.config.retry_delay)
        
        # Get the operations
        operations = [batch_op.operation for batch_op in batch_ops]
        
        # Execute the batch
        start_time = time.time()
        
        try:
            # Execute the batch
            results = await self.execute_batch_func(operations)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            
            # Check if the number of results matches the number of operations
            if len(results) != len(operations):
                # Retry the batch
                await self._retry_batch(operation_type, batch_ops, retry_count + 1)
                
                return
            
            # Resolve the futures
            for i, batch_op in enumerate(batch_ops):
                batch_op.future.set_result(results[i])
        except Exception as e:
            logger.error(f"Error retrying batch for {operation_type}: {e}")
            
            # Retry the batch
            await self._retry_batch(operation_type, batch_ops, retry_count + 1)
    
    async def _execute_single_operation(self, operation: T) -> R:
        """Execute a single operation.
        
        Args:
            operation: The operation
            
        Returns:
            The result of the operation
        """
        # Update metrics
        self.metrics["total_operations"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Execute the operation
        start_time = time.time()
        
        try:
            # Execute the operation
            results = await self.execute_batch_func([operation])
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            
            # Check if the number of results matches the number of operations
            if len(results) != 1:
                raise Exception("Batch execution returned incorrect number of results")
            
            # Return the result
            return results[0]
        except Exception as e:
            logger.error(f"Error executing single operation: {e}")
            
            # Update metrics
            self.metrics["failed_batches"] += 1
            
            # Re-raise the exception
            raise
