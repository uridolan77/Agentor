"""
Batch operations for vector databases in the Agentor framework.

This module provides utilities for batching vector database operations,
improving performance when working with large numbers of vectors.
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable, Awaitable, Union, Tuple
import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agentor.components.memory.vector_db import VectorDBProvider

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Item type


@dataclass
class BatchItem(Generic[T]):
    """An item in a batch with metadata."""
    
    item: T
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)


class BatchProcessor(Generic[T], ABC):
    """Abstract base class for batch processors."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_time: float = 0.1,
        max_batch_size: int = 1000
    ):
        """Initialize the batch processor.
        
        Args:
            batch_size: Target batch size
            max_wait_time: Maximum time to wait for a batch to fill
            max_batch_size: Maximum batch size
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_batch_size = max_batch_size
        
        self.queue: List[BatchItem[T]] = []
        self.lock = asyncio.Lock()
        self.processing = False
        self.processor_task = None
        
        # Statistics
        self.total_items = 0
        self.total_batches = 0
        self.total_errors = 0
        self.largest_batch = 0
        self.smallest_batch = float('inf')
        self.total_wait_time = 0
        self.total_processing_time = 0
    
    @abstractmethod
    async def process_batch(self, items: List[T]) -> List[Any]:
        """Process a batch of items.
        
        Args:
            items: The items to process
            
        Returns:
            The results for each item
        """
        pass
    
    async def add(self, item: T) -> Any:
        """Add an item to the batch.
        
        Args:
            item: The item to add
            
        Returns:
            The result of processing the item
        """
        # Create a future for the result
        future = asyncio.Future()
        
        # Add the item to the queue
        async with self.lock:
            self.queue.append(BatchItem(item=item, future=future))
            
            # Start the processor if not already running
            if not self.processing:
                self.processing = True
                self.processor_task = asyncio.create_task(self._process_queue())
        
        # Wait for the result
        return await future
    
    async def _process_queue(self) -> None:
        """Process the queue in batches."""
        while True:
            # Wait for the batch to fill or the max wait time to elapse
            batch_start_time = time.time()
            
            while True:
                # Check if we have enough items
                async with self.lock:
                    if len(self.queue) >= self.batch_size:
                        break
                    
                    # Check if we've waited long enough
                    if time.time() - batch_start_time >= self.max_wait_time and self.queue:
                        break
                    
                    # Check if the queue is empty
                    if not self.queue:
                        self.processing = False
                        return
                
                # Wait a bit
                await asyncio.sleep(0.01)
            
            # Get the batch
            async with self.lock:
                # Get up to max_batch_size items
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]
            
            # Process the batch
            try:
                # Extract the items
                items = [item.item for item in batch]
                
                # Update statistics
                self.total_items += len(items)
                self.total_batches += 1
                self.largest_batch = max(self.largest_batch, len(items))
                self.smallest_batch = min(self.smallest_batch, len(items))
                self.total_wait_time += time.time() - batch_start_time
                
                # Process the batch
                processing_start_time = time.time()
                results = await self.process_batch(items)
                self.total_processing_time += time.time() - processing_start_time
                
                # Set the results
                for item, result in zip(batch, results):
                    item.future.set_result(result)
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                self.total_errors += 1
                
                # Set the error for all items
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics.
        
        Returns:
            Dictionary of batch processor statistics
        """
        async with self.lock:
            return {
                "total_items": self.total_items,
                "total_batches": self.total_batches,
                "total_errors": self.total_errors,
                "largest_batch": self.largest_batch,
                "smallest_batch": self.smallest_batch if self.smallest_batch != float('inf') else 0,
                "average_batch_size": self.total_items / self.total_batches if self.total_batches > 0 else 0,
                "average_wait_time": self.total_wait_time / self.total_batches if self.total_batches > 0 else 0,
                "average_processing_time": self.total_processing_time / self.total_batches if self.total_batches > 0 else 0,
                "queue_size": len(self.queue)
            }


class VectorDBBatchProcessor(BatchProcessor[Dict[str, Any]]):
    """Batch processor for vector database operations."""
    
    def __init__(
        self,
        vector_db: VectorDBProvider,
        operation: str,
        **kwargs
    ):
        """Initialize the vector database batch processor.
        
        Args:
            vector_db: The vector database provider
            operation: The operation to perform (add, update, delete)
            **kwargs: Additional arguments for the batch processor
        """
        super().__init__(**kwargs)
        self.vector_db = vector_db
        self.operation = operation
    
    async def process_batch(self, items: List[Dict[str, Any]]) -> List[Any]:
        """Process a batch of items.
        
        Args:
            items: The items to process
            
        Returns:
            The results for each item
        """
        if self.operation == "add":
            return await self._batch_add(items)
        elif self.operation == "update":
            return await self._batch_update(items)
        elif self.operation == "delete":
            return await self._batch_delete(items)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    async def _batch_add(self, items: List[Dict[str, Any]]) -> List[Any]:
        """Add a batch of items to the vector database.
        
        Args:
            items: The items to add
            
        Returns:
            The results for each item
        """
        # Check if the vector database supports batch operations
        if hasattr(self.vector_db, "batch_add"):
            return await self.vector_db.batch_add(items)
        
        # Fall back to individual operations
        results = []
        for item in items:
            try:
                result = await self.vector_db.add(**item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error adding item to vector database: {str(e)}")
                results.append({"error": str(e)})
        
        return results
    
    async def _batch_update(self, items: List[Dict[str, Any]]) -> List[Any]:
        """Update a batch of items in the vector database.
        
        Args:
            items: The items to update
            
        Returns:
            The results for each item
        """
        # Check if the vector database supports batch operations
        if hasattr(self.vector_db, "batch_update"):
            return await self.vector_db.batch_update(items)
        
        # Fall back to individual operations
        results = []
        for item in items:
            try:
                result = await self.vector_db.update(**item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error updating item in vector database: {str(e)}")
                results.append({"error": str(e)})
        
        return results
    
    async def _batch_delete(self, items: List[Dict[str, Any]]) -> List[Any]:
        """Delete a batch of items from the vector database.
        
        Args:
            items: The items to delete
            
        Returns:
            The results for each item
        """
        # Check if the vector database supports batch operations
        if hasattr(self.vector_db, "batch_delete"):
            return await self.vector_db.batch_delete(items)
        
        # Fall back to individual operations
        results = []
        for item in items:
            try:
                result = await self.vector_db.delete(**item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error deleting item from vector database: {str(e)}")
                results.append({"error": str(e)})
        
        return results


class BatchedVectorDB:
    """Wrapper for vector database providers that adds batching."""
    
    def __init__(
        self,
        vector_db: VectorDBProvider,
        add_batch_size: int = 100,
        update_batch_size: int = 100,
        delete_batch_size: int = 100,
        max_wait_time: float = 0.1
    ):
        """Initialize the batched vector database.
        
        Args:
            vector_db: The vector database provider
            add_batch_size: Batch size for add operations
            update_batch_size: Batch size for update operations
            delete_batch_size: Batch size for delete operations
            max_wait_time: Maximum time to wait for a batch to fill
        """
        self.vector_db = vector_db
        
        # Create batch processors
        self.add_processor = VectorDBBatchProcessor(
            vector_db=vector_db,
            operation="add",
            batch_size=add_batch_size,
            max_wait_time=max_wait_time
        )
        
        self.update_processor = VectorDBBatchProcessor(
            vector_db=vector_db,
            operation="update",
            batch_size=update_batch_size,
            max_wait_time=max_wait_time
        )
        
        self.delete_processor = VectorDBBatchProcessor(
            vector_db=vector_db,
            operation="delete",
            batch_size=delete_batch_size,
            max_wait_time=max_wait_time
        )
    
    async def add(self, **kwargs) -> Any:
        """Add a vector to the database.
        
        Args:
            **kwargs: Arguments for the add operation
            
        Returns:
            The result of the add operation
        """
        return await self.add_processor.add(kwargs)
    
    async def update(self, **kwargs) -> Any:
        """Update a vector in the database.
        
        Args:
            **kwargs: Arguments for the update operation
            
        Returns:
            The result of the update operation
        """
        return await self.update_processor.add(kwargs)
    
    async def delete(self, **kwargs) -> Any:
        """Delete a vector from the database.
        
        Args:
            **kwargs: Arguments for the delete operation
            
        Returns:
            The result of the delete operation
        """
        return await self.delete_processor.add(kwargs)
    
    async def search(self, **kwargs) -> Any:
        """Search for vectors in the database.
        
        Args:
            **kwargs: Arguments for the search operation
            
        Returns:
            The result of the search operation
        """
        # Search operations are not batched
        return await self.vector_db.search(**kwargs)
    
    async def get(self, **kwargs) -> Any:
        """Get a vector from the database.
        
        Args:
            **kwargs: Arguments for the get operation
            
        Returns:
            The result of the get operation
        """
        # Get operations are not batched
        return await self.vector_db.get(**kwargs)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the batched vector database.
        
        Returns:
            Dictionary of statistics
        """
        add_stats = await self.add_processor.get_stats()
        update_stats = await self.update_processor.get_stats()
        delete_stats = await self.delete_processor.get_stats()
        
        return {
            "add": add_stats,
            "update": update_stats,
            "delete": delete_stats
        }
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying vector database.
        
        Args:
            name: The attribute name
            
        Returns:
            The attribute value
        """
        return getattr(self.vector_db, name)
