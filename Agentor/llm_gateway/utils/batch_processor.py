import asyncio
import logging
from typing import List, Dict, Any, TypeVar, Generic, Callable, Awaitable, Optional, Union, Tuple
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


@dataclass
class BatchItem(Generic[T, R]):
    """An item in a batch."""
    input: T
    result: Optional[R] = None
    error: Optional[Exception] = None
    start_time: float = 0
    end_time: float = 0
    
    @property
    def latency(self) -> float:
        """Get the latency of processing this item."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def is_success(self) -> bool:
        """Check if the item was processed successfully."""
        return self.error is None and self.result is not None


class BatchProcessor(Generic[T, R]):
    """Process items in batches."""
    
    def __init__(
        self,
        process_func: Callable[[T], Awaitable[R]],
        batch_size: int = 10,
        max_concurrency: int = 5,
        timeout: float = 30.0
    ):
        """Initialize the batch processor.
        
        Args:
            process_func: The function to process each item
            batch_size: The maximum number of items in a batch
            max_concurrency: The maximum number of concurrent tasks
            timeout: The timeout for processing a batch in seconds
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_item(self, item: BatchItem[T, R]) -> BatchItem[T, R]:
        """Process a single item.
        
        Args:
            item: The item to process
            
        Returns:
            The processed item
        """
        item.start_time = time.time()
        try:
            async with self.semaphore:
                item.result = await self.process_func(item.input)
        except Exception as e:
            item.error = e
            logger.error(f"Error processing item: {str(e)}")
        finally:
            item.end_time = time.time()
        return item
    
    async def process_batch(self, inputs: List[T]) -> List[BatchItem[T, R]]:
        """Process a batch of items.
        
        Args:
            inputs: The inputs to process
            
        Returns:
            The processed items
        """
        items = [BatchItem(input=input_item) for input_item in inputs]
        tasks = [self.process_item(item) for item in items]
        
        try:
            # Process all items with a timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions from gather
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    items[i].error = result
                    items[i].end_time = time.time()
            
            return items
        except asyncio.TimeoutError:
            logger.error(f"Batch processing timed out after {self.timeout} seconds")
            # Mark unfinished items as timed out
            for item in items:
                if not item.end_time:
                    item.error = asyncio.TimeoutError(f"Processing timed out after {self.timeout} seconds")
                    item.end_time = time.time()
            return items
    
    async def process_all(self, inputs: List[T]) -> List[BatchItem[T, R]]:
        """Process all inputs in batches.
        
        Args:
            inputs: All inputs to process
            
        Returns:
            All processed items
        """
        # Split inputs into batches
        batches = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        all_results: List[BatchItem[T, R]] = []
        
        # Process each batch
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} items")
            batch_start = time.time()
            batch_results = await self.process_batch(batch)
            batch_end = time.time()
            
            # Log batch statistics
            success_count = sum(1 for item in batch_results if item.is_success)
            logger.info(f"Batch {i+1} completed: {success_count}/{len(batch)} successful in {batch_end - batch_start:.2f}s")
            
            all_results.extend(batch_results)
        
        return all_results


class DynamicBatchProcessor(BatchProcessor[T, R]):
    """A batch processor with dynamic batch sizing based on performance."""
    
    def __init__(
        self,
        process_func: Callable[[T], Awaitable[R]],
        initial_batch_size: int = 10,
        min_batch_size: int = 1,
        max_batch_size: int = 100,
        target_latency: float = 1.0,
        adjustment_factor: float = 0.2,
        max_concurrency: int = 5,
        timeout: float = 30.0
    ):
        """Initialize the dynamic batch processor.
        
        Args:
            process_func: The function to process each item
            initial_batch_size: The initial batch size
            min_batch_size: The minimum batch size
            max_batch_size: The maximum batch size
            target_latency: The target latency in seconds
            adjustment_factor: The factor to adjust batch size by
            max_concurrency: The maximum number of concurrent tasks
            timeout: The timeout for processing a batch in seconds
        """
        super().__init__(process_func, initial_batch_size, max_concurrency, timeout)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.adjustment_factor = adjustment_factor
        self.current_batch_size = initial_batch_size
    
    def _adjust_batch_size(self, batch_latency: float):
        """Adjust the batch size based on the latency of the last batch.
        
        Args:
            batch_latency: The latency of the last batch in seconds
        """
        if batch_latency > self.target_latency * 1.2:  # Too slow
            # Decrease batch size
            new_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adjustment_factor))
            )
            if new_size != self.current_batch_size:
                logger.info(f"Decreasing batch size from {self.current_batch_size} to {new_size} due to high latency")
                self.current_batch_size = new_size
        elif batch_latency < self.target_latency * 0.8:  # Too fast
            # Increase batch size
            new_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adjustment_factor))
            )
            if new_size != self.current_batch_size:
                logger.info(f"Increasing batch size from {self.current_batch_size} to {new_size} due to low latency")
                self.current_batch_size = new_size
    
    async def process_all(self, inputs: List[T]) -> List[BatchItem[T, R]]:
        """Process all inputs with dynamic batch sizing.
        
        Args:
            inputs: All inputs to process
            
        Returns:
            All processed items
        """
        all_results: List[BatchItem[T, R]] = []
        remaining_inputs = list(inputs)
        
        batch_num = 1
        total_batches = (len(inputs) + self.current_batch_size - 1) // self.current_batch_size
        
        while remaining_inputs:
            # Get the next batch
            batch = remaining_inputs[:self.current_batch_size]
            remaining_inputs = remaining_inputs[self.current_batch_size:]
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} items (batch size: {self.current_batch_size})")
            batch_start = time.time()
            
            # Process the batch
            batch_results = await self.process_batch(batch)
            
            batch_end = time.time()
            batch_latency = batch_end - batch_start
            
            # Log batch statistics
            success_count = sum(1 for item in batch_results if item.is_success)
            logger.info(f"Batch {batch_num} completed: {success_count}/{len(batch)} successful in {batch_latency:.2f}s")
            
            # Adjust batch size for next batch
            self._adjust_batch_size(batch_latency)
            
            # Update total batches estimate
            total_batches = batch_num + (len(remaining_inputs) + self.current_batch_size - 1) // self.current_batch_size
            
            all_results.extend(batch_results)
            batch_num += 1
        
        return all_results
