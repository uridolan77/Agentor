import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Awaitable, TypeVar
from enum import Enum
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Return type for rate-limited functions


class RateLimitStrategy(Enum):
    """Strategy for handling rate limit exceeded."""
    BLOCK = "block"  # Block until the rate limit resets
    FAIL = "fail"  # Fail immediately with an exception
    QUEUE = "queue"  # Queue the request and process when possible


@dataclass
class RateLimitRule:
    """A rule for rate limiting."""
    limit: int  # Maximum number of requests
    window: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.BLOCK  # Strategy for handling exceeded limits
    max_wait: Optional[int] = None  # Maximum time to wait in seconds (for BLOCK strategy)
    max_queue: Optional[int] = None  # Maximum queue size (for QUEUE strategy)


class RateLimitExceeded(Exception):
    """Exception raised when a rate limit is exceeded."""
    
    def __init__(self, rule: RateLimitRule, key: str, reset_after: float):
        """Initialize the exception.
        
        Args:
            rule: The rate limit rule that was exceeded
            key: The rate limit key
            reset_after: Seconds until the rate limit resets
        """
        self.rule = rule
        self.key = key
        self.reset_after = reset_after
        super().__init__(f"Rate limit exceeded for {key}: {rule.limit} requests per {rule.window}s. Resets in {reset_after:.1f}s")


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        self.rules: Dict[str, RateLimitRule] = {}
        self.counters: Dict[str, List[float]] = {}
        self.queues: Dict[str, asyncio.Queue] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
    
    def add_rule(self, key: str, rule: RateLimitRule):
        """Add a rate limit rule.
        
        Args:
            key: The rate limit key
            rule: The rate limit rule
        """
        self.rules[key] = rule
        self.counters[key] = []
        self.locks[key] = asyncio.Lock()
        
        if rule.strategy == RateLimitStrategy.QUEUE:
            self.queues[key] = asyncio.Queue(maxsize=rule.max_queue or 0)
    
    async def acquire(self, key: str) -> bool:
        """Acquire a rate limit permit.
        
        Args:
            key: The rate limit key
            
        Returns:
            True if the permit was acquired, False otherwise
            
        Raises:
            RateLimitExceeded: If the rate limit is exceeded and the strategy is FAIL
        """
        if key not in self.rules:
            # No rule for this key, allow the request
            return True
        
        rule = self.rules[key]
        
        async with self.locks[key]:
            # Clean up old timestamps
            now = time.time()
            window_start = now - rule.window
            self.counters[key] = [ts for ts in self.counters[key] if ts > window_start]
            
            # Check if we're over the limit
            if len(self.counters[key]) >= rule.limit:
                oldest = min(self.counters[key])
                reset_after = oldest + rule.window - now
                
                if rule.strategy == RateLimitStrategy.FAIL:
                    raise RateLimitExceeded(rule, key, reset_after)
                
                elif rule.strategy == RateLimitStrategy.BLOCK:
                    # If max_wait is specified and we'd have to wait too long, fail
                    if rule.max_wait is not None and reset_after > rule.max_wait:
                        raise RateLimitExceeded(rule, key, reset_after)
                    
                    # Wait until we can make a request
                    logger.debug(f"Rate limit reached for {key}, waiting {reset_after:.1f}s")
                    await asyncio.sleep(reset_after)
                    
                    # Recursively try again
                    return await self.acquire(key)
                
                elif rule.strategy == RateLimitStrategy.QUEUE:
                    # If the queue is full, fail
                    if rule.max_queue is not None and self.queues[key].qsize() >= rule.max_queue:
                        raise RateLimitExceeded(rule, key, reset_after)
                    
                    # Add to the queue and wait for our turn
                    future = asyncio.get_event_loop().create_future()
                    await self.queues[key].put(future)
                    
                    # If we're the first in the queue, schedule processing
                    if self.queues[key].qsize() == 1:
                        asyncio.create_task(self._process_queue(key))
                    
                    # Wait for our turn
                    logger.debug(f"Rate limit reached for {key}, queued (position {self.queues[key].qsize()})")
                    await future
                    
                    # Now we can proceed
                    return True
            
            # We're under the limit, add the current timestamp
            self.counters[key].append(now)
            return True
    
    async def _process_queue(self, key: str):
        """Process the queue for a rate limit key.
        
        Args:
            key: The rate limit key
        """
        rule = self.rules[key]
        queue = self.queues[key]
        
        while not queue.empty():
            # Get the next future from the queue
            future = await queue.get()
            
            # Wait until we can make a request
            async with self.locks[key]:
                now = time.time()
                window_start = now - rule.window
                self.counters[key] = [ts for ts in self.counters[key] if ts > window_start]
                
                if len(self.counters[key]) >= rule.limit:
                    oldest = min(self.counters[key])
                    reset_after = oldest + rule.window - now
                    
                    # Wait until we can make a request
                    logger.debug(f"Processing queue for {key}, waiting {reset_after:.1f}s")
                    await asyncio.sleep(reset_after)
                
                # Add the current timestamp
                self.counters[key].append(time.time())
            
            # Complete the future
            future.set_result(None)
            
            # Mark the task as done
            queue.task_done()
    
    def rate_limit(self, key: str):
        """Decorator for rate limiting a function.
        
        Args:
            key: The rate limit key
            
        Returns:
            A decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                await self.acquire(key)
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def rate_limit_sync(self, key: str):
        """Decorator for rate limiting a synchronous function.
        
        Args:
            key: The rate limit key
            
        Returns:
            A decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                asyncio.get_event_loop().run_until_complete(self.acquire(key))
                return func(*args, **kwargs)
            return wrapper
        return decorator


class TokenBucketRateLimiter:
    """Rate limiter using the token bucket algorithm."""
    
    def __init__(self, rate: float, capacity: int):
        """Initialize the token bucket rate limiter.
        
        Args:
            rate: Token refill rate per second
            capacity: Maximum number of tokens in the bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> Tuple[bool, float]:
        """Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            A tuple of (success, wait_time), where success is True if tokens were acquired,
            and wait_time is the time to wait in seconds before trying again
        """
        async with self.lock:
            # Refill the bucket
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # Calculate wait time
                wait_time = (tokens - self.tokens) / self.rate
                return False, wait_time
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens were acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            success, wait_time = await self.acquire(tokens)
            
            if success:
                return True
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    return False
            
            # Wait and try again
            await asyncio.sleep(wait_time)
    
    def rate_limit(self, tokens: int = 1, timeout: Optional[float] = None):
        """Decorator for rate limiting a function.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            A decorator function
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                success = await self.wait_for_tokens(tokens, timeout)
                
                if not success:
                    raise RateLimitExceeded(
                        RateLimitRule(limit=self.capacity, window=1, strategy=RateLimitStrategy.FAIL),
                        "token_bucket",
                        tokens / self.rate
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        
        return decorator
