"""
Enhanced metrics for LLM operations.

This module provides specialized metrics for monitoring LLM operations, including:
- Model performance metrics
- Quality metrics
- Cost tracking
- Token usage analytics
- Prompt engineering metrics
"""

import time
import logging
import functools
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from datetime import datetime

from prometheus_client import Counter, Histogram, Gauge, Summary

from agentor.llm_gateway.utils.metrics import (
    LLM_REQUESTS, LLM_LATENCY, TOKEN_USAGE, LLM_COST, RESPONSE_QUALITY,
    registry
)

logger = logging.getLogger(__name__)


# Additional LLM metrics
LLM_PROMPT_SIZE = Histogram(
    'llm_prompt_size_tokens',
    'Size of prompts sent to LLMs in tokens',
    ['provider', 'model', 'purpose'],
    buckets=(10, 50, 100, 250, 500, 1000, 2000, 4000, 8000),
    registry=registry
)

LLM_COMPLETION_SIZE = Histogram(
    'llm_completion_size_tokens',
    'Size of completions from LLMs in tokens',
    ['provider', 'model', 'purpose'],
    buckets=(10, 50, 100, 250, 500, 1000, 2000),
    registry=registry
)

LLM_COST_PER_1K_TOKENS = Gauge(
    'llm_cost_per_1k_tokens_usd',
    'Cost per 1000 tokens in USD',
    ['provider', 'model', 'token_type'],
    registry=registry
)

LLM_ESTIMATED_COST = Counter(
    'llm_estimated_cost_usd',
    'Estimated cost of LLM requests in USD',
    ['provider', 'model', 'purpose'],
    registry=registry
)

LLM_CACHE_HITS = Counter(
    'llm_cache_hits_total',
    'Number of cache hits for LLM requests',
    ['provider', 'model', 'cache_type'],
    registry=registry
)

LLM_CACHE_MISSES = Counter(
    'llm_cache_misses_total',
    'Number of cache misses for LLM requests',
    ['provider', 'model', 'cache_type'],
    registry=registry
)

LLM_CACHE_HIT_RATIO = Gauge(
    'llm_cache_hit_ratio',
    'Ratio of cache hits to total requests',
    ['provider', 'model', 'cache_type'],
    registry=registry
)

LLM_THROTTLED_REQUESTS = Counter(
    'llm_throttled_requests_total',
    'Number of throttled LLM requests',
    ['provider', 'model', 'reason'],
    registry=registry
)

LLM_QUALITY_SCORES = Histogram(
    'llm_quality_scores',
    'Quality scores for LLM responses',
    ['provider', 'model', 'metric_type'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=registry
)

LLM_HALLUCINATION_RATE = Gauge(
    'llm_hallucination_rate',
    'Rate of detected hallucinations in LLM responses',
    ['provider', 'model'],
    registry=registry
)

LLM_CONTEXT_WINDOW_USAGE = Histogram(
    'llm_context_window_usage_percent',
    'Percentage of context window used',
    ['provider', 'model'],
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99),
    registry=registry
)

LLM_FALLBACK_USAGE = Counter(
    'llm_fallback_usage_total',
    'Number of times fallback mechanisms were used',
    ['provider', 'model', 'fallback_type'],
    registry=registry
)

LLM_RETRY_ATTEMPTS = Counter(
    'llm_retry_attempts_total',
    'Number of retry attempts for LLM requests',
    ['provider', 'model', 'error_type'],
    registry=registry
)

LLM_DAILY_QUOTA = Gauge(
    'llm_daily_quota_tokens',
    'Daily quota for tokens',
    ['provider', 'model'],
    registry=registry
)

LLM_QUOTA_USAGE = Gauge(
    'llm_quota_usage_percent',
    'Percentage of quota used',
    ['provider', 'model'],
    registry=registry
)

LLM_RATE_LIMIT_REMAINING = Gauge(
    'llm_rate_limit_remaining',
    'Remaining rate limit',
    ['provider', 'model', 'limit_type'],
    registry=registry
)

LLM_RATE_LIMIT_RESET = Gauge(
    'llm_rate_limit_reset_seconds',
    'Seconds until rate limit reset',
    ['provider', 'model', 'limit_type'],
    registry=registry
)


class QualityMetricType(str, Enum):
    """Types of quality metrics for LLM responses."""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FACTUALITY = "factuality"
    TOXICITY = "toxicity"
    BIAS = "bias"
    OVERALL = "overall"


class FallbackType(str, Enum):
    """Types of fallback mechanisms."""
    CACHED_RESPONSE = "cached_response"
    SIMPLER_MODEL = "simpler_model"
    RULE_BASED = "rule_based"
    DEGRADED_MODE = "degraded_mode"
    ERROR_MESSAGE = "error_message"


def track_llm_request(provider: str, model: str, purpose: str = "general"):
    """Decorator to track an LLM request with enhanced metrics.
    
    Args:
        provider: The LLM provider (e.g., "openai", "anthropic")
        model: The model name (e.g., "gpt-4", "claude-3")
        purpose: The purpose of the request (e.g., "chat", "summarization")
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            request = kwargs.get('request') or (args[1] if len(args) > 1 else None)
            
            try:
                # Track the request
                LLM_REQUESTS.labels(provider=provider, model=model, status="started").inc()
                
                # Track prompt size if available
                if request and hasattr(request, 'prompt'):
                    if hasattr(request, 'prompt_tokens'):
                        prompt_tokens = request.prompt_tokens
                    else:
                        # Estimate tokens if not provided
                        prompt_tokens = len(request.prompt.split()) * 1.3  # Rough estimate
                    
                    LLM_PROMPT_SIZE.labels(
                        provider=provider,
                        model=model,
                        purpose=purpose
                    ).observe(prompt_tokens)
                
                # Execute the request
                result = await func(*args, **kwargs)
                
                # Track success
                LLM_REQUESTS.labels(provider=provider, model=model, status="success").inc()
                
                # Track token usage
                if hasattr(result, 'usage'):
                    for token_type, count in result.usage.items():
                        TOKEN_USAGE.labels(
                            provider=provider,
                            model=model,
                            type=token_type
                        ).inc(count)
                    
                    # Track completion size
                    if 'completion_tokens' in result.usage:
                        LLM_COMPLETION_SIZE.labels(
                            provider=provider,
                            model=model,
                            purpose=purpose
                        ).observe(result.usage['completion_tokens'])
                    
                    # Track context window usage if total tokens and max tokens are available
                    if 'total_tokens' in result.usage and hasattr(request, 'max_tokens'):
                        max_possible = get_model_context_window(provider, model)
                        if max_possible:
                            usage_percent = (result.usage['total_tokens'] / max_possible) * 100
                            LLM_CONTEXT_WINDOW_USAGE.labels(
                                provider=provider,
                                model=model
                            ).observe(usage_percent)
                
                # Track cost if pricing info is available
                cost = calculate_request_cost(provider, model, result.usage if hasattr(result, 'usage') else None)
                if cost:
                    LLM_ESTIMATED_COST.labels(
                        provider=provider,
                        model=model,
                        purpose=purpose
                    ).inc(cost)
                
                return result
            
            except Exception as e:
                # Track failure
                LLM_REQUESTS.labels(provider=provider, model=model, status="error").inc()
                
                # Track error type
                error_type = type(e).__name__
                LLM_RETRY_ATTEMPTS.labels(
                    provider=provider,
                    model=model,
                    error_type=error_type
                ).inc()
                
                raise
            
            finally:
                # Track latency
                duration = time.time() - start_time
                LLM_LATENCY.labels(provider=provider, model=model).observe(duration)
        
        return wrapper
    
    return decorator


def track_llm_cache(provider: str, model: str, cache_type: str = "semantic"):
    """Decorator to track LLM cache operations.
    
    Args:
        provider: The LLM provider
        model: The model name
        cache_type: The type of cache (e.g., "semantic", "exact")
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute the cache lookup
            result = await func(*args, **kwargs)
            
            # Track cache hit/miss
            if result:
                LLM_CACHE_HITS.labels(
                    provider=provider,
                    model=model,
                    cache_type=cache_type
                ).inc()
            else:
                LLM_CACHE_MISSES.labels(
                    provider=provider,
                    model=model,
                    cache_type=cache_type
                ).inc()
            
            # Update hit ratio
            hits = LLM_CACHE_HITS.labels(provider=provider, model=model, cache_type=cache_type)._value.get()
            misses = LLM_CACHE_MISSES.labels(provider=provider, model=model, cache_type=cache_type)._value.get()
            total = hits + misses
            
            if total > 0:
                ratio = hits / total
                LLM_CACHE_HIT_RATIO.labels(
                    provider=provider,
                    model=model,
                    cache_type=cache_type
                ).set(ratio)
            
            return result
        
        return wrapper
    
    return decorator


def track_llm_quality(provider: str, model: str):
    """Decorator to track LLM response quality.
    
    Args:
        provider: The LLM provider
        model: The model name
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute the quality evaluation
            quality_scores = await func(*args, **kwargs)
            
            # Track quality scores
            if isinstance(quality_scores, dict):
                for metric_type, score in quality_scores.items():
                    LLM_QUALITY_SCORES.labels(
                        provider=provider,
                        model=model,
                        metric_type=metric_type
                    ).observe(score)
                
                # Track overall quality if available
                if 'overall' in quality_scores:
                    RESPONSE_QUALITY.labels(
                        provider=provider,
                        model=model
                    ).observe(quality_scores['overall'])
                
                # Track hallucination rate if available
                if 'factuality' in quality_scores:
                    # Invert factuality score to get hallucination rate
                    hallucination_rate = 1.0 - quality_scores['factuality']
                    LLM_HALLUCINATION_RATE.labels(
                        provider=provider,
                        model=model
                    ).set(hallucination_rate)
            
            return quality_scores
        
        return wrapper
    
    return decorator


def track_llm_fallback(provider: str, model: str, fallback_type: FallbackType):
    """Track the use of an LLM fallback mechanism.
    
    Args:
        provider: The LLM provider
        model: The model name
        fallback_type: The type of fallback used
    """
    LLM_FALLBACK_USAGE.labels(
        provider=provider,
        model=model,
        fallback_type=fallback_type.value
    ).inc()


def track_rate_limit_info(provider: str, model: str, limit_type: str, remaining: int, reset_seconds: int):
    """Track rate limit information.
    
    Args:
        provider: The LLM provider
        model: The model name
        limit_type: The type of rate limit (e.g., "requests_per_minute")
        remaining: The number of requests remaining
        reset_seconds: The number of seconds until the rate limit resets
    """
    LLM_RATE_LIMIT_REMAINING.labels(
        provider=provider,
        model=model,
        limit_type=limit_type
    ).set(remaining)
    
    LLM_RATE_LIMIT_RESET.labels(
        provider=provider,
        model=model,
        limit_type=limit_type
    ).set(reset_seconds)


def track_quota_usage(provider: str, model: str, quota: int, used: int):
    """Track quota usage.
    
    Args:
        provider: The LLM provider
        model: The model name
        quota: The total quota
        used: The amount used
    """
    LLM_DAILY_QUOTA.labels(
        provider=provider,
        model=model
    ).set(quota)
    
    if quota > 0:
        usage_percent = (used / quota) * 100
        LLM_QUOTA_USAGE.labels(
            provider=provider,
            model=model
        ).set(usage_percent)


def update_cost_per_token(provider: str, model: str, input_cost: float, output_cost: float):
    """Update the cost per token metrics.
    
    Args:
        provider: The LLM provider
        model: The model name
        input_cost: The cost per 1000 input tokens in USD
        output_cost: The cost per 1000 output tokens in USD
    """
    LLM_COST_PER_1K_TOKENS.labels(
        provider=provider,
        model=model,
        token_type="input"
    ).set(input_cost)
    
    LLM_COST_PER_1K_TOKENS.labels(
        provider=provider,
        model=model,
        token_type="output"
    ).set(output_cost)


def calculate_request_cost(provider: str, model: str, usage: Optional[Dict[str, int]]) -> Optional[float]:
    """Calculate the cost of an LLM request.
    
    Args:
        provider: The LLM provider
        model: The model name
        usage: Token usage information
        
    Returns:
        The cost in USD, or None if pricing information is not available
    """
    if not usage:
        return None
    
    # Get pricing information
    input_cost = LLM_COST_PER_1K_TOKENS.labels(
        provider=provider,
        model=model,
        token_type="input"
    )._value.get()
    
    output_cost = LLM_COST_PER_1K_TOKENS.labels(
        provider=provider,
        model=model,
        token_type="output"
    )._value.get()
    
    if not input_cost or not output_cost:
        return None
    
    # Calculate cost
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    
    prompt_cost = (prompt_tokens / 1000) * input_cost
    completion_cost = (completion_tokens / 1000) * output_cost
    
    return prompt_cost + completion_cost


def get_model_context_window(provider: str, model: str) -> Optional[int]:
    """Get the context window size for a model.
    
    Args:
        provider: The LLM provider
        model: The model name
        
    Returns:
        The context window size in tokens, or None if not known
    """
    # This is a simplified implementation
    # In a real implementation, this would be more comprehensive
    context_windows = {
        "openai": {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
        },
        "anthropic": {
            "claude-instant-1": 100000,
            "claude-2": 100000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
        },
    }
    
    return context_windows.get(provider, {}).get(model)


# Initialize cost per token for common models
def initialize_cost_metrics():
    """Initialize cost metrics for common models."""
    # OpenAI models
    update_cost_per_token("openai", "gpt-3.5-turbo", 0.0015, 0.002)
    update_cost_per_token("openai", "gpt-3.5-turbo-16k", 0.003, 0.004)
    update_cost_per_token("openai", "gpt-4", 0.03, 0.06)
    update_cost_per_token("openai", "gpt-4-32k", 0.06, 0.12)
    update_cost_per_token("openai", "gpt-4-turbo", 0.01, 0.03)
    
    # Anthropic models
    update_cost_per_token("anthropic", "claude-instant-1", 0.0008, 0.0024)
    update_cost_per_token("anthropic", "claude-2", 0.008, 0.024)
    update_cost_per_token("anthropic", "claude-3-opus", 0.015, 0.075)
    update_cost_per_token("anthropic", "claude-3-sonnet", 0.003, 0.015)
    update_cost_per_token("anthropic", "claude-3-haiku", 0.00025, 0.00125)


# Initialize cost metrics
initialize_cost_metrics()
