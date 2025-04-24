"""
Enhanced metrics for agent operations.

This module provides specialized metrics for monitoring agent operations, including:
- Agent performance metrics
- Decision-making metrics
- Tool usage metrics
- Memory access patterns
- Agent coordination metrics
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
    AGENT_EXECUTIONS, AGENT_LATENCY, TOOL_EXECUTIONS, TOOL_LATENCY,
    MEMORY_OPERATIONS, registry
)

logger = logging.getLogger(__name__)


# Additional agent metrics
AGENT_DECISION_COUNTS = Counter(
    'agent_decision_counts_total',
    'Number of times each decision was made by an agent',
    ['agent_name', 'decision'],
    registry=registry
)

AGENT_DECISION_LATENCY = Histogram(
    'agent_decision_latency_seconds',
    'Time taken for an agent to make a decision',
    ['agent_name'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

AGENT_TOOL_USAGE = Counter(
    'agent_tool_usage_total',
    'Number of times each tool was used by an agent',
    ['agent_name', 'tool_name'],
    registry=registry
)

AGENT_MEMORY_USAGE = Counter(
    'agent_memory_usage_total',
    'Number of times each memory type was accessed by an agent',
    ['agent_name', 'memory_type', 'operation'],
    registry=registry
)

AGENT_COORDINATION_EVENTS = Counter(
    'agent_coordination_events_total',
    'Number of coordination events between agents',
    ['source_agent', 'target_agent', 'event_type'],
    registry=registry
)

AGENT_TASK_COMPLETION = Counter(
    'agent_task_completion_total',
    'Number of tasks completed by an agent',
    ['agent_name', 'task_type', 'status'],
    registry=registry
)

AGENT_TASK_LATENCY = Histogram(
    'agent_task_latency_seconds',
    'Time taken for an agent to complete a task',
    ['agent_name', 'task_type'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=registry
)

AGENT_REASONING_STEPS = Histogram(
    'agent_reasoning_steps',
    'Number of reasoning steps taken by an agent',
    ['agent_name', 'task_type'],
    buckets=(1, 2, 3, 5, 8, 13, 21, 34, 55),
    registry=registry
)

AGENT_REASONING_DEPTH = Histogram(
    'agent_reasoning_depth',
    'Maximum depth of reasoning tree',
    ['agent_name', 'task_type'],
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    registry=registry
)

AGENT_SATISFACTION_SCORE = Histogram(
    'agent_satisfaction_score',
    'User satisfaction score for agent responses',
    ['agent_name'],
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    registry=registry
)

AGENT_ACCURACY_SCORE = Histogram(
    'agent_accuracy_score',
    'Accuracy score for agent responses',
    ['agent_name', 'task_type'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=registry
)

AGENT_ACTIVE_TASKS = Gauge(
    'agent_active_tasks',
    'Number of active tasks per agent',
    ['agent_name'],
    registry=registry
)

AGENT_QUEUE_SIZE = Gauge(
    'agent_queue_size',
    'Number of tasks in the agent queue',
    ['agent_name'],
    registry=registry
)

AGENT_QUEUE_LATENCY = Histogram(
    'agent_queue_latency_seconds',
    'Time spent in the agent queue',
    ['agent_name'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

AGENT_MEMORY_SIZE = Gauge(
    'agent_memory_size',
    'Size of agent memory in number of items',
    ['agent_name', 'memory_type'],
    registry=registry
)

AGENT_CONTEXT_SIZE = Histogram(
    'agent_context_size_tokens',
    'Size of agent context in tokens',
    ['agent_name'],
    buckets=(100, 500, 1000, 2000, 5000, 10000, 20000, 50000),
    registry=registry
)

AGENT_ROUTER_DECISIONS = Counter(
    'agent_router_decisions_total',
    'Number of routing decisions made',
    ['router_name', 'source', 'destination', 'confidence_level'],
    registry=registry
)

AGENT_ROUTER_LATENCY = Histogram(
    'agent_router_latency_seconds',
    'Time taken for the router to make a decision',
    ['router_name'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)


class CoordinationEventType(str, Enum):
    """Types of coordination events between agents."""
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    TASK_DELEGATED = "task_delegated"
    TASK_ACCEPTED = "task_accepted"
    TASK_REJECTED = "task_rejected"
    TASK_COMPLETED = "task_completed"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent decisions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


def track_agent_execution(agent_name: str, task_type: str = "general"):
    """Decorator to track an agent execution with enhanced metrics.
    
    Args:
        agent_name: The name of the agent
        task_type: The type of task being executed
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Increment active tasks
            AGENT_ACTIVE_TASKS.labels(agent_name=agent_name).inc()
            
            try:
                # Track the execution
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="started").inc()
                
                # Execute the agent
                result = await func(*args, **kwargs)
                
                # Track success
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="success").inc()
                AGENT_TASK_COMPLETION.labels(
                    agent_name=agent_name,
                    task_type=task_type,
                    status="success"
                ).inc()
                
                return result
            
            except Exception as e:
                # Track failure
                AGENT_EXECUTIONS.labels(agent_name=agent_name, status="error").inc()
                AGENT_TASK_COMPLETION.labels(
                    agent_name=agent_name,
                    task_type=task_type,
                    status="error"
                ).inc()
                
                raise
            
            finally:
                # Track latency
                duration = time.time() - start_time
                AGENT_LATENCY.labels(agent_name=agent_name).observe(duration)
                AGENT_TASK_LATENCY.labels(
                    agent_name=agent_name,
                    task_type=task_type
                ).observe(duration)
                
                # Decrement active tasks
                AGENT_ACTIVE_TASKS.labels(agent_name=agent_name).dec()
        
        return wrapper
    
    return decorator


def track_agent_decision(agent_name: str):
    """Decorator to track an agent decision.
    
    Args:
        agent_name: The name of the agent
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Execute the decision function
            decision = func(*args, **kwargs)
            
            # Track the decision
            AGENT_DECISION_COUNTS.labels(
                agent_name=agent_name,
                decision=decision
            ).inc()
            
            # Track latency
            duration = time.time() - start_time
            AGENT_DECISION_LATENCY.labels(agent_name=agent_name).observe(duration)
            
            return decision
        
        return wrapper
    
    return decorator


def track_agent_tool_usage(agent_name: str, tool_name: str):
    """Track the use of a tool by an agent.
    
    Args:
        agent_name: The name of the agent
        tool_name: The name of the tool
    """
    AGENT_TOOL_USAGE.labels(
        agent_name=agent_name,
        tool_name=tool_name
    ).inc()


def track_agent_memory_usage(agent_name: str, memory_type: str, operation: str):
    """Track the use of memory by an agent.
    
    Args:
        agent_name: The name of the agent
        memory_type: The type of memory (e.g., "episodic", "semantic")
        operation: The operation (e.g., "read", "write")
    """
    AGENT_MEMORY_USAGE.labels(
        agent_name=agent_name,
        memory_type=memory_type,
        operation=operation
    ).inc()


def track_agent_coordination(source_agent: str, target_agent: str, event_type: CoordinationEventType):
    """Track a coordination event between agents.
    
    Args:
        source_agent: The name of the source agent
        target_agent: The name of the target agent
        event_type: The type of coordination event
    """
    AGENT_COORDINATION_EVENTS.labels(
        source_agent=source_agent,
        target_agent=target_agent,
        event_type=event_type.value
    ).inc()


def track_agent_reasoning(agent_name: str, task_type: str, steps: int, max_depth: int):
    """Track the reasoning process of an agent.
    
    Args:
        agent_name: The name of the agent
        task_type: The type of task
        steps: The number of reasoning steps
        max_depth: The maximum depth of the reasoning tree
    """
    AGENT_REASONING_STEPS.labels(
        agent_name=agent_name,
        task_type=task_type
    ).observe(steps)
    
    AGENT_REASONING_DEPTH.labels(
        agent_name=agent_name,
        task_type=task_type
    ).observe(max_depth)


def track_agent_satisfaction(agent_name: str, score: int):
    """Track user satisfaction with an agent.
    
    Args:
        agent_name: The name of the agent
        score: The satisfaction score (1-10)
    """
    AGENT_SATISFACTION_SCORE.labels(agent_name=agent_name).observe(score)


def track_agent_accuracy(agent_name: str, task_type: str, score: float):
    """Track the accuracy of an agent.
    
    Args:
        agent_name: The name of the agent
        task_type: The type of task
        score: The accuracy score (0.0-1.0)
    """
    AGENT_ACCURACY_SCORE.labels(
        agent_name=agent_name,
        task_type=task_type
    ).observe(score)


def update_agent_queue_metrics(agent_name: str, queue_size: int, queue_latency: Optional[float] = None):
    """Update agent queue metrics.
    
    Args:
        agent_name: The name of the agent
        queue_size: The current queue size
        queue_latency: The time spent in the queue (optional)
    """
    AGENT_QUEUE_SIZE.labels(agent_name=agent_name).set(queue_size)
    
    if queue_latency is not None:
        AGENT_QUEUE_LATENCY.labels(agent_name=agent_name).observe(queue_latency)


def update_agent_memory_size(agent_name: str, memory_type: str, size: int):
    """Update the size of an agent's memory.
    
    Args:
        agent_name: The name of the agent
        memory_type: The type of memory
        size: The size of the memory in number of items
    """
    AGENT_MEMORY_SIZE.labels(
        agent_name=agent_name,
        memory_type=memory_type
    ).set(size)


def track_agent_context_size(agent_name: str, token_count: int):
    """Track the size of an agent's context.
    
    Args:
        agent_name: The name of the agent
        token_count: The size of the context in tokens
    """
    AGENT_CONTEXT_SIZE.labels(agent_name=agent_name).observe(token_count)


def track_router_decision(
    router_name: str,
    source: str,
    destination: str,
    confidence_level: Union[ConfidenceLevel, str, float]
):
    """Track a routing decision.
    
    Args:
        router_name: The name of the router
        source: The source of the request
        destination: The destination agent
        confidence_level: The confidence level of the decision
    """
    # Convert float confidence to level
    if isinstance(confidence_level, float):
        if confidence_level < 0.2:
            confidence_level = ConfidenceLevel.VERY_LOW
        elif confidence_level < 0.4:
            confidence_level = ConfidenceLevel.LOW
        elif confidence_level < 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_level < 0.8:
            confidence_level = ConfidenceLevel.HIGH
        else:
            confidence_level = ConfidenceLevel.VERY_HIGH
    
    # Convert to string if it's an enum
    if isinstance(confidence_level, ConfidenceLevel):
        confidence_level = confidence_level.value
    
    AGENT_ROUTER_DECISIONS.labels(
        router_name=router_name,
        source=source,
        destination=destination,
        confidence_level=confidence_level
    ).inc()


def track_router_latency(router_name: str, latency: float):
    """Track the latency of a router.
    
    Args:
        router_name: The name of the router
        latency: The latency in seconds
    """
    AGENT_ROUTER_LATENCY.labels(router_name=router_name).observe(latency)
