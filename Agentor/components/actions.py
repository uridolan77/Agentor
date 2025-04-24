"""
Actions module for the Agentor framework.

This module provides interfaces and implementations for agent actions.
Actions are the ways that agents can affect their environment.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
import asyncio
import inspect
import logging

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


class ActionResult:
    """Result of an action execution."""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}

    def __str__(self) -> str:
        if self.success:
            return f"ActionResult(success={self.success}, data={self.data})"
        else:
            return f"ActionResult(success={self.success}, error={self.error})"

    @classmethod
    def success_result(cls, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> 'ActionResult':
        """Create a successful action result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_result(cls, error: Union[Exception, str], metadata: Optional[Dict[str, Any]] = None) -> 'ActionResult':
        """Create an error action result."""
        if isinstance(error, str):
            error = Exception(error)
        return cls(success=False, error=error, metadata=metadata)


class Action(Generic[T, R], ABC):
    """Base class for all actions."""

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Action: {name}"

    @abstractmethod
    async def execute(self, input_data: T) -> ActionResult:
        """Execute the action."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class FunctionAction(Action[T, R]):
    """An action that executes a function."""

    def __init__(
        self,
        func: Callable[[T], R],
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.func = func
        name = name or func.__name__
        description = description or func.__doc__ or f"Function action: {name}"
        super().__init__(name=name, description=description)

    async def execute(self, input_data: T) -> ActionResult:
        """Execute the function."""
        try:
            # Check if the function is a coroutine function
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(input_data)
            else:
                # Run synchronous functions in a thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.func, input_data)

            return ActionResult.success_result(data=result)
        except Exception as e:
            logger.error(f"Error executing function action {self.name}: {e}")
            return ActionResult.error_result(error=e)


class CompositeAction(Action[T, List[ActionResult]]):
    """An action that executes multiple actions in sequence."""

    def __init__(
        self,
        actions: List[Action],
        name: Optional[str] = None,
        description: Optional[str] = None,
        stop_on_error: bool = False
    ):
        self.actions = actions
        name = name or f"CompositeAction({len(actions)})"
        description = description or f"Composite action with {len(actions)} sub-actions"
        self.stop_on_error = stop_on_error
        super().__init__(name=name, description=description)

    async def execute(self, input_data: T) -> ActionResult:
        """Execute all actions in sequence."""
        results = []

        for action in self.actions:
            try:
                result = await action.execute(input_data)
                results.append(result)

                if not result.success and self.stop_on_error:
                    logger.warning(f"Stopping composite action {self.name} due to error in {action.name}")
                    break
            except Exception as e:
                logger.error(f"Error executing action {action.name} in composite action {self.name}: {e}")
                error_result = ActionResult.error_result(error=e)
                results.append(error_result)

                if self.stop_on_error:
                    break

        # The composite action is successful if all sub-actions were successful
        success = all(result.success for result in results)

        return ActionResult(
            success=success,
            data=results,
            metadata={"action_count": len(self.actions), "completed_count": len(results)}
        )


class ParallelAction(Action[T, List[ActionResult]]):
    """An action that executes multiple actions in parallel."""

    def __init__(
        self,
        actions: List[Action],
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_concurrency: Optional[int] = None
    ):
        self.actions = actions
        name = name or f"ParallelAction({len(actions)})"
        description = description or f"Parallel action with {len(actions)} sub-actions"
        self.max_concurrency = max_concurrency
        super().__init__(name=name, description=description)

    async def execute(self, input_data: T) -> ActionResult:
        """Execute all actions in parallel."""
        if self.max_concurrency:
            # Use a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def execute_with_semaphore(action: Action) -> ActionResult:
                async with semaphore:
                    try:
                        return await action.execute(input_data)
                    except Exception as e:
                        logger.error(f"Error executing action {action.name} in parallel action {self.name}: {e}")
                        return ActionResult.error_result(error=e)

            # Create tasks with semaphore
            tasks = [execute_with_semaphore(action) for action in self.actions]
        else:
            # Create tasks without concurrency limit
            tasks = []
            for action in self.actions:
                task = asyncio.create_task(action.execute(input_data))
                task.action_name = action.name  # Store action name for error reporting
                tasks.append(task)

        # Wait for all tasks to complete
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                action_name = getattr(task, "action_name", "unknown")
                logger.error(f"Error executing action {action_name} in parallel action {self.name}: {e}")
                results.append(ActionResult.error_result(error=e))

        # The parallel action is successful if all sub-actions were successful
        success = all(result.success for result in results)

        return ActionResult(
            success=success,
            data=results,
            metadata={"action_count": len(self.actions), "completed_count": len(results)}
        )


class ConditionalAction(Action[T, R]):
    """An action that executes one of two actions based on a condition."""

    def __init__(
        self,
        condition: Callable[[T], bool],
        true_action: Action,
        false_action: Optional[Action] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.condition = condition
        self.true_action = true_action
        self.false_action = false_action
        name = name or f"ConditionalAction({true_action.name})"
        description = description or f"Conditional action that executes {true_action.name} if condition is true"
        super().__init__(name=name, description=description)

    async def execute(self, input_data: T) -> ActionResult:
        """Execute the appropriate action based on the condition."""
        try:
            # Evaluate the condition
            condition_result = self.condition(input_data)

            # Execute the appropriate action
            if condition_result:
                logger.debug(f"Condition is true, executing {self.true_action.name}")
                result = await self.true_action.execute(input_data)
            elif self.false_action:
                logger.debug(f"Condition is false, executing {self.false_action.name}")
                result = await self.false_action.execute(input_data)
            else:
                logger.debug("Condition is false and no false_action specified, returning success with None")
                result = ActionResult.success_result()

            return result
        except Exception as e:
            logger.error(f"Error executing conditional action {self.name}: {e}")
            return ActionResult.error_result(error=e)


class RetryAction(Action[T, R]):
    """An action that retries another action multiple times."""

    def __init__(
        self,
        action: Action,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.action = action
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        name = name or f"RetryAction({action.name})"
        description = description or f"Retry action that executes {action.name} up to {max_retries} times"
        super().__init__(name=name, description=description)

    async def execute(self, input_data: T) -> ActionResult:
        """Execute the action with retries."""
        retries = 0
        delay = self.retry_delay
        last_error = None

        while retries <= self.max_retries:
            try:
                if retries > 0:
                    logger.info(f"Retrying action {self.action.name} (attempt {retries}/{self.max_retries})")

                result = await self.action.execute(input_data)

                if result.success:
                    if retries > 0:
                        logger.info(f"Action {self.action.name} succeeded after {retries} retries")
                    return result

                last_error = result.error
                logger.warning(f"Action {self.action.name} failed: {last_error}")
            except Exception as e:
                last_error = e
                logger.warning(f"Action {self.action.name} raised exception: {e}")

            retries += 1

            if retries <= self.max_retries:
                logger.debug(f"Waiting {delay:.2f}s before retry {retries}")
                await asyncio.sleep(delay)
                delay *= self.backoff_factor

        logger.error(f"Action {self.action.name} failed after {self.max_retries} retries")
        return ActionResult.error_result(
            error=last_error or Exception(f"Action failed after {self.max_retries} retries"),
            metadata={"retries": retries - 1}
        )


class ActionRegistry:
    """Registry for actions."""

    def __init__(self):
        self.actions: Dict[str, Action] = {}

    def register(self, action: Action) -> None:
        """Register an action."""
        if action.name in self.actions:
            logger.warning(f"Overwriting existing action with name {action.name}")
        self.actions[action.name] = action
        logger.debug(f"Registered action {action.name}")

    def unregister(self, name: str) -> None:
        """Unregister an action."""
        if name in self.actions:
            del self.actions[name]
            logger.debug(f"Unregistered action {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent action {name}")

    def get(self, name: str) -> Optional[Action]:
        """Get an action by name."""
        return self.actions.get(name)

    def list_actions(self) -> List[str]:
        """List all registered action names."""
        return list(self.actions.keys())

    def clear(self) -> None:
        """Clear all registered actions."""
        self.actions.clear()
        logger.debug("Cleared all registered actions")


# Global action registry
default_registry = ActionRegistry()