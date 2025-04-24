"""
Dependency Injection system for the Agentor framework.

This module provides a dependency injection container that supports:
- Registering and resolving dependencies
- Singleton and transient lifetimes
- Factory functions
- Automatic dependency resolution
"""

from typing import Dict, Any, Optional, Union, List, Type, TypeVar, Generic, Callable, get_type_hints, Set
import inspect
import logging
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifetime:
    """Enumeration of dependency lifetimes."""
    SINGLETON = "singleton"  # One instance for the entire application
    TRANSIENT = "transient"  # New instance each time
    SCOPED = "scoped"  # One instance per scope
    LAZY = "lazy"  # Instantiated only when first used


class LazyProxy:
    """Proxy for lazy-loaded dependencies."""

    def __init__(self, factory: Callable[[], Any]):
        """Initialize the lazy proxy.

        Args:
            factory: Factory function to create the instance
        """
        self._factory = factory
        self._instance = None

    def __getattr__(self, name):
        """Get an attribute from the proxied instance.

        Args:
            name: The attribute name

        Returns:
            The attribute value
        """
        if self._instance is None:
            self._instance = self._factory()
        return getattr(self._instance, name)

    def __call__(self, *args, **kwargs):
        """Call the proxied instance.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of calling the instance
        """
        if self._instance is None:
            self._instance = self._factory()
        return self._instance(*args, **kwargs)


class Scope:
    """Scope for scoped dependencies."""

    def __init__(self, parent: Optional['Scope'] = None):
        """Initialize the scope.

        Args:
            parent: The parent scope
        """
        self.parent = parent
        self.instances: Dict[Union[str, Type], Any] = {}

    def get(self, key: Union[str, Type]) -> Any:
        """Get an instance from the scope.

        Args:
            key: The instance key

        Returns:
            The instance, or None if not found

        Raises:
            KeyError: If the instance is not found in this scope or any parent scope
        """
        if key in self.instances:
            return self.instances[key]

        if self.parent is not None:
            return self.parent.get(key)

        raise KeyError(f"Instance '{key}' not found in scope")

    def set(self, key: Union[str, Type], instance: Any) -> None:
        """Set an instance in the scope.

        Args:
            key: The instance key
            instance: The instance
        """
        self.instances[key] = instance


class DependencyRegistration:
    """Registration of a dependency in the container."""

    def __init__(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, Any],
        lifetime: str = Lifetime.SINGLETON,
        factory: bool = False,
        lazy: bool = False
    ):
        """Initialize the dependency registration.

        Args:
            service_type: The type of the service
            implementation: The implementation of the service
            lifetime: The lifetime of the service
            factory: Whether the implementation is a factory function
            lazy: Whether to load the dependency lazily
        """
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.factory = factory
        self.lazy = lazy
        self.instance = None

    def resolve(self, container: 'Container', scope: Optional[Scope] = None) -> Any:
        """Resolve the dependency.

        Args:
            container: The dependency injection container
            scope: The current scope (for scoped dependencies)

        Returns:
            The resolved dependency
        """
        # Handle different lifetimes
        if self.lifetime == Lifetime.SINGLETON:
            # If singleton and already instantiated, return the instance
            if self.instance is not None:
                return self.instance

            # Create the instance
            instance = self._create_instance(container, scope)

            # Store the instance
            self.instance = instance

            # Return the instance (or a lazy proxy)
            if self.lazy:
                return LazyProxy(lambda: self.instance)
            return instance

        elif self.lifetime == Lifetime.SCOPED:
            # Scoped dependencies require a scope
            if scope is None:
                raise ValueError(f"Cannot resolve scoped dependency '{self.service_type.__name__}' without a scope")

            # Try to get the instance from the scope
            key = self.service_type
            try:
                return scope.get(key)
            except KeyError:
                # Create the instance
                instance = self._create_instance(container, scope)

                # Store the instance in the scope
                scope.set(key, instance)

                # Return the instance (or a lazy proxy)
                if self.lazy:
                    return LazyProxy(lambda: scope.get(key))
                return instance

        elif self.lifetime == Lifetime.LAZY:
            # Create a lazy proxy
            return LazyProxy(lambda: self._create_instance(container, scope))

        else:  # TRANSIENT
            # Create a new instance each time
            return self._create_instance(container, scope)

    def _create_instance(self, container: 'Container', scope: Optional[Scope] = None) -> Any:
        """Create an instance of the dependency.

        Args:
            container: The dependency injection container
            scope: The current scope (for scoped dependencies)

        Returns:
            The created instance
        """
        # If the implementation is a factory function, call it
        if self.factory:
            return self._resolve_factory(container, scope)
        # If the implementation is a type, instantiate it
        elif inspect.isclass(self.implementation):
            return self._resolve_class(container, scope)
        # Otherwise, the implementation is an instance
        else:
            return self.implementation

    def _resolve_factory(self, container: 'Container', scope: Optional[Scope] = None) -> Any:
        """Resolve a factory function.

        Args:
            container: The dependency injection container
            scope: The current scope (for scoped dependencies)

        Returns:
            The result of calling the factory function
        """
        # Get the factory function's parameters
        params = inspect.signature(self.implementation).parameters

        # Resolve the parameters
        args = []
        kwargs = {}

        for name, param in params.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get the parameter type
            param_type = param.annotation

            # If the parameter has a default value and no type annotation, use the default
            if param_type is inspect.Parameter.empty and param.default is not inspect.Parameter.empty:
                continue

            # If the parameter has no type annotation, try to resolve by name
            if param_type is inspect.Parameter.empty:
                # Try to resolve by name
                try:
                    value = container.resolve(name, scope)
                    kwargs[name] = value
                except KeyError:
                    # If the parameter has a default value, use it
                    if param.default is not inspect.Parameter.empty:
                        continue
                    else:
                        raise ValueError(f"Cannot resolve parameter '{name}' for factory function")
            else:
                # Resolve by type
                value = container.resolve(param_type, scope)

                # If the parameter is positional-only, add it to args
                if param.kind == param.POSITIONAL_ONLY:
                    args.append(value)
                # Otherwise, add it to kwargs
                else:
                    kwargs[name] = value

        # Call the factory function
        return self.implementation(*args, **kwargs)

    def _resolve_class(self, container: 'Container', scope: Optional[Scope] = None) -> Any:
        """Resolve a class.

        Args:
            container: The dependency injection container
            scope: The current scope (for scoped dependencies)

        Returns:
            An instance of the class
        """
        # Get the constructor parameters
        params = inspect.signature(self.implementation.__init__).parameters

        # Resolve the parameters
        kwargs = {}

        for name, param in params.items():
            # Skip self, *args, and **kwargs
            if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get the parameter type
            param_type = param.annotation

            # If the parameter has a default value and no type annotation, use the default
            if param_type is inspect.Parameter.empty and param.default is not inspect.Parameter.empty:
                continue

            # If the parameter has no type annotation, try to resolve by name
            if param_type is inspect.Parameter.empty:
                # Try to resolve by name
                try:
                    kwargs[name] = container.resolve(name, scope)
                except KeyError:
                    # If the parameter has a default value, use it
                    if param.default is not inspect.Parameter.empty:
                        continue
                    else:
                        raise ValueError(f"Cannot resolve parameter '{name}' for class {self.implementation.__name__}")
            else:
                # Resolve by type
                kwargs[name] = container.resolve(param_type, scope)

        # Instantiate the class
        return self.implementation(**kwargs)


class Container:
    """Dependency injection container for the Agentor framework."""

    def __init__(self):
        """Initialize the dependency injection container."""
        self._registrations: Dict[Union[str, Type], DependencyRegistration] = {}
        self._root_scope = Scope()
        self._current_scope = self._root_scope

    def register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T], T],
        lifetime: str = Lifetime.SINGLETON,
        factory: bool = False,
        lazy: bool = False,
        name: Optional[str] = None
    ) -> None:
        """Register a dependency.

        Args:
            service_type: The type of the service
            implementation: The implementation of the service
            lifetime: The lifetime of the service
            factory: Whether the implementation is a factory function
            name: An optional name for the registration

        Raises:
            ValueError: If the service type is already registered
        """
        # Use the name if provided, otherwise use the service type
        key = name if name is not None else service_type

        # Check if the service type is already registered
        if key in self._registrations:
            raise ValueError(f"Service '{key}' is already registered")

        # Register the dependency
        self._registrations[key] = DependencyRegistration(
            service_type=service_type,
            implementation=implementation,
            lifetime=lifetime,
            factory=factory,
            lazy=lazy
        )

        logger.debug(f"Registered dependency: {key}")

    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
        name: Optional[str] = None
    ) -> None:
        """Register an instance.

        Args:
            service_type: The type of the service
            instance: The instance of the service
            name: An optional name for the registration

        Raises:
            ValueError: If the service type is already registered
        """
        self.register(
            service_type=service_type,
            implementation=instance,
            lifetime=Lifetime.SINGLETON,
            factory=False,
            name=name
        )

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        lifetime: str = Lifetime.SINGLETON,
        lazy: bool = False,
        name: Optional[str] = None
    ) -> None:
        """Register a factory function.

        Args:
            service_type: The type of the service
            factory: The factory function
            lifetime: The lifetime of the service
            name: An optional name for the registration

        Raises:
            ValueError: If the service type is already registered
        """
        self.register(
            service_type=service_type,
            implementation=factory,
            lifetime=lifetime,
            factory=True,
            lazy=lazy,
            name=name
        )

    def create_scope(self) -> Scope:
        """Create a new scope.

        Returns:
            The new scope
        """
        return Scope(parent=self._current_scope)

    @contextmanager
    def scope(self):
        """Create a new scope and set it as the current scope.

        Yields:
            The new scope
        """
        old_scope = self._current_scope
        new_scope = Scope(parent=old_scope)
        self._current_scope = new_scope

        try:
            yield new_scope
        finally:
            self._current_scope = old_scope

    def resolve(self, service: Union[str, Type[T]], scope: Optional[Scope] = None) -> T:
        """Resolve a dependency.

        Args:
            service: The service type or name
            scope: The scope to use for scoped dependencies

        Returns:
            The resolved dependency

        Raises:
            KeyError: If the service is not registered
        """
        # Use the current scope if none is provided
        if scope is None:
            scope = self._current_scope

        # Check if the service is registered
        if service not in self._registrations:
            raise KeyError(f"Service '{service}' is not registered")

        # Resolve the dependency
        return self._registrations[service].resolve(self, scope)

    def resolve_all(self, service_type: Type[T], scope: Optional[Scope] = None) -> List[T]:
        """Resolve all dependencies of a type.

        Args:
            service_type: The service type
            scope: The scope to use for scoped dependencies

        Returns:
            A list of resolved dependencies
        """
        # Use the current scope if none is provided
        if scope is None:
            scope = self._current_scope

        # Find all registrations of the service type
        registrations = [
            reg for key, reg in self._registrations.items()
            if isinstance(key, type) and issubclass(key, service_type)
        ]

        # Resolve the dependencies
        return [reg.resolve(self, scope) for reg in registrations]

    def inject(self, func: Callable) -> Callable:
        """Decorator to inject dependencies into a function.

        Args:
            func: The function to inject dependencies into

        Returns:
            The decorated function
        """
        # Get the function's parameters
        params = inspect.signature(func).parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a dictionary of resolved dependencies
            resolved_kwargs = {}

            # Skip the first parameter if it's self or cls
            skip_first = False
            if params and list(params.keys())[0] in ("self", "cls"):
                skip_first = True

            # Resolve the parameters
            for i, (name, param) in enumerate(params.items()):
                # Skip self, cls, *args, and **kwargs
                if (skip_first and i == 0) or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue

                # Skip parameters that are already provided
                if name in kwargs or i < len(args):
                    continue

                # Get the parameter type
                param_type = param.annotation

                # If the parameter has a default value and no type annotation, use the default
                if param_type is inspect.Parameter.empty and param.default is not inspect.Parameter.empty:
                    continue

                # If the parameter has no type annotation, try to resolve by name
                if param_type is inspect.Parameter.empty:
                    # Try to resolve by name
                    try:
                        resolved_kwargs[name] = self.resolve(name, self._current_scope)
                    except KeyError:
                        # If the parameter has a default value, use it
                        if param.default is not inspect.Parameter.empty:
                            continue
                        else:
                            raise ValueError(f"Cannot resolve parameter '{name}' for function {func.__name__}")
                else:
                    # Resolve by type
                    try:
                        resolved_kwargs[name] = self.resolve(param_type, self._current_scope)
                    except KeyError:
                        # If the parameter has a default value, use it
                        if param.default is not inspect.Parameter.empty:
                            continue
                        else:
                            raise ValueError(f"Cannot resolve parameter '{name}' of type {param_type.__name__} for function {func.__name__}")

            # Call the function with the resolved dependencies
            return func(*args, **{**resolved_kwargs, **kwargs})

        return wrapper


# Global dependency injection container
container = Container()


def get_container() -> Container:
    """Get the global dependency injection container.

    Returns:
        The global dependency injection container
    """
    return container


def inject(func: Callable) -> Callable:
    """Decorator to inject dependencies into a function.

    Args:
        func: The function to inject dependencies into

    Returns:
        The decorated function
    """
    return container.inject(func)
