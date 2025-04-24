"""
Sensors module for the Agentor framework.

This module provides interfaces and implementations for agent sensors.
Sensors are the ways that agents perceive their environment.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
import asyncio
import inspect
import logging
import time
from datetime import datetime

from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type


class SensorReading:
    """Result of a sensor reading."""

    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()

    def __str__(self) -> str:
        if self.success:
            return f"SensorReading(success={self.success}, data={self.data}, timestamp={self.timestamp})"
        else:
            return f"SensorReading(success={self.success}, error={self.error}, timestamp={self.timestamp})"

    @classmethod
    def success_reading(cls, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> 'SensorReading':
        """Create a successful sensor reading."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_reading(cls, error: Union[Exception, str], metadata: Optional[Dict[str, Any]] = None) -> 'SensorReading':
        """Create an error sensor reading."""
        if isinstance(error, str):
            error = Exception(error)
        return cls(success=False, error=error, metadata=metadata)


class Sensor(Generic[T, R], ABC):
    """Base class for all sensors."""

    def __init__(self, name: str, description: Optional[str] = None):
        self.name = name
        self.description = description or f"Sensor: {name}"
        self.last_reading: Optional[SensorReading] = None
        self.last_read_time: Optional[float] = None

    @abstractmethod
    async def read(self, input_data: Optional[T] = None) -> SensorReading:
        """Read data from the sensor."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class FunctionSensor(Sensor[T, R]):
    """A sensor that reads data by executing a function."""

    def __init__(
        self,
        func: callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.func = func
        name = name or func.__name__
        description = description or func.__doc__ or f"Function sensor: {name}"
        super().__init__(name=name, description=description)

    async def read(self, input_data: Optional[T] = None) -> SensorReading:
        """Read data by executing the function."""
        try:
            # Record the read time
            self.last_read_time = time.time()

            # Check if the function is a coroutine function
            if inspect.iscoroutinefunction(self.func):
                if input_data is not None:
                    result = await self.func(input_data)
                else:
                    result = await self.func()
            else:
                # Run synchronous functions in a thread pool
                loop = asyncio.get_event_loop()
                if input_data is not None:
                    result = await loop.run_in_executor(None, self.func, input_data)
                else:
                    result = await loop.run_in_executor(None, self.func)

            # Create a successful reading
            reading = SensorReading.success_reading(
                data=result,
                metadata={"read_time": self.last_read_time}
            )
            self.last_reading = reading
            return reading
        except Exception as e:
            logger.error(f"Error reading from sensor {self.name}: {e}")
            reading = SensorReading.error_reading(
                error=e,
                metadata={"read_time": self.last_read_time}
            )
            self.last_reading = reading
            return reading


class CachedSensor(Sensor[T, R]):
    """A sensor that caches readings for a specified duration."""

    def __init__(
        self,
        sensor: Sensor[T, R],
        cache_duration: float = 60.0,  # Cache duration in seconds
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.sensor = sensor
        self.cache_duration = cache_duration
        name = name or f"Cached{sensor.name}"
        description = description or f"Cached sensor for {sensor.name} (cache duration: {cache_duration}s)"
        super().__init__(name=name, description=description)

    async def read(self, input_data: Optional[T] = None) -> SensorReading:
        """Read data from the sensor, using cached data if available and not expired."""
        current_time = time.time()

        # Check if we have a cached reading that's still valid
        if (
            self.last_reading is not None and
            self.last_read_time is not None and
            current_time - self.last_read_time < self.cache_duration
        ):
            logger.debug(f"Using cached reading for sensor {self.name} (age: {current_time - self.last_read_time:.2f}s)")
            return self.last_reading

        # Otherwise, get a fresh reading
        logger.debug(f"Cache expired or not available for sensor {self.name}, getting fresh reading")
        reading = await self.sensor.read(input_data)
        self.last_reading = reading
        self.last_read_time = current_time
        return reading


class CompositeSensor(Sensor[T, Dict[str, Any]]):
    """A sensor that combines readings from multiple sensors."""

    def __init__(
        self,
        sensors: Dict[str, Sensor],
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.sensors = sensors
        name = name or f"CompositeSensor({len(sensors)})"
        description = description or f"Composite sensor with {len(sensors)} sub-sensors"
        super().__init__(name=name, description=description)

    async def read(self, input_data: Optional[T] = None) -> SensorReading:
        """Read data from all sensors."""
        results = {}
        success = True
        errors = []

        # Read from all sensors
        for sensor_name, sensor in self.sensors.items():
            try:
                reading = await sensor.read(input_data)
                results[sensor_name] = reading
                if not reading.success:
                    success = False
                    errors.append((sensor_name, reading.error))
            except Exception as e:
                logger.error(f"Error reading from sensor {sensor_name} in composite sensor {self.name}: {e}")
                results[sensor_name] = SensorReading.error_reading(error=e)
                success = False
                errors.append((sensor_name, e))

        # Create a composite reading
        if success:
            reading = SensorReading.success_reading(
                data=results,
                metadata={"sensor_count": len(self.sensors)}
            )
        else:
            error_msg = "; ".join([f"{name}: {error}" for name, error in errors])
            reading = SensorReading.error_reading(
                error=f"Errors in composite sensor: {error_msg}",
                metadata={
                    "sensor_count": len(self.sensors),
                    "error_count": len(errors),
                    "results": results
                }
            )

        self.last_reading = reading
        self.last_read_time = time.time()
        return reading


class ParallelSensor(Sensor[T, Dict[str, Any]]):
    """A sensor that reads from multiple sensors in parallel."""

    def __init__(
        self,
        sensors: Dict[str, Sensor],
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_concurrency: Optional[int] = None
    ):
        self.sensors = sensors
        self.max_concurrency = max_concurrency
        name = name or f"ParallelSensor({len(sensors)})"
        description = description or f"Parallel sensor with {len(sensors)} sub-sensors"
        super().__init__(name=name, description=description)

    async def read(self, input_data: Optional[T] = None) -> SensorReading:
        """Read data from all sensors in parallel."""
        if self.max_concurrency:
            # Use a semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def read_with_semaphore(sensor_name: str, sensor: Sensor) -> tuple[str, SensorReading]:
                async with semaphore:
                    try:
                        reading = await sensor.read(input_data)
                        return sensor_name, reading
                    except Exception as e:
                        logger.error(f"Error reading from sensor {sensor_name} in parallel sensor {self.name}: {e}")
                        return sensor_name, SensorReading.error_reading(error=e)

            # Create tasks with semaphore
            tasks = [read_with_semaphore(name, sensor) for name, sensor in self.sensors.items()]
        else:
            # Create tasks without concurrency limit
            async def read_sensor(sensor_name: str, sensor: Sensor) -> tuple[str, SensorReading]:
                try:
                    reading = await sensor.read(input_data)
                    return sensor_name, reading
                except Exception as e:
                    logger.error(f"Error reading from sensor {sensor_name} in parallel sensor {self.name}: {e}")
                    return sensor_name, SensorReading.error_reading(error=e)

            tasks = [read_sensor(name, sensor) for name, sensor in self.sensors.items()]

        # Wait for all tasks to complete
        results = {}
        success = True
        errors = []

        for result in await asyncio.gather(*tasks):
            sensor_name, reading = result
            results[sensor_name] = reading
            if not reading.success:
                success = False
                errors.append((sensor_name, reading.error))

        # Create a parallel reading
        if success:
            reading = SensorReading.success_reading(
                data=results,
                metadata={"sensor_count": len(self.sensors)}
            )
        else:
            error_msg = "; ".join([f"{name}: {error}" for name, error in errors])
            reading = SensorReading.error_reading(
                error=f"Errors in parallel sensor: {error_msg}",
                metadata={
                    "sensor_count": len(self.sensors),
                    "error_count": len(errors),
                    "results": results
                }
            )

        self.last_reading = reading
        self.last_read_time = time.time()
        return reading


class FilteringSensor(Sensor[T, R]):
    """A sensor that filters readings from another sensor."""

    def __init__(
        self,
        sensor: Sensor[T, R],
        filter_func: callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.sensor = sensor
        self.filter_func = filter_func
        name = name or f"Filtered{sensor.name}"
        description = description or f"Filtered sensor for {sensor.name}"
        super().__init__(name=name, description=description)

    async def read(self, input_data: Optional[T] = None) -> SensorReading:
        """Read data from the sensor and apply the filter."""
        reading = await self.sensor.read(input_data)

        if not reading.success:
            return reading

        try:
            # Apply the filter function
            filtered_data = self.filter_func(reading.data)

            # Create a new reading with the filtered data
            filtered_reading = SensorReading.success_reading(
                data=filtered_data,
                metadata={
                    **reading.metadata,
                    "original_data": reading.data
                }
            )
            self.last_reading = filtered_reading
            self.last_read_time = time.time()
            return filtered_reading
        except Exception as e:
            logger.error(f"Error applying filter in sensor {self.name}: {e}")
            error_reading = SensorReading.error_reading(
                error=e,
                metadata={
                    **reading.metadata,
                    "original_data": reading.data
                }
            )
            self.last_reading = error_reading
            self.last_read_time = time.time()
            return error_reading


class SensorRegistry:
    """Registry for sensors."""

    def __init__(self):
        self.sensors: Dict[str, Sensor] = {}

    def register(self, sensor: Sensor) -> None:
        """Register a sensor."""
        if sensor.name in self.sensors:
            logger.warning(f"Overwriting existing sensor with name {sensor.name}")
        self.sensors[sensor.name] = sensor
        logger.debug(f"Registered sensor {sensor.name}")

    def unregister(self, name: str) -> None:
        """Unregister a sensor."""
        if name in self.sensors:
            del self.sensors[name]
            logger.debug(f"Unregistered sensor {name}")
        else:
            logger.warning(f"Attempted to unregister non-existent sensor {name}")

    def get(self, name: str) -> Optional[Sensor]:
        """Get a sensor by name."""
        return self.sensors.get(name)

    def list_sensors(self) -> List[str]:
        """List all registered sensor names."""
        return list(self.sensors.keys())

    def clear(self) -> None:
        """Clear all registered sensors."""
        self.sensors.clear()
        logger.debug("Cleared all registered sensors")


# Global sensor registry
default_registry = SensorRegistry()