"""
Serialization utilities for the Agentor framework.

This module provides utilities for serializing and deserializing objects,
including custom serializers for complex objects like agents, tools, and pipelines.
"""

import json
import pickle
import base64
import inspect
import importlib
from typing import Any, Dict, List, Optional, Type, Union, Callable


class SerializationError(Exception):
    """Exception raised for serialization errors."""
    pass


class DeserializationError(Exception):
    """Exception raised for deserialization errors."""
    pass


class Serializable:
    """Base class for objects that can be serialized."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary."""
        raise NotImplementedError("Subclasses must implement to_dict()")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create an object from a dictionary."""
        raise NotImplementedError("Subclasses must implement from_dict()")


class SerializationRegistry:
    """Registry for serialization handlers."""

    _serializers: Dict[Type, Callable[[Any], Dict[str, Any]]] = {}
    _deserializers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    @classmethod
    def register_serializer(cls, type_: Type, serializer: Callable[[Any], Dict[str, Any]]) -> None:
        """Register a serializer for a type."""
        cls._serializers[type_] = serializer

    @classmethod
    def register_deserializer(cls, type_name: str, deserializer: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a deserializer for a type."""
        cls._deserializers[type_name] = deserializer

    @classmethod
    def get_serializer(cls, obj: Any) -> Optional[Callable[[Any], Dict[str, Any]]]:
        """Get a serializer for an object."""
        for type_, serializer in cls._serializers.items():
            if isinstance(obj, type_):
                return serializer
        return None

    @classmethod
    def get_deserializer(cls, type_name: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
        """Get a deserializer for a type."""
        return cls._deserializers.get(type_name)


def serialize(obj: Any) -> Dict[str, Any]:
    """Serialize an object to a dictionary."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"type": type(obj).__name__, "value": obj}

    if isinstance(obj, list):
        return {"type": "list", "value": [serialize(item) for item in obj]}

    if isinstance(obj, dict):
        return {
            "type": "dict",
            "value": {key: serialize(value) for key, value in obj.items()}
        }

    if isinstance(obj, Serializable):
        return {
            "type": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "value": obj.to_dict()
        }

    # Check for registered serializers
    serializer = SerializationRegistry.get_serializer(obj)
    if serializer:
        result = serializer(obj)
        result["module"] = obj.__class__.__module__
        return result

    # Fallback to pickle for objects that can't be serialized otherwise
    try:
        pickle_data = pickle.dumps(obj)
        return {
            "type": "pickle",
            "class": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "value": base64.b64encode(pickle_data).decode("utf-8")
        }
    except (pickle.PickleError, TypeError) as e:
        raise SerializationError(f"Cannot serialize object of type {type(obj).__name__}: {e}")


def deserialize(data: Dict[str, Any]) -> Any:
    """Deserialize an object from a dictionary."""
    if not isinstance(data, dict) or "type" not in data:
        raise DeserializationError("Invalid serialization data")

    type_name = data["type"]

    if type_name in ("NoneType", "bool", "int", "float", "str"):
        return data["value"]

    if type_name == "list":
        return [deserialize(item) for item in data["value"]]

    if type_name == "dict":
        return {key: deserialize(value) for key, value in data["value"].items()}

    if type_name == "pickle":
        try:
            module_name = data.get("module")
            class_name = data.get("class")
            pickle_data = base64.b64decode(data["value"])
            obj = pickle.loads(pickle_data)

            # Verify that the unpickled object is of the expected type
            if module_name and class_name:
                expected_class = getattr(importlib.import_module(module_name), class_name)
                if not isinstance(obj, expected_class):
                    raise DeserializationError(
                        f"Unpickled object is of type {type(obj).__name__}, expected {class_name}"
                    )

            return obj
        except (pickle.PickleError, TypeError, ImportError, AttributeError) as e:
            raise DeserializationError(f"Cannot deserialize pickled object: {e}")

    # Check for registered deserializers
    deserializer = SerializationRegistry.get_deserializer(type_name)
    if deserializer:
        return deserializer(data)

    # Try to instantiate the class from the module
    if "module" in data:
        try:
            module = importlib.import_module(data["module"])
            cls = getattr(module, type_name)

            if issubclass(cls, Serializable):
                return cls.from_dict(data["value"])

            # Check if the class has a from_dict method
            if hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
                from_dict = getattr(cls, "from_dict")
                if inspect.ismethod(from_dict) and inspect.isclass(from_dict.__self__):
                    return from_dict(data["value"])

        except (ImportError, AttributeError) as e:
            raise DeserializationError(f"Cannot deserialize object of type {type_name}: {e}")

    raise DeserializationError(f"No deserializer found for type {type_name}")


def to_json(obj: Any, **kwargs) -> str:
    """Serialize an object to a JSON string."""
    return json.dumps(serialize(obj), **kwargs)


def from_json(json_str: str) -> Any:
    """Deserialize an object from a JSON string."""
    try:
        data = json.loads(json_str)
        return deserialize(data)
    except json.JSONDecodeError as e:
        raise DeserializationError(f"Invalid JSON: {e}")


def register_serialization_handlers() -> None:
    """Register serialization handlers for common types."""
    # Example: Register handlers for datetime objects
    from datetime import datetime, date, time

    def serialize_datetime(dt: datetime) -> Dict[str, Any]:
        return {
            "type": "datetime",
            "value": dt.isoformat()
        }

    def deserialize_datetime(data: Dict[str, Any]) -> datetime:
        return datetime.fromisoformat(data["value"])

    def serialize_date(d: date) -> Dict[str, Any]:
        return {
            "type": "date",
            "value": d.isoformat()
        }

    def deserialize_date(data: Dict[str, Any]) -> date:
        return date.fromisoformat(data["value"])

    def serialize_time(t: time) -> Dict[str, Any]:
        return {
            "type": "time",
            "value": t.isoformat()
        }

    def deserialize_time(data: Dict[str, Any]) -> time:
        return time.fromisoformat(data["value"])

    SerializationRegistry.register_serializer(datetime, serialize_datetime)
    SerializationRegistry.register_deserializer("datetime", deserialize_datetime)

    SerializationRegistry.register_serializer(date, serialize_date)
    SerializationRegistry.register_deserializer("date", deserialize_date)

    SerializationRegistry.register_serializer(time, serialize_time)
    SerializationRegistry.register_deserializer("time", deserialize_time)


# Register serialization handlers
register_serialization_handlers()