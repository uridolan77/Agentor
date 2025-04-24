"""
Configuration system for the Agentor framework.

This module provides a unified configuration system that supports:
- Loading configuration from environment variables
- Loading configuration from files (JSON, YAML, TOML)
- Loading configuration from code
- Hierarchical configuration with overrides
- Type validation using Pydantic
"""

from typing import Dict, Any, Optional, Union, List, Type, TypeVar, Generic, get_type_hints
import os
import json
import logging
from pathlib import Path
from pydantic import BaseModel, create_model, Field
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class ConfigSource:
    """Base class for configuration sources."""
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from this source.
        
        Returns:
            The configuration as a dictionary
        """
        return {}


class EnvConfigSource(ConfigSource):
    """Configuration source that loads from environment variables."""
    
    def __init__(self, prefix: str = "AGENTOR_", separator: str = "__"):
        """Initialize the environment configuration source.
        
        Args:
            prefix: The prefix for environment variables
            separator: The separator for nested keys
        """
        self.prefix = prefix
        self.separator = separator
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from environment variables.
        
        Returns:
            The configuration as a dictionary
        """
        config = {}
        
        # Get all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove the prefix
                key = key[len(self.prefix):]
                
                # Split the key by separator
                parts = key.split(self.separator)
                
                # Build the nested dictionary
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1]] = self._parse_value(value)
        
        return config
    
    def _parse_value(self, value: str) -> Any:
        """Parse a string value into a Python object.
        
        Args:
            value: The string value
            
        Returns:
            The parsed value
        """
        # Try to parse as JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try to parse as boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Try to parse as number
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value


class FileConfigSource(ConfigSource):
    """Configuration source that loads from a file."""
    
    def __init__(self, file_path: str):
        """Initialize the file configuration source.
        
        Args:
            file_path: The path to the configuration file
        """
        self.file_path = file_path
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from the file.
        
        Returns:
            The configuration as a dictionary
        """
        path = Path(self.file_path)
        
        if not path.exists():
            logger.warning(f"Configuration file '{self.file_path}' does not exist")
            return {}
        
        try:
            # Load the file based on its extension
            if path.suffix.lower() == ".json":
                return self._load_json(path)
            elif path.suffix.lower() in (".yaml", ".yml"):
                return self._load_yaml(path)
            elif path.suffix.lower() == ".toml":
                return self._load_toml(path)
            else:
                logger.warning(f"Unsupported configuration file format: {path.suffix}")
                return {}
        
        except Exception as e:
            logger.error(f"Error loading configuration file '{self.file_path}': {str(e)}")
            return {}
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load a JSON configuration file.
        
        Args:
            path: The path to the JSON file
            
        Returns:
            The configuration as a dictionary
        """
        with open(path, "r") as f:
            return json.load(f)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            path: The path to the YAML file
            
        Returns:
            The configuration as a dictionary
        """
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML is not installed. Install it with 'pip install pyyaml'")
            return {}
    
    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load a TOML configuration file.
        
        Args:
            path: The path to the TOML file
            
        Returns:
            The configuration as a dictionary
        """
        try:
            import tomli
            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            logger.error("Tomli is not installed. Install it with 'pip install tomli'")
            return {}


class DictConfigSource(ConfigSource):
    """Configuration source that loads from a dictionary."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dictionary configuration source.
        
        Args:
            config: The configuration dictionary
        """
        self.config = config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from the dictionary.
        
        Returns:
            The configuration as a dictionary
        """
        return self.config


class ConfigManager:
    """Manager for configuration in the Agentor framework."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.sources: List[ConfigSource] = []
        self.config_cache: Dict[str, Any] = {}
        self.model_cache: Dict[Type[BaseModel], BaseModel] = {}
    
    def add_source(self, source: ConfigSource, priority: int = 0) -> None:
        """Add a configuration source.
        
        Args:
            source: The configuration source
            priority: The priority of the source (higher priority sources override lower priority sources)
        """
        self.sources.append((priority, source))
        self.sources.sort(key=lambda x: x[0], reverse=True)
        
        # Clear the cache
        self.config_cache = {}
        self.model_cache = {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get the merged configuration from all sources.
        
        Returns:
            The merged configuration as a dictionary
        """
        if not self.config_cache:
            # Merge configurations from all sources
            config = {}
            for _, source in self.sources:
                self._merge_config(config, source.get_config())
            
            self.config_cache = config
        
        return self.config_cache
    
    def get_config_section(self, section: str) -> Dict[str, Any]:
        """Get a section of the configuration.
        
        Args:
            section: The section name
            
        Returns:
            The section as a dictionary
        """
        config = self.get_config()
        
        # Split the section by dots
        parts = section.split(".")
        
        # Navigate to the section
        current = config
        for part in parts:
            if part not in current:
                return {}
            current = current[part]
        
        return current if isinstance(current, dict) else {}
    
    def get_typed_config(self, model_type: Type[T]) -> T:
        """Get a typed configuration.
        
        Args:
            model_type: The Pydantic model type
            
        Returns:
            An instance of the model with the configuration values
        """
        if model_type in self.model_cache:
            return self.model_cache[model_type]
        
        # Get the configuration
        config = self.get_config()
        
        # Get the model's field info
        field_info = get_type_hints(model_type)
        
        # Create a dictionary with the model's fields
        model_dict = {}
        for field_name, field_type in field_info.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue
            
            # Get the field value from the configuration
            field_value = self._get_field_value(config, field_name)
            
            # Add the field to the dictionary
            if field_value is not None:
                model_dict[field_name] = field_value
        
        # Create an instance of the model
        model_instance = model_type(**model_dict)
        
        # Cache the model instance
        self.model_cache[model_type] = model_instance
        
        return model_instance
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Merge a source configuration into a target configuration.
        
        Args:
            target: The target configuration
            source: The source configuration
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                self._merge_config(target[key], value)
            else:
                # Override the value
                target[key] = value
    
    def _get_field_value(self, config: Dict[str, Any], field_name: str) -> Any:
        """Get a field value from the configuration.
        
        Args:
            config: The configuration dictionary
            field_name: The field name
            
        Returns:
            The field value, or None if not found
        """
        # Check if the field exists in the configuration
        if field_name in config:
            return config[field_name]
        
        # Check if the field is in a section
        parts = field_name.split("_")
        current = config
        
        for i in range(len(parts) - 1):
            section = "_".join(parts[:i+1])
            if section in current:
                current = current[section]
                remaining = "_".join(parts[i+1:])
                if remaining in current:
                    return current[remaining]
        
        return None


# Global configuration manager
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager.
    
    Returns:
        The global configuration manager
    """
    return config_manager


def get_config() -> Dict[str, Any]:
    """Get the merged configuration from all sources.
    
    Returns:
        The merged configuration as a dictionary
    """
    return config_manager.get_config()


def get_config_section(section: str) -> Dict[str, Any]:
    """Get a section of the configuration.
    
    Args:
        section: The section name
        
    Returns:
        The section as a dictionary
    """
    return config_manager.get_config_section(section)


def get_typed_config(model_type: Type[T]) -> T:
    """Get a typed configuration.
    
    Args:
        model_type: The Pydantic model type
        
    Returns:
        An instance of the model with the configuration values
    """
    return config_manager.get_typed_config(model_type)


# Initialize the configuration manager with default sources
config_manager.add_source(EnvConfigSource(), priority=100)  # Environment variables have highest priority
config_manager.add_source(FileConfigSource("config.json"), priority=50)  # JSON file has medium priority
config_manager.add_source(FileConfigSource("config.yaml"), priority=50)  # YAML file has medium priority
config_manager.add_source(FileConfigSource("config.toml"), priority=50)  # TOML file has medium priority
