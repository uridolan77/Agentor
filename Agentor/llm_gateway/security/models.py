"""
Model permissions for the LLM Gateway.

This module provides model-specific permission management.
"""

import logging
from typing import Dict, Any, Optional, List, Set, Union
import asyncio

from agentor.llm_gateway.security.auth import Permission

logger = logging.getLogger(__name__)


class ModelPermissionError(Exception):
    """Error raised for model permission operations."""
    pass


class ModelPermission:
    """Model permission definition."""
    
    # General model permissions
    USE = "use"  # Permission to use the model
    FINE_TUNE = "fine_tune"  # Permission to fine-tune the model
    ADMIN = "admin"  # Permission to administer the model
    
    # Feature-specific permissions
    STREAM = "stream"  # Permission to use streaming
    USE_TOOLS = "use_tools"  # Permission to use tools/functions
    LONG_CONTEXT = "long_context"  # Permission to use long context
    HIGH_THROUGHPUT = "high_throughput"  # Permission to use high throughput


class ModelConfig:
    """Configuration for a model."""
    
    def __init__(
        self,
        model_id: str,
        provider: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        capabilities: List[str] = None,
        default_permissions: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize model configuration.
        
        Args:
            model_id: Model ID
            provider: Provider ID
            display_name: Display name
            description: Description
            capabilities: Model capabilities
            default_permissions: Default permissions for users
            metadata: Additional metadata
        """
        self.model_id = model_id
        self.provider = provider
        self.display_name = display_name or model_id
        self.description = description or ""
        self.capabilities = capabilities or []
        self.default_permissions = default_permissions or [ModelPermission.USE]
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model configuration to dictionary.
        
        Returns:
            Model configuration as dictionary
        """
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "description": self.description,
            "capabilities": self.capabilities,
            "default_permissions": self.default_permissions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """
        Create model configuration from dictionary.
        
        Args:
            data: Model configuration data
            
        Returns:
            Model configuration instance
        """
        return cls(
            model_id=data["model_id"],
            provider=data["provider"],
            display_name=data.get("display_name"),
            description=data.get("description"),
            capabilities=data.get("capabilities"),
            default_permissions=data.get("default_permissions"),
            metadata=data.get("metadata")
        )


class ModelPermissionManager:
    """Model permission management for the LLM Gateway."""
    
    def __init__(self):
        """Initialize model permission manager."""
        self.models: Dict[str, ModelConfig] = {}
        self.lock = asyncio.Lock()
    
    async def register_model(self, model: ModelConfig) -> None:
        """
        Register a model.
        
        Args:
            model: Model configuration
            
        Raises:
            ModelPermissionError: If model already exists
        """
        async with self.lock:
            if model.model_id in self.models:
                raise ModelPermissionError(f"Model '{model.model_id}' already exists")
            
            self.models[model.model_id] = model
            logger.info(f"Registered model {model.model_id}")
    
    async def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if model was unregistered, False if not found
        """
        async with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                logger.info(f"Unregistered model {model_id}")
                return True
            return False
    
    async def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get a model by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model configuration if found, None otherwise
        """
        async with self.lock:
            return self.models.get(model_id)
    
    async def get_models(self, provider: Optional[str] = None) -> List[ModelConfig]:
        """
        Get all models.
        
        Args:
            provider: Filter by provider
            
        Returns:
            List of model configurations
        """
        async with self.lock:
            if provider:
                return [model for model in self.models.values() if model.provider == provider]
            return list(self.models.values())
    
    async def update_model(self, model: ModelConfig) -> None:
        """
        Update a model.
        
        Args:
            model: Model configuration
            
        Raises:
            ModelPermissionError: If model does not exist
        """
        async with self.lock:
            if model.model_id not in self.models:
                raise ModelPermissionError(f"Model '{model.model_id}' does not exist")
            
            self.models[model.model_id] = model
            logger.info(f"Updated model {model.model_id}")
    
    async def check_permission(
        self,
        user_permissions: Dict[str, Any],
        model_id: str,
        required_permission: str
    ) -> bool:
        """
        Check if a user has permission to use a model.
        
        Args:
            user_permissions: User permissions from JWT token
            model_id: Model ID
            required_permission: Required permission
            
        Returns:
            True if user has permission, False otherwise
        """
        # Check if user has admin permission
        if Permission.ADMIN in user_permissions.get("permissions", []):
            return True
        
        # Check if user has model-specific admin permission
        model_permissions = user_permissions.get("model_permissions", {})
        if model_id in model_permissions and ModelPermission.ADMIN in model_permissions[model_id]:
            return True
        
        # Check if user has the specific permission
        if model_id in model_permissions and required_permission in model_permissions[model_id]:
            return True
        
        # Check if model exists
        model = await self.get_model(model_id)
        if not model:
            return False
        
        # Check if permission is in default permissions
        if required_permission in model.default_permissions:
            return True
        
        return False
    
    async def load_models(self, models_data: List[Dict[str, Any]]) -> None:
        """
        Load models from data.
        
        Args:
            models_data: Model data
        """
        async with self.lock:
            for model_data in models_data:
                model = ModelConfig.from_dict(model_data)
                self.models[model.model_id] = model
            
            logger.info(f"Loaded {len(models_data)} models")
    
    async def save_models(self) -> List[Dict[str, Any]]:
        """
        Save models to data.
        
        Returns:
            Model data
        """
        async with self.lock:
            models_data = []
            
            for model in self.models.values():
                models_data.append(model.to_dict())
            
            return models_data
    
    async def create_default_models(self) -> None:
        """Create default models if no models exist."""
        async with self.lock:
            if not self.models:
                # Create default models
                await self.register_model(ModelConfig(
                    model_id="gpt-4",
                    provider="openai",
                    display_name="GPT-4",
                    description="OpenAI's most advanced model",
                    capabilities=["chat", "tools", "streaming"],
                    default_permissions=[ModelPermission.USE, ModelPermission.STREAM]
                ))
                
                await self.register_model(ModelConfig(
                    model_id="gpt-3.5-turbo",
                    provider="openai",
                    display_name="GPT-3.5 Turbo",
                    description="OpenAI's efficient model",
                    capabilities=["chat", "tools", "streaming"],
                    default_permissions=[
                        ModelPermission.USE,
                        ModelPermission.STREAM,
                        ModelPermission.USE_TOOLS,
                        ModelPermission.HIGH_THROUGHPUT
                    ]
                ))
                
                await self.register_model(ModelConfig(
                    model_id="claude-3-opus",
                    provider="anthropic",
                    display_name="Claude 3 Opus",
                    description="Anthropic's most advanced model",
                    capabilities=["chat", "tools", "streaming"],
                    default_permissions=[ModelPermission.USE, ModelPermission.STREAM]
                ))
                
                logger.info("Created default models")
