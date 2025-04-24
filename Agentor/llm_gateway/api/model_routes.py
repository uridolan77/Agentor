"""
Model management routes for the LLM Gateway.
"""

from fastapi import APIRouter, Depends, HTTPException, Security, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from agentor.llm_gateway.security.auth import JWTAuth, Role, Permission
from agentor.llm_gateway.security.models import ModelPermissionManager, ModelConfig, ModelPermission, ModelPermissionError
from agentor.llm_gateway.security.users import UserManager, UserError


# Models for request and response
class ModelConfigRequest(BaseModel):
    """Model configuration request."""
    model_id: str = Field(..., description="Model ID")
    provider: str = Field(..., description="Provider ID")
    display_name: Optional[str] = Field(None, description="Display name")
    description: Optional[str] = Field(None, description="Description")
    capabilities: Optional[List[str]] = Field(None, description="Model capabilities")
    default_permissions: Optional[List[str]] = Field(None, description="Default permissions for users")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelConfigResponse(BaseModel):
    """Model configuration response."""
    model_id: str = Field(..., description="Model ID")
    provider: str = Field(..., description="Provider ID")
    display_name: str = Field(..., description="Display name")
    description: str = Field(..., description="Description")
    capabilities: List[str] = Field(..., description="Model capabilities")
    default_permissions: List[str] = Field(..., description="Default permissions for users")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class ModelPermissionRequest(BaseModel):
    """Model permission request."""
    user_id: str = Field(..., description="User ID")
    model_id: str = Field(..., description="Model ID")
    permission: str = Field(..., description="Permission to add")


# Create router
router = APIRouter(prefix="/models", tags=["Models"])


# Dependency to get model permission manager from app state
def get_model_permission_manager(request: Request) -> ModelPermissionManager:
    """Get model permission manager from app state."""
    return request.app.state.security_components["model_permission_manager"]


# Dependency to get user manager from app state
def get_user_manager(request: Request) -> UserManager:
    """Get user manager from app state."""
    return request.app.state.security_components["user_manager"]


# Dependency to get JWT auth from app state
def get_jwt_auth(request: Request) -> JWTAuth:
    """Get JWT auth from app state."""
    return request.app.state.security_components["jwt_auth"]


@router.get("", response_model=List[ModelConfigResponse])
async def get_models(
    provider: Optional[str] = None,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.get_current_user)
):
    """
    Get all models.
    
    Args:
        provider: Filter by provider
        model_permission_manager: Model permission manager
        current_user: Current user
        
    Returns:
        List of model configurations
    """
    # Get models
    models = await model_permission_manager.get_models(provider)
    
    # Convert to response
    return [
        ModelConfigResponse(
            model_id=model.model_id,
            provider=model.provider,
            display_name=model.display_name,
            description=model.description,
            capabilities=model.capabilities,
            default_permissions=model.default_permissions,
            metadata=model.metadata
        )
        for model in models
    ]


@router.get("/{model_id}", response_model=ModelConfigResponse)
async def get_model(
    model_id: str,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.get_current_user)
):
    """
    Get a model by ID.
    
    Args:
        model_id: Model ID
        model_permission_manager: Model permission manager
        current_user: Current user
        
    Returns:
        Model configuration
        
    Raises:
        HTTPException: If model not found
    """
    # Get model
    model = await model_permission_manager.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    # Convert to response
    return ModelConfigResponse(
        model_id=model.model_id,
        provider=model.provider,
        display_name=model.display_name,
        description=model.description,
        capabilities=model.capabilities,
        default_permissions=model.default_permissions,
        metadata=model.metadata
    )


@router.post("", response_model=ModelConfigResponse)
async def create_model(
    request: ModelConfigRequest,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.requires_role(Role.ADMIN))
):
    """
    Create a new model.
    
    Args:
        request: Model configuration request
        model_permission_manager: Model permission manager
        current_user: Current user (must be admin)
        
    Returns:
        Created model configuration
        
    Raises:
        HTTPException: If model creation fails
    """
    try:
        # Create model
        model = ModelConfig(
            model_id=request.model_id,
            provider=request.provider,
            display_name=request.display_name,
            description=request.description,
            capabilities=request.capabilities,
            default_permissions=request.default_permissions,
            metadata=request.metadata
        )
        
        # Register model
        await model_permission_manager.register_model(model)
        
        # Convert to response
        return ModelConfigResponse(
            model_id=model.model_id,
            provider=model.provider,
            display_name=model.display_name,
            description=model.description,
            capabilities=model.capabilities,
            default_permissions=model.default_permissions,
            metadata=model.metadata
        )
    except ModelPermissionError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating model: {str(e)}"
        )


@router.put("/{model_id}", response_model=ModelConfigResponse)
async def update_model(
    model_id: str,
    request: ModelConfigRequest,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.requires_role(Role.ADMIN))
):
    """
    Update a model.
    
    Args:
        model_id: Model ID
        request: Model configuration request
        model_permission_manager: Model permission manager
        current_user: Current user (must be admin)
        
    Returns:
        Updated model configuration
        
    Raises:
        HTTPException: If model update fails
    """
    try:
        # Check if model exists
        existing_model = await model_permission_manager.get_model(model_id)
        if not existing_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found"
            )
        
        # Create updated model
        model = ModelConfig(
            model_id=model_id,
            provider=request.provider,
            display_name=request.display_name,
            description=request.description,
            capabilities=request.capabilities,
            default_permissions=request.default_permissions,
            metadata=request.metadata
        )
        
        # Update model
        await model_permission_manager.update_model(model)
        
        # Convert to response
        return ModelConfigResponse(
            model_id=model.model_id,
            provider=model.provider,
            display_name=model.display_name,
            description=model.description,
            capabilities=model.capabilities,
            default_permissions=model.default_permissions,
            metadata=model.metadata
        )
    except ModelPermissionError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating model: {str(e)}"
        )


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.requires_role(Role.ADMIN))
):
    """
    Delete a model.
    
    Args:
        model_id: Model ID
        model_permission_manager: Model permission manager
        current_user: Current user (must be admin)
        
    Raises:
        HTTPException: If model deletion fails
    """
    # Unregister model
    success = await model_permission_manager.unregister_model(model_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    return {"status": "success"}


@router.post("/permissions")
async def add_model_permission(
    request: ModelPermissionRequest,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    user_manager: UserManager = Depends(get_user_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.requires_role(Role.ADMIN))
):
    """
    Add a model-specific permission to a user.
    
    Args:
        request: Model permission request
        model_permission_manager: Model permission manager
        user_manager: User manager
        current_user: Current user (must be admin)
        
    Raises:
        HTTPException: If permission addition fails
    """
    try:
        # Check if model exists
        model = await model_permission_manager.get_model(request.model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' not found"
            )
        
        # Add permission
        await user_manager.add_model_permission(
            user_id=request.user_id,
            model_id=request.model_id,
            permission=request.permission
        )
        
        return {"status": "success"}
    except UserError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding model permission: {str(e)}"
        )


@router.delete("/permissions")
async def remove_model_permission(
    request: ModelPermissionRequest,
    model_permission_manager: ModelPermissionManager = Depends(get_model_permission_manager),
    user_manager: UserManager = Depends(get_user_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.requires_role(Role.ADMIN))
):
    """
    Remove a model-specific permission from a user.
    
    Args:
        request: Model permission request
        model_permission_manager: Model permission manager
        user_manager: User manager
        current_user: Current user (must be admin)
        
    Raises:
        HTTPException: If permission removal fails
    """
    try:
        # Check if model exists
        model = await model_permission_manager.get_model(request.model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' not found"
            )
        
        # Remove permission
        await user_manager.remove_model_permission(
            user_id=request.user_id,
            model_id=request.model_id,
            permission=request.permission
        )
        
        return {"status": "success"}
    except UserError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error removing model permission: {str(e)}"
        )
