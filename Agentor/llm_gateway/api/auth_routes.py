"""
Authentication and user management routes for the LLM Gateway.
"""

from fastapi import APIRouter, Depends, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from typing import Dict, Any, Optional, List

from agentor.llm_gateway.security.auth import JWTAuth, Role, Permission, AuthenticationError, AuthorizationError
from agentor.llm_gateway.security.users import UserManager, UserError


# Models for request and response
class LoginRequest(BaseModel):
    """Login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(..., description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class RefreshRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str = Field(..., description="Refresh token")


class UserCreateRequest(BaseModel):
    """User creation request."""
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    roles: Optional[List[str]] = Field(None, description="User roles")


class UserResponse(BaseModel):
    """User response."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    roles: List[str] = Field(..., description="User roles")
    permissions: List[str] = Field(..., description="User permissions")
    model_permissions: Dict[str, List[str]] = Field(..., description="Model-specific permissions")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class ApiKeyResponse(BaseModel):
    """API key response."""
    api_key: str = Field(..., description="API key")


# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])


# Dependency to get JWT auth from app state
def get_jwt_auth(request: Request) -> JWTAuth:
    """Get JWT auth from app state."""
    return request.app.state.security_components["jwt_auth"]


# Dependency to get user manager from app state
def get_user_manager(request: Request) -> UserManager:
    """Get user manager from app state."""
    return request.app.state.security_components["user_manager"]


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    user_manager: UserManager = Depends(get_user_manager),
    jwt_auth: JWTAuth = Depends(get_jwt_auth)
):
    """
    Login with username and password.
    
    Args:
        request: Login request
        user_manager: User manager
        jwt_auth: JWT auth
        
    Returns:
        Token response
        
    Raises:
        HTTPException: If login fails
    """
    # Authenticate user
    user = await user_manager.authenticate(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    
    # Create tokens
    tokens = await user_manager.create_tokens(user)
    
    # Add expiration time
    tokens["expires_in"] = jwt_auth.config.token_expiration_minutes * 60
    
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    user_manager: UserManager = Depends(get_user_manager),
    jwt_auth: JWTAuth = Depends(get_jwt_auth)
):
    """
    Refresh access token.
    
    Args:
        request: Refresh token request
        user_manager: User manager
        jwt_auth: JWT auth
        
    Returns:
        Token response
        
    Raises:
        HTTPException: If refresh fails
    """
    try:
        # Refresh tokens
        tokens = await user_manager.refresh_tokens(request.refresh_token)
        
        # Add expiration time
        tokens["expires_in"] = jwt_auth.config.token_expiration_minutes * 60
        
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing token: {str(e)}"
        )


@router.post("/users", response_model=UserResponse)
async def create_user(
    request: UserCreateRequest,
    user_manager: UserManager = Depends(get_user_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.requires_role(Role.ADMIN))
):
    """
    Create a new user.
    
    Args:
        request: User creation request
        user_manager: User manager
        current_user: Current user (must be admin)
        
    Returns:
        Created user
        
    Raises:
        HTTPException: If user creation fails
    """
    try:
        # Create user
        user = await user_manager.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            roles=request.roles
        )
        
        # Convert to response
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            roles=user.roles,
            permissions=user.permissions,
            model_permissions=user.model_permissions,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    except UserError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating user: {str(e)}"
        )


@router.get("/users/me", response_model=UserResponse)
async def get_current_user(
    user_manager: UserManager = Depends(get_user_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.get_current_user)
):
    """
    Get current user.
    
    Args:
        user_manager: User manager
        current_user: Current user
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If user not found
    """
    # Get user
    user = await user_manager.get_user(current_user["sub"])
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    # Convert to response
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        email=user.email,
        roles=user.roles,
        permissions=user.permissions,
        model_permissions=user.model_permissions,
        created_at=user.created_at,
        updated_at=user.updated_at
    )


@router.post("/users/me/api-keys", response_model=ApiKeyResponse)
async def create_api_key(
    user_manager: UserManager = Depends(get_user_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.get_current_user)
):
    """
    Create a new API key for the current user.
    
    Args:
        user_manager: User manager
        current_user: Current user
        
    Returns:
        Created API key
        
    Raises:
        HTTPException: If API key creation fails
    """
    try:
        # Generate API key
        api_key = await user_manager.generate_api_key(current_user["sub"])
        
        return ApiKeyResponse(api_key=api_key)
    except UserError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating API key: {str(e)}"
        )


@router.delete("/users/me/api-keys/{api_key}")
async def revoke_api_key(
    api_key: str,
    user_manager: UserManager = Depends(get_user_manager),
    current_user: Dict[str, Any] = Depends(lambda jwt_auth: jwt_auth.get_current_user)
):
    """
    Revoke an API key.
    
    Args:
        api_key: API key to revoke
        user_manager: User manager
        current_user: Current user
        
    Raises:
        HTTPException: If API key revocation fails
    """
    try:
        # Revoke API key
        success = await user_manager.revoke_api_key(current_user["sub"], api_key)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="API key not found"
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
            detail=f"Error revoking API key: {str(e)}"
        )
