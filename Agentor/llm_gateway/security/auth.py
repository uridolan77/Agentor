"""
Enhanced authentication and authorization for the LLM Gateway.

This module provides JWT-based authentication and role-based access control.
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union

import jwt
from fastapi import Request, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Error raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Error raised when authorization fails."""
    pass


class JWTConfig:
    """Configuration for JWT authentication."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_expiration_minutes: int = 60,
        refresh_token_expiration_days: int = 7,
        issuer: Optional[str] = None,
        audience: Optional[str] = None
    ):
        """
        Initialize JWT configuration.
        
        Args:
            secret_key: Secret key for signing JWTs (generated if not provided)
            algorithm: Algorithm for signing JWTs
            token_expiration_minutes: Expiration time for access tokens in minutes
            refresh_token_expiration_days: Expiration time for refresh tokens in days
            issuer: Issuer claim for JWTs
            audience: Audience claim for JWTs
        """
        self.secret_key = secret_key or os.urandom(32).hex()
        self.algorithm = algorithm
        self.token_expiration_minutes = token_expiration_minutes
        self.refresh_token_expiration_days = refresh_token_expiration_days
        self.issuer = issuer
        self.audience = audience


class Role:
    """Role definitions for RBAC."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    MODEL_ADMIN = "model_admin"
    
    # Define role hierarchies (higher roles include permissions of lower roles)
    HIERARCHIES = {
        ADMIN: [USER, READONLY, MODEL_ADMIN],
        MODEL_ADMIN: [USER],
        USER: [READONLY]
    }


class Permission:
    """Permission definitions for RBAC."""
    # General permissions
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    
    # LLM-specific permissions
    GENERATE = "generate"
    STREAM = "stream"
    USE_TOOLS = "use_tools"
    FINE_TUNE = "fine_tune"
    
    # Model-specific permissions (will be combined with model IDs)
    USE_MODEL = "use_model"  # e.g., "use_model:gpt-4"
    ADMIN_MODEL = "admin_model"  # e.g., "admin_model:gpt-4"


class JWTAuth:
    """JWT authentication and authorization manager."""
    
    def __init__(self, config: Optional[JWTConfig] = None):
        """
        Initialize JWT authentication.
        
        Args:
            config: JWT configuration
        """
        self.config = config or JWTConfig()
        
        # Define role permissions
        self.role_permissions = {
            Role.ADMIN: {
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.ADMIN,
                Permission.GENERATE,
                Permission.STREAM,
                Permission.USE_TOOLS,
                Permission.FINE_TUNE
            },
            Role.MODEL_ADMIN: {
                Permission.READ,
                Permission.WRITE,
                Permission.GENERATE,
                Permission.STREAM,
                Permission.USE_TOOLS,
                Permission.FINE_TUNE
            },
            Role.USER: {
                Permission.READ,
                Permission.GENERATE,
                Permission.STREAM,
                Permission.USE_TOOLS
            },
            Role.READONLY: {
                Permission.READ
            }
        }
        
        # Security scheme for FastAPI
        self.security = HTTPBearer()
    
    def create_access_token(
        self,
        user_id: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        model_permissions: Dict[str, List[str]] = None,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            user_id: User ID
            roles: User roles
            permissions: Additional permissions
            model_permissions: Model-specific permissions
            additional_claims: Additional claims to include in the token
            
        Returns:
            JWT access token
        """
        now = datetime.utcnow()
        expiration = now + timedelta(minutes=self.config.token_expiration_minutes)
        
        # Create payload
        payload = {
            "sub": user_id,
            "iat": now.timestamp(),
            "exp": expiration.timestamp(),
            "type": "access"
        }
        
        # Add issuer and audience if configured
        if self.config.issuer:
            payload["iss"] = self.config.issuer
        if self.config.audience:
            payload["aud"] = self.config.audience
        
        # Add roles and permissions
        if roles:
            payload["roles"] = roles
        if permissions:
            payload["permissions"] = permissions
        if model_permissions:
            payload["model_permissions"] = model_permissions
        
        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Create token
        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        return token
    
    def create_refresh_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """
        Create a JWT refresh token.
        
        Args:
            user_id: User ID
            additional_claims: Additional claims to include in the token
            
        Returns:
            JWT refresh token
        """
        now = datetime.utcnow()
        expiration = now + timedelta(days=self.config.refresh_token_expiration_days)
        
        # Create payload
        payload = {
            "sub": user_id,
            "iat": now.timestamp(),
            "exp": expiration.timestamp(),
            "type": "refresh"
        }
        
        # Add issuer and audience if configured
        if self.config.issuer:
            payload["iss"] = self.config.issuer
        if self.config.audience:
            payload["aud"] = self.config.audience
        
        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Create token
        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        return token
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True
                }
            )
            
            # Verify issuer and audience if configured
            if self.config.issuer and payload.get("iss") != self.config.issuer:
                raise AuthenticationError("Invalid token issuer")
            if self.config.audience and payload.get("aud") != self.config.audience:
                raise AuthenticationError("Invalid token audience")
            
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """
        Refresh an access token using a refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New access token
            
        Raises:
            AuthenticationError: If refresh token is invalid
        """
        try:
            # Decode refresh token
            payload = self.decode_token(refresh_token)
            
            # Verify token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type")
            
            # Create new access token
            user_id = payload["sub"]
            roles = payload.get("roles", [])
            permissions = payload.get("permissions", [])
            model_permissions = payload.get("model_permissions", {})
            
            # Create new access token
            return self.create_access_token(
                user_id=user_id,
                roles=roles,
                permissions=permissions,
                model_permissions=model_permissions
            )
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Error refreshing token: {str(e)}")
    
    def get_user_permissions(self, roles: List[str], additional_permissions: List[str] = None) -> Set[str]:
        """
        Get all permissions for a user based on roles and additional permissions.
        
        Args:
            roles: User roles
            additional_permissions: Additional permissions
            
        Returns:
            Set of permissions
        """
        permissions = set()
        
        # Add permissions from roles
        for role in roles:
            # Add permissions for this role
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
            
            # Add permissions from role hierarchy
            if role in Role.HIERARCHIES:
                for inherited_role in Role.HIERARCHIES[role]:
                    if inherited_role in self.role_permissions:
                        permissions.update(self.role_permissions[inherited_role])
        
        # Add additional permissions
        if additional_permissions:
            permissions.update(additional_permissions)
        
        return permissions
    
    def has_permission(
        self,
        token_payload: Dict[str, Any],
        required_permission: str,
        model_id: Optional[str] = None
    ) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            token_payload: Token payload
            required_permission: Required permission
            model_id: Model ID for model-specific permissions
            
        Returns:
            True if user has permission, False otherwise
        """
        # Get user roles and permissions
        roles = token_payload.get("roles", [])
        additional_permissions = token_payload.get("permissions", [])
        model_permissions = token_payload.get("model_permissions", {})
        
        # Get all permissions for the user
        permissions = self.get_user_permissions(roles, additional_permissions)
        
        # Check if user has the required permission
        if required_permission in permissions or Permission.ADMIN in permissions:
            return True
        
        # Check model-specific permissions
        if model_id and model_id in model_permissions:
            model_specific_permissions = model_permissions[model_id]
            if required_permission in model_specific_permissions:
                return True
            
            # Check for model-specific permission patterns
            if required_permission.startswith(Permission.USE_MODEL) and Permission.USE_MODEL in model_specific_permissions:
                return True
            if required_permission.startswith(Permission.ADMIN_MODEL) and Permission.ADMIN_MODEL in model_specific_permissions:
                return True
        
        return False
    
    def verify_permission(
        self,
        token_payload: Dict[str, Any],
        required_permission: str,
        model_id: Optional[str] = None
    ) -> None:
        """
        Verify that a user has a specific permission.
        
        Args:
            token_payload: Token payload
            required_permission: Required permission
            model_id: Model ID for model-specific permissions
            
        Raises:
            AuthorizationError: If user does not have permission
        """
        if not self.has_permission(token_payload, required_permission, model_id):
            if model_id:
                raise AuthorizationError(f"User does not have permission '{required_permission}' for model '{model_id}'")
            else:
                raise AuthorizationError(f"User does not have permission '{required_permission}'")
    
    async def get_token_from_request(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> str:
        """
        Get token from request.
        
        Args:
            credentials: HTTP Authorization credentials
            
        Returns:
            JWT token
        """
        return credentials.credentials
    
    async def get_current_user(self, token: str = Depends(get_token_from_request)) -> Dict[str, Any]:
        """
        Get current user from token.
        
        Args:
            token: JWT token
            
        Returns:
            User information
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            # Decode token
            payload = self.decode_token(token)
            
            # Verify token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return payload
        except AuthenticationError as e:
            raise HTTPException(
                status_code=401,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    def requires_permission(self, permission: str, model_id: Optional[str] = None):
        """
        Decorator for requiring a specific permission.
        
        Args:
            permission: Required permission
            model_id: Model ID for model-specific permissions
            
        Returns:
            Dependency function
        """
        async def dependency(user: Dict[str, Any] = Depends(self.get_current_user)):
            try:
                self.verify_permission(user, permission, model_id)
                return user
            except AuthorizationError as e:
                raise HTTPException(
                    status_code=403,
                    detail=str(e)
                )
        
        return dependency
    
    def requires_role(self, role: str):
        """
        Decorator for requiring a specific role.
        
        Args:
            role: Required role
            
        Returns:
            Dependency function
        """
        async def dependency(user: Dict[str, Any] = Depends(self.get_current_user)):
            roles = user.get("roles", [])
            if role not in roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"User does not have role '{role}'"
                )
            return user
        
        return dependency


class RBACMiddleware:
    """Role-Based Access Control middleware."""
    
    def __init__(self, app, jwt_auth: JWTAuth):
        """
        Initialize the RBAC middleware.
        
        Args:
            app: The FastAPI application
            jwt_auth: JWT authentication manager
        """
        self.app = app
        self.jwt_auth = jwt_auth
        
        # Define endpoint permissions
        self.endpoint_permissions = {
            "/generate": Permission.GENERATE,
            "/chat/completions": Permission.GENERATE,
            "/models": Permission.READ,
            "/fine-tunes": Permission.FINE_TUNE,
            "/admin": Permission.ADMIN
        }
    
    async def __call__(self, scope, receive, send):
        """
        Process an incoming request.
        
        Args:
            scope: The ASGI scope
            receive: The ASGI receive function
            send: The ASGI send function
        """
        if scope["type"] == "http":
            # Get the path
            path = scope["path"]
            
            # Check if path requires permission
            required_permission = None
            for endpoint, permission in self.endpoint_permissions.items():
                if path.startswith(endpoint):
                    required_permission = permission
                    break
            
            # If path requires permission, check authorization
            if required_permission:
                # Get the authorization header
                headers = dict(scope["headers"])
                auth_header = headers.get(b"authorization", b"").decode()
                
                if not auth_header.startswith("Bearer "):
                    # No token provided, return 401
                    return await self._unauthorized_response(scope, receive, send)
                
                # Extract token
                token = auth_header.split(" ")[1]
                
                try:
                    # Decode token
                    payload = self.jwt_auth.decode_token(token)
                    
                    # Get model ID from query parameters or request body
                    model_id = None
                    # This is a simplified implementation - in a real app, you would
                    # parse query parameters or request body to get the model ID
                    
                    # Check permission
                    if not self.jwt_auth.has_permission(payload, required_permission, model_id):
                        # User does not have permission, return 403
                        return await self._forbidden_response(scope, receive, send)
                except AuthenticationError:
                    # Invalid token, return 401
                    return await self._unauthorized_response(scope, receive, send)
                except Exception as e:
                    # Other error, log and return 500
                    logger.error(f"Error in RBAC middleware: {e}")
                    return await self._server_error_response(scope, receive, send)
        
        # Process the request
        await self.app(scope, receive, send)
    
    async def _unauthorized_response(self, scope, receive, send):
        """Send a 401 Unauthorized response."""
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer")
            ]
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps({"detail": "Not authenticated"}).encode()
        })
    
    async def _forbidden_response(self, scope, receive, send):
        """Send a 403 Forbidden response."""
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json")
            ]
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps({"detail": "Not authorized"}).encode()
        })
    
    async def _server_error_response(self, scope, receive, send):
        """Send a 500 Internal Server Error response."""
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json")
            ]
        })
        await send({
            "type": "http.response.body",
            "body": json.dumps({"detail": "Internal server error"}).encode()
        })
