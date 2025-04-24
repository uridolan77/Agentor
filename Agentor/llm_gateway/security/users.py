"""
User management for the LLM Gateway.

This module provides user management functionality, including:
- User registration and authentication
- Role and permission management
- API key management
"""

import os
import time
import logging
import secrets
import hashlib
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union
import asyncio

from passlib.hash import pbkdf2_sha256

from agentor.llm_gateway.security.auth import JWTAuth, Role, Permission, AuthenticationError

logger = logging.getLogger(__name__)


class UserError(Exception):
    """Error raised for user-related operations."""
    pass


class User:
    """User model."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        password_hash: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        model_permissions: Dict[str, List[str]] = None,
        api_keys: List[str] = None,
        metadata: Dict[str, Any] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None
    ):
        """
        Initialize a user.
        
        Args:
            user_id: User ID
            username: Username
            email: Email address
            password_hash: Hashed password
            roles: User roles
            permissions: Additional permissions
            model_permissions: Model-specific permissions
            api_keys: API keys
            metadata: Additional metadata
            created_at: Creation timestamp
            updated_at: Last update timestamp
        """
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.roles = roles or [Role.USER]
        self.permissions = permissions or []
        self.model_permissions = model_permissions or {}
        self.api_keys = api_keys or []
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Convert user to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            User as dictionary
        """
        result = {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "model_permissions": self.model_permissions,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        if include_sensitive:
            result["password_hash"] = self.password_hash
            result["api_keys"] = self.api_keys
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Create user from dictionary.
        
        Args:
            data: User data
            
        Returns:
            User instance
        """
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            roles=data.get("roles"),
            permissions=data.get("permissions"),
            model_permissions=data.get("model_permissions"),
            api_keys=data.get("api_keys"),
            metadata=data.get("metadata"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at")
        )
    
    def verify_password(self, password: str) -> bool:
        """
        Verify password.
        
        Args:
            password: Password to verify
            
        Returns:
            True if password is correct, False otherwise
        """
        return pbkdf2_sha256.verify(password, self.password_hash)
    
    def has_role(self, role: str) -> bool:
        """
        Check if user has a specific role.
        
        Args:
            role: Role to check
            
        Returns:
            True if user has role, False otherwise
        """
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        return permission in self.permissions
    
    def has_model_permission(self, model_id: str, permission: str) -> bool:
        """
        Check if user has a specific permission for a model.
        
        Args:
            model_id: Model ID
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        if model_id in self.model_permissions:
            return permission in self.model_permissions[model_id]
        return False
    
    def add_role(self, role: str) -> None:
        """
        Add a role to the user.
        
        Args:
            role: Role to add
        """
        if role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.utcnow().isoformat()
    
    def remove_role(self, role: str) -> None:
        """
        Remove a role from the user.
        
        Args:
            role: Role to remove
        """
        if role in self.roles:
            self.roles.remove(role)
            self.updated_at = datetime.utcnow().isoformat()
    
    def add_permission(self, permission: str) -> None:
        """
        Add a permission to the user.
        
        Args:
            permission: Permission to add
        """
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.updated_at = datetime.utcnow().isoformat()
    
    def remove_permission(self, permission: str) -> None:
        """
        Remove a permission from the user.
        
        Args:
            permission: Permission to remove
        """
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.updated_at = datetime.utcnow().isoformat()
    
    def add_model_permission(self, model_id: str, permission: str) -> None:
        """
        Add a model-specific permission to the user.
        
        Args:
            model_id: Model ID
            permission: Permission to add
        """
        if model_id not in self.model_permissions:
            self.model_permissions[model_id] = []
        
        if permission not in self.model_permissions[model_id]:
            self.model_permissions[model_id].append(permission)
            self.updated_at = datetime.utcnow().isoformat()
    
    def remove_model_permission(self, model_id: str, permission: str) -> None:
        """
        Remove a model-specific permission from the user.
        
        Args:
            model_id: Model ID
            permission: Permission to remove
        """
        if model_id in self.model_permissions and permission in self.model_permissions[model_id]:
            self.model_permissions[model_id].remove(permission)
            if not self.model_permissions[model_id]:
                del self.model_permissions[model_id]
            self.updated_at = datetime.utcnow().isoformat()
    
    def generate_api_key(self) -> str:
        """
        Generate a new API key for the user.
        
        Returns:
            New API key
        """
        # Generate a secure API key
        api_key = f"sk-{secrets.token_hex(24)}"
        
        # Add to user's API keys
        self.api_keys.append(api_key)
        self.updated_at = datetime.utcnow().isoformat()
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if API key was revoked, False if not found
        """
        if api_key in self.api_keys:
            self.api_keys.remove(api_key)
            self.updated_at = datetime.utcnow().isoformat()
            return True
        return False
    
    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify an API key.
        
        Args:
            api_key: API key to verify
            
        Returns:
            True if API key is valid, False otherwise
        """
        return api_key in self.api_keys


class UserManager:
    """User management for the LLM Gateway."""
    
    def __init__(self, jwt_auth: JWTAuth):
        """
        Initialize user manager.
        
        Args:
            jwt_auth: JWT authentication manager
        """
        self.jwt_auth = jwt_auth
        self.users: Dict[str, User] = {}
        self.username_to_id: Dict[str, str] = {}
        self.email_to_id: Dict[str, str] = {}
        self.api_key_to_id: Dict[str, str] = {}
        self.lock = asyncio.Lock()
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        model_permissions: Dict[str, List[str]] = None,
        metadata: Dict[str, Any] = None
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            roles: User roles
            permissions: Additional permissions
            model_permissions: Model-specific permissions
            metadata: Additional metadata
            
        Returns:
            Created user
            
        Raises:
            UserError: If username or email already exists
        """
        async with self.lock:
            # Check if username or email already exists
            if username in self.username_to_id:
                raise UserError(f"Username '{username}' already exists")
            if email in self.email_to_id:
                raise UserError(f"Email '{email}' already exists")
            
            # Generate user ID
            user_id = f"user_{int(time.time())}_{secrets.token_hex(8)}"
            
            # Hash password
            password_hash = pbkdf2_sha256.hash(password)
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles,
                permissions=permissions,
                model_permissions=model_permissions,
                metadata=metadata
            )
            
            # Generate initial API key
            api_key = user.generate_api_key()
            
            # Store user
            self.users[user_id] = user
            self.username_to_id[username] = user_id
            self.email_to_id[email] = user_id
            self.api_key_to_id[api_key] = user_id
            
            logger.info(f"Created user {user_id} with username '{username}'")
            
            return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User if found, None otherwise
        """
        async with self.lock:
            return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by username.
        
        Args:
            username: Username
            
        Returns:
            User if found, None otherwise
        """
        async with self.lock:
            user_id = self.username_to_id.get(username)
            if user_id:
                return self.users.get(user_id)
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.
        
        Args:
            email: Email address
            
        Returns:
            User if found, None otherwise
        """
        async with self.lock:
            user_id = self.email_to_id.get(email)
            if user_id:
                return self.users.get(user_id)
            return None
    
    async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """
        Get a user by API key.
        
        Args:
            api_key: API key
            
        Returns:
            User if found, None otherwise
        """
        async with self.lock:
            user_id = self.api_key_to_id.get(api_key)
            if user_id:
                return self.users.get(user_id)
            return None
    
    async def update_user(self, user: User) -> None:
        """
        Update a user.
        
        Args:
            user: User to update
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            if user.user_id not in self.users:
                raise UserError(f"User '{user.user_id}' does not exist")
            
            # Update user
            user.updated_at = datetime.utcnow().isoformat()
            self.users[user.user_id] = user
            
            # Update mappings
            self.username_to_id[user.username] = user.user_id
            self.email_to_id[user.email] = user.user_id
            
            # Update API key mappings
            for api_key in user.api_keys:
                self.api_key_to_id[api_key] = user.user_id
            
            logger.info(f"Updated user {user.user_id}")
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user was deleted, False if not found
        """
        async with self.lock:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            
            # Remove from mappings
            del self.username_to_id[user.username]
            del self.email_to_id[user.email]
            
            # Remove API key mappings
            for api_key in user.api_keys:
                if api_key in self.api_key_to_id:
                    del self.api_key_to_id[api_key]
            
            # Remove user
            del self.users[user_id]
            
            logger.info(f"Deleted user {user_id}")
            
            return True
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User if authentication successful, None otherwise
        """
        # Get user by username
        user = await self.get_user_by_username(username)
        if not user:
            return None
        
        # Verify password
        if not user.verify_password(password):
            return None
        
        return user
    
    async def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate a user using an API key.
        
        Args:
            api_key: API key
            
        Returns:
            User if authentication successful, None otherwise
        """
        return await self.get_user_by_api_key(api_key)
    
    async def create_tokens(self, user: User) -> Dict[str, str]:
        """
        Create access and refresh tokens for a user.
        
        Args:
            user: User
            
        Returns:
            Dictionary with access and refresh tokens
        """
        # Create access token
        access_token = self.jwt_auth.create_access_token(
            user_id=user.user_id,
            roles=user.roles,
            permissions=user.permissions,
            model_permissions=user.model_permissions,
            additional_claims={
                "username": user.username,
                "email": user.email
            }
        )
        
        # Create refresh token
        refresh_token = self.jwt_auth.create_refresh_token(
            user_id=user.user_id,
            additional_claims={
                "username": user.username,
                "roles": user.roles,
                "permissions": user.permissions,
                "model_permissions": user.model_permissions
            }
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    async def refresh_tokens(self, refresh_token: str) -> Dict[str, str]:
        """
        Refresh tokens.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            Dictionary with new access and refresh tokens
            
        Raises:
            AuthenticationError: If refresh token is invalid
        """
        try:
            # Decode refresh token
            payload = self.jwt_auth.decode_token(refresh_token)
            
            # Verify token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid token type")
            
            # Get user
            user_id = payload["sub"]
            user = await self.get_user(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Create new tokens
            return await self.create_tokens(user)
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Error refreshing tokens: {str(e)}")
    
    async def add_role(self, user_id: str, role: str) -> None:
        """
        Add a role to a user.
        
        Args:
            user_id: User ID
            role: Role to add
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            user.add_role(role)
            await self.update_user(user)
    
    async def remove_role(self, user_id: str, role: str) -> None:
        """
        Remove a role from a user.
        
        Args:
            user_id: User ID
            role: Role to remove
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            user.remove_role(role)
            await self.update_user(user)
    
    async def add_permission(self, user_id: str, permission: str) -> None:
        """
        Add a permission to a user.
        
        Args:
            user_id: User ID
            permission: Permission to add
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            user.add_permission(permission)
            await self.update_user(user)
    
    async def remove_permission(self, user_id: str, permission: str) -> None:
        """
        Remove a permission from a user.
        
        Args:
            user_id: User ID
            permission: Permission to remove
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            user.remove_permission(permission)
            await self.update_user(user)
    
    async def add_model_permission(self, user_id: str, model_id: str, permission: str) -> None:
        """
        Add a model-specific permission to a user.
        
        Args:
            user_id: User ID
            model_id: Model ID
            permission: Permission to add
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            user.add_model_permission(model_id, permission)
            await self.update_user(user)
    
    async def remove_model_permission(self, user_id: str, model_id: str, permission: str) -> None:
        """
        Remove a model-specific permission from a user.
        
        Args:
            user_id: User ID
            model_id: Model ID
            permission: Permission to remove
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            user.remove_model_permission(model_id, permission)
            await self.update_user(user)
    
    async def generate_api_key(self, user_id: str) -> str:
        """
        Generate a new API key for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            New API key
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            # Generate API key
            api_key = user.generate_api_key()
            
            # Update API key mapping
            self.api_key_to_id[api_key] = user_id
            
            # Update user
            await self.update_user(user)
            
            return api_key
    
    async def revoke_api_key(self, user_id: str, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            user_id: User ID
            api_key: API key to revoke
            
        Returns:
            True if API key was revoked, False if not found
            
        Raises:
            UserError: If user does not exist
        """
        async with self.lock:
            user = await self.get_user(user_id)
            if not user:
                raise UserError(f"User '{user_id}' does not exist")
            
            # Revoke API key
            if user.revoke_api_key(api_key):
                # Remove API key mapping
                if api_key in self.api_key_to_id:
                    del self.api_key_to_id[api_key]
                
                # Update user
                await self.update_user(user)
                
                return True
            
            return False
    
    async def load_users(self, users_data: List[Dict[str, Any]]) -> None:
        """
        Load users from data.
        
        Args:
            users_data: User data
        """
        async with self.lock:
            for user_data in users_data:
                user = User.from_dict(user_data)
                
                # Store user
                self.users[user.user_id] = user
                self.username_to_id[user.username] = user.user_id
                self.email_to_id[user.email] = user.user_id
                
                # Store API key mappings
                for api_key in user.api_keys:
                    self.api_key_to_id[api_key] = user.user_id
            
            logger.info(f"Loaded {len(users_data)} users")
    
    async def save_users(self) -> List[Dict[str, Any]]:
        """
        Save users to data.
        
        Returns:
            User data
        """
        async with self.lock:
            users_data = []
            
            for user in self.users.values():
                users_data.append(user.to_dict(include_sensitive=True))
            
            return users_data
    
    async def create_default_users(self) -> None:
        """Create default users if no users exist."""
        async with self.lock:
            if not self.users:
                # Create admin user
                await self.create_user(
                    username="admin",
                    email="admin@example.com",
                    password="admin",  # This should be changed in production
                    roles=[Role.ADMIN]
                )
                
                # Create regular user
                await self.create_user(
                    username="user",
                    email="user@example.com",
                    password="user",  # This should be changed in production
                    roles=[Role.USER]
                )
                
                logger.info("Created default users")
