from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
import re

class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True

    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must be alphanumeric with optional underscores and hyphens')
        return v

class UserCreate(UserBase):
    """User creation model."""
    password: str
    role: str = "user"  # Default role is "user"

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v

    @validator('role')
    def valid_role(cls, v):
        valid_roles = ["admin", "manager", "user"]
        if v not in valid_roles:
            raise ValueError(f'Role must be one of {valid_roles}')
        return v

class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[str] = None

    @validator('password')
    def password_strength(cls, v):
        if v is None:
            return v
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v

    @validator('role')
    def valid_role(cls, v):
        if v is None:
            return v
        valid_roles = ["admin", "manager", "user"]
        if v not in valid_roles:
            raise ValueError(f'Role must be one of {valid_roles}')
        return v

class User(UserBase):
    """User model with ID."""
    id: int
    role: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str
    expires_in: int
    user_id: int
    username: str
    role: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None

class RolePermission(BaseModel):
    """Role permission model."""
    role: str
    permissions: List[str]
