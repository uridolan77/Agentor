"""Authentication dependencies for FastAPI endpoints."""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from bo.backend.db.database import get_db
from bo.backend.db.models import User
from .security import get_current_user

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency to get the current active user.
    
    Args:
        current_user: Current authenticated user from token
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Dependency to get the current admin user.
    
    Args:
        current_user: Current active user from token
        
    Returns:
        Current admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user