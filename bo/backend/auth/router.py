from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from bo.backend.auth.security import oauth2_scheme
from sqlalchemy.orm import Session
from datetime import timedelta
import logging
from typing import List, Optional

from bo.backend.db.database import get_db
from bo.backend.db.models import User as DBUser
from bo.backend.auth.models import User, UserCreate, UserUpdate, Token
from bo.backend.auth.utils import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    decode_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Setup logging
logger = logging.getLogger("agentor-backoffice.auth")

# Create router
router = APIRouter()

def get_user_by_username(db: Session, username: str) -> Optional[DBUser]:
    """Get a user by username.
    
    Args:
        db: Database session
        username: Username to look up
        
    Returns:
        User if found, None otherwise
    """
    return db.query(DBUser).filter(DBUser.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[DBUser]:
    """Get a user by email.
    
    Args:
        db: Database session
        email: Email to look up
        
    Returns:
        User if found, None otherwise
    """
    return db.query(DBUser).filter(DBUser.email == email).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[DBUser]:
    """Authenticate a user.
    
    Args:
        db: Database session
        username: Username to authenticate
        password: Password to verify
        
    Returns:
        User if authentication successful, None otherwise
    """
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    """Get the current user from the token.
    
    Args:
        db: Database session
        token: JWT token
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token_data = decode_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user_by_username(db, token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login endpoint to get an access token.
    
    Args:
        form_data: Form data with username and password
        db: Database session
        
    Returns:
        Access token
        
    Raises:
        HTTPException: If authentication fails
    """
    logger.info(f"Login attempt for user: {form_data.username}")
    
    # Check if the user exists
    user_exists = get_user_by_username(db, form_data.username)
    if not user_exists:
        logger.warning(f"User not found: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Authenticate the user
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(f"Authentication failed for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Authentication successful for user: {form_data.username}")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id, "role": user.role},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=user.id,
        username=user.username,
        role=user.role
    )

@router.post("/register", response_model=User)
async def register_user(
    user_create: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user.
    
    Args:
        user_create: User creation data
        db: Database session
        
    Returns:
        Created user
        
    Raises:
        HTTPException: If username or email already exists
    """
    # Check if username already exists
    db_user = get_user_by_username(db, user_create.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    db_user = get_user_by_email(db, user_create.email)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_create.password)
    db_user = DBUser(
        username=user_create.username,
        email=user_create.email,
        full_name=user_create.full_name,
        hashed_password=hashed_password,
        is_active=user_create.is_active,
        role=user_create.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return User(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        full_name=db_user.full_name,
        is_active=db_user.is_active,
        role=db_user.role,
        created_at=db_user.created_at,
        updated_at=db_user.updated_at
    )

@router.get("/users/me", response_model=User)
async def read_users_me(
    current_user: DBUser = Depends(get_current_user)
):
    """Get the current user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        Current user
    """
    return User(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        role=current_user.role,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )

@router.put("/users/me", response_model=User)
async def update_user_me(
    user_update: UserUpdate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the current user.
    
    Args:
        user_update: User update data
        current_user: Current user from token
        db: Database session
        
    Returns:
        Updated user
    """
    # Update user fields
    if user_update.email is not None:
        current_user.email = user_update.email
    if user_update.full_name is not None:
        current_user.full_name = user_update.full_name
    if user_update.password is not None:
        current_user.hashed_password = get_password_hash(user_update.password)
    
    # Only admin can change role and active status
    if current_user.role == "admin":
        if user_update.is_active is not None:
            current_user.is_active = user_update.is_active
        if user_update.role is not None:
            current_user.role = user_update.role
    
    db.commit()
    db.refresh(current_user)
    
    return User(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        role=current_user.role,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )
