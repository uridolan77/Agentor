from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
import logging

from .models import TokenData

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger("agentor-backoffice.auth")

# Get JWT settings from environment variables or use defaults
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_TO_A_RANDOM_SECRET_IN_PRODUCTION")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password
        
    Returns:
        True if the password matches the hash, False otherwise
    """
    try:
        logger.info("Verifying password")
        result = pwd_context.verify(plain_password, hashed_password)
        logger.info(f"Password verification result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Hash a password.
    
    Args:
        password: The plain text password
        
    Returns:
        The hashed password
    """
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: The data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        The encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {e}")
        raise

def decode_token(token: str) -> Optional[TokenData]:
    """Decode a JWT token.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        The decoded token data or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("user_id")
        role = payload.get("role")
        exp = datetime.fromtimestamp(payload.get("exp"))
        
        if username is None:
            return None
            
        return TokenData(
            username=username,
            user_id=user_id,
            role=role,
            exp=exp
        )
    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        return None

def get_permissions_for_role(role: str) -> list:
    """Get permissions for a role.
    
    Args:
        role: The role to get permissions for
        
    Returns:
        List of permissions for the role
    """
    # Define role-based permissions
    permissions = {
        "admin": [
            "user:create", "user:read", "user:update", "user:delete",
            "agent:create", "agent:read", "agent:update", "agent:delete",
            "tool:create", "tool:read", "tool:update", "tool:delete",
            "workflow:create", "workflow:read", "workflow:update", "workflow:delete",
            "llm:create", "llm:read", "llm:update", "llm:delete",
            "system:configure"
        ],
        "manager": [
            "user:read", "user:update",
            "agent:create", "agent:read", "agent:update", "agent:delete",
            "tool:create", "tool:read", "tool:update", "tool:delete",
            "workflow:create", "workflow:read", "workflow:update", "workflow:delete",
            "llm:read", "llm:update"
        ],
        "user": [
            "agent:read", "agent:execute",
            "tool:read", "tool:execute",
            "workflow:read", "workflow:execute",
            "llm:read"
        ]
    }
    
    return permissions.get(role, [])

def has_permission(user_role: str, required_permission: str) -> bool:
    """Check if a role has a specific permission.
    
    Args:
        user_role: The user's role
        required_permission: The permission to check
        
    Returns:
        True if the role has the permission, False otherwise
    """
    permissions = get_permissions_for_role(user_role)
    
    # Check for direct permission
    if required_permission in permissions:
        return True
        
    # Check for wildcard permissions (e.g., "agent:*")
    resource = required_permission.split(":")[0]
    wildcard_permission = f"{resource}:*"
    
    return wildcard_permission in permissions
