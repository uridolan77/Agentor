"""Authentication module for the Agentor BackOffice."""

from .models import User, UserCreate, UserUpdate, Token, TokenData, RolePermission
from .utils import (
    verify_password, 
    get_password_hash, 
    create_access_token, 
    decode_token,
    get_permissions_for_role,
    has_permission
)
from .router import router
from .dependencies import get_current_active_user, get_current_admin_user
