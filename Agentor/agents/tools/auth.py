"""
Authentication and authorization for tools in the Agentor framework.

This module provides classes and functions to secure tool access through:
- Authentication mechanisms for tool APIs
- Role-based access control
- Per-tool permission settings
- Audit logging for tool usage
"""

from typing import Dict, Any, List, Optional, Set, Union, Callable
import logging
import datetime
import json
import asyncio
import time
import uuid
from enum import Enum
from pydantic import BaseModel, Field

from agentor.core.interfaces.tool import ITool, ToolResult

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for tool authorization."""
    
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    GUEST = "guest"


class User(BaseModel):
    """User information for authentication and authorization."""
    
    id: str
    username: str
    roles: List[UserRole] = Field(default_factory=lambda: [UserRole.GUEST])
    api_keys: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolPermission(BaseModel):
    """Permission settings for a tool."""
    
    tool_name: str
    allowed_roles: List[UserRole] = Field(default_factory=lambda: [UserRole.ADMIN])
    allowed_users: List[str] = Field(default_factory=list)
    requires_authentication: bool = True
    rate_limit: Optional[int] = None  # Requests per minute
    
    def can_access(self, user: User) -> bool:
        """Check if a user can access this tool.
        
        Args:
            user: The user to check
            
        Returns:
            True if the user has access, False otherwise
        """
        # Check if user is explicitly allowed
        if user.id in self.allowed_users or user.username in self.allowed_users:
            return True
            
        # Check if user has an allowed role
        for role in user.roles:
            if role in self.allowed_roles:
                return True
                
        return False


class AuditLogEntry(BaseModel):
    """An audit log entry for tool usage."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    tool_name: str
    tool_version: Optional[str] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    action: str
    success: bool
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None


class ToolAuditLogger:
    """Audit logger for tool usage."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the audit logger.
        
        Args:
            log_file: Optional path to a log file
        """
        self.log_file = log_file
        self.in_memory_logs: List[AuditLogEntry] = []
        self.max_in_memory_logs = 1000  # Maximum number of logs to keep in memory
        
    async def log(self, entry: AuditLogEntry) -> None:
        """Log a tool usage event.
        
        Args:
            entry: The audit log entry
        """
        # Add to in-memory logs
        self.in_memory_logs.append(entry)
        
        # Truncate in-memory logs if needed
        if len(self.in_memory_logs) > self.max_in_memory_logs:
            self.in_memory_logs = self.in_memory_logs[-self.max_in_memory_logs:]
            
        # Write to log file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(entry.dict()) + '\n')
            except Exception as e:
                logger.error(f"Failed to write to audit log file: {str(e)}")
                
        # Log to the standard logger as well
        log_msg = (
            f"TOOL_AUDIT: {entry.tool_name} v{entry.tool_version or 'unknown'} "
            f"by {entry.username or entry.user_id or 'anonymous'} "
            f"action={entry.action} success={entry.success}"
        )
        
        if entry.success:
            logger.info(log_msg)
        else:
            logger.warning(f"{log_msg} error={entry.error_message}")
    
    def get_logs(
        self, 
        tool_name: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Get audit logs with optional filtering.
        
        Args:
            tool_name: Filter by tool name
            user_id: Filter by user ID
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            success: Filter by success status
            limit: Maximum number of logs to return
            
        Returns:
            List of matching audit log entries
        """
        filtered_logs = self.in_memory_logs
        
        if tool_name is not None:
            filtered_logs = [log for log in filtered_logs if log.tool_name == tool_name]
            
        if user_id is not None:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            
        if start_time is not None:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
            
        if end_time is not None:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
            
        if success is not None:
            filtered_logs = [log for log in filtered_logs if log.success == success]
            
        # Sort by timestamp (newest first) and limit results
        filtered_logs.sort(key=lambda log: log.timestamp, reverse=True)
        return filtered_logs[:limit]


class RateLimiter:
    """Rate limiter for tool usage."""
    
    def __init__(self):
        """Initialize the rate limiter."""
        # Map of {user_id: {tool_name: [timestamps]}}
        self.request_timestamps: Dict[str, Dict[str, List[float]]] = {}
        
    def is_rate_limited(
        self, 
        user_id: str, 
        tool_name: str, 
        limit: int, 
        window_seconds: int = 60
    ) -> bool:
        """Check if a user is rate limited for a tool.
        
        Args:
            user_id: The ID of the user
            tool_name: The name of the tool
            limit: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
            
        Returns:
            True if rate limited, False otherwise
        """
        now = time.time()
        cutoff = now - window_seconds
        
        # Initialize data structures if needed
        if user_id not in self.request_timestamps:
            self.request_timestamps[user_id] = {}
            
        if tool_name not in self.request_timestamps[user_id]:
            self.request_timestamps[user_id][tool_name] = []
            
        # Clean up old timestamps
        timestamps = [ts for ts in self.request_timestamps[user_id][tool_name] if ts > cutoff]
        self.request_timestamps[user_id][tool_name] = timestamps
        
        # Check if rate limited
        return len(timestamps) >= limit
        
    def record_request(self, user_id: str, tool_name: str) -> None:
        """Record a request for rate limiting.
        
        Args:
            user_id: The ID of the user
            tool_name: The name of the tool
        """
        now = time.time()
        
        # Initialize data structures if needed
        if user_id not in self.request_timestamps:
            self.request_timestamps[user_id] = {}
            
        if tool_name not in self.request_timestamps[user_id]:
            self.request_timestamps[user_id][tool_name] = []
            
        # Add the timestamp
        self.request_timestamps[user_id][tool_name].append(now)


class AuthenticatedTool:
    """A wrapper for a tool that adds authentication and authorization."""
    
    def __init__(
        self,
        tool: ITool,
        permissions: ToolPermission,
        audit_logger: Optional[ToolAuditLogger] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """Initialize the authenticated tool.
        
        Args:
            tool: The tool to wrap
            permissions: Permission settings for the tool
            audit_logger: Optional audit logger
            rate_limiter: Optional rate limiter
        """
        self.tool = tool
        self.permissions = permissions
        self.audit_logger = audit_logger
        self.rate_limiter = rate_limiter
        
    async def run(
        self,
        user: User,
        request_metadata: Dict[str, Any] = None,
        **kwargs
    ) -> ToolResult:
        """Run the tool with authentication and authorization.
        
        Args:
            user: The user running the tool
            request_metadata: Optional metadata about the request
            **kwargs: Parameters for the tool
            
        Returns:
            The result of running the tool
        """
        request_metadata = request_metadata or {}
        start_time = time.time()
        error_message = None
        
        # Check if authentication is required
        if self.permissions.requires_authentication and user is None:
            error_message = "Authentication required"
            result = ToolResult(success=False, error=error_message)
            
            # Log the failed authentication
            if self.audit_logger:
                await self.audit_logger.log(AuditLogEntry(
                    tool_name=self.tool.name,
                    tool_version=getattr(self.tool, "version", None),
                    action="run",
                    success=False,
                    input_data={},  # Don't log input data for security
                    error_message=error_message,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    client_ip=request_metadata.get("client_ip"),
                    user_agent=request_metadata.get("user_agent")
                ))
                
            return result
            
        # Check permissions
        if not self.permissions.can_access(user):
            error_message = "Access denied"
            result = ToolResult(success=False, error=error_message)
            
            # Log the unauthorized access
            if self.audit_logger:
                await self.audit_logger.log(AuditLogEntry(
                    tool_name=self.tool.name,
                    tool_version=getattr(self.tool, "version", None),
                    user_id=user.id,
                    username=user.username,
                    action="run",
                    success=False,
                    input_data={},  # Don't log input data for security
                    error_message=error_message,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    client_ip=request_metadata.get("client_ip"),
                    user_agent=request_metadata.get("user_agent")
                ))
                
            return result
            
        # Check rate limit
        if (
            self.rate_limiter and 
            self.permissions.rate_limit and 
            self.rate_limiter.is_rate_limited(
                user.id, 
                self.tool.name, 
                self.permissions.rate_limit
            )
        ):
            error_message = f"Rate limit exceeded: {self.permissions.rate_limit} requests per minute"
            result = ToolResult(success=False, error=error_message)
            
            # Log the rate limit exceeded
            if self.audit_logger:
                await self.audit_logger.log(AuditLogEntry(
                    tool_name=self.tool.name,
                    tool_version=getattr(self.tool, "version", None),
                    user_id=user.id,
                    username=user.username,
                    action="run",
                    success=False,
                    input_data={},  # Don't log input data for security
                    error_message=error_message,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    client_ip=request_metadata.get("client_ip"),
                    user_agent=request_metadata.get("user_agent")
                ))
                
            return result
            
        # Record the request for rate limiting
        if self.rate_limiter:
            self.rate_limiter.record_request(user.id, self.tool.name)
            
        # Run the tool
        try:
            result = await self.tool.run(**kwargs)
            
            # Log successful execution
            if self.audit_logger:
                await self.audit_logger.log(AuditLogEntry(
                    tool_name=self.tool.name,
                    tool_version=getattr(self.tool, "version", None),
                    user_id=user.id,
                    username=user.username,
                    action="run",
                    success=True,
                    input_data=kwargs,
                    output_data=result.data if result.success else None,
                    error_message=result.error if not result.success else None,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    client_ip=request_metadata.get("client_ip"),
                    user_agent=request_metadata.get("user_agent")
                ))
                
            return result
        except Exception as e:
            error_message = f"Tool execution error: {str(e)}"
            result = ToolResult(success=False, error=error_message)
            
            # Log the error
            if self.audit_logger:
                await self.audit_logger.log(AuditLogEntry(
                    tool_name=self.tool.name,
                    tool_version=getattr(self.tool, "version", None),
                    user_id=user.id,
                    username=user.username,
                    action="run",
                    success=False,
                    input_data=kwargs,
                    error_message=error_message,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    client_ip=request_metadata.get("client_ip"),
                    user_agent=request_metadata.get("user_agent")
                ))
                
            return result


class AuthenticatedToolRegistry:
    """A registry for authenticated tools."""
    
    def __init__(self):
        """Initialize the authenticated tool registry."""
        self.tools: Dict[str, AuthenticatedTool] = {}
        self.default_permissions = ToolPermission(
            tool_name="*",
            allowed_roles=[UserRole.ADMIN],
            requires_authentication=True,
            rate_limit=100  # Default rate limit: 100 requests per minute
        )
        self.audit_logger = ToolAuditLogger()
        self.rate_limiter = RateLimiter()
        self.users: Dict[str, User] = {}
        self.api_key_to_user: Dict[str, str] = {}
        
    def register_tool(self, tool: ITool, permissions: Optional[ToolPermission] = None) -> None:
        """Register a tool with permissions.
        
        Args:
            tool: The tool to register
            permissions: Optional custom permissions, or None to use defaults
        """
        if permissions is None:
            permissions = ToolPermission(
                tool_name=tool.name,
                allowed_roles=self.default_permissions.allowed_roles,
                requires_authentication=self.default_permissions.requires_authentication,
                rate_limit=self.default_permissions.rate_limit
            )
            
        self.tools[tool.name] = AuthenticatedTool(
            tool=tool,
            permissions=permissions,
            audit_logger=self.audit_logger,
            rate_limiter=self.rate_limiter
        )
        
    def register_user(self, user: User) -> None:
        """Register a user.
        
        Args:
            user: The user to register
        """
        self.users[user.id] = user
        
        # Create mappings for API keys
        for api_key in user.api_keys:
            self.api_key_to_user[api_key] = user.id
            
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get a user by API key.
        
        Args:
            api_key: The API key
            
        Returns:
            The user, or None if not found
        """
        user_id = self.api_key_to_user.get(api_key)
        if user_id:
            return self.users.get(user_id)
        return None
        
    def get_tool(self, tool_name: str) -> Optional[AuthenticatedTool]:
        """Get a tool by name.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            The authenticated tool, or None if not found
        """
        return self.tools.get(tool_name)
        
    def get_tools(self, user: Optional[User] = None) -> Dict[str, AuthenticatedTool]:
        """Get all tools accessible to a user.
        
        Args:
            user: Optional user to filter by access permission
            
        Returns:
            Dictionary of tool names to authenticated tools
        """
        if user is None:
            return {
                name: tool for name, tool in self.tools.items() 
                if not tool.permissions.requires_authentication
            }
            
        return {
            name: tool for name, tool in self.tools.items() 
            if tool.permissions.can_access(user)
        }
        
    def get_audit_logs(self, **filters) -> List[AuditLogEntry]:
        """Get audit logs with optional filters.
        
        Args:
            **filters: Filters to apply
            
        Returns:
            List of audit log entries
        """
        return self.audit_logger.get_logs(**filters)


# API Key generation and validation

def generate_api_key() -> str:
    """Generate a new API key.
    
    Returns:
        A new API key
    """
    return str(uuid.uuid4())