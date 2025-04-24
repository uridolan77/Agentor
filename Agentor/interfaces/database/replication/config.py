"""
Replication configuration for the Agentor framework.

This module provides configuration classes for database replication.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, validator


class ReplicationRole(str, Enum):
    """Replication roles."""
    
    PRIMARY = "primary"  # Primary server
    REPLICA = "replica"  # Replica server
    AUTO = "auto"  # Auto-detect role


class ReplicationMode(str, Enum):
    """Replication modes."""
    
    SINGLE_PRIMARY = "single_primary"  # Single primary with multiple replicas
    MULTI_PRIMARY = "multi_primary"  # Multiple primaries
    PRIMARY_PRIMARY = "primary_primary"  # Two primaries replicating to each other
    CUSTOM = "custom"  # Custom replication mode


class ServerConfig(BaseModel):
    """Configuration for a database server in a replication setup."""
    
    # Server settings
    host: str = Field(..., description="The host of the server")
    port: int = Field(3306, description="The port of the server")
    user: str = Field(..., description="The user to connect with")
    password: Optional[str] = Field(None, description="The password to connect with")
    database: Optional[str] = Field(None, description="The database to connect to")
    
    # Connection settings
    max_connections: int = Field(10, description="Maximum number of connections")
    connection_timeout: float = Field(10.0, description="Connection timeout in seconds")
    
    # Replication settings
    role: ReplicationRole = Field(ReplicationRole.AUTO, description="The role of the server")
    weight: int = Field(1, description="The weight of the server for load balancing")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("port")
    def validate_port(cls, v):
        """Validate that the port is valid."""
        if v <= 0 or v > 65535:
            raise ValueError(f"Port ({v}) must be between 1 and 65535")
        
        return v
    
    @validator("max_connections")
    def validate_max_connections(cls, v):
        """Validate that the maximum connections is positive."""
        if v <= 0:
            raise ValueError(f"Maximum connections ({v}) must be positive")
        
        return v
    
    @validator("connection_timeout")
    def validate_connection_timeout(cls, v):
        """Validate that the connection timeout is positive."""
        if v <= 0:
            raise ValueError(f"Connection timeout ({v}) must be positive")
        
        return v
    
    @validator("weight")
    def validate_weight(cls, v):
        """Validate that the weight is positive."""
        if v <= 0:
            raise ValueError(f"Weight ({v}) must be positive")
        
        return v


class ReplicationConfig(BaseModel):
    """Configuration for database replication."""
    
    # Replication settings
    mode: ReplicationMode = Field(ReplicationMode.SINGLE_PRIMARY, description="Replication mode")
    servers: List[ServerConfig] = Field(..., description="List of servers in the replication setup")
    
    # Read/write splitting settings
    read_write_splitting: bool = Field(True, description="Whether to use read/write splitting")
    read_from_primary: bool = Field(False, description="Whether to allow reads from the primary")
    write_to_replica: bool = Field(False, description="Whether to allow writes to replicas")
    
    # Load balancing settings
    load_balancing_strategy: str = Field("round_robin", description="Load balancing strategy")
    
    # Failover settings
    failover_enabled: bool = Field(True, description="Whether to enable failover")
    failover_timeout: float = Field(30.0, description="Failover timeout in seconds")
    failover_retry_interval: float = Field(5.0, description="Failover retry interval in seconds")
    max_failover_attempts: int = Field(3, description="Maximum number of failover attempts")
    
    # Health check settings
    health_check_interval: float = Field(60.0, description="Health check interval in seconds")
    health_check_timeout: float = Field(5.0, description="Health check timeout in seconds")
    
    # Monitoring settings
    monitoring_enabled: bool = Field(True, description="Whether to enable monitoring")
    monitoring_interval: float = Field(60.0, description="Monitoring interval in seconds")
    
    # Additional settings
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional settings")
    
    @validator("servers")
    def validate_servers(cls, v, values):
        """Validate that there are enough servers for the replication mode."""
        mode = values.get("mode", ReplicationMode.SINGLE_PRIMARY)
        
        if mode == ReplicationMode.SINGLE_PRIMARY and len(v) < 1:
            raise ValueError(f"Single primary mode requires at least 1 server")
        
        if mode == ReplicationMode.MULTI_PRIMARY and len(v) < 2:
            raise ValueError(f"Multi primary mode requires at least 2 servers")
        
        if mode == ReplicationMode.PRIMARY_PRIMARY and len(v) != 2:
            raise ValueError(f"Primary-primary mode requires exactly 2 servers")
        
        # Check that there is at least one primary server
        primary_count = sum(1 for server in v if server.role == ReplicationRole.PRIMARY)
        
        if mode == ReplicationMode.SINGLE_PRIMARY and primary_count > 1:
            raise ValueError(f"Single primary mode cannot have more than 1 primary server")
        
        if mode == ReplicationMode.PRIMARY_PRIMARY and primary_count != 2:
            raise ValueError(f"Primary-primary mode requires exactly 2 primary servers")
        
        return v
    
    @validator("failover_timeout")
    def validate_failover_timeout(cls, v):
        """Validate that the failover timeout is positive."""
        if v <= 0:
            raise ValueError(f"Failover timeout ({v}) must be positive")
        
        return v
    
    @validator("failover_retry_interval")
    def validate_failover_retry_interval(cls, v):
        """Validate that the failover retry interval is positive."""
        if v <= 0:
            raise ValueError(f"Failover retry interval ({v}) must be positive")
        
        return v
    
    @validator("max_failover_attempts")
    def validate_max_failover_attempts(cls, v):
        """Validate that the maximum failover attempts is positive."""
        if v <= 0:
            raise ValueError(f"Maximum failover attempts ({v}) must be positive")
        
        return v
    
    @validator("health_check_interval")
    def validate_health_check_interval(cls, v):
        """Validate that the health check interval is positive."""
        if v <= 0:
            raise ValueError(f"Health check interval ({v}) must be positive")
        
        return v
    
    @validator("health_check_timeout")
    def validate_health_check_timeout(cls, v):
        """Validate that the health check timeout is positive."""
        if v <= 0:
            raise ValueError(f"Health check timeout ({v}) must be positive")
        
        return v
    
    @validator("monitoring_interval")
    def validate_monitoring_interval(cls, v):
        """Validate that the monitoring interval is positive."""
        if v <= 0:
            raise ValueError(f"Monitoring interval ({v}) must be positive")
        
        return v
