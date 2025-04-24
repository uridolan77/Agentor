"""
MySQL configuration model for the Agentor framework.

This module provides a specialized configuration model for MySQL databases.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator

from ..config import SqlDatabaseConfig
from ..sql import SqlDialect


class MySqlConfig(SqlDatabaseConfig):
    """Configuration for MySQL database connections."""
    
    # Ensure the dialect is MySQL
    dialect: SqlDialect = Field(SqlDialect.MYSQL, const=True, description="SQL dialect (always MySQL)")
    
    # MySQL-specific settings
    mysql_host: str = Field("localhost", description="MySQL host")
    mysql_port: int = Field(3306, description="MySQL port")
    mysql_user: str = Field(..., description="MySQL user")
    mysql_password: Optional[str] = Field(None, description="MySQL password")
    mysql_database: str = Field(..., description="MySQL database name")
    mysql_charset: str = Field("utf8mb4", description="MySQL charset")
    mysql_collation: str = Field("utf8mb4_unicode_ci", description="MySQL collation")
    mysql_autocommit: bool = Field(False, description="MySQL autocommit mode")
    mysql_use_unicode: bool = Field(True, description="MySQL use unicode")
    mysql_get_warnings: bool = Field(True, description="MySQL get warnings")
    mysql_ssl: bool = Field(False, description="MySQL SSL mode")
    mysql_ssl_ca: Optional[str] = Field(None, description="MySQL SSL CA certificate")
    mysql_ssl_cert: Optional[str] = Field(None, description="MySQL SSL client certificate")
    mysql_ssl_key: Optional[str] = Field(None, description="MySQL SSL client key")
    mysql_ssl_verify_cert: bool = Field(False, description="MySQL SSL verify server certificate")
    mysql_init_command: Optional[str] = Field(None, description="MySQL initialization command")
    
    # Query settings
    mysql_query_timeout: float = Field(30.0, description="MySQL query timeout in seconds")
    mysql_connect_timeout: float = Field(10.0, description="MySQL connect timeout in seconds")
    mysql_read_default_file: Optional[str] = Field(None, description="MySQL configuration file")
    mysql_read_default_group: Optional[str] = Field(None, description="MySQL configuration group")
    
    # Connection pool settings
    mysql_pool_min_size: int = Field(1, description="Minimum number of connections in the pool")
    mysql_pool_max_size: int = Field(10, description="Maximum number of connections in the pool")
    mysql_pool_recycle: int = Field(3600, description="Connection recycle time in seconds")
    
    @validator("connection_string", pre=True, always=True)
    def validate_connection_string(cls, v, values):
        """Validate and generate the connection string if not provided."""
        if v:
            # If a connection string is provided, validate it
            if not v.startswith("mysql"):
                raise ValueError("MySQL connection string must start with 'mysql'")
            return v
        
        # Generate a connection string from the individual parameters
        host = values.get("mysql_host", "localhost")
        port = values.get("mysql_port", 3306)
        user = values.get("mysql_user")
        password = values.get("mysql_password", "")
        database = values.get("mysql_database")
        
        if not user or not database:
            raise ValueError("MySQL user and database must be provided")
        
        # Build the connection string
        if password:
            return f"mysql://{user}:{password}@{host}:{port}/{database}"
        else:
            return f"mysql://{user}@{host}:{port}/{database}"
    
    def to_connection_params(self) -> Dict[str, Any]:
        """Convert the configuration to connection parameters for aiomysql.
        
        Returns:
            Dictionary of connection parameters
        """
        params = {
            "host": self.mysql_host,
            "port": self.mysql_port,
            "user": self.mysql_user,
            "db": self.mysql_database,
            "charset": self.mysql_charset,
            "autocommit": self.mysql_autocommit,
            "use_unicode": self.mysql_use_unicode,
            "connect_timeout": self.mysql_connect_timeout,
            "echo": False  # Don't echo SQL
        }
        
        # Add password if provided
        if self.mysql_password:
            params["password"] = self.mysql_password
        
        # Add SSL parameters if enabled
        if self.mysql_ssl:
            params["ssl"] = {
                "ca": self.mysql_ssl_ca,
                "cert": self.mysql_ssl_cert,
                "key": self.mysql_ssl_key,
                "verify_cert": self.mysql_ssl_verify_cert
            }
        
        # Add initialization command if provided
        if self.mysql_init_command:
            params["init_command"] = self.mysql_init_command
        
        # Add read default file and group if provided
        if self.mysql_read_default_file:
            params["read_default_file"] = self.mysql_read_default_file
        
        if self.mysql_read_default_group:
            params["read_default_group"] = self.mysql_read_default_group
        
        # Add any additional connection parameters
        params.update(self.connection_params)
        
        return params
