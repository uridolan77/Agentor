"""
Database configuration models for the Agentor framework.

This module provides Pydantic models for configuring database connections.
"""

from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

from .sql import SqlDialect
from .nosql import NoSqlType


class DatabaseConfig(BaseModel):
    """Base configuration for database connections."""
    
    # Connection settings
    name: str = Field(..., description="Name of the database connection")
    connection_string: str = Field(..., description="Connection string for the database")
    
    # Resilience settings
    retry_max_attempts: int = Field(3, description="Maximum number of retry attempts")
    retry_base_delay: float = Field(1.0, description="Base delay between retries in seconds")
    retry_max_delay: float = Field(30.0, description="Maximum delay between retries in seconds")
    retry_jitter: float = Field(0.1, description="Jitter factor for randomizing retry delays")
    
    # Timeout settings
    timeout_seconds: float = Field(30.0, description="Timeout for operations in seconds")
    timeout_strategy: str = Field("adaptive", description="Timeout strategy (fixed, adaptive, percentile)")
    
    # Circuit breaker settings
    circuit_breaker_failures: int = Field(5, description="Number of failures before opening circuit")
    circuit_breaker_recovery: int = Field(60, description="Recovery timeout in seconds")
    circuit_breaker_half_open_calls: int = Field(1, description="Maximum calls in half-open state")
    
    # Bulkhead settings
    bulkhead_max_concurrent: int = Field(10, description="Maximum number of concurrent operations")
    bulkhead_max_queue_size: int = Field(20, description="Maximum size of the waiting queue")
    
    # Connection pool settings
    pool_min_size: int = Field(1, description="Minimum number of connections in the pool")
    pool_max_size: int = Field(10, description="Maximum number of connections in the pool")
    pool_recycle: int = Field(3600, description="Connection recycle time in seconds")
    
    # Additional connection parameters
    connection_params: Dict[str, Any] = Field(default_factory=dict, description="Additional connection parameters")


class SqlDatabaseConfig(DatabaseConfig):
    """Configuration for SQL database connections."""
    
    dialect: SqlDialect = Field(..., description="SQL dialect")
    
    # SQLite specific settings
    sqlite_pragmas: Dict[str, Any] = Field(default_factory=dict, description="SQLite pragmas")
    
    # PostgreSQL specific settings
    pg_schema: Optional[str] = Field(None, description="PostgreSQL schema")
    pg_application_name: Optional[str] = Field(None, description="PostgreSQL application name")
    
    # MySQL specific settings
    mysql_charset: str = Field("utf8mb4", description="MySQL charset")
    mysql_collation: str = Field("utf8mb4_unicode_ci", description="MySQL collation")
    
    # Query settings
    query_timeout: float = Field(30.0, description="Query timeout in seconds")
    max_query_size: int = Field(1024 * 1024, description="Maximum query size in bytes")
    
    # Transaction settings
    transaction_timeout: float = Field(60.0, description="Transaction timeout in seconds")
    
    @validator("connection_string")
    def validate_connection_string(cls, v, values):
        """Validate that the connection string matches the dialect."""
        dialect = values.get("dialect")
        if dialect == SqlDialect.SQLITE and not v.startswith("sqlite:"):
            raise ValueError("SQLite connection string must start with 'sqlite:'")
        elif dialect == SqlDialect.POSTGRESQL and not v.startswith(("postgresql:", "postgres:")):
            raise ValueError("PostgreSQL connection string must start with 'postgresql:' or 'postgres:'")
        elif dialect == SqlDialect.MYSQL and not v.startswith("mysql:"):
            raise ValueError("MySQL connection string must start with 'mysql:'")
        return v


class DocumentStoreConfig(DatabaseConfig):
    """Configuration for document store connections."""
    
    db_type: NoSqlType = Field(..., description="NoSQL database type")
    database_name: str = Field(..., description="Database name")
    
    # MongoDB specific settings
    mongodb_write_concern: Dict[str, Any] = Field(default_factory=dict, description="MongoDB write concern")
    mongodb_read_concern: str = Field("local", description="MongoDB read concern")
    
    # Firestore specific settings
    firestore_project_id: Optional[str] = Field(None, description="Firestore project ID")
    
    # Query settings
    query_timeout: float = Field(30.0, description="Query timeout in seconds")
    
    @validator("connection_string")
    def validate_connection_string(cls, v, values):
        """Validate that the connection string matches the database type."""
        db_type = values.get("db_type")
        if db_type == NoSqlType.MONGODB and not v.startswith("mongodb"):
            raise ValueError("MongoDB connection string must start with 'mongodb'")
        elif db_type == NoSqlType.FIRESTORE and not v.startswith("firestore"):
            raise ValueError("Firestore connection string must start with 'firestore'")
        return v


class KeyValueStoreConfig(DatabaseConfig):
    """Configuration for key-value store connections."""
    
    db_type: NoSqlType = Field(..., description="NoSQL database type")
    
    # Redis specific settings
    redis_db: int = Field(0, description="Redis database number")
    redis_password: Optional[str] = Field(None, description="Redis password")
    
    # DynamoDB specific settings
    dynamodb_region: Optional[str] = Field(None, description="DynamoDB region")
    dynamodb_table: Optional[str] = Field(None, description="DynamoDB table")
    
    # Memcached specific settings
    memcached_max_key_length: int = Field(250, description="Memcached maximum key length")
    
    # Operation settings
    default_ttl: int = Field(0, description="Default TTL for keys in seconds (0 = no expiration)")
    
    @validator("connection_string")
    def validate_connection_string(cls, v, values):
        """Validate that the connection string matches the database type."""
        db_type = values.get("db_type")
        if db_type == NoSqlType.REDIS and not v.startswith("redis"):
            raise ValueError("Redis connection string must start with 'redis'")
        elif db_type == NoSqlType.DYNAMODB and not v.startswith("dynamodb"):
            raise ValueError("DynamoDB connection string must start with 'dynamodb'")
        elif db_type == NoSqlType.MEMCACHED and not v.startswith("memcached"):
            raise ValueError("Memcached connection string must start with 'memcached'")
        return v


class GraphDatabaseConfig(DatabaseConfig):
    """Configuration for graph database connections."""
    
    db_type: NoSqlType = Field(..., description="NoSQL database type")
    
    # Neo4j specific settings
    neo4j_database: Optional[str] = Field(None, description="Neo4j database name")
    
    # Neptune specific settings
    neptune_region: Optional[str] = Field(None, description="Neptune region")
    
    # ArangoDB specific settings
    arango_database: Optional[str] = Field(None, description="ArangoDB database name")
    
    # Query settings
    query_timeout: float = Field(60.0, description="Query timeout in seconds")
    
    @validator("connection_string")
    def validate_connection_string(cls, v, values):
        """Validate that the connection string matches the database type."""
        db_type = values.get("db_type")
        if db_type == NoSqlType.NEO4J and not v.startswith("neo4j"):
            raise ValueError("Neo4j connection string must start with 'neo4j'")
        elif db_type == NoSqlType.NEPTUNE and not v.startswith("neptune"):
            raise ValueError("Neptune connection string must start with 'neptune'")
        elif db_type == NoSqlType.ARANGO and not v.startswith("arango"):
            raise ValueError("ArangoDB connection string must start with 'arango'")
        return v


def create_database_config(config_type: str, **kwargs) -> DatabaseConfig:
    """Create a database configuration of the specified type.
    
    Args:
        config_type: Type of configuration to create (sql, document, key_value, graph)
        **kwargs: Configuration parameters
        
    Returns:
        Database configuration
        
    Raises:
        ValueError: If the configuration type is not supported
    """
    if config_type == "sql":
        return SqlDatabaseConfig(**kwargs)
    elif config_type == "document":
        return DocumentStoreConfig(**kwargs)
    elif config_type == "key_value":
        return KeyValueStoreConfig(**kwargs)
    elif config_type == "graph":
        return GraphDatabaseConfig(**kwargs)
    else:
        raise ValueError(f"Unsupported database configuration type: {config_type}")
