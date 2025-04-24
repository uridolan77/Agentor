"""
Factory functions for creating enhanced database connections.

This module provides factory functions for creating enhanced database connections
with resilience patterns, connection pooling, and other advanced features.
"""

from typing import Dict, Any, Optional, Union, Type
import logging

from ..base import DatabaseConnection
from ..sql import SqlConnection, SqlDialect
from ..nosql import DocumentStore, KeyValueStore, GraphDatabase, NoSqlType
from ..config import (
    DatabaseConfig, SqlDatabaseConfig, DocumentStoreConfig,
    KeyValueStoreConfig, GraphDatabaseConfig, create_database_config
)
from .sql import EnhancedSqlConnection
from .document_store import EnhancedDocumentStore
from .key_value_store import EnhancedKeyValueStore
from .graph_database import EnhancedGraphDatabase

logger = logging.getLogger(__name__)


def create_database_connection(
    connection_type: str,
    config: Optional[Union[Dict[str, Any], DatabaseConfig]] = None,
    **kwargs
) -> DatabaseConnection:
    """Create an enhanced database connection.
    
    Args:
        connection_type: Type of connection to create (sql, document, key_value, graph)
        config: Configuration for the connection (dict or DatabaseConfig)
        **kwargs: Additional connection parameters (used if config is not provided)
        
    Returns:
        Enhanced database connection
        
    Raises:
        ValueError: If the connection type is not supported
    """
    # Convert dict config to DatabaseConfig
    if isinstance(config, dict):
        config = create_database_config(connection_type, **config)
    
    # Create the connection based on the type
    if connection_type == "sql":
        return _create_sql_connection(config, **kwargs)
    elif connection_type == "document":
        return _create_document_store(config, **kwargs)
    elif connection_type == "key_value":
        return _create_key_value_store(config, **kwargs)
    elif connection_type == "graph":
        return _create_graph_database(config, **kwargs)
    else:
        raise ValueError(f"Unsupported database connection type: {connection_type}")


def _create_sql_connection(
    config: Optional[SqlDatabaseConfig] = None,
    **kwargs
) -> EnhancedSqlConnection:
    """Create an enhanced SQL connection.
    
    Args:
        config: Configuration for the connection
        **kwargs: Additional connection parameters (used if config is not provided)
        
    Returns:
        Enhanced SQL connection
    """
    if config is None:
        # Create a configuration from kwargs
        name = kwargs.pop("name", "sql")
        dialect = kwargs.pop("dialect", SqlDialect.SQLITE)
        connection_string = kwargs.pop("connection_string", "sqlite:///database.db")
        
        config = SqlDatabaseConfig(
            name=name,
            dialect=dialect,
            connection_string=connection_string,
            connection_params=kwargs
        )
    
    # Create the connection
    return EnhancedSqlConnection(
        name=config.name,
        dialect=config.dialect,
        connection_string=config.connection_string,
        pool_size=config.pool_max_size,
        pool_recycle=config.pool_recycle,
        **config.connection_params
    )


def _create_document_store(
    config: Optional[DocumentStoreConfig] = None,
    **kwargs
) -> EnhancedDocumentStore:
    """Create an enhanced document store.
    
    Args:
        config: Configuration for the connection
        **kwargs: Additional connection parameters (used if config is not provided)
        
    Returns:
        Enhanced document store
    """
    if config is None:
        # Create a configuration from kwargs
        name = kwargs.pop("name", "document")
        db_type = kwargs.pop("db_type", NoSqlType.MONGODB)
        connection_string = kwargs.pop("connection_string", "mongodb://localhost:27017")
        database_name = kwargs.pop("database_name", "agentor")
        
        config = DocumentStoreConfig(
            name=name,
            db_type=db_type,
            connection_string=connection_string,
            database_name=database_name,
            connection_params=kwargs
        )
    
    # Create the connection
    return EnhancedDocumentStore(
        name=config.name,
        db_type=config.db_type,
        connection_string=config.connection_string,
        database_name=config.database_name,
        **config.connection_params
    )


def _create_key_value_store(
    config: Optional[KeyValueStoreConfig] = None,
    **kwargs
) -> EnhancedKeyValueStore:
    """Create an enhanced key-value store.
    
    Args:
        config: Configuration for the connection
        **kwargs: Additional connection parameters (used if config is not provided)
        
    Returns:
        Enhanced key-value store
    """
    if config is None:
        # Create a configuration from kwargs
        name = kwargs.pop("name", "key_value")
        db_type = kwargs.pop("db_type", NoSqlType.REDIS)
        connection_string = kwargs.pop("connection_string", "redis://localhost:6379")
        
        config = KeyValueStoreConfig(
            name=name,
            db_type=db_type,
            connection_string=connection_string,
            connection_params=kwargs
        )
    
    # Create the connection
    return EnhancedKeyValueStore(
        name=config.name,
        db_type=config.db_type,
        connection_string=config.connection_string,
        **config.connection_params
    )


def _create_graph_database(
    config: Optional[GraphDatabaseConfig] = None,
    **kwargs
) -> EnhancedGraphDatabase:
    """Create an enhanced graph database.
    
    Args:
        config: Configuration for the connection
        **kwargs: Additional connection parameters (used if config is not provided)
        
    Returns:
        Enhanced graph database
    """
    if config is None:
        # Create a configuration from kwargs
        name = kwargs.pop("name", "graph")
        db_type = kwargs.pop("db_type", NoSqlType.NEO4J)
        connection_string = kwargs.pop("connection_string", "neo4j://localhost:7687")
        
        config = GraphDatabaseConfig(
            name=name,
            db_type=db_type,
            connection_string=connection_string,
            connection_params=kwargs
        )
    
    # Create the connection
    return EnhancedGraphDatabase(
        name=config.name,
        db_type=config.db_type,
        connection_string=config.connection_string,
        **config.connection_params
    )
