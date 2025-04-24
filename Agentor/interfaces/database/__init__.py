"""
Database interfaces for the Agentor framework.

This module provides interfaces for interacting with various database systems,
including SQL databases, NoSQL databases, and other data storage systems.
"""

from .base import (
    DatabaseError, ConnectionError, QueryError, TransactionError,
    DatabaseResult, DatabaseConnection
)
from .sql import SqlDialect, SqlConnection
from .nosql import NoSqlType, DocumentStore, KeyValueStore, GraphDatabase
from .registry import DatabaseRegistry, default_registry

# Import enhanced database interfaces
from .enhanced import (
    EnhancedSqlConnection,
    EnhancedDocumentStore,
    EnhancedKeyValueStore,
    EnhancedGraphDatabase,
    create_database_connection
)

# Import database adapters
from .adapters import (
    MySqlAdapter,
    create_mysql_adapter
)

# Import configuration models
from .config import (
    DatabaseConfig,
    SqlDatabaseConfig,
    DocumentStoreConfig,
    KeyValueStoreConfig,
    GraphDatabaseConfig,
    create_database_config
)

__all__ = [
    'DatabaseError', 'ConnectionError', 'QueryError', 'TransactionError',
    'DatabaseResult', 'DatabaseConnection',
    'SqlDialect', 'SqlConnection',
    'NoSqlType', 'DocumentStore', 'KeyValueStore', 'GraphDatabase',
    'DatabaseRegistry', 'default_registry',

    # Enhanced database interfaces
    'EnhancedSqlConnection',
    'EnhancedDocumentStore',
    'EnhancedKeyValueStore',
    'EnhancedGraphDatabase',
    'create_database_connection',

    # Configuration models
    'DatabaseConfig',
    'SqlDatabaseConfig',
    'DocumentStoreConfig',
    'KeyValueStoreConfig',
    'GraphDatabaseConfig',
    'create_database_config',

    # Database adapters
    'MySqlAdapter',
    'create_mysql_adapter'
]
