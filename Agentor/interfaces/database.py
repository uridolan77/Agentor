"""
Database interfaces for the Agentor framework.

This module provides interfaces for interacting with various database systems,
including SQL databases, NoSQL databases, and other data storage systems.

This file is a simple import file that re-exports the database interfaces from
the database package.
"""

from .database import (
    DatabaseError, ConnectionError, QueryError, TransactionError,
    DatabaseResult, DatabaseConnection,
    SqlDialect, SqlConnection,
    NoSqlType, DocumentStore, KeyValueStore, GraphDatabase,
    DatabaseRegistry, default_registry,
    # Enhanced database interfaces
    EnhancedSqlConnection, EnhancedDocumentStore, EnhancedKeyValueStore, EnhancedGraphDatabase,
    create_database_connection,
    # Database adapters
    MySqlAdapter, create_mysql_adapter
)

__all__ = [
    'DatabaseError', 'ConnectionError', 'QueryError', 'TransactionError',
    'DatabaseResult', 'DatabaseConnection',
    'SqlDialect', 'SqlConnection',
    'NoSqlType', 'DocumentStore', 'KeyValueStore', 'GraphDatabase',
    'DatabaseRegistry', 'default_registry',
    # Enhanced database interfaces
    'EnhancedSqlConnection', 'EnhancedDocumentStore', 'EnhancedKeyValueStore', 'EnhancedGraphDatabase',
    'create_database_connection',
    # Database adapters
    'MySqlAdapter', 'create_mysql_adapter'
]
