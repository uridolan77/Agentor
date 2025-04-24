"""
Enhanced database interfaces for the Agentor framework.

This package provides enhanced database interfaces with resilience patterns,
connection pooling, and other advanced features.
"""

from .sql import EnhancedSqlConnection
from .document_store import EnhancedDocumentStore
from .key_value_store import EnhancedKeyValueStore
from .graph_database import EnhancedGraphDatabase
from .factory import create_database_connection

__all__ = [
    'EnhancedSqlConnection',
    'EnhancedDocumentStore',
    'EnhancedKeyValueStore',
    'EnhancedGraphDatabase',
    'create_database_connection'
]
