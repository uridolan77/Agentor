"""
Database connection pooling for the Agentor framework.

This package provides advanced connection pooling capabilities for database connections.
"""

from .config import ConnectionPoolConfig, ConnectionValidationMode
from .manager import ConnectionPoolManager
from .mysql import MySqlConnectionPool

__all__ = [
    'ConnectionPoolConfig',
    'ConnectionValidationMode',
    'ConnectionPoolManager',
    'MySqlConnectionPool'
]
