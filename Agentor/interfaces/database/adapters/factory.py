"""
Factory functions for creating database adapters.

This module provides factory functions for creating specialized database adapters.
"""

from typing import Dict, Any, Optional, Union
import logging

from ..config import SqlDatabaseConfig
from ..config.mysql import MySqlConfig
from ..sql import SqlDialect
from .mysql import MySqlAdapter

logger = logging.getLogger(__name__)


def create_mysql_adapter(
    config: Optional[Union[Dict[str, Any], MySqlConfig]] = None,
    **kwargs
) -> MySqlAdapter:
    """Create a MySQL adapter.
    
    Args:
        config: Configuration for the connection (dict or MySqlConfig)
        **kwargs: Additional connection parameters (used if config is not provided)
        
    Returns:
        MySQL adapter
    """
    # Convert dict config to MySqlConfig
    if isinstance(config, dict):
        config = MySqlConfig(**config)
    elif config is None:
        # Create a configuration from kwargs
        name = kwargs.pop("name", "mysql")
        user = kwargs.pop("user", None)
        password = kwargs.pop("password", None)
        host = kwargs.pop("host", "localhost")
        port = kwargs.pop("port", 3306)
        database = kwargs.pop("database", None)
        charset = kwargs.pop("charset", "utf8mb4")
        collation = kwargs.pop("collation", "utf8mb4_unicode_ci")
        autocommit = kwargs.pop("autocommit", False)
        
        # Validate required parameters
        if not user or not database:
            raise ValueError("MySQL user and database must be provided")
        
        # Create the configuration
        config = MySqlConfig(
            name=name,
            mysql_user=user,
            mysql_password=password,
            mysql_host=host,
            mysql_port=port,
            mysql_database=database,
            mysql_charset=charset,
            mysql_collation=collation,
            mysql_autocommit=autocommit,
            connection_params=kwargs
        )
    
    # Create the adapter
    return MySqlAdapter(
        name=config.name,
        connection_string=config.connection_string,
        pool_size=config.mysql_pool_max_size,
        pool_recycle=config.mysql_pool_recycle,
        charset=config.mysql_charset,
        collation=config.mysql_collation,
        autocommit=config.mysql_autocommit,
        **config.connection_params
    )
