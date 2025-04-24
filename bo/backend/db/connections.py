"""Database connection utilities."""

import pymysql
from typing import Dict, Any, Optional
from .reporting import DataSource

def get_connection_for_data_source(data_source: DataSource) -> pymysql.Connection:
    """Get a database connection for a data source.
    
    Args:
        data_source: The data source to connect to
        
    Returns:
        A database connection
        
    Raises:
        ValueError: If the connection type is not supported
        Exception: If the connection fails
    """
    if data_source.connection_type.lower() == "mysql":
        return get_mysql_connection(data_source.connection_config)
    else:
        raise ValueError(f"Unsupported connection type: {data_source.connection_type}")

def get_mysql_connection(config: Dict[str, Any]) -> pymysql.Connection:
    """Get a MySQL database connection.
    
    Args:
        config: The connection configuration
        
    Returns:
        A MySQL database connection
        
    Raises:
        Exception: If the connection fails
    """
    # Extract connection parameters
    host = config.get("host", "localhost")
    port = config.get("port")
    port = int(port) if port else 3306
    database = config.get("database", "")
    username = config.get("username", "")
    password = config.get("password", "")
    ssl_enabled = config.get("ssl_enabled", False)
    
    # Validate required parameters
    if not database:
        raise ValueError("Database name is required")
    
    # Create connection
    conn_args = {
        "host": host,
        "port": port,
        "user": username,
        "password": password,
        "database": database,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor
    }
    
    # Add SSL if enabled
    if ssl_enabled:
        conn_args["ssl"] = {
            "ca": config.get("ssl_ca", None),
            "cert": config.get("ssl_cert", None),
            "key": config.get("ssl_key", None),
        }
    
    # Connect to the database
    try:
        connection = pymysql.connect(**conn_args)
        return connection
    except Exception as e:
        raise Exception(f"Failed to connect to MySQL database: {str(e)}")
