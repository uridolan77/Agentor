"""
Database adapter factory for the Agentor framework.

This module provides factory functions for creating database adapters.
"""

from typing import Dict, Any, Optional, List, Union, Type

from .base import DatabaseAdapter
from .adapters.mysql import MySqlAdapter
from .adapters.mysql_replication import MySqlReplicationAdapter
from .adapters.mysql_optimized import MySqlOptimizedAdapter
from .replication import ReplicationConfig, ReplicationMode, ReplicationRole, ServerConfig
from .optimization import OptimizationConfig, OptimizationLevel
from .optimization.config import (
    QueryOptimizationConfig,
    IndexOptimizationConfig,
    ServerOptimizationConfig,
    PerformanceMonitoringConfig
)


def create_mysql_adapter(config: Dict[str, Any]) -> Union[MySqlAdapter, MySqlReplicationAdapter, MySqlOptimizedAdapter]:
    """Create a MySQL adapter.

    Args:
        config: The adapter configuration

    Returns:
        A MySQL adapter
    """
    # Check if replication is enabled
    if "mysql_replication_mode" in config:
        # Create a replication configuration
        replication_mode = config.get("mysql_replication_mode", "single_primary").lower()

        if replication_mode == "single_primary":
            mode = ReplicationMode.SINGLE_PRIMARY
        elif replication_mode == "multi_primary":
            mode = ReplicationMode.MULTI_PRIMARY
        elif replication_mode == "primary_primary":
            mode = ReplicationMode.PRIMARY_PRIMARY
        else:
            mode = ReplicationMode.CUSTOM

        # Create server configurations
        servers = []
        for server_config in config.get("mysql_servers", []):
            # Get the server role
            role_str = server_config.get("role", "auto").lower()

            if role_str == "primary":
                role = ReplicationRole.PRIMARY
            elif role_str == "replica":
                role = ReplicationRole.REPLICA
            else:
                role = ReplicationRole.AUTO

            # Create a server configuration
            server = ServerConfig(
                host=server_config.get("host", "localhost"),
                port=server_config.get("port", 3306),
                user=server_config.get("user", "root"),
                password=server_config.get("password"),
                database=server_config.get("database"),
                max_connections=server_config.get("max_connections", 10),
                connection_timeout=server_config.get("connection_timeout", 10.0),
                role=role,
                weight=server_config.get("weight", 1)
            )

            servers.append(server)

        # Create a replication configuration
        replication_config = ReplicationConfig(
            mode=mode,
            servers=servers,
            read_write_splitting=config.get("mysql_read_write_splitting", True),
            read_from_primary=config.get("mysql_read_from_primary", False),
            write_to_replica=config.get("mysql_write_to_replica", False),
            load_balancing_strategy=config.get("mysql_load_balancing_strategy", "round_robin"),
            failover_enabled=config.get("mysql_failover_enabled", True),
            failover_timeout=config.get("mysql_failover_timeout", 30.0),
            failover_retry_interval=config.get("mysql_failover_retry_interval", 5.0),
            max_failover_attempts=config.get("mysql_max_failover_attempts", 3),
            health_check_interval=config.get("mysql_health_check_interval", 60.0),
            health_check_timeout=config.get("mysql_health_check_timeout", 5.0),
            monitoring_enabled=config.get("mysql_monitoring_enabled", True),
            monitoring_interval=config.get("mysql_monitoring_interval", 60.0)
        )

        # Create a MySQL replication adapter
        return MySqlReplicationAdapter(
            name=config.get("name", "mysql"),
            replication_config=replication_config,
            charset=config.get("mysql_charset", "utf8mb4"),
            collation=config.get("mysql_collation", "utf8mb4_unicode_ci"),
            autocommit=config.get("mysql_autocommit", True)
        )
    # Check if optimization is enabled
    elif "mysql_optimization_enabled" in config and config.get("mysql_optimization_enabled", False):
        # Create an optimization configuration
        optimization_level_str = config.get("mysql_optimization_level", "moderate").lower()

        if optimization_level_str == "none":
            level = OptimizationLevel.NONE
        elif optimization_level_str == "basic":
            level = OptimizationLevel.BASIC
        elif optimization_level_str == "moderate":
            level = OptimizationLevel.MODERATE
        elif optimization_level_str == "aggressive":
            level = OptimizationLevel.AGGRESSIVE
        else:
            level = OptimizationLevel.CUSTOM

        # Create query optimization configuration
        query_optimization = QueryOptimizationConfig(
            enabled=config.get("mysql_query_optimization_enabled", True),
            level=level,
            rewrite_queries=config.get("mysql_rewrite_queries", True),
            add_missing_indexes=config.get("mysql_add_missing_indexes", True),
            optimize_joins=config.get("mysql_optimize_joins", True),
            optimize_where_clauses=config.get("mysql_optimize_where_clauses", True),
            optimize_order_by=config.get("mysql_optimize_order_by", True),
            optimize_group_by=config.get("mysql_optimize_group_by", True),
            optimize_limit=config.get("mysql_optimize_limit", True),
            analyze_queries=config.get("mysql_analyze_queries", True),
            slow_query_threshold=config.get("mysql_slow_query_threshold", 1.0),
            very_slow_query_threshold=config.get("mysql_very_slow_query_threshold", 10.0),
            collect_query_plans=config.get("mysql_collect_query_plans", True),
            analyze_query_plans=config.get("mysql_analyze_query_plans", True)
        )

        # Create index optimization configuration
        index_optimization = IndexOptimizationConfig(
            enabled=config.get("mysql_index_optimization_enabled", True),
            level=level,
            analyze_indexes=config.get("mysql_analyze_indexes", True),
            collect_index_stats=config.get("mysql_collect_index_stats", True),
            recommend_indexes=config.get("mysql_recommend_indexes", True),
            recommend_composite_indexes=config.get("mysql_recommend_composite_indexes", True),
            recommend_covering_indexes=config.get("mysql_recommend_covering_indexes", True),
            auto_create_indexes=config.get("mysql_auto_create_indexes", False),
            max_indexes_per_table=config.get("mysql_max_indexes_per_table", 5)
        )

        # Create server optimization configuration
        server_optimization = ServerOptimizationConfig(
            enabled=config.get("mysql_server_optimization_enabled", True),
            level=level,
            analyze_server=config.get("mysql_analyze_server", True),
            collect_server_stats=config.get("mysql_collect_server_stats", True),
            recommend_server_settings=config.get("mysql_recommend_server_settings", True),
            auto_configure_server=config.get("mysql_auto_configure_server", False)
        )

        # Create performance monitoring configuration
        performance_monitoring = PerformanceMonitoringConfig(
            enabled=config.get("mysql_performance_monitoring_enabled", True),
            level=level,
            monitoring_interval=config.get("mysql_monitoring_interval", 60.0),
            detailed_monitoring_interval=config.get("mysql_detailed_monitoring_interval", 300.0),
            collect_query_metrics=config.get("mysql_collect_query_metrics", True),
            collect_index_metrics=config.get("mysql_collect_index_metrics", True),
            collect_server_metrics=config.get("mysql_collect_server_metrics", True),
            alert_on_slow_queries=config.get("mysql_alert_on_slow_queries", True),
            alert_on_high_load=config.get("mysql_alert_on_high_load", True),
            alert_threshold=config.get("mysql_alert_threshold", 0.8)
        )

        # Create an optimization configuration
        optimization_config = OptimizationConfig(
            enabled=config.get("mysql_optimization_enabled", True),
            level=level,
            query_optimization=query_optimization,
            index_optimization=index_optimization,
            server_optimization=server_optimization,
            performance_monitoring=performance_monitoring
        )

        # Create a MySQL optimized adapter
        return MySqlOptimizedAdapter(
            name=config.get("name", "mysql"),
            optimization_config=optimization_config,
            host=config.get("mysql_host", "localhost"),
            port=config.get("mysql_port", 3306),
            user=config.get("mysql_user", "root"),
            password=config.get("mysql_password"),
            database=config.get("mysql_database"),
            charset=config.get("mysql_charset", "utf8mb4"),
            collation=config.get("mysql_collation", "utf8mb4_unicode_ci"),
            autocommit=config.get("mysql_autocommit", True),
            pool_min_size=config.get("mysql_pool_min_size", 1),
            pool_max_size=config.get("mysql_pool_max_size", 10),
            pool_recycle=config.get("mysql_pool_recycle", 3600)
        )
    else:
        # Create a regular MySQL adapter
        return MySqlAdapter(
            name=config.get("name", "mysql"),
            host=config.get("mysql_host", "localhost"),
            port=config.get("mysql_port", 3306),
            user=config.get("mysql_user", "root"),
            password=config.get("mysql_password"),
            database=config.get("mysql_database"),
            charset=config.get("mysql_charset", "utf8mb4"),
            collation=config.get("mysql_collation", "utf8mb4_unicode_ci"),
            autocommit=config.get("mysql_autocommit", True),
            pool_min_size=config.get("mysql_pool_min_size", 1),
            pool_max_size=config.get("mysql_pool_max_size", 10),
            pool_recycle=config.get("mysql_pool_recycle", 3600)
        )


def create_adapter(adapter_type: str, config: Dict[str, Any]) -> Optional[DatabaseAdapter]:
    """Create a database adapter.

    Args:
        adapter_type: The type of adapter to create
        config: The adapter configuration

    Returns:
        A database adapter, or None if the adapter type is not supported
    """
    if adapter_type.lower() == "mysql":
        return create_mysql_adapter(config)
    else:
        return None
