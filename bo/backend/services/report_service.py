"""
Report execution service for the reporting system.
"""
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Union
import asyncio
from datetime import datetime

import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..db.reporting import DataSource, Dimension, Metric, CalculatedMetric, Report
from ..db.database import get_db

logger = logging.getLogger(__name__)

class ReportExecutionError(Exception):
    """Exception raised for errors during report execution."""
    pass

class DataSourceConnectionError(Exception):
    """Exception raised when a connection to a data source cannot be established."""
    pass

class ReportService:
    """Service for executing reports against different data sources."""
    
    def __init__(self, db: Session):
        self.db = db
        self.connections = {}
    
    async def get_data_source(self, data_source_id: str) -> DataSource:
        """Get a data source by ID."""
        data_source = self.db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise ValueError(f"Data source with ID {data_source_id} not found")
        return data_source
    
    async def connect_to_data_source(self, data_source: DataSource) -> Any:
        """Connect to a data source based on its connection type."""
        if data_source.id in self.connections:
            return self.connections[data_source.id]
        
        connection_type = data_source.connection_type
        connection_config = data_source.connection_config
        
        try:
            if connection_type == "database":
                # Handle SQL database connections
                connection = await self._create_database_connection(connection_config)
            elif connection_type == "api":
                # Handle API connections
                connection = await self._create_api_connection(connection_config)
            elif connection_type == "file":
                # Handle file connections
                connection = await self._create_file_connection(connection_config)
            else:
                raise ValueError(f"Unsupported connection type: {connection_type}")
            
            # Cache the connection
            self.connections[data_source.id] = connection
            return connection
        except Exception as e:
            logger.error(f"Error connecting to data source '{data_source.name}': {str(e)}")
            raise DataSourceConnectionError(f"Failed to connect to data source: {str(e)}") from e
    
    async def _create_database_connection(self, config: Dict[str, Any]) -> Any:
        """Create a connection to a SQL database."""
        connection_string = self._get_connection_string(config)
        try:
            engine = sqlalchemy.create_engine(connection_string)
            return engine
        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {str(e)}")
            raise DataSourceConnectionError(f"Failed to connect to database: {str(e)}") from e
    
    def _get_connection_string(self, config: Dict[str, Any]) -> str:
        """Get a SQLAlchemy connection string from a database configuration."""
        db_type = config.get("dbType", "").lower()
        
        if db_type == "postgresql":
            return f"postgresql://{config.get('username')}:{config.get('password')}@{config.get('host')}:{config.get('port', 5432)}/{config.get('database')}"
        elif db_type == "mysql":
            return f"mysql+pymysql://{config.get('username')}:{config.get('password')}@{config.get('host')}:{config.get('port', 3306)}/{config.get('database')}"
        elif db_type == "sqlite":
            return f"sqlite:///{config.get('database')}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    async def _create_api_connection(self, config: Dict[str, Any]) -> Any:
        """Create a connection configuration for an API data source."""
        # For API connections, we just return the configuration
        # The actual connection will be made when executing the query
        return config
    
    async def _create_file_connection(self, config: Dict[str, Any]) -> Any:
        """Create a connection configuration for a file data source."""
        # For file connections, we just return the configuration
        # The actual file reading will be done when executing the query
        return config
    
    async def execute_report(self, report_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a report and return the results."""
        # Get the report
        report = self.db.query(Report).filter(Report.id == report_id).first()
        if not report:
            raise ValueError(f"Report with ID {report_id} not found")
        
        # Get the data source
        data_source = await self.get_data_source(report.data_source_id)
        
        # Connect to the data source
        connection = await self.connect_to_data_source(data_source)
        
        # Execute the report based on its configuration
        try:
            configuration = report.configuration
            report_type = configuration.get("reportType", "table")
            
            # Get dimensions and metrics for the report
            dimensions = self._get_report_dimensions(configuration, data_source)
            metrics = self._get_report_metrics(configuration, data_source)
            
            # Build and execute the query
            if data_source.connection_type == "database":
                result = await self._execute_database_report(
                    connection,
                    dimensions,
                    metrics,
                    configuration,
                    params
                )
            elif data_source.connection_type == "api":
                result = await self._execute_api_report(
                    connection,
                    dimensions,
                    metrics,
                    configuration,
                    params
                )
            elif data_source.connection_type == "file":
                result = await self._execute_file_report(
                    connection,
                    dimensions,
                    metrics,
                    configuration,
                    params
                )
            else:
                raise ValueError(f"Unsupported data source type: {data_source.connection_type}")
            
            # Update the report's last run time
            report.last_run_at = datetime.now()
            self.db.commit()
            
            return {
                "id": str(uuid.uuid4()),
                "report_id": report_id,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "data": result,
                "params": params or {}
            }
        except Exception as e:
            logger.error(f"Error executing report '{report.name}': {str(e)}")
            raise ReportExecutionError(f"Failed to execute report: {str(e)}") from e
    
    def _get_report_dimensions(
        self, configuration: Dict[str, Any], data_source: DataSource
    ) -> List[Dimension]:
        """Get the dimensions for a report based on its configuration."""
        dimension_ids = [
            field["id"]
            for field in configuration.get("fields", [])
            if field["type"] == "dimension"
        ]
        
        dimensions = (
            self.db.query(Dimension)
            .filter(Dimension.id.in_(dimension_ids))
            .filter(Dimension.data_source_id == data_source.id)
            .all()
        )
        
        return dimensions
    
    def _get_report_metrics(
        self, configuration: Dict[str, Any], data_source: DataSource
    ) -> List[Union[Metric, CalculatedMetric]]:
        """Get the metrics for a report based on its configuration."""
        metric_ids = [
            field["id"]
            for field in configuration.get("fields", [])
            if field["type"] in ["metric", "calculated_metric"]
        ]
        
        # Get regular metrics
        metrics = (
            self.db.query(Metric)
            .filter(Metric.id.in_(metric_ids))
            .filter(Metric.data_source_id == data_source.id)
            .all()
        )
        
        # Get calculated metrics
        calculated_metrics = (
            self.db.query(CalculatedMetric)
            .filter(CalculatedMetric.id.in_(metric_ids))
            .filter(CalculatedMetric.data_source_id == data_source.id)
            .all()
        )
        
        return metrics + calculated_metrics
    
    async def _execute_database_report(
        self,
        connection,
        dimensions: List[Dimension],
        metrics: List[Union[Metric, CalculatedMetric]],
        configuration: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a report against a SQL database."""
        # Build the SQL query
        query = self._build_sql_query(dimensions, metrics, configuration, params)
        
        try:
            # Execute the query
            with connection.connect() as conn:
                result = conn.execute(sqlalchemy.text(query))
                rows = [dict(row) for row in result]
            
            return {
                "columns": result.keys(),
                "rows": rows
            }
        except SQLAlchemyError as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise ReportExecutionError(f"Failed to execute database query: {str(e)}") from e
    
    def _build_sql_query(
        self,
        dimensions: List[Dimension],
        metrics: List[Union[Metric, CalculatedMetric]],
        configuration: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a SQL query for a report."""
        # Table to query
        tables = set()
        for dim in dimensions:
            tables.add(dim.table_name)
        
        for metric in metrics:
            if hasattr(metric, "table_name"):  # Regular metrics have table_name
                tables.add(metric.table_name)
        
        # Simple case: only one table
        if len(tables) == 1:
            table_name = tables.pop()
        else:
            # TODO: Handle joins for multiple tables
            raise NotImplementedError("Multi-table queries are not yet supported")
        
        # SELECT clause
        select_clause = []
        for dim in dimensions:
            select_clause.append(f"{dim.table_name}.{dim.column_name} AS {dim.name}")
        
        for metric in metrics:
            if isinstance(metric, Metric):
                if metric.calculation_type == "sum":
                    select_clause.append(f"SUM({metric.table_name}.{metric.column_name}) AS {metric.name}")
                elif metric.calculation_type == "avg":
                    select_clause.append(f"AVG({metric.table_name}.{metric.column_name}) AS {metric.name}")
                elif metric.calculation_type == "count":
                    select_clause.append(f"COUNT({metric.table_name}.{metric.column_name}) AS {metric.name}")
                elif metric.calculation_type == "min":
                    select_clause.append(f"MIN({metric.table_name}.{metric.column_name}) AS {metric.name}")
                elif metric.calculation_type == "max":
                    select_clause.append(f"MAX({metric.table_name}.{metric.column_name}) AS {metric.name}")
                else:
                    select_clause.append(f"{metric.table_name}.{metric.column_name} AS {metric.name}")
            else:  # CalculatedMetric
                select_clause.append(f"({metric.formula}) AS {metric.name}")
        
        # GROUP BY clause
        group_by_clause = []
        for dim in dimensions:
            group_by_clause.append(f"{dim.table_name}.{dim.column_name}")
        
        # WHERE clause
        where_clause = []
        if configuration.get("filters"):
            for filter_item in configuration["filters"]:
                column = filter_item["column"]
                operator = filter_item["operator"]
                value = filter_item["value"]
                
                # Replace with parameter if provided
                if params and column in params:
                    value = params[column]
                
                if operator == "equals":
                    where_clause.append(f"{column} = '{value}'")
                elif operator == "not_equals":
                    where_clause.append(f"{column} <> '{value}'")
                elif operator == "greater_than":
                    where_clause.append(f"{column} > {value}")
                elif operator == "less_than":
                    where_clause.append(f"{column} < {value}")
                elif operator == "contains":
                    where_clause.append(f"{column} LIKE '%{value}%'")
                # Add more operators as needed
        
        # ORDER BY clause
        order_by_clause = []
        if configuration.get("sorting"):
            for sort_item in configuration["sorting"]:
                column = sort_item["column"]
                direction = sort_item["direction"].upper()
                order_by_clause.append(f"{column} {direction}")
        
        # Build the complete query
        query = f"SELECT {', '.join(select_clause)} FROM {table_name}"
        
        if where_clause:
            query += f" WHERE {' AND '.join(where_clause)}"
        
        if group_by_clause:
            query += f" GROUP BY {', '.join(group_by_clause)}"
        
        if order_by_clause:
            query += f" ORDER BY {', '.join(order_by_clause)}"
        
        # Add limit and offset
        if configuration.get("limit"):
            query += f" LIMIT {configuration['limit']}"
        
        if configuration.get("offset"):
            query += f" OFFSET {configuration['offset']}"
        
        return query
    
    async def _execute_api_report(
        self,
        connection_config,
        dimensions: List[Dimension],
        metrics: List[Union[Metric, CalculatedMetric]],
        configuration: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a report against an API."""
        # This is a placeholder implementation
        # In a real implementation, you would make HTTP requests to the API
        return {
            "columns": ["dimension1", "metric1"],
            "rows": [
                {"dimension1": "value1", "metric1": 100},
                {"dimension1": "value2", "metric1": 200},
            ]
        }
    
    async def _execute_file_report(
        self,
        connection_config,
        dimensions: List[Dimension],
        metrics: List[Union[Metric, CalculatedMetric]],
        configuration: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a report against a file."""
        # This is a placeholder implementation
        # In a real implementation, you would read and parse files
        return {
            "columns": ["dimension1", "metric1"],
            "rows": [
                {"dimension1": "value1", "metric1": 100},
                {"dimension1": "value2", "metric1": 200},
            ]
        }
    
    def close(self):
        """Close all connections."""
        for connection_id, connection in self.connections.items():
            try:
                if hasattr(connection, "dispose"):
                    connection.dispose()
                elif hasattr(connection, "close"):
                    connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection {connection_id}: {str(e)}")
        
        self.connections.clear()


# Factory function to create a report service
def get_report_service(db: Session = None) -> ReportService:
    """Get a report service instance."""
    if db is None:
        db = next(get_db())
    return ReportService(db)