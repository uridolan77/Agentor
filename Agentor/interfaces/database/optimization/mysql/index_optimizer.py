"""
MySQL index optimizer for the Agentor framework.

This module provides a specialized optimizer for MySQL indexes.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
import weakref

from ..config import IndexOptimizationConfig, OptimizationLevel

logger = logging.getLogger(__name__)


class MySqlIndexOptimizer:
    """MySQL index optimizer with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: IndexOptimizationConfig,
        connection_func: Optional[Callable[[], Any]] = None
    ):
        """Initialize the MySQL index optimizer.
        
        Args:
            name: The name of the optimizer
            config: The index optimization configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Index cache
        self.index_cache: Dict[str, Dict[str, Any]] = {}
        self.table_stats: Dict[str, Dict[str, Any]] = {}
        self.index_stats: Dict[str, Dict[str, Any]] = {}
        
        # Optimizer metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_tables_analyzed": 0,
            "total_indexes_analyzed": 0,
            "recommended_indexes": 0,
            "created_indexes": 0
        }
        
        # Optimizer lock
        self.optimizer_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the index optimizer."""
        logger.info(f"Initialized MySQL index optimizer {self.name}")
    
    async def close(self) -> None:
        """Close the index optimizer."""
        # Clear caches
        self.index_cache.clear()
        self.table_stats.clear()
        self.index_stats.clear()
        
        logger.info(f"Closed MySQL index optimizer {self.name}")
    
    async def optimize_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Optimize indexes for a table.
        
        Args:
            table_name: The name of the table
            
        Returns:
            List of index recommendations
        """
        # Update metrics
        self.metrics["last_activity"] = time.time()
        self.metrics["total_tables_analyzed"] += 1
        
        # Check if we have a connection function
        if not self.connection_func:
            return []
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the table structure
            table_structure = await self._get_table_structure(conn, table_name)
            if not table_structure:
                return []
            
            # Get the existing indexes
            existing_indexes = await self._get_existing_indexes(conn, table_name)
            
            # Get the table statistics
            table_stats = await self._get_table_stats(conn, table_name)
            
            # Get the index statistics
            index_stats = await self._get_index_stats(conn, table_name)
            
            # Get the query patterns
            query_patterns = await self._get_query_patterns(conn, table_name)
            
            # Analyze the indexes
            analysis = await self.analyze_indexes(table_name)
            
            # Generate recommendations
            recommendations = []
            
            # Check for missing primary key
            if not any(index.get("Key_name") == "PRIMARY" for index in existing_indexes):
                # Find a suitable column for a primary key
                primary_key_column = None
                
                for column in table_structure:
                    # Check if the column is NOT NULL and unique
                    if column.get("Null") == "NO" and column.get("Key") == "UNI":
                        primary_key_column = column.get("Field")
                        break
                
                if primary_key_column:
                    recommendations.append({
                        "type": "primary_key",
                        "table": table_name,
                        "columns": [primary_key_column],
                        "reason": "Table does not have a primary key",
                        "sql": f"ALTER TABLE {table_name} ADD PRIMARY KEY ({primary_key_column})"
                    })
            
            # Check for missing indexes on foreign keys
            for column in table_structure:
                # Check if the column is a foreign key
                if "MUL" in column.get("Key", ""):
                    column_name = column.get("Field")
                    
                    # Check if the column already has an index
                    if not any(column_name in index.get("Column_name", "").split(",") for index in existing_indexes):
                        recommendations.append({
                            "type": "foreign_key",
                            "table": table_name,
                            "columns": [column_name],
                            "reason": f"Foreign key column {column_name} does not have an index",
                            "sql": f"ALTER TABLE {table_name} ADD INDEX idx_{table_name}_{column_name} ({column_name})"
                        })
            
            # Check for missing indexes on frequently queried columns
            for pattern in query_patterns:
                # Get the columns in the WHERE clause
                where_columns = pattern.get("where_columns", [])
                
                # Check if the columns already have an index
                for column in where_columns:
                    if not any(column in index.get("Column_name", "").split(",") for index in existing_indexes):
                        recommendations.append({
                            "type": "query_pattern",
                            "table": table_name,
                            "columns": [column],
                            "reason": f"Frequently queried column {column} does not have an index",
                            "sql": f"ALTER TABLE {table_name} ADD INDEX idx_{table_name}_{column} ({column})"
                        })
            
            # Check for composite indexes
            if self.config.recommend_composite_indexes:
                for pattern in query_patterns:
                    # Get the columns in the WHERE clause
                    where_columns = pattern.get("where_columns", [])
                    
                    # Check if there are multiple columns
                    if len(where_columns) > 1:
                        # Check if the columns already have a composite index
                        composite_index_exists = False
                        
                        for index in existing_indexes:
                            index_columns = index.get("Column_name", "").split(",")
                            if all(column in index_columns for column in where_columns):
                                composite_index_exists = True
                                break
                        
                        if not composite_index_exists:
                            columns_str = ", ".join(where_columns)
                            index_name = f"idx_{table_name}_{'_'.join(where_columns)}"
                            
                            recommendations.append({
                                "type": "composite",
                                "table": table_name,
                                "columns": where_columns,
                                "reason": f"Frequently queried columns {columns_str} do not have a composite index",
                                "sql": f"ALTER TABLE {table_name} ADD INDEX {index_name} ({columns_str})"
                            })
            
            # Check for covering indexes
            if self.config.recommend_covering_indexes:
                for pattern in query_patterns:
                    # Get the columns in the WHERE clause
                    where_columns = pattern.get("where_columns", [])
                    
                    # Get the columns in the SELECT clause
                    select_columns = pattern.get("select_columns", [])
                    
                    # Check if there are WHERE and SELECT columns
                    if where_columns and select_columns:
                        # Check if the columns already have a covering index
                        covering_index_exists = False
                        
                        for index in existing_indexes:
                            index_columns = index.get("Column_name", "").split(",")
                            if all(column in index_columns for column in where_columns + select_columns):
                                covering_index_exists = True
                                break
                        
                        if not covering_index_exists:
                            all_columns = where_columns + [col for col in select_columns if col not in where_columns]
                            columns_str = ", ".join(all_columns)
                            index_name = f"idx_{table_name}_covering"
                            
                            recommendations.append({
                                "type": "covering",
                                "table": table_name,
                                "columns": all_columns,
                                "reason": f"Query could benefit from a covering index on {columns_str}",
                                "sql": f"ALTER TABLE {table_name} ADD INDEX {index_name} ({columns_str})"
                            })
            
            # Check for redundant indexes
            redundant_indexes = analysis.get("redundant_indexes", [])
            for index in redundant_indexes:
                recommendations.append({
                    "type": "redundant",
                    "table": table_name,
                    "columns": index.get("columns", []),
                    "reason": f"Index {index.get('name')} is redundant",
                    "sql": f"ALTER TABLE {table_name} DROP INDEX {index.get('name')}"
                })
            
            # Check for unused indexes
            unused_indexes = analysis.get("unused_indexes", [])
            for index in unused_indexes:
                recommendations.append({
                    "type": "unused",
                    "table": table_name,
                    "columns": index.get("columns", []),
                    "reason": f"Index {index.get('name')} is unused",
                    "sql": f"ALTER TABLE {table_name} DROP INDEX {index.get('name')}"
                })
            
            # Limit the number of recommendations
            if len(recommendations) > self.config.max_indexes_per_table:
                recommendations = recommendations[:self.config.max_indexes_per_table]
            
            # Update metrics
            self.metrics["recommended_indexes"] += len(recommendations)
            
            # Create indexes if auto-creation is enabled
            if self.config.auto_create_indexes and recommendations:
                created_indexes = await self._create_indexes(conn, recommendations)
                self.metrics["created_indexes"] += created_indexes
            
            return recommendations
        except Exception as e:
            logger.error(f"Error optimizing indexes for table {table_name}: {e}")
            return []
    
    async def analyze_indexes(self, table_name: str) -> Dict[str, Any]:
        """Analyze indexes for a table.
        
        Args:
            table_name: The name of the table
            
        Returns:
            Dictionary of analysis results
        """
        # Check if index analysis is enabled
        if not self.config.analyze_indexes:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Check if the analysis is in the cache
        if table_name in self.index_cache:
            return self.index_cache[table_name]
        
        # Check if we have a connection function
        if not self.connection_func:
            return {}
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Get the existing indexes
            existing_indexes = await self._get_existing_indexes(conn, table_name)
            self.metrics["total_indexes_analyzed"] += len(existing_indexes)
            
            # Get the index statistics
            index_stats = await self._get_index_stats(conn, table_name)
            
            # Analyze the indexes
            analysis = {
                "table": table_name,
                "total_indexes": len(existing_indexes),
                "indexes": existing_indexes,
                "stats": index_stats,
                "redundant_indexes": [],
                "unused_indexes": [],
                "recommendations": []
            }
            
            # Check for redundant indexes
            redundant_indexes = []
            
            for i, index1 in enumerate(existing_indexes):
                # Skip the primary key
                if index1.get("Key_name") == "PRIMARY":
                    continue
                
                index1_columns = index1.get("Column_name", "").split(",")
                
                for j, index2 in enumerate(existing_indexes):
                    # Skip the same index and the primary key
                    if i == j or index2.get("Key_name") == "PRIMARY":
                        continue
                    
                    index2_columns = index2.get("Column_name", "").split(",")
                    
                    # Check if index1 is a prefix of index2
                    if len(index1_columns) < len(index2_columns):
                        if all(index1_columns[k] == index2_columns[k] for k in range(len(index1_columns))):
                            redundant_indexes.append({
                                "name": index1.get("Key_name"),
                                "columns": index1_columns,
                                "redundant_with": index2.get("Key_name")
                            })
                            break
            
            analysis["redundant_indexes"] = redundant_indexes
            
            # Check for unused indexes
            unused_indexes = []
            
            for index in existing_indexes:
                # Skip the primary key
                if index.get("Key_name") == "PRIMARY":
                    continue
                
                index_name = index.get("Key_name")
                index_columns = index.get("Column_name", "").split(",")
                
                # Check if the index is used
                index_used = False
                
                for stat in index_stats:
                    if stat.get("index_name") == index_name and stat.get("rows_read", 0) > 0:
                        index_used = True
                        break
                
                if not index_used:
                    unused_indexes.append({
                        "name": index_name,
                        "columns": index_columns
                    })
            
            analysis["unused_indexes"] = unused_indexes
            
            # Cache the analysis
            self.index_cache[table_name] = analysis
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing indexes for table {table_name}: {e}")
            return {}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get index optimizer metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    async def _get_table_structure(self, conn: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get the structure of a table.
        
        Args:
            conn: The database connection
            table_name: The name of the table
            
        Returns:
            List of column definitions
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                await cursor.execute(f"DESCRIBE {table_name}")
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                columns = []
                for row in results:
                    column = {}
                    for i, column_name in enumerate(cursor.description):
                        column[column_name[0]] = row[i]
                    columns.append(column)
                
                return columns
        except Exception as e:
            logger.error(f"Error getting table structure for {table_name}: {e}")
            return []
    
    async def _get_existing_indexes(self, conn: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get the existing indexes for a table.
        
        Args:
            conn: The database connection
            table_name: The name of the table
            
        Returns:
            List of index definitions
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                await cursor.execute(f"SHOW INDEX FROM {table_name}")
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                indexes = []
                current_index = None
                
                for row in results:
                    # Create a dictionary for the row
                    index_row = {}
                    for i, column_name in enumerate(cursor.description):
                        index_row[column_name[0]] = row[i]
                    
                    # Check if this is a new index
                    if current_index is None or current_index.get("Key_name") != index_row.get("Key_name"):
                        # Add the current index to the list
                        if current_index is not None:
                            indexes.append(current_index)
                        
                        # Create a new index
                        current_index = {
                            "Table": index_row.get("Table"),
                            "Key_name": index_row.get("Key_name"),
                            "Non_unique": index_row.get("Non_unique"),
                            "Index_type": index_row.get("Index_type"),
                            "Column_name": index_row.get("Column_name")
                        }
                    else:
                        # Add the column to the current index
                        current_index["Column_name"] += f",{index_row.get('Column_name')}"
                
                # Add the last index
                if current_index is not None:
                    indexes.append(current_index)
                
                return indexes
        except Exception as e:
            logger.error(f"Error getting existing indexes for {table_name}: {e}")
            return []
    
    async def _get_table_stats(self, conn: Any, table_name: str) -> Dict[str, Any]:
        """Get statistics for a table.
        
        Args:
            conn: The database connection
            table_name: The name of the table
            
        Returns:
            Dictionary of table statistics
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                await cursor.execute(f"SHOW TABLE STATUS LIKE '{table_name}'")
                
                # Fetch the results
                result = await cursor.fetchone()
                
                # Convert the result to a dictionary
                stats = {}
                for i, column_name in enumerate(cursor.description):
                    stats[column_name[0]] = result[i]
                
                # Store the table stats
                self.table_stats[table_name] = stats
                
                return stats
        except Exception as e:
            logger.error(f"Error getting table statistics for {table_name}: {e}")
            return {}
    
    async def _get_index_stats(self, conn: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get statistics for indexes on a table.
        
        Args:
            conn: The database connection
            table_name: The name of the table
            
        Returns:
            List of index statistics
        """
        try:
            # Check if index statistics collection is enabled
            if not self.config.collect_index_stats:
                return []
            
            # Execute the query
            async with conn.cursor() as cursor:
                # Get the database name
                await cursor.execute("SELECT DATABASE()")
                db_result = await cursor.fetchone()
                db_name = db_result[0]
                
                # Get index statistics from information_schema
                await cursor.execute(f"""
                    SELECT
                        i.INDEX_NAME AS index_name,
                        i.COLUMN_NAME AS column_name,
                        i.SEQ_IN_INDEX AS seq_in_index,
                        i.CARDINALITY AS cardinality,
                        s.ROWS_READ AS rows_read,
                        s.ROWS_INSERTED AS rows_inserted,
                        s.ROWS_UPDATED AS rows_updated,
                        s.ROWS_DELETED AS rows_deleted
                    FROM
                        information_schema.STATISTICS i
                    LEFT JOIN
                        information_schema.INDEX_STATISTICS s
                    ON
                        i.TABLE_SCHEMA = s.INDEX_SCHEMA
                        AND i.TABLE_NAME = s.TABLE_NAME
                        AND i.INDEX_NAME = s.INDEX_NAME
                    WHERE
                        i.TABLE_SCHEMA = %s
                        AND i.TABLE_NAME = %s
                    ORDER BY
                        i.INDEX_NAME, i.SEQ_IN_INDEX
                """, (db_name, table_name))
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                stats = []
                for row in results:
                    stat = {}
                    for i, column_name in enumerate(cursor.description):
                        stat[column_name[0]] = row[i]
                    stats.append(stat)
                
                # Store the index stats
                self.index_stats[table_name] = stats
                
                return stats
        except Exception as e:
            logger.error(f"Error getting index statistics for {table_name}: {e}")
            return []
    
    async def _get_query_patterns(self, conn: Any, table_name: str) -> List[Dict[str, Any]]:
        """Get query patterns for a table.
        
        Args:
            conn: The database connection
            table_name: The name of the table
            
        Returns:
            List of query patterns
        """
        try:
            # Execute the query
            async with conn.cursor() as cursor:
                # Get query patterns from the slow query log
                await cursor.execute(f"""
                    SELECT
                        SUBSTRING_INDEX(SUBSTRING_INDEX(query, 'FROM', -1), 'WHERE', 1) AS table_clause,
                        SUBSTRING_INDEX(query, 'WHERE', -1) AS where_clause,
                        SUBSTRING_INDEX(query, 'SELECT', -1) AS select_clause,
                        COUNT(*) AS count,
                        AVG(query_time) AS avg_time,
                        MAX(query_time) AS max_time
                    FROM
                        mysql.slow_log
                    WHERE
                        query LIKE %s
                        AND query LIKE %s
                    GROUP BY
                        table_clause, where_clause
                    ORDER BY
                        count DESC
                    LIMIT 10
                """, (f"%FROM%{table_name}%", "%WHERE%"))
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to dictionaries
                patterns = []
                for row in results:
                    pattern = {}
                    for i, column_name in enumerate(cursor.description):
                        pattern[column_name[0]] = row[i]
                    
                    # Extract columns from the WHERE clause
                    where_clause = pattern.get("where_clause", "")
                    where_columns = self._extract_where_columns(where_clause)
                    pattern["where_columns"] = where_columns
                    
                    # Extract columns from the SELECT clause
                    select_clause = pattern.get("select_clause", "")
                    select_columns = self._extract_select_columns(select_clause)
                    pattern["select_columns"] = select_columns
                    
                    patterns.append(pattern)
                
                return patterns
        except Exception as e:
            logger.error(f"Error getting query patterns for {table_name}: {e}")
            return []
    
    async def _create_indexes(self, conn: Any, recommendations: List[Dict[str, Any]]) -> int:
        """Create indexes based on recommendations.
        
        Args:
            conn: The database connection
            recommendations: The index recommendations
            
        Returns:
            Number of indexes created
        """
        created_indexes = 0
        
        for recommendation in recommendations:
            # Skip recommendations for dropping indexes
            if recommendation.get("type") in ["redundant", "unused"]:
                continue
            
            try:
                # Execute the SQL statement
                async with conn.cursor() as cursor:
                    await cursor.execute(recommendation.get("sql"))
                
                created_indexes += 1
                logger.info(f"Created index: {recommendation.get('sql')}")
            except Exception as e:
                logger.error(f"Error creating index: {e}")
        
        return created_indexes
    
    def _extract_where_columns(self, where_clause: str) -> List[str]:
        """Extract columns from a WHERE clause.
        
        Args:
            where_clause: The WHERE clause
            
        Returns:
            List of columns
        """
        # Extract columns from the WHERE clause
        column_pattern = re.compile(r'(\w+(?:\.\w+)?)\s*(?:=|!=|<>|>|<|>=|<=|LIKE|IN|NOT IN|IS|IS NOT)', re.IGNORECASE)
        columns = column_pattern.findall(where_clause)
        
        return columns
    
    def _extract_select_columns(self, select_clause: str) -> List[str]:
        """Extract columns from a SELECT clause.
        
        Args:
            select_clause: The SELECT clause
            
        Returns:
            List of columns
        """
        # Extract columns from the SELECT clause
        column_pattern = re.compile(r'(\w+(?:\.\w+)?)\s*(?:,|$|\s)', re.IGNORECASE)
        columns = column_pattern.findall(select_clause)
        
        return columns
