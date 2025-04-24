"""
MySQL query optimizer for the Agentor framework.

This module provides a specialized optimizer for MySQL queries.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
import weakref

from ..config import QueryOptimizationConfig, OptimizationLevel

logger = logging.getLogger(__name__)


class MySqlQueryOptimizer:
    """MySQL query optimizer with advanced features."""
    
    def __init__(
        self,
        name: str,
        config: QueryOptimizationConfig,
        connection_func: Optional[Callable[[], Any]] = None
    ):
        """Initialize the MySQL query optimizer.
        
        Args:
            name: The name of the optimizer
            config: The query optimization configuration
            connection_func: Function to get a database connection
        """
        self.name = name
        self.config = config
        self.connection_func = connection_func
        
        # Query patterns
        self.select_pattern = re.compile(r'^\s*SELECT\s+', re.IGNORECASE)
        self.from_pattern = re.compile(r'\s+FROM\s+', re.IGNORECASE)
        self.where_pattern = re.compile(r'\s+WHERE\s+', re.IGNORECASE)
        self.order_by_pattern = re.compile(r'\s+ORDER\s+BY\s+', re.IGNORECASE)
        self.group_by_pattern = re.compile(r'\s+GROUP\s+BY\s+', re.IGNORECASE)
        self.limit_pattern = re.compile(r'\s+LIMIT\s+', re.IGNORECASE)
        self.join_pattern = re.compile(r'\s+JOIN\s+', re.IGNORECASE)
        
        # Query cache
        self.query_cache: Dict[str, str] = {}
        self.query_plan_cache: Dict[str, Dict[str, Any]] = {}
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        
        # Optimizer metrics
        self.metrics = {
            "created_at": time.time(),
            "last_activity": time.time(),
            "total_queries": 0,
            "optimized_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "slow_queries": 0,
            "very_slow_queries": 0
        }
        
        # Optimizer lock
        self.optimizer_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the query optimizer."""
        logger.info(f"Initialized MySQL query optimizer {self.name}")
    
    async def close(self) -> None:
        """Close the query optimizer."""
        # Clear caches
        self.query_cache.clear()
        self.query_plan_cache.clear()
        self.query_stats.clear()
        
        logger.info(f"Closed MySQL query optimizer {self.name}")
    
    async def optimize_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Optimize a query.
        
        Args:
            query: The query to optimize
            params: The query parameters
            
        Returns:
            The optimized query
        """
        # Update metrics
        self.metrics["total_queries"] += 1
        self.metrics["last_activity"] = time.time()
        
        # Check if the query is in the cache
        cache_key = self._get_cache_key(query, params)
        if cache_key in self.query_cache:
            # Update metrics
            self.metrics["cache_hits"] += 1
            
            return self.query_cache[cache_key]
        
        # Update metrics
        self.metrics["cache_misses"] += 1
        
        # Check if the query is a SELECT query
        if not self.select_pattern.match(query):
            return query
        
        # Apply optimizations based on the level
        optimized_query = query
        
        if self.config.level in [OptimizationLevel.BASIC, OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE]:
            # Basic optimizations
            optimized_query = self._optimize_select_clause(optimized_query)
            
            if self.config.optimize_where_clauses:
                optimized_query = self._optimize_where_clause(optimized_query)
        
        if self.config.level in [OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE]:
            # Moderate optimizations
            if self.config.optimize_joins:
                optimized_query = self._optimize_joins(optimized_query)
            
            if self.config.optimize_order_by:
                optimized_query = self._optimize_order_by(optimized_query)
            
            if self.config.optimize_group_by:
                optimized_query = self._optimize_group_by(optimized_query)
        
        if self.config.level == OptimizationLevel.AGGRESSIVE:
            # Aggressive optimizations
            if self.config.optimize_limit:
                optimized_query = self._optimize_limit(optimized_query)
            
            if self.config.add_missing_indexes:
                optimized_query = self._add_index_hints(optimized_query)
        
        # Check if the query was optimized
        if optimized_query != query:
            # Update metrics
            self.metrics["optimized_queries"] += 1
            
            # Cache the optimized query
            self.query_cache[cache_key] = optimized_query
        
        return optimized_query
    
    async def analyze_query(self, query: str, params: Optional[Dict[str, Any]] = None, execution_time: float = 0.0) -> Dict[str, Any]:
        """Analyze a query.
        
        Args:
            query: The query to analyze
            params: The query parameters
            execution_time: The execution time of the query in seconds
            
        Returns:
            Dictionary of analysis results
        """
        # Check if query analysis is enabled
        if not self.config.analyze_queries:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Check if the query is slow
        is_slow = execution_time >= self.config.slow_query_threshold
        is_very_slow = execution_time >= self.config.very_slow_query_threshold
        
        if is_slow:
            self.metrics["slow_queries"] += 1
        
        if is_very_slow:
            self.metrics["very_slow_queries"] += 1
        
        # Get the query plan
        query_plan = {}
        if self.config.collect_query_plans:
            query_plan = await self.get_query_plan(query, params)
        
        # Create the analysis results
        analysis = {
            "query": query,
            "execution_time": execution_time,
            "is_slow": is_slow,
            "is_very_slow": is_very_slow,
            "query_plan": query_plan
        }
        
        # Add recommendations
        recommendations = []
        
        if is_slow:
            # Check if the query has a WHERE clause
            if not self.where_pattern.search(query):
                recommendations.append("Add a WHERE clause to filter the results")
            
            # Check if the query has a LIMIT clause
            if not self.limit_pattern.search(query):
                recommendations.append("Add a LIMIT clause to limit the number of results")
            
            # Check if the query has multiple JOINs
            join_count = len(self.join_pattern.findall(query))
            if join_count > 2:
                recommendations.append(f"Reduce the number of JOINs ({join_count})")
            
            # Check if the query plan indicates missing indexes
            if query_plan and "missing_indexes" in query_plan:
                for index in query_plan["missing_indexes"]:
                    recommendations.append(f"Add an index on {index}")
        
        analysis["recommendations"] = recommendations
        
        # Update query stats
        cache_key = self._get_cache_key(query, params)
        
        async with self.optimizer_lock:
            if cache_key not in self.query_stats:
                self.query_stats[cache_key] = {
                    "count": 0,
                    "total_time": 0.0,
                    "min_time": float('inf'),
                    "max_time": 0.0,
                    "avg_time": 0.0,
                    "slow_count": 0,
                    "very_slow_count": 0,
                    "last_executed": 0.0
                }
            
            stats = self.query_stats[cache_key]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["min_time"] = min(stats["min_time"], execution_time)
            stats["max_time"] = max(stats["max_time"], execution_time)
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["last_executed"] = time.time()
            
            if is_slow:
                stats["slow_count"] += 1
            
            if is_very_slow:
                stats["very_slow_count"] += 1
        
        # Add query stats to the analysis
        analysis["stats"] = stats
        
        return analysis
    
    async def get_query_plan(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the query plan for a query.
        
        Args:
            query: The query to get the plan for
            params: The query parameters
            
        Returns:
            Dictionary of query plan
        """
        # Check if query plan collection is enabled
        if not self.config.collect_query_plans:
            return {}
        
        # Update metrics
        self.metrics["last_activity"] = time.time()
        
        # Check if the query plan is in the cache
        cache_key = self._get_cache_key(query, params)
        if cache_key in self.query_plan_cache:
            return self.query_plan_cache[cache_key]
        
        # Check if we have a connection function
        if not self.connection_func:
            return {}
        
        try:
            # Get a connection
            conn = self.connection_func()
            
            # Execute EXPLAIN
            async with conn.cursor() as cursor:
                # Convert the query to EXPLAIN
                explain_query = f"EXPLAIN {query}"
                
                # Execute the EXPLAIN query
                await cursor.execute(explain_query, params or {})
                
                # Fetch the results
                results = await cursor.fetchall()
                
                # Convert the results to a dictionary
                plan = []
                for row in results:
                    plan_row = {}
                    for i, column in enumerate(cursor.description):
                        plan_row[column[0]] = row[i]
                    plan.append(plan_row)
                
                # Execute EXPLAIN FORMAT=JSON
                explain_json_query = f"EXPLAIN FORMAT=JSON {query}"
                
                try:
                    # Execute the EXPLAIN JSON query
                    await cursor.execute(explain_json_query, params or {})
                    
                    # Fetch the results
                    json_results = await cursor.fetchall()
                    
                    # Parse the JSON results
                    import json
                    json_plan = json.loads(json_results[0][0])
                except Exception:
                    # EXPLAIN FORMAT=JSON not supported
                    json_plan = {}
                
                # Create the query plan
                query_plan = {
                    "plan": plan,
                    "json_plan": json_plan
                }
                
                # Analyze the query plan
                if self.config.analyze_query_plans:
                    # Check for missing indexes
                    missing_indexes = []
                    
                    for row in plan:
                        # Check if the row indicates a table scan
                        if row.get("type") in ["ALL", "index"]:
                            # Check if there's a possible key
                            possible_key = row.get("possible_keys")
                            if not possible_key:
                                # Get the table name
                                table = row.get("table")
                                
                                # Get the columns in the WHERE clause
                                where_columns = self._extract_where_columns(query)
                                
                                # Add the missing index
                                if table and where_columns:
                                    missing_indexes.append(f"{table}({', '.join(where_columns)})")
                    
                    query_plan["missing_indexes"] = missing_indexes
                
                # Cache the query plan
                self.query_plan_cache[cache_key] = query_plan
                
                return query_plan
        except Exception as e:
            logger.error(f"Error getting query plan: {e}")
            return {}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get query optimizer metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics
    
    def _get_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Get a cache key for a query.
        
        Args:
            query: The query
            params: The query parameters
            
        Returns:
            The cache key
        """
        # Normalize the query
        normalized_query = re.sub(r'\s+', ' ', query.strip())
        
        # Create the cache key
        if params:
            import hashlib
            import json
            
            # Convert params to a string
            params_str = json.dumps(params, sort_keys=True)
            
            # Create a hash of the params
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            
            return f"{normalized_query}_{params_hash}"
        else:
            return normalized_query
    
    def _optimize_select_clause(self, query: str) -> str:
        """Optimize the SELECT clause of a query.
        
        Args:
            query: The query to optimize
            
        Returns:
            The optimized query
        """
        # Check if the query has a SELECT * clause
        select_star_pattern = re.compile(r'^\s*SELECT\s+\*\s+FROM', re.IGNORECASE)
        if select_star_pattern.match(query):
            # Get the columns used in the WHERE, ORDER BY, and GROUP BY clauses
            where_columns = self._extract_where_columns(query)
            order_by_columns = self._extract_order_by_columns(query)
            group_by_columns = self._extract_group_by_columns(query)
            
            # Combine the columns
            columns = set(where_columns + order_by_columns + group_by_columns)
            
            # If there are columns, replace SELECT * with SELECT columns
            if columns:
                # Replace SELECT * with SELECT columns
                return re.sub(r'^\s*SELECT\s+\*', f"SELECT {', '.join(columns)}", query, flags=re.IGNORECASE)
        
        return query
    
    def _optimize_where_clause(self, query: str) -> str:
        """Optimize the WHERE clause of a query.
        
        Args:
            query: The query to optimize
            
        Returns:
            The optimized query
        """
        # Check if the query has a WHERE clause
        where_match = self.where_pattern.search(query)
        if not where_match:
            return query
        
        # Get the WHERE clause
        where_start = where_match.end()
        
        # Find the end of the WHERE clause
        group_by_match = self.group_by_pattern.search(query, where_start)
        order_by_match = self.order_by_pattern.search(query, where_start)
        limit_match = self.limit_pattern.search(query, where_start)
        
        where_end = len(query)
        if group_by_match:
            where_end = min(where_end, group_by_match.start())
        if order_by_match:
            where_end = min(where_end, order_by_match.start())
        if limit_match:
            where_end = min(where_end, limit_match.start())
        
        where_clause = query[where_start:where_end].strip()
        
        # Optimize the WHERE clause
        optimized_where_clause = where_clause
        
        # Replace LIKE '%...%' with LIKE '...%' if possible
        like_pattern = re.compile(r'(\w+)\s+LIKE\s+[\'"]%([^%]+)%[\'"]', re.IGNORECASE)
        optimized_where_clause = like_pattern.sub(r'\1 LIKE "\2%"', optimized_where_clause)
        
        # Replace != with NOT IN for better index usage
        not_equal_pattern = re.compile(r'(\w+)\s+!=\s+([\'"]?\w+[\'"]?)', re.IGNORECASE)
        optimized_where_clause = not_equal_pattern.sub(r'\1 NOT IN (\2)', optimized_where_clause)
        
        # Replace the WHERE clause in the query
        if optimized_where_clause != where_clause:
            return query[:where_start] + optimized_where_clause + query[where_end:]
        
        return query
    
    def _optimize_joins(self, query: str) -> str:
        """Optimize the JOINs in a query.
        
        Args:
            query: The query to optimize
            
        Returns:
            The optimized query
        """
        # Check if the query has JOINs
        if not self.join_pattern.search(query):
            return query
        
        # Replace comma joins with INNER JOIN
        comma_join_pattern = re.compile(r'FROM\s+(\w+),\s+(\w+)\s+WHERE\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', re.IGNORECASE)
        optimized_query = comma_join_pattern.sub(r'FROM \1 INNER JOIN \2 ON \3.\4 = \5.\6 WHERE', query)
        
        # Replace LEFT JOIN with INNER JOIN if the joined table is used in the WHERE clause
        left_join_pattern = re.compile(r'LEFT\s+JOIN\s+(\w+)\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', re.IGNORECASE)
        left_joins = left_join_pattern.findall(optimized_query)
        
        for left_join in left_joins:
            table = left_join[0]
            
            # Check if the table is used in the WHERE clause
            where_match = self.where_pattern.search(optimized_query)
            if where_match:
                where_start = where_match.end()
                
                # Find the end of the WHERE clause
                group_by_match = self.group_by_pattern.search(optimized_query, where_start)
                order_by_match = self.order_by_pattern.search(optimized_query, where_start)
                limit_match = self.limit_pattern.search(optimized_query, where_start)
                
                where_end = len(optimized_query)
                if group_by_match:
                    where_end = min(where_end, group_by_match.start())
                if order_by_match:
                    where_end = min(where_end, order_by_match.start())
                if limit_match:
                    where_end = min(where_end, limit_match.start())
                
                where_clause = optimized_query[where_start:where_end].strip()
                
                # Check if the table is used in the WHERE clause
                table_pattern = re.compile(rf'{table}\.', re.IGNORECASE)
                if table_pattern.search(where_clause):
                    # Replace LEFT JOIN with INNER JOIN
                    optimized_query = optimized_query.replace(f"LEFT JOIN {table}", f"INNER JOIN {table}")
        
        return optimized_query
    
    def _optimize_order_by(self, query: str) -> str:
        """Optimize the ORDER BY clause of a query.
        
        Args:
            query: The query to optimize
            
        Returns:
            The optimized query
        """
        # Check if the query has an ORDER BY clause
        order_by_match = self.order_by_pattern.search(query)
        if not order_by_match:
            return query
        
        # Get the ORDER BY clause
        order_by_start = order_by_match.end()
        
        # Find the end of the ORDER BY clause
        limit_match = self.limit_pattern.search(query, order_by_start)
        
        order_by_end = len(query)
        if limit_match:
            order_by_end = limit_match.start()
        
        order_by_clause = query[order_by_start:order_by_end].strip()
        
        # Check if the query has a LIMIT clause
        limit_match = self.limit_pattern.search(query)
        if limit_match:
            # Get the LIMIT clause
            limit_start = limit_match.end()
            limit_clause = query[limit_start:].strip()
            
            # Check if the LIMIT is 1
            if limit_clause == "1":
                # Replace ORDER BY with ORDER BY ... LIMIT 1
                return query[:order_by_start] + order_by_clause + " LIMIT 1"
        
        return query
    
    def _optimize_group_by(self, query: str) -> str:
        """Optimize the GROUP BY clause of a query.
        
        Args:
            query: The query to optimize
            
        Returns:
            The optimized query
        """
        # Check if the query has a GROUP BY clause
        group_by_match = self.group_by_pattern.search(query)
        if not group_by_match:
            return query
        
        # Get the GROUP BY clause
        group_by_start = group_by_match.end()
        
        # Find the end of the GROUP BY clause
        order_by_match = self.order_by_pattern.search(query, group_by_start)
        limit_match = self.limit_pattern.search(query, group_by_start)
        
        group_by_end = len(query)
        if order_by_match:
            group_by_end = order_by_match.start()
        if limit_match:
            group_by_end = min(group_by_end, limit_match.start())
        
        group_by_clause = query[group_by_start:group_by_end].strip()
        
        # Optimize the GROUP BY clause
        optimized_group_by_clause = group_by_clause
        
        # Replace GROUP BY with GROUP BY ... WITH ROLLUP if appropriate
        if "," in optimized_group_by_clause and "WITH ROLLUP" not in optimized_group_by_clause:
            # Check if the query has aggregate functions
            aggregate_pattern = re.compile(r'(SUM|COUNT|AVG|MIN|MAX)\s*\(', re.IGNORECASE)
            if aggregate_pattern.search(query):
                optimized_group_by_clause += " WITH ROLLUP"
        
        # Replace the GROUP BY clause in the query
        if optimized_group_by_clause != group_by_clause:
            return query[:group_by_start] + optimized_group_by_clause + query[group_by_end:]
        
        return query
    
    def _optimize_limit(self, query: str) -> str:
        """Optimize the LIMIT clause of a query.
        
        Args:
            query: The query to optimize
            
        Returns:
            The optimized query
        """
        # Check if the query has a LIMIT clause
        limit_match = self.limit_pattern.search(query)
        if limit_match:
            return query
        
        # Add a LIMIT clause if the query doesn't have one
        return query + " LIMIT 1000"
    
    def _add_index_hints(self, query: str) -> str:
        """Add index hints to a query.
        
        Args:
            query: The query to add index hints to
            
        Returns:
            The query with index hints
        """
        # Check if the query has a FROM clause
        from_match = self.from_pattern.search(query)
        if not from_match:
            return query
        
        # Get the FROM clause
        from_start = from_match.end()
        
        # Find the end of the FROM clause
        where_match = self.where_pattern.search(query, from_start)
        join_match = self.join_pattern.search(query, from_start)
        
        from_end = len(query)
        if where_match:
            from_end = where_match.start()
        if join_match:
            from_end = min(from_end, join_match.start())
        
        from_clause = query[from_start:from_end].strip()
        
        # Check if the FROM clause has a table
        table_match = re.match(r'^(\w+)(?:\s+(\w+))?$', from_clause)
        if not table_match:
            return query
        
        # Get the table name and alias
        table_name = table_match.group(1)
        table_alias = table_match.group(2) or table_name
        
        # Get the columns in the WHERE clause
        where_columns = self._extract_where_columns(query)
        
        # Check if there are columns in the WHERE clause
        if not where_columns:
            return query
        
        # Add index hints
        index_hints = []
        for column in where_columns:
            # Check if the column belongs to the table
            if "." in column:
                table, col = column.split(".")
                if table == table_alias:
                    index_hints.append(col)
            else:
                index_hints.append(column)
        
        # Check if there are index hints
        if not index_hints:
            return query
        
        # Add the index hints to the query
        optimized_from_clause = f"{from_clause} USE INDEX ({', '.join(index_hints)})"
        
        # Replace the FROM clause in the query
        return query[:from_start] + optimized_from_clause + query[from_end:]
    
    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract columns from the WHERE clause of a query.
        
        Args:
            query: The query to extract columns from
            
        Returns:
            List of columns
        """
        # Check if the query has a WHERE clause
        where_match = self.where_pattern.search(query)
        if not where_match:
            return []
        
        # Get the WHERE clause
        where_start = where_match.end()
        
        # Find the end of the WHERE clause
        group_by_match = self.group_by_pattern.search(query, where_start)
        order_by_match = self.order_by_pattern.search(query, where_start)
        limit_match = self.limit_pattern.search(query, where_start)
        
        where_end = len(query)
        if group_by_match:
            where_end = min(where_end, group_by_match.start())
        if order_by_match:
            where_end = min(where_end, order_by_match.start())
        if limit_match:
            where_end = min(where_end, limit_match.start())
        
        where_clause = query[where_start:where_end].strip()
        
        # Extract columns from the WHERE clause
        column_pattern = re.compile(r'(\w+(?:\.\w+)?)\s*(?:=|!=|<>|>|<|>=|<=|LIKE|IN|NOT IN|IS|IS NOT)', re.IGNORECASE)
        columns = column_pattern.findall(where_clause)
        
        return columns
    
    def _extract_order_by_columns(self, query: str) -> List[str]:
        """Extract columns from the ORDER BY clause of a query.
        
        Args:
            query: The query to extract columns from
            
        Returns:
            List of columns
        """
        # Check if the query has an ORDER BY clause
        order_by_match = self.order_by_pattern.search(query)
        if not order_by_match:
            return []
        
        # Get the ORDER BY clause
        order_by_start = order_by_match.end()
        
        # Find the end of the ORDER BY clause
        limit_match = self.limit_pattern.search(query, order_by_start)
        
        order_by_end = len(query)
        if limit_match:
            order_by_end = limit_match.start()
        
        order_by_clause = query[order_by_start:order_by_end].strip()
        
        # Extract columns from the ORDER BY clause
        column_pattern = re.compile(r'(\w+(?:\.\w+)?)\s*(?:ASC|DESC)?', re.IGNORECASE)
        columns = column_pattern.findall(order_by_clause)
        
        return columns
    
    def _extract_group_by_columns(self, query: str) -> List[str]:
        """Extract columns from the GROUP BY clause of a query.
        
        Args:
            query: The query to extract columns from
            
        Returns:
            List of columns
        """
        # Check if the query has a GROUP BY clause
        group_by_match = self.group_by_pattern.search(query)
        if not group_by_match:
            return []
        
        # Get the GROUP BY clause
        group_by_start = group_by_match.end()
        
        # Find the end of the GROUP BY clause
        order_by_match = self.order_by_pattern.search(query, group_by_start)
        limit_match = self.limit_pattern.search(query, group_by_start)
        
        group_by_end = len(query)
        if order_by_match:
            group_by_end = order_by_match.start()
        if limit_match:
            group_by_end = min(group_by_end, limit_match.start())
        
        group_by_clause = query[group_by_start:group_by_end].strip()
        
        # Extract columns from the GROUP BY clause
        columns = [col.strip() for col in group_by_clause.split(",")]
        
        return columns
