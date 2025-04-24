"""
Tool recommender for the Agentor framework.

This module provides a tool that recommends appropriate tools based on task descriptions,
using semantic similarity and keyword matching to find the most relevant tools.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import asyncio
import re
from collections import defaultdict

from agentor.agents.enhanced_tools import EnhancedTool, ToolResult
from agentor.core.interfaces.tool import ITool, IToolRegistry
from agentor.components.memory.embedding import OpenAIEmbeddingProvider, MockEmbeddingProvider

logger = logging.getLogger(__name__)


class ToolRecommender(EnhancedTool):
    """Recommends appropriate tools based on task description.
    
    This tool analyzes a task description and recommends the most appropriate
    tools from the available tool registry based on semantic similarity.
    
    Examples:
        >>> recommender = ToolRecommender(tool_registry)
        >>> result = await recommender.run(task="Find the weather in New York")
        >>> print(result.data["recommended_tools"])
        ['weather', 'location']
    """
    
    def __init__(
        self,
        tool_registry: IToolRegistry,
        embedding_provider: Optional[Any] = None,
        name: str = "tool_recommender",
        description: str = "Recommends appropriate tools based on task description",
        version: str = "1.0.0",
        use_semantic_matching: bool = True,
        use_keyword_matching: bool = True,
        max_recommendations: int = 5,
        similarity_threshold: float = 0.6
    ):
        """Initialize the tool recommender.
        
        Args:
            tool_registry: The tool registry to recommend from
            embedding_provider: Optional embedding provider for semantic matching
            name: The name of the tool
            description: A description of what the tool does
            version: The version of the tool
            use_semantic_matching: Whether to use semantic matching
            use_keyword_matching: Whether to use keyword matching
            max_recommendations: Maximum number of recommendations to return
            similarity_threshold: Minimum similarity score for semantic matching
        """
        super().__init__(
            name=name,
            description=description,
            version=version
        )
        self.tool_registry = tool_registry
        self.use_semantic_matching = use_semantic_matching
        self.use_keyword_matching = use_keyword_matching
        self.max_recommendations = max_recommendations
        self.similarity_threshold = similarity_threshold
        
        # Set up embedding provider if semantic matching is enabled
        if use_semantic_matching:
            if embedding_provider:
                self.embedding_provider = embedding_provider
            else:
                # Use a mock embedding provider by default
                self.embedding_provider = MockEmbeddingProvider(dimension=384)
                logger.info("Using mock embedding provider for tool recommendations")
        else:
            self.embedding_provider = None
        
        # Cache for tool embeddings
        self._tool_embeddings: Dict[str, List[float]] = {}
        
        # Cache for tool keywords
        self._tool_keywords: Dict[str, Set[str]] = {}
        
        # Initialize caches
        self._initialize_caches()
    
    def _initialize_caches(self):
        """Initialize the tool embedding and keyword caches."""
        # Get all tools
        tools = self.tool_registry.get_tools()
        
        # Process each tool
        for name, tool in tools.items():
            # Skip self to avoid recursion
            if name == self.name:
                continue
            
            # Extract keywords from tool name and description
            self._tool_keywords[name] = self._extract_keywords(
                f"{tool.name} {tool.description}"
            )
            
            # Generate embeddings if semantic matching is enabled
            if self.use_semantic_matching and self.embedding_provider:
                # This will be populated on first use
                self._tool_embeddings[name] = None
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            A set of keywords
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
            'will', 'don', 'should', 'now', 'get', 'with', 'by'
        }
        
        # Return filtered keywords
        return {word for word in words if word not in stop_words and len(word) > 2}
    
    async def _get_tool_embedding(self, tool_name: str) -> List[float]:
        """Get the embedding for a tool.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            The tool embedding
        """
        # Check if we already have the embedding
        if tool_name in self._tool_embeddings and self._tool_embeddings[tool_name] is not None:
            return self._tool_embeddings[tool_name]
        
        # Get the tool
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            logger.warning(f"Tool {tool_name} not found in registry")
            return []
        
        # Generate the embedding
        text = f"{tool.name} {tool.description}"
        embedding = await self.embedding_provider.get_embedding(text)
        
        # Cache the embedding
        self._tool_embeddings[tool_name] = embedding
        
        return embedding
    
    async def _get_task_embedding(self, task: str) -> List[float]:
        """Get the embedding for a task.
        
        Args:
            task: The task description
            
        Returns:
            The task embedding
        """
        return await self.embedding_provider.get_embedding(task)
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate the cosine similarity between two embeddings.
        
        Args:
            embedding1: The first embedding
            embedding2: The second embedding
            
        Returns:
            The cosine similarity
        """
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        # Calculate cosine similarity
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def _get_semantic_recommendations(self, task: str) -> List[Tuple[str, float]]:
        """Get tool recommendations based on semantic similarity.
        
        Args:
            task: The task description
            
        Returns:
            A list of (tool_name, similarity_score) tuples
        """
        # Get the task embedding
        task_embedding = await self._get_task_embedding(task)
        
        # Calculate similarity with each tool
        similarities = []
        for tool_name in self._tool_embeddings:
            # Skip self to avoid recursion
            if tool_name == self.name:
                continue
            
            # Get the tool embedding
            tool_embedding = await self._get_tool_embedding(tool_name)
            
            # Calculate similarity
            similarity = self._calculate_similarity(task_embedding, tool_embedding)
            
            # Add to results if above threshold
            if similarity >= self.similarity_threshold:
                similarities.append((tool_name, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:self.max_recommendations]
    
    def _get_keyword_recommendations(self, task: str) -> List[Tuple[str, float]]:
        """Get tool recommendations based on keyword matching.
        
        Args:
            task: The task description
            
        Returns:
            A list of (tool_name, match_score) tuples
        """
        # Extract keywords from the task
        task_keywords = self._extract_keywords(task)
        
        # Calculate match score with each tool
        matches = []
        for tool_name, tool_keywords in self._tool_keywords.items():
            # Skip self to avoid recursion
            if tool_name == self.name:
                continue
            
            # Calculate intersection
            common_keywords = task_keywords.intersection(tool_keywords)
            
            # Calculate match score (0.0 to 1.0)
            if not task_keywords:
                match_score = 0.0
            else:
                match_score = len(common_keywords) / len(task_keywords)
            
            # Add to results if there are any matches
            if common_keywords:
                matches.append((tool_name, match_score))
        
        # Sort by match score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:self.max_recommendations]
    
    async def _combine_recommendations(
        self,
        semantic_recommendations: List[Tuple[str, float]],
        keyword_recommendations: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword recommendations.
        
        Args:
            semantic_recommendations: Recommendations from semantic matching
            keyword_recommendations: Recommendations from keyword matching
            
        Returns:
            A list of recommendation dictionaries
        """
        # Combine scores
        combined_scores = defaultdict(float)
        
        # Add semantic scores (weight: 0.7)
        for tool_name, score in semantic_recommendations:
            combined_scores[tool_name] += score * 0.7
        
        # Add keyword scores (weight: 0.3)
        for tool_name, score in keyword_recommendations:
            combined_scores[tool_name] += score * 0.3
        
        # Convert to list and sort
        combined = [(tool_name, score) for tool_name, score in combined_scores.items()]
        combined.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max recommendations
        combined = combined[:self.max_recommendations]
        
        # Convert to dictionaries with tool details
        result = []
        for tool_name, score in combined:
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                result.append({
                    "name": tool_name,
                    "description": tool.description,
                    "score": round(score, 3),
                    "version": getattr(tool, "version", "1.0.0"),
                    "schema": tool.get_schema()
                })
        
        return result
    
    async def run(
        self,
        task: str,
        max_recommendations: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> ToolResult:
        """Recommend tools for a given task.
        
        Args:
            task: The task description
            max_recommendations: Optional override for max recommendations
            similarity_threshold: Optional override for similarity threshold
            
        Returns:
            ToolResult with recommended tools
        """
        try:
            # Apply overrides if provided
            max_recommendations = max_recommendations or self.max_recommendations
            similarity_threshold = similarity_threshold or self.similarity_threshold
            
            # Get recommendations
            semantic_recommendations = []
            keyword_recommendations = []
            
            # Get semantic recommendations if enabled
            if self.use_semantic_matching and self.embedding_provider:
                semantic_recommendations = await self._get_semantic_recommendations(task)
            
            # Get keyword recommendations if enabled
            if self.use_keyword_matching:
                keyword_recommendations = self._get_keyword_recommendations(task)
            
            # Combine recommendations
            recommendations = await self._combine_recommendations(
                semantic_recommendations,
                keyword_recommendations
            )
            
            # Return the recommendations
            return ToolResult(
                success=True,
                data={
                    "task": task,
                    "recommended_tools": [r["name"] for r in recommendations],
                    "recommendations": recommendations,
                    "count": len(recommendations)
                }
            )
        except Exception as e:
            logger.error(f"Error recommending tools: {str(e)}")
            return ToolResult(
                success=False,
                error=f"Error recommending tools: {str(e)}"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for the tool parameters.
        
        Returns:
            A dictionary describing the parameters for the tool
        """
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description to recommend tools for"
                },
                "max_recommendations": {
                    "type": "integer",
                    "description": "Maximum number of recommendations to return",
                    "default": self.max_recommendations
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Minimum similarity score for semantic matching",
                    "default": self.similarity_threshold
                }
            },
            "required": ["task"]
        }
