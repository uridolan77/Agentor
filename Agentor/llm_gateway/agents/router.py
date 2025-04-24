import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import openai
import asyncio
import json
from enum import Enum
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Strategy for routing queries."""
    EXACT_MATCH = "exact_match"  # Exact string matching
    KEYWORD = "keyword"  # Keyword-based matching
    SEMANTIC = "semantic"  # Semantic similarity matching
    HYBRID = "hybrid"  # Combination of multiple strategies


@dataclass
class RouteDefinition:
    """Definition of a route."""
    intent: str  # The intent name
    description: str  # Description of the intent
    examples: List[str]  # Example queries for this intent
    keywords: List[str] = None  # Keywords for keyword-based matching
    threshold: float = 0.7  # Similarity threshold for this route
    priority: int = 0  # Priority of this route (higher = more important)
    metadata: Dict[str, Any] = None  # Additional metadata


@dataclass
class RouteMatch:
    """Result of matching a query to a route."""
    intent: str  # The matched intent
    confidence: float  # Confidence score (0-1)
    strategy: RoutingStrategy  # Strategy used for matching
    metadata: Dict[str, Any] = None  # Additional metadata


class SemanticRouter:
    """A router that uses semantic similarity to route queries to the appropriate agent."""

    def __init__(self,
                 embedding_model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 default_threshold: float = 0.7,
                 cache_embeddings: bool = True):
        """Initialize the semantic router.

        Args:
            embedding_model: The name of the embedding model to use
            api_key: The OpenAI API key (optional, can be set in the environment)
            default_threshold: Default similarity threshold for matching
            cache_embeddings: Whether to cache embeddings
        """
        self.embedding_model = embedding_model
        self.default_threshold = default_threshold
        self.cache_embeddings = cache_embeddings
        self.client = openai.AsyncOpenAI(api_key=api_key)

        # Storage for routes and embeddings
        self.routes: Dict[str, RouteDefinition] = {}
        self.route_embeddings: Dict[str, List[float]] = {}
        self.example_embeddings: Dict[str, List[List[float]]] = {}

        # Cache for query embeddings
        self.query_embedding_cache: Dict[str, List[float]] = {}

        # Lock for thread safety
        self.lock = asyncio.Lock()

    async def _get_embedding(self, text: str) -> List[float]:
        """Get an embedding for a text.

        Args:
            text: The text to embed

        Returns:
            The embedding
        """
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.

        Args:
            a: The first vector
            b: The second vector

        Returns:
            The cosine similarity
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    def _keyword_match(self, query: str, keywords: List[str]) -> float:
        """Calculate the keyword match score.

        Args:
            query: The query to match
            keywords: The keywords to match against

        Returns:
            The match score (0-1)
        """
        if not keywords:
            return 0.0

        # Normalize the query
        query = query.lower()

        # Count the number of keywords that appear in the query
        matches = sum(1 for keyword in keywords if keyword.lower() in query)

        # Calculate the score
        return matches / len(keywords)

    async def add_route(self, route: Union[RouteDefinition, Dict[str, Any]]):
        """Add a route to the router.

        Args:
            route: The route definition or a dictionary with route parameters
        """
        # Convert dictionary to RouteDefinition if necessary
        if isinstance(route, dict):
            route = RouteDefinition(**route)

        async with self.lock:
            # Store the route definition
            self.routes[route.intent] = route

            # Get embeddings for the description and examples
            description_embedding = await self._get_embedding(route.description)
            self.route_embeddings[route.intent] = description_embedding

            # Get embeddings for examples if provided
            if route.examples:
                example_embeddings = await self._get_embeddings(route.examples)
                self.example_embeddings[route.intent] = example_embeddings

    async def add_routes(self, routes: List[Union[RouteDefinition, Dict[str, Any]]]):
        """Add multiple routes to the router.

        Args:
            routes: List of route definitions or dictionaries
        """
        for route in routes:
            await self.add_route(route)

    async def route(self, query: str) -> RouteMatch:
        """Route a query to the appropriate intent.

        Args:
            query: The query to route

        Returns:
            The route match result
        """
        if not self.routes:
            raise ValueError("No routes have been added")

        # Get the query embedding
        query_embed = await self._get_embedding(query)

        # Calculate semantic similarities with route descriptions
        description_similarities = {
            intent: self._cosine_similarity(query_embed, embed)
            for intent, embed in self.route_embeddings.items()
        }

        # Calculate semantic similarities with examples
        example_similarities = {}
        for intent, embeddings in self.example_embeddings.items():
            # Find the best matching example
            best_similarity = max(
                self._cosine_similarity(query_embed, example_embed)
                for example_embed in embeddings
            ) if embeddings else 0.0
            example_similarities[intent] = best_similarity

        # Calculate keyword matches
        keyword_scores = {}
        for intent, route in self.routes.items():
            if route.keywords:
                keyword_scores[intent] = self._keyword_match(query, route.keywords)
            else:
                keyword_scores[intent] = 0.0

        # Combine scores using a weighted approach
        combined_scores = {}
        for intent in self.routes.keys():
            # Weights for different matching strategies
            description_weight = 0.4
            example_weight = 0.4
            keyword_weight = 0.2

            # Get scores from different strategies
            description_score = description_similarities.get(intent, 0.0)
            example_score = example_similarities.get(intent, 0.0)
            keyword_score = keyword_scores.get(intent, 0.0)

            # Calculate combined score
            combined_score = (
                description_weight * description_score +
                example_weight * example_score +
                keyword_weight * keyword_score
            )

            # Apply route priority as a multiplier
            priority_multiplier = 1.0 + (self.routes[intent].priority / 10.0)
            combined_scores[intent] = combined_score * priority_multiplier

        # Get the intent with the highest score
        best_intent, best_score = max(combined_scores.items(), key=lambda x: x[1])

        # Check if the score exceeds the threshold
        threshold = self.routes[best_intent].threshold or self.default_threshold
        if best_score < threshold:
            logger.warning(f"No intent matched query with sufficient confidence. Best: {best_intent} ({best_score:.4f})")
            return RouteMatch(
                intent="unknown",
                confidence=best_score,
                strategy=RoutingStrategy.HYBRID,
                metadata={
                    "best_match": best_intent,
                    "best_score": best_score,
                    "threshold": threshold
                }
            )

        # Determine which strategy contributed most to the match
        strategy_scores = {
            RoutingStrategy.SEMANTIC: max(description_similarities.get(best_intent, 0.0),
                                         example_similarities.get(best_intent, 0.0)),
            RoutingStrategy.KEYWORD: keyword_scores.get(best_intent, 0.0)
        }
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]

        # If multiple strategies contributed significantly, use HYBRID
        if len([s for s, score in strategy_scores.items() if score > 0.5]) > 1:
            best_strategy = RoutingStrategy.HYBRID

        logger.info(f"Routed query to intent: {best_intent} (confidence: {best_score:.4f}, strategy: {best_strategy.value})")

        return RouteMatch(
            intent=best_intent,
            confidence=best_score,
            strategy=best_strategy,
            metadata={
                "description_similarity": description_similarities.get(best_intent, 0.0),
                "example_similarity": example_similarities.get(best_intent, 0.0),
                "keyword_score": keyword_scores.get(best_intent, 0.0)
            }
        )

    async def save_routes(self, file_path: str):
        """Save routes to a file.

        Args:
            file_path: The path to save the routes to
        """
        # Convert routes to a serializable format
        serializable_routes = {}
        for intent, route in self.routes.items():
            serializable_routes[intent] = {
                "intent": route.intent,
                "description": route.description,
                "examples": route.examples,
                "keywords": route.keywords,
                "threshold": route.threshold,
                "priority": route.priority,
                "metadata": route.metadata
            }

        # Save to file
        with open(file_path, "w") as f:
            json.dump(serializable_routes, f, indent=2)

    async def load_routes(self, file_path: str):
        """Load routes from a file.

        Args:
            file_path: The path to load the routes from
        """
        # Load from file
        with open(file_path, "r") as f:
            serializable_routes = json.load(f)

        # Convert to RouteDefinition objects and add to router
        for route_id, route_data in serializable_routes.items():
            await self.add_route(RouteDefinition(**route_data))
