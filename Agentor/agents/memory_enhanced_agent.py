"""
Memory-enhanced agent implementation for the Agentor framework.

This module provides a memory-enhanced agent implementation that extends
the EnhancedAgent with long-term memory capabilities for retaining context
across multiple interactions. Memory-enhanced agents can recall past interactions,
store important information, and use this historical context to inform their
decision-making process.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Generic, TypeVar
import logging
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps

from agentor.agents.enhanced_base import EnhancedAgent, ConfigurationError
from agentor.agents.state_models import MemoryEnhancedAgentState
from agentor.agents.abstract_agent import AgentError
from agentor.core.interfaces.tool import IToolRegistry
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.components.memory import (
    SemanticMemory,
    VectorDBFactory,
    OpenAIEmbeddingProvider
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryError(AgentError):
    """Base exception class for memory-related errors."""
    pass


class MemoryStorageError(MemoryError):
    """Exception raised when memory storage operations fail."""
    pass


class MemoryRetrievalError(MemoryError):
    """Exception raised when memory retrieval operations fail."""
    pass


class MemoryEntry:
    """A memory entry for the memory-enhanced agent.
    
    Represents a single piece of information stored in the agent's memory,
    with metadata about the source, importance, and creation time.
    
    Attributes:
        content: The content of the memory
        source: The source of the memory (e.g., "agent", "user", "environment")
        importance: The importance of the memory (0.0 to 1.0)
        timestamp: The timestamp of the memory
        metadata: Additional metadata for the memory
        id: A unique identifier for the memory
    """

    def __init__(
        self,
        content: str,
        source: str = "agent",
        importance: float = 0.5,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the memory entry.

        Args:
            content: The content of the memory
            source: The source of the memory (e.g., "agent", "user", "environment")
            importance: The importance of the memory (0.0 to 1.0)
            timestamp: The timestamp of the memory (defaults to now)
            metadata: Additional metadata for the memory
            
        Raises:
            ValueError: If importance is not between 0.0 and 1.0
            ValueError: If content is empty
        """
        if not content:
            raise ValueError("Memory content cannot be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"Importance must be between 0.0 and 1.0, got {importance}")
            
        self.content = content
        self.source = source
        self.importance = importance
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.id = f"{int(time.time())}-{hash(content) % 10000}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory entry to a dictionary.

        Returns:
            Dictionary representation of the memory entry
        """
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create a memory entry from a dictionary.

        Args:
            data: Dictionary representation of the memory entry

        Returns:
            A memory entry
            
        Raises:
            ValueError: If required fields are missing
        """
        if "content" not in data:
            raise ValueError("Memory data must include 'content'")
        
        entry = cls(
            content=data["content"],
            source=data.get("source", "agent"),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {})
        )

        # Parse the timestamp
        if "timestamp" in data:
            try:
                entry.timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                entry.timestamp = datetime.now()
                logger.warning(f"Invalid timestamp format in memory data: {data['timestamp']}")

        # Set the ID
        if "id" in data:
            entry.id = data["id"]

        return entry
        
    def __str__(self) -> str:
        """Get a string representation of the memory entry.
        
        Returns:
            A string representation of the memory
        """
        return f"Memory({self.id}): {self.content[:50]}{'...' if len(self.content) > 50 else ''} [{self.importance}]"
        
    def __lt__(self, other: 'MemoryEntry') -> bool:
        """Compare memory entries by importance and timestamp.
        
        Args:
            other: Another memory entry
            
        Returns:
            True if this memory is less important/older than the other
        """
        # First compare by importance (higher is more important)
        if self.importance != other.importance:
            return self.importance < other.importance
        # Then by recency (newer is more important)
        return self.timestamp < other.timestamp


class MemoryEnhancedAgent(EnhancedAgent):
    """Memory-enhanced agent implementation.

    This agent extends the EnhancedAgent with long-term memory capabilities
    for retaining context across multiple interactions. It can store memories
    with different importance levels, recall related memories based on semantic
    similarity, and forget outdated information.

    Examples:
        >>> agent = MemoryEnhancedAgent("memory_agent")
        >>> await agent.add_memory("The user prefers dark mode", importance=0.8)
        >>> memories = await agent.recall("user preferences")
        >>> print(memories[0].content)
        "The user prefers dark mode"
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tool_registry: Optional[IToolRegistry] = None,
        memory_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the memory-enhanced agent.

        Args:
            name: The name of the agent
            tool_registry: Optional tool registry to use
            memory_config: Optional configuration for the memory system
            
        Memory configuration options:
            retention_days: Number of days to retain memories (default: 30)
            capacity: Maximum number of memories to store (default: 1000)
            importance_threshold: Minimum importance for storing memories (default: 0.3)
            vector_db: Configuration for the vector database (optional)
            embedding: Configuration for the embedding provider (optional)
        """
        super().__init__(name, tool_registry)
        self.state_model = MemoryEnhancedAgentState()

        # Set up memory configuration with defaults
        self.memory_config = memory_config or {}
        self.memory_retention = self.memory_config.get("retention_days", 30)
        self.memory_capacity = self.memory_config.get("capacity", 1000)
        self.memory_importance_threshold = self.memory_config.get("importance_threshold", 0.3)
        
        # Additional configuration
        self.auto_memory_enabled = self.memory_config.get("auto_memory", True)
        self.summarize_threshold = self.memory_config.get("summarize_threshold", 50)  # Summarize when > 50 memories

        # Initialize memory storage
        self._initialize_memory()

        # Register memory-related actions
        self.register_action("add_memory", self._add_memory_action)
        self.register_action("recall_memories", self._recall_memories_action)
        self.register_action("forget_memory", self._forget_memory_action)
        self.register_action("summarize_memories", self._summarize_memories_action)
        
    @property
    def version(self) -> str:
        """Get the version of the agent.
        
        Returns:
            The version of the agent
        """
        return "0.2.0"
        
    @property
    def description(self) -> str:
        """Get the description of the agent.
        
        Returns:
            The description of the agent
        """
        return "Memory-enhanced agent with long-term retention capabilities"

    def _initialize_memory(self) -> None:
        """Initialize the memory system.
        
        This method sets up either a semantic memory with vector embeddings
        or a simple in-memory storage, depending on the configuration.
        """
        try:
            # Try to use vector database if configured
            if "vector_db" in self.memory_config:
                self._initialize_semantic_memory()
            else:
                # Use simple in-memory storage
                self._initialize_simple_memory()
        except Exception as e:
            # Fall back to simple memory if there's an error
            logger.error(f"Error initializing memory system: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            self._initialize_simple_memory()
            
    def _initialize_semantic_memory(self) -> None:
        """Initialize semantic memory with vector database.
        
        Raises:
            ConfigurationError: If the semantic memory could not be initialized
        """
        try:
            db_config = self.memory_config["vector_db"]
            embedding_config = self.memory_config.get("embedding", {})

            # Create vector database
            vector_db = VectorDBFactory.create(
                db_type=db_config.get("type", "in_memory"),
                dimension=embedding_config.get("dimension", 384),
                **db_config.get("params", {})
            )

            # Create embedding provider
            if embedding_config.get("provider") == "openai":
                embedding_provider = OpenAIEmbeddingProvider(
                    api_key=embedding_config.get("api_key", ""),
                    model=embedding_config.get("model", "text-embedding-ada-002")
                )
            else:
                # Use a mock embedding provider for testing
                from agentor.components.memory.embedding import MockEmbeddingProvider
                embedding_provider = MockEmbeddingProvider(
                    dimension=embedding_config.get("dimension", 384)
                )

            # Create semantic memory
            self.memory = SemanticMemory(
                vector_db=vector_db,
                embedding_provider=embedding_provider,
                max_nodes=self.memory_capacity,
                forgetting_threshold=self.memory_importance_threshold,
                similarity_threshold=self.memory_config.get("similarity_threshold", 0.7)
            )

            self.state_model.memory_type = "semantic"
            logger.info(f"Initialized semantic memory for agent {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize semantic memory: {str(e)}")
            raise ConfigurationError(f"Failed to initialize semantic memory: {str(e)}") from e
            
    def _initialize_simple_memory(self) -> None:
        """Initialize simple in-memory storage."""
        self.memories: List[MemoryEntry] = []
        self.state_model.memory_type = "simple"
        logger.info(f"Initialized simple memory for agent {self.name}")

    async def add_memory(
        self,
        content: str,
        source: str = "agent",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a memory to the agent's long-term memory.

        Args:
            content: The content of the memory
            source: The source of the memory (e.g., "agent", "user", "environment")
            importance: The importance of the memory (0.0 to 1.0)
            metadata: Additional metadata for the memory

        Returns:
            The ID of the added memory
            
        Raises:
            MemoryStorageError: If memory storage fails
            ValueError: If input parameters are invalid
        """
        try:
            # Validate inputs
            if not content or not isinstance(content, str):
                raise ValueError("Memory content must be a non-empty string")
            if not 0.0 <= importance <= 1.0:
                raise ValueError(f"Importance must be between 0.0 and 1.0, got {importance}")
                
            # Skip if importance is below threshold
            if importance < self.memory_importance_threshold:
                logger.debug(f"Skipping memory with low importance: {importance}")
                return ""
    
            # Create memory entry
            memory_entry = MemoryEntry(
                content=content,
                source=source,
                importance=importance,
                metadata=metadata or {}
            )
    
            # Store in the appropriate memory system
            if self.state_model.memory_type == "semantic":
                await self._store_in_semantic_memory(memory_entry)
            else:
                self._store_in_simple_memory(memory_entry)
    
            # Update state
            self.state_model.memory_count += 1
            self.state_model.last_memory = memory_entry.to_dict()
            self.state_model.memory_statistics["by_source"][source] = self.state_model.memory_statistics["by_source"].get(source, 0) + 1
            
            # Check if we need to summarize memories due to capacity
            if self.auto_memory_enabled and self.state_model.memory_count >= self.summarize_threshold:
                await self._auto_summarize_memories()
    
            logger.debug(f"Added memory: {content[:50]}{'...' if len(content) > 50 else ''} (ID: {memory_entry.id})")
            return memory_entry.id
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            raise MemoryStorageError(f"Failed to add memory: {str(e)}") from e
            
    async def _store_in_semantic_memory(self, memory_entry: MemoryEntry) -> None:
        """Store a memory entry in semantic memory.
        
        Args:
            memory_entry: The memory entry to store
            
        Raises:
            MemoryStorageError: If storage fails
        """
        try:
            await self.memory.add({
                "text": memory_entry.content,
                "source": memory_entry.source,
                "importance": memory_entry.importance,
                "timestamp": memory_entry.timestamp.isoformat(),
                "id": memory_entry.id,
                **memory_entry.metadata
            })
        except Exception as e:
            logger.error(f"Error storing memory in semantic storage: {str(e)}")
            raise MemoryStorageError(f"Failed to store memory: {str(e)}") from e
            
    def _store_in_simple_memory(self, memory_entry: MemoryEntry) -> None:
        """Store a memory entry in simple memory list.
        
        Args:
            memory_entry: The memory entry to store
        """
        # Store in simple memory list
        self.memories.append(memory_entry)

        # Enforce capacity limit
        if len(self.memories) > self.memory_capacity:
            # Sort by importance and recency
            self.memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
            # Keep only the top entries
            self.memories = self.memories[:self.memory_capacity]

    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        source: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        recency_bias: float = 0.2
    ) -> List[MemoryEntry]:
        """Recall memories related to the query.

        Args:
            query: The query to search for
            limit: Maximum number of memories to return
            min_importance: Minimum importance threshold
            source: Optional filter by source
            metadata_filters: Optional filters for metadata fields
            recency_bias: Weight given to recency in ranking (0.0-1.0)

        Returns:
            List of memory entries
            
        Raises:
            MemoryRetrievalError: If memory retrieval fails
            ValueError: If input parameters are invalid
        """
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if limit <= 0:
                raise ValueError("Limit must be a positive integer")
            if not 0.0 <= min_importance <= 1.0:
                raise ValueError(f"Min importance must be between 0.0 and 1.0, got {min_importance}")
            if not 0.0 <= recency_bias <= 1.0:
                raise ValueError(f"Recency bias must be between 0.0 and 1.0, got {recency_bias}")
                
            # Update statistics
            self.state_model.memory_statistics["total_queries"] += 1
            
            results: List[MemoryEntry] = []
            
            if self.state_model.memory_type == "semantic":
                results = await self._recall_from_semantic_memory(
                    query, limit, min_importance, source, metadata_filters
                )
            else:
                results = self._recall_from_simple_memory(
                    query, limit, min_importance, source, metadata_filters, recency_bias
                )
    
            # Update state
            self.state_model.last_recall_query = query
            self.state_model.last_recall_count = len(results)
            
            if results:
                self.state_model.memory_statistics["successful_queries"] += 1
                
            return results
        except Exception as e:
            logger.error(f"Error recalling memories: {str(e)}")
            self.state_model.memory_statistics["failed_queries"] += 1
            raise MemoryRetrievalError(f"Failed to recall memories: {str(e)}") from e
            
    async def _recall_from_semantic_memory(
        self, 
        query: str, 
        limit: int, 
        min_importance: float,
        source: Optional[str],
        metadata_filters: Optional[Dict[str, Any]]
    ) -> List[MemoryEntry]:
        """Recall memories from semantic memory.
        
        Args:
            query: The query to search for
            limit: Maximum number of memories to return
            min_importance: Minimum importance threshold
            source: Optional filter by source
            metadata_filters: Optional filters for metadata fields
            
        Returns:
            List of memory entries
            
        Raises:
            MemoryRetrievalError: If retrieval fails
        """
        try:
            # Search in semantic memory
            filters = {}
            if source:
                filters["source"] = source
            if min_importance > 0:
                filters["importance"] = {"$gte": min_importance}
            if metadata_filters:
                filters.update(metadata_filters)

            memory_results = await self.memory.get(
                query=query,
                limit=limit,
                filters=filters
            )

            # Convert to memory entries
            results = []
            for result in memory_results:
                try:
                    entry = MemoryEntry(
                        content=result.get("text", ""),
                        source=result.get("source", "agent"),
                        importance=result.get("importance", 0.5),
                        metadata={k: v for k, v in result.items() if k not in ["text", "source", "importance", "timestamp", "id"]}
                    )

                    # Set ID and timestamp if available
                    if "id" in result:
                        entry.id = result["id"]
                    if "timestamp" in result:
                        try:
                            entry.timestamp = datetime.fromisoformat(result["timestamp"])
                        except (ValueError, TypeError):
                            pass

                    results.append(entry)
                except Exception as e:
                    logger.warning(f"Error converting memory result: {str(e)}")
                    
            return results
        except Exception as e:
            logger.error(f"Error recalling from semantic memory: {str(e)}")
            raise MemoryRetrievalError(f"Failed to recall from semantic memory: {str(e)}") from e

    def _recall_from_simple_memory(
        self, 
        query: str, 
        limit: int, 
        min_importance: float,
        source: Optional[str],
        metadata_filters: Optional[Dict[str, Any]],
        recency_bias: float
    ) -> List[MemoryEntry]:
        """Recall memories from simple memory storage.
        
        Args:
            query: The query to search for
            limit: Maximum number of memories to return
            min_importance: Minimum importance threshold
            source: Optional filter by source
            metadata_filters: Optional filters for metadata fields
            recency_bias: Weight given to recency in ranking (0.0-1.0)
            
        Returns:
            List of memory entries
        """
        # Search in simple memory list with a basic search implementation
        query_terms = query.lower().split()
        scored_memories = []

        for memory in self.memories:
            # Skip if importance is below threshold
            if memory.importance < min_importance:
                continue

            # Skip if source doesn't match
            if source and memory.source != source:
                continue

            # Skip if metadata doesn't match
            if metadata_filters:
                match = True
                for key, value in metadata_filters.items():
                    if key not in memory.metadata or memory.metadata[key] != value:
                        match = False
                        break
                if not match:
                    continue

            # Calculate a simple relevance score
            content_lower = memory.content.lower()
            score = 0
            for term in query_terms:
                if term in content_lower:
                    score += 1
                    
                    # Bonus for exact matches
                    if term in content_lower.split():
                        score += 0.5

            # Add recency and importance factors
            age_days = (datetime.now() - memory.timestamp).days
            recency_factor = max(0, 1 - (age_days / max(1, self.memory_retention)))
            
            # Combine factors with configurable recency bias
            importance_weight = 1.0 - recency_bias
            final_score = (
                score * 0.5 +  # Relevance factor
                memory.importance * importance_weight +  # Importance factor
                recency_factor * recency_bias  # Recency factor
            )

            if score > 0 or query == "*":  # Allow wildcard to get all memories
                scored_memories.append((memory, final_score))

        # Sort by score and take the top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in scored_memories[:limit]]

    async def forget(self, memory_id: str) -> bool:
        """Forget a specific memory.

        Args:
            memory_id: The ID of the memory to forget

        Returns:
            True if the memory was forgotten, False otherwise
            
        Raises:
            MemoryError: If memory deletion fails
        """
        try:
            if self.state_model.memory_type == "semantic":
                # Delete from semantic memory
                try:
                    await self.memory.delete(memory_id)
                    self.state_model.memory_count = max(0, self.state_model.memory_count - 1)
                    logger.debug(f"Deleted memory with ID {memory_id} from semantic storage")
                    return True
                except Exception as e:
                    logger.error(f"Error forgetting memory {memory_id}: {str(e)}")
                    return False
            else:
                # Delete from simple memory list
                initial_count = len(self.memories)
                self.memories = [m for m in self.memories if m.id != memory_id]

                # Update state if a memory was removed
                if len(self.memories) < initial_count:
                    self.state_model.memory_count = max(0, self.state_model.memory_count - 1)
                    logger.debug(f"Deleted memory with ID {memory_id} from simple storage")
                    return True

                return False
        except Exception as e:
            logger.error(f"Error forgetting memory: {str(e)}")
            raise MemoryError(f"Failed to forget memory: {str(e)}") from e
            
    async def forget_before(self, timestamp: datetime) -> int:
        """Forget all memories created before a specific timestamp.
        
        Args:
            timestamp: The cutoff timestamp
            
        Returns:
            The number of memories forgotten
            
        Raises:
            MemoryError: If memory deletion fails
        """
        try:
            if self.state_model.memory_type == "semantic":
                # For semantic memory, we need to retrieve all memories and check timestamps
                # This is inefficient but necessary if the vector DB doesn't support time-based queries
                try:
                    # Get all memories (using wildcard query)
                    all_memories = await self.recall("*", limit=self.memory_capacity)
                    
                    # Find memories to forget
                    memories_to_forget = [m for m in all_memories if m.timestamp < timestamp]
                    
                    # Delete each memory
                    deleted_count = 0
                    for memory in memories_to_forget:
                        if await self.forget(memory.id):
                            deleted_count += 1
                            
                    return deleted_count
                except Exception as e:
                    logger.error(f"Error forgetting memories before {timestamp}: {str(e)}")
                    raise MemoryError(f"Failed to forget memories by timestamp: {str(e)}") from e
            else:
                # For simple memory, we can filter directly
                initial_count = len(self.memories)
                self.memories = [m for m in self.memories if m.timestamp >= timestamp]
                
                # Update memory count
                forgotten_count = initial_count - len(self.memories)
                self.state_model.memory_count = max(0, self.state_model.memory_count - forgotten_count)
                
                if forgotten_count > 0:
                    logger.info(f"Forgot {forgotten_count} memories older than {timestamp}")
                    
                return forgotten_count
        except Exception as e:
            logger.error(f"Error forgetting memories by timestamp: {str(e)}")
            raise MemoryError(f"Failed to forget memories by timestamp: {str(e)}") from e
            
    async def _auto_summarize_memories(self) -> None:
        """Automatically summarize old memories to prevent exceeding capacity.
        
        This method identifies clusters of related memories and generates
        summaries to replace the individual memories, reducing memory usage.
        """
        # This is a placeholder for future implementation
        # In a real implementation, this would:
        # 1. Cluster related memories
        # 2. Generate summaries for each cluster
        # 3. Replace the original memories with the summaries
        
        # Reset the counter to avoid continuous summarization attempts
        self.state_model.memory_count = max(0, self.state_model.memory_count - self.summarize_threshold // 2)
        logger.debug("Auto-summarization triggered (placeholder implementation)")

    async def _add_memory_action(self, content: str, importance: float = 0.5, source: str = "agent") -> Dict[str, Any]:
        """Action to add a memory.

        Args:
            content: The content of the memory
            importance: The importance of the memory
            source: The source of the memory

        Returns:
            Result of adding the memory
        """
        try:
            memory_id = await self.add_memory(content, source, importance)
            return {
                "success": bool(memory_id),
                "memory_id": memory_id
            }
        except Exception as e:
            logger.error(f"Error in add_memory action: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _recall_memories_action(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Action to recall memories.

        Args:
            query: The query to search for
            limit: Maximum number of memories to return

        Returns:
            Result of recalling memories
        """
        try:
            memories = await self.recall(query, limit)
            return {
                "success": True,
                "count": len(memories),
                "memories": [m.to_dict() for m in memories]
            }
        except Exception as e:
            logger.error(f"Error in recall_memories action: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "count": 0,
                "memories": []
            }

    async def _forget_memory_action(self, memory_id: str) -> Dict[str, Any]:
        """Action to forget a memory.

        Args:
            memory_id: The ID of the memory to forget

        Returns:
            Result of forgetting the memory
        """
        try:
            success = await self.forget(memory_id)
            return {
                "success": success,
                "memory_id": memory_id
            }
        except Exception as e:
            logger.error(f"Error in forget_memory action: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "memory_id": memory_id
            }
            
    async def _summarize_memories_action(self, query: str = "*", limit: int = 20) -> Dict[str, Any]:
        """Action to summarize memories on a topic.
        
        Args:
            query: The query to find memories to summarize
            limit: Maximum number of memories to include in summarization
            
        Returns:
            Result of summarizing memories
        """
        try:
            # This is a placeholder implementation
            # A real implementation would:
            # 1. Retrieve memories related to the query
            # 2. Generate a summary
            # 3. Store the summary as a new higher-importance memory
            # 4. Optionally delete the summarized memories
            
            memories = await self.recall(query, limit)
            
            return {
                "success": True,
                "count": len(memories),
                "message": f"Summarized {len(memories)} memories (placeholder implementation)"
            }
        except Exception as e:
            logger.error(f"Error in summarize_memories action: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        """Preprocess the input data before running the agent.

        Args:
            input_data: The input data to preprocess

        Returns:
            The preprocessed input data
        """
        try:
            # Store the query in memory if auto-memory is enabled and query is significant
            if self.auto_memory_enabled and input_data.query and len(input_data.query) > 10:
                await self.add_memory(
                    content=f"User query: {input_data.query}",
                    source="user",
                    importance=0.4,
                    metadata={"type": "query"}
                )
    
            # Try to recall relevant memories for context
            if input_data.query:
                try:
                    memories = await self.recall(input_data.query, limit=3)
                    if memories:
                        # Add memories to context
                        memory_context = [m.content for m in memories]
                        if "context" not in input_data.context:
                            input_data.context["context"] = {}
                        input_data.context["relevant_memories"] = memory_context
    
                        # Log the recalled memories
                        logger.debug(f"Recalled {len(memories)} memories for query: {input_data.query[:50]}{'...' if len(input_data.query) > 50 else ''}")
                except Exception as e:
                    logger.error(f"Error recalling memories during preprocessing: {str(e)}")
                    # Continue even if memory recall fails
    
            return await super().preprocess(input_data)
        except Exception as e:
            logger.error(f"Error in memory agent preprocessing: {str(e)}")
            # Continue with original input data if preprocessing fails
            return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        """Postprocess the output data after running the agent.

        Args:
            output_data: The output data to postprocess

        Returns:
            The postprocessed output data
        """
        try:
            # Store the agent's response in memory if auto-memory is enabled and it's significant
            response_str = str(output_data.response)
            if self.auto_memory_enabled and response_str and len(response_str) > 10:
                await self.add_memory(
                    content=f"Agent response: {response_str}",
                    source="agent",
                    importance=0.4,
                    metadata={"type": "response"}
                )
    
            # Add memory information to the metadata
            output_data.metadata["memory_enhanced"] = {
                "memory_type": self.state_model.memory_type,
                "memory_count": self.state_model.memory_count,
                "last_recall_query": self.state_model.last_recall_query,
                "last_recall_count": self.state_model.last_recall_count,
                "statistics": self.state_model.memory_statistics
            }
    
            return await super().postprocess(output_data)
        except Exception as e:
            logger.error(f"Error in memory agent postprocessing: {str(e)}")
            # Return original output if postprocessing fails
            return output_data

    @property
    def state(self) -> Dict[str, Any]:
        """Get the agent's state.

        Returns:
            The agent's state
        """
        return self.state_model.dict()
        
    async def validate_configuration(self) -> None:
        """Validate the agent configuration.
        
        This method checks that the agent has all the necessary components
        to function properly.
        
        Raises:
            ConfigurationError: If the agent is not properly configured
        """
        # Validate memory configuration
        if self.memory_retention <= 0:
            raise ConfigurationError("Memory retention must be a positive number of days")
            
        if self.memory_capacity <= 0:
            raise ConfigurationError("Memory capacity must be a positive integer")
            
        if not 0.0 <= self.memory_importance_threshold <= 1.0:
            raise ConfigurationError(f"Memory importance threshold must be between 0.0 and 1.0, got {self.memory_importance_threshold}")
            
        # Log memory configuration
        logger.info(f"Memory agent {self.name} configured with retention={self.memory_retention} days, capacity={self.memory_capacity}, threshold={self.memory_importance_threshold}")
        
    async def initialize(self) -> None:
        """Initialize the agent.
        
        This method sets up any resources needed by the agent and validates
        that the agent has the necessary components to function properly.
        """
        await super().initialize()
        
        # Initialize memory statistics
        self.state_model.memory_statistics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "by_source": {}
        }
        
        # Perform basic memory health check
        if self.state_model.memory_type == "semantic":
            try:
                # Check if semantic memory is responsive
                test_result = await self.memory.health_check()
                if not test_result.get("healthy", False):
                    logger.warning(f"Memory system health check failed: {test_result.get('message', 'unknown issue')}")
            except Exception as e:
                logger.warning(f"Memory system health check failed: {str(e)}")
        
        logger.info(f"Memory-enhanced agent {self.name} initialized with {self.state_model.memory_type} memory type")
        
    async def shutdown(self) -> None:
        """Shutdown the agent.
        
        This method cleans up memory resources.
        """
        logger.info(f"Shutting down memory-enhanced agent: {self.name}")
        
        # Clean up memory resources if necessary
        if self.state_model.memory_type == "semantic":
            try:
                await self.memory.close()
            except Exception as e:
                logger.error(f"Error closing semantic memory: {str(e)}")
        
        await super().shutdown()
