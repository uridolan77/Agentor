"""
Unified Memory System for the Agentor framework.

This module provides a unified memory system that combines episodic, semantic, and procedural memory.
It allows for storing and retrieving information across different memory types with a consistent interface.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Literal
import time
import asyncio
import logging
import math
import uuid
from enum import Enum
from collections import defaultdict

from agentor.components.memory import Memory
from agentor.components.memory.episodic_memory import EpisodicMemory, Episode
from agentor.components.memory.semantic_memory import SemanticMemory, KnowledgeNode
from agentor.components.memory.procedural_memory import ProceduralMemory, Procedure
from agentor.llm_gateway.utils.metrics import track_memory_operation

logger = logging.getLogger(__name__)

# Add new enum for context matching strategies
class ContextMatchingStrategy(str, Enum):
    """Strategies for matching context when retrieving memories."""
    EXACT = "exact"       # Exact matching of context keys and values
    PARTIAL = "partial"   # Match subset of context keys and values
    SEMANTIC = "semantic" # Use semantic similarity for context matching
    HYBRID = "hybrid"     # Combination of exact and semantic matching
    HIERARCHICAL = "hierarchical"  # Match contexts considering hierarchical structure
    TEMPORAL = "temporal" # Consider temporal relationships in context matching


class MemoryType(str, Enum):
    """Types of memory in the unified memory system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ALL = "all"


class MemoryReflectionLevel(str, Enum):
    """Reflection levels for memory analysis."""
    NONE = "none"      # No reflection
    BASIC = "basic"    # Basic pattern recognition
    ADVANCED = "advanced"  # Deeper analysis with connections
    FULL = "full"      # Complete memory analysis with insights


class UnifiedMemory(Memory):
    """Unified memory system that combines episodic, semantic, and procedural memory."""
    
    def __init__(
        self,
        embedding_provider=None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        procedural_memory: Optional[ProceduralMemory] = None,
        consolidation_interval: int = 3600,  # 1 hour in seconds
    ):
        """Initialize the unified memory system.
        
        Args:
            embedding_provider: Provider for generating embeddings
            episodic_memory: Episodic memory instance, created if None
            semantic_memory: Semantic memory instance, created if None
            procedural_memory: Procedural memory instance, created if None
            consolidation_interval: How often to consolidate memories (in seconds)
        """
        self.embedding_provider = embedding_provider
        
        # Initialize memory subsystems
        self.episodic = episodic_memory or EpisodicMemory(embedding_provider=embedding_provider)
        self.semantic = semantic_memory or SemanticMemory(embedding_provider=embedding_provider)
        self.procedural = procedural_memory or ProceduralMemory(embedding_provider=embedding_provider)
        
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = time.time()
        self.lock = asyncio.Lock()
    
    @track_memory_operation("add", "unified")
    async def add(self, item: Dict[str, Any], memory_type: Optional[MemoryType] = None):
        """Add an item to memory.
        
        Args:
            item: The item to add
            memory_type: The type of memory to add to, or None to auto-detect
        """
        # Auto-detect memory type if not specified
        if memory_type is None:
            memory_type = self._detect_memory_type(item)
        
        # Add to the appropriate memory subsystem
        if memory_type == MemoryType.EPISODIC:
            await self.episodic.add(item)
        elif memory_type == MemoryType.SEMANTIC:
            await self.semantic.add(item)
        elif memory_type == MemoryType.PROCEDURAL:
            await self.procedural.add(item)
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
        
        # Check if we need to consolidate memories
        if time.time() - self.last_consolidation > self.consolidation_interval:
            await self.consolidate_memories()
    
    @track_memory_operation("get", "unified")
    async def get(
        self, 
        query: Dict[str, Any], 
        memory_type: MemoryType = MemoryType.ALL,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get items from memory that match a query.
        
        Args:
            query: The query to match
            memory_type: The type of memory to search, or ALL to search all types
            limit: Maximum number of items to return
            
        Returns:
            A list of matching items
        """
        results = []
        
        # Search the appropriate memory subsystems
        if memory_type in (MemoryType.ALL, MemoryType.EPISODIC):
            episodic_results = await self.episodic.get(query, limit=limit)
            for result in episodic_results:
                result['memory_type'] = MemoryType.EPISODIC.value
                results.append(result)
        
        if memory_type in (MemoryType.ALL, MemoryType.SEMANTIC):
            semantic_results = await self.semantic.get(query, limit=limit)
            for result in semantic_results:
                result['memory_type'] = MemoryType.SEMANTIC.value
                results.append(result)
        
        if memory_type in (MemoryType.ALL, MemoryType.PROCEDURAL):
            procedural_results = await self.procedural.get(query, limit=limit)
            for result in procedural_results:
                result['memory_type'] = MemoryType.PROCEDURAL.value
                results.append(result)
        
        # Sort results by relevance if we have similarity scores
        if any('similarity' in result for result in results):
            results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        
        return results[:limit]
    
    @track_memory_operation("clear", "unified")
    async def clear(self, memory_type: MemoryType = MemoryType.ALL):
        """Clear items from memory.
        
        Args:
            memory_type: The type of memory to clear, or ALL to clear all types
        """
        async with self.lock:
            if memory_type in (MemoryType.ALL, MemoryType.EPISODIC):
                await self.episodic.clear()
            
            if memory_type in (MemoryType.ALL, MemoryType.SEMANTIC):
                await self.semantic.clear()
            
            if memory_type in (MemoryType.ALL, MemoryType.PROCEDURAL):
                await self.procedural.clear()
    
    async def consolidate_memories(self):
        """Consolidate memories across all memory subsystems."""
        async with self.lock:
            # Consolidate each memory subsystem
            await self.episodic.consolidate_memories()
            await self.semantic.consolidate_memories()
            await self._prune_procedural_memory()
            
            # Extract knowledge from episodic memory and store in semantic memory
            await self._extract_knowledge_from_episodes()
            
            # Extract procedures from episodic memory and store in procedural memory
            await self._extract_procedures_from_episodes()
            
            self.last_consolidation = time.time()
    
    async def _extract_knowledge_from_episodes(self):
        """Extract knowledge from episodic memory and store in semantic memory."""
        # Get recent episodes
        recent_episodes = await self.episodic.get({}, limit=10)
        
        for episode_dict in recent_episodes:
            # Skip episodes that have already been processed for knowledge extraction
            if episode_dict.get('metadata', {}).get('knowledge_extracted'):
                continue
            
            # Extract knowledge from the episode
            # This is a simplified implementation - in a real system, this might use an LLM
            # to extract knowledge from the episode
            
            # For now, we'll just extract simple facts from events with 'fact' field
            for event in episode_dict.get('events', []):
                if 'fact' in event:
                    # Create a knowledge node from the fact
                    knowledge = {
                        'text': event['fact'],
                        'source': f"episode:{episode_dict['id']}",
                        'confidence': 0.8,
                        'importance': 0.5,
                        'created_at': event.get('timestamp', time.time())
                    }
                    
                    # Add to semantic memory
                    await self.semantic.add(knowledge)
            
            # Mark the episode as processed
            episode = await self.episodic.get_episode(episode_dict['id'])
            if episode:
                if 'metadata' not in episode.metadata:
                    episode.metadata['metadata'] = {}
                episode.metadata['knowledge_extracted'] = True
    
    async def _extract_procedures_from_episodes(self):
        """Extract procedures from episodic memory and store in procedural memory."""
        # Get recent episodes
        recent_episodes = await self.episodic.get({}, limit=10)
        
        for episode_dict in recent_episodes:
            # Skip episodes that have already been processed for procedure extraction
            if episode_dict.get('metadata', {}).get('procedures_extracted'):
                continue
            
            # Extract procedures from the episode
            # This is a simplified implementation - in a real system, this might use an LLM
            # to extract procedures from the episode
            
            # For now, we'll just extract simple procedures from events with 'procedure' field
            for event in episode_dict.get('events', []):
                if 'procedure' in event and isinstance(event['procedure'], dict):
                    proc = event['procedure']
                    
                    # Check if we have the required fields
                    if 'name' in proc and 'description' in proc:
                        # Create a procedure
                        if 'steps' in proc and isinstance(proc['steps'], list):
                            # Store as steps
                            await self.procedural.store_steps(
                                name=proc['name'],
                                description=proc['description'],
                                steps=proc['steps'],
                                tags=proc.get('tags', [])
                            )
                        elif 'code' in proc:
                            # Store as code
                            await self.procedural.store_code(
                                name=proc['name'],
                                description=proc['description'],
                                code=proc['code'],
                                tags=proc.get('tags', [])
                            )
            
            # Mark the episode as processed
            episode = await self.episodic.get_episode(episode_dict['id'])
            if episode:
                if 'metadata' not in episode.metadata:
                    episode.metadata['metadata'] = {}
                episode.metadata['procedures_extracted'] = True
    
    async def _prune_procedural_memory(self):
        """Prune procedural memory based on success rates."""
        # This is handled internally by the procedural memory
        pass
    
    def _detect_memory_type(self, item: Dict[str, Any]) -> MemoryType:
        """Detect the appropriate memory type for an item.
        
        Args:
            item: The item to classify
            
        Returns:
            The detected memory type
        """
        # If the item has an explicit memory_type field, use that
        if 'memory_type' in item:
            memory_type = item['memory_type']
            if memory_type == 'episodic':
                return MemoryType.EPISODIC
            elif memory_type == 'semantic':
                return MemoryType.SEMANTIC
            elif memory_type == 'procedural':
                return MemoryType.PROCEDURAL
        
        # Check for episodic memory indicators
        if 'events' in item or 'episode_id' in item:
            return MemoryType.EPISODIC
        
        # Check for procedural memory indicators
        if any(key in item for key in ('code', 'steps', 'function')):
            return MemoryType.PROCEDURAL
        
        # Default to semantic memory
        return MemoryType.SEMANTIC
    
    async def create_episode(self, episode_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        """Create a new episode in episodic memory.
        
        Args:
            episode_id: Optional ID for the episode
            metadata: Optional metadata for the episode
            
        Returns:
            The created episode
        """
        return await self.episodic.create_episode(episode_id, metadata)
    
    async def end_episode(self, episode_id: Optional[str] = None) -> Optional[Episode]:
        """End an episode in episodic memory.
        
        Args:
            episode_id: The ID of the episode to end, or None for current episode
            
        Returns:
            The ended episode, or None if not found
        """
        return await self.episodic.end_episode(episode_id)
    
    async def add_to_episode(self, item: Dict[str, Any], episode_id: Optional[str] = None):
        """Add an event to an episode in episodic memory.
        
        Args:
            item: The event to add
            episode_id: The ID of the episode to add to, or None for current episode
        """
        await self.episodic.add(item, episode_id)
    
    async def store_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store knowledge in semantic memory.
        
        Args:
            text: The knowledge text
            metadata: Optional metadata for the knowledge
            
        Returns:
            The ID of the stored knowledge node
        """
        # Create a knowledge item
        item = {
            'text': text,
            'created_at': time.time(),
            'importance': 0.5,
            'confidence': 1.0
        }
        
        if metadata:
            item.update(metadata)
        
        # Add to semantic memory
        await self.semantic.add(item)
        
        # Return the ID (this is a simplification - in reality, we'd need to get the ID from the add method)
        return f"knowledge-{hash(text)}"
    
    async def store_procedure(
        self, 
        name: str, 
        description: str, 
        steps: List[Dict[str, Any]], 
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a procedure in procedural memory.
        
        Args:
            name: Name of the procedure
            description: Description of the procedure
            steps: Step-by-step instructions for the procedure
            tags: Optional tags for categorizing the procedure
            
        Returns:
            The ID of the stored procedure
        """
        return await self.procedural.store_steps(name, description, steps, tags=tags)
    
    async def execute_procedure(self, proc_id: str, *args, **kwargs) -> Any:
        """Execute a procedure from procedural memory.
        
        Args:
            proc_id: The ID of the procedure to execute
            *args: Positional arguments for the procedure
            **kwargs: Keyword arguments for the procedure
            
        Returns:
            The result of the procedure
        """
        return await self.procedural.execute_procedure(proc_id, *args, **kwargs)
    
    async def search(self, query: str, memory_type: MemoryType = MemoryType.ALL, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory using a text query.
        
        Args:
            query: The text query
            memory_type: The type of memory to search
            limit: Maximum number of results to return
            
        Returns:
            A list of matching items
        """
        if not self.embedding_provider:
            raise ValueError("Embedding provider is required for semantic search")
        
        # Create a query object
        query_obj = {'text': query, 'threshold': 0.7}
        
        # Search memory
        return await self.get(query_obj, memory_type=memory_type, limit=limit)
    
    async def retrieve_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        memory_types: Optional[List[str]] = None,
        decay_factor: float = 0.05,
        recency_weight: float = 0.3,
        importance_weight: float = 0.4,
        relevance_weight: float = 0.7,
        matching_strategy: ContextMatchingStrategy = ContextMatchingStrategy.HYBRID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with contextual awareness.
        
        This method enhances memory retrieval by taking into account:
        - Current context (e.g., task, environment, agent state)
        - Recency of memories
        - Importance of memories
        - Relevance to query
        
        Args:
            query: The text query to search for
            context: Contextual information like current task, environment, etc.
            memory_types: Types of memories to search (episodic, semantic, procedural)
            decay_factor: Controls how quickly memory relevance decays with time
            recency_weight: Weight for memory recency in scoring
            importance_weight: Weight for memory importance in scoring
            relevance_weight: Weight for query relevance in scoring
            matching_strategy: Strategy for matching context
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memories sorted by contextual relevance
        """
        # Default to all memory types if not specified
        if memory_types is None:
            memory_types = ["episodic", "semantic", "procedural"]
        
        all_memories = []
        current_time = time.time()
        
        # Retrieve memories from each specified memory type
        for memory_type in memory_types:
            if memory_type == "episodic" and hasattr(self, "episodic"):
                memories = await self.episodic.search(query, limit=limit*2)
                for memory in memories:
                    memory["memory_type"] = "episodic"
                all_memories.extend(memories)
                
            elif memory_type == "semantic" and hasattr(self, "semantic"):
                memories = await self.semantic.search(query, limit=limit*2)
                for memory in memories:
                    memory["memory_type"] = "semantic"
                all_memories.extend(memories)
                
            elif memory_type == "procedural" and hasattr(self, "procedural"):
                memories = await self.procedural.search(query, limit=limit*2)
                for memory in memories:
                    memory["memory_type"] = "procedural"
                all_memories.extend(memories)
        
        # Score each memory based on context, recency, importance, and relevance
        scored_memories = []
        for memory in all_memories:
            # Calculate recency score (higher for more recent memories)
            time_diff = current_time - memory.get("created_at", current_time)
            recency_score = math.exp(-decay_factor * time_diff)
            
            # Calculate importance score (from memory metadata or default to 0.5)
            importance_score = memory.get("importance", 0.5)
            
            # Calculate relevance to query (use existing relevance or similarity)
            relevance_score = memory.get("relevance", memory.get("similarity", 0.5))
            
            # Calculate context relevance using the enhanced context matching
            context_score = self._calculate_context_match(
                memory.get("context", {}),
                context,
                matching_strategy,
                max_depth=2
            )
            
            # Calculate final score as weighted average
            final_score = (
                recency_weight * recency_score +
                importance_weight * importance_score +
                relevance_weight * relevance_score +
                (1.0 - recency_weight - importance_weight - relevance_weight) * context_score
            )
            
            memory["final_score"] = final_score
            memory["context_score"] = context_score
            memory["recency_score"] = recency_score
            scored_memories.append(memory)
        
        # Sort by final score and return top results
        scored_memories.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        # Track this retrieval for adaptive weighting
        self._track_memory_retrieval(query, context, {"combined": scored_memories[:limit]})
        
        return scored_memories[:limit]

    def _calculate_context_match(
        self,
        memory_context: Dict[str, Any],
        current_context: Dict[str, Any],
        strategy: ContextMatchingStrategy = ContextMatchingStrategy.HYBRID,
        max_depth: int = 2,
    ) -> float:
        """Calculate how well a memory's context matches the current context.
        
        Args:
            memory_context: Context stored with the memory
            current_context: Current context to match against
            strategy: Strategy to use for context matching
            max_depth: Maximum depth for hierarchical context matching
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not memory_context or not current_context:
            return 0.0
            
        # Implementation based on the matching strategy
        if strategy == ContextMatchingStrategy.EXACT:
            # Exact matching - all keys and values must match
            matching_keys = set(memory_context.keys()).intersection(current_context.keys())
            if not matching_keys:
                return 0.0
                
            matches = sum(1 for k in matching_keys if memory_context[k] == current_context[k])
            return matches / max(len(memory_context), len(current_context))
            
        elif strategy == ContextMatchingStrategy.PARTIAL:
            # Partial matching - some keys and values must match
            matching_keys = set(memory_context.keys()).intersection(current_context.keys())
            if not matching_keys:
                return 0.0
                
            matches = sum(1 for k in matching_keys if memory_context[k] == current_context[k])
            return matches / len(matching_keys) if matching_keys else 0.0
            
        elif strategy == ContextMatchingStrategy.SEMANTIC:
            # Semantic matching using embeddings
            if not self.embedding_provider:
                logger.warning("Semantic context matching requires an embedding provider")
                return 0.0
                
            try:
                # Convert contexts to strings for embedding
                memory_str = " ".join([f"{k}: {v}" for k, v in memory_context.items()])
                current_str = " ".join([f"{k}: {v}" for k, v in current_context.items()])
                
                # Get embeddings
                memory_embedding = self.embedding_provider.get_embedding(memory_str)
                current_embedding = self.embedding_provider.get_embedding(current_str)
                
                # Calculate cosine similarity
                return self.embedding_provider.calculate_similarity(memory_embedding, current_embedding)
            except Exception as e:
                logger.error(f"Error in semantic context matching: {e}")
                return 0.0
                
        elif strategy == ContextMatchingStrategy.HIERARCHICAL:
            # Hierarchical matching - traverse nested contexts
            return self._hierarchical_context_match(memory_context, current_context, current_depth=0, max_depth=max_depth)
            
        elif strategy == ContextMatchingStrategy.TEMPORAL:
            # Temporal matching - consider time-based relationships
            return self._temporal_context_match(memory_context, current_context)
            
        else:  # Default to HYBRID
            # Hybrid approach combining exact and semantic matching
            exact_score = self._calculate_context_match(
                memory_context, current_context, ContextMatchingStrategy.EXACT
            )
            
            semantic_score = self._calculate_context_match(
                memory_context, current_context, ContextMatchingStrategy.SEMANTIC
            )
            
            # Weighted combination (favor exact matches but consider semantic similarity)
            return 0.7 * exact_score + 0.3 * semantic_score
    
    def _hierarchical_context_match(
        self, 
        memory_context: Dict[str, Any], 
        current_context: Dict[str, Any],
        current_depth: int = 0,
        max_depth: int = 2
    ) -> float:
        """Match contexts hierarchically, traversing nested dictionaries.
        
        Args:
            memory_context: Context stored with the memory
            current_context: Current context to match against
            current_depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if current_depth > max_depth or not isinstance(memory_context, dict) or not isinstance(current_context, dict):
            return 0.0
            
        # Match at current level
        matching_keys = set(memory_context.keys()).intersection(current_context.keys())
        if not matching_keys:
            return 0.0
            
        total_score = 0.0
        for key in matching_keys:
            if isinstance(memory_context[key], dict) and isinstance(current_context[key], dict):
                # Recursive match for nested dictionaries
                nested_score = self._hierarchical_context_match(
                    memory_context[key], 
                    current_context[key],
                    current_depth + 1,
                    max_depth
                )
                total_score += nested_score
            elif memory_context[key] == current_context[key]:
                # Exact match for non-dict values
                total_score += 1.0
            # Partial credit for string matches
            elif isinstance(memory_context[key], str) and isinstance(current_context[key], str):
                # Simple string similarity (Jaccard similarity of words)
                mem_words = set(memory_context[key].lower().split())
                curr_words = set(current_context[key].lower().split())
                if mem_words and curr_words:
                    intersection = len(mem_words.intersection(curr_words))
                    union = len(mem_words.union(curr_words))
                    total_score += intersection / union if union > 0 else 0.0
                
        # Normalize score by number of matching keys
        return total_score / len(matching_keys) if matching_keys else 0.0
    
    def _temporal_context_match(self, memory_context: Dict[str, Any], current_context: Dict[str, Any]) -> float:
        """Match contexts based on temporal relationships.
        
        Args:
            memory_context: Context stored with the memory
            current_context: Current context to match against
            
        Returns:
            Match score between 0.0 and 1.0
        """
        # Check for temporal markers in both contexts
        memory_time = memory_context.get('timestamp', memory_context.get('time', None))
        current_time = current_context.get('timestamp', current_context.get('time', time.time()))
        
        if memory_time is None:
            return 0.0
            
        # Convert to numeric timestamps if needed
        if isinstance(memory_time, str):
            try:
                memory_time = float(memory_time)
            except ValueError:
                # Try to parse as ISO format
                try:
                    from datetime import datetime
                    memory_time = datetime.fromisoformat(memory_time).timestamp()
                except (ValueError, ImportError):
                    return 0.0
                    
        # Calculate temporal proximity
        time_diff = abs(current_time - memory_time)
        
        # Check for temporal context markers
        if 'time_context' in memory_context and 'time_context' in current_context:
            if memory_context['time_context'] == current_context['time_context']:
                # Same temporal context (e.g., "morning", "evening")
                return 1.0
                
        # Time decay function
        # Max window of 24 hours (86400 seconds)
        max_window = 86400
        return max(0.0, 1.0 - (time_diff / max_window))
    
    def _track_memory_retrieval(self, query: str, context: Dict[str, Any], results: Dict[str, List[Dict[str, Any]]]):
        """Track memory retrievals for adaptive weighting and learning.
        
        Args:
            query: The query used for retrieval
            context: The context used for retrieval
            results: The results returned for each memory type
        """
        # Store this retrieval for later analysis
        if not hasattr(self, '_retrieval_history'):
            self._retrieval_history = []
            
        self._retrieval_history.append({
            'timestamp': time.time(),
            'query': query,
            'context': context,
            'result_counts': {k: len(v) for k, v in results.items()}
        })
        
        # Keep history to a reasonable size
        if len(self._retrieval_history) > 100:
            self._retrieval_history = self._retrieval_history[-100:]
    
    async def retrieve_by_contextual_cues(
        self,
        context_cues: Dict[str, Any],
        memory_types: Optional[List[str]] = None,
        matching_strategy: ContextMatchingStrategy = ContextMatchingStrategy.HIERARCHICAL,
        min_confidence: float = 0.2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories based on contextual cues without an explicit query.
        
        This method allows for memory activation based purely on contextual triggering,
        simulating the way human memories can be triggered by environmental cues.
        
        Args:
            context_cues: Dictionary of contextual cues (location, people, objects, etc.)
            memory_types: Types of memories to search (default: all types)
            matching_strategy: Strategy to use for context matching
            min_confidence: Minimum confidence threshold for matches
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memories triggered by the contextual cues
        """
        # Default to all memory types if not specified
        if memory_types is None:
            memory_types = ["episodic", "semantic", "procedural"]
            
        all_memories = []
        
        # Collect memories from each memory type
        for memory_type in memory_types:
            if memory_type == "episodic" and hasattr(self, "episodic"):
                # For episodic memory, retrieve recent episodes and filter by context
                recent_episodes = await self.episodic.get({}, limit=limit*3)
                all_memories.extend([(memory, "episodic") for memory in recent_episodes])
                
            elif memory_type == "semantic" and hasattr(self, "semantic"):
                # For semantic memory, retrieve knowledge nodes related to context cues
                semantic_memories = await self.semantic.get({}, limit=limit*3)
                all_memories.extend([(memory, "semantic") for memory in semantic_memories])
                
            elif memory_type == "procedural" and hasattr(self, "procedural"):
                # For procedural memory, retrieve procedures related to context cues
                procedural_memories = await self.procedural.get({}, limit=limit*3)
                all_memories.extend([(memory, "procedural") for memory in procedural_memories])
        
        # Score memories based on contextual match
        scored_memories = []
        for memory, memory_type in all_memories:
            memory_context = memory.get("context", {})
            context_score = self._calculate_context_match(
                memory_context,
                context_cues,
                matching_strategy
            )
            
            if context_score >= min_confidence:
                memory["memory_type"] = memory_type
                memory["context_score"] = context_score
                scored_memories.append(memory)
        
        # Sort by context score and return top results
        scored_memories.sort(key=lambda x: x.get("context_score", 0), reverse=True)
        
        # Track this retrieval for adaptive weighting
        self._track_memory_retrieval("contextual_cues", context_cues, {"combined": scored_memories[:limit]})
        
        return scored_memories[:limit]
    
    async def associative_memory_retrieval(
        self,
        seed_memory_id: str,
        association_type: str = "semantic",
        depth: int = 1,
        breadth: int = 5,
        memory_types: Optional[List[str]] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Retrieve memories associated with a seed memory through associative links.
        
        This simulates the human mind's ability to follow associative pathways, 
        retrieving memories connected to a starting point.
        
        Args:
            seed_memory_id: ID of the memory to start from
            association_type: Type of association to follow (semantic, temporal, causal)
            depth: How many steps to follow in the association chain
            breadth: How many associations to follow at each step
            memory_types: Types of memories to include (default: all types)
            min_similarity: Minimum similarity threshold for associations
            
        Returns:
            List of associated memories with association metadata
        """
        if memory_types is None:
            memory_types = ["episodic", "semantic", "procedural"]
            
        # Find the seed memory
        seed_memory = None
        for memory_type_str in memory_types:
            if memory_type_str == "episodic" and hasattr(self, "episodic"):
                try:
                    seed_memory = await self.episodic.get_episode(seed_memory_id)
                    if seed_memory:
                        seed_memory = seed_memory.to_dict()
                        seed_memory["memory_type"] = "episodic"
                        break
                except:
                    pass
            elif memory_type_str == "semantic" and hasattr(self, "semantic"):
                try:
                    seed_memory = await self.semantic.get_node(seed_memory_id)
                    if seed_memory:
                        seed_memory = seed_memory.to_dict() if hasattr(seed_memory, "to_dict") else seed_memory
                        seed_memory["memory_type"] = "semantic"
                        break
                except:
                    pass
            elif memory_type_str == "procedural" and hasattr(self, "procedural"):
                try:
                    seed_memory = await self.procedural.get_procedure(seed_memory_id)
                    if seed_memory:
                        seed_memory = seed_memory.to_dict() if hasattr(seed_memory, "to_dict") else seed_memory
                        seed_memory["memory_type"] = "procedural"
                        break
                except:
                    pass
                    
        if not seed_memory:
            return []
            
        # Initialize the result with the seed memory
        result = [seed_memory]
        current_level = [seed_memory]
        
        # Process each level up to the specified depth
        for _ in range(depth):
            next_level = []
            
            # Process each memory in the current level
            for memory in current_level:
                memory_text = self._extract_memory_text(memory)
                if not memory_text:
                    continue
                    
                # Find associated memories based on association type
                associated_memories = []
                if association_type == "semantic":
                    # Semantic associations (content-based)
                    associated_memories = await self._find_semantic_associations(
                        memory_text, memory_types, breadth, min_similarity
                    )
                elif association_type == "temporal":
                    # Temporal associations (time-based)
                    associated_memories = await self._find_temporal_associations(
                        memory, memory_types, breadth
                    )
                elif association_type == "causal":
                    # Causal associations (cause-effect relationships)
                    associated_memories = await self._find_causal_associations(
                        memory, memory_types, breadth
                    )
                
                # Add to next level and results if not already included
                for associated_memory in associated_memories:
                    if not any(m.get("id") == associated_memory.get("id") for m in result):
                        associated_memory["association_type"] = association_type
                        associated_memory["associated_with"] = memory.get("id")
                        result.append(associated_memory)
                        next_level.append(associated_memory)
            
            # Update current level for next iteration
            current_level = next_level
            if not current_level:
                break
                
        return result
    
    def _extract_memory_text(self, memory: Dict[str, Any]) -> str:
        """Extract textual content from a memory for association purposes.
        
        Args:
            memory: Memory object to extract text from
            
        Returns:
            Extracted text content
        """
        # Extract based on memory type
        memory_type = memory.get("memory_type")
        
        if memory_type == "episodic":
            # For episodic, concatenate event descriptions
            events = memory.get("events", [])
            event_texts = []
            for event in events:
                event_text = event.get("description", "")
                if "fact" in event:
                    event_text += " " + event["fact"]
                event_texts.append(event_text)
            return " ".join(event_texts)
            
        elif memory_type == "semantic":
            # For semantic, use the text field
            return memory.get("text", "")
            
        elif memory_type == "procedural":
            # For procedural, use description and step descriptions
            text = memory.get("description", "")
            steps = memory.get("steps", [])
            if steps:
                step_texts = [step.get("description", "") for step in steps]
                text += " " + " ".join(step_texts)
            return text
            
        # Default case
        return str(memory.get("content", memory.get("text", "")))
    
    async def _find_semantic_associations(
        self,
        text: str,
        memory_types: List[str],
        limit: int,
        min_similarity: float
    ) -> List[Dict[str, Any]]:
        """Find memories semantically associated with the given text.
        
        Args:
            text: Text to find associations for
            memory_types: Types of memories to search
            limit: Maximum number of associations to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of associated memories
        """
        if not self.embedding_provider:
            return []
            
        # Create embedding for the text
        try:
            text_embedding = self.embedding_provider.get_embedding(text)
        except:
            return []
            
        all_results = []
        
        # Search each memory type
        for memory_type in memory_types:
            if memory_type == "episodic" and hasattr(self, "episodic"):
                results = await self.episodic.search_by_embedding(text_embedding, limit=limit)
                for result in results:
                    if result.get("similarity", 0) >= min_similarity:
                        result["memory_type"] = "episodic"
                        all_results.append(result)
                        
            elif memory_type == "semantic" and hasattr(self, "semantic"):
                results = await self.semantic.search_by_embedding(text_embedding, limit=limit)
                for result in results:
                    if result.get("similarity", 0) >= min_similarity:
                        result["memory_type"] = "semantic"
                        all_results.append(result)
                        
            elif memory_type == "procedural" and hasattr(self, "procedural"):
                results = await self.procedural.search_by_embedding(text_embedding, limit=limit)
                for result in results:
                    if result.get("similarity", 0) >= min_similarity:
                        result["memory_type"] = "procedural"
                        all_results.append(result)
        
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return all_results[:limit]
    
    async def _find_temporal_associations(
        self,
        memory: Dict[str, Any],
        memory_types: List[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find memories temporally associated with the given memory.
        
        Args:
            memory: Memory to find temporal associations for
            memory_types: Types of memories to search
            limit: Maximum number of associations to return
            
        Returns:
            List of temporally associated memories
        """
        # Extract timestamp from the memory
        memory_time = memory.get("timestamp", memory.get("created_at"))
        if not memory_time:
            return []
            
        # Convert string timestamps to numeric if needed
        if isinstance(memory_time, str):
            try:
                from datetime import datetime
                memory_time = datetime.fromisoformat(memory_time).timestamp()
            except (ValueError, ImportError):
                try:
                    memory_time = float(memory_time)
                except ValueError:
                    return []
        
        # Define time windows (30 minutes before and after)
        time_window = 1800  # 30 minutes in seconds
        time_before = memory_time - time_window
        time_after = memory_time + time_window
        
        all_results = []
        
        # Search each memory type
        for memory_type in memory_types:
            if memory_type == "episodic" and hasattr(self, "episodic"):
                # For episodic, find episodes in the time window
                episodes = await self.episodic.get(
                    {"time_range": {"start": time_before, "end": time_after}},
                    limit=limit
                )
                for episode in episodes:
                    # Skip the original memory
                    if episode.get("id") == memory.get("id"):
                        continue
                    episode["memory_type"] = "episodic"
                    all_results.append(episode)
            
            # Note: Semantic and procedural memories typically don't have strong
            # temporal associations, but we could use creation timestamps
                
        return all_results[:limit]
    
    async def _find_causal_associations(
        self,
        memory: Dict[str, Any],
        memory_types: List[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find memories causally associated with the given memory.
        
        Args:
            memory: Memory to find causal associations for
            memory_types: Types of memories to search
            limit: Maximum number of associations to return
            
        Returns:
            List of causally associated memories
        """
        # Causal relationships need to be explicitly modeled
        # This is a simplified implementation that looks for explicit cause/effect tags
        
        all_results = []
        
        # Look for explicit causal links
        causes = memory.get("causes", [])
        effects = memory.get("effects", [])
        
        if causes or effects:
            # Find memories by ID
            for memory_id in causes + effects:
                for memory_type in memory_types:
                    if memory_type == "episodic" and hasattr(self, "episodic"):
                        try:
                            episode = await self.episodic.get_episode(memory_id)
                            if episode:
                                episode_dict = episode.to_dict()
                                episode_dict["memory_type"] = "episodic"
                                episode_dict["causal_relation"] = "cause" if memory_id in causes else "effect"
                                all_results.append(episode_dict)
                                if len(all_results) >= limit:
                                    return all_results
                        except:
                            pass
                    
                    elif memory_type == "semantic" and hasattr(self, "semantic"):
                        try:
                            node = await self.semantic.get_node(memory_id)
                            if node:
                                node_dict = node.to_dict() if hasattr(node, "to_dict") else node
                                node_dict["memory_type"] = "semantic"
                                node_dict["causal_relation"] = "cause" if memory_id in causes else "effect"
                                all_results.append(node_dict)
                                if len(all_results) >= limit:
                                    return all_results
                        except:
                            pass
        
        # For additional causal relationships, could implement keyword-based search
        # For example, looking for phrases like "caused by", "resulted in", etc.
        
        return all_results[:limit]
    
    async def reflect(
        self,
        time_period: Optional[Dict[str, float]] = None,
        reflection_level: MemoryReflectionLevel = MemoryReflectionLevel.BASIC,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Analyze memories to extract patterns, insights, and connections.
        
        This simulates the process of reflection and consolidation that occurs
        during human memory consolidation.
        
        Args:
            time_period: Optional time range to analyze memories from
            reflection_level: Depth of reflection to perform
            limit: Maximum number of memories to analyze
            
        Returns:
            Dictionary containing reflection results
        """
        # Get memories to reflect on
        memories = await self._get_memories_for_reflection(time_period, limit)
        
        if not memories:
            return {"insights": [], "patterns": [], "connections": []}
            
        # Perform reflection based on the specified level
        if reflection_level == MemoryReflectionLevel.NONE:
            return {"memories_analyzed": len(memories)}
            
        elif reflection_level == MemoryReflectionLevel.BASIC:
            return await self._perform_basic_reflection(memories)
            
        elif reflection_level == MemoryReflectionLevel.ADVANCED:
            return await self._perform_advanced_reflection(memories)
            
        elif reflection_level == MemoryReflectionLevel.FULL:
            return await self._perform_full_reflection(memories)
        
        return {"insights": [], "patterns": [], "connections": []}
    
    async def _get_memories_for_reflection(
        self,
        time_period: Optional[Dict[str, float]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get memories for reflection within a specified time period.
        
        Args:
            time_period: Optional time range (start and end timestamps)
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memories for reflection
        """
        all_memories = []
        
        # Query parameters
        query = {}
        if time_period and "start" in time_period and "end" in time_period:
            query["time_range"] = time_period
        
        # Get episodic memories
        if hasattr(self, "episodic"):
            episodic_memories = await self.episodic.get(query, limit=limit)
            for memory in episodic_memories:
                memory["memory_type"] = "episodic"
                all_memories.append(memory)
        
        # Get semantic memories
        if hasattr(self, "semantic"):
            semantic_memories = await self.semantic.get(query, limit=limit)
            for memory in semantic_memories:
                memory["memory_type"] = "semantic"
                all_memories.append(memory)
        
        # Get procedural memories
        if hasattr(self, "procedural"):
            procedural_memories = await self.procedural.get(query, limit=limit)
            for memory in procedural_memories:
                memory["memory_type"] = "procedural"
                all_memories.append(memory)
        
        # Sort by timestamp if available
        all_memories.sort(
            key=lambda x: x.get("timestamp", x.get("created_at", 0)), 
            reverse=True
        )
        
        return all_memories[:limit]
    
    async def _perform_basic_reflection(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform basic reflection on memories.
        
        Args:
            memories: List of memories to reflect on
            
        Returns:
            Dictionary with basic reflection results
        """
        # Basic statistics
        memory_types = defaultdict(int)
        topics = defaultdict(int)
        entities = defaultdict(int)
        
        for memory in memories:
            memory_type = memory.get("memory_type", "unknown")
            memory_types[memory_type] += 1
            
            # Extract topics and entities (simplified implementation)
            content = self._extract_memory_text(memory)
            
            # Simple topic extraction based on metadata
            if "topics" in memory:
                for topic in memory.get("topics", []):
                    topics[topic] += 1
                    
            # Simple entity extraction based on metadata
            if "entities" in memory:
                for entity in memory.get("entities", []):
                    entity_name = entity if isinstance(entity, str) else entity.get("name", "")
                    if entity_name:
                        entities[entity_name] += 1
        
        # Find most common topics and entities
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "memories_analyzed": len(memories),
            "memory_type_distribution": dict(memory_types),
            "top_topics": top_topics,
            "top_entities": top_entities
        }
    
    async def _perform_advanced_reflection(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform advanced reflection on memories with pattern detection.
        
        Args:
            memories: List of memories to reflect on
            
        Returns:
            Dictionary with advanced reflection results
        """
        # First get basic reflection
        basic_results = await self._perform_basic_reflection(memories)
        
        # Pattern detection
        patterns = []
        connections = []
        
        # Simple co-occurrence patterns (entities that appear together)
        entity_pairs = defaultdict(int)
        
        for memory in memories:
            entities = memory.get("entities", [])
            entity_names = [e if isinstance(e, str) else e.get("name", "") for e in entities]
            entity_names = [e for e in entity_names if e]  # Filter out empty names
            
            # Count co-occurrences
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    pair = tuple(sorted([entity_names[i], entity_names[j]]))
                    entity_pairs[pair] += 1
        
        # Extract top entity co-occurrences
        top_pairs = sorted(entity_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        for (entity1, entity2), count in top_pairs:
            connections.append({
                "type": "co-occurrence",
                "entities": [entity1, entity2],
                "strength": count
            })
        
        # Look for temporal patterns (events that frequently happen in sequence)
        if len(memories) >= 2:
            for i in range(len(memories) - 1):
                if memories[i].get("memory_type") == "episodic" and memories[i+1].get("memory_type") == "episodic":
                    # Check for sequential patterns
                    curr_memory = memories[i]
                    next_memory = memories[i+1]
                    
                    # Extract memory properties for comparison
                    curr_actions = curr_memory.get("actions", [])
                    next_actions = next_memory.get("actions", [])
                    
                    if curr_actions and next_actions:
                        patterns.append({
                            "type": "sequential",
                            "description": f"Action sequence: {curr_actions[0]} -> {next_actions[0]}",
                            "memories": [curr_memory.get("id"), next_memory.get("id")]
                        })
        
        # Return combined results
        return {
            **basic_results,
            "patterns": patterns,
            "connections": connections
        }
    
    async def _perform_full_reflection(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform full reflection with deep insights and connections.
        
        Args:
            memories: List of memories to reflect on
            
        Returns:
            Dictionary with full reflection results
        """
        # Build on advanced reflection
        advanced_results = await self._perform_advanced_reflection(memories)
        
        # This would typically use an LLM to generate insights, but we'll simulate
        # with a simple implementation that looks for cause-effect relationships
        insights = []
        hypotheses = []
        
        # Look for cause-effect relationships
        for memory in memories:
            if memory.get("causes") or memory.get("effects"):
                insights.append({
                    "type": "causal_relationship",
                    "description": f"Causal relationship found for memory {memory.get('id')}",
                    "memory_id": memory.get("id"),
                    "causes": memory.get("causes", []),
                    "effects": memory.get("effects", [])
                })
        
        # Generate hypotheses about unseen connections
        # This is a placeholder for what would typically be LLM-generated content
        if len(advanced_results.get("connections", [])) >= 2:
            hypotheses.append({
                "type": "predicted_relationship",
                "description": "Potential indirect relationship between frequently co-occurring entities",
                "confidence": 0.6
            })
        
        # Return combined results
        return {
            **advanced_results,
            "insights": insights,
            "hypotheses": hypotheses
        }
