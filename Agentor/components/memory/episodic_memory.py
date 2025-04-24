"""
Episodic Memory implementation for the Agentor framework.

Episodic memory stores sequences of events or experiences that the agent has encountered.
It allows for temporal reasoning and recalling past experiences in a sequential manner.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import time
import json
import numpy as np
import asyncio
from datetime import datetime
from dataclasses import dataclass, field

from agentor.components.memory import Memory
from agentor.llm_gateway.utils.metrics import track_memory_operation


@dataclass
class Episode:
    """A single episode in episodic memory."""
    
    id: str
    """Unique identifier for the episode."""
    
    events: List[Dict[str, Any]] = field(default_factory=list)
    """List of events in the episode."""
    
    start_time: float = field(default_factory=time.time)
    """When the episode started."""
    
    end_time: Optional[float] = None
    """When the episode ended, or None if ongoing."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the episode."""
    
    importance: float = 0.0
    """Importance score for this episode (0.0 to 1.0)."""
    
    embedding: Optional[List[float]] = None
    """Vector embedding of the episode for semantic search."""
    
    related_episodes: List[str] = field(default_factory=list)
    """List of related episode IDs."""
    
    decay_factor: float = 0.95
    """Memory decay factor (0.0 to 1.0) - higher values decay more slowly."""

    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to this episode.
        
        Args:
            event: The event to add
        """
        if 'timestamp' not in event:
            event['timestamp'] = time.time()
        
        self.events.append(event)
        self.end_time = event['timestamp']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the episode to a dictionary.
        
        Returns:
            Dictionary representation of the episode
        """
        return {
            'id': self.id,
            'events': self.events,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metadata': self.metadata,
            'importance': self.importance,
            'related_episodes': self.related_episodes,
            'decay_factor': self.decay_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create an episode from a dictionary.
        
        Args:
            data: Dictionary representation of an episode
            
        Returns:
            An Episode object
        """
        episode = cls(
            id=data['id'],
            events=data['events'],
            start_time=data['start_time'],
            end_time=data['end_time'],
            metadata=data['metadata'],
            importance=data['importance']
        )
        # Add optional fields if they exist
        if 'related_episodes' in data:
            episode.related_episodes = data['related_episodes']
        if 'decay_factor' in data:
            episode.decay_factor = data['decay_factor']
        if 'embedding' in data:
            episode.embedding = data['embedding']
        return episode
    
    def get_summary(self) -> str:
        """Generate a summary of this episode.
        
        Returns:
            A string summary of the episode
        """
        start_time_str = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
        
        if self.end_time:
            end_time_str = datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')
            duration = self.end_time - self.start_time
            duration_str = f"{duration:.1f} seconds"
        else:
            end_time_str = "ongoing"
            duration_str = "ongoing"
        
        event_count = len(self.events)
        
        summary = f"Episode {self.id}\n"
        summary += f"Time: {start_time_str} to {end_time_str} ({duration_str})\n"
        summary += f"Events: {event_count}\n"
        
        if self.metadata.get('title'):
            summary += f"Title: {self.metadata['title']}\n"
        
        if self.importance > 0:
            summary += f"Importance: {self.importance:.2f}\n"
        
        if self.related_episodes:
            summary += f"Related episodes: {', '.join(self.related_episodes[:3])}"
            if len(self.related_episodes) > 3:
                summary += f" and {len(self.related_episodes) - 3} more"
            summary += "\n"
        
        return summary
    
    def calculate_recency_score(self, current_time: float) -> float:
        """Calculate a recency score based on time elapsed and decay factor.
        
        Args:
            current_time: Current timestamp to compare against
            
        Returns:
            A score between 0 and 1, where 1 is most recent
        """
        # If the episode is ongoing, it's maximally recent
        if self.end_time is None:
            return 1.0
            
        # Calculate elapsed time in hours
        elapsed_hours = (current_time - self.end_time) / 3600
        
        # Apply decay function: score = decay_factor^elapsed_hours
        # This creates an exponential decay curve
        score = self.decay_factor ** elapsed_hours
        
        # Ensure the score is between 0 and 1
        return max(0.0, min(1.0, score))
        
    def add_related_episode(self, episode_id: str) -> None:
        """Add a related episode ID if not already present.
        
        Args:
            episode_id: ID of the related episode
        """
        if episode_id not in self.related_episodes:
            self.related_episodes.append(episode_id)


class EpisodicMemory(Memory):
    """Episodic memory implementation that stores sequences of events."""
    
    def __init__(
        self,
        embedding_provider=None,
        max_episodes: int = 100,
        forgetting_threshold: float = 0.2,
        consolidation_interval: int = 3600,  # 1 hour in seconds
    ):
        """Initialize the episodic memory.
        
        Args:
            embedding_provider: Provider for generating embeddings
            max_episodes: Maximum number of episodes to store
            forgetting_threshold: Importance threshold below which episodes may be forgotten
            consolidation_interval: How often to consolidate memories (in seconds)
        """
        self.episodes: Dict[str, Episode] = {}
        self.current_episode: Optional[Episode] = None
        self.embedding_provider = embedding_provider
        self.max_episodes = max_episodes
        self.forgetting_threshold = forgetting_threshold
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = time.time()
        self.lock = asyncio.Lock()
    
    @track_memory_operation("create", "episodic")
    async def create_episode(self, episode_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Episode:
        """Create a new episode.
        
        Args:
            episode_id: Optional ID for the episode, generated if not provided
            metadata: Optional metadata for the episode
            
        Returns:
            The created episode
        """
        async with self.lock:
            # Generate ID if not provided
            if episode_id is None:
                episode_id = f"ep-{int(time.time())}-{hash(str(time.time()))}"[:16]
            
            # Create the episode
            episode = Episode(
                id=episode_id,
                metadata=metadata or {}
            )
            
            # Store the episode
            self.episodes[episode_id] = episode
            
            # Set as current episode
            self.current_episode = episode
            
            return episode
    
    @track_memory_operation("add", "episodic")
    async def add(self, item: Dict[str, Any], episode_id: Optional[str] = None):
        """Add an event to an episode.
        
        Args:
            item: The event to add
            episode_id: The ID of the episode to add to, or None for current episode
        """
        async with self.lock:
            # If no episode ID provided, use current episode
            if episode_id is None:
                if self.current_episode is None:
                    # Create a new episode if none exists
                    await self.create_episode()
                
                episode = self.current_episode
            else:
                # Get the specified episode
                episode = self.episodes.get(episode_id)
                if episode is None:
                    raise ValueError(f"Episode {episode_id} not found")
            
            # Add the event to the episode
            episode.add_event(item)
            
            # Check if we need to consolidate memories
            if time.time() - self.last_consolidation > self.consolidation_interval:
                await self.consolidate_memories()
    
    @track_memory_operation("get", "episodic")
    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get episodes that match a query.
        
        Args:
            query: The query to match
            limit: Maximum number of episodes to return
            
        Returns:
            A list of matching episodes as dictionaries
        """
        results = []
        
        # If querying by episode ID
        if 'episode_id' in query:
            episode = self.episodes.get(query['episode_id'])
            if episode:
                results.append(episode.to_dict())
            return results[:limit]
        
        # If querying by time range
        if 'start_time' in query or 'end_time' in query:
            start_time = query.get('start_time', 0)
            end_time = query.get('end_time', float('inf'))
            
            for episode in self.episodes.values():
                if episode.start_time >= start_time and (episode.end_time is None or episode.end_time <= end_time):
                    results.append(episode.to_dict())
        
        # If querying by semantic similarity
        elif 'text' in query and self.embedding_provider:
            query_embedding = await self.embedding_provider.get_embedding(query['text'])
            
            # Get episodes with embeddings
            episodes_with_embeddings = [
                (episode, episode.embedding)
                for episode in self.episodes.values()
                if episode.embedding is not None
            ]
            
            if episodes_with_embeddings:
                # Calculate similarities
                similarities = [
                    (self._cosine_similarity(query_embedding, embedding), episode)
                    for episode, embedding in episodes_with_embeddings
                ]
                
                # Sort by similarity
                similarities.sort(reverse=True, key=lambda x: x[0])
                
                # Get top results
                for similarity, episode in similarities[:limit]:
                    if similarity >= query.get('threshold', 0.0):
                        results.append(episode.to_dict())
        
        # Otherwise, return most recent episodes
        else:
            sorted_episodes = sorted(
                self.episodes.values(),
                key=lambda ep: ep.end_time or float('inf'),
                reverse=True
            )
            
            for episode in sorted_episodes[:limit]:
                results.append(episode.to_dict())
        
        return results
    
    @track_memory_operation("clear", "episodic")
    async def clear(self):
        """Clear all episodes from memory."""
        async with self.lock:
            self.episodes = {}
            self.current_episode = None
    
    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID.
        
        Args:
            episode_id: The ID of the episode to get
            
        Returns:
            The episode, or None if not found
        """
        return self.episodes.get(episode_id)
    
    async def end_episode(self, episode_id: Optional[str] = None) -> Optional[Episode]:
        """End an episode.
        
        Args:
            episode_id: The ID of the episode to end, or None for current episode
            
        Returns:
            The ended episode, or None if not found
        """
        async with self.lock:
            # If no episode ID provided, use current episode
            if episode_id is None:
                if self.current_episode is None:
                    return None
                
                episode = self.current_episode
                self.current_episode = None
            else:
                # Get the specified episode
                episode = self.episodes.get(episode_id)
                if episode is None:
                    return None
                
                # If ending the current episode, clear it
                if self.current_episode and self.current_episode.id == episode_id:
                    self.current_episode = None
            
            # Set the end time if not already set
            if episode.end_time is None:
                episode.end_time = time.time()
            
            # Generate embedding if we have a provider
            if self.embedding_provider and not episode.embedding:
                # Create a text representation of the episode
                episode_text = self._episode_to_text(episode)
                
                # Generate the embedding
                episode.embedding = await self.embedding_provider.get_embedding(episode_text)
            
            return episode
    
    async def update_importance(self, episode_id: str, importance: float) -> bool:
        """Update the importance score of an episode.
        
        Args:
            episode_id: The ID of the episode to update
            importance: The new importance score (0.0 to 1.0)
            
        Returns:
            True if the episode was updated, False otherwise
        """
        async with self.lock:
            episode = self.episodes.get(episode_id)
            if episode is None:
                return False
            
            episode.importance = max(0.0, min(1.0, importance))
            return True
    
    async def consolidate_memories(self):
        """Consolidate memories by removing less important episodes if needed."""
        async with self.lock:
            # Only consolidate if we have more episodes than the maximum
            if len(self.episodes) <= self.max_episodes:
                self.last_consolidation = time.time()
                return
            
            # Sort episodes by importance
            sorted_episodes = sorted(
                self.episodes.values(),
                key=lambda ep: ep.importance
            )
            
            # Remove the least important episodes
            episodes_to_remove = len(self.episodes) - self.max_episodes
            for episode in sorted_episodes[:episodes_to_remove]:
                # Only remove episodes below the forgetting threshold
                if episode.importance <= self.forgetting_threshold:
                    del self.episodes[episode.id]
            
            self.last_consolidation = time.time()
    
    def _episode_to_text(self, episode: Episode) -> str:
        """Convert an episode to a text representation for embedding.
        
        Args:
            episode: The episode to convert
            
        Returns:
            A text representation of the episode
        """
        # Start with metadata
        text_parts = []
        
        if episode.metadata.get('title'):
            text_parts.append(f"Title: {episode.metadata['title']}")
        
        # Add event descriptions
        for i, event in enumerate(episode.events):
            event_text = f"Event {i+1}: "
            
            if 'action' in event:
                event_text += f"Action: {event['action']} "
            
            if 'observation' in event:
                event_text += f"Observation: {event['observation']} "
            
            if 'result' in event:
                event_text += f"Result: {event['result']}"
            
            text_parts.append(event_text)
        
        return "\n".join(text_parts)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            The cosine similarity (between -1 and 1)
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
    
    @track_memory_operation("search_temporal_sequence", "episodic")
    async def search_temporal_sequence(
        self, 
        pattern: List[Dict[str, Any]], 
        match_threshold: float = 0.7, 
        recency_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for episodes matching a temporal pattern of events.
        
        Args:
            pattern: A list of event patterns to match in sequence
            match_threshold: Minimum similarity threshold for a match (0.0 to 1.0)
            recency_weight: Weight to give to recency in scoring (0.0 to 1.0)
            
        Returns:
            A list of matching episodes with match score
        """
        results = []
        current_time = time.time()
        
        for episode in self.episodes.values():
            if len(episode.events) < len(pattern):
                continue
                
            # Calculate pattern match score
            match_score = self._calculate_pattern_match(episode.events, pattern)
            
            if match_score >= match_threshold:
                # Calculate recency score
                recency_score = episode.calculate_recency_score(current_time)
                
                # Calculate combined score
                combined_score = (match_score * (1 - recency_weight)) + (recency_score * recency_weight)
                
                results.append({
                    'episode': episode.to_dict(),
                    'match_score': match_score,
                    'recency_score': recency_score,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def _calculate_pattern_match(self, events: List[Dict[str, Any]], pattern: List[Dict[str, Any]]) -> float:
        """Calculate how well a sequence of events matches a pattern.
        
        Args:
            events: The sequence of events to check
            pattern: The pattern to match against
            
        Returns:
            A match score between 0.0 and 1.0
        """
        # Implement a sliding window pattern matching algorithm
        if not events or not pattern:
            return 0.0
            
        max_score = 0.0
        
        # Try matching the pattern at different starting positions
        for i in range(len(events) - len(pattern) + 1):
            window = events[i:i+len(pattern)]
            score = 0.0
            
            # Calculate match score for this window
            for j, pattern_event in enumerate(pattern):
                event = window[j]
                event_score = self._match_event_to_pattern(event, pattern_event)
                score += event_score
            
            score /= len(pattern)  # Normalize to 0.0-1.0
            max_score = max(max_score, score)
        
        return max_score
    
    def _match_event_to_pattern(self, event: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calculate how well an event matches a pattern.
        
        Args:
            event: The event to check
            pattern: The pattern to match against
            
        Returns:
            A match score between 0.0 and 1.0
        """
        # No pattern keys means any event matches
        if not pattern:
            return 1.0
            
        # Count how many pattern keys match the event
        matches = 0
        total = 0
        
        for key, pattern_value in pattern.items():
            if key not in event:
                continue
                
            total += 1
            event_value = event[key]
            
            # Exact match
            if event_value == pattern_value:
                matches += 1
            # String partial match
            elif isinstance(event_value, str) and isinstance(pattern_value, str):
                if pattern_value.lower() in event_value.lower():
                    matches += 0.5
        
        # Avoid division by zero
        if total == 0:
            return 0.0
            
        return matches / total
    
    @track_memory_operation("reinforce", "episodic")
    async def reinforce_episode(self, episode_id: str, reinforcement_factor: float = 0.1) -> bool:
        """Reinforce an episode by increasing its importance.
        
        Args:
            episode_id: The ID of the episode to reinforce
            reinforcement_factor: How much to increase importance (0.0 to 1.0)
            
        Returns:
            True if the episode was reinforced, False otherwise
        """
        async with self.lock:
            episode = self.episodes.get(episode_id)
            if episode is None:
                return False
            
            # Increase importance with diminishing returns
            new_importance = episode.importance + (1 - episode.importance) * reinforcement_factor
            episode.importance = min(1.0, new_importance)
            
            return True
    
    @track_memory_operation("link_episodes", "episodic")
    async def link_episodes(self, source_id: str, target_id: str) -> bool:
        """Link two episodes as related to each other.
        
        Args:
            source_id: ID of the source episode
            target_id: ID of the target episode
            
        Returns:
            True if episodes were linked, False otherwise
        """
        async with self.lock:
            source = self.episodes.get(source_id)
            target = self.episodes.get(target_id)
            
            if source is None or target is None:
                return False
            
            # Add bidirectional links
            source.add_related_episode(target_id)
            target.add_related_episode(source_id)
            
            return True
    
    @track_memory_operation("get_context_aware", "episodic")
    async def get_context_aware(
        self,
        context: Dict[str, Any],
        limit: int = 5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        similarity_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Get episodes based on context-aware relevance scoring.
        
        Args:
            context: The current context (can include text, time, etc.)
            limit: Maximum number of episodes to return
            recency_weight: Weight for recency in scoring
            importance_weight: Weight for importance in scoring
            similarity_weight: Weight for semantic similarity in scoring
            
        Returns:
            A list of relevant episodes as dictionaries
        """
        scored_episodes = []
        current_time = time.time()
        
        # Get context embedding if we have text and a provider
        context_embedding = None
        if 'text' in context and self.embedding_provider:
            context_embedding = await self.embedding_provider.get_embedding(context['text'])
        
        # Score each episode
        for episode in self.episodes.values():
            # Calculate recency score
            recency_score = episode.calculate_recency_score(current_time)
            
            # Importance score is directly from the episode
            importance_score = episode.importance
            
            # Calculate similarity score if we have embeddings
            similarity_score = 0.0
            if context_embedding and episode.embedding:
                similarity_score = self._cosine_similarity(context_embedding, episode.embedding)
            
            # Calculate combined relevance score
            relevance_score = (
                (recency_score * recency_weight) +
                (importance_score * importance_weight) +
                (similarity_score * similarity_weight)
            )
            
            scored_episodes.append((relevance_score, episode))
        
        # Sort by relevance score
        scored_episodes.sort(reverse=True, key=lambda x: x[0])
        
        # Return top episodes
        results = []
        for score, episode in scored_episodes[:limit]:
            result = episode.to_dict()
            result['relevance_score'] = score
            results.append(result)
        
        return results
