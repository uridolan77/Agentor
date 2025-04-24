"""
Temporal Coordination Patterns for Multi-Agent Systems.

This module provides coordination patterns that leverage temporal memory
for more effective agent coordination and decision making across time.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta

from agentor.agents import Agent, AgentInput, AgentOutput
from agentor.components.coordination import AgentGroup, AgentMessage
from agentor.components.memory.temporal_memory import TemporalMemory, TemporalMemoryNode

logger = logging.getLogger(__name__)


class TemporalCoordinator:
    """Coordinator that uses temporal memory to make time-aware decisions.
    
    This coordinator enhances agent coordination by:
    1. Preserving conversation history with temporal decay
    2. Prioritizing information based on recency and importance
    3. Supporting time-based retrieval of relevant context
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        max_memory_nodes: int = 1000,
        memory_decay_rate: float = 0.05,
        retention_period: int = 30  # days
    ):
        """Initialize the temporal coordinator.
        
        Args:
            name: The name of the coordinator
            max_memory_nodes: Maximum number of memory nodes
            memory_decay_rate: Rate at which memories decay over time
            retention_period: Maximum number of days to retain memories
        """
        self.name = name or f"TemporalCoordinator-{id(self)}"
        self.agents: Dict[str, Agent] = {}
        self.agent_groups: Dict[str, AgentGroup] = {}
        self.memory = TemporalMemory(
            max_nodes=max_memory_nodes,
            default_decay_rate=memory_decay_rate,
            retention_period=retention_period
        )
        
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the coordinator.
        
        Args:
            agent: The agent to add
        """
        self.agents[agent.name] = agent
        logger.info(f"Added agent {agent.name} to coordinator {self.name}")
        
    def create_group(self, name: str, agents: List[Agent] = None) -> AgentGroup:
        """Create a new agent group.
        
        Args:
            name: The name of the group
            agents: Optional list of agents to add to the group
            
        Returns:
            The created agent group
        """
        group = AgentGroup(name=name)
        self.agent_groups[name] = group
        
        # Add agents if provided
        if agents:
            for agent in agents:
                if agent.name not in self.agents:
                    self.add_agent(agent)
                group.add_agent(agent)
                
        return group
        
    async def store_interaction(
        self,
        content: str,
        agents: List[str],
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an agent interaction in temporal memory.
        
        Args:
            content: The interaction content
            agents: List of agent names involved
            importance: Importance score (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            ID of the stored memory node
        """
        # Add basic metadata if not provided
        if metadata is None:
            metadata = {}
            
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "agents": agents,
            "interaction_type": "conversation"
        })
        
        # Store in memory
        node_id = await self.memory.add(
            content=content,
            importance=importance,
            metadata=metadata
        )
        
        return node_id
        
    async def retrieve_context(
        self,
        query: str,
        agent_names: Optional[List[str]] = None,
        time_window: Optional[int] = None,  # hours
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant temporal context for coordination.
        
        Args:
            query: The query to match
            agent_names: Optional filter by agent names
            time_window: Optional time window in hours
            limit: Maximum number of results
            
        Returns:
            List of relevant memory entries
        """
        # Set up time range if specified
        time_range = None
        if time_window:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window)
            time_range = (start_time, end_time)
            
        # Search memory
        nodes = await self.memory.search(
            query=query,
            limit=limit,
            time_range=time_range
        )
        
        # Filter by agent names if specified
        if agent_names:
            filtered_nodes = []
            for node in nodes:
                node_agents = node.metadata.get("agents", [])
                if any(agent in agent_names for agent in node_agents):
                    filtered_nodes.append(node)
            nodes = filtered_nodes
            
        # Convert to dicts for easier handling
        results = []
        for node in nodes:
            results.append({
                "id": node.id,
                "content": node.content,
                "importance": node.get_current_importance(),
                "created_at": node.created_at.isoformat(),
                "last_accessed": node.last_accessed.isoformat(),
                "metadata": node.metadata
            })
            
        return results
        
    async def coordinate_with_memory(
        self,
        task: str,
        group_name: str,
        context: Optional[Dict[str, Any]] = None,
        time_window: Optional[int] = None,  # hours
        max_turns: int = 10
    ) -> Dict[str, Any]:
        """Coordinate agents using temporal memory for context.
        
        Args:
            task: The task to coordinate
            group_name: The name of the agent group
            context: Additional context for the coordination
            time_window: Optional time window for historical context (hours)
            max_turns: Maximum conversation turns
            
        Returns:
            Coordination results including conversation history
        """
        if group_name not in self.agent_groups:
            raise ValueError(f"Agent group {group_name} not found")
            
        group = self.agent_groups[group_name]
        
        # Create context if not provided
        if context is None:
            context = {}
            
        # Get temporal context if time window specified
        if time_window is not None:
            # Get agent names in the group
            agent_names = list(group.agents.keys())
            
            # Retrieve relevant historical context
            temporal_context = await self.retrieve_context(
                query=task,
                agent_names=agent_names,
                time_window=time_window
            )
            
            # Add to context
            context["temporal_context"] = temporal_context
        
        # Run the conversation
        conversation = await group.run_conversation(
            query=task,
            context=context,
            max_turns=max_turns
        )
        
        # Store the conversation in temporal memory
        conversation_text = "\n".join([
            f"{msg.sender} -> {msg.receiver}: {msg.content}" 
            for msg in conversation
        ])
        
        memory_id = await self.store_interaction(
            content=conversation_text,
            agents=list(group.agents.keys()),
            importance=0.7,  # Higher importance for full conversations
            metadata={"task": task, "group": group_name}
        )
        
        # Prepare the result
        result = {
            "conversation": conversation,
            "memory_id": memory_id,
            "participating_agents": list(group.agents.keys())
        }
        
        return result
        
    async def maintenance(self) -> Dict[str, int]:
        """Run maintenance on the temporal memory.
        
        Returns:
            Statistics about the maintenance operation
        """
        forgotten_count = await self.memory.run_maintenance()
        
        return {
            "forgotten_memories": forgotten_count,
            "remaining_memories": len(self.memory.nodes)
        }


class TemporalConsensus:
    """Consensus algorithm that incorporates temporal aspects.
    
    This algorithm builds consensus among agents while considering:
    1. The decay of opinions and confidence over time
    2. Changing relevance of information
    3. Agent's historical reliability
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        temporal_coordinator: Optional[TemporalCoordinator] = None,
        confidence_threshold: float = 0.6,
        opinion_decay_rate: float = 0.1,
        historical_weight: float = 0.3
    ):
        """Initialize the temporal consensus system.
        
        Args:
            name: Name of the consensus system
            temporal_coordinator: Temporal coordinator
            confidence_threshold: Minimum confidence to reach consensus
            opinion_decay_rate: Rate at which opinions decay
            historical_weight: Weight given to historical performance
        """
        self.name = name or f"TemporalConsensus-{id(self)}"
        self.coordinator = temporal_coordinator or TemporalCoordinator()
        self.confidence_threshold = confidence_threshold
        self.opinion_decay_rate = opinion_decay_rate
        self.historical_weight = historical_weight
        self.agent_reliability: Dict[str, float] = {}
        
    async def gather_opinions(
        self,
        query: str,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
        time_window: Optional[int] = None  # hours
    ) -> Dict[str, Dict[str, Any]]:
        """Gather opinions from agents, with temporal context.
        
        Args:
            query: The question to ask agents
            agents: List of agents to query
            context: Additional context
            time_window: Time window for historical context
            
        Returns:
            Dictionary mapping agent names to their opinions and metadata
        """
        if context is None:
            context = {}
            
        # Add temporal context if requested
        if time_window is not None and self.coordinator:
            agent_names = [agent.name for agent in agents]
            temporal_context = await self.coordinator.retrieve_context(
                query=query,
                agent_names=agent_names,
                time_window=time_window
            )
            context["temporal_context"] = temporal_context
            
        # Query each agent
        opinions = {}
        for agent in agents:
            # Build agent-specific context
            agent_context = context.copy()
            agent_context["request_type"] = "consensus_opinion"
            
            try:
                # Run the agent
                result = await agent.run(AgentInput(query=query, context=agent_context))
                
                # Parse response (assuming structured output)
                opinion = self._parse_opinion(result.response)
                
                # Store the opinion
                opinions[agent.name] = {
                    "value": opinion.get("value", ""),
                    "confidence": opinion.get("confidence", 0.5),
                    "reasoning": opinion.get("reasoning", ""),
                    "timestamp": datetime.now()
                }
                
                # Update agent reliability based on confidence
                if agent.name in self.agent_reliability:
                    # Smooth update
                    current = self.agent_reliability[agent.name]
                    self.agent_reliability[agent.name] = (current * 0.7) + (opinion.get("confidence", 0.5) * 0.3)
                else:
                    # Initial reliability is their stated confidence
                    self.agent_reliability[agent.name] = opinion.get("confidence", 0.5)
                    
            except Exception as e:
                logger.error(f"Error getting opinion from {agent.name}: {e}")
                opinions[agent.name] = {
                    "value": "",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "timestamp": datetime.now()
                }
                
        return opinions
        
    def _parse_opinion(self, response: str) -> Dict[str, Any]:
        """Parse an agent's opinion from their response.
        
        Args:
            response: The agent's response text
            
        Returns:
            Parsed opinion with value, confidence, and reasoning
        """
        # Default values
        opinion = {
            "value": "",
            "confidence": 0.5,
            "reasoning": ""
        }
        
        # Simple parsing - in practice would be more robust
        lines = response.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == "opinion" or key == "answer" or key == "value":
                    opinion["value"] = value
                elif key == "confidence":
                    try:
                        conf = float(value)
                        opinion["confidence"] = max(0.0, min(1.0, conf))  # Clamp to [0,1]
                    except ValueError:
                        pass
                elif key == "reasoning" or key == "explanation":
                    opinion["reasoning"] = value
                    
        return opinion
        
    def _apply_temporal_decay(self, opinions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply temporal decay to opinions based on age.
        
        Args:
            opinions: Dictionary of opinions
            
        Returns:
            Opinions with decayed confidence based on age
        """
        now = datetime.now()
        decayed_opinions = {}
        
        for agent_name, opinion in opinions.items():
            # Create a copy to avoid modifying the original
            decayed_opinion = opinion.copy()
            
            # Calculate age in hours
            timestamp = opinion.get("timestamp")
            if timestamp and isinstance(timestamp, datetime):
                hours_since_opinion = (now - timestamp).total_seconds() / 3600.0
                decay_factor = 1.0 - (self.opinion_decay_rate * hours_since_opinion)
                decay_factor = max(0.1, decay_factor)  # Limit decay
                
                # Apply decay to confidence
                decayed_opinion["confidence"] = opinion.get("confidence", 0.5) * decay_factor
                decayed_opinion["decayed"] = True
                
            decayed_opinions[agent_name] = decayed_opinion
            
        return decayed_opinions
        
    async def reach_consensus(
        self,
        query: str,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
        time_window: Optional[int] = None,  # hours
        apply_decay: bool = True
    ) -> Dict[str, Any]:
        """Reach consensus among agents with temporal awareness.
        
        Args:
            query: The question to ask agents
            agents: List of agents to query
            context: Additional context
            time_window: Time window for historical context
            apply_decay: Whether to apply temporal decay
            
        Returns:
            Consensus result with additional metadata
        """
        # Gather opinions
        opinions = await self.gather_opinions(
            query=query,
            agents=agents,
            context=context,
            time_window=time_window
        )
        
        # Apply temporal decay if requested
        if apply_decay:
            opinions = self._apply_temporal_decay(opinions)
            
        # Calculate agent weights based on reliability and confidence
        weights = {}
        for agent_name, opinion in opinions.items():
            # Base weight is the confidence
            weight = opinion.get("confidence", 0.5)
            
            # Include historical reliability if available
            if agent_name in self.agent_reliability:
                historical_factor = self.agent_reliability[agent_name]
                weight = (weight * (1 - self.historical_weight)) + (historical_factor * self.historical_weight)
                
            weights[agent_name] = weight
            
        # Find the most common opinion, weighted by confidence
        opinion_weights: Dict[str, float] = {}
        for agent_name, opinion in opinions.items():
            value = opinion.get("value", "")
            if value:
                opinion_weights[value] = opinion_weights.get(value, 0) + weights[agent_name]
                
        # Select the opinion with highest weight
        if opinion_weights:
            consensus_value = max(opinion_weights.items(), key=lambda x: x[1])[0]
            consensus_weight = opinion_weights[consensus_value]
            
            # Calculate total weight and confidence
            total_weight = sum(weights.values())
            if total_weight > 0:
                consensus_confidence = consensus_weight / total_weight
            else:
                consensus_confidence = 0.0
                
            # Check if consensus meets threshold
            consensus_reached = consensus_confidence >= self.confidence_threshold
        else:
            # No opinions to form consensus
            consensus_value = ""
            consensus_confidence = 0.0
            consensus_reached = False
            
        # Find supporting agents
        supporting_agents = [
            agent_name for agent_name, opinion in opinions.items()
            if opinion.get("value", "") == consensus_value
        ]
            
        # Prepare result
        result = {
            "query": query,
            "consensus_value": consensus_value,
            "consensus_confidence": consensus_confidence,
            "consensus_reached": consensus_reached,
            "supporting_agents": supporting_agents,
            "agent_opinions": opinions,
            "agent_weights": weights
        }
        
        # Store consensus in temporal memory if coordinator available
        if self.coordinator:
            result_str = (
                f"Query: {query}\n"
                f"Consensus: {consensus_value}\n"
                f"Confidence: {consensus_confidence:.2f}\n"
                f"Supporting agents: {', '.join(supporting_agents)}"
            )
            
            memory_id = await self.coordinator.store_interaction(
                content=result_str,
                agents=[agent.name for agent in agents],
                importance=consensus_confidence,
                metadata={"type": "consensus", "reached": consensus_reached}
            )
            
            result["memory_id"] = memory_id
            
        return result