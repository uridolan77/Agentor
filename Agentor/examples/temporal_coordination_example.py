"""
Example demonstrating temporal coordination patterns for multi-agent systems.

This example shows how to use temporal coordination to enhance agent collaboration
by leveraging time-aware decision making and historical context.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.components.coordination import AgentGroup, AgentMessage
from agentor.components.coordination.temporal_coordination import (
    TemporalCoordinator, 
    TemporalConsensus
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeAwareAgent(EnhancedAgent):
    """Agent that is aware of temporal context."""
    
    def __init__(self, name: str, expertise: str, memory_retention: float = 0.8):
        """Initialize the time-aware agent.
        
        Args:
            name: The name of the agent
            expertise: The agent's area of expertise
            memory_retention: How well the agent retains past interactions (0-1)
        """
        super().__init__(name=name)
        self.expertise = expertise
        self.memory_retention = memory_retention
        self.experience = 0  # Increases over time as the agent interacts
        
    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Run the agent.
        
        Args:
            input_data: The input data
            
        Returns:
            The agent's output
        """
        query = input_data.query
        context = input_data.context or {}
        
        # Extract temporal context if available
        temporal_context = context.get("temporal_context", [])
        historical_knowledge = ""
        
        if temporal_context:
            # Format historical knowledge from temporal context
            historical_knowledge = "\n\nHistorical Context:\n"
            for i, memory in enumerate(temporal_context, 1):
                # Only include relevant parts to avoid overwhelming the agent
                historical_knowledge += f"{i}. {memory['content'][:200]}...\n"
                
                # Add metadata like creation time if available
                if "created_at" in memory:
                    historical_knowledge += f"   (from {memory['created_at']})\n"
        
        # Handle consensus requests differently
        if context.get("request_type") == "consensus_opinion":
            # Provide a structured response for consensus algorithms
            response = self._form_consensus_opinion(query, context, historical_knowledge)
        else:
            # Normal processing with expertise factored in
            response = f"As an agent specializing in {self.expertise}, I processed: {query}"
            
            # Add temporal context influence
            if historical_knowledge:
                response += f"\n\nMy response is informed by {len(temporal_context)} historical records."
                response += "\nBased on historical context, I've determined that:"
                
                # Simulate how the agent uses historical knowledge
                if "update" in query.lower() or "status" in query.lower():
                    response += "\n- This appears to be a recurring topic."
                    response += "\n- Previous discussions have covered similar ground."
                elif "decision" in query.lower() or "choose" in query.lower():
                    response += "\n- Similar decisions have been made in the past."
                    response += "\n- Historical outcomes might inform this decision."
        
        # Simulate experience gain from the interaction
        self.experience += 0.05
        
        return AgentOutput(response=response, context=context)
    
    def _form_consensus_opinion(self, query: str, context: Dict[str, Any], historical_knowledge: str) -> str:
        """Form a structured opinion for consensus algorithms.
        
        Args:
            query: The query to respond to
            context: Additional context
            historical_knowledge: Formatted historical knowledge
            
        Returns:
            Structured opinion response
        """
        # This is a simplified simulation of how the agent would form an opinion
        # In a real system, this would use the agent's LLM or other reasoning abilities
        
        # Simulate different opinions based on expertise
        if self.expertise == "research":
            value = "Option A is the best approach"
            confidence = min(0.7 + (self.experience * 0.1), 0.9)  # Experience improves confidence
            reasoning = "Based on research data and historical patterns"
        elif self.expertise == "engineering":
            value = "Option B provides better technical implementation"
            confidence = min(0.65 + (self.experience * 0.15), 0.85)
            reasoning = "Technical analysis suggests better maintainability"
        elif self.expertise == "design":
            value = "Option A offers better user experience"
            confidence = min(0.6 + (self.experience * 0.1), 0.8)
            reasoning = "User-centered design principles favor this approach"
        else:
            value = "Option B seems preferable"
            confidence = 0.5 + (self.experience * 0.1)
            reasoning = "General assessment based on available information"
        
        # If we have significant historical knowledge, potentially adjust the opinion
        if historical_knowledge and len(historical_knowledge) > 100:
            # Simulate the agent being influenced by history
            if "Option A" in historical_knowledge:
                confidence += 0.1
                reasoning += " with additional support from historical data"
            elif "Option B" in historical_knowledge:
                confidence += 0.05
                reasoning += " partially supported by historical context"
                
        # Format the response in a structured way for the consensus algorithm to parse
        response = f"Opinion: {value}\nConfidence: {confidence:.2f}\nReasoning: {reasoning}"
        
        return response


async def temporal_coordination_example():
    """Example demonstrating temporal coordination."""
    logger.info("\n=== Temporal Coordination Example ===")
    
    # Create a temporal coordinator
    coordinator = TemporalCoordinator(
        name="ProjectCoordinator",
        max_memory_nodes=500,
        memory_decay_rate=0.1,
        retention_period=14  # days
    )
    
    # Create time-aware agents with different expertise
    research_agent = TimeAwareAgent("ResearchAgent", "research")
    engineering_agent = TimeAwareAgent("EngineeringAgent", "engineering")
    design_agent = TimeAwareAgent("DesignAgent", "design")
    
    # Add agents to coordinator
    coordinator.add_agent(research_agent)
    coordinator.add_agent(engineering_agent)
    coordinator.add_agent(design_agent)
    
    # Create an agent group for the project team
    project_group = coordinator.create_group(
        name="ProjectTeam",
        agents=[research_agent, engineering_agent, design_agent]
    )
    
    # Connect agents in a fully connected network
    project_group.connect("ResearchAgent", "EngineeringAgent")
    project_group.connect("ResearchAgent", "DesignAgent")
    project_group.connect("EngineeringAgent", "DesignAgent")
    
    # Simulate some past interactions to build history
    logger.info("Building interaction history...")
    await coordinator.store_interaction(
        content="Team discussed initial requirements for the project. ResearchAgent presented market analysis.",
        agents=["ResearchAgent", "EngineeringAgent", "DesignAgent"],
        importance=0.8,
        metadata={"meeting_type": "planning", "phase": "initial"}
    )
    
    # Simulate a past decision with high importance
    yesterday = datetime.now() - timedelta(days=1)
    await coordinator.store_interaction(
        content="Team decided to focus on Option A for the first prototype based on research findings.",
        agents=["ResearchAgent", "EngineeringAgent", "DesignAgent"],
        importance=0.9,
        metadata={"meeting_type": "decision", "decision": "Option A", "timestamp": yesterday.isoformat()}
    )
    
    # Current task: Coordinate the team with access to historical context
    logger.info("\nCoordinating team with temporal context...")
    result = await coordinator.coordinate_with_memory(
        task="Provide a status update on the prototype development and suggest next steps.",
        group_name="ProjectTeam",
        time_window=48  # Look back 48 hours for context
    )
    
    # Display the results
    logger.info(f"Coordination completed with {len(result['conversation'])} messages exchanged")
    
    # Show key exchanges
    logger.info("\nKey exchanges:")
    for i, message in enumerate(result['conversation']):
        if message.sender != "user" and i > 0:  # Skip the initial query message
            logger.info(f"{message.sender} -> {message.receiver}: {message.content[:150]}...")
    
    # Run a maintenance operation to clean up old memories
    maintenance_stats = await coordinator.maintenance()
    logger.info(f"\nMaintenance completed: {maintenance_stats}")


async def temporal_consensus_example():
    """Example demonstrating temporal consensus."""
    logger.info("\n=== Temporal Consensus Example ===")
    
    # Create a temporal coordinator and consensus system
    coordinator = TemporalCoordinator(name="ConsensusCoordinator")
    consensus = TemporalConsensus(
        name="ProjectConsensus",
        temporal_coordinator=coordinator,
        confidence_threshold=0.6,
        opinion_decay_rate=0.2,  # Higher decay rate for testing
        historical_weight=0.3
    )
    
    # Create agents with different expertise
    research_agent = TimeAwareAgent("ResearchAgent", "research")
    engineering_agent = TimeAwareAgent("EngineeringAgent", "engineering")
    design_agent = TimeAwareAgent("DesignAgent", "design")
    product_agent = TimeAwareAgent("ProductAgent", "product")
    
    # Create a list of all agents
    all_agents = [research_agent, engineering_agent, design_agent, product_agent]
    
    # Run the consensus process
    first_decision = await consensus.reach_consensus(
        query="Which approach should we choose for the new feature: Option A or Option B?",
        agents=all_agents,
        time_window=None  # No historical context for first decision
    )
    
    logger.info(f"First consensus: {first_decision['consensus_value']} with " +
                f"confidence {first_decision['consensus_confidence']:.2f}")
    logger.info(f"Supporting agents: {first_decision['supporting_agents']}")
    
    # Simulate passage of time and new information
    logger.info("\nSimulating passage of time (3 hours)...")
    
    # Update the context with new information for a second decision
    context = {
        "additional_information": "New market research shows users prefer simplicity over features."
    }
    
    # Run a second consensus process with temporal context
    second_decision = await consensus.reach_consensus(
        query="With the new market research, should we reconsider our approach: Option A or Option B?",
        agents=all_agents,
        context=context,
        time_window=24  # Include the previous decision as context
    )
    
    logger.info(f"Second consensus: {second_decision['consensus_value']} with " +
                f"confidence {second_decision['consensus_confidence']:.2f}")
    logger.info(f"Supporting agents: {second_decision['supporting_agents']}")
    
    # Show how opinions changed
    logger.info("\nOpinion changes:")
    for agent_name in second_decision['agent_opinions']:
        first_opinion = first_decision['agent_opinions'].get(agent_name, {}).get('value', 'N/A')
        second_opinion = second_decision['agent_opinions'].get(agent_name, {}).get('value', 'N/A')
        
        if first_opinion != second_opinion:
            logger.info(f"{agent_name}: Changed from '{first_opinion}' to '{second_opinion}'")
        else:
            logger.info(f"{agent_name}: Maintained '{first_opinion}'")


async def main():
    """Run all examples."""
    await temporal_coordination_example()
    await temporal_consensus_example()


if __name__ == "__main__":
    asyncio.run(main())