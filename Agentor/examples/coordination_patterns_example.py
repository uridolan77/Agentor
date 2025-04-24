"""
Example demonstrating coordination patterns for multi-agent systems.

This example shows how to use different coordination patterns to solve problems
with multiple agents.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.components.coordination import (
    MasterSlavePattern,
    PeerToPeerPattern,
    BlackboardPattern,
    ContractNetProtocol,
    MarketBasedCoordination
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpecializedAgent(EnhancedAgent):
    """Agent with a specific specialty."""
    
    def __init__(self, name: str, specialty: str):
        """Initialize the specialized agent.
        
        Args:
            name: The name of the agent
            specialty: The agent's specialty
        """
        super().__init__(name=name)
        self.specialty = specialty
    
    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Run the agent.
        
        Args:
            input_data: The input data
            
        Returns:
            The agent's output
        """
        query = input_data.query
        context = input_data.context or {}
        
        # Simulate processing based on specialty
        response = f"As a {self.specialty} specialist, I processed: {query}"
        
        # Add some specialty-specific processing
        if self.specialty == "research":
            response += "\n\nBased on my research, I found the following information..."
        elif self.specialty == "analysis":
            response += "\n\nAfter analyzing the data, I concluded that..."
        elif self.specialty == "implementation":
            response += "\n\nI implemented the solution as follows..."
        elif self.specialty == "testing":
            response += "\n\nI tested the solution and found the following results..."
        elif self.specialty == "documentation":
            response += "\n\nI documented the solution as follows..."
        
        return AgentOutput(response=response, context=context)


async def master_slave_example():
    """Example demonstrating the Master-Slave pattern."""
    logger.info("\n=== Master-Slave Pattern Example ===")
    
    # Create agents
    master_agent = SpecializedAgent("MasterAgent", "coordination")
    research_agent = SpecializedAgent("ResearchAgent", "research")
    analysis_agent = SpecializedAgent("AnalysisAgent", "analysis")
    implementation_agent = SpecializedAgent("ImplementationAgent", "implementation")
    
    # Create the pattern
    pattern = MasterSlavePattern(master_agent=master_agent)
    pattern.add_slave_agent(research_agent)
    pattern.add_slave_agent(analysis_agent)
    pattern.add_slave_agent(implementation_agent)
    
    # Coordinate the agents
    query = "Develop a new algorithm for image recognition"
    result = await pattern.coordinate(query)
    
    logger.info(f"Result: {result}")


async def peer_to_peer_example():
    """Example demonstrating the Peer-to-Peer pattern."""
    logger.info("\n=== Peer-to-Peer Pattern Example ===")
    
    # Create agents
    research_agent = SpecializedAgent("ResearchAgent", "research")
    analysis_agent = SpecializedAgent("AnalysisAgent", "analysis")
    implementation_agent = SpecializedAgent("ImplementationAgent", "implementation")
    testing_agent = SpecializedAgent("TestingAgent", "testing")
    
    # Create the pattern
    pattern = PeerToPeerPattern()
    pattern.add_agent(research_agent)
    pattern.add_agent(analysis_agent)
    pattern.add_agent(implementation_agent)
    pattern.add_agent(testing_agent)
    
    # Connect agents
    pattern.connect_agents("ResearchAgent", "AnalysisAgent")
    pattern.connect_agents("AnalysisAgent", "ImplementationAgent")
    pattern.connect_agents("ImplementationAgent", "TestingAgent")
    pattern.connect_agents("TestingAgent", "ResearchAgent")
    
    # Coordinate the agents
    query = "Develop a new algorithm for image recognition"
    result = await pattern.coordinate(query)
    
    logger.info(f"Messages exchanged: {len(result)}")
    for message in result:
        logger.info(f"From {message.sender} to {message.receiver}: {message.content[:50]}...")


async def blackboard_example():
    """Example demonstrating the Blackboard pattern."""
    logger.info("\n=== Blackboard Pattern Example ===")
    
    # Create agents
    controller_agent = SpecializedAgent("ControllerAgent", "coordination")
    research_agent = SpecializedAgent("ResearchAgent", "research")
    analysis_agent = SpecializedAgent("AnalysisAgent", "analysis")
    implementation_agent = SpecializedAgent("ImplementationAgent", "implementation")
    testing_agent = SpecializedAgent("TestingAgent", "testing")
    
    # Create the pattern
    pattern = BlackboardPattern()
    pattern.set_controller(controller_agent)
    pattern.add_agent(research_agent)
    pattern.add_agent(analysis_agent)
    pattern.add_agent(implementation_agent)
    pattern.add_agent(testing_agent)
    
    # Coordinate the agents
    query = "Develop a new algorithm for image recognition"
    result = await pattern.coordinate(query)
    
    logger.info(f"Final blackboard state:")
    for key, value in result.items():
        logger.info(f"{key}: {value}")


async def contract_net_example():
    """Example demonstrating the Contract Net Protocol."""
    logger.info("\n=== Contract Net Protocol Example ===")
    
    # Create agents
    manager_agent = SpecializedAgent("ManagerAgent", "coordination")
    research_agent = SpecializedAgent("ResearchAgent", "research")
    analysis_agent = SpecializedAgent("AnalysisAgent", "analysis")
    implementation_agent = SpecializedAgent("ImplementationAgent", "implementation")
    
    # Create the pattern
    pattern = ContractNetProtocol(manager_agent=manager_agent)
    pattern.add_contractor_agent(research_agent)
    pattern.add_contractor_agent(analysis_agent)
    pattern.add_contractor_agent(implementation_agent)
    
    # Coordinate the agents
    query = "Develop a new algorithm for image recognition"
    result = await pattern.coordinate(query)
    
    logger.info(f"Result: {result}")


async def market_based_example():
    """Example demonstrating Market-Based Coordination."""
    logger.info("\n=== Market-Based Coordination Example ===")
    
    # Create agents
    market_agent = SpecializedAgent("MarketAgent", "coordination")
    research_agent = SpecializedAgent("ResearchAgent", "research")
    analysis_agent = SpecializedAgent("AnalysisAgent", "analysis")
    implementation_agent = SpecializedAgent("ImplementationAgent", "implementation")
    testing_agent = SpecializedAgent("TestingAgent", "testing")
    documentation_agent = SpecializedAgent("DocumentationAgent", "documentation")
    
    # Create the pattern
    pattern = MarketBasedCoordination()
    pattern.set_market_agent(market_agent)
    pattern.add_agent(research_agent)
    pattern.add_agent(analysis_agent)
    pattern.add_agent(implementation_agent)
    pattern.add_agent(testing_agent)
    pattern.add_agent(documentation_agent)
    
    # Register services
    pattern.register_service("ResearchAgent", "literature_review", 10.0)
    pattern.register_service("ResearchAgent", "data_collection", 15.0)
    pattern.register_service("AnalysisAgent", "data_analysis", 20.0)
    pattern.register_service("AnalysisAgent", "algorithm_design", 25.0)
    pattern.register_service("ImplementationAgent", "code_implementation", 30.0)
    pattern.register_service("TestingAgent", "unit_testing", 15.0)
    pattern.register_service("TestingAgent", "integration_testing", 20.0)
    pattern.register_service("DocumentationAgent", "user_documentation", 10.0)
    pattern.register_service("DocumentationAgent", "technical_documentation", 15.0)
    
    # Coordinate the agents
    query = "Develop a new algorithm for image recognition"
    result = await pattern.coordinate(query)
    
    logger.info(f"Result: {result}")


async def main():
    """Run all examples."""
    await master_slave_example()
    await peer_to_peer_example()
    await blackboard_example()
    await contract_net_example()
    await market_based_example()


if __name__ == "__main__":
    asyncio.run(main())
