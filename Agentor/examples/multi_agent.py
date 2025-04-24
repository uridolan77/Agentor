"""
Multi-Agent Example for the Agentor Framework

This example demonstrates the multi-agent capabilities of the Agentor framework,
including agent coordination, specialization, and consensus mechanisms.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from agentor.agents import Agent, AgentInput, AgentOutput
from agentor.components.coordination.patterns import MasterSlaveCoordinator, PeerToPeerCoordinator
from agentor.components.coordination.consensus import VotingConsensus, WeightedVotingConsensus
from agentor.components.coordination.roles import RoleBasedCoordinator
from agentor.components.coordination.specialization import SpecializedAgentFactory
from agentor.utils.logging import configure_logging, get_logger

# Configure logging
configure_logging(level=logging.INFO, json_format=False)
logger = get_logger(__name__)


class SpecialistAgent(Agent):
    """An agent that specializes in a particular domain."""

    def __init__(self, name: str, specialty: str, confidence: float = 0.8):
        super().__init__(name=name)
        self.specialty = specialty
        self.confidence = confidence
        logger.info(f"Created specialist agent {name} with specialty {specialty}")

    async def process(self, input_data: AgentInput) -> AgentOutput:
        """Process the input data based on the agent's specialty."""
        query = input_data.query.lower()
        context = input_data.context or {}

        # Check if the query is related to the agent's specialty
        if self.specialty.lower() in query:
            logger.info(f"Agent {self.name} is handling query related to {self.specialty}")
            response = f"As a {self.specialty} specialist, I can tell you that {query} is related to my expertise."
            confidence = self.confidence
        else:
            logger.info(f"Agent {self.name} has low confidence for query not related to {self.specialty}")
            response = f"I'm a {self.specialty} specialist, but this query seems outside my expertise."
            confidence = 0.3

        return AgentOutput(
            response=response,
            source_agent=self.name,
            confidence=confidence,
            metadata={"specialty": self.specialty}
        )


async def run_master_slave_example():
    """Run an example of master-slave coordination."""
    logger.info("Running Master-Slave Coordination Example")

    # Create specialist agents
    weather_agent = SpecialistAgent("WeatherAgent", "weather")
    news_agent = SpecialistAgent("NewsAgent", "news")
    sports_agent = SpecialistAgent("SportsAgent", "sports")

    # Create a master-slave coordinator
    coordinator = MasterSlaveCoordinator(
        name="MasterCoordinator",
        slave_agents=[weather_agent, news_agent, sports_agent]
    )

    # Process queries
    queries = [
        "What's the weather like today?",
        "Tell me the latest news headlines",
        "Who won the basketball game last night?",
        "What's the meaning of life?"  # This should be handled by the master agent
    ]

    for query in queries:
        logger.info(f"\nProcessing query: {query}")
        result = await coordinator.process(AgentInput(query=query))
        logger.info(f"Response: {result.response}")
        logger.info(f"Source agent: {result.source_agent}")
        logger.info(f"Confidence: {result.confidence}")


async def run_peer_to_peer_example():
    """Run an example of peer-to-peer coordination."""
    logger.info("\nRunning Peer-to-Peer Coordination Example")

    # Create specialist agents
    weather_agent = SpecialistAgent("WeatherAgent", "weather")
    news_agent = SpecialistAgent("NewsAgent", "news")
    sports_agent = SpecialistAgent("SportsAgent", "sports")

    # Create a peer-to-peer coordinator
    coordinator = PeerToPeerCoordinator(
        name="P2PCoordinator",
        agents=[weather_agent, news_agent, sports_agent],
        confidence_threshold=0.6  # Only use responses with confidence >= 0.6
    )

    # Process queries
    queries = [
        "What's the weather like today?",
        "Tell me the latest news headlines",
        "Who won the basketball game last night?",
        "What's the meaning of life?"  # This should get responses from all agents
    ]

    for query in queries:
        logger.info(f"\nProcessing query: {query}")
        result = await coordinator.process(AgentInput(query=query))
        logger.info(f"Response: {result.response}")
        logger.info(f"Source agents: {result.metadata.get('source_agents', [])}")
        logger.info(f"Confidence: {result.confidence}")


async def run_voting_consensus_example():
    """Run an example of voting consensus."""
    logger.info("\nRunning Voting Consensus Example")

    # Create specialist agents
    agents = [
        SpecialistAgent("Agent1", "general knowledge", confidence=0.7),
        SpecialistAgent("Agent2", "general knowledge", confidence=0.8),
        SpecialistAgent("Agent3", "general knowledge", confidence=0.6),
        SpecialistAgent("Agent4", "general knowledge", confidence=0.9),
        SpecialistAgent("Agent5", "general knowledge", confidence=0.5),
    ]

    # Create a voting consensus mechanism
    consensus = VotingConsensus(agents=agents)

    # Process a query that requires consensus
    query = "Is climate change real?"
    logger.info(f"\nProcessing query with voting consensus: {query}")

    # Simulate different responses from agents
    responses = [
        AgentOutput(response="Yes, climate change is real and supported by scientific evidence.", confidence=0.9),
        AgentOutput(response="Yes, climate change is happening.", confidence=0.8),
        AgentOutput(response="The evidence suggests climate change is occurring.", confidence=0.7),
        AgentOutput(response="There is ongoing debate about climate change.", confidence=0.6),
        AgentOutput(response="No, climate change is not real.", confidence=0.5),
    ]

    # Get consensus
    result = await consensus.get_consensus(AgentInput(query=query), responses)
    logger.info(f"Consensus response: {result.response}")
    logger.info(f"Consensus confidence: {result.confidence}")
    logger.info(f"Voting results: {result.metadata.get('voting_results', {})}")


async def run_weighted_voting_example():
    """Run an example of weighted voting consensus."""
    logger.info("\nRunning Weighted Voting Consensus Example")

    # Create specialist agents with different weights
    agents = [
        SpecialistAgent("Expert1", "climate science", confidence=0.9),
        SpecialistAgent("Expert2", "climate science", confidence=0.85),
        SpecialistAgent("Generalist1", "general knowledge", confidence=0.7),
        SpecialistAgent("Generalist2", "general knowledge", confidence=0.6),
        SpecialistAgent("Novice", "beginner", confidence=0.5),
    ]

    # Assign weights to agents (experts have higher weights)
    weights = {
        "Expert1": 5.0,
        "Expert2": 4.0,
        "Generalist1": 2.0,
        "Generalist2": 1.5,
        "Novice": 1.0
    }

    # Create a weighted voting consensus mechanism
    consensus = WeightedVotingConsensus(agents=agents, weights=weights)

    # Process a query that requires consensus
    query = "What are the main causes of climate change?"
    logger.info(f"\nProcessing query with weighted voting consensus: {query}")

    # Simulate different responses from agents
    responses = [
        AgentOutput(response="Human activities like burning fossil fuels are the main cause.",
                   confidence=0.9, source_agent="Expert1"),
        AgentOutput(response="Greenhouse gas emissions from human activities are the primary driver.",
                   confidence=0.85, source_agent="Expert2"),
        AgentOutput(response="Both human activities and natural factors contribute to climate change.",
                   confidence=0.7, source_agent="Generalist1"),
        AgentOutput(response="There are various factors including human and natural causes.",
                   confidence=0.6, source_agent="Generalist2"),
        AgentOutput(response="Natural climate cycles are the main cause.",
                   confidence=0.5, source_agent="Novice"),
    ]

    # Get consensus
    result = await consensus.get_consensus(AgentInput(query=query), responses)
    logger.info(f"Weighted consensus response: {result.response}")
    logger.info(f"Weighted consensus confidence: {result.confidence}")
    logger.info(f"Voting results: {result.metadata.get('voting_results', {})}")
    logger.info(f"Weights used: {result.metadata.get('weights', {})}")


async def run_role_based_example():
    """Run an example of role-based coordination."""
    logger.info("\nRunning Role-Based Coordination Example")

    # Define roles
    roles = {
        "researcher": "Gathers information and facts about the query",
        "analyst": "Analyzes the information and provides insights",
        "writer": "Formulates the final response in a clear and concise manner"
    }

    # Create a role-based coordinator
    coordinator = RoleBasedCoordinator(
        name="RoleCoordinator",
        roles=roles
    )

    # Process a query that requires multiple roles
    query = "Explain the impact of artificial intelligence on job markets"
    logger.info(f"\nProcessing query with role-based coordination: {query}")

    result = await coordinator.process(AgentInput(query=query))
    logger.info(f"Final response: {result.response}")
    logger.info(f"Role contributions: {result.metadata.get('role_contributions', {})}")


async def run_specialized_agent_factory_example():
    """Run an example of specialized agent factory."""
    logger.info("\nRunning Specialized Agent Factory Example")

    # Create a specialized agent factory
    factory = SpecializedAgentFactory()

    # Register specializations
    factory.register_specialization(
        "weather",
        lambda: SpecialistAgent("WeatherSpecialist", "weather", confidence=0.9)
    )
    factory.register_specialization(
        "news",
        lambda: SpecialistAgent("NewsSpecialist", "news", confidence=0.85)
    )
    factory.register_specialization(
        "sports",
        lambda: SpecialistAgent("SportsSpecialist", "sports", confidence=0.8)
    )

    # Create specialized agents based on queries
    queries = [
        "What's the weather forecast for tomorrow?",
        "Tell me the latest news about technology",
        "Who won the soccer match yesterday?",
        "What's the capital of France?"  # No specific specialization
    ]

    for query in queries:
        logger.info(f"\nCreating specialized agent for query: {query}")
        agent = factory.create_agent_for_query(query)
        logger.info(f"Created agent: {agent.name if agent else 'None'} with specialty: {agent.specialty if agent else 'None'}")

        if agent:
            result = await agent.process(AgentInput(query=query))
            logger.info(f"Response: {result.response}")
            logger.info(f"Confidence: {result.confidence}")


async def main():
    """Run all examples."""
    await run_master_slave_example()
    await run_peer_to_peer_example()
    await run_voting_consensus_example()
    await run_weighted_voting_example()
    await run_role_based_example()
    await run_specialized_agent_factory_example()


if __name__ == "__main__":
    asyncio.run(main())