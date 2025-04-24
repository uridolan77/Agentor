"""
Example demonstrating consensus algorithms for multi-agent decision making.

This example shows how different consensus algorithms can be used to help
agents reach collective decisions using various voting methods and strategies.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.agents import Agent, AgentInput, AgentOutput
from agentor.components.coordination.consensus import (
    ConsensusOption,
    PluralityVoting,
    BordaCountVoting,
    InstantRunoffVoting,
    ConsensusBuilder,
    IterativeConsensus,
    DelphinConsensus,
    MajorityConcensus,
    UnanimityConsensus
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpecializedAgent(EnhancedAgent):
    """Agent with a specialty that influences its decision making."""
    
    def __init__(self, name: str, specialty: str, bias: float = 0.0):
        """Initialize the specialized agent.
        
        Args:
            name: Agent name
            specialty: Agent's specialty area
            bias: Bias factor (-1.0 to 1.0) that influences decisions
        """
        super().__init__(name=name)
        self.specialty = specialty
        self.bias = max(-1.0, min(1.0, bias))  # Clamp to [-1, 1]
        
    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Process agent input and generate a response.
        
        Args:
            input_data: Agent input
            
        Returns:
            Agent output with response
        """
        query = input_data.query
        context = input_data.context or {}
        
        # Check if this is a consensus request
        consensus_type = context.get("consensus_type")
        
        if consensus_type == "preference_ranking":
            # This is a request to rank options for consensus
            response = self._generate_preference_ranking(query, context)
        elif context.get("request_type") == "explain_vote":
            # This is a request to explain reasoning
            response = self._generate_explanation(query, context)
        else:
            # Regular query
            response = f"As {self.name} specializing in {self.specialty}, I processed: {query}"
            
        return AgentOutput(response=response, context=context)
    
    def _generate_preference_ranking(self, query: str, context: Dict[str, Any]) -> str:
        """Generate preference rankings for consensus voting.
        
        Args:
            query: Question being asked
            context: Context including available options
            
        Returns:
            Ranked preferences as a string
        """
        # Get available options
        available_options = context.get("available_options", [])
        if not available_options:
            return "No options provided to rank."
        
        # Create a list to hold our scored options
        scored_options = []
        
        # Score each option based on specialty and bias
        for i, option in enumerate(available_options):
            description = option.get("description", "")
            value = option.get("value", "")
            option_text = f"{description} ({value})" if value != description else description
            
            # Base score - randomized but deterministic based on option
            base_score = hash(option_text) % 100 / 100.0
            
            # Specialty influence - prefer options that match specialty
            specialty_match = 0
            if self.specialty.lower() in description.lower() or self.specialty.lower() in str(value).lower():
                specialty_match = 0.4
                
            # Apply bias
            biased_score = base_score + specialty_match + (self.bias * 0.2)
            
            scored_options.append((i + 1, biased_score))
        
        # Sort by score (highest first)
        scored_options.sort(key=lambda x: x[1], reverse=True)
        
        # Get the rankings
        rankings = [str(idx) for idx, _ in scored_options]
        
        # Format as comma-separated list
        return ", ".join(rankings)
    
    def _generate_explanation(self, query: str, context: Dict[str, Any]) -> str:
        """Generate an explanation for a voting choice.
        
        Args:
            query: Question asking for explanation
            context: Context
            
        Returns:
            Explanation text
        """
        # Extract the selected option from the query
        selected_option = ""
        for line in query.split("\n"):
            if "selected" in line and "'" in line:
                # Extract text between single quotes
                parts = line.split("'")
                if len(parts) >= 3:
                    selected_option = parts[1]
                break
        
        # Generate explanation based on specialty
        explanation = f"As an expert in {self.specialty}, I selected this option because "
        
        if selected_option:
            if self.specialty.lower() in selected_option.lower():
                explanation += f"it directly aligns with my expertise in {self.specialty}. "
                explanation += "This approach leverages my specialized knowledge and experience."
            else:
                explanation += f"even though it's not directly related to my {self.specialty} specialty, "
                explanation += "I believe it offers the best balance of effectiveness and practicality. "
                explanation += f"From a {self.specialty} perspective, this approach still provides key advantages."
        else:
            explanation += "it represents the optimal choice based on my analysis. "
            explanation += f"My background in {self.specialty} gives me insight into why this is the most effective approach."
            
        # Add some variation based on bias
        if self.bias > 0.5:
            explanation += "\n\nI strongly believe this is the best option, with very few drawbacks."
        elif self.bias < -0.5:
            explanation += "\n\nWhile I've selected this option, I acknowledge it has some limitations we should address."
        
        return explanation


async def demonstrate_plurality_voting():
    """Demonstrate simple plurality voting."""
    logger.info("\n=== Plurality Voting Example ===")
    
    # Create agents with different specialties
    agents = [
        SpecializedAgent("TechnicalAgent", "engineering", bias=0.2),
        SpecializedAgent("ProductAgent", "product management", bias=-0.1),
        SpecializedAgent("DesignAgent", "user experience", bias=0.3),
        SpecializedAgent("MarketingAgent", "marketing", bias=-0.2),
        SpecializedAgent("FinanceAgent", "financial analysis", bias=0.0)
    ]
    
    # Create options
    options = [
        ConsensusOption(
            value="option_a",
            description="Option A: Develop new feature set with technical focus",
            metadata={"category": "technical", "development_time": "high"}
        ),
        ConsensusOption(
            value="option_b",
            description="Option B: Enhance user experience with design improvements",
            metadata={"category": "design", "development_time": "medium"}
        ),
        ConsensusOption(
            value="option_c",
            description="Option C: Optimize existing features for better performance",
            metadata={"category": "optimization", "development_time": "low"}
        ),
        ConsensusOption(
            value="option_d",
            description="Option D: Focus on marketing and outreach for current features",
            metadata={"category": "marketing", "development_time": "low"}
        )
    ]
    
    # Create consensus builder with plurality voting
    consensus_builder = ConsensusBuilder(
        name="ProductPriorityConsensus",
        voting_method=PluralityVoting()
    )
    
    # Build consensus
    logger.info("Building consensus with plurality voting...")
    result = await consensus_builder.build_consensus(
        question="What should be our product team's priority for the next quarter?",
        options=options,
        agents=agents
    )
    
    # Show results
    logger.info(f"Consensus reached: {result['success']}")
    if result['success']:
        logger.info(f"Winning option: {result['winning_option']['description']}")
        logger.info(f"Support: {result['winner_percentage']:.1f}%")
    
    # Show how each agent voted
    logger.info("\nAgent votes:")
    for agent_name, agent_votes in result['votes'].items():
        if agent_votes:
            top_choice = agent_votes[0]['description']
            logger.info(f"- {agent_name}: {top_choice}")


async def demonstrate_ranked_choice_voting():
    """Demonstrate ranked choice (instant runoff) voting."""
    logger.info("\n=== Ranked Choice Voting Example ===")
    
    # Create agents with different specialties
    agents = [
        SpecializedAgent("FrontendDev", "frontend development", bias=0.3),
        SpecializedAgent("BackendDev", "backend development", bias=-0.2),
        SpecializedAgent("DataScientist", "data science", bias=0.1),
        SpecializedAgent("DevOpsEngineer", "infrastructure", bias=-0.3),
        SpecializedAgent("SecurityExpert", "security", bias=0.0),
        SpecializedAgent("ProjectManager", "project management", bias=0.2)
    ]
    
    # Create options
    options = [
        ConsensusOption(
            value="arch_a",
            description="Architecture A: Microservices with frontend focus",
            metadata={"complexity": "high", "scalability": "high"}
        ),
        ConsensusOption(
            value="arch_b",
            description="Architecture B: Monolith with strong backend",
            metadata={"complexity": "medium", "scalability": "medium"}
        ),
        ConsensusOption(
            value="arch_c",
            description="Architecture C: Serverless with data processing focus",
            metadata={"complexity": "medium", "scalability": "high"}
        ),
        ConsensusOption(
            value="arch_d",
            description="Architecture D: Hybrid approach with security emphasis",
            metadata={"complexity": "high", "scalability": "medium"}
        )
    ]
    
    # Create consensus builder with instant runoff voting
    consensus_builder = ConsensusBuilder(
        name="ArchitectureConsensus",
        voting_method=InstantRunoffVoting()
    )
    
    # Build consensus
    logger.info("Building consensus with instant runoff voting...")
    result = await consensus_builder.build_consensus(
        question="Which architecture should we adopt for our new system?",
        options=options,
        agents=agents
    )
    
    # Show results
    logger.info(f"Consensus reached: {result['success']}")
    if result['success']:
        logger.info(f"Winning option: {result['winning_option']['description']}")
        logger.info(f"Support: {result['winner_percentage']:.1f}%")
    
    # Show rounds if available
    rounds = result.get('rounds', [])
    if rounds:
        logger.info(f"\nRanked choice voting completed in {len(rounds)} rounds")
        for i, round_data in enumerate(rounds):
            eliminated = round_data.get('eliminated', [])
            eliminated_desc = ", ".join([str(e) for e in eliminated]) if eliminated else "none"
            logger.info(f"Round {i+1}: Eliminated options: {eliminated_desc}")


async def demonstrate_iterative_consensus():
    """Demonstrate iterative consensus building."""
    logger.info("\n=== Iterative Consensus Example ===")
    
    # Create agents with different specialties
    agents = [
        SpecializedAgent("ProductOwner", "product strategy", bias=0.4),
        SpecializedAgent("TechLead", "technical leadership", bias=-0.1),
        SpecializedAgent("UXDesigner", "user experience", bias=0.2),
        SpecializedAgent("MarketAnalyst", "market analysis", bias=-0.3),
        SpecializedAgent("CustomerSupport", "customer support", bias=0.1)
    ]
    
    # Create options
    options = [
        ConsensusOption(
            value="roadmap_a",
            description="Roadmap A: Focus on new features for power users",
            metadata={"market_size": "medium", "effort": "high"}
        ),
        ConsensusOption(
            value="roadmap_b",
            description="Roadmap B: Improve onboarding and UX for new users",
            metadata={"market_size": "large", "effort": "medium"}
        ),
        ConsensusOption(
            value="roadmap_c",
            description="Roadmap C: Add enterprise features and security",
            metadata={"market_size": "small", "effort": "high"}
        ),
        ConsensusOption(
            value="roadmap_d",
            description="Roadmap D: Expand platform integrations and API",
            metadata={"market_size": "medium", "effort": "medium"}
        ),
        ConsensusOption(
            value="roadmap_e",
            description="Roadmap E: Focus on performance and reliability",
            metadata={"market_size": "all", "effort": "medium"}
        )
    ]
    
    # Create iterative consensus builder
    consensus_builder = IterativeConsensus(
        name="RoadmapConsensus",
        voting_method=BordaCountVoting()
    )
    
    # Build consensus
    logger.info("Building consensus iteratively with Borda count voting...")
    result = await consensus_builder.build_consensus(
        question="Which product roadmap should we pursue for the next year?",
        options=options,
        agents=agents,
        max_rounds=3,
        agreement_threshold=70.0  # 70% agreement required
    )
    
    # Show results
    logger.info(f"Consensus reached: {result['consensus_reached']}")
    if result['consensus_reached']:
        logger.info(f"Winning option: {result['winning_option']['description']}")
        logger.info(f"Agreement: {result['agreement_percentage']:.1f}%")
    
    # Show rounds
    logger.info(f"\nIterative consensus completed in {result['final_round']} rounds")
    for i, round_data in enumerate(result['rounds']):
        round_num = round_data['round']
        winner = round_data.get('winning_option', {})
        winner_desc = winner.get('description', 'No winner')
        winner_pct = round_data.get('winner_percentage', 0)
        logger.info(f"Round {round_num}: Leader: {winner_desc} with {winner_pct:.1f}% support")


async def demonstrate_delphi_method():
    """Demonstrate Delphi method consensus."""
    logger.info("\n=== Delphi Method Consensus Example ===")
    
    # Create agents with different specialties
    agents = [
        SpecializedAgent("StrategyConsultant", "strategic planning", bias=0.1),
        SpecializedAgent("TechnologyAdvisor", "technology trends", bias=-0.2),
        SpecializedAgent("MarketResearcher", "market research", bias=0.3),
        SpecializedAgent("CompetitiveAnalyst", "competitive analysis", bias=-0.1),
        SpecializedAgent("IndustryExpert", "industry knowledge", bias=0.0),
        SpecializedAgent("FuturistThinker", "future trends", bias=0.4)
    ]
    
    # Create options
    options = [
        ConsensusOption(
            value="strategy_a",
            description="Strategy A: Focus on core market and deepen penetration",
            metadata={"risk": "low", "reward": "medium"}
        ),
        ConsensusOption(
            value="strategy_b",
            description="Strategy B: Expand to adjacent markets with existing technology",
            metadata={"risk": "medium", "reward": "medium"}
        ),
        ConsensusOption(
            value="strategy_c",
            description="Strategy C: Invest in disruptive innovation for new markets",
            metadata={"risk": "high", "reward": "high"}
        ),
        ConsensusOption(
            value="strategy_d",
            description="Strategy D: Acquire competitors to consolidate market position",
            metadata={"risk": "medium", "reward": "high"}
        )
    ]
    
    # Create Delphi consensus builder
    consensus_builder = DelphinConsensus(
        name="StrategyConsensus",
        voting_method=PluralityVoting()
    )
    
    # Build consensus
    logger.info("Building consensus using Delphi method with anonymous feedback...")
    result = await consensus_builder.build_consensus(
        question="What strategic direction should the company take over the next 5 years?",
        options=options,
        agents=agents,
        max_rounds=3,
        agreement_threshold=65.0  # 65% agreement required
    )
    
    # Show results
    logger.info(f"Consensus reached: {result['consensus_reached']}")
    if result['consensus_reached']:
        logger.info(f"Winning option: {result['winning_option']['description']}")
        logger.info(f"Agreement: {result['agreement_percentage']:.1f}%")
    
    # Show how consensus evolved
    logger.info("\nConsensus evolution:")
    for i, round_data in enumerate(result['rounds']):
        round_num = round_data['round']
        winner = round_data.get('winning_option', {})
        winner_desc = winner.get('description', 'No consensus')
        winner_pct = round_data.get('winner_percentage', 0)
        
        # Show abbreviated reasoning from this round
        reasoning = round_data.get('reasoning', {})
        reasoning_count = len(reasoning)
        
        logger.info(f"Round {round_num}: {winner_desc} ({winner_pct:.1f}%)")
        logger.info(f"  {reasoning_count} experts provided feedback in this round")
        
        # Show a sample of reasoning if available
        if reasoning and round_num < len(result['rounds']):
            sample_agent = next(iter(reasoning.keys()))
            sample_reason = reasoning[sample_agent]
            if sample_reason and len(sample_reason) > 100:
                sample_reason = sample_reason[:100] + "..."
            logger.info(f"  Sample feedback: \"{sample_reason}\"")


async def demonstrate_unanimity_consensus():
    """Demonstrate unanimity consensus building."""
    logger.info("\n=== Unanimity Consensus Example ===")
    
    # Create agents with different specialties
    agents = [
        SpecializedAgent("CEOAgent", "executive leadership", bias=0.1),
        SpecializedAgent("CFOAgent", "financial management", bias=0.0),
        SpecializedAgent("CTOAgent", "technology strategy", bias=-0.1),
        SpecializedAgent("CMOAgent", "marketing strategy", bias=0.2)
    ]
    
    # Create options
    options = [
        ConsensusOption(
            value="merger_a",
            description="Merger Option A: Acquire smaller competitor in core market",
            metadata={"cost": "medium", "synergy": "high"}
        ),
        ConsensusOption(
            value="merger_b",
            description="Merger Option B: Merge with equal-sized company in adjacent market",
            metadata={"cost": "high", "synergy": "medium"}
        ),
        ConsensusOption(
            value="merger_c",
            description="Merger Option C: Be acquired by larger industry player",
            metadata={"cost": "low", "synergy": "medium"}
        )
    ]
    
    # Create unanimity consensus builder with near-unanimity (90%)
    consensus_builder = UnanimityConsensus(
        name="MergerConsensus",
        unanimity_threshold=0.9,  # 90% agreement required
        max_rounds=5
    )
    
    # Build consensus
    logger.info("Building consensus requiring near-unanimity (90%)...")
    result = await consensus_builder.build_consensus(
        question="Which strategic merger option should the company pursue?",
        options=options,
        agents=agents
    )
    
    # Show results
    logger.info(f"Unanimity consensus reached: {result['consensus_reached']}")
    if result['consensus_reached']:
        logger.info(f"Selected option: {result['winning_option']['description']}")
        logger.info(f"Agreement level: {result['actual_agreement']:.1f}%")
    else:
        logger.info(f"Failed to reach required {result['unanimity_threshold']}% agreement")
        if result.get('winning_option'):
            logger.info(f"Best option was: {result['winning_option']['description']}")
            logger.info(f"With only {result['actual_agreement']:.1f}% agreement")
    
    # Show voting rounds
    logger.info("\nVoting rounds:")
    for i, round_data in enumerate(result['rounds']):
        round_num = round_data['round']
        winner = round_data.get('winning_option', {})
        winner_desc = winner.get('description', 'No winner') if winner else 'No winner'
        winner_pct = round_data.get('winner_percentage', 0)
        logger.info(f"Round {round_num}: {winner_desc} with {winner_pct:.1f}% support")


async def main():
    """Run all consensus algorithm examples."""
    # Simple plurality voting
    await demonstrate_plurality_voting()
    
    # Ranked choice voting
    await demonstrate_ranked_choice_voting()
    
    # Iterative consensus with multiple rounds
    await demonstrate_iterative_consensus()
    
    # Delphi method with anonymous feedback
    await demonstrate_delphi_method()
    
    # Unanimity consensus for critical decisions
    await demonstrate_unanimity_consensus()


if __name__ == "__main__":
    asyncio.run(main())
