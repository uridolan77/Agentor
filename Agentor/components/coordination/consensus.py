"""
Consensus Algorithms for Multi-Agent Collective Decision Making.

This module provides various consensus algorithms that enable groups of agents
to reach collective decisions through different voting and agreement mechanisms.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from collections import Counter
from datetime import datetime

from agentor.agents import Agent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class ConsensusOption:
    """Represents an option in a consensus decision process."""
    
    def __init__(
        self,
        value: Any,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize an option.
        
        Args:
            value: The option value
            description: Human-readable description
            metadata: Additional metadata
        """
        self.value = value
        self.description = description or str(value)
        self.metadata = metadata or {}
        
    def __eq__(self, other):
        if isinstance(other, ConsensusOption):
            return self.value == other.value
        return False
        
    def __hash__(self):
        return hash(self.value)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "description": self.description,
            "metadata": self.metadata
        }


class VotingMethod:
    """Base class for voting methods used in consensus algorithms."""
    
    async def tally_votes(
        self,
        votes: Dict[str, List[Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Tally votes and determine the winner.
        
        Args:
            votes: Dictionary mapping agent IDs to their votes
            weights: Optional dictionary mapping agent IDs to their weights
            
        Returns:
            Tuple of (winning option, additional result data)
        """
        raise NotImplementedError("Subclasses must implement tally_votes")


class PluralityVoting(VotingMethod):
    """Simple plurality voting (first past the post)."""
    
    async def tally_votes(
        self,
        votes: Dict[str, List[Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Tally votes using plurality (most votes wins).
        
        Args:
            votes: Dictionary mapping agent IDs to their votes (first item used)
            weights: Optional dictionary mapping agent IDs to their weights
            
        Returns:
            Tuple of (winning option, additional result data)
        """
        # Extract first choice votes only
        first_choices = {agent_id: choices[0] if choices else None 
                        for agent_id, choices in votes.items()}
        
        # Count votes with weights
        if weights is None:
            weights = {agent_id: 1.0 for agent_id in votes}
            
        weighted_votes = {}
        for agent_id, choice in first_choices.items():
            if choice is not None:
                if choice in weighted_votes:
                    weighted_votes[choice] += weights.get(agent_id, 1.0)
                else:
                    weighted_votes[choice] = weights.get(agent_id, 1.0)
        
        if not weighted_votes:
            return None, {"error": "No valid votes cast"}
        
        # Find winner
        winner = max(weighted_votes.items(), key=lambda x: x[1])
        winning_option, winning_score = winner
        
        # Calculate total weight
        total_weight = sum(weights.values())
        winning_percentage = (winning_score / total_weight) * 100 if total_weight > 0 else 0
        
        return winning_option, {
            "vote_counts": weighted_votes,
            "winner_score": winning_score,
            "winner_percentage": winning_percentage,
            "total_votes": len([v for v in first_choices.values() if v is not None]),
            "total_weight": total_weight
        }


class BordaCountVoting(VotingMethod):
    """Borda count rank-based voting method."""
    
    async def tally_votes(
        self,
        votes: Dict[str, List[Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Tally votes using Borda count (rank-based scoring).
        
        Args:
            votes: Dictionary mapping agent IDs to their ranked choices
            weights: Optional dictionary mapping agent IDs to their weights
            
        Returns:
            Tuple of (winning option, additional result data)
        """
        if weights is None:
            weights = {agent_id: 1.0 for agent_id in votes}
            
        # Determine all unique options
        all_options = set()
        for choices in votes.values():
            all_options.update(choices)
            
        max_score = len(all_options)
        scores = {option: 0 for option in all_options}
        
        # Calculate Borda scores
        for agent_id, choices in votes.items():
            agent_weight = weights.get(agent_id, 1.0)
            for i, option in enumerate(choices):
                # Score is (max_score - rank)
                # Higher ranks (earlier in list) get higher scores
                rank = i + 1
                score = (max_score - rank + 1) * agent_weight
                scores[option] += score
                
        if not scores:
            return None, {"error": "No valid votes cast"}
        
        # Find winner
        winner = max(scores.items(), key=lambda x: x[1])
        winning_option, winning_score = winner
        
        # Calculate percentages
        total_score = sum(scores.values())
        percentages = {option: (score / total_score) * 100 if total_score > 0 else 0
                      for option, score in scores.items()}
        
        return winning_option, {
            "scores": scores,
            "percentages": percentages,
            "winner_score": winning_score,
            "winner_percentage": percentages[winning_option]
        }


class InstantRunoffVoting(VotingMethod):
    """Instant runoff voting (IRV) / Ranked choice voting."""
    
    async def tally_votes(
        self,
        votes: Dict[str, List[Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Tally votes using instant runoff voting.
        
        Args:
            votes: Dictionary mapping agent IDs to their ranked choices
            weights: Optional dictionary mapping agent IDs to their weights
            
        Returns:
            Tuple of (winning option, additional result data)
        """
        if weights is None:
            weights = {agent_id: 1.0 for agent_id in votes}
            
        # Make a copy of votes to manipulate
        current_votes = {agent_id: list(choices) for agent_id, choices in votes.items()}
        
        # Track eliminated options
        eliminated = set()
        
        # Track rounds of voting
        rounds = []
        
        while True:
            # Count first preferences
            first_prefs = {}
            for agent_id, choices in current_votes.items():
                # Skip to first non-eliminated choice
                choice = next((c for c in choices if c not in eliminated), None)
                if choice is not None:
                    weight = weights.get(agent_id, 1.0)
                    if choice in first_prefs:
                        first_prefs[choice] += weight
                    else:
                        first_prefs[choice] = weight
            
            if not first_prefs:
                # No valid votes left
                return None, {"error": "No valid votes remained", "rounds": rounds}
            
            # Calculate total weight of votes
            total_weight = sum(weights.values())
            
            # Record this round
            rounds.append({
                "vote_counts": first_prefs.copy(),
                "eliminated": list(eliminated)
            })
            
            # Check if any option has majority
            for option, score in first_prefs.items():
                if score > total_weight / 2:
                    # We have a winner with majority
                    return option, {
                        "rounds": rounds,
                        "winner_score": score,
                        "winner_percentage": (score / total_weight) * 100 if total_weight > 0 else 0
                    }
            
            # No majority, eliminate lowest scoring option
            min_option = min(first_prefs.items(), key=lambda x: x[1])
            eliminated.add(min_option[0])
            
            # If only one option remains, it wins
            if len(first_prefs) - len(eliminated) <= 1:
                remaining = [opt for opt in first_prefs.keys() if opt not in eliminated]
                if remaining:
                    winner = remaining[0]
                    return winner, {
                        "rounds": rounds,
                        "winner_score": first_prefs.get(winner, 0),
                        "winner_percentage": (first_prefs.get(winner, 0) / total_weight) * 100 if total_weight > 0 else 0,
                        "winner_by_elimination": True
                    }
                return None, {"error": "No options remained after elimination", "rounds": rounds}


class ConsensusBuilder:
    """Base class for consensus building among agents."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        voting_method: Optional[VotingMethod] = None
    ):
        """Initialize the consensus builder.
        
        Args:
            name: Name for this consensus builder
            voting_method: Method to use for vote tallying
        """
        self.name = name or f"ConsensusBuilder-{id(self)}"
        self.voting_method = voting_method or PluralityVoting()
        
    async def gather_votes(
        self,
        question: str,
        options: List[ConsensusOption],
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 1
    ) -> Dict[str, List[ConsensusOption]]:
        """Gather votes from agents.
        
        Args:
            question: The question to ask agents
            options: List of options to choose from
            agents: List of agents to query
            context: Additional context
            max_retries: Maximum number of retries for failed votes
            
        Returns:
            Dictionary mapping agent IDs to their votes (ranked)
        """
        votes = {}
        options_dict = {str(opt.value): opt for opt in options}
        
        # Prepare options information
        options_info = "\n".join([
            f"{i+1}. {opt.description}" for i, opt in enumerate(options)
        ])
        
        for agent in agents:
            # Prepare agent prompt
            prompt = (
                f"{question}\n\n"
                f"Options:\n{options_info}\n\n"
                "Please rank your preferences in order, starting with your most preferred option. "
                "Provide the option numbers in a comma-separated list (e.g., '3,1,2')."
            )
            
            # Add context information if provided
            agent_context = context.copy() if context else {}
            agent_context.update({
                "consensus_type": "preference_ranking",
                "available_options": [opt.to_dict() for opt in options]
            })
            
            # Try to get vote from agent
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    # Run the agent
                    result = await agent.run(AgentInput(query=prompt, context=agent_context))
                    
                    # Parse the agent's response
                    agent_votes = self._parse_agent_vote(result.response, options)
                    
                    if agent_votes:
                        votes[agent.name] = agent_votes
                        break
                    else:
                        logger.warning(f"Failed to parse vote from {agent.name}: {result.response}")
                except Exception as e:
                    logger.error(f"Error getting vote from {agent.name}: {e}")
                
                retry_count += 1
            
        return votes
    
    def _parse_agent_vote(self, response: str, options: List[ConsensusOption]) -> List[ConsensusOption]:
        """Parse an agent's vote from their response.
        
        Args:
            response: The agent's response text
            options: Available options
            
        Returns:
            List of ranked options
        """
        # Default implementation: look for comma-separated list of numbers
        # or option values in the response
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            
            # Check for comma-separated numbers
            if "," in line:
                try:
                    # Try to parse as option numbers (1-based)
                    numbers = [int(n.strip()) for n in line.split(",")]
                    # Convert to 0-based indices and filter valid indices
                    indices = [n-1 for n in numbers if 0 < n <= len(options)]
                    if indices:
                        return [options[i] for i in indices]
                except ValueError:
                    # Not numbers, try to match option values
                    values = [v.strip() for v in line.split(",")]
                    matches = []
                    for v in values:
                        for opt in options:
                            if str(opt.value) == v or opt.description == v:
                                matches.append(opt)
                                break
                    if matches:
                        return matches
        
        # If no clear format found, try to find mentions of options in order
        found_options = []
        for opt in options:
            if str(opt.value) in response or opt.description in response:
                found_options.append(opt)
        
        return found_options
    
    async def build_consensus(
        self,
        question: str,
        options: List[ConsensusOption],
        agents: List[Agent],
        agent_weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build consensus among agents.
        
        Args:
            question: The question to ask agents
            options: List of options to choose from
            agents: List of agents to query
            agent_weights: Optional weights for each agent
            context: Additional context
            
        Returns:
            Consensus result with voting data
        """
        # Gather votes from all agents
        votes = await self.gather_votes(
            question=question,
            options=options,
            agents=agents,
            context=context
        )
        
        # Apply voting method
        if not votes:
            return {
                "success": False,
                "error": "No valid votes collected",
                "winning_option": None,
                "votes": {}
            }
            
        winning_option, results = await self.voting_method.tally_votes(
            votes=votes,
            weights=agent_weights
        )
        
        # Prepare the consensus result
        consensus_result = {
            "success": winning_option is not None,
            "winning_option": winning_option.to_dict() if winning_option else None,
            "voting_method": self.voting_method.__class__.__name__,
            "votes": {
                agent: [opt.to_dict() for opt in options]
                for agent, options in votes.items()
            },
            **results
        }
        
        return consensus_result


class IterativeConsensus(ConsensusBuilder):
    """Consensus builder that uses multiple rounds to reach agreement."""
    
    async def build_consensus(
        self,
        question: str,
        options: List[ConsensusOption],
        agents: List[Agent],
        agent_weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        max_rounds: int = 3,
        agreement_threshold: float = 0.75  # Percentage required for consensus
    ) -> Dict[str, Any]:
        """Build consensus through multiple rounds if needed.
        
        Args:
            question: The question to ask agents
            options: List of options to choose from
            agents: List of agents to query
            agent_weights: Optional weights for each agent
            context: Additional context
            max_rounds: Maximum number of voting rounds
            agreement_threshold: Threshold for considering consensus reached
            
        Returns:
            Consensus result with voting data
        """
        rounds = []
        current_options = options.copy()
        current_context = context.copy() if context else {}
        
        for round_num in range(1, max_rounds + 1):
            # Set round information in context
            current_context["round_num"] = round_num
            current_context["max_rounds"] = max_rounds
            current_context["previous_rounds"] = rounds.copy()
            
            # Adjust question for later rounds
            round_question = question
            if round_num > 1:
                round_question = (
                    f"Round {round_num}/{max_rounds}: {question}\n\n"
                    f"Previous round results: {self._summarize_previous_round(rounds[-1])}"
                )
            
            # Gather votes for this round
            votes = await self.gather_votes(
                question=round_question,
                options=current_options,
                agents=agents,
                context=current_context
            )
            
            # Apply voting method
            winning_option, results = await self.voting_method.tally_votes(
                votes=votes,
                weights=agent_weights
            )
            
            # Record this round
            round_result = {
                "round": round_num,
                "votes": {
                    agent: [opt.to_dict() for opt in options]
                    for agent, options in votes.items()
                },
                "winning_option": winning_option.to_dict() if winning_option else None,
                **results
            }
            rounds.append(round_result)
            
            # Check if we reached agreement threshold
            if winning_option and results.get("winner_percentage", 0) >= agreement_threshold:
                # Consensus reached
                break
                
            # For later rounds, possibly reduce options
            if round_num < max_rounds and winning_option:
                # Keep top options - e.g., top half
                scores = results.get("vote_counts", {})
                if not scores:
                    scores = results.get("scores", {})
                    
                # Sort options by score
                sorted_options = sorted(
                    [(opt, scores.get(opt, 0)) for opt in current_options],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Keep at least 2 options, or top half
                keep_count = max(2, len(sorted_options) // 2)
                current_options = [opt for opt, _ in sorted_options[:keep_count]]
        
        # Prepare final result
        winning_option = rounds[-1].get("winning_option")
        consensus_reached = (
            winning_option is not None and
            rounds[-1].get("winner_percentage", 0) >= agreement_threshold
        )
        
        consensus_result = {
            "success": winning_option is not None,
            "consensus_reached": consensus_reached,
            "rounds": rounds,
            "final_round": len(rounds),
            "winning_option": winning_option,
            "voting_method": self.voting_method.__class__.__name__,
            "agreement_threshold": agreement_threshold,
            "agreement_percentage": rounds[-1].get("winner_percentage", 0) if winning_option else 0
        }
        
        return consensus_result
    
    def _summarize_previous_round(self, round_result: Dict[str, Any]) -> str:
        """Create a summary of the previous round.
        
        Args:
            round_result: Previous round results
            
        Returns:
            Summary text
        """
        winning_option = round_result.get("winning_option")
        if not winning_option:
            return "No clear winner emerged."
            
        winner_percentage = round_result.get("winner_percentage", 0)
        return f"Option '{winning_option.get('description', winning_option['value'])}' led with {winner_percentage:.1f}% support."


class DelphinConsensus(ConsensusBuilder):
    """Consensus builder using a Delphi method with anonymous feedback."""
    
    async def build_consensus(
        self,
        question: str,
        options: List[ConsensusOption],
        agents: List[Agent],
        agent_weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
        max_rounds: int = 3,
        agreement_threshold: float = 0.75
    ) -> Dict[str, Any]:
        """Build consensus using the Delphi method.
        
        Args:
            question: The question to ask agents
            options: List of options to choose from
            agents: List of agents to query
            agent_weights: Optional weights for each agent
            context: Additional context
            max_rounds: Maximum number of voting rounds
            agreement_threshold: Threshold for considering consensus reached
            
        Returns:
            Consensus result with voting data
        """
        rounds = []
        current_context = context.copy() if context else {}
        
        for round_num in range(1, max_rounds + 1):
            # Set round information in context
            current_context["round_num"] = round_num
            current_context["max_rounds"] = max_rounds
            current_context["previous_rounds"] = rounds.copy()
            
            # Adjust question for later rounds, adding anonymous feedback
            round_question = question
            if round_num > 1:
                round_question = (
                    f"Round {round_num}/{max_rounds}: {question}\n\n"
                    f"{self._gather_anonymous_feedback(rounds[-1])}\n\n"
                    f"Previous round statistics: {self._summarize_statistics(rounds[-1])}"
                )
            
            # Gather votes for this round
            votes = await self.gather_votes(
                question=round_question,
                options=options,
                agents=agents,
                context=current_context
            )
            
            # For Delphi, also collect reasoning/explanations
            reasoning = await self._collect_reasoning(
                question=question,
                options=options,
                votes=votes,
                agents=agents,
                context=current_context
            )
            
            # Apply voting method
            winning_option, results = await self.voting_method.tally_votes(
                votes=votes,
                weights=agent_weights
            )
            
            # Record this round
            round_result = {
                "round": round_num,
                "votes": {
                    agent: [opt.to_dict() for opt in options]
                    for agent, options in votes.items()
                },
                "reasoning": reasoning,
                "winning_option": winning_option.to_dict() if winning_option else None,
                **results
            }
            rounds.append(round_result)
            
            # Check if we reached agreement threshold
            if winning_option and results.get("winner_percentage", 0) >= agreement_threshold:
                # Consensus reached
                break
        
        # Prepare final result
        winning_option = rounds[-1].get("winning_option")
        consensus_reached = (
            winning_option is not None and
            rounds[-1].get("winner_percentage", 0) >= agreement_threshold
        )
        
        consensus_result = {
            "success": winning_option is not None,
            "consensus_reached": consensus_reached,
            "rounds": rounds,
            "final_round": len(rounds),
            "winning_option": winning_option,
            "voting_method": self.voting_method.__class__.__name__,
            "agreement_threshold": agreement_threshold,
            "agreement_percentage": rounds[-1].get("winner_percentage", 0) if winning_option else 0
        }
        
        return consensus_result
    
    async def _collect_reasoning(
        self,
        question: str,
        options: List[ConsensusOption],
        votes: Dict[str, List[ConsensusOption]],
        agents: List[Agent],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Collect reasoning from agents about their votes.
        
        Args:
            question: Original question
            options: Available options
            votes: Votes collected
            agents: List of agents
            context: Current context
            
        Returns:
            Dictionary mapping agent names to their reasoning
        """
        reasoning = {}
        
        for agent in agents:
            if agent.name not in votes:
                continue
                
            agent_vote = votes[agent.name]
            if not agent_vote:
                continue
                
            # Get agent's top choice
            top_choice = agent_vote[0]
                
            # Prepare prompt for explanation
            prompt = (
                f"You selected '{top_choice.description}' as your top choice for the question:\n"
                f"{question}\n\n"
                "Please provide a brief explanation of why you made this choice. "
                "Your reasoning will be shared anonymously with other participants."
            )
            
            # Add context
            agent_context = context.copy()
            agent_context["request_type"] = "explain_vote"
            
            try:
                # Run the agent
                result = await agent.run(AgentInput(query=prompt, context=agent_context))
                reasoning[agent.name] = result.response
            except Exception as e:
                logger.error(f"Error collecting reasoning from {agent.name}: {e}")
                reasoning[agent.name] = "No explanation provided."
        
        return reasoning
    
    def _gather_anonymous_feedback(self, round_result: Dict[str, Any]) -> str:
        """Create anonymous feedback from previous round.
        
        Args:
            round_result: Previous round results
            
        Returns:
            Anonymous feedback text
        """
        reasoning = round_result.get("reasoning", {})
        if not reasoning:
            return "No feedback from previous round."
            
        feedback = ["Anonymous feedback from participants:"]
        for explanation in reasoning.values():
            # Extract a concise version (first paragraph or sentence)
            if explanation:
                if "\n" in explanation:
                    concise = explanation.split("\n")[0]
                else:
                    concise = explanation.split(". ")[0]
                feedback.append(f"- {concise}")
                
        return "\n".join(feedback)
    
    def _summarize_statistics(self, round_result: Dict[str, Any]) -> str:
        """Create statistical summary of the previous round.
        
        Args:
            round_result: Previous round results
            
        Returns:
            Statistical summary text
        """
        vote_counts = round_result.get("vote_counts") or round_result.get("scores", {})
        if not vote_counts:
            return "No voting statistics available."
            
        # Find option descriptions
        option_descriptions = {}
        votes = round_result.get("votes", {})
        for agent_votes in votes.values():
            for opt in agent_votes:
                option_descriptions[opt["value"]] = opt["description"]
        
        # Create summary
        stats = ["Voting statistics:"]
        total = sum(vote_counts.values())
        
        for option, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True):
            if isinstance(option, dict):
                option_key = option.get("value")
                desc = option.get("description")
            else:
                option_key = option
                desc = option_descriptions.get(option, str(option))
                
            percentage = (count / total) * 100 if total > 0 else 0
            stats.append(f"- {desc}: {percentage:.1f}%")
            
        return "\n".join(stats)


class MajorityConcensus(ConsensusBuilder):
    """Simple majority-based consensus builder."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        required_majority: float = 0.5,  # >50% is a simple majority
        abstention_allowed: bool = True
    ):
        """Initialize the majority consensus builder.
        
        Args:
            name: Name for this consensus builder
            required_majority: Required percentage for majority
            abstention_allowed: Whether agents can abstain
        """
        super().__init__(name=name, voting_method=PluralityVoting())
        self.required_majority = required_majority
        self.abstention_allowed = abstention_allowed
        
    async def build_consensus(
        self,
        question: str,
        options: List[ConsensusOption],
        agents: List[Agent],
        agent_weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build consensus using majority rule.
        
        Args:
            question: The question to ask agents
            options: List of options to choose from
            agents: List of agents to query
            agent_weights: Optional weights for each agent
            context: Additional context
            
        Returns:
            Consensus result with voting data
        """
        # If abstention is allowed, add an abstain option
        working_options = options.copy()
        if self.abstention_allowed:
            abstain_option = ConsensusOption(
                value="abstain",
                description="Abstain from voting",
                metadata={"is_abstain": True}
            )
            working_options.append(abstain_option)
        
        # Gather votes
        votes = await self.gather_votes(
            question=question,
            options=working_options,
            agents=agents,
            context=context
        )
        
        # Apply voting method
        winning_option, results = await self.voting_method.tally_votes(
            votes=votes,
            weights=agent_weights
        )
        
        # Check if winning option is abstention
        if winning_option and hasattr(winning_option, 'metadata') and winning_option.metadata.get("is_abstain"):
            consensus_reached = False
            abstain_percentage = results.get("winner_percentage", 0)
            
            # Find the top non-abstain option
            vote_counts = results.get("vote_counts", {})
            non_abstain_options = []
            for option, count in vote_counts.items():
                if not (hasattr(option, 'metadata') and option.metadata.get("is_abstain")):
                    non_abstain_options.append((option, count))
                    
            if non_abstain_options:
                winning_option = max(non_abstain_options, key=lambda x: x[1])[0]
                winning_score = vote_counts[winning_option]
                
                # Recalculate percentage against non-abstaining votes
                active_vote_total = sum(count for option, count in vote_counts.items() 
                                      if not (hasattr(option, 'metadata') and option.metadata.get("is_abstain")))
                
                if active_vote_total > 0:
                    winning_percentage = (winning_score / active_vote_total) * 100
                else:
                    winning_percentage = 0
                    
                # Update results
                results["winner_score"] = winning_score
                results["winner_percentage"] = winning_percentage  
                results["abstain_percentage"] = abstain_percentage
            else:
                # Everyone abstained
                consensus_reached = False
                winning_option = None
                
        # Check majority threshold
        consensus_reached = (
            winning_option is not None and
            not (hasattr(winning_option, 'metadata') and winning_option.metadata.get("is_abstain")) and
            results.get("winner_percentage", 0) >= (self.required_majority * 100)
        )
        
        # Prepare result
        consensus_result = {
            "success": winning_option is not None,
            "consensus_reached": consensus_reached,
            "winning_option": winning_option.to_dict() if winning_option else None,
            "required_majority": self.required_majority * 100,
            "actual_majority": results.get("winner_percentage", 0),
            "votes": {
                agent: [opt.to_dict() for opt in options]
                for agent, options in votes.items()
            },
            "abstention_allowed": self.abstention_allowed,
            **results
        }
        
        return consensus_result


class UnanimityConsensus(ConsensusBuilder):
    """Requires complete unanimity (or near unanimity) among agents."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        unanimity_threshold: float = 1.0,  # 1.0 = 100% agreement required
        max_rounds: int = 5
    ):
        """Initialize the unanimity consensus builder.
        
        Args:
            name: Name for this consensus builder
            unanimity_threshold: Required percentage (near unanimity)
            max_rounds: Maximum rounds for discussion
        """
        super().__init__(name=name, voting_method=PluralityVoting())
        self.unanimity_threshold = unanimity_threshold
        self.max_rounds = max_rounds
        
    async def build_consensus(
        self,
        question: str,
        options: List[ConsensusOption],
        agents: List[Agent],
        agent_weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build consensus requiring unanimity or near-unanimity.
        
        Args:
            question: The question to ask agents
            options: List of options to choose from
            agents: List of agents to query
            agent_weights: Optional weights for each agent
            context: Additional context
            
        Returns:
            Consensus result with voting data
        """
        rounds = []
        current_context = context.copy() if context else {}
        
        for round_num in range(1, self.max_rounds + 1):
            # Set round information in context
            current_context["round_num"] = round_num
            current_context["max_rounds"] = self.max_rounds
            current_context["previous_rounds"] = rounds.copy()
            
            # Adjust question for later rounds
            round_question = question
            if round_num > 1:
                round_question = (
                    f"Round {round_num}/{self.max_rounds}: {question}\n\n"
                    f"Unanimity is required. Previous round showed "
                    f"{self._summarize_votes(rounds[-1])}\n\n"
                    f"Please reconsider to reach consensus."
                )
            
            # Gather votes for this round
            votes = await self.gather_votes(
                question=round_question,
                options=options,
                agents=agents,
                context=current_context
            )
            
            # Apply voting method
            winning_option, results = await self.voting_method.tally_votes(
                votes=votes,
                weights=agent_weights
            )
            
            # Record this round
            round_result = {
                "round": round_num,
                "votes": {
                    agent: [opt.to_dict() for opt in options]
                    for agent, options in votes.items()
                },
                "winning_option": winning_option.to_dict() if winning_option else None,
                **results
            }
            rounds.append(round_result)
            
            # Check for unanimity or near-unanimity
            winner_percentage = results.get("winner_percentage", 0)
            if winner_percentage >= (self.unanimity_threshold * 100):
                # Consensus reached
                break
        
        # Prepare final result
        winning_option = rounds[-1].get("winning_option")
        winner_percentage = rounds[-1].get("winner_percentage", 0)
        consensus_reached = (
            winning_option is not None and
            winner_percentage >= (self.unanimity_threshold * 100)
        )
        
        consensus_result = {
            "success": winning_option is not None,
            "consensus_reached": consensus_reached,
            "rounds": rounds,
            "final_round": len(rounds),
            "winning_option": winning_option,
            "unanimity_threshold": self.unanimity_threshold * 100,
            "actual_agreement": winner_percentage,
            "voting_method": self.voting_method.__class__.__name__
        }
        
        return consensus_result
        
    def _summarize_votes(self, round_result: Dict[str, Any]) -> str:
        """Summarize voting from previous round.
        
        Args:
            round_result: Previous round results
            
        Returns:
            Vote summary text
        """
        winning_option = round_result.get("winning_option")
        if not winning_option:
            return "no clear preference"
            
        winner_percentage = round_result.get("winner_percentage", 0)
        option_desc = winning_option.get("description", winning_option.get("value", "unknown"))
        
        return f"{winner_percentage:.1f}% agreement for '{option_desc}'"
