from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import random
import json
import os
import logging
import re
from pathlib import Path

from agentor.agents import Agent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class ReinforcementLearningAgent(Agent):
    """Base class for reinforcement learning agents."""

    def __init__(self, name=None, states=None, actions=None):
        """Initialize the reinforcement learning agent.

        Args:
            name: The name of the agent
            states: A list of possible states
            actions: A list of possible actions
        """
        super().__init__(name)
        self.states = states or []
        self.actions = actions or []

    def get_state(self) -> str:
        """Get the current state of the agent.

        Returns:
            The current state
        """
        raise NotImplementedError("Subclasses must implement get_state()")

    def get_reward(self, state: str, action: str, next_state: str) -> float:
        """Get the reward for an action.

        Args:
            state: The state before the action
            action: The action taken
            next_state: The state after the action

        Returns:
            The reward
        """
        raise NotImplementedError("Subclasses must implement get_reward()")


class QLearningAgent(ReinforcementLearningAgent):
    """An agent that learns using Q-learning."""

    def __init__(
        self,
        name=None,
        states=None,
        actions=None,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.1
    ):
        """Initialize the Q-learning agent.

        Args:
            name: The name of the agent
            states: A list of possible states
            actions: A list of possible actions
            learning_rate: The learning rate (alpha)
            discount_factor: The discount factor (gamma)
            exploration_rate: The exploration rate (epsilon)
        """
        super().__init__(name, states, actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # Initialize the Q-table
        self.q_table: Dict[str, Dict[str, float]] = {}
        for state in self.states:
            self.q_table[state] = {}
            for action in self.actions:
                self.q_table[state][action] = 0.0

    def decide(self) -> str:
        """Choose an action using epsilon-greedy policy.

        Returns:
            The name of the action to take
        """
        # Get the current state
        state = self.get_state()

        # Ensure the state is in the Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
            for action in self.actions:
                self.q_table[state][action] = 0.0

        # Choose an action using epsilon-greedy policy
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            return random.choice(self.actions)
        else:
            # Exploit: choose the best action
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]

    def learn(self, state: str, action: str, reward: float, next_state: str):
        """Update the Q-table based on an experience.

        Args:
            state: The state before the action
            action: The action taken
            reward: The reward received
            next_state: The state after the action
        """
        # Ensure the states are in the Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
            for a in self.actions:
                self.q_table[state][a] = 0.0

        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            for a in self.actions:
                self.q_table[next_state][a] = 0.0

        # Get the current Q-value
        current_q = self.q_table[state][action]

        # Get the maximum Q-value for the next state
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0

        # Calculate the new Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        # Update the Q-table
        self.q_table[state][action] = new_q

        logger.debug(f"Updated Q-value for state {state}, action {action}: {current_q} -> {new_q}")

    def save_q_table(self, file_path: str):
        """Save the Q-table to a file.

        Args:
            file_path: The path to save the Q-table to
        """
        # Ensure the directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.q_table, f, indent=2)

        logger.info(f"Saved Q-table to {file_path}")

    def load_q_table(self, file_path: str):
        """Load the Q-table from a file.

        Args:
            file_path: The path to load the Q-table from
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.q_table = json.load(f)

            logger.info(f"Loaded Q-table from {file_path}")
        else:
            logger.warning(f"Q-table file {file_path} not found")

    async def run_once(self):
        """Run one perception-decision-action-learning cycle.

        Returns:
            The result of the action
        """
        # Get the current state
        state = self.get_state()

        # Choose an action
        action = self.decide()

        # Execute the action
        result = self.act(action)

        # Get the new state
        next_state = self.get_state()

        # Get the reward
        reward = self.get_reward(state, action, next_state)

        # Learn from the experience
        self.learn(state, action, reward, next_state)

        return result


class LLMFeedbackAgent(Agent):
    """An agent that learns from LLM feedback."""

    def __init__(self, name=None, llm=None, learning_rate=0.1):
        """Initialize the LLM feedback agent.

        Args:
            name: The name of the agent
            llm: The LLM to use for feedback
            learning_rate: The learning rate
        """
        super().__init__(name)
        self.llm = llm
        self.learning_rate = learning_rate
        self.feedback_history: List[Dict[str, Any]] = []
        self.prompt_templates: Dict[str, str] = {}

    def add_prompt_template(self, name: str, template: str):
        """Add a prompt template.

        Args:
            name: The name of the template
            template: The template string
        """
        self.prompt_templates[name] = template

    async def get_feedback(self, action: str, result: str) -> Tuple[float, str]:
        """Get feedback from the LLM.

        Args:
            action: The action taken
            result: The result of the action

        Returns:
            A tuple of (score, feedback)
        """
        if not self.llm:
            return 0.5, "No LLM available for feedback"

        prompt = self.prompt_templates.get("feedback", """
        Please evaluate the following action and result.

        Action: {action}
        Result: {result}

        Rate the action on a scale from 0.0 to 1.0, where 0.0 is terrible and 1.0 is perfect.
        Also provide feedback on how to improve.

        Format your response as:
        Score: [score]
        Feedback: [feedback]
        """).format(action=action, result=result)

        from agentor.llm_gateway.llm.base import LLMRequest

        response = await self.llm.generate(
            LLMRequest(
                prompt=prompt,
                model="gpt-4",  # Use a model with good reasoning capabilities
                temperature=0.3
            )
        )

        # Parse the response
        score_match = re.search(r'Score: ([\d\.]+)', response.text)
        feedback_match = re.search(r'Feedback: (.*)', response.text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.5
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided"

        return score, feedback

    async def learn_from_feedback(self, state: str, action: str, result: str):
        """Learn from LLM feedback.

        Args:
            state: The state before the action
            action: The action taken
            result: The result of the action
        """
        score, feedback = await self.get_feedback(action, result)

        # Store the feedback
        self.feedback_history.append({
            "state": state,
            "action": action,
            "result": result,
            "score": score,
            "feedback": feedback
        })

        # Use the feedback to improve (specific implementation depends on the agent)
        # For example, we could update prompt templates or action selection strategies

        logger.info(f"Received feedback for action {action}: {score:.2f} - {feedback}")

        return score, feedback