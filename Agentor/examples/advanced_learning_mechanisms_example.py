"""
Example demonstrating advanced learning mechanisms in the Agentor framework.

This example shows how to use the advanced learning capabilities:
- Adaptive learning rate scheduling
- Meta-learning with feedback loops
- Prioritized experience replay
- Curriculum learning
"""

import asyncio
import logging
import numpy as np
import time
import os
import random
from typing import Dict, Any, List, Optional

from agentor.components.learning import DeepQLearning, PPOAgent, Experience
from agentor.components.learning.advanced_learning import (
    FeedbackEnhancedLearning,
    AdaptiveLearningRateScheduler,
    FeedbackCollector,
    FeedbackEntry,
    PrioritizedExperienceReplay,
    CurriculumLearning
)
from agentor.components.environments import CartPoleEnv
from agentor.agents.learning import EnhancedDeepQLearningAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CartPoleFeedbackEnhancedAgent(EnhancedDeepQLearningAgent):
    """CartPole agent with feedback-enhanced learning."""
    
    def __init__(self, **kwargs):
        """Initialize the feedback-enhanced CartPole agent."""
        super().__init__(**kwargs)
        
        # Create feedback collector
        self.feedback_collector = FeedbackCollector(
            feedback_sources=["agent", "environment", "analytics"],
            feedback_window=1000
        )
        
        # Wrap the base learning algorithm with feedback enhancement
        self.feedback_enhanced_learning = FeedbackEnhancedLearning(
            base_algorithm=self.learning_algorithm,
            feedback_collector=self.feedback_collector,
            prioritized_replay=True,
            use_curriculum=True,
            meta_learning=True
        )
        
        # Replace the learning algorithm with the enhanced version
        self._original_learning_algorithm = self.learning_algorithm
        self.learning_algorithm = self.feedback_enhanced_learning
    
    async def update(self, state, action, reward, next_state, done, info=None):
        """Update the agent with an experience and generate feedback.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional information
        """
        # Create experience object
        experience = Experience(state, action, reward, next_state, done, info or {})
        
        # Generate internal feedback based on experience
        feedback = await self._generate_feedback(experience)
        
        # Update using enhanced learning with feedback
        metrics = await self.feedback_enhanced_learning.update(experience, feedback)
        
        return metrics
    
    async def _generate_feedback(self, experience: Experience) -> FeedbackEntry:
        """Generate internal feedback on the experience.
        
        Args:
            experience: Experience to evaluate
            
        Returns:
            FeedbackEntry object
        """
        # Simple heuristic: higher rewards = better feedback
        reward = experience.reward
        
        # Scale to 0-1 range (assuming rewards are typically in [-1, 1])
        normalized_reward = (reward + 1) / 2
        
        # Generate some simple feedback text
        if reward > 0.5:
            feedback_text = "Good action maintaining balance"
        elif reward > 0:
            feedback_text = "Acceptable action but could be improved"
        else:
            feedback_text = "Poor action leading to instability"
        
        return FeedbackEntry(
            state=experience.state,
            action=experience.action,
            result=reward,
            feedback_score=normalized_reward,
            feedback_text=feedback_text,
            source="agent"
        )
    
    async def save_model(self, path: str):
        """Save the enhanced learning model.
        
        Args:
            path: Path to save to
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Save model using the feedback enhanced learning's save method
        self.feedback_enhanced_learning.save(path)
    
    async def load_model(self, path: str):
        """Load the enhanced learning model.
        
        Args:
            path: Path to load from
        """
        # Load model using the feedback enhanced learning's load method
        self.feedback_enhanced_learning.load(path)


async def adaptive_learning_rate_example():
    """Example demonstrating adaptive learning rate scheduling."""
    logger.info("\n=== Adaptive Learning Rate Example ===")
    
    # Create a base learning algorithm
    dql = DeepQLearning(
        state_dim=4,  # CartPole has 4 state dimensions
        action_dim=2,  # CartPole has 2 actions
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=0.1,
        batch_size=32
    )
    
    # Create a learning rate scheduler
    scheduler = AdaptiveLearningRateScheduler(
        initial_lr=0.001,
        min_lr=0.0001,
        max_lr=0.01,
        patience=5,
        factor=0.5,
        mode='min',  # Lower loss is better
        warmup_steps=50,
        warmup_factor=0.1
    )
    
    # Simulate training with changing loss patterns
    logger.info("Simulating training with adaptive learning rate:")
    
    # Initial high loss that gradually decreases
    for i in range(1, 21):
        loss = 1.0 / i
        new_lr = scheduler.step(loss)
        logger.info(f"Step {i}: Loss = {loss:.4f}, Learning Rate = {new_lr:.6f}")
    
    # Loss plateau (no improvement) - should trigger LR reduction
    for i in range(21, 31):
        loss = 0.05 + random.uniform(-0.005, 0.005)  # Small random fluctuations
        new_lr = scheduler.step(loss)
        logger.info(f"Step {i}: Loss = {loss:.4f}, Learning Rate = {new_lr:.6f}")
    
    # Improvement after LR reduction
    for i in range(31, 41):
        loss = 0.05 / (i - 29)  # Continuing improvement
        new_lr = scheduler.step(loss)
        logger.info(f"Step {i}: Loss = {loss:.4f}, Learning Rate = {new_lr:.6f}")
    
    logger.info("Adaptive learning rate scheduling complete")


async def prioritized_replay_example():
    """Example demonstrating prioritized experience replay."""
    logger.info("\n=== Prioritized Experience Replay Example ===")
    
    # Create prioritized replay buffer
    buffer = PrioritizedExperienceReplay(
        capacity=100,
        alpha=0.6,
        beta=0.4
    )
    
    # Add some experiences with varying priorities
    logger.info("Adding experiences with different priorities...")
    for i in range(50):
        # Generate synthetic experience
        state = np.array([random.uniform(-1, 1) for _ in range(4)])
        action = random.randint(0, 1)
        reward = random.uniform(-1, 1)
        next_state = np.array([random.uniform(-1, 1) for _ in range(4)])
        done = random.random() < 0.1
        
        # TD error (priority) - higher for some experiences
        td_error = abs(reward) + random.uniform(0, 1)
        
        # Create experience
        exp = Experience(state, action, reward, next_state, done)
        
        # Add to buffer with priority
        buffer.add(exp, priority=td_error)
    
    # Sample from the buffer multiple times
    logger.info("Sampling from prioritized replay buffer:")
    for i in range(5):
        experiences, indices, weights = buffer.sample(batch_size=10)
        
        # Show sampling results
        priorities = [buffer.priorities[idx] for idx in indices]
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Mean priority: {np.mean(priorities):.4f}")
        logger.info(f"  Min priority: {np.min(priorities):.4f}")
        logger.info(f"  Max priority: {np.max(priorities):.4f}")
        logger.info(f"  Mean IS weight: {np.mean(weights):.4f}")
    
    # Update some priorities
    logger.info("Updating priorities for some experiences...")
    buffer.update_priorities(
        indices=indices[:5],
        priorities=np.array([2.0, 1.5, 1.0, 0.5, 0.1])
    )
    
    # Sample again to see the effect
    experiences, indices, weights = buffer.sample(batch_size=10)
    priorities = [buffer.priorities[idx] for idx in indices]
    logger.info("After priority update:")
    logger.info(f"  Mean priority: {np.mean(priorities):.4f}")
    logger.info(f"  Min priority: {np.min(priorities):.4f}")
    logger.info(f"  Max priority: {np.max(priorities):.4f}")
    
    logger.info("Prioritized experience replay example complete")


async def curriculum_learning_example():
    """Example demonstrating curriculum learning with progressive difficulty."""
    logger.info("\n=== Curriculum Learning Example ===")
    
    # Define difficulty levels
    difficulty_levels = [
        {"name": "beginner", "pole_length": 0.5, "force_mag": 10.0, "time_limit": 200},
        {"name": "intermediate", "pole_length": 0.3, "force_mag": 8.0, "time_limit": 150},
        {"name": "advanced", "pole_length": 0.2, "force_mag": 7.0, "time_limit": 100},
        {"name": "expert", "pole_length": 0.1, "force_mag": 6.0, "time_limit": 75}
    ]
    
    # Create curriculum learning manager
    curriculum = CurriculumLearning(
        difficulty_levels=difficulty_levels,
        promotion_threshold=0.8,
        min_episodes=5,
        success_window=3,
        demotion_threshold=0.3
    )
    
    # Simulate episodes with different success patterns
    logger.info("Simulating agent training with curriculum:")
    
    # Initial episodes - learning basics
    for i in range(1, 6):
        # Simulate success with increasing probability
        success = random.random() < (i / 5)
        metrics = {"reward": 10 if success else 0, "steps": 100 if success else 50}
        curriculum.record_episode_result(success, metrics)
        
        config = curriculum.get_current_config()
        progress = curriculum.get_progress()
        logger.info(f"Episode {i}: Level={config['name']}, Success={success}, "
                   f"Success Rate={progress['success_rate']:.2f}")
    
    # More episodes - getting better
    for i in range(6, 11):
        success = random.random() < 0.8  # High success rate
        metrics = {"reward": 15 if success else 5, "steps": 120 if success else 60}
        curriculum.record_episode_result(success, metrics)
        
        config = curriculum.get_current_config()
        progress = curriculum.get_progress()
        logger.info(f"Episode {i}: Level={config['name']}, Success={success}, "
                   f"Success Rate={progress['success_rate']:.2f}")
    
    # More episodes - struggling with new difficulty
    for i in range(11, 16):
        success = random.random() < 0.4  # Lower success rate
        metrics = {"reward": 20 if success else 0, "steps": 150 if success else 30}
        curriculum.record_episode_result(success, metrics)
        
        config = curriculum.get_current_config()
        progress = curriculum.get_progress()
        logger.info(f"Episode {i}: Level={config['name']}, Success={success}, "
                   f"Success Rate={progress['success_rate']:.2f}")
    
    # Final episodes - mastering current level
    for i in range(16, 21):
        success = random.random() < 0.9  # High success rate again
        metrics = {"reward": 25 if success else 5, "steps": 180 if success else 40}
        curriculum.record_episode_result(success, metrics)
        
        config = curriculum.get_current_config()
        progress = curriculum.get_progress()
        logger.info(f"Episode {i}: Level={config['name']}, Success={success}, "
                   f"Success Rate={progress['success_rate']:.2f}")
    
    # Show final progress
    progress = curriculum.get_progress()
    logger.info(f"\nFinal curriculum progress: Level {progress['current_level']+1}/{progress['max_level']+1}")
    logger.info(f"Overall progress: {progress['progress']*100:.1f}%")
    
    # Show level history
    logger.info("Level progression history:")
    for change in progress["level_history"]:
        logger.info(f"  Episode {change['episode']}: {change['direction']} to level {change['level']+1}")


async def feedback_enhanced_agent_example():
    """Example demonstrating a feedback-enhanced reinforcement learning agent."""
    logger.info("\n=== Feedback-Enhanced Agent Example ===")
    
    # Create CartPole environment
    env = CartPoleEnv()
    
    # Create the agent
    agent = CartPoleFeedbackEnhancedAgent(
        state_dim=4,  # CartPole has 4 state dimensions
        action_dim=2,  # CartPole has 2 actions
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=0.1,
        batch_size=32,
        memory_size=10000,
        hidden_dim=64
    )
    
    # Train the agent
    logger.info("Training the feedback-enhanced agent...")
    num_episodes = 50
    max_steps = 200
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = await agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update agent with experience and feedback
            metrics = await agent.update(state, action, reward, next_state, done, info)
            
            # Accumulate reward
            episode_reward += reward
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Record episode statistics
        total_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Adjust learning rate based on recent performance
        if episode >= 5:  # Wait for a few episodes to get meaningful metrics
            mean_reward = np.mean(total_rewards[-5:])
            await agent.feedback_enhanced_learning.adjust_learning_rate(1.0 / mean_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            logger.info(f"Episode {episode+1}/{num_episodes}: "
                       f"Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_length:.2f}")
    
    # Show final results
    avg_reward = np.mean(total_rewards[-10:])
    avg_length = np.mean(episode_lengths[-10:])
    logger.info(f"Training completed: Final Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_length:.2f}")
    
    # Get feedback statistics
    feedback_stats = agent.feedback_collector.get_feedback_statistics()
    logger.info(f"Feedback collected: {feedback_stats['count']} entries")
    logger.info(f"Average feedback score: {feedback_stats['avg_score']:.4f}")
    
    # Save the model
    os.makedirs("./models", exist_ok=True)
    await agent.save_model("./models/feedback_enhanced_cartpole.pt")
    logger.info("Model saved to ./models/feedback_enhanced_cartpole.pt")


async def main():
    """Run all examples."""
    await adaptive_learning_rate_example()
    await prioritized_replay_example()
    await curriculum_learning_example()
    await feedback_enhanced_agent_example()


if __name__ == "__main__":
    asyncio.run(main())