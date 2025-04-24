"""
Advanced Learning Mechanisms for the Agentor framework.

This module provides advanced learning techniques including:
- Adaptive learning rate scheduling
- Meta-learning capabilities
- Enhanced feedback loops
- Experience prioritization
- Curriculum learning
"""

import logging
import numpy as np
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Type
from dataclasses import dataclass
import math
import random

from agentor.components.learning import Experience, LearningAlgorithm
from agentor.agents import Agent
from agentor.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackEntry:
    """Feedback data for learning improvement."""
    state: Any
    action: Any
    result: Any
    feedback_score: float
    feedback_text: Optional[str] = None
    source: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class AdaptiveLearningRateScheduler:
    """Dynamically adjusts learning rates based on performance metrics."""
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        min_lr: float = 0.00001,
        max_lr: float = 0.01,
        patience: int = 5,
        cooldown: int = 2,
        factor: float = 0.5,
        mode: str = 'min',  # 'min' for loss, 'max' for reward/accuracy
        warmup_steps: int = 0,
        warmup_factor: float = 1.0
    ):
        """Initialize the learning rate scheduler.
        
        Args:
            initial_lr: Starting learning rate
            min_lr: Minimum allowable learning rate
            max_lr: Maximum allowable learning rate
            patience: How many steps with no improvement before reducing LR
            cooldown: How many steps after LR reduction before monitoring again
            factor: Factor by which to reduce learning rate
            mode: 'min' if lower metric is better, 'max' if higher is better
            warmup_steps: Number of steps for learning rate warmup
            warmup_factor: Factor to multiply learning rate during warmup
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.cooldown = cooldown
        self.factor = factor
        self.mode = mode
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        
        # Internal state
        self.step_count = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait_count = 0
        self.cooldown_counter = 0
        self.history: List[Dict[str, Any]] = []
    
    def step(self, metric_value: float) -> float:
        """Update learning rate based on the latest metric value.
        
        Args:
            metric_value: Current performance metric
            
        Returns:
            Updated learning rate
        """
        self.step_count += 1
        self.history.append({
            'step': self.step_count,
            'lr': self.current_lr,
            'metric': metric_value
        })
        
        # Handle warmup phase
        if self.step_count <= self.warmup_steps:
            progress = self.step_count / max(1, self.warmup_steps)
            warmup_lr = self.initial_lr * (1 - (1 - self.warmup_factor) * (1 - progress))
            self.current_lr = min(self.max_lr, warmup_lr)
            return self.current_lr
            
        # Skip if in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr
        
        # Check if performance improved
        improved = (self.mode == 'min' and metric_value < self.best_metric) or \
                  (self.mode == 'max' and metric_value > self.best_metric)
        
        if improved:
            self.best_metric = metric_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        # Reduce LR if no improvement for patience steps
        if self.wait_count >= self.patience:
            # Reduce learning rate
            self.current_lr = max(self.min_lr, self.current_lr * self.factor)
            self.wait_count = 0
            self.cooldown_counter = self.cooldown
            logger.info(f"Learning rate reduced to {self.current_lr:.6f}")
        
        return self.current_lr
    
    def reset(self):
        """Reset the scheduler state."""
        self.current_lr = self.initial_lr
        self.step_count = 0
        self.wait_count = 0
        self.cooldown_counter = 0
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.history = []


class PrioritizedExperienceReplay:
    """Experience replay buffer with prioritized sampling based on TD error."""
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001,
        epsilon: float = 1e-6  # Small constant to avoid zero priority
    ):
        """Initialize the prioritized experience replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Priority exponent - how much prioritization to use (0=none, 1=full)
            beta: Importance sampling correction exponent
            beta_increment: How much to increase beta over time
            epsilon: Small constant to avoid zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        # Storage
        self.experiences = []  
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0
    
    def add(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add an experience to the buffer.
        
        Args:
            experience: Experience to add
            priority: Optional priority value. If None, max priority is used.
        """
        if priority is None:
            priority = self.max_priority
            
        # Convert to float32 and ensure it's positive
        priority = float(max(abs(priority), self.epsilon))
        self.max_priority = max(self.max_priority, priority)
        
        # Add experience
        if self.size < self.capacity:
            # Still filling buffer
            self.experiences.append(experience)
            self.priorities[self.size] = priority
            self.size += 1
        else:
            # Overwrite old experiences
            self.experiences[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, importance_sampling_weights)
        """
        if self.size < batch_size:
            batch_size = self.size
            
        # Get sampling probabilities from priorities
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increase beta over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Return experiences, indices, and weights
        return [self.experiences[i] for i in indices], indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for experiences at specified indices.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for i, priority in zip(indices, priorities):
            # Convert to float32 and ensure it's positive
            priority = float(max(abs(priority), self.epsilon))
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size


class FeedbackCollector:
    """Collects and processes feedback for learning improvement."""
    
    def __init__(
        self,
        feedback_sources: Optional[List[str]] = None,
        feedback_window: int = 1000,
        min_feedback_count: int = 10
    ):
        """Initialize the feedback collector.
        
        Args:
            feedback_sources: Names of feedback sources
            feedback_window: Maximum feedback entries to keep
            min_feedback_count: Minimum entries before using feedback
        """
        self.feedback_sources = feedback_sources or ["agent", "environment", "human", "llm"]
        self.feedback_history: List[FeedbackEntry] = []
        self.feedback_window = feedback_window
        self.min_feedback_count = min_feedback_count
        self.source_weights = {source: 1.0 for source in self.feedback_sources}
    
    async def add_feedback(self, feedback: FeedbackEntry) -> None:
        """Add a feedback entry to the history.
        
        Args:
            feedback: Feedback entry to add
        """
        self.feedback_history.append(feedback)
        
        # Keep history within size limits
        if len(self.feedback_history) > self.feedback_window:
            self.feedback_history = self.feedback_history[-self.feedback_window:]
    
    async def get_similar_feedback(
        self, 
        state: Any, 
        action: Any, 
        similarity_fn: Optional[Callable] = None,
        limit: int = 5
    ) -> List[FeedbackEntry]:
        """Get feedback entries similar to the current state-action pair.
        
        Args:
            state: Current state
            action: Current action
            similarity_fn: Function to compute similarity
            limit: Maximum entries to return
            
        Returns:
            List of similar feedback entries
        """
        if not self.feedback_history:
            return []
            
        if similarity_fn is None:
            # Default similarity function - very basic, should be replaced
            def default_similarity(entry):
                # Simple heuristic - more recent feedback is more relevant
                recency = 1.0 / (1.0 + (time.time() - entry.timestamp) / 86400)  # Decay over a day
                return recency
                
            similarity_fn = default_similarity
        
        # Calculate similarities and sort
        entries_with_similarity = [
            (entry, similarity_fn(entry)) 
            for entry in self.feedback_history
        ]
        
        sorted_entries = sorted(entries_with_similarity, key=lambda x: x[1], reverse=True)
        
        return [entry for entry, _ in sorted_entries[:limit]]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback.
        
        Returns:
            Dictionary of feedback statistics
        """
        if not self.feedback_history:
            return {"count": 0, "avg_score": 0.0}
            
        scores = [entry.feedback_score for entry in self.feedback_history]
        sources = [entry.source for entry in self.feedback_history]
        
        stats = {
            "count": len(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "sources": {source: sources.count(source) for source in set(sources)},
            "recent_trend": self._calculate_trend()
        }
        
        return stats
    
    def _calculate_trend(self, window: int = 20) -> float:
        """Calculate trend in recent feedback scores.
        
        Args:
            window: Number of recent entries to consider
            
        Returns:
            Trend value (positive=improving, negative=declining)
        """
        if len(self.feedback_history) < window:
            return 0.0
            
        recent = self.feedback_history[-window:]
        scores = [entry.feedback_score for entry in recent]
        
        if len(scores) < 2:
            return 0.0
            
        # Simple linear regression slope
        x = np.arange(len(scores))
        x_mean = np.mean(x)
        y_mean = np.mean(scores)
        
        numerator = np.sum((x - x_mean) * (scores - y_mean))
        denominator = np.sum((x - x_mean)**2)
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator


class CurriculumLearning:
    """Implements curriculum learning for progressive task difficulty."""
    
    def __init__(
        self,
        difficulty_levels: List[Dict[str, Any]],
        promotion_threshold: float = 0.8,
        min_episodes: int = 10,
        success_window: int = 5,
        demotion_threshold: float = 0.3
    ):
        """Initialize curriculum learning.
        
        Args:
            difficulty_levels: List of difficulty configurations
            promotion_threshold: Success rate needed for promotion
            min_episodes: Minimum episodes before considering promotion
            success_window: Number of episodes to consider for success rate
            demotion_threshold: Success rate below which to demote
        """
        self.difficulty_levels = difficulty_levels
        self.promotion_threshold = promotion_threshold
        self.min_episodes = min_episodes
        self.success_window = success_window
        self.demotion_threshold = demotion_threshold
        
        self.current_level = 0
        self.current_config = difficulty_levels[0]
        self.episode_results = []
        self.level_history = []
    
    def record_episode_result(self, success: bool, metrics: Dict[str, Any]) -> None:
        """Record the result of an episode.
        
        Args:
            success: Whether the episode was successful
            metrics: Additional performance metrics
        """
        self.episode_results.append({
            "success": success,
            "metrics": metrics,
            "level": self.current_level,
            "timestamp": time.time()
        })
        
        # Check if we should update difficulty
        self._update_difficulty()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current difficulty configuration.
        
        Returns:
            Current difficulty level configuration
        """
        return self.current_config.copy()
    
    def _update_difficulty(self) -> None:
        """Update difficulty based on recent performance."""
        # Need minimum number of episodes
        if len(self.episode_results) < self.min_episodes:
            return
            
        # Calculate success rate over recent episodes
        recent = self.episode_results[-self.success_window:]
        success_rate = sum(1 for r in recent if r["success"]) / len(recent)
        
        # Check for promotion
        if (self.current_level < len(self.difficulty_levels) - 1 and 
                success_rate >= self.promotion_threshold):
            self.current_level += 1
            self.current_config = self.difficulty_levels[self.current_level]
            self.level_history.append({
                "level": self.current_level,
                "direction": "promotion",
                "success_rate": success_rate,
                "episode": len(self.episode_results),
                "timestamp": time.time()
            })
            logger.info(f"Curriculum level promoted to {self.current_level}")
            
        # Check for demotion
        elif self.current_level > 0 and success_rate <= self.demotion_threshold:
            self.current_level -= 1
            self.current_config = self.difficulty_levels[self.current_level]
            self.level_history.append({
                "level": self.current_level,
                "direction": "demotion",
                "success_rate": success_rate,
                "episode": len(self.episode_results),
                "timestamp": time.time()
            })
            logger.info(f"Curriculum level demoted to {self.current_level}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get learning progress information.
        
        Returns:
            Dictionary with progress information
        """
        if not self.episode_results:
            return {
                "current_level": self.current_level,
                "max_level": len(self.difficulty_levels) - 1,
                "progress": 0.0,
                "success_rate": 0.0
            }
            
        recent = self.episode_results[-min(self.success_window, len(self.episode_results)):]
        success_rate = sum(1 for r in recent if r["success"]) / len(recent)
        
        return {
            "current_level": self.current_level,
            "max_level": len(self.difficulty_levels) - 1,
            "progress": self.current_level / max(1, len(self.difficulty_levels) - 1),
            "success_rate": success_rate,
            "episodes": len(self.episode_results),
            "level_history": self.level_history
        }


class MetaLearningOptimizer:
    """Optimizes hyperparameters using meta-learning and feedback."""
    
    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],
        feedback_collector: FeedbackCollector,
        exploration_factor: float = 0.2,
        learning_rate: float = 0.05,
        update_interval: int = 10
    ):
        """Initialize the meta-learning optimizer.
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) ranges
            feedback_collector: Feedback collector to use
            exploration_factor: How much to explore parameter space
            learning_rate: How quickly to adjust parameters
            update_interval: How often to update parameters
        """
        self.param_space = param_space
        self.feedback_collector = feedback_collector
        self.exploration_factor = exploration_factor
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        
        # Current parameter values (initialize to middle of range)
        self.current_params = {
            name: (min_val + max_val) / 2
            for name, (min_val, max_val) in param_space.items()
        }
        
        # History of parameters and performance
        self.param_history = []
        self.update_counter = 0
    
    async def get_parameters(self) -> Dict[str, float]:
        """Get current parameter values with exploration.
        
        Returns:
            Dictionary of parameter values
        """
        self.update_counter += 1
        
        # Update parameters if it's time
        if self.update_counter >= self.update_interval:
            await self._update_parameters()
            self.update_counter = 0
        
        # Add exploration noise
        params = {}
        for name, value in self.current_params.items():
            min_val, max_val = self.param_space[name]
            range_val = max_val - min_val
            noise = np.random.normal(0, self.exploration_factor * range_val)
            params[name] = np.clip(value + noise, min_val, max_val)
        
        return params
    
    async def _update_parameters(self) -> None:
        """Update parameters based on feedback."""
        stats = self.feedback_collector.get_feedback_statistics()
        if stats["count"] < self.feedback_collector.min_feedback_count:
            logger.debug("Not enough feedback to update meta-learning parameters")
            return
            
        # Get recent performance trend
        trend = stats.get("recent_trend", 0)
        
        # If performance is improving, reduce exploration
        if trend > 0:
            self.exploration_factor = max(0.05, self.exploration_factor * 0.9)
        else:
            # If performance is declining, increase exploration
            self.exploration_factor = min(0.5, self.exploration_factor * 1.1)
        
        # Get best performing parameters if available
        if self.param_history:
            sorted_history = sorted(self.param_history, key=lambda x: x["score"], reverse=True)
            best_params = sorted_history[0]["params"]
            
            # Move current parameters toward best ones
            for name, value in self.current_params.items():
                self.current_params[name] = value + self.learning_rate * (best_params[name] - value)
                
        self.param_history.append({
            "params": self.current_params.copy(),
            "score": stats["avg_score"],
            "timestamp": time.time()
        })
        
        logger.debug(f"Updated meta-learning parameters: {self.current_params}")


class FeedbackEnhancedLearning:
    """Enhances learning algorithms with feedback loops."""
    
    def __init__(
        self,
        base_algorithm: LearningAlgorithm,
        feedback_collector: Optional[FeedbackCollector] = None,
        prioritized_replay: bool = True,
        use_curriculum: bool = False,
        meta_learning: bool = True
    ):
        """Initialize the feedback-enhanced learning wrapper.
        
        Args:
            base_algorithm: Base learning algorithm to enhance
            feedback_collector: Feedback collector to use
            prioritized_replay: Whether to use prioritized experience replay
            use_curriculum: Whether to use curriculum learning
            meta_learning: Whether to use meta-learning optimization
        """
        self.base_algorithm = base_algorithm
        self.feedback_collector = feedback_collector or FeedbackCollector()
        
        # Set up advanced components
        self.prioritized_replay = None
        if prioritized_replay:
            self.prioritized_replay = PrioritizedExperienceReplay()
            
        self.curriculum = None
        if use_curriculum:
            # Default curriculum with 3 difficulty levels
            self.curriculum = CurriculumLearning([
                {"difficulty": "easy", "reward_scale": 1.0, "time_limit": 1000},
                {"difficulty": "medium", "reward_scale": 0.7, "time_limit": 750},
                {"difficulty": "hard", "reward_scale": 0.5, "time_limit": 500}
            ])
        
        self.meta_optimizer = None
        if meta_learning:
            # Define hyperparameter space based on base algorithm
            param_space = self._get_param_space(base_algorithm)
            if param_space:
                self.meta_optimizer = MetaLearningOptimizer(
                    param_space=param_space,
                    feedback_collector=self.feedback_collector
                )
        
        # Set up learning rate scheduler
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=getattr(base_algorithm, "learning_rate", 0.001)
        )
    
    async def update(self, experience: Experience, feedback: Optional[FeedbackEntry] = None) -> Dict[str, Any]:
        """Update the algorithm with a new experience and optional feedback.
        
        Args:
            experience: Experience to learn from
            feedback: Optional feedback about the experience
            
        Returns:
            Dictionary of update metrics
        """
        # Add feedback if provided
        if feedback:
            await self.feedback_collector.add_feedback(feedback)
        
        # Apply meta-learning if enabled
        if self.meta_optimizer:
            # Get optimized parameters
            params = await self.meta_optimizer.get_parameters()
            
            # Update base algorithm parameters
            for name, value in params.items():
                if hasattr(self.base_algorithm, name):
                    setattr(self.base_algorithm, name, value)
        
        # Use prioritized replay if enabled
        if self.prioritized_replay:
            # Add to prioritized buffer
            td_error = experience.info.get("td_error", 1.0)
            self.prioritized_replay.add(experience, priority=abs(td_error))
            
            # Sample batch if we have enough experiences
            if len(self.prioritized_replay) >= self.base_algorithm.batch_size:
                batch, indices, weights = self.prioritized_replay.sample(self.base_algorithm.batch_size)
                
                # Update with weighted batch
                metrics = self.base_algorithm.batch_update(batch)
                
                # Update priorities based on new TD errors
                if "td_errors" in metrics:
                    new_priorities = np.abs(metrics["td_errors"])
                    self.prioritized_replay.update_priorities(indices, new_priorities)
                
                return metrics
            else:
                # Not enough experiences yet
                return self.base_algorithm.update(experience)
        else:
            # Use standard update
            return self.base_algorithm.update(experience)
    
    async def select_action(self, state: Any) -> Any:
        """Select an action using the base algorithm with potential enhancements.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Get similar feedback
        similar_feedback = await self.feedback_collector.get_similar_feedback(state, None)
        
        # If we have relevant feedback with high scores, can use it to guide action selection
        # For now, let the base algorithm select the action
        return self.base_algorithm.select_action(state)
    
    async def adjust_learning_rate(self, metric_value: float) -> float:
        """Adjust learning rate based on performance metrics.
        
        Args:
            metric_value: Performance metric to use for adjustment
            
        Returns:
            New learning rate
        """
        new_lr = self.lr_scheduler.step(metric_value)
        
        # Update learning rate in base algorithm if it has one
        if hasattr(self.base_algorithm, "learning_rate"):
            self.base_algorithm.learning_rate = new_lr
            
            # If algorithm has an optimizer with learning rate
            if hasattr(self.base_algorithm, "optimizer") and hasattr(self.base_algorithm.optimizer, "learning_rate"):
                import tensorflow as tf
                self.base_algorithm.optimizer.learning_rate = tf.Variable(new_lr)
            
        return new_lr
    
    def _get_param_space(self, algorithm: LearningAlgorithm) -> Dict[str, Tuple[float, float]]:
        """Get parameter space for meta-optimization based on algorithm type.
        
        Args:
            algorithm: Learning algorithm
            
        Returns:
            Dictionary mapping parameter names to (min, max) ranges
        """
        # Different ranges based on algorithm type
        param_space = {}
        
        if hasattr(algorithm, "learning_rate"):
            param_space["learning_rate"] = (0.0001, 0.01)
            
        if hasattr(algorithm, "discount_factor"):
            param_space["discount_factor"] = (0.9, 0.999)
            
        if hasattr(algorithm, "exploration_rate"):
            param_space["exploration_rate"] = (0.01, 0.3)
            
        if hasattr(algorithm, "batch_size"):
            # Batch size should be discrete, but for simplicity we'll make it continuous
            # and round it when used
            param_space["batch_size"] = (16, 128)
            
        # PPO-specific parameters
        if hasattr(algorithm, "clip_ratio"):
            param_space["clip_ratio"] = (0.1, 0.3)
            
        if hasattr(algorithm, "entropy_coef"):
            param_space["entropy_coef"] = (0.001, 0.05)
            
        return param_space
    
    def save(self, path: str) -> None:
        """Save the enhanced learning algorithm.
        
        Args:
            path: Path to save to
        """
        # Save base algorithm
        self.base_algorithm.save(f"{path}_base")
        
        # Save additional components
        import pickle
        with open(f"{path}_enhanced", 'wb') as f:
            pickle.dump({
                'lr_scheduler': self.lr_scheduler,
                'feedback_stats': self.feedback_collector.get_feedback_statistics(),
                'curriculum_progress': self.curriculum.get_progress() if self.curriculum else None,
                'meta_params': self.meta_optimizer.current_params if self.meta_optimizer else None
            }, f)
    
    def load(self, path: str) -> None:
        """Load the enhanced learning algorithm.
        
        Args:
            path: Path to load from
        """
        # Load base algorithm
        self.base_algorithm.load(f"{path}_base")
        
        # Load additional components if they exist
        try:
            import pickle
            with open(f"{path}_enhanced", 'rb') as f:
                data = pickle.load(f)
                self.lr_scheduler = data.get('lr_scheduler', self.lr_scheduler)
                
                # Restore meta-learning parameters if they exist
                if self.meta_optimizer and 'meta_params' in data:
                    self.meta_optimizer.current_params = data['meta_params']
        except FileNotFoundError:
            logger.warning(f"Enhanced learning data not found at {path}_enhanced")