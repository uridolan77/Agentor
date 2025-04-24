"""
Example demonstrating the enhanced environment extensions in Agentor.

This example shows how to use the new environment extensions:
- Streaming environments for real-time data
- Hierarchical environments for complex scenarios
- Environment wrappers for common transformations
- Enhanced multi-agent environments
- Visualization tools for environment states
"""

import asyncio
import logging
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from agentor.components.environments import (
    # Base environment components
    BaseEnvironment, TimeLimit, Monitor,
    
    # Standard environments
    GridWorldEnv, CartPoleEnv,
    
    # Streaming environment
    StreamingEnvironment, DataStream, MarketDataStream, NewsStream,
    
    # Hierarchical environments
    HierarchicalEnvironment, SubEnvironment, TaskHierarchicalEnvironment,
    
    # Environment wrappers
    EnvironmentWrapper, NormalizeObservation, ClipReward, FrameStack,
    VideoRecorder, ActionRepeat,
    
    # Enhanced multi-agent environments
    CommunicativeMultiAgentEnv, TeamBasedMultiAgentEnv,
    DynamicMultiAgentEnv, CompetitiveMultiAgentEnv,
    Message, CommunicationChannel,
    
    # Visualization tools
    EnvironmentRenderer, MatplotlibRenderer, DashboardRenderer, EnvironmentMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgent:
    """A simple agent for testing environments."""
    
    def __init__(self, name: str = "SimpleAgent"):
        """Initialize the simple agent.
        
        Args:
            name: The name of the agent
        """
        self.name = name
        self.total_reward = 0.0
        self.episode_count = 0
    
    async def act(self, observation: Any, reward: float = 0.0, done: bool = False) -> Any:
        """Choose an action based on the observation.
        
        Args:
            observation: The current observation
            reward: The reward from the last action
            done: Whether the episode is done
            
        Returns:
            The action to take
        """
        # Update statistics
        self.total_reward += reward
        
        if done:
            logger.info(f"Episode {self.episode_count} finished with total reward: {self.total_reward}")
            self.episode_count += 1
            self.total_reward = 0.0
        
        # Choose a random action
        if isinstance(observation, dict) and "action_space" in observation:
            # For hierarchical environments
            return observation["action_space"].sample()
        else:
            # For standard environments
            return random.randint(0, 3)  # Assuming 4 actions (up, right, down, left)


async def run_streaming_environment_example():
    """Example of using streaming environments."""
    logger.info("\n=== Streaming Environment Example ===")
    
    # Create a streaming environment
    env = StreamingEnvironment()
    
    # Create data streams
    market_stream = MarketDataStream(
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
        update_interval=1.0,
        volatility=0.02
    )
    
    news_stream = NewsStream(
        topics=["technology", "finance", "health"],
        update_interval=2.0
    )
    
    # Register the streams
    env.register_stream("market_data", market_stream)
    env.register_stream("news_data", news_stream)
    
    # Start all streams
    await env.start_all_streams()
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Run for a few steps
    observation, info = env.reset()
    
    for i in range(10):
        logger.info(f"Step {i+1}")
        
        # Get the current observation
        logger.info(f"Observation: {observation}")
        
        # Choose an action
        action = await agent.act(observation)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Wait a bit
        await asyncio.sleep(1.0)
    
    # Stop all streams
    await env.stop_all_streams()
    
    # Close the environment
    env.close()


async def run_hierarchical_environment_example():
    """Example of using hierarchical environments."""
    logger.info("\n=== Hierarchical Environment Example ===")
    
    # Create sub-environments
    grid_world = GridWorldEnv(width=5, height=5)
    cart_pole = CartPoleEnv()
    
    # Create sub-environment wrappers
    grid_sub = SubEnvironment(
        env=grid_world,
        name="grid_world",
        reward_scale=1.0
    )
    
    cart_sub = SubEnvironment(
        env=cart_pole,
        name="cart_pole",
        reward_scale=2.0
    )
    
    # Create a hierarchical environment
    env = HierarchicalEnvironment(max_episode_steps=100)
    
    # Add sub-environments
    env.add_sub_environment(grid_sub)
    env.add_sub_environment(cart_sub)
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Run for a few episodes
    for episode in range(3):
        logger.info(f"Episode {episode+1}")
        
        observation, info = env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            # Choose actions for each sub-environment
            actions = {}
            for name, obs in observation.items():
                actions[name] = await agent.act(obs)
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(actions)
            
            total_reward += reward
            done = terminated or truncated
        
        logger.info(f"Episode {episode+1} finished with total reward: {total_reward}")
    
    # Close the environment
    env.close()


async def run_task_hierarchical_environment_example():
    """Example of using task hierarchical environments."""
    logger.info("\n=== Task Hierarchical Environment Example ===")
    
    # Create sub-environments for different tasks
    task1 = SubEnvironment(
        env=GridWorldEnv(width=3, height=3),
        name="task1",
        reward_scale=1.0
    )
    
    task2 = SubEnvironment(
        env=GridWorldEnv(width=4, height=4),
        name="task2",
        reward_scale=2.0
    )
    
    task3 = SubEnvironment(
        env=GridWorldEnv(width=5, height=5),
        name="task3",
        reward_scale=3.0
    )
    
    # Create a task hierarchical environment
    env = TaskHierarchicalEnvironment(max_episode_steps=100)
    
    # Add tasks with dependencies
    env.add_task(task1, dependencies=[])  # No dependencies
    env.add_task(task2, dependencies=["task1"])  # Depends on task1
    env.add_task(task3, dependencies=["task2"])  # Depends on task2
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Run for a few episodes
    for episode in range(3):
        logger.info(f"Episode {episode+1}")
        
        observation, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < 100:
            step += 1
            
            # Choose actions for active tasks
            actions = {}
            for name, obs in observation.items():
                actions[name] = await agent.act(obs)
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(actions)
            
            total_reward += reward
            done = terminated or truncated
            
            # Log active and completed tasks
            active_tasks = info.get("active_tasks", [])
            completed_tasks = info.get("completed_tasks", [])
            
            if step % 10 == 0:
                logger.info(f"Step {step} - Active tasks: {active_tasks}, Completed tasks: {completed_tasks}")
        
        logger.info(f"Episode {episode+1} finished with total reward: {total_reward}")
        logger.info(f"Completed tasks: {info.get('completed_tasks', [])}")
    
    # Close the environment
    env.close()


async def run_environment_wrappers_example():
    """Example of using environment wrappers."""
    logger.info("\n=== Environment Wrappers Example ===")
    
    # Create a base environment
    base_env = CartPoleEnv()
    
    # Apply wrappers
    env = base_env
    env = NormalizeObservation(env)
    env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
    env = ActionRepeat(env, repeat=2)
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Run for a few episodes
    for episode in range(3):
        logger.info(f"Episode {episode+1}")
        
        observation, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < 100:
            step += 1
            
            # Choose an action
            action = await agent.act(observation, reward=0.0 if step == 1 else reward, done=done)
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
            
            if step % 20 == 0:
                logger.info(f"Step {step} - Reward: {reward}, Total reward: {total_reward}")
        
        logger.info(f"Episode {episode+1} finished with total reward: {total_reward}")
    
    # Close the environment
    env.close()


async def run_video_recorder_example():
    """Example of using the video recorder wrapper."""
    logger.info("\n=== Video Recorder Example ===")
    
    # Create a base environment
    base_env = CartPoleEnv()
    
    # Apply the video recorder wrapper
    env = VideoRecorder(
        env=base_env,
        video_dir="videos",
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix="cart_pole"
    )
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Run for a few episodes
    for episode in range(2):
        logger.info(f"Episode {episode+1}")
        
        observation, info = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < 100:
            step += 1
            
            # Choose an action
            action = await agent.act(observation, reward=0.0 if step == 1 else reward, done=done)
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            done = terminated or truncated
        
        logger.info(f"Episode {episode+1} finished with total reward: {total_reward}")
    
    # Close the environment
    env.close()


async def run_multi_agent_environment_example():
    """Example of using enhanced multi-agent environments."""
    logger.info("\n=== Enhanced Multi-Agent Environment Example ===")
    
    # Create a team-based multi-agent environment
    env = TeamBasedMultiAgentEnv(
        agent_ids=["agent1", "agent2", "agent3", "agent4"],
        team_memberships={
            "agent1": "team1",
            "agent2": "team1",
            "agent3": "team2",
            "agent4": "team2"
        },
        observation_spaces={
            "agent1": BoxSpace(low=0, high=1, shape=(4,)),
            "agent2": BoxSpace(low=0, high=1, shape=(4,)),
            "agent3": BoxSpace(low=0, high=1, shape=(4,)),
            "agent4": BoxSpace(low=0, high=1, shape=(4,))
        },
        action_spaces={
            "agent1": DiscreteSpace(2),
            "agent2": DiscreteSpace(2),
            "agent3": DiscreteSpace(2),
            "agent4": DiscreteSpace(2)
        },
        max_episode_steps=100,
        shared_team_reward=True,
        team_reward_scale=1.0,
        individual_reward_scale=0.5
    )
    
    # Create agents
    agents = {
        "agent1": SimpleAgent(name="agent1"),
        "agent2": SimpleAgent(name="agent2"),
        "agent3": SimpleAgent(name="agent3"),
        "agent4": SimpleAgent(name="agent4")
    }
    
    # Run for a few episodes
    for episode in range(3):
        logger.info(f"Episode {episode+1}")
        
        observations, info = env.reset()
        total_rewards = {agent_id: 0.0 for agent_id in env._agent_ids}
        done = False
        step = 0
        
        while not done and step < 100:
            step += 1
            
            # Choose actions for each agent
            actions = {}
            for agent_id, observation in observations.items():
                actions[agent_id] = await agents[agent_id].act(observation)
            
            # Take a step in the environment
            observations, rewards, terminated, truncated, info = env.step(actions)
            
            # Update total rewards
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
            
            # Check if done
            all_terminated = all(terminated.values())
            any_truncated = any(truncated.values())
            done = all_terminated or any_truncated
            
            if step % 20 == 0:
                logger.info(f"Step {step} - Rewards: {rewards}")
        
        logger.info(f"Episode {episode+1} finished with total rewards: {total_rewards}")
        
        # Log team information
        team_info = info.get("teams", {})
        logger.info(f"Team memberships: {team_info.get('memberships', {})}")
        logger.info(f"Team rewards: {team_info.get('team_rewards', {})}")
    
    # Close the environment
    env.close()


async def run_communicative_multi_agent_example():
    """Example of using communicative multi-agent environments."""
    logger.info("\n=== Communicative Multi-Agent Environment Example ===")
    
    # Create a communicative multi-agent environment
    env = CommunicativeMultiAgentEnv(
        agent_ids=["agent1", "agent2", "agent3"],
        observation_spaces={
            "agent1": BoxSpace(low=0, high=1, shape=(4,)),
            "agent2": BoxSpace(low=0, high=1, shape=(4,)),
            "agent3": BoxSpace(low=0, high=1, shape=(4,))
        },
        action_spaces={
            "agent1": DiscreteSpace(2),
            "agent2": DiscreteSpace(2),
            "agent3": DiscreteSpace(2)
        },
        max_episode_steps=100,
        communication_enabled=True
    )
    
    # Add a private channel
    private_channel = CommunicationChannel(
        name="private",
        message_limit=10,
        delivery_delay=0.5,
        delivery_probability=0.9
    )
    
    env.add_channel(private_channel)
    
    # Create agents
    agents = {
        "agent1": SimpleAgent(name="agent1"),
        "agent2": SimpleAgent(name="agent2"),
        "agent3": SimpleAgent(name="agent3")
    }
    
    # Run for a few steps
    observations, info = env.reset()
    
    for step in range(10):
        logger.info(f"Step {step+1}")
        
        # Choose actions for each agent
        actions = {}
        for agent_id, observation in observations.items():
            # Basic action
            action = await agents[agent_id].act(observation)
            
            # Add communication
            if step % 3 == 0:
                # Send a message
                actions[agent_id] = {
                    "action": action,
                    "message": {
                        "content": f"Hello from {agent_id} at step {step}",
                        "recipients": None,  # Broadcast
                        "channel": "global"
                    }
                }
            elif step % 3 == 1:
                # Send a private message
                recipient = random.choice([a for a in env._agent_ids if a != agent_id])
                actions[agent_id] = {
                    "action": action,
                    "message": {
                        "content": f"Private message from {agent_id} to {recipient}",
                        "recipients": [recipient],
                        "channel": "private"
                    }
                }
            else:
                actions[agent_id] = action
        
        # Take a step in the environment
        observations, rewards, terminated, truncated, info = env.step(actions)
        
        # Log messages
        for agent_id, observation in observations.items():
            if isinstance(observation, dict) and "messages" in observation:
                messages = observation["messages"]
                if messages:
                    logger.info(f"Messages for {agent_id}: {len(messages)}")
                    for msg in messages:
                        logger.info(f"  From: {msg.sender}, Content: {msg.content}")
        
        # Wait a bit
        await asyncio.sleep(0.5)
    
    # Close the environment
    env.close()


async def run_visualization_example():
    """Example of using environment visualization tools."""
    logger.info("\n=== Environment Visualization Example ===")
    
    # Create an environment
    env = GridWorldEnv(width=10, height=10)
    
    # Create a matplotlib renderer
    renderer = MatplotlibRenderer(
        env=env,
        figsize=(8, 8),
        title="Grid World Environment",
        update_interval=0.5,
        block=False
    )
    
    # Create an environment monitor
    monitor = EnvironmentMonitor(
        env=env,
        log_interval=1,
        window_size=10
    )
    
    # Create a dashboard
    monitor.create_dashboard()
    
    # Create a simple agent
    agent = SimpleAgent()
    
    # Run for a few episodes
    for episode in range(3):
        logger.info(f"Episode {episode+1}")
        
        observation, info = env.reset()
        monitor.start_episode()
        
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < 100:
            step += 1
            
            # Render the environment
            renderer.render()
            
            # Choose an action
            action = await agent.act(observation, reward=0.0 if step == 1 else reward, done=done)
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Record the step
            monitor.record_step(reward)
            
            total_reward += reward
            done = terminated or truncated
            
            # Update the dashboard
            monitor.update_dashboard()
            
            # Wait a bit
            await asyncio.sleep(0.1)
        
        # End the episode
        episode_stats = monitor.end_episode()
        logger.info(f"Episode {episode+1} stats: {episode_stats}")
    
    # Get overall metrics
    metrics = monitor.get_metrics()
    logger.info(f"Overall metrics: {metrics}")
    
    # Close the renderer and monitor
    renderer.close()
    monitor.close()
    
    # Close the environment
    env.close()


async def main():
    """Run all examples."""
    # Run the streaming environment example
    await run_streaming_environment_example()
    
    # Run the hierarchical environment example
    await run_hierarchical_environment_example()
    
    # Run the task hierarchical environment example
    await run_task_hierarchical_environment_example()
    
    # Run the environment wrappers example
    await run_environment_wrappers_example()
    
    # Run the video recorder example
    await run_video_recorder_example()
    
    # Run the multi-agent environment example
    await run_multi_agent_environment_example()
    
    # Run the communicative multi-agent example
    await run_communicative_multi_agent_example()
    
    # Run the visualization example
    await run_visualization_example()


if __name__ == "__main__":
    asyncio.run(main())
