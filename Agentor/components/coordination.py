from typing import Dict, Any, List, Optional, Set, Union
import uuid
import asyncio
import logging
from agentor.agents import Agent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)

class AgentMessage:
    """A message passed between agents."""

    def __init__(
        self,
        sender: str,
        receiver: str,
        content: Union[str, Dict[str, Any]],
        message_type: str = "text"
    ):
        """Initialize the agent message.

        Args:
            sender: The sender agent's name
            receiver: The receiver agent's name
            content: The message content
            message_type: The message type
        """
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.timestamp = asyncio.get_event_loop().time()


class AgentGroup:
    """A group of agents that can collaborate."""

    def __init__(self, name: Optional[str] = None):
        """Initialize the agent group.

        Args:
            name: The name of the group
        """
        self.name = name or f"AgentGroup-{uuid.uuid4().hex[:8]}"
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, Set[str]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: List[AgentMessage] = []

    def add_agent(self, agent: Agent):
        """Add an agent to the group.

        Args:
            agent: The agent to add
        """
        self.agents[agent.name] = agent
        self.connections[agent.name] = set()
        logger.info(f"Added agent {agent.name} to group {self.name}")

    def connect(self, agent1: str, agent2: str, bidirectional: bool = True):
        """Connect two agents.

        Args:
            agent1: The name of the first agent
            agent2: The name of the second agent
            bidirectional: Whether the connection is bidirectional
        """
        if agent1 not in self.agents:
            raise ValueError(f"Agent {agent1} not found")
        if agent2 not in self.agents:
            raise ValueError(f"Agent {agent2} not found")

        self.connections[agent1].add(agent2)
        logger.info(f"Connected agent {agent1} -> {agent2}")

        if bidirectional:
            self.connections[agent2].add(agent1)
            logger.info(f"Connected agent {agent2} -> {agent1}")

    async def send_message(self, message: AgentMessage):
        """Send a message between agents.

        Args:
            message: The message to send
        """
        # Add to history
        self.message_history.append(message)

        # Add to queue
        await self.message_queue.put(message)

        logger.info(f"Message sent from {message.sender} to {message.receiver}")

    async def run_agent(self, agent_name: str, query: str, context: Dict[str, Any] = None):
        """Run a specific agent.

        Args:
            agent_name: The name of the agent to run
            query: The query to process
            context: Additional context for the query

        Returns:
            The agent's output
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")

        # Create a context with the message history if not provided
        if context is None:
            context = {}

        # Add relevant message history to the context
        relevant_messages = []
        for message in self.message_history:
            if message.receiver == agent_name:
                relevant_messages.append(message)

        context["messages"] = relevant_messages

        # Run the agent
        output = await self.agents[agent_name].run(query, context)

        return output

    async def run_parallel(self, query: str, context: Dict[str, Any] = None) -> Dict[str, AgentOutput]:
        """Run all agents in parallel.

        Args:
            query: The query to process
            context: Additional context for the query

        Returns:
            A dictionary of agent outputs
        """
        # Create tasks for all agents
        tasks = [
            self.run_agent(agent_name, query, context)
            for agent_name in self.agents
        ]

        # Run all tasks
        outputs = await asyncio.gather(*tasks)

        # Map outputs to agent names
        return {
            agent_name: output
            for agent_name, output in zip(self.agents.keys(), outputs)
        }

    async def run_conversation(
        self,
        query: str,
        context: Dict[str, Any] = None,
        start_agent: str = None,
        max_turns: int = 10
    ) -> List[AgentMessage]:
        """Run a multi-agent conversation.

        Args:
            query: The query to process
            context: Additional context for the query
            start_agent: The name of the agent to start with (default: first agent)
            max_turns: Maximum number of conversation turns

        Returns:
            The conversation history
        """
        if not self.agents:
            return []

        # Clear the message queue
        while not self.message_queue.empty():
            await self.message_queue.get()

        # Use the first agent if no start agent is specified
        current_agent = start_agent or next(iter(self.agents.keys()))

        # Create a context if not provided
        if context is None:
            context = {}

        # Add the query as a message from the user to the start agent
        first_message = AgentMessage(
            sender="user",
            receiver=current_agent,
            content=query
        )
        await self.send_message(first_message)

        # Run the conversation
        turn = 0
        while turn < max_turns:
            # Get all messages for the current agent
            agent_messages = []
            while not self.message_queue.empty():
                message = await self.message_queue.get()
                if message.receiver == current_agent:
                    agent_messages.append(message)
                else:
                    # Put it back for other agents
                    await self.message_queue.put(message)

            if not agent_messages:
                # No messages for the current agent, we're done
                break

            # Set up context for the agent
            agent_context = context.copy()
            agent_context["messages"] = agent_messages

            # Combine all messages into a single query
            combined_query = "\n".join([
                f"Message from {message.sender}: {message.content}"
                for message in agent_messages
            ])

            # Run the current agent
            output = await self.agents[current_agent].run(combined_query, agent_context)

            # Process the agent's response - send messages to connected agents
            for connected_agent in self.connections[current_agent]:
                response_message = AgentMessage(
                    sender=current_agent,
                    receiver=connected_agent,
                    content=output.response
                )
                await self.send_message(response_message)

            # Move to the next agent with messages
            current_agent = None
            for agent_name in self.agents:
                # Check if there are messages for this agent
                for message in self.message_queue._queue:
                    if message.receiver == agent_name:
                        current_agent = agent_name
                        break
                if current_agent:
                    break

            if not current_agent:
                # No more messages, we're done
                break

            turn += 1

        # Return the message history
        return self.message_history