"""
Coordination patterns for multi-agent systems.

This module provides various coordination patterns for multi-agent systems,
including master-slave, peer-to-peer, blackboard, contract net protocol,
and market-based coordination.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from abc import ABC, abstractmethod

from agentor.core.interfaces.agent import IAgent, AgentInput, AgentOutput
from agentor.components.coordination.base import AgentMessage

logger = logging.getLogger(__name__)


class CoordinationPattern(ABC):
    """Base class for coordination patterns."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the coordination pattern.
        
        Args:
            name: The name of the pattern
        """
        self.name = name or f"{self.__class__.__name__}-{uuid.uuid4().hex[:8]}"
        self.agents: Dict[str, IAgent] = {}
    
    def add_agent(self, agent: IAgent) -> None:
        """Add an agent to the pattern.
        
        Args:
            agent: The agent to add
        """
        self.agents[agent.name] = agent
        logger.info(f"Added agent {agent.name} to pattern {self.name}")
    
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the pattern.
        
        Args:
            agent_name: The name of the agent to remove
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Removed agent {agent_name} from pattern {self.name}")
    
    @abstractmethod
    async def coordinate(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate the agents to solve a problem.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        pass


class MasterSlavePattern(CoordinationPattern):
    """Master-slave coordination pattern.
    
    In this pattern, one agent (the master) delegates tasks to other agents (slaves)
    and aggregates their results.
    """
    
    def __init__(self, name: Optional[str] = None, master_agent: Optional[IAgent] = None):
        """Initialize the master-slave pattern.
        
        Args:
            name: The name of the pattern
            master_agent: The master agent
        """
        super().__init__(name=name or "MasterSlavePattern")
        self.master_agent = master_agent
        if master_agent:
            self.add_agent(master_agent)
        self.slave_agents: Dict[str, IAgent] = {}
    
    def add_slave_agent(self, agent: IAgent) -> None:
        """Add a slave agent to the pattern.
        
        Args:
            agent: The agent to add
        """
        self.slave_agents[agent.name] = agent
        self.add_agent(agent)
        logger.info(f"Added slave agent {agent.name} to pattern {self.name}")
    
    def set_master_agent(self, agent: IAgent) -> None:
        """Set the master agent.
        
        Args:
            agent: The agent to set as master
        """
        self.master_agent = agent
        self.add_agent(agent)
        logger.info(f"Set {agent.name} as master agent in pattern {self.name}")
    
    async def coordinate(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate the agents to solve a problem.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        if not self.master_agent:
            raise ValueError("Master agent not set")
        
        if not self.slave_agents:
            raise ValueError("No slave agents added")
        
        # Create a context if not provided
        if context is None:
            context = {}
        
        # Run the master agent to delegate tasks
        master_output = await self.master_agent.run(
            AgentInput(query=f"Delegate tasks for: {query}", context=context)
        )
        
        # Parse the master's output to get tasks for slave agents
        tasks = self._parse_master_output(master_output.response)
        
        # Run slave agents in parallel
        slave_results = {}
        for agent_name, task in tasks.items():
            if agent_name in self.slave_agents:
                slave_agent = self.slave_agents[agent_name]
                slave_output = await slave_agent.run(
                    AgentInput(query=task, context=context)
                )
                slave_results[agent_name] = slave_output.response
        
        # Run the master agent again to aggregate results
        aggregation_query = f"Aggregate results for: {query}\n\nResults:\n"
        for agent_name, result in slave_results.items():
            aggregation_query += f"\n{agent_name}: {result}"
        
        final_output = await self.master_agent.run(
            AgentInput(query=aggregation_query, context=context)
        )
        
        return final_output.response
    
    def _parse_master_output(self, output: str) -> Dict[str, str]:
        """Parse the master agent's output to get tasks for slave agents.
        
        Args:
            output: The master agent's output
            
        Returns:
            Dictionary mapping agent names to tasks
        """
        # Simple parsing: assume the master outputs tasks in the format "AgentName: Task"
        tasks = {}
        for line in output.strip().split("\n"):
            if ":" in line:
                agent_name, task = line.split(":", 1)
                agent_name = agent_name.strip()
                task = task.strip()
                if agent_name in self.slave_agents:
                    tasks[agent_name] = task
        
        return tasks


class PeerToPeerPattern(CoordinationPattern):
    """Peer-to-peer coordination pattern.
    
    In this pattern, agents communicate directly with each other without
    a central coordinator.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the peer-to-peer pattern.
        
        Args:
            name: The name of the pattern
        """
        super().__init__(name=name or "PeerToPeerPattern")
        self.connections: Dict[str, Set[str]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: List[AgentMessage] = []
    
    def add_agent(self, agent: IAgent) -> None:
        """Add an agent to the pattern.
        
        Args:
            agent: The agent to add
        """
        super().add_agent(agent)
        self.connections[agent.name] = set()
    
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the pattern.
        
        Args:
            agent_name: The name of the agent to remove
        """
        super().remove_agent(agent_name)
        if agent_name in self.connections:
            del self.connections[agent_name]
        
        # Remove connections to this agent
        for connections in self.connections.values():
            if agent_name in connections:
                connections.remove(agent_name)
    
    def connect_agents(self, agent1_name: str, agent2_name: str) -> None:
        """Connect two agents.
        
        Args:
            agent1_name: The name of the first agent
            agent2_name: The name of the second agent
        """
        if agent1_name not in self.agents:
            raise ValueError(f"Agent {agent1_name} not found")
        
        if agent2_name not in self.agents:
            raise ValueError(f"Agent {agent2_name} not found")
        
        self.connections[agent1_name].add(agent2_name)
        self.connections[agent2_name].add(agent1_name)
        
        logger.info(f"Connected agents {agent1_name} and {agent2_name} in pattern {self.name}")
    
    def disconnect_agents(self, agent1_name: str, agent2_name: str) -> None:
        """Disconnect two agents.
        
        Args:
            agent1_name: The name of the first agent
            agent2_name: The name of the second agent
        """
        if agent1_name in self.connections and agent2_name in self.connections[agent1_name]:
            self.connections[agent1_name].remove(agent2_name)
        
        if agent2_name in self.connections and agent1_name in self.connections[agent2_name]:
            self.connections[agent2_name].remove(agent1_name)
        
        logger.info(f"Disconnected agents {agent1_name} and {agent2_name} in pattern {self.name}")
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message between agents.
        
        Args:
            message: The message to send
        """
        # Add to history
        self.message_history.append(message)
        
        # Add to queue
        await self.message_queue.put(message)
        
        logger.info(f"Message sent from {message.sender} to {message.receiver}")
    
    async def coordinate(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate the agents to solve a problem.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        if not self.agents:
            raise ValueError("No agents added")
        
        # Clear the message queue
        while not self.message_queue.empty():
            await self.message_queue.get()
        
        # Create a context if not provided
        if context is None:
            context = {}
        
        # Start with a random agent
        current_agent_name = next(iter(self.agents.keys()))
        
        # Run the conversation for a maximum number of turns
        max_turns = 10
        turn = 0
        
        while turn < max_turns:
            # Get the current agent
            current_agent = self.agents[current_agent_name]
            
            # Get messages for the current agent
            agent_messages = []
            for message in self.message_history:
                if message.receiver == current_agent_name:
                    agent_messages.append(message)
            
            # Create a query with the messages
            combined_query = query
            if agent_messages:
                combined_query += "\n\nMessages:\n"
                for message in agent_messages:
                    combined_query += f"\nFrom {message.sender}: {message.content}"
            
            # Run the current agent
            agent_context = context.copy()
            agent_context["messages"] = agent_messages
            
            output = await current_agent.run(
                AgentInput(query=combined_query, context=agent_context)
            )
            
            # Process the agent's response - send messages to connected agents
            for connected_agent in self.connections[current_agent_name]:
                response_message = AgentMessage(
                    sender=current_agent_name,
                    receiver=connected_agent,
                    content=output.response
                )
                await self.send_message(response_message)
            
            # Move to the next agent with messages
            current_agent_name = None
            for agent_name in self.agents:
                # Check if there are messages for this agent
                for message in self.message_queue._queue:
                    if message.receiver == agent_name:
                        current_agent_name = agent_name
                        break
                if current_agent_name:
                    break
            
            if not current_agent_name:
                # No more messages, we're done
                break
            
            turn += 1
        
        # Return the message history
        return self.message_history


class BlackboardPattern(CoordinationPattern):
    """Blackboard coordination pattern.
    
    In this pattern, agents share a common knowledge base (the blackboard)
    and contribute to solving a problem by updating the blackboard.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the blackboard pattern.
        
        Args:
            name: The name of the pattern
        """
        super().__init__(name=name or "BlackboardPattern")
        self.blackboard: Dict[str, Any] = {}
        self.controller: Optional[IAgent] = None
    
    def set_controller(self, agent: IAgent) -> None:
        """Set the controller agent.
        
        Args:
            agent: The agent to set as controller
        """
        self.controller = agent
        self.add_agent(agent)
        logger.info(f"Set {agent.name} as controller in pattern {self.name}")
    
    def update_blackboard(self, key: str, value: Any) -> None:
        """Update the blackboard.
        
        Args:
            key: The key to update
            value: The value to set
        """
        self.blackboard[key] = value
        logger.info(f"Updated blackboard key {key} in pattern {self.name}")
    
    async def coordinate(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate the agents to solve a problem.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        if not self.agents:
            raise ValueError("No agents added")
        
        if not self.controller:
            raise ValueError("Controller not set")
        
        # Create a context if not provided
        if context is None:
            context = {}
        
        # Clear the blackboard
        self.blackboard.clear()
        
        # Initialize the blackboard with the query
        self.update_blackboard("query", query)
        
        # Maximum number of iterations
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # Run the controller to select the next agent
            controller_query = f"Select the next agent to run based on the current blackboard state:\n\n{self.blackboard_to_string()}"
            controller_output = await self.controller.run(
                AgentInput(query=controller_query, context=context)
            )
            
            # Parse the controller's output to get the next agent
            next_agent_name = self._parse_controller_output(controller_output.response)
            
            if not next_agent_name or next_agent_name not in self.agents:
                # No more agents to run or invalid agent, we're done
                break
            
            # Run the selected agent
            agent = self.agents[next_agent_name]
            agent_query = f"Update the blackboard based on the current state:\n\n{self.blackboard_to_string()}"
            agent_output = await agent.run(
                AgentInput(query=agent_query, context=context)
            )
            
            # Parse the agent's output to update the blackboard
            updates = self._parse_agent_output(agent_output.response)
            for key, value in updates.items():
                self.update_blackboard(key, value)
            
            # Check if the problem is solved
            controller_query = f"Is the problem solved based on the current blackboard state?\n\n{self.blackboard_to_string()}"
            controller_output = await self.controller.run(
                AgentInput(query=controller_query, context=context)
            )
            
            if "yes" in controller_output.response.lower():
                # Problem solved, we're done
                break
            
            iteration += 1
        
        # Return the final blackboard state
        return self.blackboard
    
    def blackboard_to_string(self) -> str:
        """Convert the blackboard to a string.
        
        Returns:
            String representation of the blackboard
        """
        result = "Blackboard:\n"
        for key, value in self.blackboard.items():
            result += f"{key}: {value}\n"
        return result
    
    def _parse_controller_output(self, output: str) -> Optional[str]:
        """Parse the controller's output to get the next agent.
        
        Args:
            output: The controller's output
            
        Returns:
            The name of the next agent, or None if no agent is selected
        """
        # Simple parsing: assume the controller outputs the agent name
        for agent_name in self.agents:
            if agent_name in output:
                return agent_name
        return None
    
    def _parse_agent_output(self, output: str) -> Dict[str, Any]:
        """Parse the agent's output to get blackboard updates.
        
        Args:
            output: The agent's output
            
        Returns:
            Dictionary of blackboard updates
        """
        # Simple parsing: assume the agent outputs updates in the format "Key: Value"
        updates = {}
        for line in output.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                updates[key] = value
        return updates


class ContractNetProtocol(CoordinationPattern):
    """Contract Net Protocol coordination pattern.
    
    In this pattern, agents bid on tasks and the manager selects the best bid.
    """
    
    def __init__(self, name: Optional[str] = None, manager_agent: Optional[IAgent] = None):
        """Initialize the Contract Net Protocol pattern.
        
        Args:
            name: The name of the pattern
            manager_agent: The manager agent
        """
        super().__init__(name=name or "ContractNetProtocol")
        self.manager_agent = manager_agent
        if manager_agent:
            self.add_agent(manager_agent)
        self.contractor_agents: Dict[str, IAgent] = {}
    
    def add_contractor_agent(self, agent: IAgent) -> None:
        """Add a contractor agent to the pattern.
        
        Args:
            agent: The agent to add
        """
        self.contractor_agents[agent.name] = agent
        self.add_agent(agent)
        logger.info(f"Added contractor agent {agent.name} to pattern {self.name}")
    
    def set_manager_agent(self, agent: IAgent) -> None:
        """Set the manager agent.
        
        Args:
            agent: The agent to set as manager
        """
        self.manager_agent = agent
        self.add_agent(agent)
        logger.info(f"Set {agent.name} as manager agent in pattern {self.name}")
    
    async def coordinate(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate the agents to solve a problem.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        if not self.manager_agent:
            raise ValueError("Manager agent not set")
        
        if not self.contractor_agents:
            raise ValueError("No contractor agents added")
        
        # Create a context if not provided
        if context is None:
            context = {}
        
        # Step 1: Task announcement
        announcement_query = f"Announce task: {query}"
        announcement_output = await self.manager_agent.run(
            AgentInput(query=announcement_query, context=context)
        )
        
        # Step 2: Bid submission
        bids = {}
        for agent_name, agent in self.contractor_agents.items():
            bid_query = f"Submit bid for task: {query}\n\nTask announcement: {announcement_output.response}"
            bid_output = await agent.run(
                AgentInput(query=bid_query, context=context)
            )
            bids[agent_name] = bid_output.response
        
        # Step 3: Bid evaluation
        evaluation_query = f"Evaluate bids for task: {query}\n\nBids:\n"
        for agent_name, bid in bids.items():
            evaluation_query += f"\n{agent_name}: {bid}"
        
        evaluation_output = await self.manager_agent.run(
            AgentInput(query=evaluation_query, context=context)
        )
        
        # Step 4: Task allocation
        winner = self._parse_evaluation_output(evaluation_output.response)
        
        if not winner or winner not in self.contractor_agents:
            raise ValueError(f"Invalid winner: {winner}")
        
        # Step 5: Task execution
        execution_query = f"Execute task: {query}\n\nYou have been selected to execute this task."
        execution_output = await self.contractor_agents[winner].run(
            AgentInput(query=execution_query, context=context)
        )
        
        # Step 6: Result reporting
        result_query = f"Report result for task: {query}\n\nExecution result: {execution_output.response}"
        result_output = await self.manager_agent.run(
            AgentInput(query=result_query, context=context)
        )
        
        return result_output.response
    
    def _parse_evaluation_output(self, output: str) -> Optional[str]:
        """Parse the evaluation output to get the winner.
        
        Args:
            output: The evaluation output
            
        Returns:
            The name of the winning agent, or None if no winner is selected
        """
        # Simple parsing: assume the output contains "Winner: AgentName"
        for line in output.strip().split("\n"):
            if "winner" in line.lower() and ":" in line:
                _, winner = line.split(":", 1)
                winner = winner.strip()
                if winner in self.contractor_agents:
                    return winner
        
        # If no winner is found, try to find any agent name in the output
        for agent_name in self.contractor_agents:
            if agent_name in output:
                return agent_name
        
        return None


class MarketBasedCoordination(CoordinationPattern):
    """Market-based coordination pattern.
    
    In this pattern, agents buy and sell services in a market.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the market-based coordination pattern.
        
        Args:
            name: The name of the pattern
        """
        super().__init__(name=name or "MarketBasedCoordination")
        self.services: Dict[str, Dict[str, float]] = {}  # agent_name -> {service_name: price}
        self.market_agent: Optional[IAgent] = None
    
    def set_market_agent(self, agent: IAgent) -> None:
        """Set the market agent.
        
        Args:
            agent: The agent to set as market agent
        """
        self.market_agent = agent
        self.add_agent(agent)
        logger.info(f"Set {agent.name} as market agent in pattern {self.name}")
    
    def register_service(self, agent_name: str, service_name: str, price: float) -> None:
        """Register a service.
        
        Args:
            agent_name: The name of the agent providing the service
            service_name: The name of the service
            price: The price of the service
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        if agent_name not in self.services:
            self.services[agent_name] = {}
        
        self.services[agent_name][service_name] = price
        logger.info(f"Registered service {service_name} by {agent_name} at price {price}")
    
    def get_service_providers(self, service_name: str) -> Dict[str, float]:
        """Get all providers of a service.
        
        Args:
            service_name: The name of the service
            
        Returns:
            Dictionary mapping agent names to prices
        """
        providers = {}
        for agent_name, services in self.services.items():
            if service_name in services:
                providers[agent_name] = services[service_name]
        return providers
    
    async def coordinate(self, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate the agents to solve a problem.
        
        Args:
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        if not self.market_agent:
            raise ValueError("Market agent not set")
        
        if not self.services:
            raise ValueError("No services registered")
        
        # Create a context if not provided
        if context is None:
            context = {}
        
        # Step 1: Identify required services
        service_query = f"Identify services required for: {query}\n\nAvailable services:\n"
        for agent_name, services in self.services.items():
            for service_name, price in services.items():
                service_query += f"\n{agent_name} - {service_name}: {price}"
        
        service_output = await self.market_agent.run(
            AgentInput(query=service_query, context=context)
        )
        
        # Step 2: Select service providers
        required_services = self._parse_service_output(service_output.response)
        
        # Step 3: Execute services
        results = {}
        for service_name, _ in required_services:
            providers = self.get_service_providers(service_name)
            if not providers:
                continue
            
            # Select the cheapest provider
            cheapest_provider = min(providers.items(), key=lambda x: x[1])[0]
            
            # Execute the service
            execution_query = f"Execute service {service_name} for: {query}"
            execution_output = await self.agents[cheapest_provider].run(
                AgentInput(query=execution_query, context=context)
            )
            
            results[service_name] = execution_output.response
        
        # Step 4: Aggregate results
        aggregation_query = f"Aggregate results for: {query}\n\nResults:\n"
        for service_name, result in results.items():
            aggregation_query += f"\n{service_name}: {result}"
        
        aggregation_output = await self.market_agent.run(
            AgentInput(query=aggregation_query, context=context)
        )
        
        return aggregation_output.response
    
    def _parse_service_output(self, output: str) -> List[Tuple[str, float]]:
        """Parse the service output to get required services.
        
        Args:
            output: The service output
            
        Returns:
            List of (service_name, budget) tuples
        """
        # Simple parsing: assume the output contains "Service: Budget"
        services = []
        for line in output.strip().split("\n"):
            if ":" in line:
                service_name, budget_str = line.split(":", 1)
                service_name = service_name.strip()
                try:
                    budget = float(budget_str.strip())
                    services.append((service_name, budget))
                except ValueError:
                    pass
        return services
