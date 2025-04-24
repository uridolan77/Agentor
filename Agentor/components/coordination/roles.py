"""
Role-based coordination for multi-agent systems.

This module provides mechanisms for role-based coordination in multi-agent systems,
including role definition, assignment, transition, and team formation.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from abc import ABC, abstractmethod

from agentor.core.interfaces.agent import IAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class Role:
    """Definition of an agent role."""
    
    def __init__(self, name: str, description: str, responsibilities: List[str], requirements: Optional[Dict[str, Any]] = None):
        """Initialize the role.
        
        Args:
            name: The name of the role
            description: The description of the role
            responsibilities: List of responsibilities for this role
            requirements: Optional requirements for agents to fulfill this role
        """
        self.name = name
        self.description = description
        self.responsibilities = responsibilities
        self.requirements = requirements or {}
    
    def __str__(self) -> str:
        """Get a string representation of the role.
        
        Returns:
            String representation
        """
        return f"{self.name}: {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the role to a dictionary.
        
        Returns:
            Dictionary representation of the role
        """
        return {
            "name": self.name,
            "description": self.description,
            "responsibilities": self.responsibilities,
            "requirements": self.requirements
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create a role from a dictionary.
        
        Args:
            data: The dictionary data
            
        Returns:
            The created role
        """
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            responsibilities=data.get("responsibilities", []),
            requirements=data.get("requirements", {})
        )


class RoleAssignment:
    """Assignment of a role to an agent."""
    
    def __init__(self, agent: IAgent, role: Role, start_time: float, end_time: Optional[float] = None):
        """Initialize the role assignment.
        
        Args:
            agent: The agent
            role: The role
            start_time: The start time of the assignment
            end_time: The end time of the assignment, or None if ongoing
        """
        self.agent = agent
        self.role = role
        self.start_time = start_time
        self.end_time = end_time
    
    @property
    def is_active(self) -> bool:
        """Check if the assignment is active.
        
        Returns:
            True if the assignment is active
        """
        return self.end_time is None
    
    def end(self, end_time: float) -> None:
        """End the assignment.
        
        Args:
            end_time: The end time
        """
        self.end_time = end_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the assignment to a dictionary.
        
        Returns:
            Dictionary representation of the assignment
        """
        return {
            "agent_name": self.agent.name,
            "role_name": self.role.name,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class RoleManager:
    """Manager for roles and role assignments."""
    
    def __init__(self):
        """Initialize the role manager."""
        self.roles: Dict[str, Role] = {}
        self.assignments: List[RoleAssignment] = []
    
    def add_role(self, role: Role) -> None:
        """Add a role.
        
        Args:
            role: The role to add
        """
        self.roles[role.name] = role
        logger.info(f"Added role {role.name}")
    
    def remove_role(self, role_name: str) -> None:
        """Remove a role.
        
        Args:
            role_name: The name of the role to remove
        """
        if role_name in self.roles:
            del self.roles[role_name]
            logger.info(f"Removed role {role_name}")
    
    def get_role(self, role_name: str) -> Optional[Role]:
        """Get a role by name.
        
        Args:
            role_name: The name of the role
            
        Returns:
            The role, or None if not found
        """
        return self.roles.get(role_name)
    
    def assign_role(self, agent: IAgent, role_name: str, start_time: float) -> Optional[RoleAssignment]:
        """Assign a role to an agent.
        
        Args:
            agent: The agent
            role_name: The name of the role
            start_time: The start time of the assignment
            
        Returns:
            The created assignment, or None if the role doesn't exist
        """
        role = self.get_role(role_name)
        if not role:
            logger.warning(f"Role {role_name} not found")
            return None
        
        # End any active assignments for this agent with this role
        for assignment in self.assignments:
            if (
                assignment.agent.name == agent.name and
                assignment.role.name == role_name and
                assignment.is_active
            ):
                assignment.end(start_time)
        
        # Create a new assignment
        assignment = RoleAssignment(agent, role, start_time)
        self.assignments.append(assignment)
        
        logger.info(f"Assigned role {role_name} to agent {agent.name}")
        
        return assignment
    
    def unassign_role(self, agent_name: str, role_name: str, end_time: float) -> None:
        """Unassign a role from an agent.
        
        Args:
            agent_name: The name of the agent
            role_name: The name of the role
            end_time: The end time of the assignment
        """
        for assignment in self.assignments:
            if (
                assignment.agent.name == agent_name and
                assignment.role.name == role_name and
                assignment.is_active
            ):
                assignment.end(end_time)
                logger.info(f"Unassigned role {role_name} from agent {agent_name}")
    
    def get_active_assignments(self, agent_name: Optional[str] = None, role_name: Optional[str] = None) -> List[RoleAssignment]:
        """Get active role assignments.
        
        Args:
            agent_name: Optional name of the agent to filter by
            role_name: Optional name of the role to filter by
            
        Returns:
            List of active assignments
        """
        return [
            assignment
            for assignment in self.assignments
            if assignment.is_active and
               (agent_name is None or assignment.agent.name == agent_name) and
               (role_name is None or assignment.role.name == role_name)
        ]
    
    def get_agents_with_role(self, role_name: str) -> List[IAgent]:
        """Get all agents with a specific role.
        
        Args:
            role_name: The name of the role
            
        Returns:
            List of agents with the role
        """
        return [
            assignment.agent
            for assignment in self.get_active_assignments(role_name=role_name)
        ]
    
    def get_agent_roles(self, agent_name: str) -> List[Role]:
        """Get all roles assigned to an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of roles assigned to the agent
        """
        return [
            assignment.role
            for assignment in self.get_active_assignments(agent_name=agent_name)
        ]
    
    def has_role(self, agent_name: str, role_name: str) -> bool:
        """Check if an agent has a specific role.
        
        Args:
            agent_name: The name of the agent
            role_name: The name of the role
            
        Returns:
            True if the agent has the role
        """
        return any(
            assignment.agent.name == agent_name and assignment.role.name == role_name
            for assignment in self.get_active_assignments()
        )


class RoleTransitionRule:
    """Rule for transitioning between roles."""
    
    def __init__(
        self,
        from_role: Optional[str],
        to_role: str,
        condition: Callable[[IAgent, Dict[str, Any]], bool],
        description: str
    ):
        """Initialize the role transition rule.
        
        Args:
            from_role: The name of the source role, or None for any role
            to_role: The name of the target role
            condition: Function that determines if the transition should occur
            description: Description of the rule
        """
        self.from_role = from_role
        self.to_role = to_role
        self.condition = condition
        self.description = description
    
    def check(self, agent: IAgent, context: Dict[str, Any]) -> bool:
        """Check if the rule applies.
        
        Args:
            agent: The agent
            context: Additional context
            
        Returns:
            True if the rule applies
        """
        return self.condition(agent, context)


class RoleTransitionManager:
    """Manager for role transitions."""
    
    def __init__(self, role_manager: RoleManager):
        """Initialize the role transition manager.
        
        Args:
            role_manager: The role manager
        """
        self.role_manager = role_manager
        self.rules: List[RoleTransitionRule] = []
    
    def add_rule(self, rule: RoleTransitionRule) -> None:
        """Add a transition rule.
        
        Args:
            rule: The rule to add
        """
        self.rules.append(rule)
        logger.info(f"Added transition rule from {rule.from_role or 'any'} to {rule.to_role}")
    
    def remove_rule(self, from_role: Optional[str], to_role: str) -> None:
        """Remove a transition rule.
        
        Args:
            from_role: The name of the source role, or None for any role
            to_role: The name of the target role
        """
        self.rules = [
            rule
            for rule in self.rules
            if rule.from_role != from_role or rule.to_role != to_role
        ]
        logger.info(f"Removed transition rule from {from_role or 'any'} to {to_role}")
    
    def check_transitions(self, agent: IAgent, context: Dict[str, Any], current_time: float) -> List[str]:
        """Check for possible role transitions.
        
        Args:
            agent: The agent
            context: Additional context
            current_time: The current time
            
        Returns:
            List of role names that the agent can transition to
        """
        # Get the agent's current roles
        current_roles = self.role_manager.get_agent_roles(agent.name)
        current_role_names = {role.name for role in current_roles}
        
        # Check each rule
        transitions = []
        
        for rule in self.rules:
            # Skip if the from_role doesn't match
            if rule.from_role is not None and rule.from_role not in current_role_names:
                continue
            
            # Skip if the agent already has the to_role
            if rule.to_role in current_role_names:
                continue
            
            # Check if the rule applies
            if rule.check(agent, context):
                transitions.append(rule.to_role)
        
        return transitions
    
    def apply_transitions(self, agent: IAgent, context: Dict[str, Any], current_time: float) -> List[str]:
        """Apply role transitions.
        
        Args:
            agent: The agent
            context: Additional context
            current_time: The current time
            
        Returns:
            List of role names that the agent transitioned to
        """
        transitions = self.check_transitions(agent, context, current_time)
        
        for role_name in transitions:
            self.role_manager.assign_role(agent, role_name, current_time)
        
        return transitions


class Team:
    """Team of agents with specific roles."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize the team.
        
        Args:
            name: The name of the team
            description: The description of the team
        """
        self.name = name
        self.description = description
        self.members: Dict[str, IAgent] = {}
        self.roles: Dict[str, Set[str]] = {}  # agent_name -> set of role names
    
    def add_member(self, agent: IAgent, roles: Optional[List[str]] = None) -> None:
        """Add a member to the team.
        
        Args:
            agent: The agent to add
            roles: Optional list of role names for the agent
        """
        self.members[agent.name] = agent
        self.roles[agent.name] = set(roles or [])
        logger.info(f"Added agent {agent.name} to team {self.name}")
    
    def remove_member(self, agent_name: str) -> None:
        """Remove a member from the team.
        
        Args:
            agent_name: The name of the agent to remove
        """
        if agent_name in self.members:
            del self.members[agent_name]
            
            if agent_name in self.roles:
                del self.roles[agent_name]
            
            logger.info(f"Removed agent {agent_name} from team {self.name}")
    
    def add_role(self, agent_name: str, role_name: str) -> None:
        """Add a role to a team member.
        
        Args:
            agent_name: The name of the agent
            role_name: The name of the role
        """
        if agent_name in self.members:
            if agent_name not in self.roles:
                self.roles[agent_name] = set()
            
            self.roles[agent_name].add(role_name)
            logger.info(f"Added role {role_name} to agent {agent_name} in team {self.name}")
        else:
            logger.warning(f"Agent {agent_name} is not a member of team {self.name}")
    
    def remove_role(self, agent_name: str, role_name: str) -> None:
        """Remove a role from a team member.
        
        Args:
            agent_name: The name of the agent
            role_name: The name of the role
        """
        if agent_name in self.roles and role_name in self.roles[agent_name]:
            self.roles[agent_name].remove(role_name)
            logger.info(f"Removed role {role_name} from agent {agent_name} in team {self.name}")
    
    def get_members_with_role(self, role_name: str) -> List[IAgent]:
        """Get all team members with a specific role.
        
        Args:
            role_name: The name of the role
            
        Returns:
            List of agents with the role
        """
        return [
            self.members[agent_name]
            for agent_name, roles in self.roles.items()
            if role_name in roles
        ]
    
    def get_member_roles(self, agent_name: str) -> List[str]:
        """Get all roles of a team member.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of role names
        """
        return list(self.roles.get(agent_name, set()))
    
    def has_role(self, agent_name: str, role_name: str) -> bool:
        """Check if a team member has a specific role.
        
        Args:
            agent_name: The name of the agent
            role_name: The name of the role
            
        Returns:
            True if the agent has the role
        """
        return agent_name in self.roles and role_name in self.roles[agent_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the team to a dictionary.
        
        Returns:
            Dictionary representation of the team
        """
        return {
            "name": self.name,
            "description": self.description,
            "members": [
                {
                    "agent_name": agent_name,
                    "roles": list(self.roles.get(agent_name, set()))
                }
                for agent_name in self.members
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], agents: Dict[str, IAgent]) -> 'Team':
        """Create a team from a dictionary.
        
        Args:
            data: The dictionary data
            agents: Dictionary mapping agent names to agents
            
        Returns:
            The created team
        """
        team = cls(
            name=data.get("name", ""),
            description=data.get("description", "")
        )
        
        for member_data in data.get("members", []):
            agent_name = member_data.get("agent_name")
            roles = member_data.get("roles", [])
            
            if agent_name and agent_name in agents:
                team.add_member(agents[agent_name], roles)
        
        return team


class TeamManager:
    """Manager for teams."""
    
    def __init__(self, role_manager: Optional[RoleManager] = None):
        """Initialize the team manager.
        
        Args:
            role_manager: Optional role manager to use
        """
        self.teams: Dict[str, Team] = {}
        self.role_manager = role_manager
    
    def create_team(self, name: str, description: str = "") -> Team:
        """Create a new team.
        
        Args:
            name: The name of the team
            description: The description of the team
            
        Returns:
            The created team
        """
        team = Team(name, description)
        self.teams[name] = team
        logger.info(f"Created team {name}")
        return team
    
    def remove_team(self, team_name: str) -> None:
        """Remove a team.
        
        Args:
            team_name: The name of the team to remove
        """
        if team_name in self.teams:
            del self.teams[team_name]
            logger.info(f"Removed team {team_name}")
    
    def get_team(self, team_name: str) -> Optional[Team]:
        """Get a team by name.
        
        Args:
            team_name: The name of the team
            
        Returns:
            The team, or None if not found
        """
        return self.teams.get(team_name)
    
    def add_member(self, team_name: str, agent: IAgent, roles: Optional[List[str]] = None) -> None:
        """Add a member to a team.
        
        Args:
            team_name: The name of the team
            agent: The agent to add
            roles: Optional list of role names for the agent
        """
        team = self.get_team(team_name)
        if team:
            team.add_member(agent, roles)
            
            # If a role manager is available, assign the roles
            if self.role_manager and roles:
                current_time = time.time()
                for role_name in roles:
                    self.role_manager.assign_role(agent, role_name, current_time)
        else:
            logger.warning(f"Team {team_name} not found")
    
    def remove_member(self, team_name: str, agent_name: str) -> None:
        """Remove a member from a team.
        
        Args:
            team_name: The name of the team
            agent_name: The name of the agent to remove
        """
        team = self.get_team(team_name)
        if team:
            # Get the agent's roles before removing
            roles = team.get_member_roles(agent_name)
            
            team.remove_member(agent_name)
            
            # If a role manager is available, unassign the roles
            if self.role_manager and roles:
                current_time = time.time()
                for role_name in roles:
                    self.role_manager.unassign_role(agent_name, role_name, current_time)
        else:
            logger.warning(f"Team {team_name} not found")
    
    def get_agent_teams(self, agent_name: str) -> List[Team]:
        """Get all teams that an agent is a member of.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of teams
        """
        return [
            team
            for team in self.teams.values()
            if agent_name in team.members
        ]
    
    def get_teams_with_role(self, role_name: str) -> List[Team]:
        """Get all teams that have members with a specific role.
        
        Args:
            role_name: The name of the role
            
        Returns:
            List of teams
        """
        return [
            team
            for team in self.teams.values()
            if any(role_name in roles for roles in team.roles.values())
        ]
    
    def form_team_by_roles(self, team_name: str, description: str, role_requirements: Dict[str, int], agents: List[IAgent]) -> Optional[Team]:
        """Form a team based on role requirements.
        
        Args:
            team_name: The name of the team
            description: The description of the team
            role_requirements: Dictionary mapping role names to required counts
            agents: List of available agents
            
        Returns:
            The formed team, or None if the requirements couldn't be met
        """
        if not self.role_manager:
            logger.warning("No role manager available")
            return None
        
        # Check if we have enough agents with the required roles
        for role_name, count in role_requirements.items():
            agents_with_role = self.role_manager.get_agents_with_role(role_name)
            if len(agents_with_role) < count:
                logger.warning(f"Not enough agents with role {role_name}")
                return None
        
        # Create the team
        team = self.create_team(team_name, description)
        
        # Assign agents to roles
        for role_name, count in role_requirements.items():
            agents_with_role = self.role_manager.get_agents_with_role(role_name)
            
            # Sort by whether they're already in the team
            agents_with_role.sort(key=lambda a: a.name not in team.members)
            
            # Add agents to the team
            for i in range(count):
                if i < len(agents_with_role):
                    agent = agents_with_role[i]
                    if agent.name not in team.members:
                        team.add_member(agent, [role_name])
                    else:
                        team.add_role(agent.name, role_name)
        
        return team
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the team manager to a dictionary.
        
        Returns:
            Dictionary representation of the team manager
        """
        return {
            "teams": [team.to_dict() for team in self.teams.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], agents: Dict[str, IAgent], role_manager: Optional[RoleManager] = None) -> 'TeamManager':
        """Create a team manager from a dictionary.
        
        Args:
            data: The dictionary data
            agents: Dictionary mapping agent names to agents
            role_manager: Optional role manager to use
            
        Returns:
            The created team manager
        """
        manager = cls(role_manager=role_manager)
        
        for team_data in data.get("teams", []):
            team = Team.from_dict(team_data, agents)
            manager.teams[team.name] = team
        
        return manager


class RoleBasedCoordinator:
    """Coordinator for role-based agent coordination."""
    
    def __init__(self, role_manager: RoleManager, team_manager: TeamManager):
        """Initialize the role-based coordinator.
        
        Args:
            role_manager: The role manager
            team_manager: The team manager
        """
        self.role_manager = role_manager
        self.team_manager = team_manager
    
    async def coordinate(self, team_name: str, query: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate a team to solve a problem.
        
        Args:
            team_name: The name of the team
            query: The query to process
            context: Additional context for the query
            
        Returns:
            The result of the coordination
        """
        team = self.team_manager.get_team(team_name)
        if not team:
            raise ValueError(f"Team {team_name} not found")
        
        # Create a context if not provided
        if context is None:
            context = {}
        
        # Get all roles in the team
        all_roles = set()
        for roles in team.roles.values():
            all_roles.update(roles)
        
        # Get role definitions
        role_definitions = {}
        for role_name in all_roles:
            role = self.role_manager.get_role(role_name)
            if role:
                role_definitions[role_name] = role
        
        # Analyze the query to determine which roles are needed
        analysis_query = f"Analyze the following query and determine which roles are needed to solve it:\n\n{query}\n\nAvailable roles:\n"
        for role_name, role in role_definitions.items():
            analysis_query += f"\n{role_name}: {role.description}"
        
        # Use a random agent for analysis
        analysis_agent = next(iter(team.members.values()))
        analysis_output = await analysis_agent.run(
            AgentInput(query=analysis_query, context=context)
        )
        
        # Parse the analysis to get required roles
        required_roles = self._parse_required_roles(analysis_output.response, role_definitions.keys())
        
        # Get agents for each required role
        role_agents = {}
        for role_name in required_roles:
            agents = team.get_members_with_role(role_name)
            if agents:
                role_agents[role_name] = agents[0]  # Use the first agent with the role
        
        # Create a plan
        plan_query = f"Create a plan to solve the following problem using the available roles:\n\n{query}\n\nAvailable roles:\n"
        for role_name, agent in role_agents.items():
            role = role_definitions.get(role_name)
            if role:
                plan_query += f"\n{role_name} ({agent.name}): {role.description}"
        
        plan_output = await analysis_agent.run(
            AgentInput(query=plan_query, context=context)
        )
        
        # Parse the plan to get steps
        steps = self._parse_plan(plan_output.response)
        
        # Execute the plan
        results = []
        for step in steps:
            step_role = step.get("role")
            step_task = step.get("task")
            
            if step_role and step_task and step_role in role_agents:
                agent = role_agents[step_role]
                
                # Execute the task
                task_query = f"Execute the following task as the {step_role} role:\n\n{step_task}"
                task_output = await agent.run(
                    AgentInput(query=task_query, context=context)
                )
                
                results.append({
                    "role": step_role,
                    "agent": agent.name,
                    "task": step_task,
                    "result": task_output.response
                })
        
        # Aggregate the results
        aggregation_query = f"Aggregate the results for the following problem:\n\n{query}\n\nResults:\n"
        for result in results:
            aggregation_query += f"\n{result['role']} ({result['agent']}): {result['result']}"
        
        aggregation_output = await analysis_agent.run(
            AgentInput(query=aggregation_query, context=context)
        )
        
        return aggregation_output.response
    
    def _parse_required_roles(self, analysis: str, available_roles: Set[str]) -> List[str]:
        """Parse the analysis to get required roles.
        
        Args:
            analysis: The analysis output
            available_roles: Set of available role names
            
        Returns:
            List of required role names
        """
        required_roles = []
        
        for role_name in available_roles:
            if role_name.lower() in analysis.lower():
                required_roles.append(role_name)
        
        return required_roles
    
    def _parse_plan(self, plan: str) -> List[Dict[str, str]]:
        """Parse the plan to get steps.
        
        Args:
            plan: The plan output
            
        Returns:
            List of step dictionaries with 'role' and 'task' keys
        """
        steps = []
        
        # Simple parsing: look for lines with role names
        for line in plan.strip().split("\n"):
            for role_name in self.role_manager.roles:
                if role_name.lower() in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    task = parts[1].strip()
                    steps.append({
                        "role": role_name,
                        "task": task
                    })
                    break
        
        return steps
