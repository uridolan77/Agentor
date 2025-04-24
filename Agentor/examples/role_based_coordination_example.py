"""
Example demonstrating role-based coordination for multi-agent systems.

This example shows how to use role-based coordination to organize agents
into teams with specific roles and responsibilities.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.components.coordination import (
    Role,
    RoleManager,
    RoleTransitionRule,
    RoleTransitionManager,
    Team,
    TeamManager,
    RoleBasedCoordinator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RoleBasedAgent(EnhancedAgent):
    """Agent that can fulfill different roles."""
    
    def __init__(self, name: str, skills: Dict[str, float] = None):
        """Initialize the role-based agent.
        
        Args:
            name: The name of the agent
            skills: Dictionary mapping skill names to proficiency levels
        """
        super().__init__(name=name)
        self.skills = skills or {}
        self.current_roles: List[str] = []
    
    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Run the agent.
        
        Args:
            input_data: The input data
            
        Returns:
            The agent's output
        """
        query = input_data.query
        context = input_data.context or {}
        
        # Get the agent's current roles
        roles = context.get("roles", self.current_roles)
        
        if roles:
            # Act according to the agent's roles
            role_str = ", ".join(roles)
            response = f"Agent {self.name} acting as {role_str} processed: {query}"
            
            # Add role-specific processing
            for role in roles:
                if role == "ProjectManager":
                    response += "\n\nAs a Project Manager, I'll coordinate the team and ensure we meet our deadlines."
                elif role == "Architect":
                    response += "\n\nAs an Architect, I'll design the system architecture and ensure it meets our requirements."
                elif role == "Developer":
                    response += "\n\nAs a Developer, I'll implement the required functionality according to the specifications."
                elif role == "Tester":
                    response += "\n\nAs a Tester, I'll verify that the implementation meets the requirements and is free of defects."
                elif role == "DevOps":
                    response += "\n\nAs a DevOps engineer, I'll set up the deployment pipeline and ensure smooth operation."
        else:
            # No roles assigned
            response = f"Agent {self.name} processed: {query}"
        
        return AgentOutput(response=response, context=context)


async def role_management_example():
    """Example demonstrating role management."""
    logger.info("\n=== Role Management Example ===")
    
    # Create the role manager
    role_manager = RoleManager()
    
    # Define roles
    project_manager_role = Role(
        name="ProjectManager",
        description="Manages the project and coordinates the team",
        responsibilities=[
            "Create and maintain project plans",
            "Coordinate team members",
            "Track progress and report status",
            "Manage risks and issues"
        ],
        requirements={
            "skills": ["project_management", "leadership", "communication"],
            "min_experience": 3
        }
    )
    
    architect_role = Role(
        name="Architect",
        description="Designs the system architecture",
        responsibilities=[
            "Design the system architecture",
            "Make technology choices",
            "Ensure the architecture meets requirements",
            "Guide the development team"
        ],
        requirements={
            "skills": ["system_design", "technical_leadership", "problem_solving"],
            "min_experience": 5
        }
    )
    
    developer_role = Role(
        name="Developer",
        description="Implements the system",
        responsibilities=[
            "Implement features",
            "Write clean, maintainable code",
            "Follow coding standards",
            "Fix bugs"
        ],
        requirements={
            "skills": ["programming", "problem_solving", "testing"],
            "min_experience": 1
        }
    )
    
    tester_role = Role(
        name="Tester",
        description="Tests the system",
        responsibilities=[
            "Write and execute test cases",
            "Report defects",
            "Verify fixes",
            "Ensure quality"
        ],
        requirements={
            "skills": ["testing", "attention_to_detail", "communication"],
            "min_experience": 1
        }
    )
    
    devops_role = Role(
        name="DevOps",
        description="Manages deployment and operations",
        responsibilities=[
            "Set up deployment pipelines",
            "Configure environments",
            "Monitor system performance",
            "Troubleshoot issues"
        ],
        requirements={
            "skills": ["devops", "automation", "troubleshooting"],
            "min_experience": 2
        }
    )
    
    # Add roles to the manager
    role_manager.add_role(project_manager_role)
    role_manager.add_role(architect_role)
    role_manager.add_role(developer_role)
    role_manager.add_role(tester_role)
    role_manager.add_role(devops_role)
    
    # Create agents
    alice = RoleBasedAgent("Alice", skills={"project_management": 0.9, "leadership": 0.8, "communication": 0.9})
    bob = RoleBasedAgent("Bob", skills={"system_design": 0.9, "technical_leadership": 0.8, "problem_solving": 0.9})
    charlie = RoleBasedAgent("Charlie", skills={"programming": 0.9, "problem_solving": 0.8, "testing": 0.7})
    dave = RoleBasedAgent("Dave", skills={"programming": 0.8, "problem_solving": 0.7, "testing": 0.6})
    eve = RoleBasedAgent("Eve", skills={"testing": 0.9, "attention_to_detail": 0.9, "communication": 0.8})
    frank = RoleBasedAgent("Frank", skills={"devops": 0.9, "automation": 0.8, "troubleshooting": 0.9})
    
    # Assign roles to agents
    current_time = time.time()
    role_manager.assign_role(alice, "ProjectManager", current_time)
    role_manager.assign_role(bob, "Architect", current_time)
    role_manager.assign_role(charlie, "Developer", current_time)
    role_manager.assign_role(dave, "Developer", current_time)
    role_manager.assign_role(eve, "Tester", current_time)
    role_manager.assign_role(frank, "DevOps", current_time)
    
    # Get active assignments
    active_assignments = role_manager.get_active_assignments()
    logger.info(f"Active role assignments: {len(active_assignments)}")
    
    # Get agents with a specific role
    developers = role_manager.get_agents_with_role("Developer")
    logger.info(f"Developers: {[agent.name for agent in developers]}")
    
    # Check if an agent has a role
    is_pm = role_manager.has_role("Alice", "ProjectManager")
    logger.info(f"Is Alice a Project Manager? {is_pm}")
    
    # Run agents with their roles
    for agent, role_name in [
        (alice, "ProjectManager"),
        (bob, "Architect"),
        (charlie, "Developer"),
        (eve, "Tester"),
        (frank, "DevOps")
    ]:
        query = "How would you approach this project?"
        context = {"roles": [role_name]}
        result = await agent.run(AgentInput(query=query, context=context))
        logger.info(f"{agent.name} as {role_name}: {result.response}")


async def role_transition_example():
    """Example demonstrating role transitions."""
    logger.info("\n=== Role Transition Example ===")
    
    # Create the role manager
    role_manager = RoleManager()
    
    # Define roles
    junior_dev_role = Role(
        name="JuniorDeveloper",
        description="Junior software developer",
        responsibilities=[
            "Implement simple features",
            "Fix bugs",
            "Write tests"
        ],
        requirements={
            "skills": ["programming"],
            "min_experience": 0
        }
    )
    
    senior_dev_role = Role(
        name="SeniorDeveloper",
        description="Senior software developer",
        responsibilities=[
            "Implement complex features",
            "Review code",
            "Mentor junior developers",
            "Participate in architecture discussions"
        ],
        requirements={
            "skills": ["programming", "code_review", "mentoring"],
            "min_experience": 3
        }
    )
    
    tech_lead_role = Role(
        name="TechLead",
        description="Technical team leader",
        responsibilities=[
            "Lead the development team",
            "Make technical decisions",
            "Ensure code quality",
            "Coordinate with other teams"
        ],
        requirements={
            "skills": ["programming", "leadership", "architecture"],
            "min_experience": 5
        }
    )
    
    # Add roles to the manager
    role_manager.add_role(junior_dev_role)
    role_manager.add_role(senior_dev_role)
    role_manager.add_role(tech_lead_role)
    
    # Create the transition manager
    transition_manager = RoleTransitionManager(role_manager)
    
    # Define transition rules
    
    # Junior Developer -> Senior Developer
    def can_become_senior(agent: RoleBasedAgent, context: Dict[str, Any]) -> bool:
        # Check if the agent has the required skills and experience
        return (
            agent.skills.get("programming", 0) >= 0.8 and
            agent.skills.get("code_review", 0) >= 0.7 and
            context.get("experience", 0) >= 3
        )
    
    junior_to_senior_rule = RoleTransitionRule(
        from_role="JuniorDeveloper",
        to_role="SeniorDeveloper",
        condition=can_become_senior,
        description="Junior developers can become senior developers if they have sufficient skills and experience"
    )
    
    # Senior Developer -> Tech Lead
    def can_become_tech_lead(agent: RoleBasedAgent, context: Dict[str, Any]) -> bool:
        # Check if the agent has the required skills and experience
        return (
            agent.skills.get("programming", 0) >= 0.8 and
            agent.skills.get("leadership", 0) >= 0.8 and
            agent.skills.get("architecture", 0) >= 0.7 and
            context.get("experience", 0) >= 5
        )
    
    senior_to_tech_lead_rule = RoleTransitionRule(
        from_role="SeniorDeveloper",
        to_role="TechLead",
        condition=can_become_tech_lead,
        description="Senior developers can become tech leads if they have sufficient skills and experience"
    )
    
    # Add rules to the manager
    transition_manager.add_rule(junior_to_senior_rule)
    transition_manager.add_rule(senior_to_tech_lead_rule)
    
    # Create agents
    alice = RoleBasedAgent("Alice", skills={
        "programming": 0.9,
        "code_review": 0.8,
        "mentoring": 0.7,
        "leadership": 0.6,
        "architecture": 0.5
    })
    
    bob = RoleBasedAgent("Bob", skills={
        "programming": 0.8,
        "code_review": 0.7,
        "mentoring": 0.6,
        "leadership": 0.8,
        "architecture": 0.7
    })
    
    charlie = RoleBasedAgent("Charlie", skills={
        "programming": 0.7,
        "code_review": 0.5,
        "mentoring": 0.4,
        "leadership": 0.3,
        "architecture": 0.2
    })
    
    # Assign initial roles
    current_time = time.time()
    role_manager.assign_role(alice, "SeniorDeveloper", current_time)
    role_manager.assign_role(bob, "SeniorDeveloper", current_time)
    role_manager.assign_role(charlie, "JuniorDeveloper", current_time)
    
    # Check for transitions
    context = {"experience": 6}  # Alice has 6 years of experience
    transitions = transition_manager.check_transitions(alice, context, current_time)
    logger.info(f"Possible transitions for Alice: {transitions}")
    
    context = {"experience": 5}  # Bob has 5 years of experience
    transitions = transition_manager.check_transitions(bob, context, current_time)
    logger.info(f"Possible transitions for Bob: {transitions}")
    
    context = {"experience": 2}  # Charlie has 2 years of experience
    transitions = transition_manager.check_transitions(charlie, context, current_time)
    logger.info(f"Possible transitions for Charlie: {transitions}")
    
    # Apply transitions
    context = {"experience": 6}
    applied = transition_manager.apply_transitions(alice, context, current_time)
    logger.info(f"Applied transitions for Alice: {applied}")
    
    context = {"experience": 5}
    applied = transition_manager.apply_transitions(bob, context, current_time)
    logger.info(f"Applied transitions for Bob: {applied}")
    
    context = {"experience": 4}  # Charlie now has 4 years of experience
    applied = transition_manager.apply_transitions(charlie, context, current_time)
    logger.info(f"Applied transitions for Charlie: {applied}")
    
    # Get active assignments after transitions
    active_assignments = role_manager.get_active_assignments()
    for assignment in active_assignments:
        logger.info(f"{assignment.agent.name} is assigned to {assignment.role.name}")


async def team_formation_example():
    """Example demonstrating team formation."""
    logger.info("\n=== Team Formation Example ===")
    
    # Create the role manager
    role_manager = RoleManager()
    
    # Define roles
    project_manager_role = Role(
        name="ProjectManager",
        description="Manages the project and coordinates the team",
        responsibilities=[
            "Create and maintain project plans",
            "Coordinate team members",
            "Track progress and report status",
            "Manage risks and issues"
        ]
    )
    
    architect_role = Role(
        name="Architect",
        description="Designs the system architecture",
        responsibilities=[
            "Design the system architecture",
            "Make technology choices",
            "Ensure the architecture meets requirements",
            "Guide the development team"
        ]
    )
    
    frontend_dev_role = Role(
        name="FrontendDeveloper",
        description="Implements the frontend",
        responsibilities=[
            "Implement the user interface",
            "Ensure good user experience",
            "Optimize performance",
            "Fix frontend bugs"
        ]
    )
    
    backend_dev_role = Role(
        name="BackendDeveloper",
        description="Implements the backend",
        responsibilities=[
            "Implement the server-side logic",
            "Design and implement APIs",
            "Optimize performance",
            "Fix backend bugs"
        ]
    )
    
    tester_role = Role(
        name="Tester",
        description="Tests the system",
        responsibilities=[
            "Write and execute test cases",
            "Report defects",
            "Verify fixes",
            "Ensure quality"
        ]
    )
    
    devops_role = Role(
        name="DevOps",
        description="Manages deployment and operations",
        responsibilities=[
            "Set up deployment pipelines",
            "Configure environments",
            "Monitor system performance",
            "Troubleshoot issues"
        ]
    )
    
    # Add roles to the manager
    role_manager.add_role(project_manager_role)
    role_manager.add_role(architect_role)
    role_manager.add_role(frontend_dev_role)
    role_manager.add_role(backend_dev_role)
    role_manager.add_role(tester_role)
    role_manager.add_role(devops_role)
    
    # Create the team manager
    team_manager = TeamManager(role_manager)
    
    # Create agents
    alice = RoleBasedAgent("Alice", skills={"project_management": 0.9, "leadership": 0.8})
    bob = RoleBasedAgent("Bob", skills={"system_design": 0.9, "technical_leadership": 0.8})
    charlie = RoleBasedAgent("Charlie", skills={"frontend": 0.9, "ui_design": 0.8})
    dave = RoleBasedAgent("Dave", skills={"backend": 0.9, "database": 0.8})
    eve = RoleBasedAgent("Eve", skills={"testing": 0.9, "automation": 0.8})
    frank = RoleBasedAgent("Frank", skills={"devops": 0.9, "cloud": 0.8})
    grace = RoleBasedAgent("Grace", skills={"frontend": 0.8, "ui_design": 0.7})
    henry = RoleBasedAgent("Henry", skills={"backend": 0.8, "database": 0.7})
    
    # Assign roles to agents
    current_time = time.time()
    role_manager.assign_role(alice, "ProjectManager", current_time)
    role_manager.assign_role(bob, "Architect", current_time)
    role_manager.assign_role(charlie, "FrontendDeveloper", current_time)
    role_manager.assign_role(dave, "BackendDeveloper", current_time)
    role_manager.assign_role(eve, "Tester", current_time)
    role_manager.assign_role(frank, "DevOps", current_time)
    role_manager.assign_role(grace, "FrontendDeveloper", current_time)
    role_manager.assign_role(henry, "BackendDeveloper", current_time)
    
    # Create teams
    web_team = team_manager.create_team("WebTeam", "Team responsible for the web application")
    
    # Add members to the team
    team_manager.add_member("WebTeam", alice, ["ProjectManager"])
    team_manager.add_member("WebTeam", bob, ["Architect"])
    team_manager.add_member("WebTeam", charlie, ["FrontendDeveloper"])
    team_manager.add_member("WebTeam", dave, ["BackendDeveloper"])
    team_manager.add_member("WebTeam", eve, ["Tester"])
    team_manager.add_member("WebTeam", frank, ["DevOps"])
    
    # Get team members with a specific role
    frontend_devs = web_team.get_members_with_role("FrontendDeveloper")
    logger.info(f"Frontend developers in WebTeam: {[agent.name for agent in frontend_devs]}")
    
    # Create another team using role requirements
    mobile_team = team_manager.form_team_by_roles(
        team_name="MobileTeam",
        description="Team responsible for the mobile application",
        role_requirements={
            "ProjectManager": 1,
            "Architect": 1,
            "FrontendDeveloper": 1,
            "BackendDeveloper": 1,
            "Tester": 1
        },
        agents=[alice, bob, charlie, dave, eve, frank, grace, henry]
    )
    
    if mobile_team:
        logger.info(f"Created MobileTeam with {len(mobile_team.members)} members")
        for agent_name, roles in mobile_team.roles.items():
            logger.info(f"{agent_name} has roles: {roles}")
    else:
        logger.warning("Failed to create MobileTeam")


async def role_based_coordination_example():
    """Example demonstrating role-based coordination."""
    logger.info("\n=== Role-Based Coordination Example ===")
    
    # Create the role manager
    role_manager = RoleManager()
    
    # Define roles
    project_manager_role = Role(
        name="ProjectManager",
        description="Manages the project and coordinates the team",
        responsibilities=[
            "Create and maintain project plans",
            "Coordinate team members",
            "Track progress and report status",
            "Manage risks and issues"
        ]
    )
    
    architect_role = Role(
        name="Architect",
        description="Designs the system architecture",
        responsibilities=[
            "Design the system architecture",
            "Make technology choices",
            "Ensure the architecture meets requirements",
            "Guide the development team"
        ]
    )
    
    frontend_dev_role = Role(
        name="FrontendDeveloper",
        description="Implements the frontend",
        responsibilities=[
            "Implement the user interface",
            "Ensure good user experience",
            "Optimize performance",
            "Fix frontend bugs"
        ]
    )
    
    backend_dev_role = Role(
        name="BackendDeveloper",
        description="Implements the backend",
        responsibilities=[
            "Implement the server-side logic",
            "Design and implement APIs",
            "Optimize performance",
            "Fix backend bugs"
        ]
    )
    
    tester_role = Role(
        name="Tester",
        description="Tests the system",
        responsibilities=[
            "Write and execute test cases",
            "Report defects",
            "Verify fixes",
            "Ensure quality"
        ]
    )
    
    # Add roles to the manager
    role_manager.add_role(project_manager_role)
    role_manager.add_role(architect_role)
    role_manager.add_role(frontend_dev_role)
    role_manager.add_role(backend_dev_role)
    role_manager.add_role(tester_role)
    
    # Create the team manager
    team_manager = TeamManager(role_manager)
    
    # Create agents
    alice = RoleBasedAgent("Alice", skills={"project_management": 0.9, "leadership": 0.8})
    bob = RoleBasedAgent("Bob", skills={"system_design": 0.9, "technical_leadership": 0.8})
    charlie = RoleBasedAgent("Charlie", skills={"frontend": 0.9, "ui_design": 0.8})
    dave = RoleBasedAgent("Dave", skills={"backend": 0.9, "database": 0.8})
    eve = RoleBasedAgent("Eve", skills={"testing": 0.9, "automation": 0.8})
    
    # Assign roles to agents
    current_time = time.time()
    role_manager.assign_role(alice, "ProjectManager", current_time)
    role_manager.assign_role(bob, "Architect", current_time)
    role_manager.assign_role(charlie, "FrontendDeveloper", current_time)
    role_manager.assign_role(dave, "BackendDeveloper", current_time)
    role_manager.assign_role(eve, "Tester", current_time)
    
    # Create a team
    dev_team = team_manager.create_team("DevTeam", "Development team")
    
    # Add members to the team
    team_manager.add_member("DevTeam", alice, ["ProjectManager"])
    team_manager.add_member("DevTeam", bob, ["Architect"])
    team_manager.add_member("DevTeam", charlie, ["FrontendDeveloper"])
    team_manager.add_member("DevTeam", dave, ["BackendDeveloper"])
    team_manager.add_member("DevTeam", eve, ["Tester"])
    
    # Create the coordinator
    coordinator = RoleBasedCoordinator(role_manager, team_manager)
    
    # Coordinate the team to solve a problem
    problem = "Develop a web application for managing inventory"
    result = await coordinator.coordinate("DevTeam", problem)
    
    logger.info(f"Coordination result: {result}")


async def main():
    """Run all examples."""
    await role_management_example()
    await role_transition_example()
    await team_formation_example()
    await role_based_coordination_example()


if __name__ == "__main__":
    asyncio.run(main())
