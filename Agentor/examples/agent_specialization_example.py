"""
Example demonstrating agent specialization and hierarchy.

This example shows how to use agent specialization and hierarchy mechanisms
to organize agents based on their skills, knowledge, and roles.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from agentor.agents.enhanced_base import EnhancedAgent
from agentor.core.interfaces.agent import AgentInput, AgentOutput
from agentor.components.coordination import (
    Skill,
    Knowledge,
    AgentProfile,
    SpecializationManager,
    HierarchyNode,
    AgentHierarchy
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpecializedAgent(EnhancedAgent):
    """Agent with specific skills and knowledge."""
    
    def __init__(self, name: str):
        """Initialize the specialized agent.
        
        Args:
            name: The name of the agent
        """
        super().__init__(name=name)
    
    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Run the agent.
        
        Args:
            input_data: The input data
            
        Returns:
            The agent's output
        """
        query = input_data.query
        context = input_data.context or {}
        
        # Get the agent's profile if available
        profile = context.get("profile")
        
        if profile:
            # Use the agent's skills and knowledge
            skills = ", ".join(skill.name for skill in profile.skills.values())
            knowledge_domains = ", ".join(knowledge.domain for knowledge in profile.knowledge.values())
            
            response = f"Agent {self.name} with skills ({skills}) and knowledge in ({knowledge_domains}) processed: {query}"
        else:
            response = f"Agent {self.name} processed: {query}"
        
        return AgentOutput(response=response, context=context)


async def specialization_example():
    """Example demonstrating agent specialization."""
    logger.info("\n=== Agent Specialization Example ===")
    
    # Create agents
    frontend_agent = SpecializedAgent("FrontendAgent")
    backend_agent = SpecializedAgent("BackendAgent")
    database_agent = SpecializedAgent("DatabaseAgent")
    devops_agent = SpecializedAgent("DevOpsAgent")
    
    # Create the specialization manager
    manager = SpecializationManager()
    
    # Add agents to the manager
    manager.add_agent(frontend_agent)
    manager.add_agent(backend_agent)
    manager.add_agent(database_agent)
    manager.add_agent(devops_agent)
    
    # Get agent profiles
    frontend_profile = manager.get_profile("FrontendAgent")
    backend_profile = manager.get_profile("BackendAgent")
    database_profile = manager.get_profile("DatabaseAgent")
    devops_profile = manager.get_profile("DevOpsAgent")
    
    # Add skills to agents
    frontend_profile.add_skill(Skill("HTML", "HyperText Markup Language", 0.9))
    frontend_profile.add_skill(Skill("CSS", "Cascading Style Sheets", 0.8))
    frontend_profile.add_skill(Skill("JavaScript", "Client-side scripting language", 0.9))
    frontend_profile.add_skill(Skill("React", "JavaScript library for building user interfaces", 0.7))
    
    backend_profile.add_skill(Skill("Python", "General-purpose programming language", 0.9))
    backend_profile.add_skill(Skill("Django", "Python web framework", 0.8))
    backend_profile.add_skill(Skill("API Design", "Designing application programming interfaces", 0.7))
    
    database_profile.add_skill(Skill("SQL", "Structured Query Language", 0.9))
    database_profile.add_skill(Skill("Database Design", "Designing database schemas", 0.8))
    database_profile.add_skill(Skill("PostgreSQL", "Open-source relational database", 0.7))
    database_profile.add_skill(Skill("MongoDB", "NoSQL database", 0.6))
    
    devops_profile.add_skill(Skill("Docker", "Containerization platform", 0.8))
    devops_profile.add_skill(Skill("Kubernetes", "Container orchestration", 0.7))
    devops_profile.add_skill(Skill("CI/CD", "Continuous Integration/Continuous Deployment", 0.9))
    
    # Add knowledge to agents
    frontend_profile.add_knowledge(Knowledge("Web Development", "Knowledge of web development principles", 0.9))
    frontend_profile.add_knowledge(Knowledge("UI/UX", "User Interface and User Experience design", 0.7))
    
    backend_profile.add_knowledge(Knowledge("Server Architecture", "Knowledge of server architecture", 0.8))
    backend_profile.add_knowledge(Knowledge("RESTful APIs", "Representational State Transfer APIs", 0.9))
    
    database_profile.add_knowledge(Knowledge("Database Systems", "Knowledge of database systems", 0.9))
    database_profile.add_knowledge(Knowledge("Data Modeling", "Techniques for data modeling", 0.8))
    
    devops_profile.add_knowledge(Knowledge("Cloud Computing", "Knowledge of cloud computing platforms", 0.8))
    devops_profile.add_knowledge(Knowledge("Infrastructure as Code", "Managing infrastructure using code", 0.9))
    
    # Find agents with specific skills
    javascript_agents = manager.find_agents_with_skill("JavaScript")
    logger.info(f"Agents with JavaScript skill: {javascript_agents}")
    
    database_agents = manager.find_agents_with_skill("Database Design")
    logger.info(f"Agents with Database Design skill: {database_agents}")
    
    # Find agents with specific knowledge
    cloud_agents = manager.find_agents_with_knowledge("Cloud Computing")
    logger.info(f"Agents with Cloud Computing knowledge: {cloud_agents}")
    
    # Find the best agent for a skill
    best_python_agent = manager.find_best_agent_for_skill("Python")
    logger.info(f"Best agent for Python: {best_python_agent}")
    
    # Run an agent with its profile
    query = "Design a web application architecture"
    context = {"profile": frontend_profile}
    result = await frontend_agent.run(AgentInput(query=query, context=context))
    
    logger.info(f"Frontend Agent Result: {result.response}")
    
    # Run another agent with its profile
    query = "Design a database schema for a social media application"
    context = {"profile": database_profile}
    result = await database_agent.run(AgentInput(query=query, context=context))
    
    logger.info(f"Database Agent Result: {result.response}")


async def hierarchy_example():
    """Example demonstrating agent hierarchy."""
    logger.info("\n=== Agent Hierarchy Example ===")
    
    # Create agents
    cto_agent = SpecializedAgent("CTOAgent")
    dev_manager_agent = SpecializedAgent("DevManagerAgent")
    ops_manager_agent = SpecializedAgent("OpsManagerAgent")
    frontend_dev_agent = SpecializedAgent("FrontendDevAgent")
    backend_dev_agent = SpecializedAgent("BackendDevAgent")
    database_admin_agent = SpecializedAgent("DatabaseAdminAgent")
    network_admin_agent = SpecializedAgent("NetworkAdminAgent")
    
    # Create the hierarchy
    hierarchy = AgentHierarchy(root_agent=cto_agent)
    
    # Add agents to the hierarchy
    hierarchy.add_agent(dev_manager_agent, parent_name="CTOAgent")
    hierarchy.add_agent(ops_manager_agent, parent_name="CTOAgent")
    hierarchy.add_agent(frontend_dev_agent, parent_name="DevManagerAgent")
    hierarchy.add_agent(backend_dev_agent, parent_name="DevManagerAgent")
    hierarchy.add_agent(database_admin_agent, parent_name="OpsManagerAgent")
    hierarchy.add_agent(network_admin_agent, parent_name="OpsManagerAgent")
    
    # Get information about the hierarchy
    logger.info(f"Root agent: {hierarchy.root.agent.name}")
    
    # Get children of an agent
    dev_manager_children = hierarchy.get_children("DevManagerAgent")
    logger.info(f"Children of DevManagerAgent: {[agent.name for agent in dev_manager_children]}")
    
    # Get parent of an agent
    frontend_parent = hierarchy.get_parent("FrontendDevAgent")
    logger.info(f"Parent of FrontendDevAgent: {frontend_parent.name if frontend_parent else 'None'}")
    
    # Get ancestors of an agent
    frontend_ancestors = hierarchy.get_ancestors("FrontendDevAgent")
    logger.info(f"Ancestors of FrontendDevAgent: {[agent.name for agent in frontend_ancestors]}")
    
    # Get descendants of an agent
    cto_descendants = hierarchy.get_descendants("CTOAgent")
    logger.info(f"Descendants of CTOAgent: {[agent.name for agent in cto_descendants]}")
    
    # Get the level of an agent
    frontend_level = hierarchy.get_level("FrontendDevAgent")
    logger.info(f"Level of FrontendDevAgent: {frontend_level}")
    
    # Check if an agent is an ancestor of another
    is_ancestor = hierarchy.is_ancestor("CTOAgent", "FrontendDevAgent")
    logger.info(f"Is CTOAgent an ancestor of FrontendDevAgent? {is_ancestor}")
    
    # Get all agents at a specific level
    level_1_agents = hierarchy.get_agents_at_level(1)
    logger.info(f"Agents at level 1: {[agent.name for agent in level_1_agents]}")
    
    # Convert the hierarchy to a dictionary
    hierarchy_dict = hierarchy.to_dict()
    logger.info(f"Hierarchy as dictionary: {hierarchy_dict}")


async def combined_example():
    """Example combining specialization and hierarchy."""
    logger.info("\n=== Combined Specialization and Hierarchy Example ===")
    
    # Create agents
    cto_agent = SpecializedAgent("CTOAgent")
    dev_manager_agent = SpecializedAgent("DevManagerAgent")
    ops_manager_agent = SpecializedAgent("OpsManagerAgent")
    frontend_dev_agent = SpecializedAgent("FrontendDevAgent")
    backend_dev_agent = SpecializedAgent("BackendDevAgent")
    database_admin_agent = SpecializedAgent("DatabaseAdminAgent")
    network_admin_agent = SpecializedAgent("NetworkAdminAgent")
    
    # Create the specialization manager
    spec_manager = SpecializationManager()
    
    # Add agents to the manager
    spec_manager.add_agent(cto_agent)
    spec_manager.add_agent(dev_manager_agent)
    spec_manager.add_agent(ops_manager_agent)
    spec_manager.add_agent(frontend_dev_agent)
    spec_manager.add_agent(backend_dev_agent)
    spec_manager.add_agent(database_admin_agent)
    spec_manager.add_agent(network_admin_agent)
    
    # Add skills to agents
    cto_profile = spec_manager.get_profile("CTOAgent")
    cto_profile.add_skill(Skill("Leadership", "Leading teams", 0.9))
    cto_profile.add_skill(Skill("Strategy", "Strategic planning", 0.9))
    
    dev_profile = spec_manager.get_profile("DevManagerAgent")
    dev_profile.add_skill(Skill("Project Management", "Managing development projects", 0.8))
    dev_profile.add_skill(Skill("Team Leadership", "Leading development teams", 0.8))
    
    ops_profile = spec_manager.get_profile("OpsManagerAgent")
    ops_profile.add_skill(Skill("Operations Management", "Managing operations", 0.8))
    ops_profile.add_skill(Skill("Infrastructure Planning", "Planning infrastructure", 0.8))
    
    frontend_profile = spec_manager.get_profile("FrontendDevAgent")
    frontend_profile.add_skill(Skill("HTML", "HyperText Markup Language", 0.9))
    frontend_profile.add_skill(Skill("CSS", "Cascading Style Sheets", 0.8))
    frontend_profile.add_skill(Skill("JavaScript", "Client-side scripting language", 0.9))
    
    backend_profile = spec_manager.get_profile("BackendDevAgent")
    backend_profile.add_skill(Skill("Python", "General-purpose programming language", 0.9))
    backend_profile.add_skill(Skill("Django", "Python web framework", 0.8))
    
    db_profile = spec_manager.get_profile("DatabaseAdminAgent")
    db_profile.add_skill(Skill("SQL", "Structured Query Language", 0.9))
    db_profile.add_skill(Skill("Database Administration", "Administering databases", 0.9))
    
    network_profile = spec_manager.get_profile("NetworkAdminAgent")
    network_profile.add_skill(Skill("Network Configuration", "Configuring networks", 0.9))
    network_profile.add_skill(Skill("Security", "Network security", 0.8))
    
    # Create the hierarchy
    hierarchy = AgentHierarchy(root_agent=cto_agent)
    
    # Add agents to the hierarchy
    hierarchy.add_agent(dev_manager_agent, parent_name="CTOAgent")
    hierarchy.add_agent(ops_manager_agent, parent_name="CTOAgent")
    hierarchy.add_agent(frontend_dev_agent, parent_name="DevManagerAgent")
    hierarchy.add_agent(backend_dev_agent, parent_name="DevManagerAgent")
    hierarchy.add_agent(database_admin_agent, parent_name="OpsManagerAgent")
    hierarchy.add_agent(network_admin_agent, parent_name="OpsManagerAgent")
    
    # Find the best agent for a task based on skills and position in the hierarchy
    task = "Develop a new web application"
    
    # First, find agents with relevant skills
    web_dev_agents = (
        spec_manager.find_agents_with_skill("HTML") +
        spec_manager.find_agents_with_skill("JavaScript") +
        spec_manager.find_agents_with_skill("Python")
    )
    
    # Then, find the agent with the lowest level in the hierarchy
    best_agent = None
    best_level = float('inf')
    
    for agent_name in web_dev_agents:
        level = hierarchy.get_level(agent_name)
        if level < best_level:
            best_agent = agent_name
            best_level = level
    
    logger.info(f"Best agent for task '{task}': {best_agent}")
    
    # Find all managers (level 1)
    managers = hierarchy.get_agents_at_level(1)
    logger.info(f"Managers: {[agent.name for agent in managers]}")
    
    # Find all developers and admins (level 2)
    workers = hierarchy.get_agents_at_level(2)
    logger.info(f"Workers: {[agent.name for agent in workers]}")
    
    # Run a task through the hierarchy
    # 1. CTO delegates to the appropriate manager
    query = "We need a new web application with a database backend"
    context = {"profile": cto_profile}
    cto_result = await cto_agent.run(AgentInput(query=query, context=context))
    logger.info(f"CTO: {cto_result.response}")
    
    # 2. Development Manager delegates to developers
    dev_query = "Implement the web application frontend and backend"
    context = {"profile": dev_profile}
    dev_manager_result = await dev_manager_agent.run(AgentInput(query=dev_query, context=context))
    logger.info(f"Dev Manager: {dev_manager_result.response}")
    
    # 3. Operations Manager delegates to admins
    ops_query = "Set up the database and network infrastructure"
    context = {"profile": ops_profile}
    ops_manager_result = await ops_manager_agent.run(AgentInput(query=ops_query, context=context))
    logger.info(f"Ops Manager: {ops_manager_result.response}")
    
    # 4. Frontend Developer implements the frontend
    frontend_query = "Implement the user interface using HTML, CSS, and JavaScript"
    context = {"profile": frontend_profile}
    frontend_result = await frontend_dev_agent.run(AgentInput(query=frontend_query, context=context))
    logger.info(f"Frontend Developer: {frontend_result.response}")
    
    # 5. Backend Developer implements the backend
    backend_query = "Implement the server-side logic using Python and Django"
    context = {"profile": backend_profile}
    backend_result = await backend_dev_agent.run(AgentInput(query=backend_query, context=context))
    logger.info(f"Backend Developer: {backend_result.response}")
    
    # 6. Database Admin sets up the database
    db_query = "Set up a PostgreSQL database for the application"
    context = {"profile": db_profile}
    db_result = await database_admin_agent.run(AgentInput(query=db_query, context=context))
    logger.info(f"Database Admin: {db_result.response}")
    
    # 7. Network Admin configures the network
    network_query = "Configure the network for the application"
    context = {"profile": network_profile}
    network_result = await network_admin_agent.run(AgentInput(query=network_query, context=context))
    logger.info(f"Network Admin: {network_result.response}")


async def main():
    """Run all examples."""
    await specialization_example()
    await hierarchy_example()
    await combined_example()


if __name__ == "__main__":
    asyncio.run(main())
