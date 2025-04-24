"""
Agent specialization and hierarchy mechanisms.

This module provides mechanisms for agent specialization and hierarchy,
including skill-based, knowledge-based, and role-based specialization.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Tuple, Union
from abc import ABC, abstractmethod

from agentor.core.interfaces.agent import IAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class Skill:
    """Representation of an agent skill."""
    
    def __init__(self, name: str, description: str, proficiency: float = 0.0):
        """Initialize the skill.
        
        Args:
            name: The name of the skill
            description: The description of the skill
            proficiency: The proficiency level (0.0 to 1.0)
        """
        self.name = name
        self.description = description
        self.proficiency = max(0.0, min(1.0, proficiency))
    
    def __str__(self) -> str:
        """Get a string representation of the skill.
        
        Returns:
            String representation
        """
        return f"{self.name} (Proficiency: {self.proficiency:.2f})"


class Knowledge:
    """Representation of agent knowledge."""
    
    def __init__(self, domain: str, description: str, confidence: float = 0.0):
        """Initialize the knowledge.
        
        Args:
            domain: The knowledge domain
            description: The description of the knowledge
            confidence: The confidence level (0.0 to 1.0)
        """
        self.domain = domain
        self.description = description
        self.confidence = max(0.0, min(1.0, confidence))
    
    def __str__(self) -> str:
        """Get a string representation of the knowledge.
        
        Returns:
            String representation
        """
        return f"{self.domain} (Confidence: {self.confidence:.2f})"


class AgentProfile:
    """Profile of an agent's capabilities."""
    
    def __init__(self, agent: IAgent):
        """Initialize the agent profile.
        
        Args:
            agent: The agent
        """
        self.agent = agent
        self.skills: Dict[str, Skill] = {}
        self.knowledge: Dict[str, Knowledge] = {}
        self.roles: Set[str] = set()
    
    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the profile.
        
        Args:
            skill: The skill to add
        """
        self.skills[skill.name] = skill
        logger.info(f"Added skill {skill.name} to agent {self.agent.name}")
    
    def remove_skill(self, skill_name: str) -> None:
        """Remove a skill from the profile.
        
        Args:
            skill_name: The name of the skill to remove
        """
        if skill_name in self.skills:
            del self.skills[skill_name]
            logger.info(f"Removed skill {skill_name} from agent {self.agent.name}")
    
    def add_knowledge(self, knowledge: Knowledge) -> None:
        """Add knowledge to the profile.
        
        Args:
            knowledge: The knowledge to add
        """
        self.knowledge[knowledge.domain] = knowledge
        logger.info(f"Added knowledge {knowledge.domain} to agent {self.agent.name}")
    
    def remove_knowledge(self, domain: str) -> None:
        """Remove knowledge from the profile.
        
        Args:
            domain: The domain of the knowledge to remove
        """
        if domain in self.knowledge:
            del self.knowledge[domain]
            logger.info(f"Removed knowledge {domain} from agent {self.agent.name}")
    
    def add_role(self, role: str) -> None:
        """Add a role to the profile.
        
        Args:
            role: The role to add
        """
        self.roles.add(role)
        logger.info(f"Added role {role} to agent {self.agent.name}")
    
    def remove_role(self, role: str) -> None:
        """Remove a role from the profile.
        
        Args:
            role: The role to remove
        """
        if role in self.roles:
            self.roles.remove(role)
            logger.info(f"Removed role {role} from agent {self.agent.name}")
    
    def has_skill(self, skill_name: str, min_proficiency: float = 0.0) -> bool:
        """Check if the agent has a skill.
        
        Args:
            skill_name: The name of the skill
            min_proficiency: The minimum proficiency level required
            
        Returns:
            True if the agent has the skill with sufficient proficiency
        """
        return (
            skill_name in self.skills and
            self.skills[skill_name].proficiency >= min_proficiency
        )
    
    def has_knowledge(self, domain: str, min_confidence: float = 0.0) -> bool:
        """Check if the agent has knowledge in a domain.
        
        Args:
            domain: The knowledge domain
            min_confidence: The minimum confidence level required
            
        Returns:
            True if the agent has the knowledge with sufficient confidence
        """
        return (
            domain in self.knowledge and
            self.knowledge[domain].confidence >= min_confidence
        )
    
    def has_role(self, role: str) -> bool:
        """Check if the agent has a role.
        
        Args:
            role: The role to check
            
        Returns:
            True if the agent has the role
        """
        return role in self.roles
    
    def get_skill_proficiency(self, skill_name: str) -> float:
        """Get the proficiency level for a skill.
        
        Args:
            skill_name: The name of the skill
            
        Returns:
            The proficiency level, or 0.0 if the agent doesn't have the skill
        """
        return self.skills.get(skill_name, Skill(skill_name, "", 0.0)).proficiency
    
    def get_knowledge_confidence(self, domain: str) -> float:
        """Get the confidence level for knowledge in a domain.
        
        Args:
            domain: The knowledge domain
            
        Returns:
            The confidence level, or 0.0 if the agent doesn't have the knowledge
        """
        return self.knowledge.get(domain, Knowledge(domain, "", 0.0)).confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary.
        
        Returns:
            Dictionary representation of the profile
        """
        return {
            "agent_name": self.agent.name,
            "skills": {
                name: {
                    "description": skill.description,
                    "proficiency": skill.proficiency
                }
                for name, skill in self.skills.items()
            },
            "knowledge": {
                domain: {
                    "description": knowledge.description,
                    "confidence": knowledge.confidence
                }
                for domain, knowledge in self.knowledge.items()
            },
            "roles": list(self.roles)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent: IAgent) -> 'AgentProfile':
        """Create a profile from a dictionary.
        
        Args:
            data: The dictionary data
            agent: The agent
            
        Returns:
            The created profile
        """
        profile = cls(agent)
        
        # Add skills
        for name, skill_data in data.get("skills", {}).items():
            profile.add_skill(Skill(
                name=name,
                description=skill_data.get("description", ""),
                proficiency=skill_data.get("proficiency", 0.0)
            ))
        
        # Add knowledge
        for domain, knowledge_data in data.get("knowledge", {}).items():
            profile.add_knowledge(Knowledge(
                domain=domain,
                description=knowledge_data.get("description", ""),
                confidence=knowledge_data.get("confidence", 0.0)
            ))
        
        # Add roles
        for role in data.get("roles", []):
            profile.add_role(role)
        
        return profile


class SpecializationManager:
    """Manager for agent specialization."""
    
    def __init__(self):
        """Initialize the specialization manager."""
        self.profiles: Dict[str, AgentProfile] = {}
    
    def add_agent(self, agent: IAgent) -> None:
        """Add an agent to the manager.
        
        Args:
            agent: The agent to add
        """
        if agent.name not in self.profiles:
            self.profiles[agent.name] = AgentProfile(agent)
            logger.info(f"Added agent {agent.name} to specialization manager")
    
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the manager.
        
        Args:
            agent_name: The name of the agent to remove
        """
        if agent_name in self.profiles:
            del self.profiles[agent_name]
            logger.info(f"Removed agent {agent_name} from specialization manager")
    
    def get_profile(self, agent_name: str) -> Optional[AgentProfile]:
        """Get an agent's profile.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The agent's profile, or None if not found
        """
        return self.profiles.get(agent_name)
    
    def find_agents_with_skill(self, skill_name: str, min_proficiency: float = 0.0) -> List[str]:
        """Find agents with a specific skill.
        
        Args:
            skill_name: The name of the skill
            min_proficiency: The minimum proficiency level required
            
        Returns:
            List of agent names
        """
        return [
            agent_name
            for agent_name, profile in self.profiles.items()
            if profile.has_skill(skill_name, min_proficiency)
        ]
    
    def find_agents_with_knowledge(self, domain: str, min_confidence: float = 0.0) -> List[str]:
        """Find agents with knowledge in a specific domain.
        
        Args:
            domain: The knowledge domain
            min_confidence: The minimum confidence level required
            
        Returns:
            List of agent names
        """
        return [
            agent_name
            for agent_name, profile in self.profiles.items()
            if profile.has_knowledge(domain, min_confidence)
        ]
    
    def find_agents_with_role(self, role: str) -> List[str]:
        """Find agents with a specific role.
        
        Args:
            role: The role to find
            
        Returns:
            List of agent names
        """
        return [
            agent_name
            for agent_name, profile in self.profiles.items()
            if profile.has_role(role)
        ]
    
    def find_best_agent_for_skill(self, skill_name: str) -> Optional[str]:
        """Find the best agent for a specific skill.
        
        Args:
            skill_name: The name of the skill
            
        Returns:
            The name of the best agent, or None if no agent has the skill
        """
        agents = [
            (agent_name, profile.get_skill_proficiency(skill_name))
            for agent_name, profile in self.profiles.items()
            if profile.has_skill(skill_name)
        ]
        
        if not agents:
            return None
        
        return max(agents, key=lambda x: x[1])[0]
    
    def find_best_agent_for_knowledge(self, domain: str) -> Optional[str]:
        """Find the best agent for knowledge in a specific domain.
        
        Args:
            domain: The knowledge domain
            
        Returns:
            The name of the best agent, or None if no agent has the knowledge
        """
        agents = [
            (agent_name, profile.get_knowledge_confidence(domain))
            for agent_name, profile in self.profiles.items()
            if profile.has_knowledge(domain)
        ]
        
        if not agents:
            return None
        
        return max(agents, key=lambda x: x[1])[0]


class HierarchyNode:
    """Node in an agent hierarchy."""
    
    def __init__(self, agent: IAgent):
        """Initialize the hierarchy node.
        
        Args:
            agent: The agent
        """
        self.agent = agent
        self.parent: Optional[HierarchyNode] = None
        self.children: List[HierarchyNode] = []
    
    def add_child(self, child: 'HierarchyNode') -> None:
        """Add a child node.
        
        Args:
            child: The child node to add
        """
        child.parent = self
        self.children.append(child)
        logger.info(f"Added {child.agent.name} as child of {self.agent.name}")
    
    def remove_child(self, child: 'HierarchyNode') -> None:
        """Remove a child node.
        
        Args:
            child: The child node to remove
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            logger.info(f"Removed {child.agent.name} as child of {self.agent.name}")
    
    def get_ancestors(self) -> List['HierarchyNode']:
        """Get all ancestors of this node.
        
        Returns:
            List of ancestor nodes
        """
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self) -> List['HierarchyNode']:
        """Get all descendants of this node.
        
        Returns:
            List of descendant nodes
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def get_siblings(self) -> List['HierarchyNode']:
        """Get all siblings of this node.
        
        Returns:
            List of sibling nodes
        """
        if not self.parent:
            return []
        
        return [child for child in self.parent.children if child != self]
    
    def is_ancestor_of(self, node: 'HierarchyNode') -> bool:
        """Check if this node is an ancestor of another node.
        
        Args:
            node: The node to check
            
        Returns:
            True if this node is an ancestor of the given node
        """
        return self in node.get_ancestors()
    
    def is_descendant_of(self, node: 'HierarchyNode') -> bool:
        """Check if this node is a descendant of another node.
        
        Args:
            node: The node to check
            
        Returns:
            True if this node is a descendant of the given node
        """
        return node in self.get_ancestors()
    
    def get_level(self) -> int:
        """Get the level of this node in the hierarchy.
        
        Returns:
            The level (0 for root, 1 for children of root, etc.)
        """
        level = 0
        current = self.parent
        while current:
            level += 1
            current = current.parent
        return level


class AgentHierarchy:
    """Hierarchy of agents."""
    
    def __init__(self, root_agent: Optional[IAgent] = None):
        """Initialize the agent hierarchy.
        
        Args:
            root_agent: The root agent
        """
        self.nodes: Dict[str, HierarchyNode] = {}
        self.root: Optional[HierarchyNode] = None
        
        if root_agent:
            self.set_root(root_agent)
    
    def set_root(self, agent: IAgent) -> None:
        """Set the root agent.
        
        Args:
            agent: The agent to set as root
        """
        if agent.name in self.nodes:
            node = self.nodes[agent.name]
            
            # Remove from parent if it has one
            if node.parent:
                node.parent.remove_child(node)
            
            self.root = node
        else:
            self.root = HierarchyNode(agent)
            self.nodes[agent.name] = self.root
        
        logger.info(f"Set {agent.name} as root of hierarchy")
    
    def add_agent(self, agent: IAgent, parent_name: Optional[str] = None) -> None:
        """Add an agent to the hierarchy.
        
        Args:
            agent: The agent to add
            parent_name: The name of the parent agent, or None to add as root
        """
        if agent.name in self.nodes:
            logger.warning(f"Agent {agent.name} already in hierarchy")
            return
        
        node = HierarchyNode(agent)
        self.nodes[agent.name] = node
        
        if parent_name is None:
            # Add as root if no root exists
            if self.root is None:
                self.root = node
                logger.info(f"Set {agent.name} as root of hierarchy")
            else:
                logger.warning(f"Cannot add {agent.name} as root, root already exists")
        else:
            # Add as child of parent
            parent_node = self.nodes.get(parent_name)
            if parent_node:
                parent_node.add_child(node)
            else:
                logger.warning(f"Parent {parent_name} not found")
    
    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the hierarchy.
        
        Args:
            agent_name: The name of the agent to remove
        """
        if agent_name not in self.nodes:
            logger.warning(f"Agent {agent_name} not in hierarchy")
            return
        
        node = self.nodes[agent_name]
        
        # Remove from parent if it has one
        if node.parent:
            node.parent.remove_child(node)
        
        # Move children to parent
        if node.parent:
            for child in node.children[:]:  # Copy to avoid modification during iteration
                node.remove_child(child)
                node.parent.add_child(child)
        
        # If this is the root, set a new root
        if node == self.root:
            if node.children:
                self.root = node.children[0]
                
                # Remove from parent (which is now itself)
                self.root.parent = None
                
                # Add siblings as children
                for child in node.children[1:]:
                    child.parent = None  # Remove old parent
                    self.root.add_child(child)
            else:
                self.root = None
        
        # Remove from nodes
        del self.nodes[agent_name]
        logger.info(f"Removed {agent_name} from hierarchy")
    
    def get_node(self, agent_name: str) -> Optional[HierarchyNode]:
        """Get a node by agent name.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The node, or None if not found
        """
        return self.nodes.get(agent_name)
    
    def get_parent(self, agent_name: str) -> Optional[IAgent]:
        """Get the parent of an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The parent agent, or None if the agent has no parent
        """
        node = self.nodes.get(agent_name)
        if node and node.parent:
            return node.parent.agent
        return None
    
    def get_children(self, agent_name: str) -> List[IAgent]:
        """Get the children of an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of child agents
        """
        node = self.nodes.get(agent_name)
        if node:
            return [child.agent for child in node.children]
        return []
    
    def get_ancestors(self, agent_name: str) -> List[IAgent]:
        """Get the ancestors of an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of ancestor agents
        """
        node = self.nodes.get(agent_name)
        if node:
            return [ancestor.agent for ancestor in node.get_ancestors()]
        return []
    
    def get_descendants(self, agent_name: str) -> List[IAgent]:
        """Get the descendants of an agent.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of descendant agents
        """
        node = self.nodes.get(agent_name)
        if node:
            return [descendant.agent for descendant in node.get_descendants()]
        return []
    
    def get_level(self, agent_name: str) -> int:
        """Get the level of an agent in the hierarchy.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            The level (0 for root, 1 for children of root, etc.)
        """
        node = self.nodes.get(agent_name)
        if node:
            return node.get_level()
        return -1
    
    def is_ancestor(self, ancestor_name: str, descendant_name: str) -> bool:
        """Check if an agent is an ancestor of another agent.
        
        Args:
            ancestor_name: The name of the potential ancestor
            descendant_name: The name of the potential descendant
            
        Returns:
            True if the first agent is an ancestor of the second
        """
        ancestor_node = self.nodes.get(ancestor_name)
        descendant_node = self.nodes.get(descendant_name)
        
        if ancestor_node and descendant_node:
            return ancestor_node.is_ancestor_of(descendant_node)
        
        return False
    
    def get_all_agents(self) -> List[IAgent]:
        """Get all agents in the hierarchy.
        
        Returns:
            List of all agents
        """
        return [node.agent for node in self.nodes.values()]
    
    def get_agents_at_level(self, level: int) -> List[IAgent]:
        """Get all agents at a specific level in the hierarchy.
        
        Args:
            level: The level (0 for root, 1 for children of root, etc.)
            
        Returns:
            List of agents at the specified level
        """
        return [
            node.agent
            for node in self.nodes.values()
            if node.get_level() == level
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the hierarchy to a dictionary.
        
        Returns:
            Dictionary representation of the hierarchy
        """
        def node_to_dict(node: HierarchyNode) -> Dict[str, Any]:
            return {
                "agent_name": node.agent.name,
                "children": [node_to_dict(child) for child in node.children]
            }
        
        if self.root:
            return node_to_dict(self.root)
        return {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], agents: Dict[str, IAgent]) -> 'AgentHierarchy':
        """Create a hierarchy from a dictionary.
        
        Args:
            data: The dictionary data
            agents: Dictionary mapping agent names to agents
            
        Returns:
            The created hierarchy
        """
        hierarchy = cls()
        
        def add_node(node_data: Dict[str, Any], parent_name: Optional[str] = None) -> None:
            agent_name = node_data.get("agent_name")
            if agent_name and agent_name in agents:
                agent = agents[agent_name]
                hierarchy.add_agent(agent, parent_name)
                
                for child_data in node_data.get("children", []):
                    add_node(child_data, agent_name)
        
        add_node(data)
        return hierarchy
