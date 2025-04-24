"""
Coordination module for multi-agent systems.

This module provides various coordination mechanisms for multi-agent systems,
including coordination patterns, agent specialization, consensus algorithms,
and role-based coordination.
"""

from agentor.components.coordination.base import (
    AgentMessage,
    AgentRegistry,
    CoordinationContext
)
from agentor.components.coordination.patterns import (
    CoordinationPattern,
    MasterSlavePattern,
    PeerToPeerPattern,
    BlackboardPattern,
    ContractNetProtocol,
    MarketBasedCoordination
)
from agentor.components.coordination.specialization import (
    Skill,
    Knowledge,
    AgentProfile,
    SpecializationManager,
    HierarchyNode,
    AgentHierarchy
)
from agentor.components.coordination.consensus import (
    ConsensusAlgorithm,
    VotingConsensus,
    WeightedVotingConsensus,
    BordaCountConsensus,
    SimplePaxosConsensus,
    SimpleRaftConsensus
)
from agentor.components.coordination.roles import (
    Role,
    RoleAssignment,
    RoleManager,
    RoleTransitionRule,
    RoleTransitionManager,
    Team,
    TeamManager,
    RoleBasedCoordinator
)

__all__ = [
    # Base classes
    'AgentMessage',
    'AgentRegistry',
    'CoordinationContext',
    
    # Coordination patterns
    'CoordinationPattern',
    'MasterSlavePattern',
    'PeerToPeerPattern',
    'BlackboardPattern',
    'ContractNetProtocol',
    'MarketBasedCoordination',
    
    # Agent specialization
    'Skill',
    'Knowledge',
    'AgentProfile',
    'SpecializationManager',
    'HierarchyNode',
    'AgentHierarchy',
    
    # Consensus algorithms
    'ConsensusAlgorithm',
    'VotingConsensus',
    'WeightedVotingConsensus',
    'BordaCountConsensus',
    'SimplePaxosConsensus',
    'SimpleRaftConsensus',
    
    # Role-based coordination
    'Role',
    'RoleAssignment',
    'RoleManager',
    'RoleTransitionRule',
    'RoleTransitionManager',
    'Team',
    'TeamManager',
    'RoleBasedCoordinator'
]
