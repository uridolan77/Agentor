"""
Components module for the Agentor framework.

This module provides various components for building agent-based systems, including:
- Memory components for storing and retrieving information
- Environment components for agent-environment interaction
- Coordination components for multi-agent systems
- Learning components for agent learning and adaptation
"""

# Import submodules to make them available through the components package
from agentor.components import memory
from agentor.components import environments
from agentor.components import coordination
from agentor.components import learning

__all__ = [
    'memory',
    'environments',
    'coordination',
    'learning',
]
