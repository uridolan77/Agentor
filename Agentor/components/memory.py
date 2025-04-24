"""Memory module for the Agentor framework.

This module re-exports the base memory classes from the memory submodule.
For backward compatibility, this module should be kept, but new code should
use the memory submodule directly.
"""

# Re-export the base memory classes from the memory submodule
from agentor.components.memory.base import Memory, SimpleMemory, VectorMemory