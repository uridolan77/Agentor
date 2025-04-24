"""
Versioning utilities for the Agentor framework.

This module provides tools for version parsing, comparison, and compatibility checking
to support semantic versioning across the framework. It includes specialized components 
for tracking and validating agent versions, tool versions, and ensuring compatibility
between different components in the agent ecosystem.
"""

from typing import Optional, Tuple, List, Union, Dict, Any, Set, Callable, TypeVar
import re
import logging
from packaging import version
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class VersioningError(Exception):
    """Base exception for versioning-related errors."""
    pass


class IncompatibleVersionError(VersioningError):
    """Exception raised when incompatible versions are detected."""
    pass


class DeprecationLevel(Enum):
    """Levels of deprecation for versioned components."""
    
    NONE = 0  # Not deprecated
    WARNING = 1  # Deprecated with warning
    ERROR = 2  # Deprecated with error (raises exception)


class SemanticVersion:
    """A semantic version according to SemVer 2.0.0 specification."""

    def __init__(self, version_str: str):
        """Initialize the semantic version.

        Args:
            version_str: The version string (e.g., "1.2.3", "1.0.0-beta.1")
        
        Raises:
            ValueError: If the version string is invalid
        """
        try:
            self.version = version.parse(version_str)
            self._version_str = version_str
        except Exception as e:
            raise ValueError(f"Invalid version string '{version_str}': {str(e)}") from e
    
    @property
    def major(self) -> int:
        """Get the major version.
        
        Returns:
            The major version number
        """
        return self.version.major
    
    @property
    def minor(self) -> int:
        """Get the minor version.
        
        Returns:
            The minor version number
        """
        return self.version.minor
    
    @property
    def patch(self) -> int:
        """Get the patch version.
        
        Returns:
            The patch version number
        """
        return self.version.micro
    
    @property
    def prerelease(self) -> Optional[Tuple]:
        """Get the prerelease components.
        
        Returns:
            The prerelease components, or None if not a prerelease
        """
        return self.version.pre
    
    @property
    def build(self) -> Optional[Tuple]:
        """Get the build components.
        
        Returns:
            The build components, or None if no build metadata
        """
        return self.version.dev
        
    @property
    def is_prerelease(self) -> bool:
        """Check if this version is a prerelease.
        
        Returns:
            True if this is a prerelease version, False otherwise
        """
        return self.prerelease is not None
        
    @property
    def is_stable(self) -> bool:
        """Check if this version is stable (not prerelease and major > 0).
        
        Returns:
            True if this is a stable version, False otherwise
        """
        return not self.is_prerelease and self.major > 0
    
    def next_major(self) -> 'SemanticVersion':
        """Get the next major version.
        
        Returns:
            A new SemanticVersion with the major version incremented
        """
        return SemanticVersion(f"{self.major + 1}.0.0")
    
    def next_minor(self) -> 'SemanticVersion':
        """Get the next minor version.
        
        Returns:
            A new SemanticVersion with the minor version incremented
        """
        return SemanticVersion(f"{self.major}.{self.minor + 1}.0")
    
    def next_patch(self) -> 'SemanticVersion':
        """Get the next patch version.
        
        Returns:
            A new SemanticVersion with the patch version incremented
        """
        return SemanticVersion(f"{self.major}.{self.minor}.{self.patch + 1}")
    
    def __str__(self) -> str:
        """Get the version as a string.
        
        Returns:
            The version string
        """
        return self._version_str
    
    def __repr__(self) -> str:
        """Get the representation of the version.
        
        Returns:
            The string representation
        """
        return f"SemanticVersion('{self._version_str}')"
    
    def __eq__(self, other) -> bool:
        """Check if this version equals another version.
        
        Args:
            other: The other version to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if isinstance(other, str):
            other = SemanticVersion(other)
        elif not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return self.version == other.version
    
    def __lt__(self, other) -> bool:
        """Check if this version is less than another version.
        
        Args:
            other: The other version to compare with
            
        Returns:
            True if less than, False otherwise
        """
        if isinstance(other, str):
            other = SemanticVersion(other)
        elif not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return self.version < other.version
    
    def __gt__(self, other) -> bool:
        """Check if this version is greater than another version.
        
        Args:
            other: The other version to compare with
            
        Returns:
            True if greater than, False otherwise
        """
        if isinstance(other, str):
            other = SemanticVersion(other)
        elif not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return self.version > other.version
    
    def __le__(self, other) -> bool:
        """Check if this version is less than or equal to another version.
        
        Args:
            other: The other version to compare with
            
        Returns:
            True if less than or equal, False otherwise
        """
        if isinstance(other, str):
            other = SemanticVersion(other)
        elif not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return self.version <= other.version
    
    def __ge__(self, other) -> bool:
        """Check if this version is greater than or equal to another version.
        
        Args:
            other: The other version to compare with
            
        Returns:
            True if greater than or equal, False otherwise
        """
        if isinstance(other, str):
            other = SemanticVersion(other)
        elif not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return self.version >= other.version
        
    def __hash__(self) -> int:
        """Get the hash of this version.
        
        Returns:
            The hash value
        """
        return hash(str(self.version))
    
    def is_compatible_with(self, other) -> bool:
        """Check if this version is compatible with another version.
        
        According to semantic versioning rules, versions are compatible if:
        1. For 0.y.z versions, only exact matches are compatible
        2. For x.y.z where x > 0, versions with the same major version are compatible
        
        Args:
            other: The other version to check compatibility with
            
        Returns:
            True if compatible, False otherwise
        """
        if isinstance(other, str):
            other = SemanticVersion(other)
        elif not isinstance(other, SemanticVersion):
            return NotImplemented
        
        # For 0.y.z versions, only exact matches are compatible
        if self.major == 0 or other.major == 0:
            return self.version == other.version
        
        # For x.y.z where x > 0, versions with the same major version are compatible
        return self.major == other.major


class VersionConstraint:
    """A constraint for tool versions."""

    def __init__(
        self, 
        min_version: Optional[str] = None,
        max_version: Optional[str] = None,
        exact_version: Optional[str] = None,
        compatible_with: Optional[str] = None
    ):
        """Initialize the version constraint.
        
        Args:
            min_version: The minimum version (inclusive)
            max_version: The maximum version (exclusive)
            exact_version: The exact version required
            compatible_with: Version to be compatible with
            
        Raises:
            ValueError: If the constraint is invalid
        """
        if exact_version is not None and (min_version is not None or max_version is not None or compatible_with is not None):
            raise ValueError("Cannot specify exact_version with other constraints")
        
        if compatible_with is not None and (min_version is not None or max_version is not None):
            raise ValueError("Cannot specify compatible_with with min_version or max_version")
        
        self.min_version = SemanticVersion(min_version) if min_version else None
        self.max_version = SemanticVersion(max_version) if max_version else None
        self.exact_version = SemanticVersion(exact_version) if exact_version else None
        self.compatible_with = SemanticVersion(compatible_with) if compatible_with else None
    
    def is_satisfied_by(self, version_str: str) -> bool:
        """Check if a version satisfies this constraint.
        
        Args:
            version_str: The version to check
            
        Returns:
            True if the version satisfies the constraint, False otherwise
            
        Raises:
            ValueError: If the version string is invalid
        """
        version = SemanticVersion(version_str)
        
        # Check exact version
        if self.exact_version is not None:
            return version == self.exact_version
        
        # Check compatibility
        if self.compatible_with is not None:
            return version.is_compatible_with(self.compatible_with)
        
        # Check range
        min_satisfied = self.min_version is None or version >= self.min_version
        max_satisfied = self.max_version is None or version < self.max_version
        
        return min_satisfied and max_satisfied
    
    def __str__(self) -> str:
        """Get the string representation of this constraint.
        
        Returns:
            The constraint as a string
        """
        if self.exact_version is not None:
            return f"=={self.exact_version}"
        
        if self.compatible_with is not None:
            return f"~{self.compatible_with}"
        
        parts = []
        if self.min_version is not None:
            parts.append(f">={self.min_version}")
        if self.max_version is not None:
            parts.append(f"<{self.max_version}")
        
        return " and ".join(parts) or "any version"
        
    def __eq__(self, other) -> bool:
        """Check if this constraint equals another constraint.
        
        Args:
            other: The other constraint to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, VersionConstraint):
            return NotImplemented
            
        return (
            self.min_version == other.min_version and
            self.max_version == other.max_version and
            self.exact_version == other.exact_version and
            self.compatible_with == other.compatible_with
        )
        
    def __hash__(self) -> int:
        """Get the hash of this constraint.
        
        Returns:
            The hash value
        """
        return hash((
            str(self.min_version) if self.min_version else None,
            str(self.max_version) if self.max_version else None,
            str(self.exact_version) if self.exact_version else None,
            str(self.compatible_with) if self.compatible_with else None
        ))


class VersionRange:
    """A range of versions, defined as a set of constraints."""
    
    def __init__(self, constraints: List[VersionConstraint]):
        """Initialize the version range.
        
        Args:
            constraints: The constraints that define the range
        """
        self.constraints = constraints
    
    def is_satisfied_by(self, version_str: str) -> bool:
        """Check if a version satisfies this range.
        
        Args:
            version_str: The version to check
            
        Returns:
            True if the version satisfies all constraints, False otherwise
            
        Raises:
            ValueError: If the version string is invalid
        """
        if not self.constraints:
            return True  # No constraints means any version is accepted
        
        for constraint in self.constraints:
            if not constraint.is_satisfied_by(version_str):
                return False
        
        return True

    def get_best_matching_version(self, versions: List[str]) -> Optional[str]:
        """Get the best matching version from a list of versions.
        
        This selects the highest version that satisfies the constraints.
        
        Args:
            versions: The versions to choose from
            
        Returns:
            The best matching version, or None if no version satisfies the constraints
            
        Raises:
            ValueError: If any version string is invalid
        """
        if not versions:
            return None
        
        valid_versions = [v for v in versions if self.is_satisfied_by(v)]
        if not valid_versions:
            return None
        
        # Return the highest valid version
        return str(max(map(SemanticVersion, valid_versions)))
        
    def overlaps_with(self, other: 'VersionRange') -> bool:
        """Check if this version range overlaps with another.
        
        Args:
            other: The other version range
            
        Returns:
            True if there's at least one version that satisfies both ranges
        """
        # A proper implementation would need to solve for the intersection of the ranges,
        # but for simplicity, we'll use some common test versions
        test_versions = []
        
        # Add explicit versions from constraints
        for constraint in self.constraints + other.constraints:
            if constraint.exact_version:
                test_versions.append(str(constraint.exact_version))
            if constraint.compatible_with:
                test_versions.append(str(constraint.compatible_with))
            if constraint.min_version:
                test_versions.append(str(constraint.min_version))
            if constraint.max_version:
                test_versions.append(str(constraint.max_version))
                
        # Add intermediate versions for better coverage
        all_versions = set(test_versions)
        for v in test_versions:
            sem_v = SemanticVersion(v)
            all_versions.add(str(sem_v.next_patch()))
            all_versions.add(str(sem_v.next_minor()))
        
        # Check if any version satisfies both ranges
        for v in all_versions:
            if self.is_satisfied_by(v) and other.is_satisfied_by(v):
                return True
                
        return False
    
    def __str__(self) -> str:
        """Get the string representation of this range.
        
        Returns:
            The range as a string
        """
        if not self.constraints:
            return "any version"
            
        return " and ".join(str(c) for c in self.constraints)
        
    def __eq__(self, other) -> bool:
        """Check if this range equals another range.
        
        Args:
            other: The other range to compare with
            
        Returns:
            True if equal, False otherwise
        """
        if not isinstance(other, VersionRange):
            return NotImplemented
            
        return set(self.constraints) == set(other.constraints)
    
    @classmethod
    def parse(cls, spec: str) -> 'VersionRange':
        """Parse a version specification string.
        
        Supports formats like:
        - ">=1.0.0"
        - ">=1.0.0,<2.0.0"
        - "==1.2.3"
        - "~1.2.3" (compatible with 1.2.3)
        
        Args:
            spec: The specification string
            
        Returns:
            A VersionRange object
            
        Raises:
            ValueError: If the specification is invalid
        """
        constraints = []
        specs = spec.split(',')
        
        for s in specs:
            s = s.strip()
            if not s:
                continue
            
            if s.startswith('=='):
                constraints.append(VersionConstraint(exact_version=s[2:]))
            elif s.startswith('>='):
                constraints.append(VersionConstraint(min_version=s[2:]))
            elif s.startswith('>'):
                # Convert exclusive min to inclusive min by incrementing patch
                v = SemanticVersion(s[1:])
                new_ver = f"{v.major}.{v.minor}.{v.patch + 1}"
                constraints.append(VersionConstraint(min_version=new_ver))
            elif s.startswith('<='):
                # Convert inclusive max to exclusive max by incrementing patch
                v = SemanticVersion(s[2:])
                new_ver = f"{v.major}.{v.minor}.{v.patch + 1}"
                constraints.append(VersionConstraint(max_version=new_ver))
            elif s.startswith('<'):
                constraints.append(VersionConstraint(max_version=s[1:]))
            elif s.startswith('~'):
                constraints.append(VersionConstraint(compatible_with=s[1:]))
            else:
                # Assume exact version
                constraints.append(VersionConstraint(exact_version=s))
        
        return cls(constraints)


class ComponentVersion:
    """Version information for a component in the Agentor framework."""
    
    def __init__(
        self, 
        component_type: str,
        component_id: str,
        version: str,
        requires: Optional[Dict[str, str]] = None,
        provides: Optional[Dict[str, str]] = None,
        deprecated: bool = False,
        supported_until: Optional[str] = None,
        minimum_framework_version: Optional[str] = None
    ):
        """Initialize the component version.
        
        Args:
            component_type: The type of component (e.g., "agent", "tool", "memory")
            component_id: The ID of the component
            version: The version of the component
            requires: Component dependencies with version specifications
            provides: Features/capabilities provided by this component
            deprecated: Whether this component is deprecated
            supported_until: Version until which this component is supported
            minimum_framework_version: Minimum required framework version
            
        Raises:
            ValueError: If any version string is invalid
        """
        self.component_type = component_type
        self.component_id = component_id
        self.version = SemanticVersion(version)
        self.requires = requires or {}
        self.provides = provides or {}
        self.deprecated = deprecated
        self.supported_until = SemanticVersion(supported_until) if supported_until else None
        self.minimum_framework_version = SemanticVersion(minimum_framework_version) if minimum_framework_version else None
        
        # Parse dependencies
        self.parsed_requires: Dict[str, VersionRange] = {}
        for comp, ver_spec in self.requires.items():
            try:
                self.parsed_requires[comp] = VersionRange.parse(ver_spec)
            except ValueError as e:
                raise ValueError(f"Invalid dependency specification for {comp}: {str(e)}") from e
                
    def satisfies_dependency(self, dependency_id: str, version: str) -> bool:
        """Check if this component satisfies a dependency requirement.
        
        Args:
            dependency_id: The ID of the dependency
            version: The version of the dependency
            
        Returns:
            True if this component can satisfy the dependency, False otherwise
        """
        # Check if this component matches the dependency ID
        if self.component_id != dependency_id:
            return False
            
        # Check if this version satisfies the requirement
        try:
            version_range = VersionRange.parse(version)
            return version_range.is_satisfied_by(str(self.version))
        except ValueError:
            return False
            
    def is_compatible_with(self, other: 'ComponentVersion') -> bool:
        """Check if this component is compatible with another component.
        
        Args:
            other: The other component to check compatibility with
            
        Returns:
            True if compatible, False otherwise
        """
        # Different component types are always compatible
        if self.component_type != other.component_type:
            return True
            
        # Same component ID must have compatible versions
        if self.component_id == other.component_id:
            return self.version.is_compatible_with(other.version)
            
        # Check if this component depends on the other
        for dep_id, dep_range in self.parsed_requires.items():
            if dep_id == other.component_id:
                if not dep_range.is_satisfied_by(str(other.version)):
                    return False
                    
        # Check if the other component depends on this
        for dep_id, dep_range in other.parsed_requires.items():
            if dep_id == self.component_id:
                if not dep_range.is_satisfied_by(str(self.version)):
                    return False
                    
        return True
        
    def is_deprecated_for(self, framework_version: str) -> Optional[DeprecationLevel]:
        """Check if this component is deprecated for a specific framework version.
        
        Args:
            framework_version: The framework version to check against
            
        Returns:
            The deprecation level, or None if not deprecated
        """
        if not self.deprecated:
            return None
            
        if not self.supported_until:
            # Generic deprecation
            return DeprecationLevel.WARNING
            
        # Check if the framework version is beyond the supported version
        try:
            fv = SemanticVersion(framework_version)
            if fv > self.supported_until:
                return DeprecationLevel.ERROR
            else:
                return DeprecationLevel.WARNING
        except ValueError:
            # If we can't parse the framework version, assume warning
            return DeprecationLevel.WARNING
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation.
        
        Returns:
            Dictionary representation of this component version
        """
        result = {
            "component_type": self.component_type,
            "component_id": self.component_id,
            "version": str(self.version),
            "requires": self.requires,
            "provides": self.provides,
            "deprecated": self.deprecated
        }
        
        if self.supported_until:
            result["supported_until"] = str(self.supported_until)
            
        if self.minimum_framework_version:
            result["minimum_framework_version"] = str(self.minimum_framework_version)
            
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentVersion':
        """Create from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            A ComponentVersion instance
            
        Raises:
            ValueError: If the data is invalid
        """
        return cls(
            component_type=data["component_type"],
            component_id=data["component_id"],
            version=data["version"],
            requires=data.get("requires"),
            provides=data.get("provides"),
            deprecated=data.get("deprecated", False),
            supported_until=data.get("supported_until"),
            minimum_framework_version=data.get("minimum_framework_version")
        )
        
    def __str__(self) -> str:
        """Get a string representation.
        
        Returns:
            String representation
        """
        status = " (deprecated)" if self.deprecated else ""
        return f"{self.component_type}:{self.component_id}@{self.version}{status}"


class AgentVersion(ComponentVersion):
    """Version information for an agent in the Agentor framework."""
    
    def __init__(
        self,
        agent_id: str,
        version: str,
        requires: Optional[Dict[str, str]] = None,
        provides: Optional[Dict[str, str]] = None,
        deprecated: bool = False,
        supported_until: Optional[str] = None,
        minimum_framework_version: Optional[str] = None,
        compatible_tools: Optional[Dict[str, str]] = None,
        compatible_agents: Optional[Dict[str, str]] = None
    ):
        """Initialize the agent version.
        
        Args:
            agent_id: The ID of the agent
            version: The version of the agent
            requires: Component dependencies with version specifications
            provides: Features/capabilities provided by this agent
            deprecated: Whether this agent is deprecated
            supported_until: Version until which this agent is supported
            minimum_framework_version: Minimum required framework version
            compatible_tools: Tool dependencies with version specifications
            compatible_agents: Agent dependencies with version specifications
            
        Raises:
            ValueError: If any version string is invalid
        """
        super().__init__(
            component_type="agent",
            component_id=agent_id,
            version=version,
            requires=requires,
            provides=provides,
            deprecated=deprecated,
            supported_until=supported_until,
            minimum_framework_version=minimum_framework_version
        )
        
        self.compatible_tools = compatible_tools or {}
        self.compatible_agents = compatible_agents or {}
        
        # Parse tool dependencies
        self.parsed_tool_reqs: Dict[str, VersionRange] = {}
        for tool, ver_spec in self.compatible_tools.items():
            try:
                self.parsed_tool_reqs[tool] = VersionRange.parse(ver_spec)
            except ValueError as e:
                raise ValueError(f"Invalid tool dependency specification for {tool}: {str(e)}") from e
                
        # Parse agent dependencies
        self.parsed_agent_reqs: Dict[str, VersionRange] = {}
        for agent, ver_spec in self.compatible_agents.items():
            try:
                self.parsed_agent_reqs[agent] = VersionRange.parse(ver_spec)
            except ValueError as e:
                raise ValueError(f"Invalid agent dependency specification for {agent}: {str(e)}") from e
                
    def is_compatible_with_tool(self, tool_id: str, version: str) -> bool:
        """Check if this agent is compatible with a specific tool version.
        
        Args:
            tool_id: The ID of the tool
            version: The version of the tool
            
        Returns:
            True if compatible, False otherwise
        """
        if tool_id in self.parsed_tool_reqs:
            try:
                return self.parsed_tool_reqs[tool_id].is_satisfied_by(version)
            except ValueError:
                return False
        
        # If no specific compatibility is defined, assume compatible
        return True
        
    def is_compatible_with_agent(self, agent_id: str, version: str) -> bool:
        """Check if this agent is compatible with another agent version.
        
        Args:
            agent_id: The ID of the other agent
            version: The version of the other agent
            
        Returns:
            True if compatible, False otherwise
        """
        if agent_id in self.parsed_agent_reqs:
            try:
                return self.parsed_agent_reqs[agent_id].is_satisfied_by(version)
            except ValueError:
                return False
        
        # If no specific compatibility is defined, assume compatible
        return True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation.
        
        Returns:
            Dictionary representation of this agent version
        """
        result = super().to_dict()
        result["compatible_tools"] = self.compatible_tools
        result["compatible_agents"] = self.compatible_agents
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentVersion':
        """Create from a dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            An AgentVersion instance
            
        Raises:
            ValueError: If the data is invalid
        """
        return cls(
            agent_id=data["component_id"],
            version=data["version"],
            requires=data.get("requires"),
            provides=data.get("provides"),
            deprecated=data.get("deprecated", False),
            supported_until=data.get("supported_until"),
            minimum_framework_version=data.get("minimum_framework_version"),
            compatible_tools=data.get("compatible_tools"),
            compatible_agents=data.get("compatible_agents")
        )


class VersionRegistry:
    """Registry of component versions in the Agentor framework."""
    
    def __init__(self):
        """Initialize the version registry."""
        self.components: Dict[str, Dict[str, ComponentVersion]] = {}
        self.framework_version: Optional[SemanticVersion] = None
        
    def register_component(self, component: ComponentVersion) -> None:
        """Register a component version.
        
        Args:
            component: The component version to register
            
        Raises:
            ValueError: If a component with the same ID and version already exists
        """
        component_type = component.component_type
        component_id = component.component_id
        
        if component_type not in self.components:
            self.components[component_type] = {}
            
        component_versions = self.components[component_type]
        key = f"{component_id}@{component.version}"
        
        if key in component_versions:
            raise ValueError(f"Component {key} is already registered")
            
        component_versions[key] = component
        logger.debug(f"Registered component {component}")
        
    def set_framework_version(self, version: str) -> None:
        """Set the framework version.
        
        Args:
            version: The framework version
            
        Raises:
            ValueError: If the version string is invalid
        """
        self.framework_version = SemanticVersion(version)
        logger.info(f"Set framework version to {version}")
        
    def get_component(
        self, 
        component_type: str, 
        component_id: str, 
        version: Optional[str] = None
    ) -> Optional[ComponentVersion]:
        """Get a component by type, ID, and optional version.
        
        Args:
            component_type: The type of component
            component_id: The ID of the component
            version: The version of the component, or None for the latest version
            
        Returns:
            The component version, or None if not found
        """
        if component_type not in self.components:
            return None
            
        component_versions = self.components[component_type]
        matching_components = []
        
        for key, component in component_versions.items():
            if component.component_id == component_id:
                if version is None or str(component.version) == version:
                    matching_components.append(component)
                    
        if not matching_components:
            return None
            
        if version is not None and len(matching_components) == 1:
            return matching_components[0]
            
        # Return the latest version
        return max(matching_components, key=lambda c: c.version)
        
    def get_components_by_type(self, component_type: str) -> List[ComponentVersion]:
        """Get all components of a specific type.
        
        Args:
            component_type: The type of component
            
        Returns:
            List of component versions
        """
        if component_type not in self.components:
            return []
            
        return list(self.components[component_type].values())
        
    def get_compatible_components(
        self, 
        component: ComponentVersion
    ) -> Dict[str, List[ComponentVersion]]:
        """Get all components compatible with a given component.
        
        Args:
            component: The component to check compatibility with
            
        Returns:
            Dictionary mapping component types to lists of compatible components
        """
        result: Dict[str, List[ComponentVersion]] = {}
        
        for ctype, components in self.components.items():
            compatible = []
            
            for c in components.values():
                if component.is_compatible_with(c):
                    compatible.append(c)
                    
            if compatible:
                result[ctype] = compatible
                
        return result
        
    def check_compatibility(
        self, 
        components: List[ComponentVersion]
    ) -> Dict[Tuple[str, str], str]:
        """Check compatibility between a list of components.
        
        Args:
            components: The components to check
            
        Returns:
            Dictionary mapping (component1, component2) to error message,
            empty if all components are compatible
        """
        errors: Dict[Tuple[str, str], str] = {}
        
        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
                if not c1.is_compatible_with(c2):
                    key = (str(c1), str(c2))
                    errors[key] = f"{c1} is not compatible with {c2}"
                    
        return errors
        
    def get_deprecated_components(self) -> List[Tuple[ComponentVersion, DeprecationLevel]]:
        """Get all deprecated components.
        
        Returns:
            List of (component, deprecation_level) tuples
        """
        if self.framework_version is None:
            return []
            
        result = []
        for components in self.components.values():
            for component in components.values():
                level = component.is_deprecated_for(str(self.framework_version))
                if level:
                    result.append((component, level))
                    
        return result
        
    def clear(self) -> None:
        """Clear the registry."""
        self.components.clear()
        logger.debug("Cleared version registry")


# Global registry instance
_registry = VersionRegistry()

def get_version_registry() -> VersionRegistry:
    """Get the global version registry.
    
    Returns:
        The global version registry instance
    """
    return _registry


def versioned(
    component_type: str,
    component_id: str,
    version: str,
    requires: Optional[Dict[str, str]] = None,
    provides: Optional[Dict[str, str]] = None,
    deprecated: bool = False,
    supported_until: Optional[str] = None,
    minimum_framework_version: Optional[str] = None
) -> Callable[[T], T]:
    """Decorator to mark a class or function with version information.
    
    Args:
        component_type: The type of component
        component_id: The ID of the component
        version: The version of the component
        requires: Component dependencies with version specifications
        provides: Features/capabilities provided by this component
        deprecated: Whether this component is deprecated
        supported_until: Version until which this component is supported
        minimum_framework_version: Minimum required framework version
        
    Returns:
        Decorator function
        
    Raises:
        ValueError: If any version string is invalid
    """
    def decorator(func_or_class: T) -> T:
        # Create component version
        component = ComponentVersion(
            component_type=component_type,
            component_id=component_id,
            version=version,
            requires=requires,
            provides=provides,
            deprecated=deprecated,
            supported_until=supported_until,
            minimum_framework_version=minimum_framework_version
        )
        
        # Register component in the global registry
        try:
            get_version_registry().register_component(component)
        except ValueError as e:
            logger.warning(f"Failed to register component: {str(e)}")
            
        # Attach version information to the function or class
        func_or_class._version_info = component
        
        # For functions, add a wrapper to check deprecation
        if callable(func_or_class) and not isinstance(func_or_class, type):
            @wraps(func_or_class)
            def wrapper(*args, **kwargs):
                registry = get_version_registry()
                if registry.framework_version and component.deprecated:
                    level = component.is_deprecated_for(str(registry.framework_version))
                    
                    if level == DeprecationLevel.WARNING:
                        logger.warning(f"Deprecated: {component} is deprecated")
                    elif level == DeprecationLevel.ERROR:
                        raise IncompatibleVersionError(f"{component} is deprecated and no longer supported")
                        
                return func_or_class(*args, **kwargs)
                
            return wrapper  # type: ignore
            
        return func_or_class
        
    return decorator


def versioned_agent(
    agent_id: str,
    version: str,
    requires: Optional[Dict[str, str]] = None,
    provides: Optional[Dict[str, str]] = None,
    deprecated: bool = False,
    supported_until: Optional[str] = None,
    minimum_framework_version: Optional[str] = None,
    compatible_tools: Optional[Dict[str, str]] = None,
    compatible_agents: Optional[Dict[str, str]] = None
) -> Callable[[T], T]:
    """Decorator to mark an agent class with version information.
    
    Args:
        agent_id: The ID of the agent
        version: The version of the agent
        requires: Component dependencies with version specifications
        provides: Features/capabilities provided by this agent
        deprecated: Whether this agent is deprecated
        supported_until: Version until which this agent is supported
        minimum_framework_version: Minimum required framework version
        compatible_tools: Tool dependencies with version specifications
        compatible_agents: Agent dependencies with version specifications
        
    Returns:
        Decorator function
        
    Raises:
        ValueError: If any version string is invalid
    """
    def decorator(agent_class: T) -> T:
        # Create agent version
        agent_version = AgentVersion(
            agent_id=agent_id,
            version=version,
            requires=requires,
            provides=provides,
            deprecated=deprecated,
            supported_until=supported_until,
            minimum_framework_version=minimum_framework_version,
            compatible_tools=compatible_tools,
            compatible_agents=compatible_agents
        )
        
        # Register agent in the global registry
        try:
            get_version_registry().register_component(agent_version)
        except ValueError as e:
            logger.warning(f"Failed to register agent: {str(e)}")
            
        # Attach version information to the class
        agent_class._version_info = agent_version  # type: ignore
        
        return agent_class
        
    return decorator


def validate_version_compatibility(component1: Any, component2: Any) -> bool:
    """Validate that two versioned components are compatible.
    
    Args:
        component1: The first component
        component2: The second component
        
    Returns:
        True if compatible, False otherwise
        
    Raises:
        ValueError: If either component is not versioned
    """
    if not hasattr(component1, '_version_info') or not hasattr(component2, '_version_info'):
        raise ValueError("Both components must be versioned")
        
    v1 = component1._version_info
    v2 = component2._version_info
    
    if not isinstance(v1, ComponentVersion) or not isinstance(v2, ComponentVersion):
        raise ValueError("Invalid version information")
        
    return v1.is_compatible_with(v2)


def check_compatibility(*components: Any) -> Dict[Tuple[str, str], str]:
    """Check compatibility between multiple versioned components.
    
    Args:
        components: The components to check
        
    Returns:
        Dictionary mapping (component1, component2) to error message,
        empty if all components are compatible
        
    Raises:
        ValueError: If any component is not versioned
    """
    version_infos = []
    
    for comp in components:
        if not hasattr(comp, '_version_info'):
            raise ValueError(f"Component {comp} is not versioned")
            
        v = comp._version_info
        if not isinstance(v, ComponentVersion):
            raise ValueError(f"Invalid version information for {comp}")
            
        version_infos.append(v)
        
    return get_version_registry().check_compatibility(version_infos)


def get_version_info(component: Any) -> Optional[ComponentVersion]:
    """Get version information for a component.
    
    Args:
        component: The component to get version information for
        
    Returns:
        Component version information, or None if not versioned
    """
    if hasattr(component, '_version_info'):
        v = component._version_info
        if isinstance(v, ComponentVersion):
            return v
            
    return None
