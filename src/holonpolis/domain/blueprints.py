"""Blueprint and Species definitions - the DNA of Holons."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvolutionStrategy(Enum):
    """How this species handles evolution requests."""
    CONSERVATIVE = "conservative"  # Requires strict validation
    BALANCED = "balanced"  # Standard red-green-verify
    AGGRESSIVE = "aggressive"  # Fast iteration, higher risk


@dataclass
class Boundary:
    """Runtime boundaries for a Holon (security sandbox)."""

    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    max_episodes_per_hour: int = 100
    max_tokens_per_episode: int = 10000
    allow_file_write: bool = False
    allow_network: bool = False
    allow_subprocess: bool = False
    workspace_path: Optional[str] = None


@dataclass
class EvolutionPolicy:
    """Policy for evolving new skills."""

    strategy: EvolutionStrategy = EvolutionStrategy.BALANCED
    auto_promote_to_global: bool = False  # Requires Genesis approval
    require_tests: bool = True
    max_evolution_attempts: int = 3
    pytest_timeout_seconds: int = 30
    allowed_imports: List[str] = field(default_factory=lambda: ["typing", "re", "json"])
    denied_functions: List[str] = field(default_factory=lambda: ["eval", "exec", "compile"])


@dataclass
class Species:
    """Species defines the archetype/template for Holons."""

    species_id: str
    name: str
    description: str
    system_prompt_template: str
    default_boundary: Boundary
    default_evolution_policy: EvolutionPolicy
    initial_tools: List[str] = field(default_factory=list)
    parent_species: Optional[str] = None


@dataclass
class Blueprint:
    """Blueprint is the instantiation recipe for a specific Holon.

    Created by Genesis when spawning a new Holon.
    """

    blueprint_id: str
    holon_id: str
    species_id: str
    name: str
    purpose: str  # Why this Holon was created
    boundary: Boundary
    evolution_policy: EvolutionPolicy
    initial_memory_tags: List[str] = field(default_factory=list)
    parent_blueprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage."""
        return {
            "blueprint_id": self.blueprint_id,
            "holon_id": self.holon_id,
            "species_id": self.species_id,
            "name": self.name,
            "purpose": self.purpose,
            "boundary": {
                "allowed_tools": self.boundary.allowed_tools,
                "denied_tools": self.boundary.denied_tools,
                "max_episodes_per_hour": self.boundary.max_episodes_per_hour,
                "max_tokens_per_episode": self.boundary.max_tokens_per_episode,
                "allow_file_write": self.boundary.allow_file_write,
                "allow_network": self.boundary.allow_network,
                "allow_subprocess": self.boundary.allow_subprocess,
                "workspace_path": self.boundary.workspace_path,
            },
            "evolution_policy": {
                "strategy": self.evolution_policy.strategy.value,
                "auto_promote_to_global": self.evolution_policy.auto_promote_to_global,
                "require_tests": self.evolution_policy.require_tests,
                "max_evolution_attempts": self.evolution_policy.max_evolution_attempts,
                "pytest_timeout_seconds": self.evolution_policy.pytest_timeout_seconds,
                "allowed_imports": self.evolution_policy.allowed_imports,
                "denied_functions": self.evolution_policy.denied_functions,
            },
            "initial_memory_tags": self.initial_memory_tags,
            "parent_blueprint": self.parent_blueprint,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Blueprint":
        """Deserialize from dict."""
        boundary_data = data.get("boundary", {})
        boundary = Boundary(
            allowed_tools=boundary_data.get("allowed_tools", []),
            denied_tools=boundary_data.get("denied_tools", []),
            max_episodes_per_hour=boundary_data.get("max_episodes_per_hour", 100),
            max_tokens_per_episode=boundary_data.get("max_tokens_per_episode", 10000),
            allow_file_write=boundary_data.get("allow_file_write", False),
            allow_network=boundary_data.get("allow_network", False),
            allow_subprocess=boundary_data.get("allow_subprocess", False),
            workspace_path=boundary_data.get("workspace_path"),
        )

        policy_data = data.get("evolution_policy", {})
        policy = EvolutionPolicy(
            strategy=EvolutionStrategy(policy_data.get("strategy", "balanced")),
            auto_promote_to_global=policy_data.get("auto_promote_to_global", False),
            require_tests=policy_data.get("require_tests", True),
            max_evolution_attempts=policy_data.get("max_evolution_attempts", 3),
            pytest_timeout_seconds=policy_data.get("pytest_timeout_seconds", 30),
            allowed_imports=policy_data.get("allowed_imports", ["typing", "re", "json"]),
            denied_functions=policy_data.get("denied_functions", ["eval", "exec", "compile"]),
        )

        return cls(
            blueprint_id=data["blueprint_id"],
            holon_id=data["holon_id"],
            species_id=data["species_id"],
            name=data["name"],
            purpose=data["purpose"],
            boundary=boundary,
            evolution_policy=policy,
            initial_memory_tags=data.get("initial_memory_tags", []),
            parent_blueprint=data.get("parent_blueprint"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
        )
