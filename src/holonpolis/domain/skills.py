"""Skill definitions and manifests."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from holonpolis.infrastructure.time_utils import utc_now_iso


@dataclass
class ToolSchema:
    """JSON Schema for a tool's input."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object
    required: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
        }


@dataclass
class SkillVersion:
    """Specific version of a skill."""

    version: str  # semver
    created_at: str = field(default_factory=utc_now_iso)
    created_by: str = ""  # holon_id that evolved this

    # Code location
    code_path: str = ""  # Relative to skills directory
    test_path: str = ""  # Relative to skills directory

    # Attestation
    attestation_id: Optional[str] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    static_scan_passed: bool = False

    # Promotion
    is_global: bool = False
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None


@dataclass
class SkillManifest:
    """Manifest for a skill - the "interface definition"."""

    skill_id: str
    name: str
    description: str
    version: str  # Current version

    # Schema
    tool_schema: ToolSchema

    # Metadata
    tags: List[str] = field(default_factory=list)
    author_holon: Optional[str] = None
    origin_species: Optional[str] = None

    # Versioning
    versions: List[SkillVersion] = field(default_factory=list)

    # Evolution tracking
    parent_skill: Optional[str] = None  # If evolved from another
    evolution_count: int = 0

    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tool_schema": self.tool_schema.to_dict(),
            "tags": self.tags,
            "author_holon": self.author_holon,
            "origin_species": self.origin_species,
            "versions": [
                {
                    "version": v.version,
                    "created_at": v.created_at,
                    "created_by": v.created_by,
                    "code_path": v.code_path,
                    "test_path": v.test_path,
                    "attestation_id": v.attestation_id,
                    "test_results": v.test_results,
                    "static_scan_passed": v.static_scan_passed,
                    "is_global": v.is_global,
                    "promoted_at": v.promoted_at,
                    "promoted_by": v.promoted_by,
                }
                for v in self.versions
            ],
            "parent_skill": self.parent_skill,
            "evolution_count": self.evolution_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        """Deserialize from dict."""
        tool_schema = ToolSchema(
            name=data["tool_schema"]["name"],
            description=data["tool_schema"]["description"],
            parameters=data["tool_schema"]["parameters"],
            required=data["tool_schema"].get("required", []),
        )

        versions = []
        for v_data in data.get("versions", []):
            sv = SkillVersion(
                version=v_data["version"],
                created_at=v_data["created_at"],
                created_by=v_data.get("created_by", ""),
                code_path=v_data.get("code_path", ""),
                test_path=v_data.get("test_path", ""),
                attestation_id=v_data.get("attestation_id"),
                test_results=v_data.get("test_results", {}),
                static_scan_passed=v_data.get("static_scan_passed", False),
                is_global=v_data.get("is_global", False),
                promoted_at=v_data.get("promoted_at"),
                promoted_by=v_data.get("promoted_by"),
            )
            versions.append(sv)

        return cls(
            skill_id=data["skill_id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            tool_schema=tool_schema,
            tags=data.get("tags", []),
            author_holon=data.get("author_holon"),
            origin_species=data.get("origin_species"),
            versions=versions,
            parent_skill=data.get("parent_skill"),
            evolution_count=data.get("evolution_count", 0),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
        )
