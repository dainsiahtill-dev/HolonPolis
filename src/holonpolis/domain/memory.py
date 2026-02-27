"""Memory models - episodes and retrievable memories."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryKind(Enum):
    """Types of memories."""

    EPISODE_SUMMARY = "episode_summary"  # Summarized episode
    FACT = "fact"  # Extracted fact
    PROCEDURE = "procedure"  # Learned procedure
    PATTERN = "pattern"  # Recognized pattern
    SKILL_USAGE = "skill_usage"  # How a skill was used
    EVOLUTION = "evolution"  # Evolution attempt result
    ROUTE_EXPERIENCE = "route_experience"  # Routing decision outcome
    SKILL = "skill"  # Evolved skill code


@dataclass
class MemoryRecord:
    """A retrievable memory unit stored in LanceDB.

    This is the "condensed" form optimized for retrieval.
    """

    memory_id: str
    holon_id: str
    kind: MemoryKind
    content: str  # Text content (embedded)
    embedding: Optional[List[float]] = None  # Vector embedding

    # Metadata for retrieval
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0  # 0.0 - 2.0, higher = more important
    success_score: Optional[float] = None  # 0.0 - 1.0, outcome quality

    # Source tracking
    source_episode_id: Optional[str] = None
    source_skill: Optional[str] = None

    # Temporal
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed_at: Optional[str] = None
    access_count: int = 0

    # Decay
    decay_factor: float = 1.0  # Multiplied by importance over time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LanceDB storage."""
        return {
            "memory_id": self.memory_id,
            "holon_id": self.holon_id,
            "kind": self.kind.value,
            "content": self.content,
            "embedding": self.embedding or [],
            "tags": self.tags,
            "importance": self.importance,
            "success_score": self.success_score,
            "source_episode_id": self.source_episode_id,
            "source_skill": self.source_skill,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "decay_factor": self.decay_factor,
        }


@dataclass
class Episode:
    """A complete interaction episode stored in LanceDB.

    This is the "full record" used for analysis and consolidation.
    """

    episode_id: str
    holon_id: str

    # The conversation/tool chain
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    # [{"role": "user", "content": "..."}, {"role": "assistant", "tool_calls": [...]}]

    # Tool invocations
    tool_chain: List[Dict[str, Any]] = field(default_factory=list)
    # [{"tool": "search", "input": {...}, "output": {...}, "latency_ms": 123}]

    # Outcome
    outcome: str = ""  # "success", "failure", "abandoned", "evolved"
    outcome_details: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    cost: float = 0.0  # Estimated cost in tokens/$
    latency_ms: int = 0  # Total time

    # Evolution reference
    evolution_id: Optional[str] = None  # If this episode spawned evolution

    # Temporal
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ended_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LanceDB storage."""
        return {
            "episode_id": self.episode_id,
            "holon_id": self.holon_id,
            "transcript": self.transcript,
            "tool_chain": self.tool_chain,
            "outcome": self.outcome,
            "outcome_details": self.outcome_details,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "evolution_id": self.evolution_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class RetrievalQuery:
    """Query for memory retrieval."""

    text: Optional[str] = None  # For semantic search
    embedding: Optional[List[float]] = None  # Pre-computed embedding
    filters: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"kind": "procedure", "tags": ["math"], "min_importance": 0.5}

    top_k: int = 5
    min_similarity: float = 0.7

    # Temporal filters
    after: Optional[str] = None  # ISO timestamp
    before: Optional[str] = None

    # Boost recent
    recency_bias: float = 0.1  # 0 = no bias, 1 = strong recency preference
