"""Event envelope structure for all internal communication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import json


class EventType(Enum):
    """Types of events in the system."""

    # User interaction
    USER_MESSAGE = "user_message"
    HOLON_RESPONSE = "holon_response"

    # Lifecycle
    HOLON_CREATED = "holon_created"
    HOLON_FROZEN = "holon_frozen"
    HOLON_RESUMED = "holon_resumed"

    # Memory
    EPISODE_RECORDED = "episode_recorded"
    MEMORY_CONSOLIDATED = "memory_consolidated"

    # Evolution
    EVOLUTION_REQUESTED = "evolution_requested"
    EVOLUTION_SUCCEEDED = "evolution_succeeded"
    EVOLUTION_FAILED = "evolution_failed"
    SKILL_PROMOTED = "skill_promoted"

    # Routing
    ROUTE_DECISION = "route_decision"
    SPAWN_DECISION = "spawn_decision"

    # System
    AUDIT_LOG = "audit_log"
    ERROR = "error"


@dataclass
class Event:
    """The actual event payload."""

    type: EventType
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "system"  # Component that emitted the event


@dataclass
class Envelope:
    """Envelope wraps events with routing and tracing info."""

    envelope_id: str
    event: Event

    # Routing
    from_holon: Optional[str] = None  # None = user/system
    to_holon: Optional[str] = None  # None = broadcast/all

    # Tracing
    correlation_id: Optional[str] = None  # Trace across multiple events
    parent_envelope_id: Optional[str] = None  # Causal chain

    # Metadata
    priority: int = 0  # Higher = more urgent
    ttl_seconds: Optional[int] = None  # Expiration

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "envelope_id": self.envelope_id,
            "event": {
                "type": self.event.type.value,
                "payload": self.event.payload,
                "timestamp": self.event.timestamp,
                "source": self.event.source,
            },
            "from_holon": self.from_holon,
            "to_holon": self.to_holon,
            "correlation_id": self.correlation_id,
            "parent_envelope_id": self.parent_envelope_id,
            "priority": self.priority,
            "ttl_seconds": self.ttl_seconds,
        }
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Envelope":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        event_data = data["event"]
        event = Event(
            type=EventType(event_data["type"]),
            payload=event_data["payload"],
            timestamp=event_data["timestamp"],
            source=event_data["source"],
        )
        return cls(
            envelope_id=data["envelope_id"],
            event=event,
            from_holon=data.get("from_holon"),
            to_holon=data.get("to_holon"),
            correlation_id=data.get("correlation_id"),
            parent_envelope_id=data.get("parent_envelope_id"),
            priority=data.get("priority", 0),
            ttl_seconds=data.get("ttl_seconds"),
        )
