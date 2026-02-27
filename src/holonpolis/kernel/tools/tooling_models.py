"""Tool chain data models for HolonPolis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ToolChainStep:
    """Enhanced tool step with chain execution metadata."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    step_id: Optional[str] = None
    save_as: Optional[str] = None  # Store result under this key for later steps
    input_from: Optional[str] = None  # Use result from this key as input
    on_error: Literal["stop", "retry", "continue"] = "stop"
    max_retries: int = 2
    retry_count: int = 0


@dataclass
class ToolChainResult:
    """Result of a tool chain execution."""

    ok: bool
    outputs: List[Dict[str, Any]]
    errors: List[str]
    total_steps: int
    completed_steps: int
    failed_steps: int
    retried_steps: int
    saved_results: Dict[str, Any] = field(default_factory=dict)
