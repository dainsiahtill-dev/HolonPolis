"""Evolution Lord - Genesis's decision engine.

Produces structured decisions: Route or Spawn.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.blueprints import EvolutionStrategy
from holonpolis.kernel.llm.llm_runtime import LLMConfig, LLMMessage, get_llm_runtime

logger = structlog.get_logger()


@dataclass
class RouteDecision:
    """Decision to route to an existing Holon."""

    decision: str = "route_to"  # Literal "route_to"
    holon_id: str = ""
    confidence: float = 0.0  # 0.0-1.0
    reasoning: str = ""
    context_to_inject: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpawnDecision:
    """Decision to spawn a new Holon."""

    decision: str = "spawn"  # Literal "spawn"
    blueprint: Optional[Blueprint] = None
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class DenyDecision:
    """Decision to deny the request."""

    decision: str = "deny"  # Literal "deny"
    reason: str = ""
    suggested_alternative: Optional[str] = None


@dataclass
class ClarifyDecision:
    """Decision to ask for clarification."""

    decision: str = "clarify"  # Literal "clarify"
    question: str = ""


DecisionType = Union[RouteDecision, SpawnDecision, DenyDecision, ClarifyDecision]


class EvolutionLord:
    """The Genesis decision engine.

    Uses LLM to make routing/spawning decisions based on user requests.
    """

    def __init__(self):
        self.runtime = get_llm_runtime()
        self._system_prompt = self._load_system_prompt()
        self._blueprint_schema = self._load_blueprint_schema()

    def _load_system_prompt(self) -> str:
        """Load the Genesis system prompt."""
        prompt_path = Path(__file__).parent / "prompts" / "genesis_system_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "You are the Evolution Lord. Make routing decisions."

    def _load_blueprint_schema(self) -> Dict:
        """Load the blueprint JSON schema."""
        schema_path = Path(__file__).parent / "prompts" / "blueprint_schema.json"
        if schema_path.exists():
            return json.loads(schema_path.read_text(encoding="utf-8"))
        return {}

    async def decide(
        self,
        user_request: str,
        available_holons: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]] = None,
    ) -> DecisionType:
        """Make a routing/spawning decision.

        Args:
            user_request: The user's request text
            available_holons: List of existing holon info dicts with keys:
                - holon_id
                - name
                - purpose
                - species_id
                - capabilities
                - recent_success_rate
            conversation_history: Optional conversation context

        Returns:
            One of the decision types
        """
        # Build the prompt for the Evolution Lord
        prompt = self._build_decision_prompt(
            user_request, available_holons, conversation_history
        )

        config = LLMConfig(
            model="gpt-4o-mini",  # Can be configured
            temperature=0.3,  # Lower for more deterministic decisions
            max_tokens=2048,
        )

        try:
            response = await self.runtime.chat(
                system_prompt=self._system_prompt,
                user_prompt=prompt,
                config=config,
            )

            # Parse the JSON response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            decision_data = json.loads(content)
            return self._parse_decision(decision_data)

        except json.JSONDecodeError as e:
            logger.error("decision_parse_failed", error=str(e), content=response.content[:500])
            # Fallback to clarify
            return ClarifyDecision(question="I couldn't understand your request. Could you rephrase?")
        except Exception as e:
            logger.error("decision_failed", error=str(e))
            raise

    def _build_decision_prompt(
        self,
        user_request: str,
        available_holons: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]],
    ) -> str:
        """Build the decision prompt."""
        lines = [
            "# Decision Request",
            "",
            f"User Request: {user_request}",
            "",
        ]

        if conversation_history:
            lines.extend(["# Conversation History", ""])
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"{role}: {content[:200]}")  # Truncate long messages
            lines.append("")

        lines.extend(["# Available Holons", ""])
        if available_holons:
            for h in available_holons:
                lines.append(f"- ID: {h.get('holon_id')}")
                lines.append(f"  Name: {h.get('name')}")
                lines.append(f"  Purpose: {h.get('purpose', 'N/A')}")
                lines.append(f"  Species: {h.get('species_id')}")
                lines.append(f"  Success Rate: {h.get('success_rate', 'N/A')}")
                lines.append("")
        else:
            lines.append("No existing Holons available.")
            lines.append("")

        lines.extend([
            "# Your Task",
            "",
            "Decide how to handle this request.",
            "",
            "Response must be valid JSON with one of these structures:",
            "",
            "1. Route to existing Holon:",
            json.dumps({
                "decision": "route_to",
                "holon_id": "existing_holon_id",
                "confidence": 0.85,
                "reasoning": "Why this Holon is appropriate",
                "context_to_inject": {}
            }, indent=2),
            "",
            "2. Spawn new Holon:",
            json.dumps({
                "decision": "spawn",
                "blueprint": {
                    "species_id": "generalist",
                    "name": "Descriptive Name",
                    "purpose": "Specific purpose for routing",
                    "boundary": {
                        "allowed_tools": ["search", "calculate"],
                        "denied_tools": [],
                        "max_episodes_per_hour": 100,
                        "max_tokens_per_episode": 10000,
                        "allow_file_write": False,
                        "allow_network": False,
                        "allow_subprocess": False
                    },
                    "evolution_policy": {
                        "strategy": "balanced",
                        "auto_promote_to_global": False,
                        "require_tests": True,
                        "max_evolution_attempts": 3
                    },
                    "initial_memory_tags": ["domain_tag"]
                },
                "confidence": 0.9,
                "reasoning": "Why a new Holon is needed"
            }, indent=2),
            "",
            "3. Deny request:",
            json.dumps({
                "decision": "deny",
                "reason": "Security or policy violation",
                "suggested_alternative": "Optional alternative"
            }, indent=2),
            "",
            "4. Ask for clarification:",
            json.dumps({
                "decision": "clarify",
                "question": "What specifically do you need?"
            }, indent=2),
            "",
            "Choose the most appropriate response format.",
        ])

        return "\n".join(lines)

    def _parse_decision(self, data: Dict[str, Any]) -> DecisionType:
        """Parse decision JSON into typed decision."""
        decision_type = data.get("decision")

        if decision_type == "route_to":
            return RouteDecision(
                holon_id=data.get("holon_id", ""),
                confidence=data.get("confidence", 0.0),
                reasoning=data.get("reasoning", ""),
                context_to_inject=data.get("context_to_inject", {}),
            )

        elif decision_type == "spawn":
            blueprint_data = data.get("blueprint", {})
            blueprint = self._create_blueprint_from_decision(blueprint_data)
            return SpawnDecision(
                blueprint=blueprint,
                confidence=data.get("confidence", 0.0),
                reasoning=data.get("reasoning", ""),
            )

        elif decision_type == "deny":
            return DenyDecision(
                reason=data.get("reason", ""),
                suggested_alternative=data.get("suggested_alternative"),
            )

        elif decision_type == "clarify":
            return ClarifyDecision(question=data.get("question", ""))

        else:
            logger.warning("unknown_decision_type", decision_type=decision_type)
            return ClarifyDecision(question="Could you clarify your request?")

    def _create_blueprint_from_decision(self, data: Dict[str, Any]) -> Blueprint:
        """Create a Blueprint from decision data."""
        holon_id = f"holon_{uuid.uuid4().hex[:12]}"
        blueprint_id = f"blueprint_{uuid.uuid4().hex[:12]}"

        boundary_data = data.get("boundary", {})
        boundary = Boundary(
            allowed_tools=boundary_data.get("allowed_tools", []),
            denied_tools=boundary_data.get("denied_tools", []),
            max_episodes_per_hour=boundary_data.get("max_episodes_per_hour", 100),
            max_tokens_per_episode=boundary_data.get("max_tokens_per_episode", 10000),
            allow_file_write=boundary_data.get("allow_file_write", False),
            allow_network=boundary_data.get("allow_network", False),
            allow_subprocess=boundary_data.get("allow_subprocess", False),
        )

        policy_data = data.get("evolution_policy", {})
        strategy_str = policy_data.get("strategy", "balanced")
        policy = EvolutionPolicy(
            strategy=EvolutionStrategy(strategy_str),
            auto_promote_to_global=policy_data.get("auto_promote_to_global", False),
            require_tests=policy_data.get("require_tests", True),
            max_evolution_attempts=policy_data.get("max_evolution_attempts", 3),
        )

        return Blueprint(
            blueprint_id=blueprint_id,
            holon_id=holon_id,
            species_id=data.get("species_id", "generalist"),
            name=data.get("name", "Unnamed Holon"),
            purpose=data.get("purpose", ""),
            boundary=boundary,
            evolution_policy=policy,
            initial_memory_tags=data.get("initial_memory_tags", []),
            created_at=datetime.utcnow().isoformat(),
        )
