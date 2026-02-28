"""Genesis Service - orchestration layer for routing and spawning.

This is the entry point for user requests.
It decides whether to route to an existing Holon or spawn a new one.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.domain import Blueprint
from holonpolis.domain.errors import GenesisError
from holonpolis.genesis import EvolutionLord, RouteDecision, SpawnDecision
from holonpolis.genesis.genesis_memory import GenesisMemory
from holonpolis.kernel.embeddings.default_embedder import get_embedder

from .holon_service import HolonService
from .memory_service import MemoryService

logger = structlog.get_logger()


@dataclass
class RouteResult:
    """Result of a Genesis routing decision."""
    decision: str  # route_to, spawn, deny, clarify
    holon_id: Optional[str] = None
    blueprint: Optional[Blueprint] = None
    reasoning: str = ""
    message: Optional[str] = None  # For clarify or deny
    route_id: Optional[str] = None


class GenesisService:
    """The Genesis orchestration service.

    Responsibilities:
    - Receive user requests
    - Decide route/spawn/deny/clarify
    - Spawn new Holons when needed
    - Track all decisions in Genesis memory
    """

    def __init__(self):
        self.evolution_lord = EvolutionLord()
        self.genesis_memory = GenesisMemory()
        self.holon_service = HolonService()
        self.embedder = get_embedder()

    async def route_or_spawn(
        self,
        user_request: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> RouteResult:
        """Main entry point: decide what to do with a user request.

        Args:
            user_request: The user's request
            conversation_history: Optional conversation context

        Returns:
            RouteResult with decision and relevant data
        """
        # Get available Holons from Genesis memory
        available_holons = await self.genesis_memory.list_active_holons()

        # Also check local filesystem for any not yet indexed
        local_holons = self.holon_service.list_holons()
        local_holon_ids = {h["holon_id"] for h in local_holons}
        genesis_holon_ids = {h["holon_id"] for h in available_holons}

        # Index any missing Holons
        for holon_data in local_holons:
            if holon_data["holon_id"] not in genesis_holon_ids:
                await self._index_holon(holon_data["holon_id"])
                available_holons.append({
                    "holon_id": holon_data["holon_id"],
                    "name": holon_data["name"],
                    "purpose": holon_data["purpose"],
                    "species_id": holon_data["species_id"],
                    "capabilities": [],
                    "success_rate": 0.0,
                })

        available_holons = [
            item
            for item in available_holons
            if self.holon_service.holon_exists(str(item.get("holon_id") or ""))
            and self.holon_service.is_active(str(item.get("holon_id") or ""))
        ]

        # Ask Evolution Lord for decision
        try:
            decision = await self.evolution_lord.decide(
                user_request=user_request,
                available_holons=available_holons,
                conversation_history=conversation_history,
            )
        except Exception as e:
            logger.error("evolution_lord_failed", error=str(e))
            raise GenesisError(f"Failed to make routing decision: {e}")

        # Handle different decision types
        if isinstance(decision, RouteDecision):
            # Verify the Holon exists
            if not self.holon_service.holon_exists(decision.holon_id):
                logger.error("route_to_nonexistent_holon", holon_id=decision.holon_id)
                # Fallback to spawn
                return await self._spawn_for_request(user_request)
            if not self.holon_service.is_active(decision.holon_id):
                logger.info("route_to_non_runnable_holon", holon_id=decision.holon_id)
                return await self._spawn_for_request(user_request)

            # Record the route decision
            route_id = await self.genesis_memory.record_route_decision(
                query=user_request,
                decision="route_to",
                target_holon_id=decision.holon_id,
                spawned_blueprint_id=None,
                reasoning=decision.reasoning,
            )

            logger.info(
                "routed_to_holon",
                holon_id=decision.holon_id,
                confidence=decision.confidence,
            )

            return RouteResult(
                decision="route_to",
                holon_id=decision.holon_id,
                reasoning=decision.reasoning,
                route_id=route_id,
            )

        elif isinstance(decision, SpawnDecision):
            if decision.blueprint is None:
                raise GenesisError("Spawn decision missing blueprint")

            # Record spawn decision before creating
            route_id = await self.genesis_memory.record_route_decision(
                query=user_request,
                decision="spawn",
                target_holon_id=None,
                spawned_blueprint_id=decision.blueprint.blueprint_id,
                reasoning=decision.reasoning,
            )

            # Create the Holon
            holon_id = await self._create_holon_from_blueprint(decision.blueprint)

            logger.info(
                "spawned_new_holon",
                holon_id=holon_id,
                name=decision.blueprint.name,
                confidence=decision.confidence,
            )

            return RouteResult(
                decision="spawn",
                holon_id=holon_id,
                blueprint=decision.blueprint,
                reasoning=decision.reasoning,
                route_id=route_id,
            )

        elif hasattr(decision, 'decision') and decision.decision == "deny":
            from holonpolis.genesis.evolution_lord import DenyDecision
            route_id = await self.genesis_memory.record_route_decision(
                query=user_request,
                decision="deny",
                target_holon_id=None,
                spawned_blueprint_id=None,
                reasoning=getattr(decision, "reason", "Denied by policy"),
            )
            if isinstance(decision, DenyDecision):
                return RouteResult(
                    decision="deny",
                    reasoning=decision.reason,
                    message=decision.suggested_alternative,
                    route_id=route_id,
                )
            return RouteResult(
                decision="deny",
                reasoning=getattr(decision, 'reason', 'Request denied by policy'),
                route_id=route_id,
            )

        elif hasattr(decision, 'decision') and decision.decision == "clarify":
            from holonpolis.genesis.evolution_lord import ClarifyDecision
            route_id = await self.genesis_memory.record_route_decision(
                query=user_request,
                decision="clarify",
                target_holon_id=None,
                spawned_blueprint_id=None,
                reasoning=getattr(decision, "question", "Need clarification"),
            )
            if isinstance(decision, ClarifyDecision):
                return RouteResult(
                    decision="clarify",
                    message=decision.question,
                    route_id=route_id,
                )
            return RouteResult(
                decision="clarify",
                message=getattr(decision, 'question', 'Could you clarify your request?'),
                route_id=route_id,
            )

        else:
            logger.warning("unexpected_decision_type", decision_type=type(decision).__name__)
            return RouteResult(
                decision="clarify",
                message="I'm not sure how to handle that request. Could you provide more details?",
            )

    async def _spawn_for_request(self, user_request: str) -> RouteResult:
        """Fallback: spawn a generalist Holon for a request."""
        from holonpolis.domain import Boundary, EvolutionPolicy
        from holonpolis.domain.blueprints import EvolutionStrategy
        import uuid

        blueprint = Blueprint(
            blueprint_id=f"blueprint_{uuid.uuid4().hex[:12]}",
            holon_id=f"holon_{uuid.uuid4().hex[:12]}",
            species_id="generalist",
            name="General Assistant",
            purpose=f"Handle request: {user_request[:100]}",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(strategy=EvolutionStrategy.BALANCED),
        )

        holon_id = await self._create_holon_from_blueprint(blueprint)

        return RouteResult(
            decision="spawn",
            holon_id=holon_id,
            blueprint=blueprint,
            reasoning="Fallback spawn for unroutable request",
        )

    async def _create_holon_from_blueprint(self, blueprint: Blueprint) -> str:
        """Create a Holon and register it in Genesis memory."""
        # Create the Holon
        holon_id = await self.holon_service.create_holon(blueprint)

        # Register in Genesis
        await self.genesis_memory.register_holon(
            holon_id=holon_id,
            blueprint_id=blueprint.blueprint_id,
            species_id=blueprint.species_id,
            name=blueprint.name,
            purpose=blueprint.purpose,
            capabilities=blueprint.boundary.allowed_tools,
            skills=[],  # No skills yet
        )

        return holon_id

    async def _index_holon(self, holon_id: str) -> None:
        """Index an existing Holon into Genesis memory."""
        try:
            blueprint = self.holon_service.get_blueprint(holon_id)
            await self.genesis_memory.register_holon(
                holon_id=holon_id,
                blueprint_id=blueprint.blueprint_id,
                species_id=blueprint.species_id,
                name=blueprint.name,
                purpose=blueprint.purpose,
                capabilities=blueprint.boundary.allowed_tools,
                skills=[],
            )
        except Exception as e:
            logger.warning("failed_to_index_holon", holon_id=holon_id, error=str(e))

    async def get_holon_for_request(
        self,
        user_request: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """Get the best Holon for a request, spawning if necessary.

        Convenience method that returns just the holon_id.
        """
        result = await self.route_or_spawn(user_request, conversation_history)

        if result.decision in ("route_to", "spawn"):
            return result.holon_id

        return None

    async def list_all_holons(self) -> List[Dict[str, Any]]:
        """List all Holons with their Genesis metadata."""
        return await self.genesis_memory.list_active_holons()

    async def get_evolution_history(self, holon_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evolution history for a Holon or all Holons."""
        if holon_id:
            return await self.genesis_memory.get_holon_evolutions(holon_id)

        # Get all evolutions - would need to query all holons
        # For now, return empty list
        return []
