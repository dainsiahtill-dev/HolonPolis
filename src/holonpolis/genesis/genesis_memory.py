"""Genesis Memory - specialized memory for the Evolution Lord.

Genesis has its own LanceDB with three tables:
- holons: Registry of all holons
- routes: History of routing decisions
- evolutions: History of evolution attempts
"""

from typing import Any, Dict, List, Optional
import uuid

import structlog

from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory
from holonpolis.kernel.embeddings.default_embedder import get_embedder
from holonpolis.infrastructure.time_utils import utc_now_iso

logger = structlog.get_logger()


class GenesisMemory:
    """Genesis's memory store.

    Tracks:
    - All holons in the system
    - All routing decisions
    - All evolution attempts
    """

    def __init__(self):
        self.factory = get_lancedb_factory()
        self.embedder = get_embedder()

    def _get_connection(self):
        """Get Genesis's LanceDB connection."""
        return self.factory.open("genesis")

    @staticmethod
    def _latest_record(records: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
        """Pick the latest record by ISO timestamp field."""
        if not records:
            return None
        return max(records, key=lambda item: item.get(key) or "")

    async def register_holon(
        self,
        holon_id: str,
        blueprint_id: str,
        species_id: str,
        name: str,
        purpose: str,
        capabilities: List[str],
        skills: List[str],
    ) -> None:
        """Register a new holon in Genesis's memory."""
        conn = self._get_connection()
        table = conn.get_table("holons")

        # Embed the purpose for semantic search
        purpose_embedding = await self.embedder.embed_single(purpose)

        now = utc_now_iso()

        table.add([{
            "holon_id": holon_id,
            "blueprint_id": blueprint_id,
            "species_id": species_id,
            "name": name,
            "purpose": purpose,
            "status": "active",
            "capabilities": capabilities,
            "skills": skills,
            "total_episodes": 0,
            "success_rate": 0.0,
            "last_active_at": now,
            "created_at": now,
            "embedding": purpose_embedding,
        }])

        logger.info("holon_registered_in_genesis", holon_id=holon_id, name=name)

    async def update_holon_stats(
        self,
        holon_id: str,
        episode_increment: int = 0,
        success: Optional[bool] = None,
    ) -> None:
        """Update holon statistics after episode completion."""
        conn = self._get_connection()
        table = conn.get_table("holons")
        rows = table.search().where(f"holon_id = '{holon_id}'").limit(1000).to_list()
        latest = self._latest_record(rows, "created_at")
        if latest is None:
            logger.warning("holon_stats_update_skipped", holon_id=holon_id, reason="holon_not_found")
            return

        previous_total = int(latest.get("total_episodes", 0))
        new_total = max(0, previous_total + int(episode_increment))

        prev_rate = float(latest.get("success_rate", 0.0))
        if success is None or episode_increment <= 0:
            new_success_rate = prev_rate
        else:
            prev_successes = int(round(previous_total * prev_rate))
            new_successes = prev_successes + (1 if success else 0)
            new_success_rate = float(new_successes / new_total) if new_total > 0 else 0.0

        now = utc_now_iso()
        table.add([{
            "holon_id": latest["holon_id"],
            "blueprint_id": latest["blueprint_id"],
            "species_id": latest["species_id"],
            "name": latest["name"],
            "purpose": latest["purpose"],
            "status": latest.get("status", "active"),
            "capabilities": latest.get("capabilities", []),
            "skills": latest.get("skills", []),
            "total_episodes": new_total,
            "success_rate": new_success_rate,
            "last_active_at": now,
            "created_at": now,
            "embedding": latest["embedding"],
        }])

        logger.debug(
            "holon_stats_updated",
            holon_id=holon_id,
            total_episodes=new_total,
            success_rate=new_success_rate,
        )

    async def find_holons_for_task(
        self,
        task_description: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find holons that might handle a task (semantic search)."""
        conn = self._get_connection()
        table = conn.get_table("holons")

        # Embed the task
        task_embedding = await self.embedder.embed_single(task_description)

        # Search by purpose embedding
        results = table.search(task_embedding, vector_column_name="embedding").limit(top_k * 5).to_list()
        latest_by_holon: Dict[str, Dict[str, Any]] = {}

        for row in results:
            holon_id = row["holon_id"]
            previous = latest_by_holon.get(holon_id)
            if previous is None or row.get("created_at", "") > previous.get("created_at", ""):
                latest_by_holon[holon_id] = row

        deduped = list(latest_by_holon.values())[:top_k]
        return [
            {
                "holon_id": r["holon_id"],
                "name": r["name"],
                "purpose": r["purpose"],
                "species_id": r["species_id"],
                "capabilities": r["capabilities"],
                "success_rate": r["success_rate"],
                "total_episodes": r["total_episodes"],
            }
            for r in deduped
        ]

    async def list_active_holons(self) -> List[Dict[str, Any]]:
        """List all active holons."""
        conn = self._get_connection()
        table = conn.get_table("holons")

        # Use filter for active status
        results = table.search().where("status = 'active'").limit(5000).to_list()
        latest_by_holon: Dict[str, Dict[str, Any]] = {}
        for row in results:
            holon_id = row["holon_id"]
            previous = latest_by_holon.get(holon_id)
            if previous is None or row.get("created_at", "") > previous.get("created_at", ""):
                latest_by_holon[holon_id] = row

        return [
            {
                "holon_id": r["holon_id"],
                "name": r["name"],
                "purpose": r["purpose"],
                "species_id": r["species_id"],
                "capabilities": r["capabilities"],
                "success_rate": r["success_rate"],
            }
            for r in latest_by_holon.values()
        ]

    async def record_route_decision(
        self,
        query: str,
        decision: str,  # route_to, spawn, deny, clarify
        target_holon_id: Optional[str],
        spawned_blueprint_id: Optional[str],
        reasoning: str,
    ) -> str:
        """Record a routing decision."""
        conn = self._get_connection()
        table = conn.get_table("routes")

        query_embedding = await self.embedder.embed_single(query)
        route_id = f"route_{uuid.uuid4().hex[:12]}"

        table.add([{
            "route_id": route_id,
            "query": query,
            "query_embedding": query_embedding,
            "decision": decision,
            "target_holon_id": target_holon_id,
            "spawned_blueprint_id": spawned_blueprint_id,
            "reasoning": reasoning,
            "outcome": "pending",
            "created_at": utc_now_iso(),
        }])

        logger.info(
            "route_decision_recorded",
            route_id=route_id,
            decision=decision,
            target=target_holon_id or spawned_blueprint_id,
        )

        return route_id

    async def update_route_outcome(
        self,
        route_id: str,
        outcome: str,  # success, failure
    ) -> None:
        """Update the outcome of a routing decision."""
        conn = self._get_connection()
        table = conn.get_table("routes")
        rows = table.search().where(f"route_id = '{route_id}'").limit(1000).to_list()
        latest = self._latest_record(rows, "created_at")
        if latest is None:
            logger.warning("route_outcome_update_skipped", route_id=route_id, reason="route_not_found")
            return

        now = utc_now_iso()
        table.add([{
            "route_id": latest["route_id"],
            "query": latest["query"],
            "query_embedding": latest["query_embedding"],
            "decision": latest["decision"],
            "target_holon_id": latest.get("target_holon_id"),
            "spawned_blueprint_id": latest.get("spawned_blueprint_id"),
            "reasoning": latest["reasoning"],
            "outcome": outcome,
            "created_at": now,
        }])

        logger.info("route_outcome_updated", route_id=route_id, outcome=outcome)

    async def record_evolution(
        self,
        holon_id: str,
        skill_name: str,
        status: str,  # pending, success, failed
        attestation_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Record an evolution attempt."""
        conn = self._get_connection()
        table = conn.get_table("evolutions")

        evolution_id = f"evo_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()

        table.add([{
            "evolution_id": evolution_id,
            "holon_id": holon_id,
            "skill_name": skill_name,
            "status": status,
            "attempt_count": 1,
            "test_passed": status == "success",
            "static_scan_passed": status == "success",
            "attestation_id": attestation_id,
            "error_message": error_message,
            "created_at": now,
            "completed_at": now if status != "pending" else None,
        }])

        logger.info(
            "evolution_recorded",
            evolution_id=evolution_id,
            holon_id=holon_id,
            skill=skill_name,
            status=status,
        )

        return evolution_id

    async def get_holon_evolutions(self, holon_id: str) -> List[Dict[str, Any]]:
        """Get all evolutions for a holon."""
        conn = self._get_connection()
        table = conn.get_table("evolutions")

        results = table.search().where(f"holon_id = '{holon_id}'").limit(1000).to_list()

        normalized = [
            {
                "evolution_id": r["evolution_id"],
                "skill_name": r["skill_name"],
                "status": r["status"],
                "test_passed": r["test_passed"],
                "created_at": r["created_at"],
            }
            for r in results
        ]
        normalized.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return normalized
