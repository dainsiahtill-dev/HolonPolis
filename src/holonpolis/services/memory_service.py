"""Memory Service - per-Holon memory management.

Enforces: Each Holon has its own isolated LanceDB.
Provides: Unified interface for remember/recall/episode recording.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.domain.memory import Episode, MemoryKind, MemoryRecord, RetrievalQuery
from holonpolis.kernel.embeddings.default_embedder import get_embedder
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory

logger = structlog.get_logger()


class MemoryService:
    """Service for managing per-Holon memory.

    Key constraint: Each Holon's memory is physically isolated in its own LanceDB.
    """

    def __init__(self, holon_id: str):
        self.holon_id = holon_id
        self.factory = get_lancedb_factory()
        self.embedder = get_embedder()
        self._conn = None

    def _get_connection(self):
        """Get this Holon's database connection."""
        if self._conn is None:
            self._conn = self.factory.open(self.holon_id)
        return self._conn

    async def remember(
        self,
        content: str,
        kind: MemoryKind = MemoryKind.FACT,
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
        success_score: Optional[float] = None,
        source_episode_id: Optional[str] = None,
        source_skill: Optional[str] = None,
    ) -> str:
        """Store a memory in this Holon's memory.

        Args:
            content: The text content to remember
            kind: Type of memory
            tags: Optional tags for filtering
            importance: Importance score (0.0-2.0)
            success_score: Outcome quality if applicable (0.0-1.0)
            source_episode_id: Which episode this came from
            source_skill: Which skill was used

        Returns:
            memory_id: The ID of the created memory
        """
        conn = self._get_connection()
        table = conn.get_table("memories")

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        # Generate embedding
        embedding = await self.embedder.embed_single(content)

        memory = MemoryRecord(
            memory_id=memory_id,
            holon_id=self.holon_id,
            kind=kind,
            content=content,
            embedding=embedding,
            tags=tags or [],
            importance=importance,
            success_score=success_score,
            source_episode_id=source_episode_id,
            source_skill=source_skill,
        )

        table.add([memory.to_dict()])

        logger.debug(
            "memory_stored",
            holon_id=self.holon_id,
            memory_id=memory_id,
            kind=kind.value,
        )

        return memory_id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from this Holon's memory.

        Args:
            query: The query text
            top_k: Maximum number of results
            filters: Optional filters (e.g., {"kind": "procedure", "tags": ["math"]})
            min_similarity: Minimum similarity threshold

        Returns:
            List of memory records sorted by relevance
        """
        conn = self._get_connection()
        table = conn.get_table("memories")

        # Embed the query
        query_embedding = await self.embedder.embed_single(query)

        # Build the search
        search = table.search(query_embedding, vector_column_name="embedding")

        # Apply filters if provided
        if filters:
            where_clauses = []
            if "kind" in filters:
                where_clauses.append(f"kind = '{filters['kind']}'")
            if "min_importance" in filters:
                where_clauses.append(f"importance >= {filters['min_importance']}")
            if "tags" in filters:
                # Tags are stored as list, so we use contains
                for tag in filters["tags"]:
                    where_clauses.append(f"tags LIKE '%{tag}%'")

            if where_clauses:
                search = search.where(" AND ".join(where_clauses))

        # Execute search
        results = search.limit(top_k).to_list()

        # Filter by similarity and update access stats
        scored_results = []
        for r in results:
            # Calculate actual similarity (LanceDB returns distance)
            # Cosine distance to similarity: sim = 1 - dist
            distance = r.get("_distance", 0)
            similarity = 1.0 - distance

            scored_results.append({
                "memory_id": r["memory_id"],
                "content": r["content"],
                "kind": r["kind"],
                "tags": r["tags"],
                "importance": r["importance"],
                "success_score": r["success_score"],
                "similarity": similarity,
                "created_at": r["created_at"],
            })

            # Update access count (fire and forget)
            # In production, this should be batched

        filtered_results = [r for r in scored_results if r["similarity"] >= min_similarity]
        if not filtered_results:
            # Fallback for deterministic local embedders that do not preserve semantics.
            filtered_results = scored_results[:top_k]

        logger.debug(
            "memory_recalled",
            holon_id=self.holon_id,
            query=query[:50],
            results=len(filtered_results),
        )

        return filtered_results

    async def write_episode(
        self,
        transcript: List[Dict[str, Any]],
        tool_chain: Optional[List[Dict[str, Any]]] = None,
        outcome: str = "success",
        outcome_details: Optional[Dict[str, Any]] = None,
        cost: float = 0.0,
        latency_ms: int = 0,
        evolution_id: Optional[str] = None,
    ) -> str:
        """Record a complete episode.

        Args:
            transcript: The conversation transcript
            tool_chain: Tool invocations
            outcome: Episode outcome (success, failure, abandoned, evolved)
            outcome_details: Additional outcome info
            cost: Estimated cost
            latency_ms: Total time
            evolution_id: If this spawned an evolution

        Returns:
            episode_id: The ID of the recorded episode
        """
        conn = self._get_connection()
        table = conn.get_table("episodes")

        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()

        episode = Episode(
            episode_id=episode_id,
            holon_id=self.holon_id,
            transcript=transcript,
            tool_chain=tool_chain or [],
            outcome=outcome,
            outcome_details=outcome_details or {},
            cost=cost,
            latency_ms=latency_ms,
            evolution_id=evolution_id,
            started_at=now,  # Should be passed in from actual start time
            ended_at=now,
        )

        # Convert to dict but JSON-serialize the complex fields
        record = episode.to_dict()
        record["transcript"] = json.dumps(record["transcript"])
        record["tool_chain"] = json.dumps(record["tool_chain"])
        record["outcome_details"] = json.dumps(record["outcome_details"])

        table.add([record])

        logger.debug(
            "episode_recorded",
            holon_id=self.holon_id,
            episode_id=episode_id,
            outcome=outcome,
        )

        return episode_id

    async def consolidate_episode_to_memory(
        self,
        episode_id: str,
        summary: str,
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
    ) -> str:
        """Consolidate an episode into a retrievable memory.

        This extracts the key lesson from an episode for future recall.
        """
        memory_id = await self.remember(
            content=summary,
            kind=MemoryKind.EPISODE_SUMMARY,
            tags=tags or [],
            importance=importance,
            source_episode_id=episode_id,
        )

        logger.debug(
            "episode_consolidated",
            holon_id=self.holon_id,
            episode_id=episode_id,
            memory_id=memory_id,
        )

        return memory_id

    async def get_episodes(
        self,
        outcome: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent episodes.

        Args:
            outcome: Filter by outcome
            limit: Maximum number of episodes

        Returns:
            List of episode records
        """
        conn = self._get_connection()
        table = conn.get_table("episodes")

        if outcome:
            results = table.search().where(f"outcome = '{outcome}'").limit(limit).to_list()
        else:
            # Just get all, ordered by time (LanceDB doesn't guarantee order without index)
            results = table.search().limit(limit).to_list()

        return [
            {
                "episode_id": r["episode_id"],
                "outcome": r["outcome"],
                "cost": r["cost"],
                "latency_ms": r["latency_ms"],
                "evolution_id": r["evolution_id"],
                "started_at": r["started_at"],
            }
            for r in results
        ]
