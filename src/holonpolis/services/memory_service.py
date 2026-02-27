"""Memory Service - per-Holon memory management.

Enforces: Each Holon has its own isolated LanceDB.
Provides: Unified interface for remember/recall/episode recording.

铁律：所有检索必须使用 Hybrid Search (FTS + Vector)。
Sniper Mode: 通过精确检索最大化上下文质量，最小化 token 消耗。
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from holonpolis.domain.memory import Episode, MemoryKind, MemoryRecord, RetrievalQuery
from holonpolis.kernel.embeddings.default_embedder import get_embedder
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory

logger = structlog.get_logger()


@dataclass
class HybridSearchResult:
    """Hybrid Search 结果项。

    结合 FTS 和 Vector 搜索的结果，包含融合分数。
    """
    memory_id: str
    content: str
    kind: str
    tags: List[str]
    importance: float
    success_score: Optional[float]
    created_at: str

    # 搜索分数
    vector_score: float  # 向量相似度 (0-1)
    text_score: float  # 文本匹配分数 (0-1)
    hybrid_score: float  # 融合分数 (加权组合)

    # 源信息
    source_episode_id: Optional[str] = None
    source_skill: Optional[str] = None


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

        # 确保 kind 是 MemoryKind 枚举
        if isinstance(kind, str):
            kind = MemoryKind(kind)

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

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.5,
    ) -> List[HybridSearchResult]:
        """混合搜索 (Hybrid Search): 结合 FTS + Vector 检索。

        铁律：所有检索必须使用 Hybrid Search。

        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量搜索权重 (0-1)
            text_weight: 全文搜索权重 (0-1)
            filters: 过滤条件 {"kind": "fact", "tags": ["math"]}
            min_score: 最低融合分数阈值

        Returns:
            按 hybrid_score 排序的结果列表
        """
        conn = self._get_connection()
        table = conn.get_table("memories")

        # 1. 向量搜索
        query_embedding = await self.embedder.embed_single(query)
        vector_results = table.search(
            query_embedding, vector_column_name="embedding"
        ).limit(top_k * 2).to_list()

        vector_scores = {}
        for r in vector_results:
            dist = r.get("_distance", 0.0)
            # Cosine distance -> similarity
            sim = 1.0 - float(dist)
            vector_scores[r["memory_id"]] = {
                "record": r,
                "score": max(0.0, sim),
            }

        # 2. 全文搜索 (FTS)
        try:
            # LanceDB FTS 语法: 使用 search(query, query_type="fts")
            text_results = table.search(
                query, query_type="fts"
            ).limit(top_k * 2).to_list()

            text_scores = {}
            for r in text_results:
                # FTS 返回的是 BM25 分数，需要归一化
                score = r.get("_score", 0.0)
                # BM25 分数无上限，使用 sigmoid 归一化到 0-1
                import math
                normalized = 1.0 / (1.0 + math.exp(-score / 5.0))
                text_scores[r["memory_id"]] = {
                    "record": r,
                    "score": normalized,
                }
        except Exception as e:
            # FTS 可能未启用或失败，只使用向量结果
            logger.warning("fts_search_failed", error=str(e), holon_id=self.holon_id)
            text_scores = {}

        # 3. 融合分数 (RRF - Reciprocal Rank Fusion)
        all_ids = set(vector_scores.keys()) | set(text_scores.keys())
        fused_results = []

        for memory_id in all_ids:
            v_data = vector_scores.get(memory_id, {})
            t_data = text_scores.get(memory_id, {})

            v_score = v_data.get("score", 0.0)
            t_score = t_data.get("score", 0.0)

            # 加权融合
            hybrid_score = (vector_weight * v_score) + (text_weight * t_score)

            # 使用任一侧的记录数据
            record = v_data.get("record") or t_data.get("record")
            if record is None:
                continue

            # 应用过滤器
            if filters:
                if "kind" in filters and record.get("kind") != filters["kind"]:
                    continue
                if "min_importance" in filters:
                    if record.get("importance", 0) < filters["min_importance"]:
                        continue
                if "tags" in filters:
                    record_tags = set(record.get("tags", []))
                    if not any(tag in record_tags for tag in filters["tags"]):
                        continue

            if hybrid_score >= min_score:
                fused_results.append(
                    HybridSearchResult(
                        memory_id=record["memory_id"],
                        content=record["content"],
                        kind=record["kind"],
                        tags=record.get("tags", []),
                        importance=record.get("importance", 1.0),
                        success_score=record.get("success_score"),
                        created_at=record.get("created_at", ""),
                        vector_score=v_score,
                        text_score=t_score,
                        hybrid_score=hybrid_score,
                        source_episode_id=record.get("source_episode_id"),
                        source_skill=record.get("source_skill"),
                    )
                )

        # 按 hybrid_score 降序排序
        fused_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        logger.debug(
            "hybrid_search_completed",
            holon_id=self.holon_id,
            query=query[:50],
            results=len(fused_results),
            vector_hits=len(vector_scores),
            text_hits=len(text_scores),
        )

        return fused_results[:top_k]

    async def sniper_mode_retrieval(
        self,
        query: str,
        max_context_tokens: int = 4000,
        context_per_item: int = 200,  # 预估每个记忆项的 token 数
    ) -> Dict[str, Any]:
        """Sniper Mode 上下文汇聚 - 精确检索，最小冗余。

        策略:
        1. 使用 Hybrid Search 获取候选集
        2. 根据重要性、成功分数、时效性重排序
        3. 在 token 预算内选择最高质量的上下文

        Args:
            query: 查询文本
            max_context_tokens: 最大 token 预算
            context_per_item: 预估每个记忆项占用的 token 数

        Returns:
            {
                "memories": List[HybridSearchResult],
                "total_tokens": int,
                "coverage": float,  # 查询意图覆盖率估计
            }
        """
        # 计算可以包含的记忆项数量
        max_items = max(1, max_context_tokens // context_per_item)

        # 获取候选集 (2x 以确保有足够的选择)
        candidates = await self.hybrid_search(
            query=query,
            top_k=max_items * 2,
            vector_weight=0.6,
            text_weight=0.4,
            min_score=0.3,
        )

        if not candidates:
            return {
                "memories": [],
                "total_tokens": 0,
                "coverage": 0.0,
            }

        # Sniper Mode 重排序算法
        # 综合因素: hybrid_score, importance, success_score, recency
        scored_candidates = []
        now = datetime.utcnow()

        for c in candidates:
            # 基础分数: hybrid search 结果
            score = c.hybrid_score

            # 重要性加权
            score *= (0.5 + c.importance)  # importance 范围通常 0-2

            # 成功分数加权 (如果有)
            if c.success_score is not None:
                score *= (0.8 + 0.4 * c.success_score)  # 0.8-1.2x

            # 时效性衰减 (简单的线性衰减)
            try:
                created = datetime.fromisoformat(c.created_at.replace("Z", "+00:00"))
                days_old = (now - created).days
                recency_factor = max(0.5, 1.0 - (days_old / 365))  # 一年内线性衰减
                score *= recency_factor
            except Exception:
                pass

            scored_candidates.append((score, c))

        # 按综合分数排序
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 在 token 预算内选择
        selected = []
        total_tokens = 0

        for score, candidate in scored_candidates:
            if len(selected) >= max_items:
                break
            selected.append(candidate)
            total_tokens += context_per_item

        # 计算覆盖率 (基于 hybrid_score 的加权平均)
        coverage = sum(c.hybrid_score for c in selected) / len(selected) if selected else 0.0

        logger.debug(
            "sniper_mode_retrieval_completed",
            holon_id=self.holon_id,
            query=query[:50],
            selected=len(selected),
            candidates=len(candidates),
            coverage=coverage,
        )

        return {
            "memories": selected,
            "total_tokens": total_tokens,
            "coverage": coverage,
        }

    async def recall_with_sniper_mode(
        self,
        query: str,
        top_k: int = 5,
        **filters
    ) -> List[Dict[str, Any]]:
        """使用 Sniper Mode 的记忆检索 (兼容旧接口)。

        这是 recall 方法的增强版，使用 Hybrid Search + Sniper Mode。
        """
        # 估算 token 预算: top_k * 200 tokens per item
        max_tokens = top_k * 200

        result = await self.sniper_mode_retrieval(
            query=query,
            max_context_tokens=max_tokens,
            context_per_item=200,
        )

        # 转换为旧接口格式
        memories = []
        for m in result["memories"]:
            memories.append({
                "memory_id": m.memory_id,
                "content": m.content,
                "kind": m.kind,
                "tags": m.tags,
                "importance": m.importance,
                "success_score": m.success_score,
                "similarity": m.hybrid_score,  # 使用 hybrid_score 作为相似度
                "created_at": m.created_at,
            })

        return memories
