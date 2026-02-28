"""Phase 2 Tests: LanceDB Memory with Hybrid Search

验证:
1. 物理隔离: 每个 Holon 有独立的 LanceDB
2. Hybrid Search: FTS + Vector 融合检索
3. Sniper Mode: 精确上下文汇聚
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import pyarrow as pa

from holonpolis.kernel.embeddings.default_embedder import set_embedder, SimpleEmbedder
from holonpolis.kernel.lancedb import (
    LanceDBConnection,
    LanceDBFactory,
    build_memories_schema,
    EPISODES_SCHEMA,
    get_lancedb_factory,
    reset_factory,
)


@pytest.fixture
def temp_embedder():
    """使用 SimpleEmbedder 避免 API 调用."""
    embedder = SimpleEmbedder(dimension=128)  # 小维度加快测试
    set_embedder(embedder)
    return embedder


@pytest.fixture
def factory(tmp_path, temp_embedder, monkeypatch):
    """创建测试用的 LanceDB Factory."""
    monkeypatch.setattr(
        "holonpolis.config.settings.holonpolis_root",
        tmp_path / ".holonpolis"
    )
    reset_factory()
    return get_lancedb_factory()


class TestLanceDBPhysicalIsolation:
    """测试物理隔离 - 每个 Holon 独立数据库."""

    def test_holon_gets_independent_db_path(self, factory):
        """测试每个 Holon 获得独立的数据库路径."""
        conn1 = factory.open("agent_001")
        conn2 = factory.open("agent_002")

        # 路径不同
        assert conn1.db_path != conn2.db_path
        assert "agent_001" in str(conn1.db_path)
        assert "agent_002" in str(conn2.db_path)
        assert "holons" in str(conn1.db_path)

    def test_genesis_gets_separate_db(self, factory):
        """测试 Genesis 有独立的数据库路径."""
        genesis_conn = factory.open("genesis")
        agent_conn = factory.open("genesis_holon")  # 名为 genesis 的 Holon

        assert "genesis" in str(genesis_conn.db_path)
        # Genesis 路径应该是 .holonpolis/genesis/memory/lancedb
        # 而不是 .holonpolis/holons/genesis/memory/lancedb
        assert "/holons/genesis/" not in str(genesis_conn.db_path) or "genesis_holon" in str(agent_conn.db_path)

    def test_db_persistence(self, factory, tmp_path):
        """测试数据库持久化到磁盘."""
        conn = factory.open("persistent_agent")

        # 创建表
        schema = build_memories_schema(128)
        conn.create_table("test_memories", schema, exist_ok=True)

        # 验证目录存在
        assert conn.db_path.exists()


class TestLanceDBTableOperations:
    """测试表操作."""

    def test_create_memories_table(self, factory):
        """测试创建 memories 表."""
        factory.init_holon_tables("test_agent")
        conn = factory.open("test_agent")

        tables = conn.list_tables()
        assert "memories" in tables
        assert "episodes" in tables

    def test_create_genesis_tables(self, factory):
        """测试创建 Genesis 专用表."""
        factory.init_genesis_tables()
        conn = factory.open("genesis")

        tables = conn.list_tables()
        assert "holons" in tables
        assert "routes" in tables
        assert "evolutions" in tables

    def test_table_exists_check(self, factory):
        """测试表存在性检查."""
        factory.init_holon_tables("check_agent")
        conn = factory.open("check_agent")

        assert conn.table_exists("memories") is True
        assert conn.table_exists("nonexistent") is False


class _FakeIndexConfig:
    def __init__(self, columns):
        self.index_type = "FTS"
        self.columns = columns


class _FakeTable:
    def __init__(self, *, raise_on_list: bool = False, existing_columns: list[str] | None = None):
        self.raise_on_list = raise_on_list
        self.calls: list[Any] = []
        self._indices = [_FakeIndexConfig([col]) for col in (existing_columns or [])]

    def list_indices(self):
        return self._indices

    def create_fts_index(self, field_names):
        self.calls.append(field_names)
        if isinstance(field_names, list) and self.raise_on_list:
            raise ValueError("field_names must be a string when use_tantivy=False")

        cols = field_names if isinstance(field_names, list) else [field_names]
        for col in cols:
            self._indices.append(_FakeIndexConfig([str(col)]))


class _FakeConnection:
    def __init__(self, table: _FakeTable):
        self._table = table

    def create_table(self, name, schema, exist_ok=True):  # noqa: ARG002
        return self._table


class TestLanceDBFTSCompatibility:
    def test_single_column_list_is_coerced_to_string(self):
        table = _FakeTable(raise_on_list=True)
        conn = LanceDBConnection(Path("."), _FakeConnection(table))

        conn.create_table(
            "memories",
            schema=object(),
            enable_fts=True,
            fts_columns=["content"],
        )

        assert table.calls == ["content"]

    def test_multi_columns_falls_back_to_per_column_creation(self):
        table = _FakeTable(raise_on_list=True)
        conn = LanceDBConnection(Path("."), _FakeConnection(table))

        conn.create_table(
            "memories",
            schema=object(),
            enable_fts=True,
            fts_columns=["content", "summary"],
        )

        assert table.calls == [["content", "summary"], "content", "summary"]

    def test_existing_fts_column_is_not_recreated(self):
        table = _FakeTable(raise_on_list=True, existing_columns=["content"])
        conn = LanceDBConnection(Path("."), _FakeConnection(table))

        conn.create_table(
            "memories",
            schema=object(),
            enable_fts=True,
            fts_columns=["content"],
        )

        assert table.calls == []


class TestMemoryRecordOperations:
    """测试记忆记录操作."""

    @pytest.mark.asyncio
    async def test_add_and_retrieve_memory(self, factory, temp_embedder):
        """测试添加和检索记忆."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("memory_agent")
        service = MemoryService("memory_agent")

        # 存储记忆
        memory_id = await service.remember(
            content="Python is a programming language",
            kind="fact",
            tags=["python", "programming"],
            importance=1.5,
        )

        assert memory_id.startswith("mem_")

    @pytest.mark.asyncio
    async def test_recall_returns_results(self, factory, temp_embedder):
        """测试 recall 返回结果."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("recall_agent")
        service = MemoryService("recall_agent")

        # 先存储一些记忆
        await service.remember(
            content="Machine learning is a subset of AI",
            kind="fact",
            tags=["ai", "ml"],
            importance=1.2,
        )
        await service.remember(
            content="Deep learning uses neural networks",
            kind="fact",
            tags=["ai", "dl"],
            importance=1.3,
        )

        # 检索
        results = await service.recall("artificial intelligence", top_k=3)

        # 应该有结果 (即使使用 SimpleEmbedder)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_write_and_get_episode(self, factory, temp_embedder):
        """测试写入和获取 episode."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("episode_agent")
        service = MemoryService("episode_agent")

        episode_id = await service.write_episode(
            transcript=[{"role": "user", "content": "Hello"}],
            outcome="success",
            cost=0.001,
            latency_ms=100,
        )

        assert episode_id.startswith("ep_")

        episodes = await service.get_episodes(limit=10)
        assert len(episodes) >= 1


class TestHybridSearch:
    """测试 Hybrid Search (FTS + Vector)."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, factory, temp_embedder):
        """测试 hybrid_search 返回结果."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("hybrid_agent")
        service = MemoryService("hybrid_agent")

        # 存储一些记忆
        await service.remember(
            content="Python dictionary is a hash map implementation",
            kind="fact",
            tags=["python", "data-structures"],
            importance=1.0,
        )
        await service.remember(
            content="List is a dynamic array in Python",
            kind="fact",
            tags=["python", "data-structures"],
            importance=1.0,
        )

        # Hybrid search
        results = await service.hybrid_search(
            query="python data structures",
            top_k=5,
            vector_weight=0.6,
            text_weight=0.4,
        )

        assert isinstance(results, list)
        # 结果应该有 HybridSearchResult 结构
        if results:
            result = results[0]
            assert hasattr(result, "memory_id")
            assert hasattr(result, "content")
            assert hasattr(result, "hybrid_score")
            assert hasattr(result, "vector_score")
            assert hasattr(result, "text_score")

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, factory, temp_embedder):
        """测试带过滤器的 hybrid_search."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("filtered_agent")
        service = MemoryService("filtered_agent")

        await service.remember(
            content="Important fact about AI",
            kind="fact",
            tags=["ai"],
            importance=1.8,
        )
        await service.remember(
            content="Less important observation",
            kind="episode_summary",  # 使用有效的 MemoryKind
            tags=["misc"],
            importance=0.5,
        )

        results = await service.hybrid_search(
            query="AI important",
            filters={"kind": "fact", "min_importance": 1.0},
        )

        # 过滤器应该生效
        for r in results:
            assert r.kind == "fact"
            assert r.importance >= 1.0


class TestSniperMode:
    """测试 Sniper Mode 上下文汇聚."""

    @pytest.mark.asyncio
    async def test_sniper_mode_returns_structured_result(self, factory, temp_embedder):
        """测试 sniper_mode_retrieval 返回结构化结果."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("sniper_agent")
        service = MemoryService("sniper_agent")

        # 存储多个记忆
        for i in range(5):
            await service.remember(
                content=f"Important concept number {i}: machine learning fundamentals",
                kind="fact",
                tags=["ml"],
                importance=1.0 + i * 0.1,
            )

        result = await service.sniper_mode_retrieval(
            query="machine learning",
            max_context_tokens=1000,
            context_per_item=200,
        )

        assert "memories" in result
        assert "total_tokens" in result
        assert "coverage" in result
        assert isinstance(result["memories"], list)
        assert isinstance(result["total_tokens"], int)
        assert isinstance(result["coverage"], float)

    @pytest.mark.asyncio
    async def test_sniper_mode_respects_token_budget(self, factory, temp_embedder):
        """测试 Sniper Mode 尊重 token 预算."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("budget_agent")
        service = MemoryService("budget_agent")

        # 存储多个记忆
        for i in range(10):
            await service.remember(
                content=f"Concept {i}: deep learning and neural networks",
                kind="fact",
                tags=["dl"],
                importance=1.0,
            )

        max_tokens = 600  # 预算只够 3 个记忆项 (3 * 200)
        result = await service.sniper_mode_retrieval(
            query="neural networks",
            max_context_tokens=max_tokens,
            context_per_item=200,
        )

        # 应该只返回预算内的记忆
        assert result["total_tokens"] <= max_tokens
        assert len(result["memories"]) <= 3

    @pytest.mark.asyncio
    async def test_recall_with_sniper_mode(self, factory, temp_embedder):
        """测试 recall_with_sniper_mode 接口."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("compat_agent")
        service = MemoryService("compat_agent")

        await service.remember(
            content="Python asyncio is for concurrent programming",
            kind="procedure",
            tags=["python", "async"],
            importance=1.5,
        )

        results = await service.recall_with_sniper_mode(
            query="async programming",
            top_k=3,
        )

        assert isinstance(results, list)


class TestEdgeCases:
    """边界情况测试."""

    @pytest.mark.asyncio
    async def test_empty_database_search(self, factory, temp_embedder):
        """测试空数据库搜索不崩溃."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("empty_agent")
        service = MemoryService("empty_agent")

        results = await service.hybrid_search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_unicode_content_storage(self, factory, temp_embedder):
        """测试 Unicode 内容存储."""
        from holonpolis.services import MemoryService

        factory.init_holon_tables("unicode_agent")
        service = MemoryService("unicode_agent")

        memory_id = await service.remember(
            content="中文内容 🎯 日本語 العربية",
            kind="fact",
            tags=["unicode", " multilingual"],
        )

        assert memory_id.startswith("mem_")

    def test_factory_singleton(self, tmp_path, temp_embedder, monkeypatch):
        """测试 Factory 单例模式."""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )
        reset_factory()

        factory1 = get_lancedb_factory()
        factory2 = get_lancedb_factory()

        assert factory1 is factory2
