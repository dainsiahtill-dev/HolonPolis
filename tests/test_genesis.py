"""Phase 3 Tests: Genesis Engine - 创世引擎测试

验证:
1. Evolution Lord 决策引擎 (Route/Spawn/Deny/Clarify)
2. Genesis Memory (Holon 注册、路由记录)
3. Genesis Service 完整链路 (需求 -> 决策 -> 实体创建)
4. Blueprint 解析与验证
"""

import json
import tempfile
from pathlib import Path

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.blueprints import EvolutionStrategy
from holonpolis.genesis import (
    ClarifyDecision,
    DenyDecision,
    EvolutionLord,
    GenesisMemory,
    RouteDecision,
    SpawnDecision,
)
from holonpolis.genesis.evolution_lord import DecisionType
from holonpolis.kernel.embeddings.default_embedder import set_embedder, SimpleEmbedder
from holonpolis.kernel.lancedb import get_lancedb_factory, reset_factory


@pytest.fixture
def temp_embedder():
    """使用 SimpleEmbedder 避免 API 调用."""
    embedder = SimpleEmbedder(dimension=128)
    set_embedder(embedder)
    return embedder


@pytest.fixture
def genesis_setup(tmp_path, temp_embedder, monkeypatch):
    """设置 Genesis 测试环境."""
    holonpolis_root = tmp_path / ".holonpolis"
    monkeypatch.setattr(
        "holonpolis.config.settings.holonpolis_root",
        holonpolis_root
    )
    # 同时更新 holons_path 和 genesis_memory_path
    monkeypatch.setattr(
        "holonpolis.config.settings.holons_path",
        holonpolis_root / "holons"
    )
    monkeypatch.setattr(
        "holonpolis.config.settings.genesis_memory_path",
        holonpolis_root / "genesis" / "memory" / "lancedb"
    )
    reset_factory()

    # 初始化 Genesis 表
    factory = get_lancedb_factory(base_path=holonpolis_root)
    factory.init_genesis_tables()

    return tmp_path


class TestEvolutionLord:
    """Evolution Lord 决策引擎测试."""

    @pytest.fixture
    def evolution_lord(self):
        """创建 Evolution Lord 实例."""
        return EvolutionLord()

    def test_load_system_prompt(self, evolution_lord):
        """测试加载 System Prompt."""
        assert evolution_lord._system_prompt is not None
        assert len(evolution_lord._system_prompt) > 0
        assert "Evolution Lord" in evolution_lord._system_prompt

    def test_load_blueprint_schema(self, evolution_lord):
        """测试加载 Blueprint Schema."""
        assert evolution_lord._blueprint_schema is not None
        assert isinstance(evolution_lord._blueprint_schema, dict)

    def test_parse_route_decision(self, evolution_lord):
        """测试解析 Route 决策."""
        data = {
            "decision": "route_to",
            "holon_id": "holon_abc123",
            "confidence": 0.85,
            "reasoning": "This Holon is appropriate for the task",
            "context_to_inject": {"key": "value"},
        }

        decision = evolution_lord._parse_decision(data)

        assert isinstance(decision, RouteDecision)
        assert decision.holon_id == "holon_abc123"
        assert decision.confidence == 0.85
        assert decision.decision == "route_to"

    def test_parse_spawn_decision(self, evolution_lord):
        """测试解析 Spawn 决策."""
        data = {
            "decision": "spawn",
            "blueprint": {
                "species_id": "generalist",
                "name": "Test Holon",
                "purpose": "Handle test requests",
                "boundary": {
                    "allowed_tools": ["search", "calculate"],
                    "denied_tools": [],
                    "max_episodes_per_hour": 100,
                    "max_tokens_per_episode": 10000,
                    "allow_file_write": False,
                    "allow_network": False,
                    "allow_subprocess": False,
                },
                "evolution_policy": {
                    "strategy": "balanced",
                    "auto_promote_to_global": False,
                    "require_tests": True,
                    "max_evolution_attempts": 3,
                },
                "initial_memory_tags": ["test"],
            },
            "confidence": 0.9,
            "reasoning": "Need a new Holon for this task",
        }

        decision = evolution_lord._parse_decision(data)

        assert isinstance(decision, SpawnDecision)
        assert decision.blueprint is not None
        assert decision.blueprint.species_id == "generalist"
        assert decision.blueprint.name == "Test Holon"
        assert decision.decision == "spawn"

    def test_parse_deny_decision(self, evolution_lord):
        """测试解析 Deny 决策."""
        data = {
            "decision": "deny",
            "reason": "Security violation",
            "suggested_alternative": "Try a different approach",
        }

        decision = evolution_lord._parse_decision(data)

        assert isinstance(decision, DenyDecision)
        assert decision.reason == "Security violation"
        assert decision.suggested_alternative == "Try a different approach"

    def test_parse_clarify_decision(self, evolution_lord):
        """测试解析 Clarify 决策."""
        data = {
            "decision": "clarify",
            "question": "What specifically do you need?",
        }

        decision = evolution_lord._parse_decision(data)

        assert isinstance(decision, ClarifyDecision)
        assert decision.question == "What specifically do you need?"

    def test_parse_unknown_decision(self, evolution_lord):
        """测试解析未知决策类型."""
        data = {"decision": "unknown_type"}

        decision = evolution_lord._parse_decision(data)

        assert isinstance(decision, ClarifyDecision)

    def test_create_blueprint_from_decision(self, evolution_lord):
        """测试从决策数据创建 Blueprint."""
        data = {
            "species_id": "specialist",
            "name": "Code Reviewer",
            "purpose": "Review code for quality",
            "boundary": {
                "allowed_tools": ["read_file", "analyze_code"],
                "allow_file_write": False,
            },
            "evolution_policy": {
                "strategy": "conservative",
            },
            "initial_memory_tags": ["code_review", "quality"],
        }

        blueprint = evolution_lord._create_blueprint_from_decision(data)

        assert isinstance(blueprint, Blueprint)
        assert blueprint.species_id == "specialist"
        assert blueprint.name == "Code Reviewer"
        assert blueprint.boundary.allow_file_write is False
        assert blueprint.evolution_policy.strategy == EvolutionStrategy.CONSERVATIVE
        assert blueprint.blueprint_id.startswith("blueprint_")
        assert blueprint.holon_id.startswith("holon_")


class TestGenesisMemory:
    """Genesis Memory 测试."""

    @pytest.fixture
    def genesis_memory(self):
        """创建 Genesis Memory 实例."""
        return GenesisMemory()

    @pytest.mark.asyncio
    async def test_register_holon(self, genesis_setup, genesis_memory):
        """测试注册 Holon."""
        await genesis_memory.register_holon(
            holon_id="holon_test_001",
            blueprint_id="blueprint_001",
            species_id="generalist",
            name="Test Holon",
            purpose="Testing Genesis memory",
            capabilities=["search", "calculate"],
            skills=["basic_math"],
        )

        # 验证可以通过 list_active_holons 找到
        holons = await genesis_memory.list_active_holons()
        assert len(holons) >= 1

        holon_ids = [h["holon_id"] for h in holons]
        assert "holon_test_001" in holon_ids

    @pytest.mark.asyncio
    async def test_find_holons_for_task(self, genesis_setup, genesis_memory, temp_embedder):
        """测试语义搜索 Holon."""
        # 注册多个 Holons
        await genesis_memory.register_holon(
            holon_id="holon_code",
            blueprint_id="bp_code",
            species_id="specialist",
            name="Code Expert",
            purpose="Handle programming and coding tasks",
            capabilities=["code_analysis", "refactoring"],
            skills=[],
        )
        await genesis_memory.register_holon(
            holon_id="holon_data",
            blueprint_id="bp_data",
            species_id="analyst",
            name="Data Analyst",
            purpose="Process and analyze data",
            capabilities=["statistics", "visualization"],
            skills=[],
        )

        # 搜索与编程相关的任务
        results = await genesis_memory.find_holons_for_task(
            "I need help with Python programming",
            top_k=5,
        )

        # 应该返回结果
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_record_route_decision(self, genesis_setup, genesis_memory):
        """测试记录路由决策."""
        route_id = await genesis_memory.record_route_decision(
            query="Help me with Python",
            decision="route_to",
            target_holon_id="holon_python",
            spawned_blueprint_id=None,
            reasoning="Python Holon is best suited",
        )

        assert route_id.startswith("route_")

    @pytest.mark.asyncio
    async def test_record_evolution(self, genesis_setup, genesis_memory):
        """测试记录演化尝试."""
        evolution_id = await genesis_memory.record_evolution(
            holon_id="holon_test",
            skill_name="file_parser",
            status="success",
            attestation_id="att_123",
        )

        assert evolution_id.startswith("evo_")

        # 验证可以获取
        evolutions = await genesis_memory.get_holon_evolutions("holon_test")
        assert len(evolutions) >= 1
        assert evolutions[0]["skill_name"] == "file_parser"


class TestBlueprintSerialization:
    """Blueprint 序列化测试."""

    def test_blueprint_to_dict(self):
        """测试 Blueprint 转字典."""
        blueprint = Blueprint(
            blueprint_id="bp_001",
            holon_id="holon_001",
            species_id="generalist",
            name="Test",
            purpose="Testing",
            boundary=Boundary(
                allowed_tools=["tool1"],
                allow_file_write=True,
            ),
            evolution_policy=EvolutionPolicy(
                strategy=EvolutionStrategy.BALANCED,
            ),
            initial_memory_tags=["test"],
        )

        data = blueprint.to_dict()

        assert data["blueprint_id"] == "bp_001"
        assert data["holon_id"] == "holon_001"
        assert data["boundary"]["allowed_tools"] == ["tool1"]
        assert data["boundary"]["allow_file_write"] is True
        assert data["evolution_policy"]["strategy"] == "balanced"

    def test_blueprint_from_dict(self):
        """测试从字典创建 Blueprint."""
        data = {
            "blueprint_id": "bp_002",
            "holon_id": "holon_002",
            "species_id": "specialist",
            "name": "Specialist Test",
            "purpose": "Specialized testing",
            "boundary": {
                "allowed_tools": ["tool2"],
                "denied_tools": ["dangerous"],
                "max_episodes_per_hour": 50,
            },
            "evolution_policy": {
                "strategy": "conservative",
                "require_tests": True,
            },
            "initial_memory_tags": ["specialist"],
            "created_at": "2024-01-01T00:00:00",
        }

        blueprint = Blueprint.from_dict(data)

        assert blueprint.blueprint_id == "bp_002"
        assert blueprint.species_id == "specialist"
        assert blueprint.boundary.max_episodes_per_hour == 50
        assert blueprint.evolution_policy.strategy == EvolutionStrategy.CONSERVATIVE

    def test_blueprint_roundtrip(self):
        """测试 Blueprint 序列化往返."""
        original = Blueprint(
            blueprint_id="bp_003",
            holon_id="holon_003",
            species_id="worker",
            name="Worker Test",
            purpose="Work work work",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(),
        )

        data = original.to_dict()
        restored = Blueprint.from_dict(data)

        assert restored.blueprint_id == original.blueprint_id
        assert restored.species_id == original.species_id
        assert restored.name == original.name


class TestDecisionTypes:
    """决策类型测试."""

    def test_route_decision_defaults(self):
        """测试 RouteDecision 默认值."""
        decision = RouteDecision()

        assert decision.decision == "route_to"
        assert decision.holon_id == ""
        assert decision.confidence == 0.0
        assert decision.reasoning == ""
        assert decision.context_to_inject == {}

    def test_spawn_decision_defaults(self):
        """测试 SpawnDecision 默认值."""
        decision = SpawnDecision()

        assert decision.decision == "spawn"
        assert decision.blueprint is None
        assert decision.confidence == 0.0
        assert decision.reasoning == ""

    def test_deny_decision_defaults(self):
        """测试 DenyDecision 默认值."""
        decision = DenyDecision()

        assert decision.decision == "deny"
        assert decision.reason == ""
        assert decision.suggested_alternative is None

    def test_clarify_decision_defaults(self):
        """测试 ClarifyDecision 默认值."""
        decision = ClarifyDecision()

        assert decision.decision == "clarify"
        assert decision.question == ""


class TestGenesisIntegration:
    """Genesis 集成测试."""

    @pytest.mark.asyncio
    async def test_full_spawn_workflow(self, genesis_setup):
        """测试完整的 Spawn 工作流."""
        from holonpolis.services import GenesisService

        # 创建 Genesis Service
        genesis_svc = GenesisService()

        # 模拟 Spawn 决策
        blueprint = Blueprint(
            blueprint_id="bp_spawn_test",
            holon_id="holon_spawn_test",
            species_id="generalist",
            name="Spawned Holon",
            purpose="Test spawn workflow",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(),
        )

        # 创建 Holon
        from holonpolis.services import HolonService
        holon_svc = HolonService()
        holon_id = await holon_svc.create_holon(blueprint)

        assert holon_id == "holon_spawn_test"

        # 验证 Holon 目录结构
        holon_path = genesis_setup / ".holonpolis" / "holons" / holon_id
        assert holon_path.exists(), f"Holon path {holon_path} does not exist"
        blueprint_path = holon_path / "blueprint.json"
        assert blueprint_path.exists(), f"Blueprint not found at {blueprint_path}"
        assert (holon_path / "workspace").exists()
        assert (holon_path / "skills_local").exists()

    @pytest.mark.asyncio
    async def test_holon_lifecycle(self, genesis_setup):
        """测试 Holon 生命周期 (create -> freeze -> resume -> delete)."""
        from holonpolis.services import HolonService

        holon_svc = HolonService()

        # 1. 创建
        blueprint = Blueprint(
            blueprint_id="bp_lifecycle",
            holon_id="holon_lifecycle",
            species_id="worker",
            name="Lifecycle Test",
            purpose="Test lifecycle",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(),
        )
        holon_id = await holon_svc.create_holon(blueprint)

        # 2. 冻结
        holon_svc.freeze_holon(holon_id)
        assert holon_svc.is_frozen(holon_id) is True

        # 3. 恢复
        holon_svc.resume_holon(holon_id)
        assert holon_svc.is_frozen(holon_id) is False

        # 4. 删除 (在删除前重新获取服务以确保路径正确)
        from holonpolis.services import HolonService
        holon_svc_fresh = HolonService()
        holon_svc_fresh.delete_holon(holon_id)
        assert holon_svc_fresh.holon_exists(holon_id) is False


class TestEdgeCases:
    """边界情况测试."""

    def test_evolution_lord_with_missing_prompt_file(self, tmp_path, monkeypatch):
        """测试 Evolution Lord 处理缺失的 Prompt 文件."""
        # 模拟不存在的路径
        monkeypatch.setattr(
            Path,
            "exists",
            lambda self: False if "genesis_system_prompt" in str(self) else True
        )

        lord = EvolutionLord()
        assert lord._system_prompt == "You are the Evolution Lord. Make routing decisions."

    def test_blueprint_from_dict_missing_fields(self):
        """测试从缺少字段的字典创建 Blueprint."""
        data = {
            "blueprint_id": "bp_minimal",
            "holon_id": "holon_minimal",
            "species_id": "minimal",
            "name": "Minimal",
            "purpose": "Test minimal fields",
        }

        blueprint = Blueprint.from_dict(data)

        assert blueprint.boundary is not None
        assert blueprint.evolution_policy is not None

    @pytest.mark.asyncio
    async def test_genesis_memory_empty_database(self, genesis_setup):
        """测试 Genesis Memory 处理空数据库."""
        memory = GenesisMemory()

        holons = await memory.list_active_holons()
        assert isinstance(holons, list)

        evolutions = await memory.get_holon_evolutions("nonexistent")
        assert isinstance(evolutions, list)
