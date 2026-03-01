"""Runtime capability tests for evolved skills and social graph wrappers."""

import asyncio

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.genesis.genesis_memory import GenesisMemory
from holonpolis.domain.skills import ToolSchema
from holonpolis.kernel.embeddings.default_embedder import SimpleEmbedder, set_embedder
from holonpolis.kernel.llm.llm_runtime import LLMResponse, LLMUsage
from holonpolis.kernel.lancedb import reset_factory
from holonpolis.runtime.holon_runtime import (
    CapabilityDeniedError,
    EvolutionRequest,
    EvolutionStatus,
    HolonRuntime,
    SkillPayloadValidationError,
)
from holonpolis.services.collaboration_service import CollaborationService
from holonpolis.services.evolution_service import EvolutionResult, EvolutionService
from holonpolis.services.holon_service import HolonService, HolonUnavailableError
from holonpolis.services.market_service import MarketService


def _blueprint(holon_id: str) -> Blueprint:
    return Blueprint(
        blueprint_id=f"bp_{holon_id}",
        holon_id=holon_id,
        species_id="generalist",
        name=f"{holon_id}-name",
        purpose="Runtime capability tests",
        boundary=Boundary(),
        evolution_policy=EvolutionPolicy(),
    )


@pytest.fixture
def runtime_setup(tmp_path, monkeypatch):
    """Isolate runtime storage and disable external embedding dependencies."""
    root = tmp_path / ".holonpolis"
    monkeypatch.setattr("holonpolis.config.settings.holonpolis_root", root)
    monkeypatch.setattr("holonpolis.config.settings.holons_path", root / "holons")
    monkeypatch.setattr(
        "holonpolis.config.settings.genesis_memory_path",
        root / "genesis" / "memory" / "lancedb",
    )
    set_embedder(SimpleEmbedder(dimension=128))
    reset_factory()
    MarketService.reset_in_memory_cache()
    CollaborationService.reset_in_memory_cache()
    return root


@pytest.mark.asyncio
async def test_runtime_can_load_and_execute_persisted_skill(runtime_setup):
    """Runtime should load evolved skill manifests and execute entrypoints."""
    holon_id = "runtime_holon_skill"
    service = EvolutionService()
    schema = ToolSchema(
        name="execute",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        },
        required=["a", "b"],
    )

    await service._phase_persist(
        holon_id=holon_id,
        skill_name="Adder Skill",
        code="""
def execute(a, b):
    return a + b
""",
        tests="""
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
""",
        description="Simple adder",
        tool_schema=schema,
        version="0.1.0",
        green_result={"passed": True, "details": {"exit_code": 0}},
        verify_result={"passed": True, "violations": []},
    )

    runtime = HolonRuntime(holon_id=holon_id, blueprint=_blueprint(holon_id))
    loaded = runtime.get_skill("Adder Skill")
    assert loaded.skill_id == "adder_skill"

    direct_result = await loaded.execute({"a": 2, "b": 8})
    assert direct_result == 10

    wrapper_result = await runtime.execute_skill("adder_skill", a=7, b=5)
    assert wrapper_result == 12

    skills = runtime.list_skills()
    assert any(item["skill_id"] == "adder_skill" for item in skills)


@pytest.mark.asyncio
async def test_runtime_social_relationship_wrappers(runtime_setup, monkeypatch):
    """Runtime wrappers should persist relationship edges and support trust propagation."""
    runtime = HolonRuntime(
        holon_id="social_holon_a",
        blueprint=_blueprint("social_holon_a"),
    )

    async def _noop_remember(*args, **kwargs):
        return "mem_noop"

    monkeypatch.setattr(runtime, "remember", _noop_remember)

    relationship_id = await runtime.register_relationship(
        target_holon_id="social_holon_b",
        relationship_type="mentor",
        strength=0.7,
        trust_score=0.8,
    )
    assert relationship_id.startswith("rel_")

    trust = await runtime.propagate_trust("social_holon_b", max_hops=2)
    assert trust == pytest.approx(0.8)

    service = CollaborationService()
    rels = service.social_graph.get_relationships("social_holon_a")
    assert any(rel.relationship_id == relationship_id for rel in rels)


@pytest.mark.asyncio
async def test_runtime_execute_skill_enforces_schema(runtime_setup):
    """execute_skill should strictly validate payload against tool schema."""
    holon_id = "runtime_schema_holon"
    service = EvolutionService()
    schema = ToolSchema(
        name="execute",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
            },
            "additionalProperties": False,
        },
        required=["x", "y"],
    )

    await service._phase_persist(
        holon_id=holon_id,
        skill_name="Multiplier Skill",
        code="""
def execute(x, y):
    return x * y
""",
        tests="""
from skill_module import execute

def test_execute():
    assert execute(3, 4) == 12
""",
        description="Simple multiplier",
        tool_schema=schema,
        version="0.1.0",
        green_result={"passed": True, "details": {"exit_code": 0}},
        verify_result={"passed": True, "violations": []},
    )

    runtime = HolonRuntime(holon_id=holon_id, blueprint=_blueprint(holon_id))

    with pytest.raises(SkillPayloadValidationError):
        await runtime.execute_skill("multiplier_skill", payload={"x": 2, "extra": 1})


@pytest.mark.asyncio
async def test_runtime_selection_denied_by_boundary_policy(runtime_setup):
    """High-impact operations should be blocked by boundary capability mapping."""
    holon_id = "runtime_policy_holon"
    restricted_blueprint = Blueprint(
        blueprint_id=f"bp_{holon_id}",
        holon_id=holon_id,
        species_id="generalist",
        name="Restricted Runtime",
        purpose="Policy test",
        boundary=Boundary(allowed_tools=["market.read"], denied_tools=[]),
        evolution_policy=EvolutionPolicy(),
    )

    runtime = HolonRuntime(holon_id=holon_id, blueprint=restricted_blueprint)
    with pytest.raises(CapabilityDeniedError):
        await runtime.run_selection(threshold=0.5)


@pytest.mark.asyncio
async def test_runtime_evolution_status_survives_genesis_audit_failure(runtime_setup, monkeypatch):
    """Evolution should remain completed even if Genesis audit persistence fails."""
    holon_id = "runtime_evo_resilience"
    blueprint = _blueprint(holon_id)
    await HolonService().create_holon(blueprint)
    runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    async def fake_evolve_skill_autonomous(self, **kwargs):
        persist_result = await self._phase_persist(
            holon_id=kwargs["holon_id"],
            skill_name=kwargs["skill_name"],
            code="""
def execute(a, b):
    return a + b
""",
            tests="""
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
""",
            description=kwargs["description"],
            tool_schema=kwargs["tool_schema"],
            version="0.1.0",
            green_result={"passed": True, "details": {"exit_code": 0}},
            verify_result={"passed": True, "violations": []},
        )
        self._holon_service.mark_active(
            kwargs["holon_id"],
            reason="fake_evolution_complete",
            details={"service": "test_runtime_capabilities"},
        )
        return EvolutionResult(
            success=True,
            phase="complete",
            skill_id=persist_result["skill_id"],
            attestation=persist_result["attestation"],
            code_path=persist_result["code_path"],
            test_path=persist_result["test_path"],
            manifest_path=persist_result["manifest_path"],
        )

    async def fake_record_evolution(self, *args, **kwargs):
        raise RuntimeError("genesis evolutions table unavailable")

    monkeypatch.setattr(EvolutionService, "evolve_skill_autonomous", fake_evolve_skill_autonomous)
    monkeypatch.setattr(GenesisMemory, "record_evolution", fake_record_evolution)

    request = await runtime.request_evolution(
        skill_name="ResilientAdder",
        description="Add two numbers",
        requirements=["Expose execute(a, b)", "Return a + b"],
    )
    status = await runtime.wait_for_evolution(
        request_id=request.request_id,
        timeout_seconds=20,
        poll_interval_seconds=0.1,
    )

    assert status.status == EvolutionStatus.COMPLETED
    assert status.result is not None
    assert status.result["skill_id"] == "resilientadder"
    assert await runtime.execute_skill("resilientadder", a=9, b=4) == 13


@pytest.mark.asyncio
async def test_runtime_can_index_and_search_ui_component_library(runtime_setup):
    """UI component libraries should be indexable into memory and searchable."""
    holon_id = "runtime_ui_library"
    runtime = HolonRuntime(holon_id=holon_id, blueprint=_blueprint(holon_id))

    ui_root = runtime_setup / "ui_library"
    ui_root.mkdir(parents=True, exist_ok=True)
    (ui_root / "Button.tsx").write_text(
        """
import React from "react";

export function Button() {
    return <button className="btn-primary">Save</button>;
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (ui_root / "SearchInput.tsx").write_text(
        """
import React from "react";

export const SearchInput = () => {
    return <input placeholder="Search" />;
};
""".strip()
        + "\n",
        encoding="utf-8",
    )

    first_index = await runtime.index_ui_component_library(
        source_path=str(ui_root),
        library_name="acme-ui",
        framework="react",
    )
    assert first_index["indexed_components"] == 2
    assert first_index["reused_components"] == 0

    second_index = await runtime.index_ui_component_library(
        source_path=str(ui_root),
        library_name="acme-ui",
        framework="react",
    )
    assert second_index["indexed_components"] == 0
    assert second_index["reused_components"] == 2

    hits = await runtime.search_ui_component_library("loading button", top_k=2)
    assert hits
    assert any(hit["component_name"] == "Button" for hit in hits)
    button_hit = next(hit for hit in hits if hit["component_name"] == "Button")
    assert "button" in button_hit["code_content"].lower()


@pytest.mark.asyncio
async def test_runtime_can_index_and_search_reusable_code_library(runtime_setup):
    """Generic code libraries should be indexed and retrievable for reuse."""
    holon_id = "runtime_code_library"
    runtime = HolonRuntime(holon_id=holon_id, blueprint=_blueprint(holon_id))

    code_root = runtime_setup / "shared_sdk"
    code_root.mkdir(parents=True, exist_ok=True)
    (code_root / "client.py").write_text(
        """
class ApiClient:
    def request(self, endpoint: str) -> dict:
        return {"endpoint": endpoint, "signed": True}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = await runtime.index_reusable_code_library(
        source_path=str(code_root),
        library_name="shared-sdk",
        library_kind="code_asset",
        framework="python",
    )
    assert result["indexed_assets"] == 1

    hits = await runtime.search_reusable_code_library(
        "api client request signing",
        top_k=2,
    )
    assert hits
    assert hits[0]["asset_name"] == "ApiClient"
    assert "signed" in hits[0]["code_content"].lower()


@pytest.mark.asyncio
async def test_chat_injects_ui_component_context_for_frontend_requests(runtime_setup, monkeypatch):
    """chat() should inject retrieved UI component snippets for frontend prompts."""
    holon_id = "runtime_ui_prompt"
    runtime = HolonRuntime(holon_id=holon_id, blueprint=_blueprint(holon_id))

    ui_root = runtime_setup / "ui_prompt_library"
    ui_root.mkdir(parents=True, exist_ok=True)
    (ui_root / "Button.tsx").write_text(
        """
import React from "react";

export function Button() {
    return <button className="btn-primary">Save</button>;
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    await runtime.index_ui_component_library(
        source_path=str(ui_root),
        library_name="prompt-ui",
        framework="react",
    )

    captured: dict = {}

    async def fake_chat_with_history(messages, config=None):
        captured["system_prompt"] = messages[0].content
        return LLMResponse(
            content="ok",
            usage=LLMUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16),
        )

    monkeypatch.setattr(runtime.llm, "chat_with_history", fake_chat_with_history)

    result = await runtime.chat("请帮我构建一个带保存按钮的前端页面")
    assert result["content"] == "ok"
    assert "# Retrieved UI Components" in captured["system_prompt"]
    assert "Button" in captured["system_prompt"]


@pytest.mark.asyncio
async def test_chat_injects_reusable_code_context_for_generation_requests(runtime_setup, monkeypatch):
    """chat() should inject reusable code assets for generic code generation prompts."""
    holon_id = "runtime_code_prompt"
    runtime = HolonRuntime(holon_id=holon_id, blueprint=_blueprint(holon_id))

    code_root = runtime_setup / "code_prompt_library"
    code_root.mkdir(parents=True, exist_ok=True)
    (code_root / "client.py").write_text(
        """
class ApiClient:
    def request(self, endpoint: str) -> dict:
        return {"endpoint": endpoint, "signed": True}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    await runtime.index_reusable_code_library(
        source_path=str(code_root),
        library_name="shared-sdk",
        library_kind="code_asset",
        framework="python",
    )

    captured: dict = {}

    async def fake_chat_with_history(messages, config=None):
        captured["system_prompt"] = messages[0].content
        return LLMResponse(
            content="ok",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
        )

    monkeypatch.setattr(runtime.llm, "chat_with_history", fake_chat_with_history)

    result = await runtime.chat("请为新项目构建一个可复用的 API client SDK")
    assert result["content"] == "ok"
    assert "# Retrieved Reusable Code Assets" in captured["system_prompt"]
    assert "ApiClient" in captured["system_prompt"]


@pytest.mark.asyncio
async def test_runtime_blocks_skill_execution_while_evolution_is_pending(runtime_setup, monkeypatch):
    """A Holon should not execute skills while it has an in-flight self-evolution."""
    holon_id = "runtime_pending_block"
    blueprint = _blueprint(holon_id)
    await HolonService().create_holon(blueprint)

    service = EvolutionService()
    schema = ToolSchema(
        name="execute",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        },
        required=["a", "b"],
    )
    await service._phase_persist(
        holon_id=holon_id,
        skill_name="Adder Skill",
        code="""
def execute(a, b):
    return a + b
""",
        tests="""
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
""",
        description="Simple adder",
        tool_schema=schema,
        version="0.1.0",
        green_result={"passed": True, "details": {"exit_code": 0}},
        verify_result={"passed": True, "violations": []},
    )

    runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)
    gate = asyncio.Event()

    async def fake_execute_evolution(self, request):
        request.status = EvolutionStatus.EVOLVING
        await gate.wait()
        request.status = EvolutionStatus.FAILED
        request.error_message = "cancelled_for_test"
        request.completed_at = "2026-03-01T00:00:00+00:00"
        self._holon_service.mark_active(
            self.holon_id,
            reason="test_pending_released",
            details={"service": "test_runtime_capabilities", "request_id": request.request_id},
        )

    monkeypatch.setattr(HolonRuntime, "_execute_evolution", fake_execute_evolution)

    request = await runtime.request_evolution(
        skill_name="BlockedDuringUpgrade",
        description="Should keep runtime pending",
        requirements=["Any requirement"],
    )

    with pytest.raises(HolonUnavailableError):
        await runtime.execute_skill("adder_skill", a=2, b=3)

    gate.set()
    status = await runtime.wait_for_evolution(
        request_id=request.request_id,
        timeout_seconds=5,
        poll_interval_seconds=0.05,
    )
    assert status.status == EvolutionStatus.FAILED
    assert await runtime.execute_skill("adder_skill", a=2, b=3) == 5


@pytest.mark.asyncio
async def test_failed_evolution_produces_next_round_plan(runtime_setup, monkeypatch):
    """Failed evolution should leave behind a concrete self-improvement plan."""
    holon_id = "runtime_failure_plan"
    blueprint = _blueprint(holon_id)
    await HolonService().create_holon(blueprint)
    runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    async def fake_evolve_skill_autonomous(self, **kwargs):
        return EvolutionResult(
            success=False,
            phase="green",
            error_message="pytest failed because execute raised ValueError on empty input",
        )

    async def fake_record_evolution(self, *args, **kwargs):
        return None

    monkeypatch.setattr(EvolutionService, "evolve_skill_autonomous", fake_evolve_skill_autonomous)
    monkeypatch.setattr(GenesisMemory, "record_evolution", fake_record_evolution)

    request = await runtime.request_evolution(
        skill_name="PlannerSkill",
        description="Should fail and produce a retry plan",
        requirements=["Return a deterministic execute payload"],
    )
    status = await runtime.wait_for_evolution(
        request_id=request.request_id,
        timeout_seconds=5,
        poll_interval_seconds=0.05,
    )

    assert status.status == EvolutionStatus.FAILED
    assert isinstance(status.result, dict)
    assert status.result["retry_recommended"] is True
    assert status.result["next_round_plan"]["focus"] == "stabilize_execution"
    assert "revised_requirements" in status.result["next_round_plan"]


@pytest.mark.asyncio
async def test_failed_evolution_can_schedule_reflective_follow_up(runtime_setup, monkeypatch):
    """A failed evolution should schedule one follow-up request within policy budget."""
    holon_id = "runtime_follow_up_retry"
    blueprint = _blueprint(holon_id)
    await HolonService().create_holon(blueprint)
    runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    captured: list[dict[str, object]] = []

    async def fake_request_evolution(
        *,
        skill_name,
        description,
        requirements,
        test_cases=None,
        parent_skills=None,
        origin="manual",
        attempt_index=1,
        lineage_id=None,
    ):
        captured.append(
            {
                "skill_name": skill_name,
                "description": description,
                "requirements": list(requirements),
                "origin": origin,
                "attempt_index": attempt_index,
                "lineage_id": lineage_id,
            }
        )
        return EvolutionRequest(
            request_id="evo_follow_up_001",
            holon_id=holon_id,
            skill_name=skill_name,
            description=description,
            requirements=list(requirements),
            test_cases=list(test_cases or []),
            parent_skills=list(parent_skills or []),
            status=EvolutionStatus.PENDING,
            created_at="2026-03-01T00:00:00+00:00",
            origin=origin,
            attempt_index=attempt_index,
            lineage_id=lineage_id,
        )

    monkeypatch.setattr(runtime, "request_evolution", fake_request_evolution)

    request = EvolutionRequest(
        request_id="evo_origin_001",
        holon_id=holon_id,
        skill_name="RetryableSkill",
        description="Stabilize a brittle skill",
        requirements=["Return deterministic output"],
        test_cases=[],
        parent_skills=[],
        status=EvolutionStatus.FAILED,
        created_at="2026-03-01T00:00:00+00:00",
        attempt_index=1,
    )
    result = EvolutionResult(
        success=False,
        phase="green",
        error_message="pytest failed because execute raised ValueError on empty input",
    )

    await runtime._learn_from_evolution_failure(request, result)

    assert request.result is not None
    assert request.result["follow_up"]["triggered"] is True
    assert captured[0]["skill_name"] == "RetryableSkill"
    assert captured[0]["origin"] == "self_reflection_recovery"
    assert captured[0]["attempt_index"] == 2
    assert captured[0]["lineage_id"] == "evo_origin_001"
    assert any(
        "Address previous green failure:" in item
        for item in captured[0]["requirements"]
    )

    state = HolonService().get_holon_state(holon_id)
    reflection = state.get("self_reflection")
    assert isinstance(reflection, dict)
    assert reflection["history"][0]["metrics"]["continuation_allowed"] is True
    assert reflection["history"][0]["auto_evolution"]["triggered"] is True


@pytest.mark.asyncio
async def test_self_improve_persists_reflection_and_auto_evolves_once(runtime_setup, monkeypatch):
    """Self-reflection should persist structured telemetry and turn top insights into one evolution."""
    holon_id = "runtime_self_improve"
    blueprint = _blueprint(holon_id)
    await HolonService().create_holon(blueprint)
    runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    await runtime.memory.write_episode(
        transcript=[{"role": "user", "content": "normalize payload"}],
        outcome="failure",
        outcome_details={"error": "payload missing required field email"},
        latency_ms=140,
    )
    await runtime.memory.write_episode(
        transcript=[{"role": "user", "content": "validate payload"}],
        outcome="failure",
        outcome_details={"error": "payload missing required field email"},
        latency_ms=120,
    )
    await runtime.memory.write_episode(
        transcript=[{"role": "user", "content": "ping"}],
        outcome="success",
        latency_ms=80,
    )

    triggered: list[dict[str, object]] = []

    async def fake_request_evolution(
        *,
        skill_name,
        description,
        requirements,
        test_cases=None,
        parent_skills=None,
        origin="manual",
        attempt_index=1,
        lineage_id=None,
    ):
        triggered.append(
            {
                "skill_name": skill_name,
                "description": description,
                "requirements": list(requirements),
                "origin": origin,
                "attempt_index": attempt_index,
                "lineage_id": lineage_id,
            }
        )
        return EvolutionRequest(
            request_id="evo_self_reflect_001",
            holon_id=holon_id,
            skill_name=skill_name,
            description=description,
            requirements=list(requirements),
            test_cases=list(test_cases or []),
            parent_skills=list(parent_skills or []),
            status=EvolutionStatus.PENDING,
            created_at="2026-03-01T00:00:00+00:00",
            origin=origin,
            attempt_index=attempt_index,
            lineage_id=lineage_id,
        )

    monkeypatch.setattr(runtime, "request_evolution", fake_request_evolution)

    result = await runtime.self_improve(
        auto_evolve=True,
        max_suggestions=3,
        max_evolution_requests=1,
    )

    assert result["status"] == "analyzed"
    assert result["metrics"]["failure_categories"]["contract"] >= 2
    assert any(gap["gap_id"] == "input_contracts" for gap in result["capability_gaps"])
    assert result["auto_evolution"]["triggered"] is True
    assert result["auto_evolution"]["request_count"] == 1
    assert triggered[0]["skill_name"] == "input_contract_guard"

    state = HolonService().get_holon_state(holon_id)
    reflection = state.get("self_reflection")
    assert isinstance(reflection, dict)
    assert reflection["latest_reflection_id"] == result["reflection_id"]
    assert reflection["history"][0]["reflection_id"] == result["reflection_id"]

    snapshot = runtime.get_self_reflection(history_limit=1)
    assert snapshot["reflection_id"] == result["reflection_id"]
    assert snapshot["history_count"] >= 1
    assert len(snapshot["history"]) == 1

    memory_rows = runtime.memory._get_connection().get_table("memories").search().limit(50).to_list()
    reflection_rows = [
        row for row in memory_rows
        if "self_reflection" in list(row.get("tags", []))
    ]
    assert reflection_rows
    assert any(row.get("kind") == "pattern" for row in reflection_rows)


@pytest.mark.asyncio
async def test_chat_prioritizes_self_reflection_guidance_in_prompt(runtime_setup, monkeypatch):
    """Chat should inject self-reflection guidance ahead of ordinary memory context."""
    holon_id = "runtime_chat_reflection"
    blueprint = _blueprint(holon_id)
    await HolonService().create_holon(blueprint)
    runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    reflection = {
        "status": "analyzed",
        "reflection_id": "reflect_chat_001",
        "created_at": "2026-03-01T00:00:00+00:00",
        "summary": "Recent schema failures show the holon must validate payloads before execution.",
        "capability_gaps": [
            {"gap_id": "input_contracts", "severity": "high"},
        ],
        "suggestions": [
            {"type": "evolve_skill", "suggested_skill": "input_contract_guard"},
        ],
        "auto_evolution": {
            "triggered": True,
            "request_count": 1,
            "requests": [
                {"request_id": "evo_chat_001", "skill_name": "input_contract_guard"},
            ],
        },
    }
    HolonService().record_self_reflection(holon_id, reflection)
    await runtime._safe_persist_reflection_memory(reflection, phase_tag="analysis")

    async def fake_recall(query, top_k=5):
        return [{"kind": "fact", "content": "General memory content"}]

    captured: dict[str, object] = {}

    async def fake_chat_with_history(messages, config=None, tools=None):
        captured["messages"] = messages
        return LLMResponse(
            content="ok",
            usage=LLMUsage(total_tokens=7),
            latency_ms=12,
            model="test-model",
        )

    monkeypatch.setattr(runtime.memory, "recall", fake_recall)
    monkeypatch.setattr(runtime.llm, "chat_with_history", fake_chat_with_history)

    result = await runtime.chat("How should I handle malformed payloads?")

    assert result["content"] == "ok"
    messages = captured["messages"]
    system_prompt = messages[0].content
    assert "# Self Reflection Guidance" in system_prompt
    assert "Latest reflection: Recent schema failures show the holon must validate payloads before execution." in system_prompt
    assert "Active gaps: input_contracts" in system_prompt
    assert "In-flight self-evolution: input_contract_guard (evo_chat_001)" in system_prompt
    assert "Retrieved lesson:" in system_prompt
    assert "# Relevant Memories" in system_prompt


@pytest.mark.asyncio
async def test_market_selection_skips_pending_holons(runtime_setup):
    """Selection should not evaluate pending Holons as survivors or eliminations."""
    active_id = "market_active_holon"
    pending_id = "market_pending_holon"
    service = HolonService()

    for holon_id in (active_id, pending_id):
        blueprint = _blueprint(holon_id)
        await service.create_holon(blueprint)

    service.mark_pending(pending_id, reason="upgrading")

    market = MarketService()
    result = market.run_selection(threshold=0.0)

    skipped_ids = {item["holon_id"] for item in result["skipped_list"]}
    assert pending_id in skipped_ids
    assert all(item["holon_id"] != pending_id for item in result["top_performers"])
    assert all(item["holon_id"] != pending_id for item in result["eliminated_list"])


@pytest.mark.asyncio
async def test_collaboration_skips_pending_participants(runtime_setup):
    """Pending Holons should not be enrolled as collaboration workers."""
    service = HolonService()
    coordinator_id = "collab_coordinator"
    active_worker_id = "collab_worker_active"
    pending_worker_id = "collab_worker_pending"

    for holon_id in (coordinator_id, active_worker_id, pending_worker_id):
        await service.create_holon(_blueprint(holon_id))

    service.mark_pending(pending_worker_id, reason="upgrading")

    collaboration = CollaborationService()
    task = await collaboration.create_collaboration(
        name="Pending Filter",
        description="Ensure pending workers are skipped",
        coordinator_id=coordinator_id,
        participant_ids=[active_worker_id, pending_worker_id],
        task_structure={"subtasks": [{"name": "A", "description": "Do A"}]},
    )

    assert active_worker_id in task.participants
    assert pending_worker_id not in task.participants
