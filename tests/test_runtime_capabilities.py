"""Runtime capability tests for evolved skills and social graph wrappers."""

import asyncio

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.genesis.genesis_memory import GenesisMemory
from holonpolis.domain.skills import ToolSchema
from holonpolis.kernel.embeddings.default_embedder import SimpleEmbedder, set_embedder
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

    async def fake_generate_tests(self, *args, **kwargs):
        return """
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
"""

    async def fake_generate_code(self, *args, **kwargs):
        return """
def execute(a, b):
    return a + b
"""

    async def fake_record_evolution(self, *args, **kwargs):
        raise RuntimeError("genesis evolutions table unavailable")

    monkeypatch.setattr(EvolutionService, "_generate_tests_via_llm", fake_generate_tests)
    monkeypatch.setattr(EvolutionService, "_generate_code_via_llm", fake_generate_code)
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
    ):
        triggered.append(
            {
                "skill_name": skill_name,
                "description": description,
                "requirements": list(requirements),
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
