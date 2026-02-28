"""Runtime capability tests for evolved skills and social graph wrappers."""

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.skills import ToolSchema
from holonpolis.kernel.embeddings.default_embedder import SimpleEmbedder, set_embedder
from holonpolis.kernel.lancedb import reset_factory
from holonpolis.runtime.holon_runtime import HolonRuntime
from holonpolis.services.collaboration_service import CollaborationService
from holonpolis.services.evolution_service import EvolutionService
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
