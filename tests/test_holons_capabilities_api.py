"""API tests for advanced Holon runtime capability endpoints."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from holonpolis.api.main import app
from holonpolis.bootstrap import bootstrap
from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.skills import ToolSchema
from holonpolis.kernel.embeddings.default_embedder import SimpleEmbedder, set_embedder
from holonpolis.kernel.lancedb import reset_factory
from holonpolis.runtime.holon_manager import get_holon_manager
from holonpolis.runtime.holon_runtime import EvolutionRequest, EvolutionStatus
from holonpolis.services.evolution_service import EvolutionService
from holonpolis.services.holon_service import HolonService


@pytest.fixture
def temp_embedder():
    embedder = SimpleEmbedder(dimension=128)
    set_embedder(embedder)
    return embedder


@pytest.fixture
def api_setup(tmp_path, temp_embedder, monkeypatch):
    root = tmp_path / ".holonpolis"
    monkeypatch.setattr("holonpolis.config.settings.holonpolis_root", root)
    monkeypatch.setattr("holonpolis.config.settings.holons_path", root / "holons")
    monkeypatch.setattr(
        "holonpolis.config.settings.genesis_memory_path",
        root / "genesis" / "memory" / "lancedb",
    )
    reset_factory()
    get_holon_manager().clear_cache()
    bootstrap()
    return root


@pytest.fixture
def client(api_setup):
    return TestClient(app)


@pytest.fixture
def created_holon(client):
    holon_id = "holon_api_caps"
    service = HolonService()
    blueprint = Blueprint(
        blueprint_id=f"bp_{holon_id}",
        holon_id=holon_id,
        species_id="generalist",
        name="API Capability Holon",
        purpose="Test advanced API endpoints",
        boundary=Boundary(),
        evolution_policy=EvolutionPolicy(),
    )
    asyncio.run(service.create_holon(blueprint))
    return holon_id


@pytest.fixture
def restricted_holon(client):
    holon_id = "holon_api_restricted"
    service = HolonService()
    blueprint = Blueprint(
        blueprint_id=f"bp_{holon_id}",
        holon_id=holon_id,
        species_id="generalist",
        name="Restricted Holon",
        purpose="Should be denied for high-impact operations",
        boundary=Boundary(allowed_tools=["market.read"], denied_tools=[]),
        evolution_policy=EvolutionPolicy(),
    )
    asyncio.run(service.create_holon(blueprint))
    return holon_id


def test_skill_list_and_execute_endpoints(client, created_holon, monkeypatch):
    manager = get_holon_manager()
    runtime = manager.get_runtime(created_holon)

    monkeypatch.setattr(
        runtime,
        "list_skills",
        lambda: [
            {
                "skill_id": "adder_skill",
                "name": "Adder Skill",
                "description": "Adds numbers",
                "version": "0.1.0",
                "path": "skills_local/adder_skill",
            }
        ],
    )

    async def fake_execute_skill(skill_name_or_id, payload=None, **kwargs):
        data = payload or kwargs
        return {"sum": data.get("a", 0) + data.get("b", 0)}

    monkeypatch.setattr(runtime, "execute_skill", fake_execute_skill)

    list_resp = client.get(f"/api/v1/holons/{created_holon}/skills")
    assert list_resp.status_code == 200
    assert list_resp.json()[0]["skill_id"] == "adder_skill"

    exec_resp = client.post(
        f"/api/v1/holons/{created_holon}/skills/adder_skill/execute",
        json={"payload": {"a": 3, "b": 4}},
    )
    assert exec_resp.status_code == 200
    assert exec_resp.json()["result"]["sum"] == 7


def test_evolution_request_and_status_endpoints(client, created_holon, monkeypatch):
    manager = get_holon_manager()
    runtime = manager.get_runtime(created_holon)

    requested = EvolutionRequest(
        request_id="evo_req_001",
        holon_id=created_holon,
        skill_name="PlannerSkill",
        description="Plan tasks",
        requirements=["accept input", "return plan"],
        test_cases=[],
        parent_skills=[],
        status=EvolutionStatus.PENDING,
        created_at="2026-02-28T00:00:00+00:00",
    )
    completed = EvolutionRequest(
        request_id="evo_req_001",
        holon_id=created_holon,
        skill_name="PlannerSkill",
        description="Plan tasks",
        requirements=["accept input", "return plan"],
        test_cases=[],
        parent_skills=[],
        status=EvolutionStatus.COMPLETED,
        created_at="2026-02-28T00:00:00+00:00",
        completed_at="2026-02-28T00:01:00+00:00",
        result={"skill_id": "planner_skill"},
    )

    async def fake_request_evolution(**kwargs):
        return requested

    monkeypatch.setattr(runtime, "request_evolution", fake_request_evolution)
    monkeypatch.setattr(runtime, "get_evolution_status", lambda rid: completed if rid == "evo_req_001" else None)

    req_resp = client.post(
        f"/api/v1/holons/{created_holon}/evolution/requests",
        json={
            "skill_name": "PlannerSkill",
            "description": "Plan tasks",
            "requirements": ["accept input", "return plan"],
        },
    )
    assert req_resp.status_code == 200
    assert req_resp.json()["request_id"] == "evo_req_001"
    assert req_resp.json()["status"] == "pending"

    status_resp = client.get(
        f"/api/v1/holons/{created_holon}/evolution/requests/evo_req_001"
    )
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["status"] == "completed"
    assert data["result"]["skill_id"] == "planner_skill"


def test_social_and_market_endpoints(client, created_holon, monkeypatch):
    manager = get_holon_manager()
    runtime = manager.get_runtime(created_holon)

    async def fake_offer_skill(skill_name, description, price_per_use=0.0):
        return "offer_001"

    async def fake_find_skill_providers(skill_query, max_price=None, top_k=3):
        return [
            {
                "holon_id": "holon_provider",
                "skill_name": skill_query,
                "price": 50.0,
                "success_rate": 0.9,
                "rating": 0.8,
                "match_score": 0.95,
            }
        ]

    async def fake_run_selection(threshold=0.3):
        return {
            "total": 2,
            "survivors": 1,
            "eliminated": 1,
            "top_performers": [{"holon_id": created_holon, "score": 0.9}],
            "eliminated_list": [{"holon_id": "holon_bad", "score": 0.1}],
            "selection_pressure": 1.0 - threshold,
        }

    async def fake_register_relationship(*args, **kwargs):
        return "rel_001"

    async def fake_propagate_trust(target_holon_id, max_hops=2):
        return 0.88

    async def fake_compete(task_description, competitors, evaluation_criteria=None):
        return {
            "competition_id": "comp_001",
            "ranking": competitors,
            "my_rank": 1,
            "reward": 1000.0,
            "scores": {"accuracy": 0.9, "speed": 0.8, "quality": 0.85},
        }

    monkeypatch.setattr(runtime, "offer_skill", fake_offer_skill)
    monkeypatch.setattr(runtime, "find_skill_providers", fake_find_skill_providers)
    monkeypatch.setattr(runtime, "get_market_stats", lambda: {"total_offers": 1, "active_offers": 1})
    monkeypatch.setattr(runtime, "run_selection", fake_run_selection)
    monkeypatch.setattr(runtime, "register_relationship", fake_register_relationship)
    monkeypatch.setattr(runtime, "propagate_trust", fake_propagate_trust)
    monkeypatch.setattr(runtime, "compete", fake_compete)

    offer_resp = client.post(
        f"/api/v1/holons/{created_holon}/market/offers",
        json={
            "skill_name": "Code Review",
            "description": "Review pull requests",
            "price_per_use": 50.0,
        },
    )
    assert offer_resp.status_code == 200
    assert offer_resp.json()["offer_id"] == "offer_001"

    providers_resp = client.get(
        f"/api/v1/holons/{created_holon}/market/providers",
        params={"skill_query": "Code Review", "top_k": 3},
    )
    assert providers_resp.status_code == 200
    assert providers_resp.json()[0]["holon_id"] == "holon_provider"

    stats_resp = client.get(f"/api/v1/holons/{created_holon}/market/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["total_offers"] == 1

    selection_resp = client.post(
        f"/api/v1/holons/{created_holon}/selection",
        json={"threshold": 0.6},
    )
    assert selection_resp.status_code == 200
    assert selection_resp.json()["survivors"] == 1

    comp_resp = client.post(
        f"/api/v1/holons/{created_holon}/competitions",
        json={
            "task_description": "Generate login form",
            "competitors": [created_holon, "holon_provider"],
            "evaluation_criteria": {"accuracy": 0.4, "speed": 0.3, "quality": 0.3},
        },
    )
    assert comp_resp.status_code == 200
    assert comp_resp.json()["competition_id"] == "comp_001"

    rel_resp = client.post(
        f"/api/v1/holons/{created_holon}/relationships",
        json={
            "target_holon_id": "holon_provider",
            "relationship_type": "peer",
            "strength": 0.7,
            "trust_score": 0.8,
        },
    )
    assert rel_resp.status_code == 200
    assert rel_resp.json()["relationship_id"] == "rel_001"

    trust_resp = client.get(
        f"/api/v1/holons/{created_holon}/trust/holon_provider",
        params={"max_hops": 2},
    )
    assert trust_resp.status_code == 200
    assert trust_resp.json()["trust_score"] == pytest.approx(0.88)


def test_skill_execute_payload_schema_validation(client, created_holon):
    """Payload violating manifest tool_schema should return 422."""
    service = EvolutionService()
    schema = ToolSchema(
        name="execute",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "additionalProperties": False,
        },
        required=["a", "b"],
    )

    asyncio.run(
        service._phase_persist(
            holon_id=created_holon,
            skill_name="StrictAdder",
            code="""
def execute(a, b):
    return a + b
""",
            tests="""
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
""",
            description="Strict adder skill",
            tool_schema=schema,
            version="0.1.0",
            green_result={"passed": True, "details": {"exit_code": 0}},
            verify_result={"passed": True, "violations": []},
        )
    )

    response = client.post(
        f"/api/v1/holons/{created_holon}/skills/strictadder/execute",
        json={"payload": {"a": 2, "extra": 1}},
    )
    assert response.status_code == 422
    assert "missing required fields" in response.json()["detail"] or "unexpected field" in response.json()["detail"]


def test_high_impact_endpoints_denied_by_boundary_policy(client, restricted_holon):
    """Selection and competition should be denied when boundary policy disallows them."""
    selection_resp = client.post(
        f"/api/v1/holons/{restricted_holon}/selection",
        json={"threshold": 0.6},
    )
    assert selection_resp.status_code == 403

    competition_resp = client.post(
        f"/api/v1/holons/{restricted_holon}/competitions",
        json={
            "task_description": "Evaluate model answers",
            "competitors": [restricted_holon],
            "evaluation_criteria": {"accuracy": 1.0},
        },
    )
    assert competition_resp.status_code == 403


def test_openapi_contract_includes_examples_and_error_codes(client):
    """OpenAPI should expose request examples and explicit 403/404/422 contracts."""
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    doc = spec.json()

    execute_path = "/api/v1/holons/{holon_id}/skills/{skill_name_or_id}/execute"
    execute_post = doc["paths"][execute_path]["post"]
    assert "403" in execute_post["responses"]
    assert "404" in execute_post["responses"]
    assert "422" in execute_post["responses"]

    schema_ref = execute_post["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    schema_name = schema_ref.split("/")[-1]
    execute_schema = doc["components"]["schemas"][schema_name]
    assert "example" in execute_schema
    assert "payload" in execute_schema["example"]
