"""Tests for Genesis-routed autonomous project incubation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.kernel.embeddings.default_embedder import SimpleEmbedder, set_embedder
from holonpolis.kernel.lancedb import reset_factory
from holonpolis.runtime.holon_runtime import EvolutionRequest, EvolutionStatus
from holonpolis.services.holon_service import HolonService
from holonpolis.services.project_incubation_service import (
    ProjectIncubationSpec,
    ProjectIncubationService,
)


@pytest.fixture
def temp_embedder():
    embedder = SimpleEmbedder(dimension=128)
    set_embedder(embedder)
    return embedder


@pytest.fixture
def incubation_setup(tmp_path, temp_embedder, monkeypatch):
    root = tmp_path / ".holonpolis"
    monkeypatch.setattr("holonpolis.config.settings.holonpolis_root", root)
    monkeypatch.setattr("holonpolis.config.settings.holons_path", root / "holons")
    monkeypatch.setattr(
        "holonpolis.config.settings.genesis_memory_path",
        root / "genesis" / "memory" / "lancedb",
    )
    reset_factory()
    return root


async def _create_holon(holon_id: str) -> None:
    service = HolonService()
    blueprint = Blueprint(
        blueprint_id=f"bp_{holon_id}",
        holon_id=holon_id,
        species_id="generalist",
        name=f"Holon {holon_id}",
        purpose="Project incubation test holon",
        boundary=Boundary(allowed_tools=[]),
        evolution_policy=EvolutionPolicy(),
    )
    await service.create_holon(blueprint)


class _StubRuntime:
    def __init__(self, holon_id, blueprint, execution_result):
        self.holon_id = holon_id
        self.blueprint = blueprint
        self.execution_result = execution_result

    async def request_evolution(self, **kwargs):
        return EvolutionRequest(
            request_id="evo_stub_001",
            holon_id=self.holon_id,
            skill_name=kwargs["skill_name"],
            description=kwargs["description"],
            requirements=kwargs["requirements"],
            test_cases=kwargs.get("test_cases", []),
            parent_skills=kwargs.get("parent_skills", []),
            status=EvolutionStatus.PENDING,
            created_at=utc_now_iso(),
        )

    async def wait_for_evolution(self, request_id, timeout_seconds, poll_interval_seconds):
        return EvolutionRequest(
            request_id=request_id,
            holon_id=self.holon_id,
            skill_name="Autonomous Stub Builder",
            description="stub",
            requirements=[],
            test_cases=[],
            parent_skills=[],
            status=EvolutionStatus.COMPLETED,
            created_at=utc_now_iso(),
            completed_at=utc_now_iso(),
            result={"skill_id": "autonomous_stub_builder"},
        )

    async def execute_skill(self, skill_name_or_id, payload=None, **kwargs):
        return self.execution_result


@pytest.mark.asyncio
async def test_incubation_routes_via_genesis_and_materializes_workspace(incubation_setup):
    holon_id = "holon_spawned_game"
    await _create_holon(holon_id)

    execution_result = {
        "project_name": "Snake Arena",
        "project_slug": "snake_arena",
        "files": {
            "README.md": "# Snake Arena\n",
            "package.json": '{"name":"snake-arena"}\n',
            "src/index.html": "<html></html>\n",
            "src/main.js": 'console.log("boot");\n',
            "src/game/engine.js": "export function tick(){}\n",
            "src/game/snake.js": "export function moveSnake(){}\n",
        },
        "run_instructions": ["npm install", "npm run dev"],
    }

    async def fake_route_or_spawn(user_request, conversation_history=None):
        return SimpleNamespace(
            decision="spawn",
            holon_id=holon_id,
        )

    genesis = SimpleNamespace(route_or_spawn=fake_route_or_spawn)

    def runtime_factory(runtime_holon_id, blueprint):
        return _StubRuntime(runtime_holon_id, blueprint, execution_result)

    service = ProjectIncubationService(
        genesis_service=genesis,
        runtime_factory=runtime_factory,
    )
    result = await service.incubate_project(
        ProjectIncubationSpec(
            project_name="Snake Arena",
            project_goal="Build a browser snake game with deterministic movement and scoring.",
            required_files=[
                "README.md",
                "package.json",
                "src/index.html",
                "src/main.js",
                "src/game/engine.js",
                "src/game/snake.js",
            ],
        )
    )

    assert result.holon_id == holon_id
    assert result.route_decision == "spawn"
    assert result.generated_file_count == 6

    report_path = incubation_setup / "holons" / holon_id / "workspace"
    report_files = list(report_path.glob("incubations/*/_incubation_report.json"))
    assert report_files, "incubation report should be materialized under workspace/incubations"
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report["project_slug"] == "snake_arena"


@pytest.mark.asyncio
async def test_incubation_blocks_path_traversal_from_generated_files(incubation_setup):
    holon_id = "holon_existing_safe"
    await _create_holon(holon_id)

    execution_result = {
        "project_name": "Fish Arena",
        "project_slug": "fish_arena",
        "files": {
            "../escape.txt": "boom",
            "README.md": "# fish\n",
            "package.json": '{"name":"fish"}\n',
            "apps/server/server.mjs": "export const x = 1;\n",
            "apps/client/index.html": "<html></html>\n",
            "apps/client/main.js": 'console.log("x");\n',
            "apps/shared/protocol.mjs": "export const P = {};\n",
        },
        "run_instructions": ["npm install", "npm run dev"],
    }

    async def fake_route_or_spawn(user_request, conversation_history=None):
        return SimpleNamespace(
            decision="route_to",
            holon_id=holon_id,
        )

    genesis = SimpleNamespace(route_or_spawn=fake_route_or_spawn)

    def runtime_factory(runtime_holon_id, blueprint):
        return _StubRuntime(runtime_holon_id, blueprint, execution_result)

    service = ProjectIncubationService(
        genesis_service=genesis,
        runtime_factory=runtime_factory,
    )

    with pytest.raises(ValueError, match="escapes sandbox"):
        await service.incubate_project(
            ProjectIncubationSpec(
                project_name="Fish Arena",
                project_goal="Build a multiplayer fish game with websocket synchronization.",
            )
        )


def test_build_requirements_adds_large_multiplayer_floor():
    service = ProjectIncubationService()
    requirements = service._build_requirements(
        "构建大型多人在线 MMO WebSocket 游戏，要求实时同步。",
        [],
    )
    assert any("at least 18 files" in item for item in requirements)
    assert any("Fish-eat-fish gameplay floor" in item for item in requirements)
    assert any("placeholder" in item.lower() for item in requirements)
