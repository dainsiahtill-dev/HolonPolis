"""Tests for Genesis-routed autonomous project incubation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.kernel.embeddings.default_embedder import SimpleEmbedder, set_embedder
from holonpolis.kernel.lancedb import reset_factory
from holonpolis.runtime.holon_runtime import EvolutionRequest, EvolutionStatus
from holonpolis.services.evolution_service import EvolutionService
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
    def __init__(self, holon_id, blueprint, execution_result, asset_hits=None):
        self.holon_id = holon_id
        self.blueprint = blueprint
        self.execution_result = execution_result
        self.asset_hits = list(asset_hits or [])
        self.last_request_kwargs = {}

    async def request_evolution(self, **kwargs):
        self.last_request_kwargs = dict(kwargs)
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

    async def search_reusable_code_library(self, query, top_k=3, library_kind=None):
        _ = (query, top_k, library_kind)
        return list(self.asset_hits)


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


def test_build_requirements_adds_generic_large_multiplayer_floor():
    service = ProjectIncubationService()
    requirements = service._build_requirements(
        "构建大型多人在线 MMO WebSocket 游戏，要求实时同步和可扩展战斗系统。",
        [],
    )
    assert any("at least 18 files" in item for item in requirements)
    assert any("project_goal-defined rules" in item for item in requirements)
    assert any("placeholder" in item.lower() for item in requirements)
    merged = "\n".join(requirements).lower()
    assert "fish-eat-fish" not in merged
    assert "rpg gameplay floor" not in merged


def test_project_contract_builder_is_callable_from_instance():
    service = EvolutionService()
    tests = service._build_project_contract_tests(
        ["Project goal: Build a dashboard frontend.", "Required output file paths: package.json, src/main.jsx"]
    )
    assert "def test_execute_output_shape" in tests
    assert "REQUIRED_PATHS" in tests


@pytest.mark.asyncio
async def test_incubation_injects_reusable_code_assets_into_evolution_requirements(incubation_setup):
    holon_id = "holon_reuse_assets"
    await _create_holon(holon_id)

    execution_result = {
        "project_name": "SDK Demo",
        "project_slug": "sdk_demo",
        "files": {
            "README.md": "# SDK Demo\n",
            "package.json": '{"name":"sdk-demo"}\n',
            "src/main.js": 'console.log("boot");\n',
        },
        "run_instructions": ["npm install", "npm run dev"],
    }
    asset_hits = [
        {
            "asset_name": "ApiClient",
            "library_name": "internal-sdk",
            "relative_path": "client.py",
            "usage_example": "from module import ApiClient",
            "code_content": "class ApiClient:\n    pass\n",
        }
    ]

    async def fake_route_or_spawn(user_request, conversation_history=None):
        return SimpleNamespace(
            decision="route_to",
            holon_id=holon_id,
        )

    genesis = SimpleNamespace(route_or_spawn=fake_route_or_spawn)
    stub_runtime = _StubRuntime(holon_id, None, execution_result, asset_hits=asset_hits)

    def runtime_factory(runtime_holon_id, blueprint):
        stub_runtime.blueprint = blueprint
        return stub_runtime

    service = ProjectIncubationService(
        genesis_service=genesis,
        runtime_factory=runtime_factory,
    )
    await service.incubate_project(
        ProjectIncubationSpec(
            project_name="SDK Demo",
            project_goal="Build a starter project that reuses our existing API client patterns.",
        )
    )

    requirements = stub_runtime.last_request_kwargs.get("requirements", [])
    joined = "\n".join(requirements)
    assert "Reference reusable code assets" in joined
    assert "ApiClient" in joined


@pytest.mark.asyncio
async def test_incubation_defaults_to_evolution_only_even_when_reusable_assets_exist(incubation_setup):
    holon_id = "holon_evolution_only"
    await _create_holon(holon_id)

    execution_result = {
        "project_name": "Evolution Demo",
        "project_slug": "evolution_demo",
        "files": {
            "README.md": "# Evolution Demo\n",
            "package.json": '{"name":"evolution-demo"}\n',
            "src/main.js": 'console.log("evolved");\n',
        },
        "run_instructions": ["npm install", "npm run dev"],
    }
    asset_hits = [
        {
            "library_key": "dashboard-seed",
            "library_name": "dashboard-seed",
            "asset_name": "DashboardShell",
            "relative_path": "src/components/DashboardShell.jsx",
            "asset_type": "source",
            "code_content": "export function DashboardShell() { return null; }\n",
        }
    ]

    async def fake_route_or_spawn(user_request, conversation_history=None):
        return SimpleNamespace(
            decision="route_to",
            holon_id=holon_id,
        )

    genesis = SimpleNamespace(route_or_spawn=fake_route_or_spawn)
    stub_runtime = _StubRuntime(holon_id, None, execution_result, asset_hits=asset_hits)

    def runtime_factory(runtime_holon_id, blueprint):
        stub_runtime.blueprint = blueprint
        return stub_runtime

    service = ProjectIncubationService(
        genesis_service=genesis,
        runtime_factory=runtime_factory,
    )
    result = await service.incubate_project(
        ProjectIncubationSpec(
            project_name="Evolution Demo",
            project_goal="Build a finance dashboard using our learned component patterns.",
        )
    )

    assert result.delivery_mode == "evolved_skill"
    assert stub_runtime.last_request_kwargs
    joined = "\n".join(stub_runtime.last_request_kwargs.get("requirements", []))
    assert "Reference reusable code assets" in joined
    assert "DashboardShell" in joined


@pytest.mark.asyncio
async def test_incubation_prefers_reusable_scaffold_when_indexed_library_is_available(incubation_setup):
    holon_id = "holon_dashboard_seed"
    await _create_holon(holon_id)

    source_root = incubation_setup.parent / "dashboard_seed"
    (source_root / "src" / "components").mkdir(parents=True, exist_ok=True)
    (source_root / "src" / "__tests__").mkdir(parents=True, exist_ok=True)
    (source_root / "node_modules" / "left-pad").mkdir(parents=True, exist_ok=True)
    (source_root / "package.json").write_text(
        json.dumps(
            {
                "name": "dashboard-seed",
                "private": True,
                "scripts": {
                    "dev": "vite",
                    "build": "vite build",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source_root / "index.html").write_text("<!doctype html>\n<div id=\"root\"></div>\n", encoding="utf-8")
    (source_root / ".gitignore").write_text("node_modules/\ndist/\n", encoding="utf-8")
    (source_root / "vite.svg").write_text("<svg></svg>\n", encoding="utf-8")
    (source_root / "index.html").write_text(
        "<!doctype html>\n<img src=\"/vite.svg\" />\n<div id=\"root\"></div>\n",
        encoding="utf-8",
    )
    (source_root / "src" / "styles").mkdir(parents=True, exist_ok=True)
    (source_root / "src" / "widgets").mkdir(parents=True, exist_ok=True)
    (source_root / "src" / "main.jsx").write_text(
        "import './styles/dashboard.css';\n"
        "import { DashboardShell } from './components/DashboardShell.jsx';\n",
        encoding="utf-8",
    )
    (source_root / "src" / "components" / "DashboardShell.jsx").write_text(
        "import { KpiCard } from '../widgets/KpiCard.jsx';\n"
        "export function DashboardShell() { return <main><KpiCard /></main>; }\n",
        encoding="utf-8",
    )
    (source_root / "src" / "widgets" / "KpiCard.jsx").write_text(
        "export function KpiCard() { return <section>kpi</section>; }\n",
        encoding="utf-8",
    )
    (source_root / "src" / "styles" / "dashboard.css").write_text(
        ".dashboard { display: grid; }\n",
        encoding="utf-8",
    )
    (source_root / "src" / "__tests__" / "DashboardShell.test.jsx").write_text(
        "test('x', () => {});\n",
        encoding="utf-8",
    )
    (source_root / "node_modules" / "left-pad" / "index.js").write_text(
        "module.exports = (x) => x;\n",
        encoding="utf-8",
    )

    HolonService().record_reusable_code_library_index(
        holon_id,
        "dashboard-seed",
        snapshot={
            "library_name": "dashboard-seed",
                "library_key": "dashboard-seed",
                "library_kind": "ui_component",
                "source_path": str(source_root),
                "allowed_extensions": [".jsx", ".json", ".html", ".css", ".svg"],
                "asset_count": 8,
            },
        )

    asset_hits = [
        {
            "library_key": "dashboard-seed",
            "library_name": "dashboard-seed",
            "asset_name": "DashboardShell",
            "relative_path": "src/components/DashboardShell.jsx",
            "asset_type": "source",
        }
    ]

    async def fake_route_or_spawn(user_request, conversation_history=None):
        return SimpleNamespace(
            decision="route_to",
            holon_id=holon_id,
        )

    genesis = SimpleNamespace(route_or_spawn=fake_route_or_spawn)
    stub_runtime = _StubRuntime(holon_id, None, {"files": {}}, asset_hits=asset_hits)

    def runtime_factory(runtime_holon_id, blueprint):
        stub_runtime.blueprint = blueprint
        return stub_runtime

    service = ProjectIncubationService(
        genesis_service=genesis,
        runtime_factory=runtime_factory,
    )
    result = await service.incubate_project(
        ProjectIncubationSpec(
            project_name="Dashboard Starter",
            project_goal="Build a dashboard project with sidebar navigation, cards, and table views.",
            required_files=["package.json", "index.html", "src/main.jsx"],
            prefer_reusable_scaffold=True,
        )
    )

    output_dir = Path(result.output_dir)
    assert result.delivery_mode == "reusable_scaffold"
    assert result.evolution_status == "reused_scaffold"
    assert result.reused_library_key == "dashboard-seed"
    assert result.run_instructions == ["npm install", "npm run dev"]
    assert result.generated_file_count == 8
    assert stub_runtime.last_request_kwargs == {}

    assert (output_dir / "package.json").is_file()
    assert (output_dir / "index.html").is_file()
    assert (output_dir / "vite.svg").is_file()
    assert (output_dir / "src" / "main.jsx").is_file()
    assert (output_dir / "src" / "styles" / "dashboard.css").is_file()
    assert (output_dir / "src" / "widgets" / "KpiCard.jsx").is_file()
    assert (output_dir / ".gitignore").is_file()
    assert not (output_dir / "node_modules").exists()
    assert not (output_dir / "src" / "__tests__").exists()

    report = json.loads((output_dir / "_incubation_report.json").read_text(encoding="utf-8"))
    assert report["delivery_mode"] == "reusable_scaffold"
    assert report["run_instructions"] == ["npm install", "npm run dev"]


@pytest.mark.asyncio
async def test_incubation_can_export_generated_project_without_mutating_runtime_source(incubation_setup):
    holon_id = "holon_export_delivery"
    await _create_holon(holon_id)

    execution_result = {
        "project_name": "Export Demo",
        "project_slug": "export_demo",
        "files": {
            "README.md": "# Export Demo\n",
            "package.json": '{"name":"export-demo"}\n',
            "src/main.js": 'console.log("export");\n',
        },
        "run_instructions": ["npm install", "npm run dev"],
    }
    export_target = incubation_setup.parent / "delivered_project"

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
    result = await service.incubate_project(
        ProjectIncubationSpec(
            project_name="Export Demo",
            project_goal="Build a simple exportable frontend starter project.",
            export_target_path=str(export_target),
        )
    )

    internal_output = Path(result.output_dir)
    assert internal_output.is_dir()
    assert result.exported_output_dir == str(export_target)
    assert export_target.is_dir()
    assert (export_target / "README.md").read_text(encoding="utf-8") == "# Export Demo\n"
    assert (export_target / "package.json").read_text(encoding="utf-8") == '{"name":"export-demo"}\n'
    assert (internal_output / "README.md").read_text(encoding="utf-8") == "# Export Demo\n"

    report = json.loads((internal_output / "_incubation_report.json").read_text(encoding="utf-8"))
    assert report["exported_output_dir"] == str(export_target)


@pytest.mark.asyncio
async def test_incubation_waits_for_self_reflective_follow_up_chain(incubation_setup):
    holon_id = "holon_follow_up_chain"
    await _create_holon(holon_id)

    execution_result = {
        "project_name": "Cyber Ledger",
        "project_slug": "cyber_ledger",
        "files": {
            "README.md": "# Cyber Ledger\n",
            "package.json": '{"name":"cyber-ledger"}\n',
            "index.html": "<html></html>\n",
            "src/main.jsx": "export {};\n",
            "src/app.jsx": "export default function App() { return null; }\n",
            "src/styles.css": "body { background: #000; }\n",
        },
        "run_instructions": ["npm install", "npm run dev"],
    }

    class _FollowUpRuntime(_StubRuntime):
        def __init__(self, holon_id, blueprint, execution_result):
            super().__init__(holon_id, blueprint, execution_result)
            self.wait_calls: list[str] = []

        async def wait_for_evolution(self, request_id, timeout_seconds, poll_interval_seconds):
            _ = (timeout_seconds, poll_interval_seconds)
            self.wait_calls.append(request_id)
            if request_id == "evo_stub_001":
                return EvolutionRequest(
                    request_id=request_id,
                    holon_id=self.holon_id,
                    skill_name="Autonomous Stub Builder",
                    description="stub",
                    requirements=[],
                    test_cases=[],
                    parent_skills=[],
                    status=EvolutionStatus.FAILED,
                    created_at=utc_now_iso(),
                    completed_at=utc_now_iso(),
                    result={
                        "status": "failed",
                        "follow_up": {
                            "triggered": True,
                            "request_id": "evo_stub_002",
                        },
                    },
                    error_message="first attempt failed",
                )
            if request_id == "evo_stub_002":
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
            raise AssertionError(f"unexpected request id: {request_id}")

    async def fake_route_or_spawn(user_request, conversation_history=None):
        return SimpleNamespace(
            decision="route_to",
            holon_id=holon_id,
        )

    genesis = SimpleNamespace(route_or_spawn=fake_route_or_spawn)
    stub_runtime = _FollowUpRuntime(holon_id, None, execution_result)

    def runtime_factory(runtime_holon_id, blueprint):
        stub_runtime.blueprint = blueprint
        return stub_runtime

    service = ProjectIncubationService(
        genesis_service=genesis,
        runtime_factory=runtime_factory,
    )
    result = await service.incubate_project(
        ProjectIncubationSpec(
            project_name="Cyber Ledger",
            project_goal="Build a cyberpunk bookkeeping frontend.",
            required_files=[
                "README.md",
                "package.json",
                "index.html",
                "src/main.jsx",
                "src/app.jsx",
                "src/styles.css",
            ],
        )
    )

    assert stub_runtime.wait_calls == ["evo_stub_001", "evo_stub_002"]
    assert result.request_id == "evo_stub_002"
    assert result.evolution_status == EvolutionStatus.COMPLETED.value
    assert result.generated_file_count == 6
