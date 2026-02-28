#!/usr/bin/env python
"""Stress test HolonPolis startup + MMO lightning game scaffolding pipeline.

This script validates:
1) Service can boot and answer health checks.
2) A Holon can evolve and expose a reusable scaffolding skill.
3) API can sustain concurrent skill executions.
4) Output can be materialized into a large multiplayer lightning game project skeleton.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy  # noqa: E402
from holonpolis.domain.skills import ToolSchema  # noqa: E402
from holonpolis.services.evolution_service import EvolutionService  # noqa: E402
from holonpolis.services.holon_service import HolonService  # noqa: E402


HOLON_ID = "holon_mmo_lightning_stress"
SKILL_NAME = "MMO Lightning Game Scaffolder"
SKILL_ID = "mmo_lightning_game_scaffolder"


@dataclass
class RequestMetric:
    ok: bool
    latency_ms: float
    status_code: int
    error: str = ""
    payload_index: int = 0


def _build_skill_code() -> str:
    return '''
"""MMO Lightning Game scaffolder skill."""

from __future__ import annotations

from typing import Dict, Any


def _slugify(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "mmo_lightning_game"


def execute(
    game_name: str,
    world_shards: int = 8,
    max_players: int = 2000,
) -> Dict[str, Any]:
    world_shards = max(2, min(64, int(world_shards)))
    max_players = max(100, min(200000, int(max_players)))
    slug = _slugify(game_name)

    files: Dict[str, str] = {}

    files["README.md"] = f"""# {game_name}

Large-scale multiplayer online lightning game scaffold.

## Architecture
- Gateway: WebSocket session ingress
- Matchmaker: room and shard assignment
- World simulation: deterministic tick loop
- Realtime transport: event fanout
- Leaderboard and telemetry

## Capacity plan
- shards: {world_shards}
- max players: {max_players}
- baseline tick rate: 30Hz
"""

    files["package.json"] = """{
  "name": "mmo-lightning-game",
  "private": true,
  "workspaces": ["apps/*", "packages/*"],
  "scripts": {
    "build": "echo build",
    "test": "echo test",
    "lint": "echo lint"
  }
}
"""

    files["apps/gateway/src/server.ts"] = """export function startGateway() {
  return {
    service: "gateway",
    transport: "websocket",
    status: "ready"
  };
}
"""

    files["apps/matchmaker/src/index.ts"] = """export type MatchRequest = { playerId: string; mmr: number };

export function assignRoom(req: MatchRequest, shardCount: number): string {
  const shard = Math.abs(req.playerId.length + req.mmr) % shardCount;
  return `room-${shard}`;
}
"""

    files["apps/world/src/simulation.ts"] = """export type PlayerState = { id: string; energy: number; x: number; y: number };

export function tick(state: PlayerState): PlayerState {
  const nextEnergy = Math.max(0, Math.min(100, state.energy + 1));
  return { ...state, energy: nextEnergy };
}
"""

    files["apps/world/src/lightning.ts"] = """export type LightningEvent = {
  sourceId: string;
  targetId: string;
  voltage: number;
};

export function applyLightningDamage(voltage: number): number {
  return Math.max(1, Math.floor(voltage / 10));
}
"""

    files["apps/frontend/src/main.tsx"] = """export function bootClient() {
  return {
    scene: "arena",
    renderer: "webgl",
    status: "ready"
  };
}
"""

    files["apps/frontend/src/ui/hud.ts"] = """export type HudState = {
  hp: number;
  energy: number;
  playersOnline: number;
};

export function renderHud(state: HudState): string {
  return `HP:${state.hp} EN:${state.energy} ONLINE:${state.playersOnline}`;
}
"""

    files["packages/shared/src/protocol.ts"] = """export type ClientToServer =
  | { type: "join"; playerId: string }
  | { type: "move"; dx: number; dy: number }
  | { type: "cast_lightning"; targetId: string };

export type ServerToClient =
  | { type: "state"; tick: number }
  | { type: "damage"; amount: number };
"""

    files["packages/shared/src/config.ts"] = f"""export const GAME_CONFIG = {{
  worldShards: {world_shards},
  maxPlayers: {max_players},
  tickRate: 30
}};
"""

    files["packages/leaderboard/src/store.ts"] = """export type ScoreEntry = { playerId: string; score: number };

export function top10(entries: ScoreEntry[]): ScoreEntry[] {
  return [...entries].sort((a, b) => b.score - a.score).slice(0, 10);
}
"""

    files["infra/docker/docker-compose.yml"] = """version: '3.8'
services:
  gateway:
    image: node:20
  matchmaker:
    image: node:20
  world:
    image: node:20
"""

    files["infra/k8s/gateway-deployment.yaml"] = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
spec:
  replicas: 3
"""

    files["infra/k8s/world-statefulset.yaml"] = """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: world
spec:
  serviceName: world
"""

    files["docs/ARCHITECTURE.md"] = """# MMO Lightning Game Architecture

## Service Mesh
- gateway
- matchmaker
- world simulation
- leaderboard

## Data flow
clients -> gateway -> matchmaker -> world -> gateway -> clients
"""

    files["docs/SCALING.md"] = f"""# Scaling Notes

- target concurrent players: {max_players}
- world shards: {world_shards}
- horizontal scaling via stateless gateways
- deterministic shard simulation loops
"""

    files["tests/world_tick.test.ts"] = """import { tick } from "../apps/world/src/simulation";

const next = tick({ id: "p1", energy: 50, x: 0, y: 0 });
if (next.energy !== 51) {
  throw new Error("tick energy increment failed");
}
"""

    for shard in range(1, world_shards + 1):
        files[f"configs/shards/shard_{shard}.json"] = (
            '{"shardId": %d, "maxPlayers": %d, "region": "auto"}\\n'
            % (shard, max(32, max_players // world_shards))
        )

    estimated_lines = sum(content.count("\\n") + 1 for content in files.values())

    return {
        "project_slug": slug,
        "project_name": game_name,
        "world_shards": world_shards,
        "max_players": max_players,
        "files": files,
        "estimated_lines": estimated_lines,
    }
'''


def _build_skill_tests() -> str:
    return '''
from skill_module import execute


def test_generates_large_project_skeleton():
    result = execute("Lightning Arena", world_shards=10, max_players=5000)
    assert result["project_name"] == "Lightning Arena"
    assert result["world_shards"] == 10
    assert "apps/gateway/src/server.ts" in result["files"]
    assert "apps/world/src/lightning.ts" in result["files"]
    assert len(result["files"]) >= 20


def test_shard_configs_match_requested_count():
    result = execute("Storm Grid", world_shards=6, max_players=1200)
    shard_files = [k for k in result["files"].keys() if k.startswith("configs/shards/")]
    assert len(shard_files) == 6
'''


async def ensure_holon_exists() -> None:
    service = HolonService()
    if service.holon_exists(HOLON_ID):
        return

    blueprint = Blueprint(
        blueprint_id=f"bp_{HOLON_ID}",
        holon_id=HOLON_ID,
        species_id="specialist",
        name="MMO Lightning Developer",
        purpose="Build large multiplayer online lightning game systems",
        boundary=Boundary(
            allowed_tools=[
                "skill.execute",
                "evolution.request",
                "social.selection.execute",
                "social.competition.execute",
                "social.*",
            ],
            denied_tools=[],
            allow_file_write=True,
            allow_network=False,
            allow_subprocess=False,
        ),
        evolution_policy=EvolutionPolicy(),
    )
    await service.create_holon(blueprint)


async def ensure_skill_evolved() -> None:
    evo = EvolutionService()
    check = await evo.validate_existing_skill(HOLON_ID, SKILL_NAME)
    if check.get("valid"):
        return

    schema = ToolSchema(
        name="execute",
        description="Generate MMO lightning game project scaffold",
        parameters={
            "type": "object",
            "properties": {
                "game_name": {"type": "string", "minLength": 3},
                "world_shards": {"type": "integer", "minimum": 2, "maximum": 64},
                "max_players": {"type": "integer", "minimum": 100, "maximum": 200000},
            },
            "required": ["game_name"],
            "additionalProperties": False,
        },
        required=["game_name"],
    )

    result = await evo.evolve_skill(
        holon_id=HOLON_ID,
        skill_name=SKILL_NAME,
        code=_build_skill_code(),
        tests=_build_skill_tests(),
        description="Scaffold a large-scale multiplayer online lightning game project",
        tool_schema=schema,
        version="1.0.0",
    )
    if not result.success:
        raise RuntimeError(f"Failed to evolve skill: {result.phase} {result.error_message}")


def _start_server(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["HOLONPOLIS_RELOAD"] = "false"
    env["HOLONPOLIS_PORT"] = str(port)

    return subprocess.Popen(
        [sys.executable, "run.py"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _wait_server_ready(base_url: str, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise TimeoutError("Server did not become healthy in time")


async def _execute_once(
    client: httpx.AsyncClient,
    base_url: str,
    payload: Dict[str, Any],
    payload_index: int,
) -> Tuple[RequestMetric, Dict[str, Any] | None]:
    url = f"{base_url}/api/v1/holons/{HOLON_ID}/skills/{SKILL_ID}/execute"
    started = time.perf_counter()
    try:
        resp = await client.post(url, json={"payload": payload})
        elapsed_ms = (time.perf_counter() - started) * 1000
        if resp.status_code == 200:
            return (
                RequestMetric(
                    ok=True,
                    latency_ms=elapsed_ms,
                    status_code=resp.status_code,
                    payload_index=payload_index,
                ),
                resp.json(),
            )
        return (
            RequestMetric(
                ok=False,
                latency_ms=elapsed_ms,
                status_code=resp.status_code,
                error=resp.text[:400],
                payload_index=payload_index,
            ),
            None,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return (
            RequestMetric(
                ok=False,
                latency_ms=elapsed_ms,
                status_code=0,
                error=repr(exc),
                payload_index=payload_index,
            ),
            None,
        )


async def run_api_stress(
    base_url: str,
    total_requests: int,
    concurrency: int,
    request_timeout_sec: float,
) -> Tuple[List[RequestMetric], Dict[str, Any] | None]:
    semaphore = asyncio.Semaphore(concurrency)
    metrics: List[RequestMetric] = []
    first_ok: Dict[str, Any] | None = None

    async with httpx.AsyncClient(timeout=request_timeout_sec) as client:
        async def worker(index: int) -> None:
            nonlocal first_ok
            payload = {
                "game_name": f"Lightning-MMO-{index:03d}",
                "world_shards": 8 + (index % 4),
                "max_players": 2500 + index * 10,
            }
            async with semaphore:
                metric, data = await _execute_once(client, base_url, payload, index)
                metrics.append(metric)
                if first_ok is None and data is not None:
                    first_ok = data

        await asyncio.gather(*(worker(i) for i in range(total_requests)))

    return metrics, first_ok


def materialize_project(api_result: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    result = api_result.get("result", {})
    files = result.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("No files returned by scaffolder skill")

    slug = str(result.get("project_slug") or "mmo_lightning_game")
    target = output_root / slug
    if target.exists():
        for path in sorted(target.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    target.mkdir(parents=True, exist_ok=True)

    file_count = 0
    line_count = 0
    for rel_path, content in files.items():
        dest = target / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        dest.write_text(text, encoding="utf-8")
        file_count += 1
        line_count += text.count("\n") + 1

    required = [
        "apps/gateway/src/server.ts",
        "apps/matchmaker/src/index.ts",
        "apps/world/src/lightning.ts",
        "apps/frontend/src/main.tsx",
        "packages/shared/src/protocol.ts",
        "infra/docker/docker-compose.yml",
    ]
    missing = [item for item in required if not (target / item).exists()]

    return {
        "target_dir": str(target),
        "file_count": file_count,
        "line_count": line_count,
        "missing_required_files": missing,
        "world_shards": result.get("world_shards"),
        "max_players": result.get("max_players"),
    }


def summarize_metrics(metrics: List[RequestMetric]) -> Dict[str, Any]:
    if not metrics:
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "success_rate": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    lat = sorted(m.latency_ms for m in metrics)
    success = [m for m in metrics if m.ok]
    p50 = statistics.median(lat)
    p95_idx = min(len(lat) - 1, int(len(lat) * 0.95))
    return {
        "total": len(metrics),
        "success": len(success),
        "failed": len(metrics) - len(success),
        "success_rate": len(success) / len(metrics),
        "p50_ms": round(p50, 2),
        "p95_ms": round(lat[p95_idx], 2),
        "max_ms": round(lat[-1], 2),
    }


def print_report(
    summary: Dict[str, Any],
    project: Dict[str, Any],
    failures: List[RequestMetric],
    base_url: str,
) -> None:
    print("=" * 88)
    print("HolonPolis MMO Lightning Stress Test Report")
    print("=" * 88)
    print(f"API: {base_url}")
    print(f"Requests: {summary['total']}")
    print(f"Success: {summary['success']}  Failed: {summary['failed']}  Rate: {summary['success_rate']:.2%}")
    print(f"Latency(ms): p50={summary['p50_ms']}  p95={summary['p95_ms']}  max={summary['max_ms']}")
    print("-" * 88)
    print(f"Project directory: {project['target_dir']}")
    print(f"Generated files: {project['file_count']}  Estimated lines: {project['line_count']}")
    print(f"World shards: {project.get('world_shards')}  Max players: {project.get('max_players')}")
    if project["missing_required_files"]:
        print(f"Missing required files: {project['missing_required_files']}")
    else:
        print("Required architecture files: OK")
    if failures:
        print("-" * 88)
        print("Top failures (up to 5):")
        for item in failures[:5]:
            print(
                f"#{item.payload_index} status={item.status_code} "
                f"latency={item.latency_ms:.2f}ms error={item.error[:180]}"
            )
    print("=" * 88)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Stress test startup and MMO lightning scaffolding")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("C:/Temp/mmo-lightning"),
    )
    args = parser.parse_args()

    await ensure_holon_exists()
    await ensure_skill_evolved()

    base_url = f"http://127.0.0.1:{args.port}"
    server = _start_server(args.port)
    try:
        await _wait_server_ready(base_url)
        metrics, first_ok = await run_api_stress(
            base_url=base_url,
            total_requests=args.requests,
            concurrency=args.concurrency,
            request_timeout_sec=args.request_timeout,
        )
        if first_ok is None:
            failures = [m for m in metrics if not m.ok]
            print("No successful responses received from API execute endpoint.")
            for item in failures[:5]:
                print(item)
            return 2

        project = materialize_project(first_ok, args.output_root)
        summary = summarize_metrics(metrics)
        failures = [m for m in metrics if not m.ok]
        print_report(summary, project, failures, base_url)

        if summary["success_rate"] < 0.95:
            return 3
        if project["missing_required_files"]:
            return 4
        return 0
    finally:
        try:
            server.terminate()
        except Exception:
            pass
        try:
            server.wait(timeout=8)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
