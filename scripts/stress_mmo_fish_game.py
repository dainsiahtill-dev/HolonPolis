#!/usr/bin/env python
"""Stress test HolonPolis for large MMO fish-eat-fish game generation.

Coverage:
1) Boot server and verify health.
2) Evolve/validate a reusable MMO fish game scaffolder skill.
3) Stress skill execution endpoint under concurrent load.
4) Materialize generated project and audit per-file quality.
5) Score each Holon with capability + reputation + skill signals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from holonpolis.config import settings  # noqa: E402
from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy  # noqa: E402
from holonpolis.domain.skills import ToolSchema  # noqa: E402
from holonpolis.runtime.holon_runtime import HolonRuntime  # noqa: E402
from holonpolis.services.evolution_service import EvolutionService  # noqa: E402
from holonpolis.services.holon_service import HolonService  # noqa: E402
from holonpolis.services.market_service import MarketService  # noqa: E402


HOLON_ID = "holon_mmo_fish_stress"
SKILL_NAME = "MMO Big-Fish-Eat-Small-Fish Scaffolder"
SKILL_ID = "mmo_big_fish_eat_small_fish_scaffolder"
SKILL_VERSION = "1.1.0"
BASELINE_SKILL_NAME = "Baseline Runtime Capability"
BASELINE_SKILL_ID = "baseline_runtime_capability"
MANDATORY_CAPABILITIES = [
    "skill.execute",
    "social.selection.execute",
    "social.competition.execute",
    "social.*",
]


@dataclass
class RequestMetric:
    ok: bool
    status_code: int
    latency_ms: float
    error: str = ""
    index: int = 0


@dataclass
class FileQuality:
    path: str
    score: int
    issues: List[str]


@dataclass
class HolonScore:
    holon_id: str
    score: float
    grade: str
    details: Dict[str, Any]


@dataclass
class UpliftResult:
    holon_id: str
    blueprint_updated: bool
    skill_added: bool
    notes: List[str]


@dataclass
class RuntimeSmokeResult:
    project_boot_ok: bool
    ws_smoke_ok: bool
    install_ok: bool
    health_url: str
    details: Dict[str, Any]


def _skill_code() -> str:
    return '''
"""MMO fish-eat-fish project scaffolder skill."""

from __future__ import annotations

from typing import Any, Dict, List


def _slugify(name: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "mmo_fish_game"


def _mk_species(name: str, tier: int) -> Dict[str, Any]:
    return {
        "name": name,
        "tier": tier,
        "baseSpeed": max(0.5, 2.5 - tier * 0.1),
        "baseMass": 2 + tier * 3,
        "lightningResistance": min(0.9, 0.1 + tier * 0.05),
    }


def execute(
    game_name: str,
    world_shards: int = 12,
    max_players: int = 8000,
    ocean_sectors: int = 16,
    fish_species: int = 12,
) -> Dict[str, Any]:
    world_shards = max(4, min(128, int(world_shards)))
    max_players = max(500, min(500000, int(max_players)))
    ocean_sectors = max(6, min(128, int(ocean_sectors)))
    fish_species = max(6, min(64, int(fish_species)))

    slug = _slugify(game_name)
    files: Dict[str, str] = {}

    files["README.md"] = f"""# {game_name}

Massive multiplayer online fish-eat-fish game scaffold.

## Vision
- Thousands of concurrent players.
- Deterministic ocean simulation per shard.
- Predation chain with growth, mass, and lightning burst skill.
- Real-time leaderboard and replay hooks.

## Capacity Targets
- world shards: {world_shards}
- ocean sectors: {ocean_sectors}
- fish species: {fish_species}
- max players: {max_players}
"""

    files["package.json"] = """{
  "name": "mmo-fish-eat-fish",
  "private": true,
  "type": "module",
  "scripts": {
    "start:gateway": "node apps/gateway/src/server.mjs",
    "dev": "node apps/gateway/src/server.mjs",
    "smoke:ws": "node scripts/ws-smoke.mjs"
  },
  "dependencies": {
    "ws": "^8.18.3"
  }
}
"""

    files["docs/ARCHITECTURE.md"] = """# Architecture

## Services
- gateway: websocket ingress + session management
- matchmaker: shard + sector assignment
- world: deterministic simulation tick
- events: lightning burst and predation resolution
- leaderboard: global and regional rank
"""
    files["docs/SCALING.md"] = """# Scaling Strategy

- stateless gateways behind L4 balancer
- shard ownership with sticky sessions
- event snapshots every 5 ticks
- replay stream via append-only event log
"""
    files["docs/GAMEPLAY.md"] = """# Gameplay

- Eat smaller fish to gain mass.
- Avoid larger fish and lightning traps.
- Trigger lightning burst at high energy.
- Leaderboard updates every simulation window.
"""

    files["apps/gateway/src/server.ts"] = """export function startGateway() {
  return { service: "gateway", protocol: "ws", status: "ready" };
}
"""
    files["apps/gateway/src/session.ts"] = """export type Session = { playerId: string; shardId: number };
export function attachSession(playerId: string, shardId: number): Session {
  return { playerId, shardId };
}
"""
    files["apps/gateway/src/rate_limit.ts"] = """export function checkRate(eventsPerSecond: number): boolean {
  return eventsPerSecond <= 40;
}
"""
    files["apps/world/src/authoritative_world.mjs"] = """const WORLD_SIZE = 2000;
const FOOD_GAIN = 0.3;
const ENERGY_GAIN = 0.6;
const ENERGY_COST_PER_BOOST = 5;

export function createWorldState() {
  return {
    tick: 0,
    players: new Map(),
    metrics: {
      joins: 0,
      leaves: 0,
      lastBroadcastSize: 0
    }
  };
}

export function addPlayer(world, playerId) {
  const spawn = {
    id: playerId,
    x: Math.random() * WORLD_SIZE,
    y: Math.random() * WORLD_SIZE,
    vx: 0,
    vy: 0,
    mass: 10,
    energy: 30,
    score: 0
  };
  world.players.set(playerId, spawn);
  world.metrics.joins += 1;
  return spawn;
}

export function removePlayer(world, playerId) {
  if (world.players.delete(playerId)) {
    world.metrics.leaves += 1;
  }
}

export function applyInput(world, playerId, payload) {
  const actor = world.players.get(playerId);
  if (!actor || typeof payload !== "object" || payload === null) return;

  if (payload.type === "swim") {
    actor.vx = clamp(Number(payload.dx || 0), -4, 4);
    actor.vy = clamp(Number(payload.dy || 0), -4, 4);
    return;
  }

  if (payload.type === "boost" && actor.energy >= ENERGY_COST_PER_BOOST) {
    actor.energy -= ENERGY_COST_PER_BOOST;
    actor.vx *= 1.6;
    actor.vy *= 1.6;
  }
}

export function tickWorld(world) {
  world.tick += 1;
  for (const fish of world.players.values()) {
    fish.x = wrap(fish.x + fish.vx, WORLD_SIZE);
    fish.y = wrap(fish.y + fish.vy, WORLD_SIZE);
    fish.energy = Math.min(100, fish.energy + ENERGY_GAIN);
    fish.mass = Math.min(600, fish.mass + FOOD_GAIN * 0.02);
    fish.score = Math.floor(fish.mass * 10 + fish.energy);
    fish.vx *= 0.92;
    fish.vy *= 0.92;
  }
}

export function buildSnapshot(world, topN = 12) {
  const players = [...world.players.values()];
  players.sort((a, b) => b.score - a.score);
  const leaders = players.slice(0, topN).map((p) => ({
    id: p.id,
    score: p.score,
    mass: round2(p.mass)
  }));

  return {
    tick: world.tick,
    online: players.length,
    leaders
  };
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function wrap(value, size) {
  if (value < 0) return size + (value % size);
  return value % size;
}

function round2(value) {
  return Math.round(value * 100) / 100;
}
"""
    files["apps/gateway/src/server.mjs"] = """import { createServer } from "node:http";
import { WebSocketServer } from "ws";
import {
  addPlayer,
  applyInput,
  buildSnapshot,
  createWorldState,
  removePlayer,
  tickWorld
} from "../../world/src/authoritative_world.mjs";

const port = Number(process.env.PORT || 8787);
const tickMs = Number(process.env.TICK_MS || 100);
const world = createWorldState();
const sockets = new Map();

const server = createServer((req, res) => {
  if (req.url === "/healthz") {
    res.writeHead(200, { "content-type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ ok: true, service: "gateway", online: world.players.size, tick: world.tick }));
    return;
  }
  res.writeHead(404, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify({ ok: false, error: "not_found" }));
});

const wss = new WebSocketServer({ server, path: "/ws" });
wss.on("connection", (socket) => {
  const playerId = `p_${Math.random().toString(16).slice(2, 10)}`;
  const state = addPlayer(world, playerId);
  sockets.set(playerId, socket);

  socket.send(JSON.stringify({
    type: "welcome",
    playerId,
    state,
    tick: world.tick
  }));

  socket.on("message", (raw) => {
    try {
      const payload = JSON.parse(String(raw));
      applyInput(world, playerId, payload);
    } catch {
      socket.send(JSON.stringify({ type: "error", error: "invalid_json" }));
    }
  });

  socket.on("close", () => {
    sockets.delete(playerId);
    removePlayer(world, playerId);
  });
});

setInterval(() => {
  tickWorld(world);
  const snapshot = buildSnapshot(world);
  const message = JSON.stringify({ type: "snapshot", ...snapshot });
  for (const ws of sockets.values()) {
    if (ws.readyState === ws.OPEN) {
      ws.send(message);
    }
  }
}, tickMs);

server.listen(port, "127.0.0.1", () => {
  // eslint-disable-next-line no-console
  console.log(`gateway_listening:${port}`);
});
"""
    files["scripts/ws-smoke.mjs"] = """const port = Number(process.argv[2] || 8787);
const clients = Number(process.argv[3] || 3);
const timeoutMs = Number(process.argv[4] || 8000);
const endpoint = `ws://127.0.0.1:${port}/ws`;

function runClient(index) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(endpoint);
    let gotSnapshot = false;
    const timer = setTimeout(() => {
      try { ws.close(); } catch {}
      reject(new Error(`client_${index}_timeout`));
    }, timeoutMs);

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "swim", dx: 1 + index * 0.1, dy: 0.4 }));
    };

    ws.onmessage = (evt) => {
      try {
        const payload = JSON.parse(String(evt.data));
        if (payload.type === "snapshot") {
          gotSnapshot = true;
          clearTimeout(timer);
          ws.close();
          resolve(true);
        }
      } catch {
        // ignore malformed payload for smoke phase
      }
    };

    ws.onerror = () => {
      clearTimeout(timer);
      reject(new Error(`client_${index}_ws_error`));
    };

    ws.onclose = () => {
      if (!gotSnapshot) {
        clearTimeout(timer);
      }
    };
  });
}

async function main() {
  const tasks = [];
  for (let i = 0; i < clients; i += 1) {
    tasks.push(runClient(i));
  }
  await Promise.all(tasks);
  console.log(JSON.stringify({ ok: true, clients }));
}

main().catch((err) => {
  console.error(err?.message || String(err));
  process.exit(1);
});
"""

    files["apps/matchmaker/src/index.ts"] = """export type JoinRequest = { playerId: string; skill: number };
export function assignShard(req: JoinRequest, shardCount: number): number {
  return Math.abs(req.playerId.length * 7 + req.skill) % shardCount;
}
"""
    files["apps/matchmaker/src/region.ts"] = """export function chooseRegion(pingMs: number): string {
  if (pingMs < 60) return "local";
  if (pingMs < 120) return "near";
  return "global";
}
"""

    files["apps/world/src/simulation.ts"] = """export type FishState = {
  id: string;
  mass: number;
  energy: number;
  x: number;
  y: number;
};
export function tickFish(fish: FishState): FishState {
  return { ...fish, energy: Math.min(100, fish.energy + 1) };
}
"""
    files["apps/world/src/collision.ts"] = """export function canEat(hunterMass: number, preyMass: number): boolean {
  return hunterMass >= preyMass * 1.1;
}
"""
    files["apps/world/src/lightning.ts"] = """export function lightningDamage(voltage: number, resistance: number): number {
  const raw = Math.max(0, voltage * (1 - resistance));
  return Math.max(1, Math.floor(raw / 10));
}
"""
    files["apps/world/src/spawn.ts"] = """export function initialMass(tier: number): number {
  return 2 + tier * 2;
}
"""
    files["apps/world/src/sector.ts"] = """export type SectorState = { id: number; population: number };
export function sectorLoadScore(state: SectorState): number {
  return state.population / 1000;
}
"""

    files["apps/frontend/src/main.tsx"] = """export function bootClient() {
  return { scene: "ocean", renderer: "webgl", ui: "hud" };
}
"""
    files["apps/frontend/src/ui/hud.ts"] = """export function renderHud(mass: number, energy: number, rank: number): string {
  return `MASS:${mass} EN:${energy} RANK:${rank}`;
}
"""
    files["apps/frontend/src/ui/minimap.ts"] = """export function minimapScale(oceanSize: number): number {
  return Math.max(0.05, Math.min(1, 1000 / oceanSize));
}
"""
    files["apps/frontend/src/ui/leaderboard.ts"] = """export function formatRank(name: string, score: number): string {
  return `${name}:${score}`;
}
"""

    files["packages/shared/src/protocol.ts"] = """export type ClientEvent =
  | { type: "join"; playerId: string }
  | { type: "swim"; dx: number; dy: number }
  | { type: "boost"; value: number }
  | { type: "cast_lightning"; targetId: string };

export type ServerEvent =
  | { type: "snapshot"; tick: number }
  | { type: "damage"; amount: number }
  | { type: "mass_gain"; amount: number }
  | { type: "eliminated"; byPlayerId: string };
"""
    files["packages/shared/src/config.ts"] = f"""export const GAME_CONFIG = {{
  worldShards: {world_shards},
  oceanSectors: {ocean_sectors},
  fishSpecies: {fish_species},
  maxPlayers: {max_players},
  tickRate: 30
}};
"""
    files["packages/shared/src/types.ts"] = """export type Vector2 = { x: number; y: number };
export type PlayerStats = { mass: number; energy: number; streak: number };
"""

    files["packages/leaderboard/src/store.ts"] = """export type Score = { playerId: string; mass: number };
export function topN(scores: Score[], n: number): Score[] {
  return [...scores].sort((a, b) => b.mass - a.mass).slice(0, n);
}
"""
    files["packages/leaderboard/src/window.ts"] = """export function shouldEmitWindow(tick: number): boolean {
  return tick % 10 === 0;
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
  leaderboard:
    image: node:20
"""
    files["infra/k8s/gateway-deployment.yaml"] = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: fish-gateway
spec:
  replicas: 4
"""
    files["infra/k8s/world-statefulset.yaml"] = """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: fish-world
spec:
  serviceName: fish-world
"""
    files["infra/k8s/matchmaker-deployment.yaml"] = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: fish-matchmaker
spec:
  replicas: 3
"""
    files["infra/observability/grafana-dashboard.json"] = """{
  "title": "Fish MMO Core Metrics",
  "panels": [{"title": "tick_latency_ms"}, {"title": "online_players"}]
}
"""

    files["tests/world_tick.test.ts"] = """import { tickFish } from "../apps/world/src/simulation";
const next = tickFish({ id: "a", mass: 10, energy: 50, x: 0, y: 0 });
if (next.energy !== 51) throw new Error("tickFish failed");
"""
    files["tests/collision.test.ts"] = """import { canEat } from "../apps/world/src/collision";
if (!canEat(11, 10)) throw new Error("canEat logic failed");
"""
    files["tests/lightning.test.ts"] = """import { lightningDamage } from "../apps/world/src/lightning";
if (lightningDamage(100, 0.2) < 1) throw new Error("lightningDamage invalid");
"""

    species: List[Dict[str, Any]] = []
    for i in range(1, fish_species + 1):
        species.append(_mk_species(f"species_{i}", i))
        files[f"configs/species/species_{i}.json"] = (
            '{"name":"species_%d","tier":%d,"baseMass":%d}\\n' % (i, i, 2 + i * 3)
        )

    for shard in range(1, world_shards + 1):
        files[f"configs/shards/shard_{shard}.json"] = (
            '{"shardId": %d, "maxPlayers": %d, "tickRate": 30}\\n'
            % (shard, max(64, max_players // world_shards))
        )

    for sector in range(1, ocean_sectors + 1):
        files[f"configs/sectors/sector_{sector}.json"] = (
            '{"sectorId": %d, "foodDensity": %.2f, "hazardLevel": %.2f}\\n'
            % (sector, 0.6 + (sector % 4) * 0.1, 0.2 + (sector % 3) * 0.15)
        )

    estimated_lines = sum(v.count("\\n") + 1 for v in files.values())
    return {
        "project_slug": slug,
        "project_name": game_name,
        "world_shards": world_shards,
        "max_players": max_players,
        "ocean_sectors": ocean_sectors,
        "fish_species": fish_species,
        "species_preview": species[:6],
        "files": files,
        "estimated_lines": estimated_lines,
    }
'''


def _skill_tests() -> str:
    return '''
from skill_module import execute


def test_large_project_generated():
    out = execute("Ocean Arena", world_shards=12, max_players=9000, ocean_sectors=16, fish_species=12)
    assert out["project_name"] == "Ocean Arena"
    assert out["world_shards"] == 12
    assert out["ocean_sectors"] == 16
    assert out["fish_species"] == 12
    assert len(out["files"]) >= 45


def test_critical_files_exist():
    out = execute("Blue Abyss", world_shards=8, max_players=5000, ocean_sectors=12, fish_species=10)
    files = out["files"]
    assert "apps/world/src/simulation.ts" in files
    assert "apps/world/src/lightning.ts" in files
    assert "apps/gateway/src/server.ts" in files
    assert "apps/gateway/src/server.mjs" in files
    assert "apps/world/src/authoritative_world.mjs" in files
    assert "scripts/ws-smoke.mjs" in files
    assert "packages/shared/src/protocol.ts" in files
    assert "infra/k8s/world-statefulset.yaml" in files
'''


async def ensure_holon_exists() -> None:
    service = HolonService()
    if service.holon_exists(HOLON_ID):
        return

    blueprint = Blueprint(
        blueprint_id=f"bp_{HOLON_ID}",
        holon_id=HOLON_ID,
        species_id="specialist",
        name="MMO Fish Engineer",
        purpose="Generate and evolve large-scale fish MMO systems",
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
    valid = await evo.validate_existing_skill(HOLON_ID, SKILL_NAME)
    if valid.get("valid"):
        try:
            svc = HolonService()
            runtime = HolonRuntime(holon_id=HOLON_ID, blueprint=svc.get_blueprint(HOLON_ID))
            probe = await runtime.execute_skill(
                SKILL_ID,
                payload={"game_name": "runtime-probe"},
            )
            files = probe.get("files", {}) if isinstance(probe, dict) else {}
            if (
                isinstance(files, dict)
                and "apps/gateway/src/server.mjs" in files
                and "apps/world/src/authoritative_world.mjs" in files
                and "scripts/ws-smoke.mjs" in files
            ):
                return
        except Exception:
            pass

    schema = ToolSchema(
        name="execute",
        description="Generate large MMO fish-eat-fish project scaffold",
        parameters={
            "type": "object",
            "properties": {
                "game_name": {"type": "string", "minLength": 3},
                "world_shards": {"type": "integer", "minimum": 4, "maximum": 128},
                "max_players": {"type": "integer", "minimum": 500, "maximum": 500000},
                "ocean_sectors": {"type": "integer", "minimum": 6, "maximum": 128},
                "fish_species": {"type": "integer", "minimum": 6, "maximum": 64},
            },
            "required": ["game_name"],
            "additionalProperties": False,
        },
        required=["game_name"],
    )
    result = await evo.evolve_skill(
        holon_id=HOLON_ID,
        skill_name=SKILL_NAME,
        code=_skill_code(),
        tests=_skill_tests(),
        description="Scaffold a large multiplayer fish-eat-fish game architecture",
        tool_schema=schema,
        version=SKILL_VERSION,
    )
    if not result.success:
        raise RuntimeError(f"Skill evolution failed: {result.phase} {result.error_message}")


def _baseline_skill_code() -> str:
    return '''
"""Minimal baseline skill to guarantee executable capability for a Holon."""

from __future__ import annotations

from typing import Any, Dict


def execute(task: str = "ping") -> Dict[str, Any]:
    clean_task = str(task).strip() or "ping"
    return {
        "status": "ok",
        "task": clean_task,
        "task_length": len(clean_task),
    }
'''


def _baseline_skill_tests() -> str:
    return '''
from skill_module import execute


def test_execute_default():
    out = execute()
    assert out["status"] == "ok"
    assert out["task"] == "ping"


def test_execute_custom_task():
    out = execute("matchmaking")
    assert out["task"] == "matchmaking"
    assert out["task_length"] == len("matchmaking")
'''


def _persist_blueprint(blueprint: Blueprint) -> None:
    path = settings.holons_path / blueprint.holon_id / "blueprint.json"
    path.write_text(
        json.dumps(blueprint.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def uplift_holons() -> List[UpliftResult]:
    svc = HolonService()
    evo = EvolutionService()
    results: List[UpliftResult] = []

    schema = ToolSchema(
        name="execute",
        description="Return baseline deterministic capability payload.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
            },
            "additionalProperties": False,
        },
        required=[],
    )

    for item in svc.list_holons():
        holon_id = item["holon_id"]
        notes: List[str] = []
        blueprint_updated = False
        skill_added = False

        try:
            blueprint = svc.get_blueprint(holon_id)
            allowed = list(blueprint.boundary.allowed_tools or [])
            for cap in MANDATORY_CAPABILITIES:
                if cap not in allowed:
                    allowed.append(cap)
                    blueprint_updated = True
            if blueprint_updated:
                blueprint.boundary.allowed_tools = sorted(set(allowed))
                _persist_blueprint(blueprint)
                notes.append("boundary capabilities upgraded")

            runtime = HolonRuntime(holon_id=holon_id, blueprint=blueprint)
            skills = runtime.list_skills()
            if not skills:
                evolved = await evo.evolve_skill(
                    holon_id=holon_id,
                    skill_name=BASELINE_SKILL_NAME,
                    code=_baseline_skill_code(),
                    tests=_baseline_skill_tests(),
                    description="Baseline executable skill for runtime quality floor",
                    tool_schema=schema,
                    version="1.0.0",
                )
                if evolved.success:
                    skill_added = True
                    notes.append("baseline skill evolved")
                else:
                    notes.append(f"baseline skill evolve failed: {evolved.phase} {evolved.error_message}")
        except Exception as exc:
            notes.append(f"uplift failed: {exc}")

        results.append(
            UpliftResult(
                holon_id=holon_id,
                blueprint_updated=blueprint_updated,
                skill_added=skill_added,
                notes=notes,
            )
        )
    return results


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    timeout_sec: float,
) -> Tuple[bool, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode == 0, proc.stdout, proc.stderr
    except Exception as exc:
        return False, "", repr(exc)


def _npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def run_project_runtime_smoke(
    project_dir: Path,
    port: int,
    ws_clients: int,
    ws_timeout_ms: int,
    npm_install_timeout_sec: float,
) -> RuntimeSmokeResult:
    npm_exe = _npm_executable()
    install_ok, _, install_err = _run_cmd(
        [npm_exe, "install", "--no-audit", "--no-fund"],
        cwd=project_dir,
        timeout_sec=npm_install_timeout_sec,
    )
    health_url = f"http://127.0.0.1:{port}/healthz"
    if not install_ok:
        return RuntimeSmokeResult(
            project_boot_ok=False,
            ws_smoke_ok=False,
            install_ok=False,
            health_url=health_url,
            details={"install_error": install_err[:1000]},
        )

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHONUTF8"] = "1"
    server = subprocess.Popen(
        [npm_exe, "run", "start:gateway"],
        cwd=str(project_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    boot_ok = False
    ws_ok = False
    details: Dict[str, Any] = {}
    try:
        deadline = time.monotonic() + 25.0
        with httpx.Client(timeout=1.5) as client:
            while time.monotonic() < deadline:
                try:
                    resp = client.get(health_url)
                    if resp.status_code == 200:
                        boot_ok = True
                        details["health_payload"] = resp.json()
                        break
                except Exception:
                    pass
                time.sleep(0.25)

        if boot_ok:
            ws_ok, ws_out, ws_err = _run_cmd(
                [
                    "node",
                    "scripts/ws-smoke.mjs",
                    str(port),
                    str(ws_clients),
                    str(ws_timeout_ms),
                ],
                cwd=project_dir,
                timeout_sec=max(20.0, ws_timeout_ms / 1000.0 + 5.0),
            )
            details["ws_stdout"] = ws_out[:1000]
            if ws_err:
                details["ws_stderr"] = ws_err[:1000]
    finally:
        try:
            server.terminate()
            server.wait(timeout=10)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass

    return RuntimeSmokeResult(
        project_boot_ok=boot_ok,
        ws_smoke_ok=ws_ok,
        install_ok=True,
        health_url=health_url,
        details=details,
    )


def start_server(port: int) -> subprocess.Popen[str]:
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


async def wait_server(base_url: str, timeout_sec: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_sec
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise TimeoutError("Service health check timeout")


async def execute_once(
    client: httpx.AsyncClient,
    base_url: str,
    payload: Dict[str, Any],
    index: int,
) -> Tuple[RequestMetric, Optional[Dict[str, Any]]]:
    url = f"{base_url}/api/v1/holons/{HOLON_ID}/skills/{SKILL_ID}/execute"
    started = time.perf_counter()
    try:
        resp = await client.post(url, json={"payload": payload})
        elapsed = (time.perf_counter() - started) * 1000
        if resp.status_code == 200:
            return RequestMetric(True, resp.status_code, elapsed, index=index), resp.json()
        return (
            RequestMetric(False, resp.status_code, elapsed, error=resp.text[:500], index=index),
            None,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return RequestMetric(False, 0, elapsed, error=repr(exc), index=index), None


async def stress_execute(
    base_url: str,
    requests: int,
    concurrency: int,
    timeout_sec: float,
) -> Tuple[List[RequestMetric], Optional[Dict[str, Any]]]:
    semaphore = asyncio.Semaphore(concurrency)
    metrics: List[RequestMetric] = []
    first_ok: Optional[Dict[str, Any]] = None
    async with httpx.AsyncClient(timeout=timeout_sec) as client:

        async def one(i: int) -> None:
            nonlocal first_ok
            payload = {
                "game_name": f"Abyss-Arena-{i:03d}",
                "world_shards": 10 + (i % 6),
                "max_players": 8000 + i * 25,
                "ocean_sectors": 16 + (i % 5),
                "fish_species": 12 + (i % 4),
            }
            async with semaphore:
                metric, data = await execute_once(client, base_url, payload, i)
                metrics.append(metric)
                if first_ok is None and data is not None:
                    first_ok = data

        await asyncio.gather(*(one(i) for i in range(requests)))
    return metrics, first_ok


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
    succ = [m for m in metrics if m.ok]
    p50 = statistics.median(lat)
    p95_idx = min(len(lat) - 1, int(len(lat) * 0.95))
    return {
        "total": len(metrics),
        "success": len(succ),
        "failed": len(metrics) - len(succ),
        "success_rate": len(succ) / len(metrics),
        "p50_ms": round(p50, 2),
        "p95_ms": round(lat[p95_idx], 2),
        "max_ms": round(lat[-1], 2),
    }


def _clean_dir(target: Path) -> None:
    if not target.exists():
        return
    for item in sorted(target.rglob("*"), reverse=True):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            try:
                item.rmdir()
            except OSError:
                pass


def materialize_project(resp: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    result = resp.get("result", {})
    files = result.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("No files in skill output")
    slug = str(result.get("project_slug") or "mmo_fish_game")
    target = output_root / slug
    _clean_dir(target)
    target.mkdir(parents=True, exist_ok=True)

    count = 0
    lines = 0
    for rel, content in files.items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        path.write_text(text, encoding="utf-8")
        count += 1
        lines += text.count("\n") + 1

    required = [
        "apps/world/src/simulation.ts",
        "apps/world/src/lightning.ts",
        "apps/gateway/src/server.ts",
        "apps/gateway/src/server.mjs",
        "apps/world/src/authoritative_world.mjs",
        "scripts/ws-smoke.mjs",
        "packages/shared/src/protocol.ts",
        "infra/k8s/world-statefulset.yaml",
        "configs/species/species_1.json",
        "configs/sectors/sector_1.json",
    ]
    missing = [p for p in required if not (target / p).exists()]
    return {
        "target_dir": str(target),
        "file_count": count,
        "line_count": lines,
        "missing_required_files": missing,
        "world_shards": result.get("world_shards"),
        "max_players": result.get("max_players"),
    }


def audit_file_quality(project_dir: Path) -> List[FileQuality]:
    findings: List[FileQuality] = []
    keywords = ("TODO", "FIXME", "XXX", "TBD", "placeholder")
    for file in sorted(project_dir.rglob("*")):
        if not file.is_file():
            continue
        rel = str(file.relative_to(project_dir)).replace("\\", "/")
        score = 100
        issues: List[str] = []
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(FileQuality(rel, 0, ["not utf-8 decodable"]))
            continue

        if not text.strip():
            score -= 50
            issues.append("empty file")

        upper = text.upper()
        for kw in keywords:
            if kw in upper:
                score -= 10
                issues.append(f"contains {kw}")

        if file.suffix == ".json":
            try:
                json.loads(text)
            except Exception:
                score -= 40
                issues.append("invalid json")
        elif file.suffix in {".ts", ".tsx", ".js", ".mjs"}:
            is_test_file = ".test." in file.name or rel.startswith("tests/")
            is_entrypoint = file.name in {"server.mjs", "main.mjs", "index.mjs"}
            if not is_test_file and not is_entrypoint and "export " not in text and "function " not in text:
                score -= 15
                issues.append("no exported symbol")
            if "any" in re.findall(r"\bany\b", text):
                score -= 8
                issues.append("uses any type")
        elif file.suffix == ".md":
            if "#" not in text:
                score -= 20
                issues.append("no markdown heading")

        if len(text.splitlines()) < 3:
            score -= 8
            issues.append("very short file")

        score = max(0, min(100, score))
        findings.append(FileQuality(rel, score, issues))
    return findings


def summarize_quality(files: List[FileQuality]) -> Dict[str, Any]:
    if not files:
        return {"avg": 0.0, "min": 0, "high_risk": []}
    avg = sum(f.score for f in files) / len(files)
    min_score = min(f.score for f in files)
    high_risk = [f for f in files if f.score < 70]
    return {
        "avg": round(avg, 2),
        "min": min_score,
        "high_risk": high_risk[:20],
    }


def grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "E"


def score_holons(target_skill_id: str) -> List[HolonScore]:
    holon_service = HolonService()
    market = MarketService()
    holons = holon_service.list_holons()

    results: List[HolonScore] = []
    for item in holons:
        hid = item["holon_id"]
        details: Dict[str, Any] = {}
        base = 40.0
        try:
            blueprint = holon_service.get_blueprint(hid)
            runtime = HolonRuntime(holon_id=hid, blueprint=blueprint)
            skills = runtime.list_skills()
            skill_count = len(skills)
            details["skill_count"] = skill_count
            details["has_target_skill"] = any(s["skill_id"] == target_skill_id for s in skills)
            score = base + min(25.0, skill_count * 5.0)

            if runtime.is_capability_allowed("skill.execute", aliases=["execute"]):
                score += 8.0
                details["can_execute_skill"] = True
            else:
                details["can_execute_skill"] = False

            if runtime.is_capability_allowed(
                "social.competition.execute",
                aliases=["competition", "execute"],
            ):
                score += 6.0
                details["can_compete"] = True
            else:
                details["can_compete"] = False

            rep = market._get_reputation(hid)
            details["reputation"] = {
                "overall": round(rep.overall_score, 3),
                "reliability": round(rep.reliability, 3),
                "competence": round(rep.competence, 3),
                "collaboration": round(rep.collaboration, 3),
            }
            score += rep.overall_score * 12.0
            score += rep.competence * 6.0
            score += rep.collaboration * 3.0

            if details["has_target_skill"]:
                score += 5.0
        except Exception as exc:
            score = 25.0
            details["error"] = str(exc)

        final = max(0.0, min(100.0, round(score, 2)))
        results.append(HolonScore(hid, final, grade(final), details))

    results.sort(key=lambda x: x.score, reverse=True)
    return results


def print_report(
    summary: Dict[str, Any],
    project: Dict[str, Any],
    quality_summary: Dict[str, Any],
    quality_files: List[FileQuality],
    holon_scores: List[HolonScore],
    failures: List[RequestMetric],
    base_url: str,
    runtime_smoke: Optional[RuntimeSmokeResult],
    uplift_results: List[UpliftResult],
) -> None:
    print("=" * 96)
    print("HolonPolis Stress Report - MMO Big Fish Eat Small Fish")
    print("=" * 96)
    print(f"API: {base_url}")
    print(
        f"Stress: total={summary['total']} success={summary['success']} failed={summary['failed']} "
        f"rate={summary['success_rate']:.2%}"
    )
    print(
        f"Latency(ms): p50={summary['p50_ms']} p95={summary['p95_ms']} max={summary['max_ms']}"
    )
    print("-" * 96)
    print(
        f"Project: dir={project['target_dir']} files={project['file_count']} "
        f"lines={project['line_count']} shards={project['world_shards']} max_players={project['max_players']}"
    )
    if project["missing_required_files"]:
        print(f"Missing required files: {project['missing_required_files']}")
    else:
        print("Required architecture files: OK")
    print("-" * 96)
    print(
        f"Code Quality: avg={quality_summary['avg']} min={quality_summary['min']} "
        f"high_risk_files={len(quality_summary['high_risk'])}"
    )
    for item in quality_summary["high_risk"][:10]:
        print(f"  - {item.path}: score={item.score} issues={item.issues}")
    print("-" * 96)
    print(f"Holon Uplift ({len(uplift_results)}):")
    changed = [u for u in uplift_results if u.blueprint_updated or u.skill_added]
    print(f"  updated_or_evolved={len(changed)}")
    for u in changed[:10]:
        print(
            f"  - {u.holon_id}: blueprint_updated={u.blueprint_updated} "
            f"skill_added={u.skill_added} notes={u.notes}"
        )
    print("-" * 96)
    if runtime_smoke is not None:
        print(
            "Runtime Smoke: "
            f"install_ok={runtime_smoke.install_ok} "
            f"boot_ok={runtime_smoke.project_boot_ok} "
            f"ws_ok={runtime_smoke.ws_smoke_ok} "
            f"health={runtime_smoke.health_url}"
        )
        if runtime_smoke.details:
            print(f"Runtime details: {runtime_smoke.details}")
        print("-" * 96)
    print(f"Holon Scores ({len(holon_scores)}):")
    for hs in holon_scores:
        print(
            f"  - {hs.holon_id}: {hs.score} ({hs.grade}) "
            f"skills={hs.details.get('skill_count', 0)} "
            f"target_skill={hs.details.get('has_target_skill')}"
        )
    if failures:
        print("-" * 96)
        print("Top failures:")
        for f in failures[:8]:
            print(
                f"  - idx={f.index} status={f.status_code} latency={round(f.latency_ms, 2)} "
                f"error={f.error[:180]}"
            )
    print("=" * 96)


def build_report_payload(
    summary: Dict[str, Any],
    project: Dict[str, Any],
    quality_summary: Dict[str, Any],
    quality_files: List[FileQuality],
    holon_scores: List[HolonScore],
    failures: List[RequestMetric],
    base_url: str,
    runtime_smoke: Optional[RuntimeSmokeResult],
    uplift_results: List[UpliftResult],
) -> Dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "stress_summary": summary,
        "project_summary": project,
        "quality_summary": {
            "avg": quality_summary.get("avg"),
            "min": quality_summary.get("min"),
            "high_risk_count": len(quality_summary.get("high_risk", [])),
        },
        "quality_files": [
            {
                "path": f.path,
                "score": f.score,
                "issues": f.issues,
            }
            for f in quality_files
        ],
        "holon_scores": [
            {
                "holon_id": h.holon_id,
                "score": h.score,
                "grade": h.grade,
                "details": h.details,
            }
            for h in holon_scores
        ],
        "failures": [
            {
                "index": f.index,
                "status_code": f.status_code,
                "latency_ms": round(f.latency_ms, 2),
                "error": f.error,
            }
            for f in failures
        ],
        "runtime_smoke": (
            {
                "install_ok": runtime_smoke.install_ok,
                "project_boot_ok": runtime_smoke.project_boot_ok,
                "ws_smoke_ok": runtime_smoke.ws_smoke_ok,
                "health_url": runtime_smoke.health_url,
                "details": runtime_smoke.details,
            }
            if runtime_smoke is not None
            else None
        ),
        "uplift_results": [
            {
                "holon_id": u.holon_id,
                "blueprint_updated": u.blueprint_updated,
                "skill_added": u.skill_added,
                "notes": u.notes,
            }
            for u in uplift_results
        ],
    }


async def run(args: argparse.Namespace) -> int:
    await ensure_holon_exists()
    uplift_results: List[UpliftResult] = []
    if args.auto_uplift_holons:
        uplift_results = await uplift_holons()
    await ensure_skill_evolved()

    base_url = f"http://127.0.0.1:{args.port}"
    server = start_server(args.port)
    try:
        await wait_server(base_url, timeout_sec=30.0)
        metrics, first_ok = await stress_execute(
            base_url=base_url,
            requests=args.requests,
            concurrency=args.concurrency,
            timeout_sec=args.request_timeout,
        )
        if first_ok is None:
            print("No successful responses returned by execute endpoint.")
            return 2

        project = materialize_project(first_ok, args.output_root)
        quality_files = audit_file_quality(Path(project["target_dir"]))
        quality_summary = summarize_quality(quality_files)
        runtime_smoke: Optional[RuntimeSmokeResult] = None
        if args.verify_project_runtime:
            runtime_smoke = await asyncio.to_thread(
                run_project_runtime_smoke,
                Path(project["target_dir"]),
                args.game_port,
                args.ws_clients,
                args.ws_timeout_ms,
                args.npm_install_timeout_sec,
            )
        holon_scores = score_holons(SKILL_ID)
        summary = summarize_metrics(metrics)
        failures = [m for m in metrics if not m.ok]

        print_report(
            summary=summary,
            project=project,
            quality_summary=quality_summary,
            quality_files=quality_files,
            holon_scores=holon_scores,
            failures=failures,
            base_url=base_url,
            runtime_smoke=runtime_smoke,
            uplift_results=uplift_results,
        )
        if args.report_json:
            report_payload = build_report_payload(
                summary=summary,
                project=project,
                quality_summary=quality_summary,
                quality_files=quality_files,
                holon_scores=holon_scores,
                failures=failures,
                base_url=base_url,
                runtime_smoke=runtime_smoke,
                uplift_results=uplift_results,
            )
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved report JSON: {args.report_json}")

        if summary["success_rate"] < args.min_success_rate:
            return 3
        if project["missing_required_files"]:
            return 4
        if quality_summary["avg"] < args.min_quality_avg:
            return 5
        if runtime_smoke is not None and not (
            runtime_smoke.install_ok and runtime_smoke.project_boot_ok and runtime_smoke.ws_smoke_ok
        ):
            return 6
        return 0
    finally:
        try:
            server.terminate()
        except Exception:
            pass
        try:
            server.wait(timeout=10)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test MMO fish game generation and Holon scoring")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-quality-avg", type=float, default=85.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("C:/Temp/mmo-fish-eat-fish"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("C:/Temp/mmo-fish-eat-fish/stress_report.json"),
    )
    parser.add_argument(
        "--auto-uplift-holons",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verify-project-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--game-port", type=int, default=8787)
    parser.add_argument("--ws-clients", type=int, default=5)
    parser.add_argument("--ws-timeout-ms", type=int, default=10000)
    parser.add_argument("--npm-install-timeout-sec", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
