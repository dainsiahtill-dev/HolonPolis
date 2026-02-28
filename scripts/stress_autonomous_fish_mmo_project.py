#!/usr/bin/env python
"""End-to-end stress test for autonomous MMO fish project incubation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from holonpolis.services.project_incubation_service import (  # noqa: E402
    ProjectIncubationService,
    ProjectIncubationSpec,
)


DEFAULT_PROJECT_GOAL = (
    "构建一个大型多人在线“大鱼吃小鱼”游戏项目。必须由 Holon 自主生成完整工程代码，"
    "包含权威 WebSocket 服务端（/healthz 与 /ws）、浏览器客户端、吞噬与成长规则、"
    "分片匹配、基础反作弊，并提供可执行脚本：npm run start:server 与 npm run smoke:ws。"
)

DEFAULT_REQUIRED_FILES = [
    "README.md",
    "package.json",
    "apps/server/src/server.mjs",
    "scripts/smoke/ws-smoke.mjs",
]


STRICT_WS_PROBE_SCRIPT = r"""
const WebSocket = require("ws");
const url = process.argv[1];
const timeoutMs = Number(process.argv[2] || "12000");

const ws = new WebSocket(url);
let done = false;
let sawWelcome = false;

function finish(code, message) {
  if (done) return;
  done = true;
  if (message) {
    if (code === 0) {
      console.log(message);
    } else {
      console.error(message);
    }
  }
  try { ws.close(); } catch {}
  process.exit(code);
}

const timer = setTimeout(() => {
  finish(2, "strict_probe_timeout");
}, timeoutMs);

ws.on("open", () => {
  ws.send(JSON.stringify({ type: "move", dx: 1, dy: 0.2 }));
});

ws.on("message", (data) => {
  const text = String(data || "");
  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch {}
  const typeValue = parsed && parsed.type ? String(parsed.type).toLowerCase() : text.toLowerCase();
  if (typeValue.includes("welcome")) {
    sawWelcome = true;
    return;
  }
  const isSnapshot = typeValue.includes("snapshot") || typeValue.includes("state");
  const hasGamePayload = Boolean(
    parsed &&
      (parsed.players || parsed.entities || parsed.world || parsed.state || parsed.fishes)
  );
  if (isSnapshot && hasGamePayload) {
    clearTimeout(timer);
    finish(0, text.slice(0, 300));
  }
});

ws.on("error", (err) => {
  clearTimeout(timer);
  finish(3, String(err && err.message ? err.message : err));
});

ws.on("close", () => {
  if (!done) {
    clearTimeout(timer);
    finish(4, "closed_without_expected_message");
  }
});

setTimeout(() => {
  if (!done && sawWelcome) {
    clearTimeout(timer);
    finish(5, "welcome_without_snapshot_payload");
  }
}, Math.max(2000, Math.floor(timeoutMs * 0.6)));
"""


LOAD_WS_PROBE_SCRIPT = r"""
const WebSocket = require("ws");
const url = process.argv[1];
const clients = Number(process.argv[2] || "40");
const timeoutMs = Number(process.argv[3] || "25000");

let completed = 0;
let success = 0;
let failed = 0;
const latencies = [];

function computeP95(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
  return sorted[idx];
}

function finish(code, reason) {
  const payload = {
    ok: code === 0,
    clients,
    completed,
    success,
    failed,
    p50_ms: latencies.length ? latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.5)] : 0,
    p95_ms: computeP95(latencies),
    max_ms: latencies.length ? Math.max(...latencies) : 0,
    reason: reason || ""
  };
  console.log(JSON.stringify(payload));
  process.exit(code);
}

const globalTimer = setTimeout(() => {
  finish(2, "load_probe_global_timeout");
}, timeoutMs);

for (let i = 0; i < clients; i += 1) {
  const started = Date.now();
  const ws = new WebSocket(url);
  let finalized = false;

  function complete(ok) {
    if (finalized) return;
    finalized = true;
    completed += 1;
    if (ok) {
      success += 1;
      latencies.push(Date.now() - started);
    } else {
      failed += 1;
    }
    try { ws.close(); } catch {}
    if (completed >= clients) {
      clearTimeout(globalTimer);
      finish(failed === 0 ? 0 : 1, failed === 0 ? "" : "load_probe_partial_failure");
    }
  }

  const perClientTimer = setTimeout(() => {
    complete(false);
  }, Math.max(2000, timeoutMs - 1000));

  ws.on("open", () => {
    ws.send(JSON.stringify({ type: "move", dx: 0.4, dy: 0.1 }));
  });

  ws.on("message", (data) => {
    const text = String(data || "");
    let parsed = null;
    try {
      parsed = JSON.parse(text);
    } catch {}
    const typeValue = parsed && parsed.type ? String(parsed.type).toLowerCase() : text.toLowerCase();
    const hasSnapshot = typeValue.includes("snapshot") || typeValue.includes("state");
    const hasPayload = Boolean(
      parsed &&
        (parsed.players || parsed.entities || parsed.world || parsed.state || parsed.fishes)
    );
    if (hasSnapshot && hasPayload) {
      clearTimeout(perClientTimer);
      complete(true);
    }
  });

  ws.on("error", () => {
    clearTimeout(perClientTimer);
    complete(false);
  });

  ws.on("close", () => {
    clearTimeout(perClientTimer);
    complete(false);
  });
}
"""


@dataclass
class RuntimeProbeResult:
    quality_ok: bool = False
    quality_generated_files: int = 0
    quality_source_files: int = 0
    install_ok: bool = False
    server_boot_ok: bool = False
    health_status: int = 0
    smoke_ok: bool = False
    strict_ws_ok: bool = False
    load_ok: bool = False
    load_clients: int = 0
    load_success: int = 0
    load_failed: int = 0
    load_p95_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def all_checks_passed(self) -> bool:
        return (
            self.quality_ok
            and self.install_ok
            and self.server_boot_ok
            and self.smoke_ok
            and self.strict_ws_ok
            and self.load_ok
        )


@dataclass
class RunReport:
    run_index: int
    project_name: str
    started_at: str
    incubation_ok: bool = False
    incubation_seconds: float = 0.0
    holon_id: str = ""
    request_id: str = ""
    skill_id: str = ""
    output_dir: str = ""
    generated_file_count: int = 0
    runtime_probe: Optional[RuntimeProbeResult] = None
    error: str = ""

    def full_success(self) -> bool:
        return self.incubation_ok and self.runtime_probe is not None and self.runtime_probe.all_checks_passed()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def _run_cmd(
    cmd: List[str],
    cwd: Path,
    timeout_sec: float,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, str, int]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        return False, str(exc.stdout or ""), str(exc.stderr or ""), 124
    except Exception as exc:
        return False, "", repr(exc), 1


def _extract_json_line(text: str) -> Dict[str, Any]:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _wait_healthz(port: int, timeout_sec: float) -> Tuple[bool, int, str]:
    deadline = time.monotonic() + timeout_sec
    url = f"http://127.0.0.1:{port}/healthz"
    status = 0
    payload = ""
    with httpx.Client(timeout=1.5) as client:
        while time.monotonic() < deadline:
            try:
                resp = client.get(url)
                status = int(resp.status_code)
                payload = resp.text[:1000]
                if status == 200:
                    return True, status, payload
            except Exception:
                pass
            time.sleep(0.25)
    return False, status, payload


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def analyze_project_quality(
    project_dir: Path,
    min_generated_files: int,
    min_source_files: int,
    max_placeholder_hits: int,
) -> Dict[str, Any]:
    placeholder_tokens = (
        "todo",
        "tbd",
        "placeholder",
        "coming soon",
        "render logic",
        "implement this",
        "to be implemented",
        "mock data",
        "stub implementation",
    )
    source_suffixes = {".js", ".mjs", ".ts", ".tsx", ".jsx", ".json", ".md"}
    all_files: List[Path] = []
    source_files: List[Path] = []
    placeholder_hits: List[Dict[str, str]] = []

    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(project_dir)).replace("\\", "/")
        if rel.startswith("node_modules/"):
            continue
        all_files.append(path)

        if path.suffix.lower() in {".js", ".mjs", ".ts", ".tsx", ".jsx"}:
            source_files.append(path)

        if path.suffix.lower() not in source_suffixes:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            placeholder_hits.append({"file": rel, "token": "non_utf8_or_unreadable"})
            continue
        lowered = text.lower()
        for token in placeholder_tokens:
            if token in lowered:
                placeholder_hits.append({"file": rel, "token": token})

    rel_paths = [str(p.relative_to(project_dir)).replace("\\", "/").lower() for p in all_files]
    source_corpus_parts: List[str] = []
    for path in all_files:
        rel = str(path.relative_to(project_dir)).replace("\\", "/")
        if rel.startswith("node_modules/"):
            continue
        if path.suffix.lower() not in source_suffixes:
            continue
        try:
            source_corpus_parts.append(path.read_text(encoding="utf-8").lower())
        except Exception:
            continue
    source_corpus = "\n".join(source_corpus_parts)
    has_fish_term = any(token in source_corpus for token in ("fish", "大鱼", "小鱼"))
    has_eat_term = any(token in source_corpus for token in ("eat", "devour", "consume", "吞噬"))
    has_growth_term = any(
        token in source_corpus for token in ("grow", "growth", "mass", "size", "score", "成长")
    )
    has_collision_term = any(
        token in source_corpus for token in ("collision", "overlap", "hitbox", "碰撞")
    )
    gameplay_ok = has_fish_term and has_eat_term and has_growth_term and has_collision_term

    domain_flags = {
        "server": any("apps/server/" in rel for rel in rel_paths),
        "client": any("apps/client/" in rel or "frontend/" in rel for rel in rel_paths),
        "world": any("world" in rel for rel in rel_paths),
        "matchmaking": any("match" in rel for rel in rel_paths),
        "anti_cheat": any("anti" in rel and "cheat" in rel for rel in rel_paths),
        "gameplay_semantics": gameplay_ok,
    }
    thin_source_files: List[str] = []
    for path in source_files:
        rel = str(path.relative_to(project_dir)).replace("\\", "/")
        if rel.startswith("node_modules/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lowered = text.lower()
        non_comment_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and not line.strip().startswith("//")
            and not line.strip().startswith("/*")
            and not line.strip().startswith("*")
        ]
        if not non_comment_lines:
            thin_source_files.append(rel)
            continue
        only_console = all("console.log" in line.lower() for line in non_comment_lines)
        has_logic_token = any(
            token in lowered
            for token in ("function", "=>", "class ", "if ", "for ", "while ", "switch ", "return ")
        )
        if len(non_comment_lines) <= 2 or only_console or not has_logic_token:
            thin_source_files.append(rel)

    reasons: List[str] = []
    if len(all_files) < min_generated_files:
        reasons.append(f"generated_files_too_few({len(all_files)}<{min_generated_files})")
    if len(source_files) < min_source_files:
        reasons.append(f"source_files_too_few({len(source_files)}<{min_source_files})")
    for key, ok in domain_flags.items():
        if not ok:
            reasons.append(f"missing_domain_module:{key}")
    if len(placeholder_hits) > max_placeholder_hits:
        reasons.append(
            f"placeholder_hits_exceeded({len(placeholder_hits)}>{max_placeholder_hits})"
        )
    if len(thin_source_files) > 2:
        reasons.append(f"thin_source_files_exceeded({len(thin_source_files)}>2)")

    return {
        "ok": len(reasons) == 0,
        "generated_files": len(all_files),
        "source_files": len(source_files),
        "domain_flags": domain_flags,
        "thin_source_files": thin_source_files[:100],
        "placeholder_hits": placeholder_hits[:100],
        "reasons": reasons,
    }


def run_runtime_probe(
    project_dir: Path,
    port: int,
    ws_timeout_ms: int,
    load_clients: int,
    load_timeout_ms: int,
    load_min_success_rate: float,
    min_generated_files: int,
    min_source_files: int,
    max_placeholder_hits: int,
    npm_install_timeout_sec: float,
) -> RuntimeProbeResult:
    result = RuntimeProbeResult(load_clients=load_clients)
    npm_exe = _npm_executable()
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHONUTF8"] = "1"

    quality = analyze_project_quality(
        project_dir=project_dir,
        min_generated_files=min_generated_files,
        min_source_files=min_source_files,
        max_placeholder_hits=max_placeholder_hits,
    )
    result.quality_ok = bool(quality.get("ok", False))
    result.quality_generated_files = int(quality.get("generated_files", 0))
    result.quality_source_files = int(quality.get("source_files", 0))
    result.details["quality"] = quality
    if not result.quality_ok:
        result.errors.extend(list(quality.get("reasons", [])))
        return result

    ok, out, err, _ = _run_cmd(
        [npm_exe, "install", "--no-audit", "--no-fund"],
        cwd=project_dir,
        timeout_sec=npm_install_timeout_sec,
        env=env,
    )
    result.install_ok = ok
    result.details["npm_install_stdout"] = out[-1200:]
    result.details["npm_install_stderr"] = err[-1200:]
    if not ok:
        result.errors.append("npm_install_failed")
        return result

    server = subprocess.Popen(
        [npm_exe, "run", "start:server"],
        cwd=str(project_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        time.sleep(0.5)
        if server.poll() is not None:
            result.errors.append("server_exited_immediately")
            stdout_text, stderr_text = server.communicate(timeout=5)
            result.details["server_stdout_tail"] = stdout_text[-1500:]
            result.details["server_stderr_tail"] = stderr_text[-1500:]
            return result

        boot_ok, health_status, health_payload = _wait_healthz(port=port, timeout_sec=30.0)
        result.server_boot_ok = boot_ok
        result.health_status = health_status
        result.details["health_payload"] = health_payload
        if not boot_ok:
            result.errors.append("server_boot_failed")
            if server.poll() is not None:
                stdout_text, stderr_text = server.communicate(timeout=5)
                result.details["server_stdout_tail"] = stdout_text[-1500:]
                result.details["server_stderr_tail"] = stderr_text[-1500:]
            return result
        if server.poll() is not None:
            result.server_boot_ok = False
            result.errors.append("server_not_running_after_health")
            stdout_text, stderr_text = server.communicate(timeout=5)
            result.details["server_stdout_tail"] = stdout_text[-1500:]
            result.details["server_stderr_tail"] = stderr_text[-1500:]
            return result

        smoke_ok, smoke_out, smoke_err, _ = _run_cmd(
            [npm_exe, "run", "smoke:ws"],
            cwd=project_dir,
            timeout_sec=max(30.0, ws_timeout_ms / 1000.0 + 15.0),
            env=env,
        )
        result.smoke_ok = smoke_ok
        result.details["smoke_stdout"] = smoke_out[-1200:]
        result.details["smoke_stderr"] = smoke_err[-1200:]
        if not smoke_ok:
            result.errors.append("smoke_ws_failed")

        strict_ok, strict_out, strict_err, _ = _run_cmd(
            ["node", "-e", STRICT_WS_PROBE_SCRIPT, f"ws://127.0.0.1:{port}/ws", str(ws_timeout_ms)],
            cwd=project_dir,
            timeout_sec=max(20.0, ws_timeout_ms / 1000.0 + 10.0),
            env=env,
        )
        result.strict_ws_ok = strict_ok
        result.details["strict_ws_stdout"] = strict_out[-1200:]
        result.details["strict_ws_stderr"] = strict_err[-1200:]
        if not strict_ok:
            result.errors.append("strict_ws_probe_failed")

        load_ok, load_out, load_err, _ = _run_cmd(
            [
                "node",
                "-e",
                LOAD_WS_PROBE_SCRIPT,
                f"ws://127.0.0.1:{port}/ws",
                str(load_clients),
                str(load_timeout_ms),
            ],
            cwd=project_dir,
            timeout_sec=max(30.0, load_timeout_ms / 1000.0 + 10.0),
            env=env,
        )
        result.details["load_stdout"] = load_out[-1200:]
        result.details["load_stderr"] = load_err[-1200:]
        payload = _extract_json_line(load_out)
        if payload:
            result.load_success = int(payload.get("success", 0))
            result.load_failed = int(payload.get("failed", 0))
            result.load_p95_ms = float(payload.get("p95_ms", 0.0))
        elif load_ok:
            result.load_success = load_clients
            result.load_failed = 0
        else:
            result.load_success = 0
            result.load_failed = load_clients

        success_rate = (result.load_success / load_clients) if load_clients > 0 else 0.0
        result.load_ok = load_ok and success_rate >= load_min_success_rate
        if not result.load_ok:
            result.errors.append(
                f"load_probe_failed(rate={success_rate:.3f},threshold={load_min_success_rate:.3f})"
            )
    finally:
        try:
            server.terminate()
            server.wait(timeout=8)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass

    return result


def summarize_runs(runs: List[RunReport]) -> Dict[str, Any]:
    total = len(runs)
    incubation_ok = [item for item in runs if item.incubation_ok]
    runtime_ok = [item for item in runs if item.runtime_probe is not None and item.runtime_probe.all_checks_passed()]
    full_ok = [item for item in runs if item.full_success()]

    incubation_latencies = [item.incubation_seconds for item in incubation_ok]
    files_counts = [item.generated_file_count for item in incubation_ok if item.generated_file_count > 0]
    if incubation_latencies:
        p50 = statistics.median(incubation_latencies)
        sorted_latency = sorted(incubation_latencies)
        p95_idx = min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95))
        incubation_stats = {
            "p50_sec": round(p50, 2),
            "p95_sec": round(sorted_latency[p95_idx], 2),
            "max_sec": round(sorted_latency[-1], 2),
        }
    else:
        incubation_stats = {"p50_sec": 0.0, "p95_sec": 0.0, "max_sec": 0.0}

    if files_counts:
        file_stats = {
            "min": min(files_counts),
            "avg": round(sum(files_counts) / len(files_counts), 2),
            "max": max(files_counts),
        }
    else:
        file_stats = {"min": 0, "avg": 0.0, "max": 0}

    failures = []
    for item in runs:
        if item.full_success():
            continue
        if item.error:
            failures.append(item.error)
            continue
        if item.runtime_probe is not None and item.runtime_probe.errors:
            failures.extend(item.runtime_probe.errors)

    top_failures: Dict[str, int] = {}
    for reason in failures:
        top_failures[reason] = top_failures.get(reason, 0) + 1
    ordered_failures = sorted(top_failures.items(), key=lambda it: it[1], reverse=True)[:10]

    return {
        "total_runs": total,
        "incubation_success": len(incubation_ok),
        "runtime_success": len(runtime_ok),
        "full_success": len(full_ok),
        "incubation_success_rate": (len(incubation_ok) / total) if total else 0.0,
        "runtime_success_rate": (len(runtime_ok) / total) if total else 0.0,
        "full_success_rate": (len(full_ok) / total) if total else 0.0,
        "incubation_latency": incubation_stats,
        "generated_files": file_stats,
        "top_failures": [{"reason": key, "count": value} for key, value in ordered_failures],
    }


async def run(args: argparse.Namespace) -> int:
    service = ProjectIncubationService()
    required_files = args.required_file or list(DEFAULT_REQUIRED_FILES)
    reports: List[RunReport] = []
    now_ts = int(time.time())
    max_runs = args.max_runs if args.until_full_success else args.runs
    run_index = 1

    while run_index <= max_runs:
        project_name = f"{args.project_name_prefix} {run_index:02d}"
        started_at = _now_iso()
        run_report = RunReport(
            run_index=run_index,
            project_name=project_name,
            started_at=started_at,
        )
        print(f"[run {run_index}/{max_runs}] incubating {project_name} ...")

        started = time.perf_counter()
        try:
            spec = ProjectIncubationSpec(
                project_name=project_name,
                project_goal=args.project_goal,
                holon_id=args.holon_id,
                skill_name=args.skill_name,
                execution_payload={
                    "project_name": project_name,
                    "project_goal": args.project_goal,
                },
                required_files=required_files,
                evolution_timeout_seconds=args.evolution_timeout,
                poll_interval_seconds=args.poll_interval,
            )
            incubation = await service.incubate_project(spec)
            run_report.incubation_ok = True
            run_report.incubation_seconds = round(time.perf_counter() - started, 3)
            run_report.holon_id = incubation.holon_id
            run_report.request_id = incubation.request_id
            run_report.skill_id = incubation.skill_id
            run_report.output_dir = incubation.output_dir
            run_report.generated_file_count = incubation.generated_file_count

            port = _find_free_port() if args.auto_port else (args.base_port + run_index - 1)
            runtime_probe = await asyncio.to_thread(
                run_runtime_probe,
                Path(incubation.output_dir),
                port,
                args.ws_timeout_ms,
                args.load_clients,
                args.load_timeout_ms,
                args.load_min_success_rate,
                args.min_generated_files,
                args.min_source_files,
                args.max_placeholder_hits,
                args.npm_install_timeout_sec,
            )
            run_report.runtime_probe = runtime_probe
            print(
                f"[run {run_index}/{max_runs}] runtime "
                f"install={runtime_probe.install_ok} boot={runtime_probe.server_boot_ok} "
                f"smoke={runtime_probe.smoke_ok} strict={runtime_probe.strict_ws_ok} "
                f"load={runtime_probe.load_ok}"
            )
        except Exception as exc:
            run_report.incubation_seconds = round(time.perf_counter() - started, 3)
            run_report.error = f"{type(exc).__name__}: {exc}"
            print(f"[run {run_index}/{max_runs}] failed: {run_report.error}")
            if not args.continue_on_failure:
                reports.append(run_report)
                break

        reports.append(run_report)
        if args.until_full_success and run_report.full_success():
            break
        run_index += 1

    summary = summarize_runs(reports)
    payload = {
        "generated_at": _now_iso(),
        "seed_timestamp": now_ts,
        "config": {
            "runs": args.runs,
            "until_full_success": args.until_full_success,
            "max_runs": args.max_runs,
            "project_name_prefix": args.project_name_prefix,
            "project_goal": args.project_goal,
            "holon_id": args.holon_id,
            "skill_name": args.skill_name,
            "required_files": required_files,
            "evolution_timeout": args.evolution_timeout,
            "poll_interval": args.poll_interval,
            "base_port": args.base_port,
            "ws_timeout_ms": args.ws_timeout_ms,
            "load_clients": args.load_clients,
            "load_timeout_ms": args.load_timeout_ms,
            "load_min_success_rate": args.load_min_success_rate,
            "min_generated_files": args.min_generated_files,
            "min_source_files": args.min_source_files,
            "max_placeholder_hits": args.max_placeholder_hits,
            "npm_install_timeout_sec": args.npm_install_timeout_sec,
            "continue_on_failure": args.continue_on_failure,
        },
        "summary": summary,
        "runs": [
            {
                **asdict(item),
                "runtime_probe": asdict(item.runtime_probe) if item.runtime_probe is not None else None,
                "full_success": item.full_success(),
            }
            for item in reports
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 88)
    print("Autonomous MMO Fish Project Stress Summary")
    print("=" * 88)
    print(f"runs={summary['total_runs']}")
    print(f"incubation_success={summary['incubation_success']} rate={summary['incubation_success_rate']:.2%}")
    print(f"runtime_success={summary['runtime_success']} rate={summary['runtime_success_rate']:.2%}")
    print(f"full_success={summary['full_success']} rate={summary['full_success_rate']:.2%}")
    print(f"incubation_latency={summary['incubation_latency']}")
    print(f"generated_files={summary['generated_files']}")
    if summary["top_failures"]:
        print(f"top_failures={summary['top_failures']}")
    print(f"report_json={args.output_json}")
    print("=" * 88)

    if args.until_full_success:
        return 0 if summary["full_success"] >= 1 else 2
    return 0 if summary["full_success_rate"] >= args.min_full_success_rate else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress autonomous Holon project incubation for MMO fish game"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--until-full-success",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--project-name-prefix", default="Abyss Arena Autonomous")
    parser.add_argument("--project-goal", default=DEFAULT_PROJECT_GOAL)
    parser.add_argument("--holon-id", default=None)
    parser.add_argument("--skill-name", default=None)
    parser.add_argument("--required-file", action="append", default=[])
    parser.add_argument("--evolution-timeout", type=float, default=420.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--base-port", type=int, default=8900)
    parser.add_argument(
        "--auto-port",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ws-timeout-ms", type=int, default=12000)
    parser.add_argument("--load-clients", type=int, default=40)
    parser.add_argument("--load-timeout-ms", type=int, default=25000)
    parser.add_argument("--load-min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-generated-files", type=int, default=18)
    parser.add_argument("--min-source-files", type=int, default=6)
    parser.add_argument("--max-placeholder-hits", type=int, default=0)
    parser.add_argument("--npm-install-timeout-sec", type=float, default=180.0)
    parser.add_argument("--min-full-success-rate", type=float, default=0.67)
    parser.add_argument(
        "--continue-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("C:/Temp/holonpolis_stress/autonomous_mmo_full_stress_report.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
