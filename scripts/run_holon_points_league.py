#!/usr/bin/env python
"""Holon points league: 100 Holons, 100 rounds, top score ranking."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from holonpolis.bootstrap import bootstrap  # noqa: E402
from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy  # noqa: E402
from holonpolis.domain.skills import ToolSchema  # noqa: E402
from holonpolis.kernel.llm.llm_runtime import get_llm_runtime  # noqa: E402
from holonpolis.kernel.storage import HolonPathGuard  # noqa: E402
from holonpolis.runtime.holon_runtime import HolonRuntime, LoadedSkill  # noqa: E402
from holonpolis.services.evolution_service import EvolutionService  # noqa: E402
from holonpolis.services.holon_service import HolonService  # noqa: E402


SKILL_NAME = "Top3 Draw Number Predictor"
STRATEGY_MODES = ("balanced", "recency", "coverage", "volatility")


@dataclass
class HolonLeagueState:
    holon_id: str
    points: int
    hit3: int = 0
    hit2: int = 0
    hit1: int = 0
    hit0: int = 0
    rounds_played: int = 0
    total_hits: int = 0
    errors: int = 0
    strategy: Dict[str, Any] = field(default_factory=dict)
    evolution_step: int = 0
    last_round_hits: int = 0
    last_round_delta: int = 0
    best_points: int = 0
    best_strategy: Dict[str, Any] = field(default_factory=dict)
    skill_id: str = ""
    skill_code_hash: str = ""
    llm_evolution_count: int = 0


@dataclass
class RoundTopItem:
    holon_id: str
    points: int
    round_hits: int
    round_delta: int


@dataclass
class RoundSummary:
    round_index: int
    draw_id: str
    top: List[RoundTopItem]


@dataclass
class PredictorHandle:
    loaded_skill: LoadedSkill
    skill_id: str
    code_hash: str
    llm_evolved: bool


def _slugify(text: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_") or "skill"


def _read_json_array(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a JSON array")
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
    return out


def _normalize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in rows:
        raw_numbers = item.get("numbers")
        if not isinstance(raw_numbers, list):
            continue
        numbers: List[int] = []
        for raw in raw_numbers:
            try:
                value = int(raw)
            except Exception:
                continue
            if 1 <= value <= 49:
                numbers.append(value)
        numbers = sorted(set(numbers))
        if len(numbers) < 6:
            continue
        numbers = numbers[:6]
        out.append(
            {
                "draw_id": str(item.get("draw_id") or ""),
                "draw_date": str(item.get("draw_date") or ""),
                "numbers": numbers,
            }
        )
    return out


def _compact_strategy_text(strategy: Dict[str, Any]) -> str:
    normalized = _normalize_strategy(strategy)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _build_skill_description(
    holon_id: str,
    round_index: int,
    feedback: Optional[Dict[str, Any]],
) -> str:
    feedback_text = ""
    if isinstance(feedback, dict) and feedback:
        fields: List[str] = []
        for key in ("points", "last_hits", "last_delta", "hit3", "hit2", "hit1", "hit0"):
            if key in feedback:
                fields.append(f"{key}={feedback[key]}")
        if fields:
            feedback_text = " | recent_metrics: " + ", ".join(fields)

    return (
        "Evolve a deterministic top-3 primary-number predictor skill for points league scoring. "
        f"Holon={holon_id}, evolution_round={round_index}.{feedback_text}"
    )


def _build_skill_requirements(
    holon_id: str,
    round_index: int,
    strategy: Dict[str, Any],
    feedback: Optional[Dict[str, Any]],
) -> List[str]:
    normalized_strategy = _normalize_strategy(strategy)
    requirements: List[str] = [
        "Implement Python function execute(train_data, predict_context=None, round_index=1, strategy=None).",
        "train_data is a list of draw records; each draw has field numbers. Ignore special_number entirely.",
        "Use only main number history (numbers field), never predict or rely on special_number.",
        "Return dict with predicted_numbers: exactly 3 unique integers in range [1,49], sorted ascending.",
        "Prediction must be deterministic for identical inputs.",
        "execute must never raise exceptions; return a valid payload even for empty/invalid train_data.",
        "If strategy mode is unknown, gracefully fallback to balanced mode.",
        "Be defensive with malformed input: ignore invalid rows/numbers rather than crashing.",
        "No file I/O, no network calls, and no subprocess usage in skill code.",
        "Use strategy fields when provided: window_size, recency_scale, jump, seed, mode.",
        "Support strategy mode values: balanced, recency, coverage, volatility.",
        "Output should include lightweight meta fields useful for diagnostics.",
        f"Holon identity seed: {holon_id}. Use this for deterministic tie-breaking so different Holons can diverge.",
        f"Current league evolution round: {round_index}.",
        f"Current suggested strategy baseline: {_compact_strategy_text(normalized_strategy)}",
    ]
    if isinstance(feedback, dict) and feedback:
        safe_feedback = {
            key: feedback.get(key)
            for key in ("points", "last_hits", "last_delta", "hit3", "hit2", "hit1", "hit0")
        }
        requirements.append(
            "Recent performance feedback (for refinement only): "
            + json.dumps(safe_feedback, ensure_ascii=False, sort_keys=True)
        )
        requirements.append(
            "If last_hits <= 1 then increase exploration; if last_hits >= 2 then keep stable signal weights."
        )
    return requirements


def _compact_error_text(error: Any, limit: int = 360) -> str:
    raw = str(error or "").strip()
    raw = re.sub(r"\s+", " ", raw)
    if len(raw) <= limit:
        return raw
    return raw[:limit].rstrip() + "..."


def _evolution_phase_label(round_index: int) -> str:
    return "init" if int(round_index) <= 0 else f"round-{int(round_index)}"


def _short_hash(value: str, length: int = 8) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return text[: max(4, int(length))]


def _emit_progress(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def _validate_llm_provider_ready() -> Dict[str, str]:
    runtime = get_llm_runtime()
    bundle = runtime.get_provider_bundle()
    provider_id = str(bundle.get("default_provider_id") or "").strip()
    providers = bundle.get("providers")
    if not isinstance(providers, dict):
        providers = {}

    provider_cfg = providers.get(provider_id)
    if not isinstance(provider_cfg, dict):
        raise RuntimeError(f"LLM provider not configured: provider_id={provider_id!r}")

    provider_type = str(provider_cfg.get("type") or "").strip()
    model = str(provider_cfg.get("model") or "").strip()
    api_key = str(provider_cfg.get("api_key") or "").strip()
    needs_api_key = provider_type in {"openai_compat", "anthropic_compat", "gemini_api"}
    if needs_api_key and not api_key:
        raise RuntimeError(
            f"LLM provider '{provider_id}' requires api_key but none is configured (type={provider_type})."
        )

    return {
        "provider_id": provider_id,
        "provider_type": provider_type,
        "model": model,
    }


def _read_skill_code_hash(holon_id: str, skill_id: str) -> str:
    guard = HolonPathGuard(holon_id)
    slug = _slugify(skill_id)
    skill_dir = guard.resolve(f"skills_local/{slug}", must_exist=False)
    if not skill_dir.exists():
        skill_dir = guard.resolve(f"skills/{slug}", must_exist=False)
    code_path = skill_dir / "skill.py"
    if not code_path.exists():
        return ""
    content = code_path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _empty_llm_audit() -> Dict[str, Any]:
    return {
        "requested": False,
        "inflight": False,
        "call_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "last_status": "not_requested",
        "last_stage": "",
        "provider_id": "",
        "provider_type": "",
        "model": "",
        "last_error": "",
        "last_latency_ms": 0,
        "last_started_at": "",
        "last_completed_at": "",
    }


def _mark_cached_reuse_audit(
    holon_service: HolonService,
    holon_id: str,
    *,
    phase_label: str,
) -> None:
    holon_service.update_evolution_audit(
        holon_id,
        patch={
            "request_id": "",
            "lifecycle": "idle",
            "phase": str(phase_label),
            "result": "cached_reuse",
            "error": "",
            "fallback_to_cached_skill": False,
            "cache_reused_without_llm": True,
            "llm": _empty_llm_audit(),
        },
    )


def _mark_cached_fallback_audit(
    holon_service: HolonService,
    holon_id: str,
    *,
    phase_label: str,
    error: str,
) -> None:
    holon_service.update_evolution_audit(
        holon_id,
        patch={
            "phase": str(phase_label),
            "result": "fallback_cached",
            "error": str(error or ""),
            "fallback_to_cached_skill": True,
            "cache_reused_without_llm": False,
            "llm": {
                "inflight": False,
                "last_status": "fallback_cached",
                "last_error": str(error or ""),
            },
        },
    )


def _resolve_evolution_concurrency(args: argparse.Namespace) -> int:
    configured = getattr(args, "evolution_concurrency", None)
    if configured is not None:
        return max(1, int(configured))
    base = max(1, int(args.concurrency))
    if (bool(args.require_llm_evolution) or bool(args.force_evolve)) and int(args.population_size) > 1:
        return max(2, base)
    return base


def _tool_schema() -> ToolSchema:
    return ToolSchema(
        name="execute",
        description="Predict three primary numbers from historical draw sequences.",
        parameters={
            "type": "object",
            "properties": {
                "train_data": {"type": "array", "items": {"type": "object"}},
                "predict_context": {"type": "object"},
                "round_index": {"type": "integer"},
                "strategy": {"type": "object"},
            },
            "required": ["train_data"],
            "additionalProperties": False,
        },
        required=["train_data"],
    )


def _seed_int(*parts: Any) -> int:
    payload = "|".join(str(item) for item in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _normalize_strategy(strategy: Dict[str, Any]) -> Dict[str, Any]:
    try:
        window_size = int(strategy.get("window_size", 420))
    except Exception:
        window_size = 420
    window_size = max(60, min(4200, window_size))

    try:
        recency_scale = float(strategy.get("recency_scale", 0.9))
    except Exception:
        recency_scale = 0.9
    recency_scale = max(0.0, min(3.0, recency_scale))

    try:
        jump = int(strategy.get("jump", 5))
    except Exception:
        jump = 5
    jump = max(1, min(17, jump))

    try:
        seed = int(strategy.get("seed", 0))
    except Exception:
        seed = 0
    seed = seed % 49

    mode = str(strategy.get("mode", "balanced")).strip().lower()
    if mode not in STRATEGY_MODES:
        mode = "balanced"

    return {
        "window_size": window_size,
        "recency_scale": round(recency_scale, 4),
        "jump": jump,
        "seed": seed,
        "mode": mode,
    }


def _strategy_for_holon(holon_id: str) -> Dict[str, Any]:
    digest = hashlib.sha256(holon_id.encode("utf-8")).hexdigest()
    a = int(digest[0:8], 16)
    b = int(digest[8:16], 16)
    c = int(digest[16:24], 16)
    d = int(digest[24:32], 16)
    mode = STRATEGY_MODES[int(digest[32:34], 16) % len(STRATEGY_MODES)]
    return _normalize_strategy(
        {
        "window_size": 100 + (a % 2200),
        "recency_scale": round(0.2 + (b % 250) / 100.0, 4),
        "jump": 1 + (c % 17),
        "seed": d % 49,
            "mode": mode,
        }
    )


def _evolve_strategy(
    holon_id: str,
    current: Dict[str, Any],
    round_index: int,
    hit_count: int,
    mentor_strategy: Optional[Dict[str, Any]],
    mentor_adopt_rate: float,
) -> Dict[str, Any]:
    rng = random.Random(_seed_int(holon_id, round_index, "strategy_evolve"))
    base = dict(current)
    if (
        mentor_strategy
        and hit_count <= 1
        and rng.random() < max(0.0, min(1.0, mentor_adopt_rate))
    ):
        base = dict(mentor_strategy)

    if hit_count >= 3:
        scale = 0.25
    elif hit_count == 2:
        scale = 0.55
    else:
        scale = 1.0

    base["window_size"] = int(base.get("window_size", 420)) + int(rng.randint(-260, 260) * scale)
    base["recency_scale"] = float(base.get("recency_scale", 0.9)) + float(rng.uniform(-0.5, 0.5) * scale)

    if rng.random() < (0.20 if hit_count >= 2 else 0.65):
        base["jump"] = int(base.get("jump", 5)) + rng.choice([-3, -2, -1, 1, 2, 3])

    if rng.random() < (0.18 if hit_count >= 2 else 0.72):
        base["seed"] = int(base.get("seed", 0)) + rng.randint(1, 19)

    if rng.random() < (0.10 if hit_count >= 2 else 0.45):
        base["mode"] = rng.choice(STRATEGY_MODES)

    evolved = _normalize_strategy(base)
    if evolved == _normalize_strategy(current):
        evolved["seed"] = (int(evolved["seed"]) + 1) % 49
    return evolved


def _mentor_strategy_for_holon(
    holon_id: str,
    round_index: int,
    ranked_states: Sequence[HolonLeagueState],
    mentor_top_k: int,
) -> Optional[Dict[str, Any]]:
    if not ranked_states:
        return None
    k = max(1, min(int(mentor_top_k), len(ranked_states)))
    mentors = ranked_states[:k]
    idx = _seed_int(holon_id, round_index, "mentor_pick") % len(mentors)
    return dict(mentors[idx].strategy)


async def _ensure_population(
    holon_service: HolonService,
    population_size: int,
    holon_prefix: str,
) -> List[str]:
    out: List[str] = []
    created = 0
    for index in range(1, population_size + 1):
        holon_id = f"{holon_prefix}_{index:03d}"
        out.append(holon_id)
        if holon_service.holon_exists(holon_id):
            continue
        blueprint = Blueprint(
            blueprint_id=f"bp_{holon_id}",
            holon_id=holon_id,
            species_id="generalist",
            name=f"Points Predictor {index:03d}",
            purpose="Compete in points-based number prediction league.",
            boundary=Boundary(
                allowed_tools=["evolution.request", "skill.execute"],
                denied_tools=[],
                allow_file_write=False,
                allow_network=False,
                allow_subprocess=False,
            ),
            evolution_policy=EvolutionPolicy(),
        )
        await holon_service.create_holon(blueprint)
        created += 1
    _emit_progress(
        "population",
        f"ready total={len(out)} created={created} reused={len(out) - created}",
    )
    return out


async def _ensure_skill(
    holon_id: str,
    holon_service: HolonService,
    evo: EvolutionService,
    skill_name: str,
    schema: ToolSchema,
    strategy: Dict[str, Any],
    round_index: int,
    feedback: Optional[Dict[str, Any]],
    force_evolve: bool,
    require_llm_evolution: bool,
    strict_llm_evolution: bool,
    llm_max_attempts: int,
    llm_outer_retries: int,
) -> tuple[str, str, bool]:
    slug = _slugify(skill_name)
    phase_label = _evolution_phase_label(round_index)
    cached_valid = False
    if not force_evolve:
        status = await evo.validate_existing_skill(holon_id, skill_name)
        cached_valid = bool(status.get("valid"))
        if cached_valid and not require_llm_evolution:
            _mark_cached_reuse_audit(
                holon_service,
                holon_id,
                phase_label=phase_label,
            )
            _emit_progress(
                "evolve",
                f"{phase_label} {holon_id} reuse cached skill={slug}",
            )
            return slug, _read_skill_code_hash(holon_id, slug), False

    if not require_llm_evolution and not cached_valid:
        raise RuntimeError(
            "No valid cached skill found while --require-llm-evolution is disabled."
        )

    base_description = _build_skill_description(
        holon_id=holon_id,
        round_index=round_index,
        feedback=feedback,
    )
    base_requirements = _build_skill_requirements(
        holon_id=holon_id,
        round_index=round_index,
        strategy=strategy,
        feedback=feedback,
    )

    last_error = ""
    attempts = max(1, int(llm_outer_retries))
    for outer_try in range(1, attempts + 1):
        requirements = list(base_requirements)
        description = f"{base_description} | outer_try={outer_try}/{attempts}"
        _emit_progress(
            "evolve",
            f"{phase_label} {holon_id} start try={outer_try}/{attempts}",
        )
        if last_error:
            requirements.append(f"Previous failure: {last_error}")
            requirements.append(
                "Fix previous failure and ensure generated tests fully pass with robust execute() behavior."
            )

        result = await evo.evolve_skill_autonomous(
            holon_id=holon_id,
            skill_name=skill_name,
            description=description,
            requirements=requirements,
            tool_schema=schema,
            version="0.1.0",
            max_attempts=max(1, int(llm_max_attempts)),
        )
        if result.success:
            skill_id = str(result.skill_id or slug)
            code_hash = ""
            if result.attestation is not None:
                code_hash = str(result.attestation.code_hash or "")
            if not code_hash:
                code_hash = _read_skill_code_hash(holon_id, skill_id)
            _emit_progress(
                "evolve",
                f"{phase_label} {holon_id} success skill={skill_id} code={_short_hash(code_hash)} try={outer_try}/{attempts}",
            )
            return skill_id, code_hash, True
        last_error = _compact_error_text(f"{result.phase}: {result.error_message}")
        _emit_progress(
            "evolve",
            f"{phase_label} {holon_id} retry try={outer_try}/{attempts} reason={last_error}",
        )

    error = f"evolve_failed after {attempts} outer tries error={last_error}"
    if strict_llm_evolution:
        holon_service.update_evolution_audit(
            holon_id,
            patch={
                "phase": str(phase_label),
                "result": "failed_strict",
                "error": str(error),
                "fallback_to_cached_skill": False,
                "llm": {
                    "inflight": False,
                    "last_status": "failed_strict",
                    "last_error": str(error),
                },
            },
        )
        _emit_progress(
            "evolve",
            f"{phase_label} {holon_id} fail strict reason={_compact_error_text(error, limit=220)}",
        )
        raise RuntimeError(error)

    if not cached_valid:
        status = await evo.validate_existing_skill(holon_id, skill_name)
        cached_valid = bool(status.get("valid"))
    if cached_valid:
        _mark_cached_fallback_audit(
            holon_service,
            holon_id,
            phase_label=phase_label,
            error=error,
        )
        _emit_progress(
            "evolve",
            f"{phase_label} {holon_id} fallback cached skill={slug}",
        )
        return slug, _read_skill_code_hash(holon_id, slug), False
    holon_service.update_evolution_audit(
        holon_id,
        patch={
            "phase": str(phase_label),
            "result": "failed",
            "error": str(error),
            "fallback_to_cached_skill": False,
            "llm": {
                "inflight": False,
                "last_status": "failed",
                "last_error": str(error),
            },
        },
    )
    _emit_progress(
        "evolve",
        f"{phase_label} {holon_id} fail reason={_compact_error_text(error, limit=220)}",
    )
    raise RuntimeError(error)


async def _prepare_predictors(
    holon_ids: Sequence[str],
    holon_service: HolonService,
    evo: EvolutionService,
    skill_name: str,
    schema: ToolSchema,
    strategy_map: Dict[str, Dict[str, Any]],
    force_evolve: bool,
    require_llm_evolution: bool,
    strict_llm_evolution: bool,
    llm_max_attempts: int,
    llm_outer_retries: int,
    evolution_concurrency: int,
) -> Dict[str, PredictorHandle]:
    _emit_progress(
        "batch",
        f"init-evolve holons={len(holon_ids)} evolution_concurrency={max(1, int(evolution_concurrency))}",
    )
    sem = asyncio.Semaphore(max(1, int(evolution_concurrency)))

    async def _build_for(hid: str) -> tuple[str, PredictorHandle]:
        async with sem:
            skill_id, code_hash, llm_evolved = await _ensure_skill(
                holon_id=hid,
                holon_service=holon_service,
                evo=evo,
                skill_name=skill_name,
                schema=schema,
                strategy=dict(strategy_map.get(hid) or _strategy_for_holon(hid)),
                round_index=0,
                feedback=None,
                force_evolve=force_evolve,
                require_llm_evolution=require_llm_evolution,
                strict_llm_evolution=strict_llm_evolution,
                llm_max_attempts=llm_max_attempts,
                llm_outer_retries=llm_outer_retries,
            )
            runtime = HolonRuntime(
                holon_id=hid,
                blueprint=holon_service.get_blueprint(hid),
            )
            loaded = runtime.get_skill(skill_id)
            return hid, PredictorHandle(
                loaded_skill=loaded,
                skill_id=skill_id,
                code_hash=code_hash,
                llm_evolved=llm_evolved,
            )

    built = await asyncio.gather(*(_build_for(hid) for hid in holon_ids))
    llm_count = sum(1 for _, handle in built if handle.llm_evolved)
    _emit_progress(
        "batch",
        f"init-evolve complete ready={len(built)} llm_evolved={llm_count}",
    )
    return {hid: handle for hid, handle in built}


def _feedback_from_state(state: HolonLeagueState) -> Dict[str, Any]:
    return {
        "points": int(state.points),
        "last_hits": int(state.last_round_hits),
        "last_delta": int(state.last_round_delta),
        "hit3": int(state.hit3),
        "hit2": int(state.hit2),
        "hit1": int(state.hit1),
        "hit0": int(state.hit0),
    }


async def _re_evolve_predictors(
    round_index: int,
    predictors: Dict[str, PredictorHandle],
    scoreboard: Dict[str, HolonLeagueState],
    holon_service: HolonService,
    evo: EvolutionService,
    skill_name: str,
    schema: ToolSchema,
    llm_max_attempts: int,
    llm_outer_retries: int,
    strict_llm_evolution: bool,
    evolution_concurrency: int,
) -> Dict[str, PredictorHandle]:
    _emit_progress(
        "batch",
        f"re-evolve { _evolution_phase_label(round_index) } holons={len(predictors)} evolution_concurrency={max(1, int(evolution_concurrency))}",
    )
    sem = asyncio.Semaphore(max(1, int(evolution_concurrency)))

    async def _refresh_one(hid: str) -> tuple[str, PredictorHandle]:
        async with sem:
            state = scoreboard[hid]
            skill_id, code_hash, llm_evolved = await _ensure_skill(
                holon_id=hid,
                holon_service=holon_service,
                evo=evo,
                skill_name=skill_name,
                schema=schema,
                strategy=dict(state.strategy),
                round_index=round_index,
                feedback=_feedback_from_state(state),
                force_evolve=True,
                require_llm_evolution=True,
                strict_llm_evolution=strict_llm_evolution,
                llm_max_attempts=llm_max_attempts,
                llm_outer_retries=llm_outer_retries,
            )
            runtime = HolonRuntime(
                holon_id=hid,
                blueprint=holon_service.get_blueprint(hid),
            )
            loaded = runtime.get_skill(skill_id)
            return hid, PredictorHandle(
                loaded_skill=loaded,
                skill_id=skill_id,
                code_hash=code_hash,
                llm_evolved=llm_evolved,
            )

    refreshed = await asyncio.gather(*(_refresh_one(hid) for hid in predictors.keys()))
    llm_count = sum(1 for _, handle in refreshed if handle.llm_evolved)
    _emit_progress(
        "batch",
        f"re-evolve { _evolution_phase_label(round_index) } complete refreshed={len(refreshed)} llm_evolved={llm_count}",
    )
    return {hid: handle for hid, handle in refreshed}


def _round_delta(hit_count: int, hit3: int, hit2: int, hit1: int, hit0: int) -> int:
    if hit_count >= 3:
        return int(hit3)
    if hit_count == 2:
        return int(hit2)
    if hit_count == 1:
        return int(hit1)
    if hit_count == 0:
        return int(hit0)
    return 0


def _skill_diversity_summary(scoreboard: Dict[str, HolonLeagueState]) -> Dict[str, Any]:
    total = len(scoreboard)
    hashes = [state.skill_code_hash for state in scoreboard.values() if state.skill_code_hash]
    unique_hashes = len(set(hashes))
    ratio = 0.0 if total <= 0 else float(unique_hashes) / float(total)
    return {
        "total_holons": total,
        "with_hash": len(hashes),
        "unique_code_hashes": unique_hashes,
        "unique_ratio": round(ratio, 4),
    }


async def run_league(args: argparse.Namespace) -> int:
    bootstrap()
    if int(args.population_size) <= 0:
        raise ValueError("population_size must be > 0")
    if int(args.rounds) <= 0:
        raise ValueError("rounds must be > 0")
    if int(args.llm_max_attempts) <= 0:
        raise ValueError("llm_max_attempts must be > 0")
    if int(args.llm_outer_retries) <= 0:
        raise ValueError("llm_outer_retries must be > 0")
    evolution_concurrency = _resolve_evolution_concurrency(args)
    if (
        (bool(args.require_llm_evolution) or bool(args.force_evolve))
        and int(args.population_size) > 1
        and getattr(args, "evolution_concurrency", None) is not None
        and evolution_concurrency < 2
    ):
        raise ValueError(
            "evolution_concurrency must be >= 2 when LLM evolution is enabled for multiple Holons."
        )

    llm_provider_info: Dict[str, str] = {}
    if bool(args.require_llm_evolution) or bool(args.force_evolve):
        llm_provider_info = _validate_llm_provider_ready()
        _emit_progress(
            "llm",
            "provider="
            f"{llm_provider_info.get('provider_id', '-')}"
            f" type={llm_provider_info.get('provider_type', '-')}"
            f" model={llm_provider_info.get('model', '-')}",
        )
        _emit_progress(
            "llm",
            f"evolution_concurrency={int(evolution_concurrency)} prediction_concurrency={max(1, int(args.concurrency))}",
        )

    train_rows = _normalize_rows(_read_json_array(Path(args.train_data)))
    predict_rows = _normalize_rows(_read_json_array(Path(args.predict_data)))
    if not train_rows:
        raise ValueError("train_data has no valid rows")
    if not predict_rows:
        raise ValueError("predict_data has no valid rows")

    holon_service = HolonService()
    evo = EvolutionService()

    holon_ids = await _ensure_population(
        holon_service=holon_service,
        population_size=args.population_size,
        holon_prefix=args.holon_prefix,
    )
    _emit_progress(
        "league",
        f"population={len(holon_ids)} rounds={int(args.rounds)} start_points={int(args.start_points)}",
    )

    strategy_map = {
        hid: _strategy_for_holon(hid)
        for hid in holon_ids
    }

    predictors = await _prepare_predictors(
        holon_ids=holon_ids,
        holon_service=holon_service,
        evo=evo,
        skill_name=args.skill_name,
        schema=_tool_schema(),
        strategy_map=strategy_map,
        force_evolve=bool(args.force_evolve or args.require_llm_evolution),
        require_llm_evolution=bool(args.require_llm_evolution),
        strict_llm_evolution=bool(args.strict_llm_evolution),
        llm_max_attempts=int(args.llm_max_attempts),
        llm_outer_retries=int(args.llm_outer_retries),
        evolution_concurrency=int(evolution_concurrency),
    )

    scoreboard: Dict[str, HolonLeagueState] = {}
    for hid, handle in predictors.items():
        initial_strategy = strategy_map.get(hid) or _strategy_for_holon(hid)
        scoreboard[hid] = HolonLeagueState(
            holon_id=hid,
            points=int(args.start_points),
            strategy=dict(initial_strategy),
            best_points=int(args.start_points),
            best_strategy=dict(initial_strategy),
            skill_id=str(handle.skill_id),
            skill_code_hash=str(handle.code_hash),
            llm_evolution_count=1 if bool(handle.llm_evolved) else 0,
        )
    rounds: List[RoundSummary] = []

    for round_index in range(1, int(args.rounds) + 1):
        target = predict_rows[(round_index - 1) % len(predict_rows)]
        truth = set(int(v) for v in target["numbers"])
        sem = asyncio.Semaphore(max(1, int(args.concurrency)))
        per_round: List[RoundTopItem] = []

        async def _run_one(hid: str, handle: PredictorHandle) -> None:
            async with sem:
                state = scoreboard[hid]
                try:
                    out = await handle.loaded_skill.execute(
                        {
                            "train_data": train_rows,
                            "predict_context": {
                                "draw_id": target["draw_id"],
                                "draw_date": target["draw_date"],
                            },
                            "round_index": round_index,
                            "strategy": dict(state.strategy),
                        }
                    )
                    raw = out.get("predicted_numbers") if isinstance(out, dict) else []
                    if not isinstance(raw, list):
                        raw = []
                    predicted = sorted({int(v) for v in raw if 1 <= int(v) <= 49})
                    predicted = predicted[:3]
                    if len(predicted) < 3:
                        raise ValueError("predicted_numbers must contain 3 valid numbers")
                    hits = len(truth.intersection(set(predicted)))
                except Exception:
                    hits = 0
                    state.errors += 1

                delta = _round_delta(
                    hit_count=hits,
                    hit3=args.points_hit3,
                    hit2=args.points_hit2,
                    hit1=args.points_hit1,
                    hit0=args.points_hit0,
                )
                state.points += int(delta)
                state.rounds_played += 1
                state.total_hits += int(hits)
                state.last_round_hits = int(hits)
                state.last_round_delta = int(delta)
                if hits >= 3:
                    state.hit3 += 1
                elif hits == 2:
                    state.hit2 += 1
                elif hits == 1:
                    state.hit1 += 1
                else:
                    state.hit0 += 1
                per_round.append(
                    RoundTopItem(
                        holon_id=hid,
                        points=state.points,
                        round_hits=hits,
                        round_delta=int(delta),
                    )
                )

        await asyncio.gather(
            *(_run_one(hid, handle) for hid, handle in predictors.items())
        )

        ranked_states = sorted(
            scoreboard.values(),
            key=lambda item: (-item.points, -item.hit3, -item.hit2, item.holon_id),
        )
        for state in ranked_states:
            if state.points > state.best_points:
                state.best_points = int(state.points)
                state.best_strategy = dict(state.strategy)

        for state in ranked_states:
            mentor = _mentor_strategy_for_holon(
                holon_id=state.holon_id,
                round_index=round_index,
                ranked_states=ranked_states,
                mentor_top_k=args.mentor_top_k,
            )
            evolved = _evolve_strategy(
                holon_id=state.holon_id,
                current=state.strategy,
                round_index=round_index,
                hit_count=state.last_round_hits,
                mentor_strategy=mentor,
                mentor_adopt_rate=args.mentor_adopt_rate,
            )
            # If drifted too far below its best checkpoint, re-anchor then mutate once.
            if state.points < state.best_points - 200 and state.best_strategy:
                evolved = _evolve_strategy(
                    holon_id=state.holon_id,
                    current=state.best_strategy,
                    round_index=round_index + 100000,
                    hit_count=state.last_round_hits,
                    mentor_strategy=mentor,
                    mentor_adopt_rate=args.mentor_adopt_rate,
                )
            state.strategy = evolved
            state.evolution_step += 1

        if int(args.llm_re_evolve_interval) > 0 and round_index % int(args.llm_re_evolve_interval) == 0:
            predictors = await _re_evolve_predictors(
                round_index=round_index,
                predictors=predictors,
                scoreboard=scoreboard,
                holon_service=holon_service,
                evo=evo,
                skill_name=args.skill_name,
                schema=_tool_schema(),
                llm_max_attempts=int(args.llm_max_attempts),
                llm_outer_retries=int(args.llm_outer_retries),
                strict_llm_evolution=bool(args.strict_llm_evolution),
                evolution_concurrency=int(evolution_concurrency),
            )
            for hid, handle in predictors.items():
                state = scoreboard[hid]
                state.skill_id = str(handle.skill_id)
                state.skill_code_hash = str(handle.code_hash)
                if handle.llm_evolved:
                    state.llm_evolution_count += 1

        ranked_round = sorted(per_round, key=lambda item: (-item.points, -item.round_hits, item.holon_id))
        rounds.append(
            RoundSummary(
                round_index=round_index,
                draw_id=target["draw_id"],
                top=ranked_round[:10],
            )
        )
        print(
            f"[round {round_index}/{args.rounds}] draw={target['draw_id']} "
            f"leader={ranked_round[0].holon_id} points={ranked_round[0].points}"
        )

    final_rank = sorted(
        scoreboard.values(),
        key=lambda item: (
            -item.points,
            -item.hit3,
            -item.hit2,
            item.holon_id,
        ),
    )

    output = {
        "config": {
            "train_data": str(args.train_data),
            "predict_data": str(args.predict_data),
            "population_size": int(args.population_size),
            "rounds": int(args.rounds),
            "start_points": int(args.start_points),
            "points_hit3": int(args.points_hit3),
            "points_hit2": int(args.points_hit2),
            "points_hit1": int(args.points_hit1),
            "points_hit0": int(args.points_hit0),
            "holon_prefix": args.holon_prefix,
            "skill_name": args.skill_name,
            "force_evolve": bool(args.force_evolve),
            "require_llm_evolution": bool(args.require_llm_evolution),
            "strict_llm_evolution": bool(args.strict_llm_evolution),
            "llm_max_attempts": int(args.llm_max_attempts),
            "llm_outer_retries": int(args.llm_outer_retries),
            "llm_re_evolve_interval": int(args.llm_re_evolve_interval),
            "concurrency": int(args.concurrency),
            "evolution_concurrency": int(evolution_concurrency),
            "mentor_top_k": int(args.mentor_top_k),
            "mentor_adopt_rate": float(args.mentor_adopt_rate),
            "llm_provider": llm_provider_info,
        },
        "skill_diversity": _skill_diversity_summary(scoreboard),
        "rounds": [
            {
                "round_index": row.round_index,
                "draw_id": row.draw_id,
                "top": [asdict(item) for item in row.top],
            }
            for row in rounds
        ],
        "final_ranking": [asdict(item) for item in final_rank],
        "top10": [asdict(item) for item in final_rank[:10]],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 88)
    print("Holon Points League Complete")
    print("=" * 88)
    print(f"rounds={args.rounds} holons={args.population_size}")
    print(f"best={final_rank[0].holon_id} points={final_rank[0].points}")
    diversity = _skill_diversity_summary(scoreboard)
    print(
        "skill_diversity="
        f"{diversity['unique_code_hashes']}/{diversity['total_holons']} "
        f"(ratio={diversity['unique_ratio']})"
    )
    print(f"report_json={output_path}")
    print("=" * 88)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run points-based Holon prediction league."
    )
    parser.add_argument(
        "--train-data",
        default=r"C:\Users\dains\Documents\Git\SixData\train_data.json",
    )
    parser.add_argument(
        "--predict-data",
        default=r"C:\Users\dains\Documents\Git\SixData\predict_data.json",
    )
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--start-points", type=int, default=10000)
    parser.add_argument("--points-hit3", type=int, default=650)
    parser.add_argument("--points-hit2", type=int, default=20)
    parser.add_argument("--points-hit1", type=int, default=-1)
    parser.add_argument("--points-hit0", type=int, default=-1)
    parser.add_argument("--holon-prefix", default="holon_points_predictor")
    parser.add_argument("--skill-name", default=SKILL_NAME)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--evolution-concurrency", type=int, default=None)
    parser.add_argument("--mentor-top-k", type=int, default=5)
    parser.add_argument("--mentor-adopt-rate", type=float, default=0.65)
    parser.add_argument(
        "--force-evolve",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-llm-evolution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-llm-evolution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--llm-max-attempts", type=int, default=6)
    parser.add_argument("--llm-outer-retries", type=int, default=3)
    parser.add_argument("--llm-re-evolve-interval", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default=r"C:\Temp\holonpolis_stress\holon_points_league_100r.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_league(parse_args())))
