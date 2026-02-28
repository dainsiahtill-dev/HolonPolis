#!/usr/bin/env python
"""Evolve 100 Holons to predict number draws and retain top-5 by win rate."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from holonpolis.bootstrap import bootstrap  # noqa: E402
from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy  # noqa: E402
from holonpolis.domain.skills import ToolSchema  # noqa: E402
from holonpolis.runtime.holon_runtime import HolonRuntime  # noqa: E402
from holonpolis.services.evolution_service import EvolutionService  # noqa: E402
from holonpolis.services.holon_service import HolonService  # noqa: E402


SKILL_NAME = "Structured Sequence Predictor"


@dataclass
class HolonScore:
    holon_id: str
    round_index: int
    main_hits: int
    special_hits: int
    exact_draw_hits: int
    draw_count: int
    main_hit_rate: float
    special_hit_rate: float
    win_rate: float
    strategy: Dict[str, Any]
    error: str = ""


@dataclass
class RoundReport:
    round_index: int
    active_holons: int
    kept_holons: int
    retired_holons: int
    top: List[HolonScore]
    retired_ids: List[str]


def _slugify(text: str) -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw.strip("_") or "skill"


def _read_json_array(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    normalized: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def _build_skill_code() -> str:
    return '''
from __future__ import annotations

from typing import Any, Dict, List

MAX_NUMBER = 49
MAIN_COUNT = 6


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _normalize_draws(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        raw_numbers = item.get("numbers")
        if not isinstance(raw_numbers, list):
            continue
        numbers: List[int] = []
        for raw in raw_numbers:
            value = _safe_int(raw, -1)
            if 1 <= value <= MAX_NUMBER:
                numbers.append(value)
        numbers = sorted(set(numbers))
        if len(numbers) < MAIN_COUNT:
            continue
        numbers = numbers[:MAIN_COUNT]
        special = _safe_int(item.get("special_number"), 1)
        if special < 1 or special > MAX_NUMBER:
            special = 1
        out.append(
            {
                "draw_id": str(item.get("draw_id") or ""),
                "numbers": numbers,
                "special_number": special,
            }
        )
    return out


def _score_space(
    train_rows: List[Dict[str, Any]],
    window_size: int,
    recency_scale: float,
) -> tuple[Dict[int, float], Dict[int, float]]:
    selected = train_rows[-window_size:] if window_size < len(train_rows) else list(train_rows)
    number_scores = {num: 0.0 for num in range(1, MAX_NUMBER + 1)}
    special_scores = {num: 0.0 for num in range(1, MAX_NUMBER + 1)}
    total = len(selected)
    if total == 0:
        return number_scores, special_scores

    for idx, row in enumerate(selected):
        recency = float(idx + 1) / float(total)
        weight = 1.0 + recency * recency_scale
        for value in row["numbers"]:
            number_scores[value] += weight
        special_scores[row["special_number"]] += weight
    return number_scores, special_scores


def _rank_scores(scores: Dict[int, float]) -> List[int]:
    return [key for key, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _pick_main_numbers(ranked: List[int], offset: int) -> List[int]:
    if not ranked:
        return [1, 2, 3, 4, 5, 6]
    normalized = offset % len(ranked)
    rotated = ranked[normalized:] + ranked[:normalized]
    picks: List[int] = []
    for value in rotated:
        if value not in picks:
            picks.append(value)
        if len(picks) >= MAIN_COUNT:
            break
    return sorted(picks[:MAIN_COUNT])


def execute(
    train_data: List[Dict[str, Any]],
    predict_data: List[Dict[str, Any]],
    strategy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    train_rows = _normalize_draws(train_data)
    predict_rows = _normalize_draws(predict_data)
    config = strategy if isinstance(strategy, dict) else {}

    window_size = _clamp_int(_safe_int(config.get("window_size"), 400), 60, 4000)
    recency_scale = _safe_float(config.get("recency_scale"), 0.8)
    if recency_scale < 0.0:
        recency_scale = 0.0
    if recency_scale > 3.0:
        recency_scale = 3.0
    offset_step = _clamp_int(_safe_int(config.get("offset_step"), 3), 1, 11)
    seed = _clamp_int(_safe_int(config.get("seed"), 0), 0, MAX_NUMBER - 1)

    number_scores, special_scores = _score_space(
        train_rows=train_rows,
        window_size=window_size,
        recency_scale=recency_scale,
    )
    ranked_main = _rank_scores(number_scores)
    ranked_special = _rank_scores(special_scores)

    predictions: List[Dict[str, Any]] = []
    for index, target in enumerate(predict_rows):
        offset = (seed + index * offset_step) % MAX_NUMBER
        numbers = _pick_main_numbers(ranked_main, offset)
        special_number = 1
        for candidate in ranked_special:
            if candidate not in numbers:
                special_number = candidate
                break
        predictions.append(
            {
                "draw_id": target["draw_id"],
                "numbers": numbers,
                "special_number": special_number,
            }
        )

    return {
        "predictions": predictions,
        "strategy_used": {
            "window_size": window_size,
            "recency_scale": recency_scale,
            "offset_step": offset_step,
            "seed": seed,
        },
        "meta": {
            "train_rows": len(train_rows),
            "predict_rows": len(predict_rows),
        },
    }
'''


def _build_skill_tests() -> str:
    return '''
from skill_module import execute


def _train_rows():
    return [
        {"draw_id": "a", "numbers": [1, 2, 3, 4, 5, 6], "special_number": 7},
        {"draw_id": "b", "numbers": [1, 2, 3, 8, 9, 10], "special_number": 11},
        {"draw_id": "c", "numbers": [1, 12, 13, 14, 15, 16], "special_number": 17},
    ]


def _predict_rows():
    return [
        {"draw_id": "p1", "numbers": [1, 2, 3, 4, 5, 6], "special_number": 7},
        {"draw_id": "p2", "numbers": [2, 3, 4, 5, 6, 7], "special_number": 8},
    ]


def test_execute_output_shape():
    out = execute(_train_rows(), _predict_rows(), strategy={"seed": 9})
    assert isinstance(out, dict)
    assert "predictions" in out
    predictions = out["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    for item in predictions:
        assert isinstance(item, dict)
        numbers = item.get("numbers")
        assert isinstance(numbers, list)
        assert len(numbers) == 6
        assert numbers == sorted(numbers)
        assert len(set(numbers)) == 6
        assert all(1 <= int(value) <= 49 for value in numbers)
        special = int(item.get("special_number"))
        assert 1 <= special <= 49
        assert special not in numbers


def test_execute_is_deterministic():
    a = execute(_train_rows(), _predict_rows(), strategy={"seed": 4, "window_size": 120})
    b = execute(_train_rows(), _predict_rows(), strategy={"seed": 4, "window_size": 120})
    assert a == b
'''


def _build_tool_schema() -> ToolSchema:
    return ToolSchema(
        name="execute",
        description="Predict structured draw numbers from historical training data.",
        parameters={
            "type": "object",
            "properties": {
                "train_data": {"type": "array", "items": {"type": "object"}},
                "predict_data": {"type": "array", "items": {"type": "object"}},
                "strategy": {"type": "object"},
            },
            "required": ["train_data", "predict_data"],
            "additionalProperties": False,
        },
        required=["train_data", "predict_data"],
    )


def _strategy_for_holon(holon_id: str) -> Dict[str, Any]:
    digest = hashlib.sha256(holon_id.encode("utf-8")).hexdigest()
    a = int(digest[0:8], 16)
    b = int(digest[8:16], 16)
    c = int(digest[16:24], 16)
    d = int(digest[24:32], 16)
    return {
        "window_size": 120 + (a % 1800),
        "recency_scale": round(0.2 + (b % 240) / 100.0, 3),
        "offset_step": 1 + (c % 11),
        "seed": d % 49,
    }


def _normalize_prediction_map(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in predictions:
        if not isinstance(item, dict):
            continue
        draw_id = str(item.get("draw_id") or "").strip()
        raw_numbers = item.get("numbers")
        if not draw_id or not isinstance(raw_numbers, list):
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
        if len(numbers) != 6:
            continue
        try:
            special = int(item.get("special_number"))
        except Exception:
            continue
        if not (1 <= special <= 49):
            continue
        out[draw_id] = {
            "numbers": numbers,
            "special_number": special,
        }
    return out


def _score_predictions(
    holon_id: str,
    round_index: int,
    truth_rows: Sequence[Dict[str, Any]],
    predicted_rows: Sequence[Dict[str, Any]],
    strategy: Dict[str, Any],
    error: str = "",
) -> HolonScore:
    truth_map = _normalize_prediction_map(truth_rows)
    pred_map = _normalize_prediction_map(predicted_rows)
    draw_count = len(truth_map)
    if draw_count <= 0:
        return HolonScore(
            holon_id=holon_id,
            round_index=round_index,
            main_hits=0,
            special_hits=0,
            exact_draw_hits=0,
            draw_count=0,
            main_hit_rate=0.0,
            special_hit_rate=0.0,
            win_rate=0.0,
            strategy=strategy,
            error=error or "empty_truth_data",
        )

    main_hits = 0
    special_hits = 0
    exact_draw_hits = 0
    for draw_id, truth in truth_map.items():
        pred = pred_map.get(draw_id)
        if not pred:
            continue
        truth_numbers = set(int(v) for v in truth["numbers"])
        pred_numbers = set(int(v) for v in pred["numbers"])
        hit = len(truth_numbers.intersection(pred_numbers))
        main_hits += int(hit)
        if int(pred["special_number"]) == int(truth["special_number"]):
            special_hits += 1
        if hit == 6 and int(pred["special_number"]) == int(truth["special_number"]):
            exact_draw_hits += 1

    main_hit_rate = float(main_hits) / float(draw_count * 6)
    special_hit_rate = float(special_hits) / float(draw_count)
    win_rate = float(main_hits + special_hits) / float(draw_count * 7)
    return HolonScore(
        holon_id=holon_id,
        round_index=round_index,
        main_hits=main_hits,
        special_hits=special_hits,
        exact_draw_hits=exact_draw_hits,
        draw_count=draw_count,
        main_hit_rate=main_hit_rate,
        special_hit_rate=special_hit_rate,
        win_rate=win_rate,
        strategy=strategy,
        error=error,
    )


async def _ensure_population(
    holon_service: HolonService,
    population_size: int,
    holon_prefix: str,
) -> List[str]:
    ids: List[str] = []
    for index in range(1, population_size + 1):
        holon_id = f"{holon_prefix}_{index:03d}"
        ids.append(holon_id)
        if holon_service.holon_exists(holon_id):
            continue
        blueprint = Blueprint(
            blueprint_id=f"bp_{holon_id}",
            holon_id=holon_id,
            species_id="generalist",
            name=f"Predictor {index:03d}",
            purpose="Evolve predictor skill and compete by prediction win rate.",
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
    return ids


async def _ensure_predictor_skill(
    holon_id: str,
    evo: EvolutionService,
    skill_name: str,
    code: str,
    tests: str,
    tool_schema: ToolSchema,
    force_evolve: bool,
) -> str:
    if not force_evolve:
        validation = await evo.validate_existing_skill(holon_id, skill_name)
        if bool(validation.get("valid")):
            return _slugify(skill_name)

    result = await evo.evolve_skill(
        holon_id=holon_id,
        skill_name=skill_name,
        code=code,
        tests=tests,
        description="Predict draw numbers using historical sequence data.",
        tool_schema=tool_schema,
        version="0.1.0",
    )
    if not result.success:
        raise RuntimeError(
            f"evolve_failed phase={result.phase} error={result.error_message}"
        )
    return str(result.skill_id or _slugify(skill_name))


async def _evaluate_holon(
    holon_id: str,
    round_index: int,
    holon_service: HolonService,
    evo: EvolutionService,
    skill_name: str,
    code: str,
    tests: str,
    tool_schema: ToolSchema,
    train_data: Sequence[Dict[str, Any]],
    predict_data: Sequence[Dict[str, Any]],
    force_evolve: bool,
) -> HolonScore:
    strategy = _strategy_for_holon(holon_id)
    try:
        skill_id = await _ensure_predictor_skill(
            holon_id=holon_id,
            evo=evo,
            skill_name=skill_name,
            code=code,
            tests=tests,
            tool_schema=tool_schema,
            force_evolve=force_evolve,
        )
        runtime = HolonRuntime(
            holon_id=holon_id,
            blueprint=holon_service.get_blueprint(holon_id),
        )
        execution = await runtime.execute_skill(
            skill_id,
            payload={
                "train_data": list(train_data),
                "predict_data": list(predict_data),
                "strategy": strategy,
            },
        )
        predicted = execution.get("predictions") if isinstance(execution, dict) else []
        if not isinstance(predicted, list):
            predicted = []
        return _score_predictions(
            holon_id=holon_id,
            round_index=round_index,
            truth_rows=predict_data,
            predicted_rows=predicted,
            strategy=strategy,
        )
    except Exception as exc:
        return _score_predictions(
            holon_id=holon_id,
            round_index=round_index,
            truth_rows=predict_data,
            predicted_rows=[],
            strategy=strategy,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _evaluate_population(
    holon_ids: Sequence[str],
    round_index: int,
    holon_service: HolonService,
    evo: EvolutionService,
    skill_name: str,
    code: str,
    tests: str,
    tool_schema: ToolSchema,
    train_data: Sequence[Dict[str, Any]],
    predict_data: Sequence[Dict[str, Any]],
    force_evolve: bool,
    concurrency: int,
) -> List[HolonScore]:
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(hid: str) -> HolonScore:
        async with sem:
            return await _evaluate_holon(
                holon_id=hid,
                round_index=round_index,
                holon_service=holon_service,
                evo=evo,
                skill_name=skill_name,
                code=code,
                tests=tests,
                tool_schema=tool_schema,
                train_data=train_data,
                predict_data=predict_data,
                force_evolve=force_evolve,
            )

    return await asyncio.gather(*(_run_one(hid) for hid in holon_ids))


def _sorted_scores(scores: Sequence[HolonScore]) -> List[HolonScore]:
    return sorted(
        scores,
        key=lambda item: (
            -item.win_rate,
            -item.main_hit_rate,
            -item.special_hit_rate,
            -item.exact_draw_hits,
            item.holon_id,
        ),
    )


async def run_tournament(args: argparse.Namespace) -> int:
    bootstrap()
    train_path = Path(args.train_data)
    predict_path = Path(args.predict_data)
    train_data = _read_json_array(train_path)
    predict_data = _read_json_array(predict_path)

    holon_service = HolonService()
    evo = EvolutionService()
    tool_schema = _build_tool_schema()
    code = _build_skill_code()
    tests = _build_skill_tests()

    active_holons = await _ensure_population(
        holon_service=holon_service,
        population_size=args.population_size,
        holon_prefix=args.holon_prefix,
    )
    if len(active_holons) < args.survivor_count:
        raise ValueError("population_size must be >= survivor_count")

    reports: List[RoundReport] = []
    last_scores: List[HolonScore] = []
    round_index = 1
    while len(active_holons) > args.survivor_count and round_index <= args.max_rounds:
        print(f"[round {round_index}] evaluating {len(active_holons)} holons ...")
        scores = await _evaluate_population(
            holon_ids=active_holons,
            round_index=round_index,
            holon_service=holon_service,
            evo=evo,
            skill_name=args.skill_name,
            code=code,
            tests=tests,
            tool_schema=tool_schema,
            train_data=train_data,
            predict_data=predict_data,
            force_evolve=args.force_evolve,
            concurrency=args.concurrency,
        )
        ranked = _sorted_scores(scores)
        last_scores = ranked

        keep_count = max(
            args.survivor_count,
            int(math.ceil(len(active_holons) * args.keep_ratio)),
        )
        keep_count = min(keep_count, len(active_holons))
        kept = ranked[:keep_count]
        retired = ranked[keep_count:]
        retired_ids = [item.holon_id for item in retired]
        for retired_id in retired_ids:
            try:
                holon_service.freeze_holon(retired_id)
            except Exception:
                pass

        reports.append(
            RoundReport(
                round_index=round_index,
                active_holons=len(active_holons),
                kept_holons=len(kept),
                retired_holons=len(retired),
                top=kept[: min(10, len(kept))],
                retired_ids=retired_ids,
            )
        )
        active_holons = [item.holon_id for item in kept]
        if kept:
            best_msg = f"best={kept[0].holon_id} win_rate={kept[0].win_rate:.4f}"
        else:
            best_msg = "best=n/a win_rate=0.0000"
        print(
            f"[round {round_index}] keep={len(active_holons)} retire={len(retired_ids)} "
            f"{best_msg}"
        )
        round_index += 1

    if len(active_holons) > args.survivor_count:
        ranked = _sorted_scores(last_scores or [])
        active_holons = [item.holon_id for item in ranked[: args.survivor_count]]

    final_scores = _sorted_scores(
        [item for item in (last_scores or []) if item.holon_id in set(active_holons)]
    )

    payload = {
        "config": {
            "train_data": str(train_path),
            "predict_data": str(predict_path),
            "population_size": args.population_size,
            "survivor_count": args.survivor_count,
            "max_rounds": args.max_rounds,
            "keep_ratio": args.keep_ratio,
            "holon_prefix": args.holon_prefix,
            "skill_name": args.skill_name,
            "force_evolve": args.force_evolve,
            "concurrency": args.concurrency,
        },
        "rounds": [
            {
                **asdict(report),
                "top": [asdict(score) for score in report.top],
            }
            for report in reports
        ],
        "final_top5": [asdict(score) for score in final_scores[: args.survivor_count]],
        "retained_holons": active_holons[: args.survivor_count],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("")
    print("=" * 88)
    print("Holon Prediction Tournament Complete")
    print("=" * 88)
    print(f"train_rows={len(train_data)} predict_rows={len(predict_data)}")
    print(f"retained={active_holons[: args.survivor_count]}")
    if final_scores:
        best = final_scores[0]
        print(
            f"best={best.holon_id} win_rate={best.win_rate:.4f} "
            f"main_hit_rate={best.main_hit_rate:.4f} special_hit_rate={best.special_hit_rate:.4f}"
        )
    print(f"report_json={output_path}")
    print("=" * 88)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve and run a 100-Holon prediction tournament until top-5 remain."
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
    parser.add_argument("--survivor-count", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--holon-prefix", default="holon_predictor")
    parser.add_argument("--skill-name", default=SKILL_NAME)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--force-evolve",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output-json",
        default=r"C:\Temp\holonpolis_stress\holon_prediction_tournament.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_tournament(parse_args())))
