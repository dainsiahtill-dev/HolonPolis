#!/usr/bin/env python
"""Incubate a Holon and autonomously evolve a runnable project skill."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from holonpolis.services.project_incubation_service import (  # noqa: E402
    ProjectIncubationSpec,
    ProjectIncubationService,
)


async def _run(args: argparse.Namespace) -> int:
    service = ProjectIncubationService()
    payload = {}
    if args.payload_json:
        payload = json.loads(Path(args.payload_json).read_text(encoding="utf-8"))

    spec = ProjectIncubationSpec(
        project_name=args.project_name,
        project_goal=args.project_goal,
        holon_id=args.holon_id,
        skill_name=args.skill_name,
        execution_payload=payload,
        required_files=args.required_file or [],
        evolution_timeout_seconds=args.timeout,
        poll_interval_seconds=args.poll_interval,
    )

    result = await service.incubate_project(spec)
    payload = result.to_dict()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genesis-routed autonomous project incubation."
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Name used inside generated project metadata.",
    )
    parser.add_argument(
        "--project-goal",
        required=True,
        help="Plain-text project requirement. Business logic comes only from this goal.",
    )
    parser.add_argument(
        "--holon-id",
        default=None,
        help="Optional explicit Holon ID. If omitted, Genesis route_or_spawn decides.",
    )
    parser.add_argument(
        "--skill-name",
        default=None,
        help="Optional evolved skill name override.",
    )
    parser.add_argument(
        "--payload-json",
        default=None,
        help="Optional JSON file path used as execute(...) payload extension.",
    )
    parser.add_argument(
        "--required-file",
        action="append",
        default=[],
        help="Optional required output file path (repeatable).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=360.0,
        help="Evolution wait timeout in seconds.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to persist run result JSON.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
