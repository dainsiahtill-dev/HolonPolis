"""Project incubation orchestration aligned with Genesis route-or-spawn.

This module intentionally contains no business-domain project templates.
Only orchestration contracts and safety guards live here.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

from holonpolis.config import settings
from holonpolis.domain import Blueprint
from holonpolis.infrastructure.storage.io_text import write_text_atomic
from holonpolis.infrastructure.storage.path_guard import ensure_within_root, safe_join
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.kernel.storage import HolonPathGuard
from holonpolis.runtime.holon_runtime import (
    CapabilityDeniedError,
    EvolutionRequest,
    EvolutionStatus,
    HolonRuntime,
)
from holonpolis.services.genesis_service import GenesisService
from holonpolis.services.holon_service import HolonService
from holonpolis.services.reusable_project_scaffold_service import (
    ReusableProjectScaffold,
    ReusableProjectScaffoldService,
)


RuntimeFactory = Callable[[str, Optional[Blueprint]], HolonRuntime]
REQUIRED_CAPABILITIES = ("skill.execute", "evolution.request", "code.library.search")


@dataclass(frozen=True)
class ProjectIncubationSpec:
    """Inputs for incubating a Holon-driven autonomous project builder."""

    project_name: str
    project_goal: str
    holon_id: Optional[str] = None
    skill_name: Optional[str] = None
    execution_payload: Dict[str, Any] = field(default_factory=dict)
    required_files: List[str] = field(default_factory=list)
    prefer_reusable_scaffold: bool = True
    allow_evolution_fallback: bool = True
    evolution_timeout_seconds: float = 360.0
    poll_interval_seconds: float = 0.5


@dataclass
class ProjectIncubationResult:
    """Outcome of one incubation run."""

    holon_id: str
    route_decision: str
    project_name: str
    project_slug: str
    request_id: str
    evolution_status: str
    skill_name: str
    skill_id: str
    output_dir: str
    generated_file_count: int
    completed_at: str
    delivery_mode: str = "evolved_skill"
    reused_library_key: str = ""
    run_instructions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "holon_id": self.holon_id,
            "route_decision": self.route_decision,
            "project_name": self.project_name,
            "project_slug": self.project_slug,
            "request_id": self.request_id,
            "evolution_status": self.evolution_status,
            "skill_name": self.skill_name,
            "skill_id": self.skill_id,
            "output_dir": self.output_dir,
            "generated_file_count": self.generated_file_count,
            "completed_at": self.completed_at,
            "delivery_mode": self.delivery_mode,
            "reused_library_key": self.reused_library_key,
            "run_instructions": list(self.run_instructions),
        }


class ProjectIncubationService:
    """End-to-end autonomous project incubation with Genesis in front."""

    def __init__(
        self,
        holon_service: Optional[HolonService] = None,
        genesis_service: Optional[GenesisService] = None,
        runtime_factory: Optional[RuntimeFactory] = None,
        reusable_scaffold_service: Optional[ReusableProjectScaffoldService] = None,
    ):
        self.holon_service = holon_service or HolonService()
        self.genesis_service = genesis_service or GenesisService()
        self.runtime_factory: RuntimeFactory = runtime_factory or self._default_runtime_factory
        self.reusable_scaffold_service = reusable_scaffold_service or ReusableProjectScaffoldService(
            holon_service=self.holon_service
        )

    async def incubate_project(self, spec: ProjectIncubationSpec) -> ProjectIncubationResult:
        """Route/spawn via Genesis, evolve capability, execute skill, and materialize files."""
        project_name = self._normalize_project_name(spec.project_name)
        project_goal = self._normalize_project_goal(spec.project_goal)
        skill_name = self._resolve_skill_name(spec.skill_name)
        required_files = [self._normalize_generated_rel_path(path) for path in spec.required_files]

        holon_id, route_decision = await self._select_holon(
            project_name=project_name,
            project_goal=project_goal,
            holon_id_hint=spec.holon_id,
        )
        blueprint = self.holon_service.get_blueprint(holon_id)
        if self._ensure_required_capabilities(blueprint):
            self._persist_blueprint(blueprint)

        runtime = self.runtime_factory(holon_id, blueprint)
        reusable_assets = await self._search_reusable_assets(runtime, project_goal)
        project_slug = self._slugify(project_name)

        if spec.prefer_reusable_scaffold:
            scaffold = self._try_materialize_reusable_scaffold(
                holon_id=holon_id,
                project_name=project_name,
                project_goal=project_goal,
                project_slug=project_slug,
                required_files=required_files,
                reusable_assets=reusable_assets,
            )
            if scaffold is not None:
                result = self._build_reusable_scaffold_result(
                    holon_id=holon_id,
                    route_decision=route_decision,
                    project_name=project_name,
                    scaffold=scaffold,
                )
                self._write_incubation_report(scaffold.output_dir, result)
                return result
            if not spec.allow_evolution_fallback:
                raise RuntimeError(
                    "No reusable scaffold matched the project goal and evolution fallback is disabled."
                )

        requirements = self._build_requirements(project_goal, required_files)
        requirements.extend(self._build_reusable_asset_requirements(reusable_assets))
        request = await runtime.request_evolution(
            skill_name=skill_name,
            description="Autonomously generate a project scaffold from requirements.",
            requirements=requirements,
            test_cases=[],
            parent_skills=[],
        )

        status = await runtime.wait_for_evolution(
            request_id=request.request_id,
            timeout_seconds=spec.evolution_timeout_seconds,
            poll_interval_seconds=spec.poll_interval_seconds,
        )
        if status.status != EvolutionStatus.COMPLETED:
            raise RuntimeError(
                f"Evolution failed for {skill_name}: "
                f"{status.error_message or status.status.value}"
            )

        skill_id = self._extract_skill_id(status, skill_name)
        execution_payload = self._build_execution_payload(
            project_name=project_name,
            project_goal=project_goal,
            extra_payload=spec.execution_payload,
        )
        execution_result = await runtime.execute_skill(skill_id, payload=execution_payload)

        files = self._extract_files(execution_result)
        self._validate_required_files(files, required_files)
        project_slug = self._extract_project_slug(execution_result, project_name)
        output_dir = self._materialize_files(
            holon_id=holon_id,
            project_slug=project_slug,
            files=files,
        )

        result = ProjectIncubationResult(
            holon_id=holon_id,
            route_decision=route_decision,
            project_name=project_name,
            project_slug=project_slug,
            request_id=request.request_id,
            evolution_status=status.status.value,
            skill_name=skill_name,
            skill_id=skill_id,
            output_dir=str(output_dir),
            generated_file_count=len(files),
            completed_at=utc_now_iso(),
            delivery_mode="evolved_skill",
            reused_library_key="",
            run_instructions=self._extract_run_instructions(execution_result),
        )
        self._write_incubation_report(output_dir, result)
        return result

    def _default_runtime_factory(self, holon_id: str, blueprint: Optional[Blueprint]) -> HolonRuntime:
        return HolonRuntime(holon_id=holon_id, blueprint=blueprint)

    async def _select_holon(
        self,
        project_name: str,
        project_goal: str,
        holon_id_hint: Optional[str],
    ) -> tuple[str, str]:
        if holon_id_hint:
            holon_id = self._normalize_holon_id(holon_id_hint)
            if not self.holon_service.holon_exists(holon_id):
                raise ValueError(f"Holon not found: {holon_id}")
            return holon_id, "explicit"

        route = await self.genesis_service.route_or_spawn(
            user_request=self._build_genesis_request(project_name, project_goal),
            conversation_history=None,
        )
        if route.decision not in {"route_to", "spawn"} or not route.holon_id:
            raise RuntimeError(
                f"Genesis did not produce a runnable Holon for incubation: {route.decision}"
            )
        return route.holon_id, route.decision

    def _build_genesis_request(self, project_name: str, project_goal: str) -> str:
        return (
            "Need a Holon that can autonomously evolve and deliver project source code "
            "through red-green-verify. "
            f"Project name: {project_name}. "
            f"Goal: {project_goal}"
        )

    def _build_requirements(self, project_goal: str, required_files: List[str]) -> List[str]:
        is_large_multiplayer = self._is_large_multiplayer_goal(project_goal)
        requirements = [
            "Implement a public execute(...) callable as the entrypoint.",
            "execute(...) must return a dict with keys: project_name, project_slug, files, run_instructions.",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
            "Generated output must be deterministic for the same input.",
            "Do not use placeholders like TODO/TBD or empty files.",
            "README must include install, run, and usage instructions.",
            "For Node.js projects, all imported third-party runtime packages must be declared in package.json dependencies/devDependencies.",
            "If a websocket server is generated, it must expose /healthz and /ws and use PORT from environment.",
            "WebSocket server protocol must emit JSON messages with a type field (at least welcome and snapshot/state semantics).",
            "When embedding JS/JSON content in Python code generation, avoid multiline Python f-strings; use plain triple-quoted strings to prevent brace interpolation bugs.",
            f"Project goal: {project_goal}",
        ]
        if is_large_multiplayer:
            requirements.append(
                "Large multiplayer project floor: generate at least 18 files across modular directories "
                "(for example apps/server, apps/client, shared or packages/shared, configs, docs). "
                "Must include modular runtime, synchronization, and security-related source modules."
            )
            requirements.append(
                "Gameplay floor: implementation must follow project_goal-defined rules with concrete "
                "interaction logic and progression/state update logic."
            )
            requirements.append(
                "Reject placeholder content in generated files (for example: TODO, TBD, placeholder, "
                "render logic, implement this, coming soon, mock data)."
            )
        if required_files:
            requirements.append(
                "Required output file paths: " + ", ".join(required_files)
            )
        return requirements

    async def _search_reusable_assets(
        self,
        runtime: Any,
        project_goal: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve reusable code assets once so all downstream paths use the same context."""
        search_fn = getattr(runtime, "search_reusable_code_library", None)
        if not callable(search_fn):
            return []

        try:
            assets = await search_fn(query=project_goal, top_k=3)
        except CapabilityDeniedError:
            return []
        except Exception:
            return []

        if not isinstance(assets, list) or not assets:
            return []
        return [item for item in assets if isinstance(item, dict)]

    def _build_reusable_asset_requirements(
        self,
        assets: List[Dict[str, Any]],
    ) -> List[str]:
        """Inject retrieved reusable code assets into evolution requirements when available."""
        if not isinstance(assets, list) or not assets:
            return []
        lines = [
            "Reference reusable code assets when they fit the project goal. Reuse compatible patterns and interfaces instead of reinventing them.",
        ]
        for item in assets[:3]:
            asset_name = str(item.get("asset_name") or item.get("component_name") or "unknown_asset").strip()
            library_name = str(item.get("library_name") or "").strip()
            relative_path = str(item.get("relative_path") or "").strip()
            usage_example = str(item.get("usage_example") or "").strip()
            code_content = str(item.get("code_content") or "").strip()

            summary = f"Reusable asset: {asset_name}"
            if library_name:
                summary += f" from {library_name}"
            if relative_path:
                summary += f" ({relative_path})"
            lines.append(summary)

            if usage_example:
                lines.append(f"Suggested usage: {usage_example}")
            if code_content:
                lines.append("Reference snippet:")
                lines.append(self._truncate_requirement_text(code_content, max_chars=700))

        return lines

    def _try_materialize_reusable_scaffold(
        self,
        *,
        holon_id: str,
        project_name: str,
        project_goal: str,
        project_slug: str,
        required_files: List[str],
        reusable_assets: List[Dict[str, Any]],
    ) -> Optional[ReusableProjectScaffold]:
        return self.reusable_scaffold_service.try_materialize(
            holon_id=holon_id,
            project_name=project_name,
            project_slug=project_slug,
            project_goal=project_goal,
            candidate_assets=reusable_assets,
            required_files=required_files,
        )

    def _build_reusable_scaffold_result(
        self,
        *,
        holon_id: str,
        route_decision: str,
        project_name: str,
        scaffold: ReusableProjectScaffold,
    ) -> ProjectIncubationResult:
        return ProjectIncubationResult(
            holon_id=holon_id,
            route_decision=route_decision,
            project_name=project_name,
            project_slug=scaffold.project_slug,
            request_id=f"reusable_scaffold_{uuid.uuid4().hex[:12]}",
            evolution_status="reused_scaffold",
            skill_name="Reusable Scaffold",
            skill_id=f"reusable_scaffold::{scaffold.library_key}",
            output_dir=str(scaffold.output_dir),
            generated_file_count=len(scaffold.copied_files),
            completed_at=utc_now_iso(),
            delivery_mode="reusable_scaffold",
            reused_library_key=scaffold.library_key,
            run_instructions=list(scaffold.run_instructions),
        )

    def _build_execution_payload(
        self,
        project_name: str,
        project_goal: str,
        extra_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "project_name": project_name,
            "project_goal": project_goal,
            # Common aliases to tolerate evolved skill signatures.
            "project_description": project_goal,
            "description": project_goal,
        }
        if isinstance(extra_payload, dict):
            payload.update(extra_payload)
        return payload

    def _ensure_required_capabilities(self, blueprint: Blueprint) -> bool:
        changed = False

        allowed = [str(item).strip() for item in (blueprint.boundary.allowed_tools or []) if str(item).strip()]
        denied = [str(item).strip() for item in (blueprint.boundary.denied_tools or []) if str(item).strip()]

        for capability in REQUIRED_CAPABILITIES:
            if capability not in allowed:
                allowed.append(capability)
                changed = True
            if capability in denied:
                denied = [item for item in denied if item != capability]
                changed = True

        if changed:
            blueprint.boundary.allowed_tools = sorted(set(allowed))
            blueprint.boundary.denied_tools = sorted(set(denied))
        return changed

    def _persist_blueprint(self, blueprint: Blueprint) -> None:
        holon_dir = safe_join(settings.holons_path, blueprint.holon_id)
        blueprint_path = safe_join(holon_dir, "blueprint.json")
        write_text_atomic(
            str(blueprint_path),
            json.dumps(blueprint.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    def _extract_skill_id(self, status: EvolutionRequest, skill_name: str) -> str:
        if isinstance(status.result, dict):
            skill_id = status.result.get("skill_id")
            if isinstance(skill_id, str) and skill_id.strip():
                return skill_id.strip()
        return self._slugify(skill_name)

    def _extract_files(self, execution_result: Any) -> Dict[str, str]:
        if not isinstance(execution_result, dict):
            raise ValueError("Skill execution result must be a dict")
        files = execution_result.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Skill execution result missing non-empty files dict")

        normalized_files: Dict[str, str] = {}
        for raw_path, raw_content in files.items():
            safe_path = self._normalize_generated_rel_path(raw_path)
            if not isinstance(raw_content, str):
                raise ValueError(f"File content must be text for {safe_path}")
            if not raw_content.strip():
                raise ValueError(f"Generated file content is empty: {safe_path}")
            normalized_files[safe_path] = raw_content
        return normalized_files

    def _validate_required_files(self, files: Dict[str, str], required_files: List[str]) -> None:
        if not required_files:
            return
        missing = [path for path in required_files if path not in files]
        if missing:
            raise ValueError(
                "Generated project missing required files: " + ", ".join(missing)
            )

    def _extract_project_slug(self, execution_result: Dict[str, Any], project_name: str) -> str:
        candidate = execution_result.get("project_slug")
        if isinstance(candidate, str) and candidate.strip():
            return self._slugify(candidate)
        return self._slugify(project_name)

    def _extract_run_instructions(self, execution_result: Dict[str, Any]) -> List[str]:
        raw = execution_result.get("run_instructions")
        if not isinstance(raw, list):
            return []
        instructions: List[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text:
                instructions.append(text)
        return instructions

    def _materialize_files(self, holon_id: str, project_slug: str, files: Dict[str, str]) -> Path:
        guard = HolonPathGuard(holon_id)
        run_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        output_dir = guard.ensure_directory(f"workspace/incubations/{project_slug}_{run_suffix}")

        for rel_path, content in files.items():
            target = output_dir / rel_path
            ensure_within_root(output_dir, target)
            write_text_atomic(str(target), content)
        return output_dir

    def _write_incubation_report(self, output_dir: Path, result: ProjectIncubationResult) -> None:
        report_path = output_dir / "_incubation_report.json"
        ensure_within_root(output_dir, report_path)
        write_text_atomic(
            str(report_path),
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )

    def _normalize_generated_rel_path(self, rel_path: Any) -> str:
        raw = str(rel_path or "").strip().replace("\\", "/")
        if not raw:
            raise ValueError("Generated file path is empty")
        if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
            raise ValueError(f"Generated file path must be relative: {raw}")

        normalized = PurePosixPath(raw)
        if normalized.is_absolute() or any(part in ("..", ".") for part in normalized.parts):
            raise ValueError(f"Generated file path escapes sandbox: {raw}")

        safe = str(normalized).strip("/")
        if not safe:
            raise ValueError(f"Invalid generated file path: {raw}")
        return safe

    def _resolve_skill_name(self, skill_name: Optional[str]) -> str:
        value = (skill_name or "").strip()
        if value:
            return value
        return "Autonomous Project Builder"

    def _normalize_project_name(self, project_name: str) -> str:
        value = str(project_name or "").strip()
        if len(value) < 3:
            raise ValueError("project_name must have at least 3 characters")
        return value

    def _normalize_project_goal(self, project_goal: str) -> str:
        value = str(project_goal or "").strip()
        if len(value) < 10:
            raise ValueError("project_goal must have at least 10 characters")
        return value

    def _normalize_holon_id(self, holon_id: Optional[str]) -> str:
        value = str(holon_id or "").strip()
        if not value:
            raise ValueError("holon_id cannot be empty")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,127}", value):
            raise ValueError(f"Invalid holon_id: {holon_id!r}")
        return value

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower()).strip("_")
        return slug or "project"

    def _is_large_multiplayer_goal(self, project_goal: str) -> bool:
        lowered = str(project_goal or "").lower()
        keywords = (
            "mmo",
            "multiplayer",
            "websocket",
            "real-time",
            "large",
            "large-scale",
            "多人在线",
            "大型",
        )
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _truncate_requirement_text(content: str, max_chars: int = 700) -> str:
        text = str(content or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."
