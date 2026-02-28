"""Evolution Service - RGV (Red-Green-Verify) core.

This module keeps the evolution crucible minimal and framework-agnostic:
1) Red   – validate test definitions (syntax only)
2) Green – run tests in a sandbox
3) Verify – AST security scan
4) Persist – write skill + manifest + attestation

All higher-level capabilities (React, Vue, etc.) must be evolved by Holons
through this generic pipeline rather than hardcoded templates.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.config import settings
from holonpolis.domain.skills import SkillManifest, SkillVersion, ToolSchema
from holonpolis.kernel.llm.llm_runtime import LLMConfig, get_llm_runtime
from holonpolis.kernel.sandbox import SandboxConfig, SandboxRunner
from holonpolis.kernel.storage import HolonPathGuard

logger = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class Attestation:
    """Evidence that a skill passed the RGV cycle."""

    attestation_id: str
    holon_id: str
    skill_name: str
    version: str

    red_phase_passed: bool
    green_phase_passed: bool
    verify_phase_passed: bool

    test_results: Dict[str, Any] = field(default_factory=dict)
    security_scan_results: Dict[str, Any] = field(default_factory=dict)
    code_hash: str = ""
    test_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attestation_id": self.attestation_id,
            "holon_id": self.holon_id,
            "skill_name": self.skill_name,
            "version": self.version,
            "red_phase_passed": self.red_phase_passed,
            "green_phase_passed": self.green_phase_passed,
            "verify_phase_passed": self.verify_phase_passed,
            "test_results": self.test_results,
            "security_scan_results": self.security_scan_results,
            "code_hash": self.code_hash,
            "test_hash": self.test_hash,
            "created_at": self.created_at,
        }


@dataclass
class EvolutionResult:
    """Outcome of an evolution run."""

    success: bool
    skill_id: Optional[str] = None
    attestation: Optional[Attestation] = None
    error_message: Optional[str] = None
    phase: str = ""  # red | green | verify | persist | complete
    code_path: Optional[str] = None
    test_path: Optional[str] = None
    manifest_path: Optional[str] = None


# --------------------------------------------------------------------------- #
# Security scanning
# --------------------------------------------------------------------------- #


class SecurityScanner:
    """Simple AST-based security scanner for Python code."""

    DANGEROUS_IMPORTS = frozenset(
        {
            "os.system",
            "subprocess.call",
            "subprocess.run",
            "subprocess.Popen",
            "eval",
            "exec",
            "compile",
            "__import__",
            "pickle.loads",
            "pickle.load",
            "yaml.load",
            "yaml.unsafe_load",
        }
    )

    DANGEROUS_CALLS = frozenset(
        {
            "eval",
            "exec",
            "compile",
            "getattr",
            "setattr",
            "delattr",
            "globals",
            "locals",
            "vars",
        }
    )

    SENSITIVE_ATTRS = frozenset(
        {
            "__code__",
            "__globals__",
            "__closure__",
            "__subclasses__",
            "__bases__",
            "__mro__",
        }
    )

    def scan(self, code: str, filename: str = "<unknown>") -> Dict[str, Any]:
        """Return security report."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {
                "passed": False,
                "violations": [{"type": "syntax_error", "message": str(exc)}],
                "complexity_score": 0.0,
            }

        violations: List[Dict[str, Any]] = []

        for node in ast.walk(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.DANGEROUS_IMPORTS:
                        violations.append(
                            {
                                "type": "dangerous_import",
                                "line": getattr(node, "lineno", 0),
                                "name": alias.name,
                                "message": f"Dangerous import: {alias.name}",
                            }
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}"
                    if full_name in self.DANGEROUS_IMPORTS:
                        violations.append(
                            {
                                "type": "dangerous_import",
                                "line": getattr(node, "lineno", 0),
                                "name": full_name,
                                "message": f"Dangerous import: {full_name}",
                            }
                        )

            # Calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.DANGEROUS_CALLS:
                    violations.append(
                        {
                            "type": "dangerous_call",
                            "line": getattr(node, "lineno", 0),
                            "name": node.func.id,
                            "message": f"Dangerous function call: {node.func.id}",
                        }
                    )
                elif isinstance(node.func, ast.Attribute):
                    dotted = self._resolve_attr(node.func)
                    if dotted in self.DANGEROUS_IMPORTS:
                        violations.append(
                            {
                                "type": "dangerous_call",
                                "line": getattr(node, "lineno", 0),
                                "name": dotted,
                                "message": f"Dangerous function call: {dotted}",
                            }
                        )

            # Sensitive attributes
            elif isinstance(node, ast.Attribute) and node.attr in self.SENSITIVE_ATTRS:
                violations.append(
                    {
                        "type": "sensitive_attribute",
                        "line": getattr(node, "lineno", 0),
                        "name": node.attr,
                        "message": f"Sensitive attribute access: {node.attr}",
                    }
                )

        complexity = self._estimate_complexity(tree)
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "complexity_score": complexity,
        }

    def _resolve_attr(self, node: ast.Attribute) -> str:
        parts = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _estimate_complexity(self, tree: ast.AST) -> float:
        decision_points = 0
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.With,
                    ast.Try,
                    ast.ExceptHandler,
                    ast.comprehension,
                ),
            ):
                decision_points += 1
            elif isinstance(node, ast.BoolOp):
                decision_points += max(0, len(node.values) - 1)
        return min(10.0, decision_points / 5.0)


# --------------------------------------------------------------------------- #
# LLM-based generators (framework-agnostic)
# --------------------------------------------------------------------------- #


class LLMCodeGenerator:
    """Lightweight, framework-agnostic code generator.

    This does not embed any business templates; it simply asks an LLM to
    generate code for the given description/requirements.
    """

    def __init__(self, provider_id: Optional[str] = None, model: Optional[str] = None):
        self.runtime = get_llm_runtime()
        self.provider_id = provider_id or settings.llm_provider
        self.model = model

    async def _generate_code(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
    ) -> str:
        req_text = "\n".join(f"- {r}" for r in requirements)
        system_prompt = (
            "You are a pragmatic senior engineer. "
            "Generate concise, production-quality code that satisfies the requirements. "
            "Return ONLY the code without explanations."
        )
        user_prompt = f"""Project: {project_name}
Description: {description}
Requirements:
{req_text}

Guidelines:
- Choose the most natural language and minimal dependencies.
- Keep the code self-contained.
- Include necessary types/defs.
- No placeholders like TODO.
"""
        config = LLMConfig(
            provider_id=self.provider_id,
            model=self.model,
            temperature=0.3,
            max_tokens=8000,
        )
        response = await self.runtime.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        return self._strip_code_fences(response.content)

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()


# --------------------------------------------------------------------------- #
# Evolution core
# --------------------------------------------------------------------------- #


class EvolutionService:
    """Framework-agnostic RGV executor."""

    def __init__(self):
        self.security_scanner = SecurityScanner()
        self._sandbox: Optional[SandboxRunner] = None
        self._llm = get_llm_runtime()

    @property
    def sandbox(self) -> SandboxRunner:
        """Lazily initialize sandbox so tests can monkeypatch settings first."""
        if self._sandbox is None:
            self._sandbox = SandboxRunner()
        return self._sandbox

    # Public API ------------------------------------------------------------- #

    async def evolve_skill(
        self,
        holon_id: str,
        skill_name: str,
        code: str,
        tests: str,
        description: str,
        tool_schema: ToolSchema,
        version: str = "0.1.0",
    ) -> EvolutionResult:
        """Run full RGV and persist."""
        red_result = await self._phase_red(tests)
        if not red_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="red",
                error_message=red_result["error"],
            )

        green_result = await self._phase_green(code, tests, holon_id)
        if not green_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="green",
                error_message=green_result.get("error"),
            )

        verify_result = self._phase_verify(code)
        if not verify_result["passed"]:
            violation_msg = "; ".join(v.get("message", "") for v in verify_result.get("violations", []))
            return EvolutionResult(
                success=False,
                phase="verify",
                error_message=violation_msg or "Security scan failed",
            )

        persist_result = await self._phase_persist(
            holon_id=holon_id,
            skill_name=skill_name,
            code=code,
            tests=tests,
            description=description,
            tool_schema=tool_schema,
            version=version,
            green_result=green_result,
            verify_result=verify_result,
        )

        if not persist_result["success"]:
            return EvolutionResult(
                success=False,
                phase="persist",
                error_message=persist_result.get("error"),
            )

        return EvolutionResult(
            success=True,
            phase="complete",
            skill_id=persist_result["skill_id"],
            attestation=persist_result["attestation"],
            code_path=persist_result["code_path"],
            test_path=persist_result["test_path"],
            manifest_path=persist_result["manifest_path"],
        )

    async def evolve_skill_autonomous(
        self,
        holon_id: str,
        skill_name: str,
        description: str,
        requirements: List[str],
        tool_schema: ToolSchema,
        version: str = "0.1.0",
    ) -> EvolutionResult:
        """End-to-end LLM-driven evolution (no hardcoded templates)."""
        tests = await self._generate_tests_via_llm(skill_name, description, requirements)
        code = await self._generate_code_via_llm(skill_name, description, requirements, tests)
        return await self.evolve_skill(
            holon_id=holon_id,
            skill_name=skill_name,
            code=code,
            tests=tests,
            description=description,
            tool_schema=tool_schema,
            version=version,
        )

    async def evolve_react_project_auto(self, *args, **kwargs) -> EvolutionResult:  # Compatibility stub
        """Previously hardcoded React generator has been removed."""
        return EvolutionResult(
            success=False,
            phase="generate",
            error_message="React auto-generator removed. Evolve a React generator skill via evolve_skill_autonomous().",
        )

    async def evolve_typescript_project_auto(self, *args, **kwargs) -> EvolutionResult:  # Compatibility stub
        """Previously hardcoded TS generator has been removed."""
        return EvolutionResult(
            success=False,
            phase="generate",
            error_message="TypeScript auto-generator removed. Use evolve_skill_autonomous().",
        )

    async def validate_existing_skill(
        self,
        holon_id: str,
        skill_name: str,
    ) -> Dict[str, Any]:
        """Re-run Green + Verify for a persisted skill."""
        guard = HolonPathGuard(holon_id)
        slug = self._slugify(skill_name)
        skill_dir = guard.resolve(f"skills_local/{slug}", must_exist=False)
        legacy_skill_dir = guard.resolve(f"skills/{slug}", must_exist=False)
        if not skill_dir.exists() and legacy_skill_dir.exists():
            skill_dir = legacy_skill_dir
        if not skill_dir.exists():
            return {"valid": False, "error": f"Skill '{skill_name}' not found"}

        manifest_path = skill_dir / "manifest.json"
        if manifest_path.exists():
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = SkillManifest.from_dict(manifest_data)
            version_entry = manifest.versions[0] if manifest.versions else None
            code_path = self._resolve_skill_path(
                guard=guard,
                skill_dir=skill_dir,
                candidate=getattr(version_entry, "code_path", None),
                fallback_name="skill.py",
            )
            test_path = self._resolve_skill_path(
                guard=guard,
                skill_dir=skill_dir,
                candidate=getattr(version_entry, "test_path", None),
                fallback_name="tests.py",
            )
        else:
            code_path = skill_dir / "skill.py"
            test_path = skill_dir / "tests.py"

        if not code_path.exists() or not test_path.exists():
            return {"valid": False, "error": "Skill files not found"}

        code = code_path.read_text(encoding="utf-8")
        tests = test_path.read_text(encoding="utf-8")

        green = await self._phase_green(code, tests, holon_id)
        verify = self._phase_verify(code)

        return {
            "valid": green["passed"] and verify["passed"],
            "green_passed": green["passed"],
            "verify_passed": verify["passed"],
            "violations": verify.get("violations", []),
        }

    # Phase implementations -------------------------------------------------- #

    async def _phase_red(self, tests: str) -> Dict[str, Any]:
        """Validate test syntax (no execution)."""
        try:
            ast.parse(tests)
            return {"passed": True, "error": None}
        except SyntaxError as exc:
            return {"passed": False, "error": f"syntax error: {exc}"}

    async def _phase_green(self, code: str, tests: str, holon_id: str) -> Dict[str, Any]:
        """Run pytest in sandbox."""
        guard = HolonPathGuard(holon_id)
        temp_root = guard.ensure_directory("temp")
        work_dir = Path(tempfile.mkdtemp(prefix="rgv_", dir=temp_root))

        code_file = work_dir / "skill_module.py"
        test_file = work_dir / "test_skill.py"
        code_file.write_text(code, encoding="utf-8")
        test_file.write_text(tests, encoding="utf-8")

        config = SandboxConfig(
            timeout_seconds=settings.evolution_pytest_timeout,
            working_dir=work_dir,
            inherit_env=False,
            enable_network=False,
            strict_exit_code=False,  # pytest will signal failures via exit code
            env_vars={
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONUTF8": "1",
                "HOME": str(work_dir),
                "USERPROFILE": str(work_dir),
            },
        )

        result = await self.sandbox.run_pytest(test_file, config=config)
        passed = result.success and result.exit_code == 0

        details = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
        }

        return {
            "passed": passed,
            "details": details,
            "error": None if passed else "pytest failed",
        }

    def _phase_verify(self, code: str) -> Dict[str, Any]:
        """Static security scan."""
        return self.security_scanner.scan(code)

    async def _phase_persist(
        self,
        holon_id: str,
        skill_name: str,
        code: str,
        tests: str,
        description: str,
        tool_schema: ToolSchema,
        version: str,
        green_result: Dict[str, Any],
        verify_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Persist code, tests, manifest, and attestation."""
        guard = HolonPathGuard(holon_id)
        skills_root = guard.ensure_directory("skills_local")
        slug = self._slugify(skill_name)
        skill_dir = skills_root / slug
        skill_dir.mkdir(parents=True, exist_ok=True)

        code_path = skill_dir / "skill.py"
        test_path = skill_dir / "tests.py"
        manifest_path = skill_dir / "manifest.json"
        attestation_path = skill_dir / "attestation.json"

        code_path.write_text(code, encoding="utf-8")
        test_path.write_text(tests, encoding="utf-8")

        code_hash = self._hash_text(code)
        test_hash = self._hash_text(tests)

        attestation = Attestation(
            attestation_id=f"att_{slug}_{code_hash[:8]}",
            holon_id=holon_id,
            skill_name=skill_name,
            version=version,
            red_phase_passed=True,
            green_phase_passed=green_result.get("passed", False),
            verify_phase_passed=verify_result.get("passed", False),
            test_results=green_result,
            security_scan_results=verify_result,
            code_hash=code_hash,
            test_hash=test_hash,
        )

        # Manifest
        version_entry = SkillVersion(
            version=version,
            created_by=holon_id,
            code_path=str(code_path.relative_to(guard.holon_base)),
            test_path=str(test_path.relative_to(guard.holon_base)),
            attestation_id=attestation.attestation_id,
            test_results=green_result,
            static_scan_passed=verify_result.get("passed", False),
        )

        manifest = SkillManifest(
            skill_id=slug,
            name=skill_name,
            description=description,
            version=version,
            tool_schema=tool_schema,
            author_holon=holon_id,
            versions=[version_entry],
        )

        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        attestation_path.write_text(
            json.dumps(attestation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "success": True,
            "skill_id": slug,
            "attestation": attestation,
            "code_path": str(code_path),
            "test_path": str(test_path),
            "manifest_path": str(manifest_path),
        }

    # Helpers ---------------------------------------------------------------- #

    async def _generate_tests_via_llm(
        self,
        skill_name: str,
        description: str,
        requirements: List[str],
    ) -> str:
        req_text = "\n".join(f"- {r}" for r in requirements)
        system_prompt = (
            "You are a careful QA engineer. "
            "Write exhaustive pytest cases for the described skill. "
            "Return only runnable test code."
        )
        user_prompt = f"""Skill: {skill_name}
Description: {description}
Requirements:
{req_text}

Rules:
- Use pytest only.
- Use deterministic assertions.
- No placeholders or TODOs.
"""
        config = LLMConfig(
            provider_id=settings.llm_provider,
            model=None,
            temperature=0.2,
            max_tokens=4000,
        )
        response = await self._llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        return self._strip_code_fences(response.content)

    async def _generate_code_via_llm(
        self,
        skill_name: str,
        description: str,
        requirements: List[str],
        tests: str,
    ) -> str:
        req_text = "\n".join(f"- {r}" for r in requirements)
        system_prompt = (
            "You are a senior Python engineer. "
            "Produce minimal, clean code that will pass the provided pytest suite. "
            "Return only the code."
        )
        user_prompt = f"""Skill: {skill_name}
Description: {description}
Requirements:
{req_text}

Tests that must pass:
{tests}
"""
        config = LLMConfig(
            provider_id=settings.llm_provider,
            temperature=0.3,
            max_tokens=8000,
        )
        response = await self._llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        return self._strip_code_fences(response.content)

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return slug or "skill"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_skill_path(
        guard: HolonPathGuard,
        skill_dir: Path,
        candidate: Optional[str],
        fallback_name: str,
    ) -> Path:
        """Resolve stored code/test path that might be relative or absolute."""
        if candidate:
            candidate_path = Path(candidate)
            if candidate_path.is_absolute():
                return candidate_path
            try:
                return guard.resolve(candidate_path)
            except Exception:
                pass
        return skill_dir / fallback_name


def create_evolution_service() -> EvolutionService:
    """Factory helper for DI usage."""
    return EvolutionService()
