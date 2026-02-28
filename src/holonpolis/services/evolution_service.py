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
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.config import settings
from holonpolis.domain.skills import SkillManifest, SkillVersion, ToolSchema
from holonpolis.infrastructure.time_utils import utc_now_iso
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
    created_at: str = field(default_factory=utc_now_iso)

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

        red_contract_result = await self._phase_red_contract(tests, holon_id)
        if not red_contract_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="red",
                error_message=red_contract_result["error"],
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
        max_attempts: Optional[int] = None,
    ) -> EvolutionResult:
        """End-to-end LLM-driven evolution (no hardcoded templates).

        Runs autonomous RGV with repair retries so Holons can iterate without
        hand-written scaffolding templates.
        """
        configured_attempts = max_attempts if max_attempts is not None else settings.evolution_max_attempts
        attempts = max(1, int(configured_attempts))
        project_mode = self._detect_project_contract_mode(requirements)
        if project_mode:
            # Project incubation is a high-variance generation task; allow more repair rounds by default.
            attempts = max(attempts, 6)
            tests = self._build_project_contract_tests(requirements)
        else:
            tests = await self._generate_tests_via_llm(skill_name, description, requirements)
            tests_validation: Dict[str, Any] = {"passed": False, "error": "tests validation not run"}
            for _test_try in range(1, attempts + 1):
                tests_validation = self._validate_generated_tests_contract(tests, requirements=requirements)
                if tests_validation["passed"]:
                    break
                tests = await self._repair_tests_via_llm(
                    skill_name=skill_name,
                    description=description,
                    requirements=requirements,
                    previous_tests=tests,
                    failure_message=tests_validation["error"] or "invalid test contract",
                )

            if not tests_validation["passed"]:
                return EvolutionResult(
                    success=False,
                    phase="red",
                    error_message=tests_validation["error"] or "unable to produce valid tests",
                )

        code = await self._generate_code_via_llm(skill_name, description, requirements, tests)
        result = await self.evolve_skill(
            holon_id=holon_id,
            skill_name=skill_name,
            code=code,
            tests=tests,
            description=description,
            tool_schema=tool_schema,
            version=version,
        )
        if result.success or attempts == 1:
            return result

        previous_code = code
        previous_tests = tests
        for _attempt in range(2, attempts + 1):
            if result.phase == "red":
                previous_tests = await self._repair_tests_via_llm(
                    skill_name=skill_name,
                    description=description,
                    requirements=requirements,
                    previous_tests=previous_tests,
                    failure_message=result.error_message or "red phase failed",
                )
                validation = self._validate_generated_tests_contract(
                    previous_tests,
                    requirements=requirements,
                )
                if not validation["passed"]:
                    continue
                previous_code = await self._generate_code_via_llm(
                    skill_name=skill_name,
                    description=description,
                    requirements=requirements,
                    tests=previous_tests,
                )
            else:
                previous_code = await self._repair_code_via_llm(
                    skill_name=skill_name,
                    description=description,
                    requirements=requirements,
                    tests=previous_tests,
                    previous_code=previous_code,
                    failure_phase=result.phase,
                    failure_message=result.error_message or "",
                )

            result = await self.evolve_skill(
                holon_id=holon_id,
                skill_name=skill_name,
                code=previous_code,
                tests=previous_tests,
                description=description,
                tool_schema=tool_schema,
                version=version,
            )
            if result.success:
                return result
        return result

    async def evolve_skill_with_test_cases(
        self,
        holon_id: str,
        skill_name: str,
        description: str,
        requirements: List[str],
        test_cases: List[Dict[str, Any]],
        tool_schema: ToolSchema,
        version: str = "0.1.0",
    ) -> EvolutionResult:
        """Evolve a skill from explicit test cases supplied by caller."""
        if not test_cases:
            return await self.evolve_skill_autonomous(
                holon_id=holon_id,
                skill_name=skill_name,
                description=description,
                requirements=requirements,
                tool_schema=tool_schema,
                version=version,
            )

        tests = self._render_pytest_from_test_cases(test_cases)
        code = await self._generate_code_via_llm(
            skill_name=skill_name,
            description=description,
            requirements=requirements,
            tests=tests,
        )
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

    async def _phase_red_contract(self, tests: str, holon_id: str) -> Dict[str, Any]:
        """Ensure tests encode behavior by failing against a baseline stub."""
        baseline_code = (
            "def __getattr__(name):\n"
            "    def _missing(*args, **kwargs):\n"
            "        raise NotImplementedError(f\"{name} not implemented\")\n"
            "    return _missing\n"
        )
        baseline_result = await self._phase_green(
            code=baseline_code,
            tests=tests,
            holon_id=holon_id,
        )
        if baseline_result["passed"]:
            return {
                "passed": False,
                "error": (
                    "tests must fail against baseline implementation; "
                    "current tests are too weak to drive Red phase"
                ),
                "details": baseline_result.get("details"),
            }
        return {"passed": True, "error": None}

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
            "error": None if passed else self._format_pytest_failure(details),
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
- Tests MUST import from skill_module (for example: from skill_module import execute).
- Do NOT define execute()/main business logic in test file.
- Do NOT mock or monkeypatch skill_module implementation.
- Ensure at least one assertion validates concrete business output, not only function existence.
- For project scaffolding skills, validate structure and invariants, not exact full-template file contents.
- Never require absolute output file paths unless requirements explicitly demand it.
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
        project_rules = self._project_generation_rules(requirements)
        project_rule_text = ""
        if project_rules:
            project_rule_text = "Project contract focus:\n" + "\n".join(
                f"- {rule}" for rule in project_rules
            )
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

{project_rule_text}

Implementation rules:
- Return valid Python syntax only.
- Do not place raw JavaScript/JSON braces inside Python f-strings unless every literal brace is escaped as {{ and }}.
- Prefer plain triple-quoted strings (without f-prefix) for JS/JSON snippets unless interpolation is strictly required.
- Keep output deterministic and fully runnable.
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

    async def _repair_code_via_llm(
        self,
        skill_name: str,
        description: str,
        requirements: List[str],
        tests: str,
        previous_code: str,
        failure_phase: str,
        failure_message: str,
    ) -> str:
        """Repair generated code from RGV feedback without introducing templates."""
        req_text = "\n".join(f"- {r}" for r in requirements)
        repair_hints = self._derive_failure_repair_hints(
            failure_message=failure_message,
            requirements=requirements,
        )
        repair_hint_text = "\n".join(f"- {hint}" for hint in repair_hints)
        system_prompt = (
            "You are a senior Python engineer. "
            "You will repair code to pass pytest and security verification. "
            "Return only the full corrected code."
        )
        user_prompt = f"""Skill: {skill_name}
Description: {description}
Requirements:
{req_text}

RGV failure:
- phase: {failure_phase}
- message: {failure_message}

Previous code:
{previous_code}

Tests that must pass:
{tests}

Targeted fixes inferred from failing pytest output:
{repair_hint_text}

Repair rules:
- Return syntactically valid Python only.
- If previous code used f-strings containing JS/JSON braces and caused syntax errors, remove f-strings or escape braces.
- Do not leave placeholders; preserve deterministic behavior.
"""
        config = LLMConfig(
            provider_id=settings.llm_provider,
            temperature=0.1,
            max_tokens=8000,
        )
        response = await self._llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        return self._strip_code_fences(response.content)

    async def _repair_tests_via_llm(
        self,
        skill_name: str,
        description: str,
        requirements: List[str],
        previous_tests: str,
        failure_message: str,
    ) -> str:
        """Repair test suite contract when Red phase validation fails."""
        req_text = "\n".join(f"- {r}" for r in requirements)
        system_prompt = (
            "You are a senior QA engineer. "
            "Repair pytest tests so they validate behavior of skill_module implementation. "
            "Return only the full corrected pytest code."
        )
        user_prompt = f"""Skill: {skill_name}
Description: {description}
Requirements:
{req_text}

Test contract failure:
{failure_message}

Previous tests:
{previous_tests}

Mandatory rules:
- Must import from skill_module.
- Must not define execute() or business logic in tests.
- Must assert concrete outputs for core requirements.
- Do not overfit to a single hardcoded template implementation.
- Do not require absolute paths unless explicitly required.
"""
        config = LLMConfig(
            provider_id=settings.llm_provider,
            temperature=0.1,
            max_tokens=6000,
        )
        response = await self._llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
        )
        return self._strip_code_fences(response.content)

    @staticmethod
    def _validate_generated_tests_contract(
        tests: str,
        requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate generated tests actually exercise skill_module behavior."""
        try:
            tree = ast.parse(tests)
        except SyntaxError as exc:
            return {"passed": False, "error": f"tests syntax error: {exc}"}

        imports_skill_module = False
        has_assertion = False
        forbidden_defs: List[str] = []
        forbidden_monkeypatch = False
        execute_calls: List[ast.Call] = []
        project_mode = EvolutionService._detect_project_contract_mode(requirements or [])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "skill_module":
                        imports_skill_module = True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "skill_module":
                    imports_skill_module = True
            elif isinstance(node, ast.Assert):
                has_assertion = True
            elif isinstance(node, ast.FunctionDef) and node.name == "execute":
                forbidden_defs.append("execute")
            elif isinstance(node, ast.AsyncFunctionDef) and node.name == "execute":
                forbidden_defs.append("execute")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "skill_module":
                        forbidden_defs.append("skill_module_reassigned")
            elif isinstance(node, ast.Attribute):
                if node.attr in {"monkeypatch", "patch"}:
                    forbidden_monkeypatch = True
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"MagicMock", "Mock"}:
                    forbidden_monkeypatch = True
                if isinstance(node.func, ast.Name) and node.func.id == "execute":
                    execute_calls.append(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                    execute_calls.append(node)

        if not imports_skill_module:
            return {"passed": False, "error": "tests must import from skill_module"}
        if not has_assertion:
            return {"passed": False, "error": "tests must contain at least one assert"}
        if forbidden_defs:
            return {
                "passed": False,
                "error": "tests must not define execute or reassign skill_module",
            }
        if forbidden_monkeypatch:
            return {
                "passed": False,
                "error": "tests must not mock/patch skill_module implementation",
            }
        if not execute_calls:
            return {"passed": False, "error": "tests must call execute from skill_module"}

        if project_mode:
            lowered = tests.lower()
            if re.search(r"assert\s+os\.path\.isabs\s*\(", lowered):
                return {
                    "passed": False,
                    "error": "project tests must validate relative paths, not assert absolute paths",
                }
            if "is_relative_to(\"/\")" in lowered or "is_relative_to('/')" in lowered:
                return {
                    "passed": False,
                    "error": "project tests must validate relative paths, not absolute-path checks",
                }
            if "expected_project_structure" in lowered:
                return {
                    "passed": False,
                    "error": "project tests must not hardcode full project template snapshots",
                }
            if "project_name=" not in lowered and "project_goal=" not in lowered:
                return {
                    "passed": False,
                    "error": "project tests should call execute with project_name/project_goal inputs",
                }
            readme_lines = [
                line.strip().lower()
                for line in tests.splitlines()
                if "readme_content" in line and "assert" in line
            ]
            for line in readme_lines:
                if " in readme_content" in line and "readme_content.lower()" not in line:
                    return {
                        "passed": False,
                        "error": "readme assertions must be case-insensitive (use readme_content.lower())",
                    }

        return {"passed": True, "error": None}

    @staticmethod
    def _format_pytest_failure(details: Dict[str, Any]) -> str:
        """Summarize pytest failure output for LLM repair prompts."""
        stdout = str(details.get("stdout") or "").strip()
        stderr = str(details.get("stderr") or "").strip()
        chunks: List[str] = ["pytest failed"]
        if stdout:
            failure_marker = "================================== FAILURES ==================================="
            marker_idx = stdout.find(failure_marker)
            if marker_idx >= 0:
                chunks.append(f"failure_excerpt:\n{stdout[marker_idx:marker_idx + 3500]}")
            short_summary_idx = stdout.lower().find("short test summary info")
            if short_summary_idx >= 0:
                chunks.append(f"short_summary:\n{stdout[short_summary_idx:short_summary_idx + 1500]}")
            elif marker_idx < 0:
                excerpt = stdout if len(stdout) <= 2500 else f"{stdout[:1200]}\n...\n{stdout[-1200:]}"
                chunks.append(f"stdout_excerpt:\n{excerpt}")
        if stderr:
            chunks.append(f"stderr:\n{stderr[:1200]}")
        return "\n\n".join(chunks)

    @staticmethod
    def _project_generation_rules(requirements: List[str]) -> List[str]:
        if not EvolutionService._detect_project_contract_mode(requirements):
            return []
        min_file_count = EvolutionService._infer_project_min_file_count(requirements)
        return [
            (
                "execute()['files'] must contain at least "
                f"{min_file_count} non-empty files spread across modular directories."
            ),
            "Never output placeholder-only files or comments (TODO/TBD/placeholder/render logic/implement here).",
            (
                "apps/server/src/server.mjs must contain explicit '/healthz' and '/ws' strings, "
                "read PORT from process.env.PORT, and send JSON websocket messages with type "
                "including welcome and snapshot/state."
            ),
            (
                "Initialize websocket server with one of these literal forms in source: "
                "`new WebSocketServer({ server })` (ESM) or `new Server({ server })` (CJS import alias)."
            ),
            (
                "Implement fish-eat-fish gameplay semantics in code/comments/tests: fish entities, "
                "collision detection, devour/eat behavior, and growth (mass/size/score) updates."
            ),
            (
                "Gameplay modules must contain substantive logic (functions/state updates), "
                "not comment-only or console.log-only stubs."
            ),
            (
                "scripts/smoke/ws-smoke.mjs must include timeout protection and explicit exits: "
                "process.exit(0) on success, non-zero exit on failure, and websocket URL built from process.env.PORT."
            ),
            "Keep output deterministic for the same project_name/project_goal input.",
        ]

    @staticmethod
    def _derive_failure_repair_hints(failure_message: str, requirements: List[str]) -> List[str]:
        message = str(failure_message or "")
        lower = message.lower()
        hints: List[str] = []

        def add_hint(text: str) -> None:
            if text not in hints:
                hints.append(text)

        min_files_match = re.search(
            r"assert\s+len\(files\)\s*>=\s*(\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if min_files_match:
            add_hint(
                "Increase execute()['files'] count to at least "
                f"{min_files_match.group(1)} with meaningful module files."
            )
        elif EvolutionService._detect_project_contract_mode(requirements):
            inferred_min = EvolutionService._infer_project_min_file_count(requirements)
            add_hint(
                "Ensure execute()['files'] contains at least "
                f"{inferred_min} files with server/client/world/matchmaking/anti-cheat modules."
            )

        placeholder_tokens = ("placeholder token in", "render logic", "todo", "tbd", "coming soon")
        if any(token in lower for token in placeholder_tokens):
            add_hint(
                "Remove placeholder comments/text from every generated file and replace them with concrete logic."
            )

        if 'assert "/ws" in server' in lower or "assert '/ws' in server" in lower:
            add_hint("Add explicit '/ws' path handling text in apps/server/src/server.mjs.")
        if "has_ws_server_ctor or has_cjs_ws_ctor" in lower:
            add_hint(
                "Construct websocket server with `new WebSocketServer({ server })` or `new Server({ server })`."
            )

        if "process.exit(0)" in lower and "smoke" in lower:
            add_hint(
                "In scripts/smoke/ws-smoke.mjs, call process.exit(0) after receiving welcome/snapshot/state."
            )
        if "process.env.port" in lower and "smoke" in lower:
            add_hint("Use process.env.PORT in smoke websocket URL; do not hardcode port 3000/8080.")

        if "keyerror" in lower or ".format(" in lower or "brace" in lower:
            add_hint(
                "Avoid .format(...) and risky f-string interpolation for JS/JSON templates; use plain literals."
            )

        if ".on('/healthz'" in lower or '.on("/healthz"' in lower or ".on('/ws'" in lower or '.on("/ws"' in lower:
            add_hint("Do not register HTTP routes with server.on('/healthz')/wss.on('/ws'); handle via request listener and ws server setup.")

        if "package_scripts_quality" in lower or '"ws" in dependencies' in lower:
            add_hint(
                "Declare ws dependency in package.json and keep scripts.start:server + scripts.smoke:ws executable."
            )

        if "test_domain_modules_present_for_large_game" in lower or "missing_domain_module" in lower:
            add_hint(
                "Provide concrete world, matchmaking, and anti-cheat modules in source file paths."
            )
        if "test_fish_gameplay_semantics_present" in lower:
            add_hint(
                "Add explicit fish-eat-fish semantics: collision/overlap, devour/eat, and growth via mass/size/score."
            )
        if "test_gameplay_modules_have_substantive_logic" in lower:
            add_hint(
                "Replace comment/console-only gameplay modules with real logic functions and state transitions."
            )

        if not hints:
            add_hint("Resolve every failing assertion from pytest output exactly.")

        return hints

    @staticmethod
    def _detect_project_contract_mode(requirements: List[str]) -> bool:
        merged_req = "\n".join(requirements).lower()
        return (
            "project_name, project_slug, files, run_instructions" in merged_req
            or "files must be dict[str, str]" in merged_req
        )

    @staticmethod
    def _extract_required_paths_from_requirements(requirements: List[str]) -> List[str]:
        marker = "required output file paths:"
        for req in requirements:
            raw = str(req or "").strip()
            if raw.lower().startswith(marker):
                values = raw[len(marker):].strip()
                paths = [item.strip() for item in values.split(",") if item.strip()]
                deduped: List[str] = []
                seen = set()
                for path in paths:
                    normalized = path.replace("\\", "/")
                    if normalized not in seen:
                        seen.add(normalized)
                        deduped.append(normalized)
                return deduped
        return []

    @staticmethod
    def _infer_project_min_file_count(requirements: List[str]) -> int:
        merged = "\n".join(str(item or "").lower() for item in requirements)
        large_keywords = (
            "mmo",
            "multiplayer",
            "websocket",
            "real-time",
            "large",
            "large-scale",
            "多人在线",
            "大型",
        )
        if any(keyword in merged for keyword in large_keywords):
            return 18
        return 6

    @staticmethod
    def _build_project_contract_tests(requirements: List[str]) -> str:
        required_paths = EvolutionService._extract_required_paths_from_requirements(requirements)
        min_file_count = EvolutionService._infer_project_min_file_count(requirements)
        required_literal = repr(required_paths)
        return f'''import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import skill_module

REQUIRED_PATHS = {required_literal}
MIN_FILE_COUNT = {min_file_count}
PLACEHOLDER_TOKENS = (
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


def _run_once():
    return skill_module.execute(
        project_name="Abyss Arena",
        project_goal="Build MMO fish-eat-fish core scaffold",
    )


def test_execute_output_shape():
    out = _run_once()
    assert isinstance(out, dict)
    expected = {{"project_name", "project_slug", "files", "run_instructions"}}
    assert expected.issubset(set(out.keys()))


def test_execute_signature_contract():
    sig = inspect.signature(skill_module.execute)
    assert "project_name" in sig.parameters
    assert "project_goal" in sig.parameters


def test_files_are_relative_utf8_text():
    out = _run_once()
    files = out["files"]
    assert isinstance(files, dict)
    assert files
    for rel_path, content in files.items():
        assert isinstance(rel_path, str) and rel_path.strip()
        assert not os.path.isabs(rel_path)
        assert ".." not in rel_path.replace("\\\\", "/")
        assert isinstance(content, str)
        assert content.strip()


def test_required_paths_present():
    out = _run_once()
    files = out["files"]
    for path in REQUIRED_PATHS:
        assert path in files


def test_file_count_meets_scale_floor():
    out = _run_once()
    files = out["files"]
    assert len(files) >= MIN_FILE_COUNT
    assert any("/" in path for path in files.keys())


def test_runtime_critical_files_present():
    out = _run_once()
    files = out["files"]
    assert "apps/server/src/server.mjs" in files
    assert "scripts/smoke/ws-smoke.mjs" in files
    assert "package.json" in files


def test_domain_modules_present_for_large_game():
    out = _run_once()
    files = out["files"]
    paths = [path.lower() for path in files.keys()]
    assert any("apps/server/" in path for path in paths)
    assert any("apps/client/" in path or "frontend/" in path for path in paths)
    assert any("world" in path for path in paths)
    assert any("match" in path for path in paths)
    assert any("anti" in path and "cheat" in path for path in paths)


def test_fish_gameplay_semantics_present():
    out = _run_once()
    files = out["files"]
    corpus = "\\n".join(str(content).lower() for content in files.values())
    fish_terms = ("fish", "大鱼", "小鱼")
    eat_terms = ("eat", "devour", "consume", "吞噬")
    grow_terms = ("grow", "growth", "mass", "size", "score", "成长")
    collision_terms = ("collision", "overlap", "hitbox", "碰撞")
    assert any(term in corpus for term in fish_terms)
    assert any(term in corpus for term in eat_terms)
    assert any(term in corpus for term in grow_terms)
    assert any(term in corpus for term in collision_terms)
    assert "move" in corpus


def test_no_placeholder_tokens():
    out = _run_once()
    files = out["files"]
    for rel_path, content in files.items():
        lower = str(content).lower()
        for token in PLACEHOLDER_TOKENS:
            assert token not in lower, f"placeholder token in {{rel_path}}: {{token}}"


def test_core_source_size_floor():
    out = _run_once()
    files = out["files"]
    server_lines = len(files.get("apps/server/src/server.mjs", "").splitlines())
    smoke_lines = len(files.get("scripts/smoke/ws-smoke.mjs", "").splitlines())
    assert server_lines >= 30
    assert smoke_lines >= 20
    source_like = [
        path for path in files.keys()
        if path.lower().endswith((".js", ".mjs", ".ts", ".tsx")) and ("/src/" in path.lower() or path.lower().startswith("scripts/"))
    ]
    assert len(source_like) >= 6


def test_gameplay_modules_have_substantive_logic():
    out = _run_once()
    files = out["files"]
    gameplay_paths = [
        path
        for path in files.keys()
        if path.lower().endswith((".js", ".mjs", ".ts"))
        and any(
            token in path.lower()
            for token in (
                "world",
                "match",
                "anti-cheat",
                "anticheat",
                "fish",
                "devour",
                "growth",
                "collision",
                "movement",
            )
        )
    ]
    assert len(gameplay_paths) >= 6
    substantive_count = 0
    for path in gameplay_paths:
        content = str(files.get(path, ""))
        lowered = content.lower()
        non_comment_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
            and not line.strip().startswith("//")
            and not line.strip().startswith("/*")
            and not line.strip().startswith("*")
        ]
        has_logic_token = any(
            token in lowered
            for token in (
                "function",
                "=>",
                "class ",
                "if ",
                "for ",
                "while ",
                "switch ",
                "return ",
                "export const",
                "export function",
            )
        )
        only_console = bool(non_comment_lines) and all("console.log" in line.lower() for line in non_comment_lines)
        if len(non_comment_lines) >= 4 and has_logic_token and not only_console:
            substantive_count += 1
    assert substantive_count >= 4


def test_core_js_files_are_node_parseable():
    out = _run_once()
    files = out["files"]
    node_bin = shutil.which("node")
    assert node_bin
    targets = (
        "apps/server/src/server.mjs",
        "scripts/smoke/ws-smoke.mjs",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        for rel_path in targets:
            content = files.get(rel_path, "")
            assert isinstance(content, str) and content.strip()
            temp_file = Path(temp_dir) / Path(rel_path).name
            temp_file.write_text(content, encoding="utf-8")
            check = subprocess.run(
                [node_bin, "--check", str(temp_file)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert check.returncode == 0, f"node --check failed for {{rel_path}}: {{(check.stderr or check.stdout)[-400:]}}"


def test_readme_has_install_run_usage():
    out = _run_once()
    readme = out["files"].get("README.md", "").lower()
    assert "install" in readme
    assert "run" in readme
    assert "usage" in readme


def test_run_instructions_non_empty():
    out = _run_once()
    value = out.get("run_instructions")
    assert isinstance(value, str)
    assert value.strip()


def test_skill_source_avoids_multiline_fstring_templates():
    source = inspect.getsource(skill_module).lower()
    assert 'f"""' not in source
    assert "f" + ("'" * 3) not in source


def test_skill_source_avoids_format_template_brace_risk():
    source = inspect.getsource(skill_module)
    assert ".format(" not in source


def test_package_scripts_quality():
    out = _run_once()
    package_raw = out["files"].get("package.json", "")
    package = json.loads(package_raw)
    scripts = package.get("scripts", {{}})
    dependencies = package.get("dependencies", {{}})
    dev_dependencies = package.get("devDependencies", {{}})
    assert "start:server" in scripts
    assert "smoke:ws" in scripts
    start_cmd = str(scripts["start:server"]).lower()
    smoke_cmd = str(scripts["smoke:ws"]).lower()
    assert "echo" not in start_cmd
    assert "node" in start_cmd or "npm" in start_cmd
    assert "node" in smoke_cmd or "npm" in smoke_cmd
    assert "server" in start_cmd
    assert "smoke" in smoke_cmd and "ws" in smoke_cmd
    assert "ws" in dependencies or "ws" in dev_dependencies


def test_server_third_party_imports_declared_in_package():
    out = _run_once()
    files = out["files"]
    server = files.get("apps/server/src/server.mjs", "")
    package = json.loads(files.get("package.json", "{{}}"))
    dependencies = package.get("dependencies", {{}})
    dev_dependencies = package.get("devDependencies", {{}})
    declared = set(str(key) for key in list(dependencies.keys()) + list(dev_dependencies.keys()))
    third_party = set()
    builtin_modules = {{
        "assert", "buffer", "child_process", "cluster", "console", "crypto", "dgram",
        "dns", "events", "fs", "http", "http2", "https", "module", "net", "os",
        "path", "perf_hooks", "process", "querystring", "readline", "repl", "stream",
        "string_decoder", "timers", "tls", "tty", "url", "util", "v8", "vm",
        "worker_threads", "zlib"
    }}
    for quoted in re.findall(r'from\\s+[\\'\\"]([^\\'\\"]+)[\\'\\"]', server):
        if quoted.startswith(".") or quoted.startswith("/") or quoted.startswith("node:"):
            continue
        segments = quoted.split("/")
        if quoted.startswith("@") and len(segments) >= 2:
            package_name = "/".join(segments[:2])
        else:
            package_name = segments[0]
        if package_name in builtin_modules:
            continue
        third_party.add(package_name)
    for required in third_party:
        assert required in declared


def test_server_contains_health_and_ws_reply_logic():
    out = _run_once()
    files = out["files"]
    server = out["files"].get("apps/server/src/server.mjs", "").lower()
    all_text = "\\n".join(str(content).lower() for content in files.values())
    assert "/healthz" in server
    assert "/ws" in server
    assert "process.env.port" in server or "process.env.port" in all_text
    assert "from 'ws'" in server or 'from "ws"' in server or "require('ws')" in server
    has_ws_server_ctor = "new websocketserver(" in server or "new server(" in server
    has_cjs_ws_ctor = "require('ws')" in server and ".server(" in server
    assert has_ws_server_ctor or has_cjs_ws_ctor
    assert "new websocket.server(" not in server
    assert ".on('/healthz'" not in server and '.on("/healthz"' not in server
    assert ".on('/ws'" not in server and '.on("/ws"' not in server
    has_inline_http_handler = "createserver((req, res)" in server or "createserver(function (req, res)" in server
    has_extra_request_handler = ".on('request'" in server or '.on("request"' in server
    assert not (has_inline_http_handler and has_extra_request_handler)
    assert "websocket" in server or "ws" in server
    assert "on('message'" in server or 'on(\"message\"' in server
    assert "ws.send" in server
    assert "json.stringify" in server
    assert "welcome" in server
    assert "snapshot" in server or "state" in server


def test_ws_smoke_has_timeout_and_failure_exit():
    out = _run_once()
    smoke_raw = out["files"].get("scripts/smoke/ws-smoke.mjs", "")
    smoke = smoke_raw.lower()
    assert "ws://127.0.0.1" in smoke or "ws://localhost" in smoke
    assert "process.env.port" in smoke
    assert "/ws" in smoke
    assert "settimeout" in smoke
    assert "ws.send(" in smoke or "send(" in smoke
    assert "on('open'" in smoke or "onopen" in smoke
    assert "on('message'" in smoke or "onmessage" in smoke
    assert "on('error'" in smoke or "onerror" in smoke
    assert "json.parse" in smoke
    assert "JSON.parse(" in smoke_raw
    assert "welcome" in smoke
    assert "snapshot" in smoke or "state" in smoke
    assert "process.exit(0)" in smoke_raw
    assert "process.exit(1)" in smoke or "process.exit(2)" in smoke or "process.exit(3)" in smoke


def test_output_deterministic():
    assert _run_once() == _run_once()
'''

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
    def _render_pytest_from_test_cases(test_cases: List[Dict[str, Any]]) -> str:
        """Render deterministic pytest code from semantic test case definitions."""
        if not test_cases:
            return (
                "import skill_module\n\n"
                "def test_skill_module_importable():\n"
                "    assert skill_module is not None\n"
            )

        lines = [
            "import asyncio",
            "import inspect",
            "import skill_module",
            "",
            "def _select_callable(function_name=None):",
            "    if function_name and hasattr(skill_module, function_name):",
            "        candidate = getattr(skill_module, function_name)",
            "        if callable(candidate):",
            "            return candidate",
            "    if hasattr(skill_module, 'execute') and callable(skill_module.execute):",
            "        return skill_module.execute",
            "    public_callables = []",
            "    for name in dir(skill_module):",
            "        if name.startswith('_'):",
            "            continue",
            "        candidate = getattr(skill_module, name)",
            "        if callable(candidate):",
            "            public_callables.append(candidate)",
            "    if len(public_callables) == 1:",
            "        return public_callables[0]",
            "    raise AssertionError('No unambiguous callable found in skill_module')",
            "",
            "def _invoke(function_name, payload):",
            "    fn = _select_callable(function_name)",
            "    if isinstance(payload, dict):",
            "        result = fn(**payload)",
            "    elif isinstance(payload, (list, tuple)):",
            "        result = fn(*payload)",
            "    elif payload is None:",
            "        result = fn()",
            "    else:",
            "        result = fn(payload)",
            "    if inspect.isawaitable(result):",
            "        return asyncio.run(result)",
            "    return result",
            "",
        ]

        for index, case in enumerate(test_cases, 1):
            description = str(case.get("description", f"test_case_{index}")).replace('"', "'")
            function_name = case.get("function")
            payload = case.get("input")
            expected = case.get("expected")

            lines.append(f"def test_case_{index}():")
            lines.append(f'    """{description}"""')
            lines.append(f"    result = _invoke({function_name!r}, {payload!r})")
            lines.append(f"    assert result == {expected!r}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

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
