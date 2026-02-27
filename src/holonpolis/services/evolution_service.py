"""Evolution Service - The RGV (Red-Green-Verify) Crucible.

演化裁判所 - 执行 Red-Green-Verify 演化闭环：
1. Red: 编写预期失败的 pytest
2. Green: 提交代码通过测试
3. Verify: AST 安全扫描

所有技能演化必须经过此裁判所才能落盘。
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from holonpolis.config import settings
from holonpolis.domain.skills import SkillManifest, SkillVersion, ToolSchema
from holonpolis.kernel.sandbox import SandboxConfig, SandboxResult, SandboxRunner
from holonpolis.kernel.storage import HolonPathGuard
from holonpolis.services.holon_service import HolonService

logger = structlog.get_logger()


@dataclass
class Attestation:
    """演化证明 - 技能通过 RGV 的证据。"""

    attestation_id: str
    holon_id: str
    skill_name: str
    version: str

    # RGV 阶段
    red_phase_passed: bool  # 测试定义有效
    green_phase_passed: bool  # 代码通过测试
    verify_phase_passed: bool  # AST 安全扫描通过

    # 详细结果
    test_results: Dict[str, Any] = field(default_factory=dict)
    security_scan_results: Dict[str, Any] = field(default_factory=dict)

    # 代码指纹
    code_hash: str = ""  # SHA256 of code
    test_hash: str = ""  # SHA256 of tests

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
    """技能演化结果。"""

    success: bool
    skill_id: Optional[str] = None
    attestation: Optional[Attestation] = None
    error_message: Optional[str] = None
    phase: str = ""  # red, green, verify, persist

    # 落盘路径
    code_path: Optional[str] = None
    test_path: Optional[str] = None
    manifest_path: Optional[str] = None


class SecurityScanner:
    """AST 安全扫描器 - Verify 阶段。"""

    # 危险导入模式
    DANGEROUS_IMPORTS = frozenset({
        "os.system", "subprocess.call", "subprocess.run", "subprocess.Popen",
        "eval", "exec", "compile", "__import__",
        "pickle.loads", "pickle.load", "yaml.load", "yaml.unsafe_load",
    })

    # 危险函数调用
    DANGEROUS_CALLS = frozenset({
        "eval", "exec", "compile",
        "getattr", "setattr", "delattr",
        "globals", "locals", "vars",
    })

    # 敏感属性访问
    SENSITIVE_ATTRS = frozenset({
        "__code__", "__globals__", "__closure__",
        "__subclasses__", "__bases__", "__mro__",
    })

    def __init__(self):
        self.violations: List[Dict[str, Any]] = []

    def scan(self, code: str, filename: str = "<unknown>") -> Dict[str, Any]:
        """扫描代码返回安全报告。

        Returns:
            {
                "passed": bool,
                "violations": List[dict],
                "complexity_score": float,
            }
        """
        self.violations = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "passed": False,
                "violations": [{"type": "syntax_error", "message": str(e)}],
                "complexity_score": 0.0,
            }

        self._analyze_node(tree, filename)

        # 计算复杂度分数 (简单的圈复杂度估计)
        complexity = self._estimate_complexity(tree)

        return {
            "passed": len(self.violations) == 0,
            "violations": self.violations,
            "complexity_score": complexity,
        }

    def _analyze_node(self, node: ast.AST, filename: str) -> None:
        """递归分析 AST 节点。"""
        for child in ast.walk(node):
            self._check_node(child, filename)

    def _check_node(self, node: ast.AST, filename: str) -> None:
        """检查单个节点。"""
        # 检查危险导入
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in self.DANGEROUS_IMPORTS:
                    self.violations.append({
                        "type": "dangerous_import",
                        "line": getattr(node, "lineno", 0),
                        "name": alias.name,
                        "message": f"Dangerous import: {alias.name}",
                    })

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                full_name = f"{module}.{alias.name}"
                if full_name in self.DANGEROUS_IMPORTS:
                    self.violations.append({
                        "type": "dangerous_import",
                        "line": getattr(node, "lineno", 0),
                        "name": full_name,
                        "message": f"Dangerous import: {full_name}",
                    })

        # 检查危险函数调用
        elif isinstance(node, ast.Call):
            # 直接调用 (如 eval())
            if isinstance(node.func, ast.Name):
                if node.func.id in self.DANGEROUS_CALLS:
                    self.violations.append({
                        "type": "dangerous_call",
                        "line": getattr(node, "lineno", 0),
                        "name": node.func.id,
                        "message": f"Dangerous function call: {node.func.id}",
                    })
            # 模块/对象方法调用 (如 os.system())
            elif isinstance(node.func, ast.Attribute):
                full_call = self._get_full_attribute_name(node.func)
                if full_call in self.DANGEROUS_IMPORTS:
                    self.violations.append({
                        "type": "dangerous_call",
                        "line": getattr(node, "lineno", 0),
                        "name": full_call,
                        "message": f"Dangerous function call: {full_call}",
                    })

        # 检查敏感属性访问
        elif isinstance(node, ast.Attribute):
            if node.attr in self.SENSITIVE_ATTRS:
                self.violations.append({
                    "type": "sensitive_attribute",
                    "line": getattr(node, "lineno", 0),
                    "name": node.attr,
                    "message": f"Sensitive attribute access: {node.attr}",
                })

    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        """获取属性链的完整名称 (如 os.system.call -> 'os.system.call')。"""
        parts = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _estimate_complexity(self, tree: ast.AST) -> float:
        """估算代码复杂度。"""
        # 简单的圈复杂度估计
        decision_points = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With,
                                ast.Try, ast.ExceptHandler, ast.comprehension)):
                decision_points += 1
            elif isinstance(node, ast.BoolOp):
                decision_points += len(node.values) - 1

        # 归一化到 0-10
        return min(10.0, decision_points / 5.0)


class EvolutionService:
    """演化服务 - RGV 裁判所。

    负责：
    1. 接收技能演化请求 (代码 + 测试)
    2. 执行 Red-Green-Verify 循环
    3. 通过后落盘到 skills_local
    4. 生成 Attestation
    """

    def __init__(self):
        self._sandbox: Optional[SandboxRunner] = None
        self.security_scanner = SecurityScanner()
        self.holon_service = HolonService()

    @property
    def sandbox(self) -> SandboxRunner:
        """Lazy initialization of sandbox runner."""
        if self._sandbox is None:
            self._sandbox = SandboxRunner()
        return self._sandbox

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
        """演化新技能 - 完整 RGV 循环。

        Args:
            holon_id: 发起演化的 Holon ID
            skill_name: 技能名称
            code: 技能代码 (Python)
            tests: pytest 测试代码
            description: 技能描述
            tool_schema: 工具 schema
            version: 版本号

        Returns:
            EvolutionResult
        """
        logger.info(
            "evolution_started",
            holon_id=holon_id,
            skill_name=skill_name,
            version=version,
        )

        # Phase 1: Red - 验证测试定义有效
        red_result = await self._phase_red(tests)
        if not red_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="red",
                error_message=f"Red phase failed: {red_result['error']}",
            )

        # Phase 2: Green - 代码通过测试
        green_result = await self._phase_green(code, tests, holon_id)
        if not green_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="green",
                error_message=f"Green phase failed: {green_result['error']}",
            )

        # Phase 3: Verify - AST 安全扫描
        verify_result = self._phase_verify(code)
        if not verify_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="verify",
                error_message=f"Verify phase failed: security violations found",
            )

        # Phase 4: Persist - 落盘到 skills_local
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
                error_message=f"Persist phase failed: {persist_result['error']}",
            )

        logger.info(
            "evolution_completed",
            holon_id=holon_id,
            skill_name=skill_name,
            attestation_id=persist_result["attestation"].attestation_id,
        )

        return EvolutionResult(
            success=True,
            skill_id=persist_result["skill_id"],
            attestation=persist_result["attestation"],
            phase="complete",
            code_path=persist_result.get("code_path"),
            test_path=persist_result.get("test_path"),
            manifest_path=persist_result.get("manifest_path"),
        )

    async def _phase_red(self, tests: str) -> Dict[str, Any]:
        """Red 阶段: 验证测试代码语法有效。

        Returns:
            {"passed": bool, "error": str or None}
        """
        # 检查测试代码语法
        try:
            ast.parse(tests)
        except SyntaxError as e:
            return {"passed": False, "error": f"Test syntax error: {e}"}

        return {"passed": True, "error": None}

    async def _phase_green(self, code: str, tests: str, holon_id: Optional[str] = None) -> Dict[str, Any]:
        """Green 阶段: 运行 pytest 验证代码。

        Returns:
            {"passed": bool, "error": str or None, "details": dict}
        """
        # 使用 Holon 工作目录 (必须在 .holonpolis 内)
        if holon_id:
            guard = HolonPathGuard(holon_id)
            work_dir = guard.ensure_directory("temp/evolution")
        else:
            # 使用沙箱临时目录
            work_dir = Path(self.sandbox._root) / "temp" / "evolution"
            work_dir.mkdir(parents=True, exist_ok=True)

        # 清理旧文件
        for f in work_dir.glob("*.py"):
            f.unlink()

        # 写入代码和测试
        code_file = work_dir / "skill_module.py"
        test_file = work_dir / "test_skill.py"

        code_file.write_text(code, encoding="utf-8")
        test_file.write_text(tests, encoding="utf-8")

        # 使用 Sandbox 运行 pytest
        result = await self.sandbox.run(
            command=[
                "python", "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
            ],
            config=SandboxConfig(
                timeout_seconds=settings.evolution_pytest_timeout,
                working_dir=work_dir,
                strict_exit_code=False,  # pytest 非零退出码表示测试失败
                inherit_env=True,  # 继承环境变量以获取 HOME 等必要变量
            ),
        )

        # 解析结果
        success = result.exit_code == 0

        return {
            "passed": success,
            "error": None if success else result.stderr,
            "details": {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
            },
        }

    def _phase_verify(self, code: str) -> Dict[str, Any]:
        """Verify 阶段: AST 安全扫描。

        Returns:
            {"passed": bool, "violations": list, "complexity": float}
        """
        scan_result = self.security_scanner.scan(code)

        return {
            "passed": scan_result["passed"],
            "violations": scan_result["violations"],
            "complexity": scan_result["complexity_score"],
        }

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
        """Persist 阶段: 落盘到 skills_local。"""
        try:
            guard = HolonPathGuard(holon_id)

            # 创建 skills 目录
            skills_path = guard.ensure_directory("skills")
            skill_dir = skills_path / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # 安全文件名
            safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", skill_name).strip("_").lower()

            # 写入代码
            code_file = skill_dir / f"{safe_name}.py"
            code_file.write_text(code, encoding="utf-8")

            # 写入测试
            test_file = skill_dir / f"test_{safe_name}.py"
            test_file.write_text(tests, encoding="utf-8")

            # 计算哈希
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
            test_hash = hashlib.sha256(tests.encode()).hexdigest()[:16]

            # 生成 Attestation
            attestation_id = f"att_{holon_id}_{safe_name}_{code_hash[:8]}"
            attestation = Attestation(
                attestation_id=attestation_id,
                holon_id=holon_id,
                skill_name=skill_name,
                version=version,
                red_phase_passed=True,
                green_phase_passed=True,
                verify_phase_passed=verify_result["passed"],
                test_results=green_result.get("details", {}),
                security_scan_results=verify_result,
                code_hash=code_hash,
                test_hash=test_hash,
            )

            # 写入 Attestation
            att_file = skill_dir / "attestation.json"
            att_file.write_text(
                json.dumps(attestation.to_dict(), indent=2),
                encoding="utf-8"
            )

            # 创建/更新 Manifest
            manifest = SkillManifest(
                skill_id=f"skill_{holon_id}_{safe_name}",
                name=skill_name,
                description=description,
                version=version,
                tool_schema=tool_schema,
                author_holon=holon_id,
                versions=[SkillVersion(
                    version=version,
                    created_by=holon_id,
                    code_path=str(code_file.relative_to(guard.holon_base)),
                    test_path=str(test_file.relative_to(guard.holon_base)),
                    attestation_id=attestation_id,
                    test_results=green_result.get("details", {}),
                    static_scan_passed=verify_result["passed"],
                )],
            )

            manifest_file = skill_dir / "manifest.json"
            manifest_file.write_text(
                json.dumps(manifest.to_dict(), indent=2),
                encoding="utf-8"
            )

            return {
                "success": True,
                "skill_id": manifest.skill_id,
                "attestation": attestation,
                "code_path": str(code_file),
                "test_path": str(test_file),
                "manifest_path": str(manifest_file),
            }

        except Exception as e:
            logger.error("persist_failed", error=str(e), holon_id=holon_id, skill_name=skill_name)
            return {"success": False, "error": str(e)}

    async def validate_existing_skill(
        self,
        holon_id: str,
        skill_name: str,
    ) -> Dict[str, Any]:
        """验证已存在的技能 (重新运行 RGV)。"""
        try:
            guard = HolonPathGuard(holon_id)
            skills_path = guard.resolve(f"skills/{skill_name}")

            if not skills_path.exists():
                return {"valid": False, "error": "Skill not found"}

            # 读取代码和测试
            safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", skill_name).strip("_").lower()
            code_file = skills_path / f"{safe_name}.py"
            test_file = skills_path / f"test_{safe_name}.py"

            if not code_file.exists() or not test_file.exists():
                return {"valid": False, "error": "Missing code or test file"}

            code = code_file.read_text(encoding="utf-8")
            tests = test_file.read_text(encoding="utf-8")

            # 重新运行 Green 和 Verify
            green_result = await self._phase_green(code, tests)
            verify_result = self._phase_verify(code)

            return {
                "valid": green_result["passed"] and verify_result["passed"],
                "green_passed": green_result["passed"],
                "verify_passed": verify_result["passed"],
                "violations": verify_result.get("violations", []),
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}


def create_evolution_service() -> EvolutionService:
    """工厂函数: 创建 EvolutionService 实例。"""
    return EvolutionService()
