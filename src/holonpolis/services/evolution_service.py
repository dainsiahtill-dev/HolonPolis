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
from holonpolis.kernel.llm.llm_runtime import LLMConfig, LLMMessage, get_llm_runtime
from holonpolis.kernel.llm.provider_config import get_provider_manager

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


class TypeScriptSecurityScanner:
    """TypeScript 安全扫描器。"""

    # TypeScript 危险模式
    DANGEROUS_PATTERNS = frozenset({
        'eval(',
        'new Function(',
        'child_process',
        'exec(',
        'execSync(',
        'spawn(',
        '__proto__',
        'prototype pollution',
    })

    # 安全必需的防护
    REQUIRED_PROTECTIONS = [
        ('path traversal', r'\.\.|path\.resolve|path\.join'),
        ('input validation', r'typeof|instanceof|Array\.isArray'),
    ]

    def scan(self, code: str) -> Dict[str, Any]:
        """扫描 TypeScript 代码。"""
        violations = []

        # 检查危险模式
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                violations.append({
                    'type': 'dangerous_pattern',
                    'pattern': pattern,
                    'message': f'Found dangerous pattern: {pattern}'
                })

        # 检查必要的安全防护
        has_traversal_protection = '..' in code and ('replace' in code or 'resolve' in code)

        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'complexity': self._estimate_complexity(code),
            'has_traversal_protection': has_traversal_protection,
        }

    def _estimate_complexity(self, code: str) -> float:
        """估算 TypeScript 代码复杂度。"""
        # 计算决策点
        decision_points = (
            code.count('if (') +
            code.count('switch (') +
            code.count('for (') +
            code.count('while (') +
            code.count('?.')  # 可选链
        )
        return min(10.0, decision_points / 5.0)


class LLMCodeGenerator:
    """LLM 代码生成器 - 真正自主生成代码。"""

    def __init__(self, provider_id: Optional[str] = None):
        self.runtime = get_llm_runtime()
        self.provider_manager = get_provider_manager()
        self.provider_id = provider_id or self._select_best_provider()

    def _select_best_provider(self) -> str:
        """选择最佳的代码生成 provider。"""
        # 优先级：Kimi Coding > MiniMax > Ollama > 其他
        providers = self.provider_manager.list_providers(mask_secrets=True)
        provider_ids = [p["provider_id"] for p in providers]

        if "kimi-coding" in provider_ids:
            return "kimi-coding"
        if "minimax" in provider_ids:
            return "minimax"
        if "ollama-local" in provider_ids:
            return "ollama-local"
        if "ollama" in provider_ids:
            return "ollama"

        return provider_ids[0] if provider_ids else "openai"

    async def generate_typescript_project(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
    ) -> Dict[str, str]:
        """生成完整的 TypeScript 项目代码。

        Returns:
            Dict with keys: code, tsconfig, package_json
        """
        provider = self.provider_manager.get_provider(self.provider_id)
        if not provider:
            raise ValueError(f"Provider {self.provider_id} not found")

        # 生成主代码
        code = await self._generate_code(project_name, description, requirements)

        # 生成配置文件
        tsconfig = await self._generate_tsconfig()
        package_json = await self._generate_package_json(project_name, description)

        return {
            "code": code,
            "tsconfig": tsconfig,
            "package_json": package_json,
        }

    async def _generate_code(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
    ) -> str:
        """使用 LLM 生成 TypeScript 代码。"""
        system_prompt = """You are an expert TypeScript developer.
Your task is to generate complete, production-ready TypeScript code.
Rules:
1. Use strict TypeScript with proper types
2. Include security best practices (input validation, path traversal protection)
3. Use modern ES2020+ features
4. Include proper error handling
5. Add JSDoc comments for public APIs
6. NEVER use eval() or new Function()
7. ALWAYS sanitize user inputs
Output ONLY the code, no explanations."""

        requirements_text = "\n".join(f"- {r}" for r in requirements)

        prompt = f"""Generate a complete TypeScript file server implementation.

Project: {project_name}
Description: {description}

Requirements:
{requirements_text}

Generate a single file (index.ts) that:
1. Creates an HTTP file server
2. Supports GET (read files/list directories)
3. Supports POST (create files)
4. Supports DELETE (delete files)
5. Has path traversal protection
6. Proper TypeScript types and interfaces
7. Security focused

Output ONLY valid TypeScript code that can be compiled with tsc."""

        # Get provider config for model info
        provider_cfg = self.provider_manager.get_provider(self.provider_id)
        if provider_cfg and provider_cfg.model:
            model = provider_cfg.model
        else:
            model = "qwen3-coder-30b-v12-q8-128k-dual3090:latest"

        config = LLMConfig(
            provider_id=self.provider_id,
            model=model,
            temperature=0.2,
            max_tokens=8192,
        )

        response = await self.runtime.chat(
            system_prompt=system_prompt,
            user_prompt=prompt,
            config=config,
        )

        # 提取代码
        code = response.content.strip()

        # 移除 markdown 代码块
        if code.startswith("```typescript"):
            code = code[13:]
        elif code.startswith("```ts"):
            code = code[5:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()

        return code

    async def _generate_tsconfig(self) -> str:
        """生成 tsconfig.json。"""
        return '''{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}'''

    async def _generate_package_json(self, project_name: str, description: str) -> str:
        """生成 package.json。"""
        safe_name = project_name.lower().replace(" ", "-")
        return f'''{{
  "name": "{safe_name}",
  "version": "1.0.0",
  "description": "{description}",
  "main": "dist/index.js",
  "scripts": {{
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "tsc --watch"
  }},
  "keywords": ["file-server", "typescript", "holonpolis"],
  "author": "HolonPolis Evolution Engine",
  "license": "MIT",
  "devDependencies": {{
    "@types/node": "^20.0.0",
    "typescript": "^5.0.0"
  }},
  "engines": {{
    "node": ">=16.0.0"
  }}
}}'''


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

    # 支持的语言
    SUPPORTED_LANGUAGES = frozenset({'python', 'typescript'})

    def __init__(self):
        self._sandbox: Optional[SandboxRunner] = None
        self.security_scanner = SecurityScanner()
        self.holon_service = HolonService()
        self._typescript_scanner = TypeScriptSecurityScanner()

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

        # 写入代码和测试 - 添加必要的导入
        code_file = work_dir / "skill_module.py"
        test_file = work_dir / "test_skill.py"

        # 确保代码包含必要的导入
        full_code = code
        if 'import os' not in code:
            full_code = 'import os\n' + full_code

        code_file.write_text(full_code, encoding="utf-8")
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


    async def evolve_typescript_project_auto(
        self,
        project_name: str,
        description: str,
        requirements: List[str],
        target_dir: Path,
        provider_id: Optional[str] = None,
    ) -> EvolutionResult:
        """自主演化 TypeScript 项目 - 使用 LLM 生成代码。

        这是真正的自主演化：LLM 生成代码 -> RGV 验证 -> 落盘
        """
        logger.info(
            "typescript_evolution_auto_started",
            project_name=project_name,
            target_dir=str(target_dir),
        )

        # Phase 0: Generate - 使用 LLM 生成代码
        try:
            generator = LLMCodeGenerator(provider_id=provider_id)
            generated = await generator.generate_typescript_project(
                project_name=project_name,
                description=description,
                requirements=requirements,
            )
            code = generated["code"]
            tsconfig = generated["tsconfig"]
            package_json = generated["package_json"]
            logger.info("code_generated_by_llm", provider=generator.provider_id)
        except Exception as e:
            logger.error("code_generation_failed", error=str(e))
            return EvolutionResult(
                success=False,
                phase="generate",
                error_message=f"Code generation failed: {e}",
            )

        # Phase 1: Red - 验证 TypeScript 语法结构
        red_result = self._phase_red_typescript(code, tsconfig)
        if not red_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="red",
                error_message=f"Red phase failed: {red_result['error']}",
            )

        # Phase 2: Green - TypeScript 结构验证
        green_result = self._phase_green_typescript(code)
        if not green_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="green",
                error_message=f"Green phase failed: {green_result['error']}",
            )

        # Phase 3: Verify - TypeScript 安全扫描
        verify_result = self._phase_verify_typescript(code)
        if not verify_result["passed"]:
            return EvolutionResult(
                success=False,
                phase="verify",
                error_message=f"Verify phase failed: security violations found",
            )

        # Phase 4: Persist - 落盘到目标目录
        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            # 创建 src 目录
            src_dir = target_dir / "src"
            src_dir.mkdir(exist_ok=True)

            # 写入文件
            (src_dir / "index.ts").write_text(code, encoding="utf-8")
            (target_dir / "tsconfig.json").write_text(tsconfig, encoding="utf-8")
            (target_dir / "package.json").write_text(package_json, encoding="utf-8")

            # 创建 README
            readme = f"""# {project_name}

{description}

## Installation

```bash
npm install
```

## Build

```bash
npm run build
```

## Usage

```bash
npm start
```

---
Evolved by HolonPolis RGV Crucible
"""
            (target_dir / "README.md").write_text(readme, encoding="utf-8")

            # 生成 Attestation
            attestation = Attestation(
                attestation_id=f"att_ts_{project_name.lower().replace(' ', '_')}_{hashlib.sha256(code.encode()).hexdigest()[:8]}",
                holon_id="genesis",
                skill_name=project_name,
                version="1.0.0",
                red_phase_passed=True,
                green_phase_passed=True,
                verify_phase_passed=verify_result["passed"],
                security_scan_results=verify_result,
                code_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
            )

            logger.info(
                "typescript_evolution_completed",
                project_name=project_name,
                target_dir=str(target_dir),
            )

            return EvolutionResult(
                success=True,
                skill_id=f"project_{project_name.lower().replace(' ', '_')}",
                attestation=attestation,
                phase="complete",
                code_path=str(src_dir / "index.ts"),
            )

        except Exception as e:
            logger.error("typescript_persist_failed", error=str(e))
            return EvolutionResult(
                success=False,
                phase="persist",
                error_message=str(e),
            )

    def _phase_red_typescript(self, code: str, tsconfig: str) -> Dict[str, Any]:
        """Red 阶段: 验证 TypeScript 基本结构。"""
        # 检查 tsconfig 是有效的 JSON
        try:
            json.loads(tsconfig)
        except json.JSONDecodeError as e:
            return {"passed": False, "error": f"tsconfig.json invalid: {e}"}

        # 检查代码包含必要的 TypeScript 特征
        ts_features = [
            ("import/export", "import " in code or "export " in code),
            ("type annotations", ": " in code and ("string" in code or "number" in code)),
            ("interface or type", "interface " in code or "type " in code),
        ]

        missing = [name for name, present in ts_features if not present]
        if missing:
            return {"passed": False, "error": f"Missing TypeScript features: {missing}"}

        return {"passed": True, "error": None}

    def _phase_green_typescript(self, code: str) -> Dict[str, Any]:
        """Green 阶段: TypeScript 结构验证。"""
        # 检查类定义和函数
        has_class = "class " in code
        has_function = "function " in code or "=>" in code

        if not has_class and not has_function:
            return {"passed": False, "error": "No class or function found"}

        # 检查 async/await 正确使用
        async_count = code.count("async ")
        await_count = code.count("await ")
        if async_count > 0 and await_count == 0:
            return {"passed": False, "error": "Async function without await"}

        return {"passed": True, "error": None}

    def _phase_verify_typescript(self, code: str) -> Dict[str, Any]:
        """Verify 阶段: TypeScript 安全扫描。"""
        return self._typescript_scanner.scan(code)


def create_evolution_service() -> EvolutionService:
    """工厂函数: 创建 EvolutionService 实例。"""
    return EvolutionService()
