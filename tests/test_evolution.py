"""Phase 4 Tests: RGV Crucible - 演化裁判所测试

验证:
1. Red 阶段: 测试代码语法检查
2. Green 阶段: Sandbox 中运行 pytest
3. Verify 阶段: AST 安全扫描
4. Persist 阶段: 落盘到 skills_local
"""

import tempfile
from pathlib import Path

import pytest

from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.skills import ToolSchema
from holonpolis.kernel.embeddings.default_embedder import set_embedder, SimpleEmbedder
from holonpolis.kernel.llm.llm_runtime import LLMResponse
from holonpolis.kernel.lancedb import get_lancedb_factory, reset_factory
from holonpolis.kernel.storage import HolonPathGuard
from holonpolis.services.evolution_service import (
    Attestation,
    EvolutionResult,
    EvolutionService,
    SecurityScanner,
)
from holonpolis.services.holon_service import HolonService


@pytest.fixture
def temp_embedder():
    """使用 SimpleEmbedder 避免 API 调用."""
    embedder = SimpleEmbedder(dimension=128)
    set_embedder(embedder)
    return embedder


@pytest.fixture
def evolution_setup(tmp_path, temp_embedder, monkeypatch):
    """设置演化测试环境."""
    holonpolis_root = tmp_path / ".holonpolis"
    monkeypatch.setattr(
        "holonpolis.config.settings.holonpolis_root",
        holonpolis_root
    )
    monkeypatch.setattr(
        "holonpolis.config.settings.holons_path",
        holonpolis_root / "holons"
    )
    reset_factory()

    return tmp_path


class TestSecurityScanner:
    """AST 安全扫描测试."""

    @pytest.fixture
    def scanner(self):
        """创建 SecurityScanner 实例."""
        return SecurityScanner()

    def test_safe_code_passes(self, scanner):
        """测试安全代码通过扫描."""
        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
'''
        result = scanner.scan(code)

        assert result["passed"] is True
        assert len(result["violations"]) == 0
        assert result["complexity_score"] >= 0

    def test_dangerous_import_detected(self, scanner):
        """测试检测到危险导入 (subprocess)."""
        code = '''
from subprocess import call
call(["rm", "-rf", "/"])
'''
        result = scanner.scan(code)

        assert result["passed"] is False
        assert any(v["type"] == "dangerous_import" for v in result["violations"])

    def test_dangerous_call_detected(self, scanner):
        """测试检测到危险函数调用."""
        code = '''
user_input = input("Enter code: ")
result = eval(user_input)
'''
        result = scanner.scan(code)

        assert result["passed"] is False
        assert any(v["type"] == "dangerous_call" for v in result["violations"])

    def test_sensitive_attribute_detected(self, scanner):
        """测试检测到敏感属性访问."""
        code = '''
obj = some_object
secret = obj.__globals__
'''
        result = scanner.scan(code)

        assert result["passed"] is False
        assert any(v["type"] == "sensitive_attribute" for v in result["violations"])

    def test_syntax_error_reported(self, scanner):
        """测试语法错误被报告."""
        code = '''
def broken_function(
    # Missing closing parenthesis and colon
    pass
'''
        result = scanner.scan(code)

        assert result["passed"] is False
        assert any(v["type"] == "syntax_error" for v in result["violations"])

    def test_complexity_estimation(self, scanner):
        """测试复杂度估计."""
        code = '''
def complex_function(n):
    result = 0
    for i in range(n):
        if i % 2 == 0:
            for j in range(i):
                if j > 5 and j < 10:
                    result += j
    return result
'''
        result = scanner.scan(code)

        assert result["complexity_score"] > 0
        # 应该检测到有决策点
        assert result["complexity_score"] > 0.5


class TestRedPhase:
    """Red 阶段测试."""

    @pytest.fixture
    def service(self):
        """创建 EvolutionService 实例."""
        return EvolutionService()

    @pytest.mark.asyncio
    async def test_valid_test_code_passes(self, service):
        """测试有效的测试代码通过 Red 阶段."""
        tests = '''
import pytest

def test_addition():
    assert 1 + 1 == 2
'''
        result = await service._phase_red(tests)

        assert result["passed"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_invalid_syntax_fails(self, service):
        """测试无效语法在 Red 阶段失败."""
        tests = '''
def broken_test(
    # Missing closing parenthesis
    pass
'''
        result = await service._phase_red(tests)

        assert result["passed"] is False
        assert "syntax error" in result["error"].lower()


@pytest.mark.asyncio
async def test_evolution_service_marks_holon_pending_during_run(evolution_setup, monkeypatch):
    """Holons should be pending while evolution is executing and active afterwards."""
    holon_id = "pending_guard_holon"
    await HolonService().create_holon(
        Blueprint(
            blueprint_id=f"bp_{holon_id}",
            holon_id=holon_id,
            species_id="generalist",
            name="Pending Guard",
            purpose="Verify pending during evolution",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(),
        )
    )

    observed: dict[str, str] = {}

    async def fake_phase_red(self, tests):
        observed["status_during_red"] = HolonService().get_holon_status(holon_id)
        return {"passed": False, "error": "stop_after_status_check"}

    monkeypatch.setattr(EvolutionService, "_phase_red", fake_phase_red)

    service = EvolutionService()
    result = await service.evolve_skill(
        holon_id=holon_id,
        skill_name="PendingSkill",
        code="def execute():\n    return 1\n",
        tests="def test_execute():\n    assert True\n",
        description="status transition test",
        tool_schema=ToolSchema(
            name="execute",
            description="No-op",
            parameters={"type": "object", "properties": {}},
        ),
    )

    assert result.success is False
    assert result.phase == "red"
    assert observed["status_during_red"] == "pending"
    assert HolonService().get_holon_status(holon_id) == "active"


@pytest.mark.asyncio
async def test_evolution_service_persists_llm_audit_state(evolution_setup, monkeypatch):
    """Successful LLM requests should be visible in state.json even after pending clears."""
    holon_id = "llm_audit_holon"
    await HolonService().create_holon(
        Blueprint(
            blueprint_id=f"bp_{holon_id}",
            holon_id=holon_id,
            species_id="generalist",
            name="LLM Audit",
            purpose="Verify LLM audit persistence",
            boundary=Boundary(),
            evolution_policy=EvolutionPolicy(),
        )
    )

    service = EvolutionService()
    llm_calls = {"count": 0}

    async def fake_chat(**kwargs):
        llm_calls["count"] += 1
        if llm_calls["count"] == 1:
            return LLMResponse(
                content=(
                    "from skill_module import execute\n\n"
                    "def test_execute():\n"
                    "    assert execute({\"value\": 1}) is not None\n"
                ),
                model="audit-test-model",
                latency_ms=11,
            )
        return LLMResponse(
            content=(
                "def execute(payload=None):\n"
                "    return {\"ok\": True, \"payload\": payload or {}}\n"
            ),
            model="audit-test-model",
            latency_ms=17,
        )

    async def fake_evolve_skill(**kwargs):
        return EvolutionResult(success=False, phase="green", error_message="stop_after_audit")

    monkeypatch.setattr(service._llm, "chat", fake_chat)
    monkeypatch.setattr(service, "evolve_skill", fake_evolve_skill)

    result = await service.evolve_skill_autonomous(
        holon_id=holon_id,
        skill_name="AuditSkill",
        description="Check LLM audit fields",
        requirements=["Return a deterministic payload"],
        tool_schema=ToolSchema(
            name="execute",
            description="Audit-only",
            parameters={"type": "object", "properties": {}},
        ),
        max_attempts=1,
        pending_token="audit-request-001",
    )

    state = HolonService().get_holon_state(holon_id)
    audit = state.get("evolution_audit")
    assert isinstance(audit, dict)
    llm = audit.get("llm")
    assert isinstance(llm, dict)
    assert result.success is False
    assert llm_calls["count"] == 2
    assert state["status"] == "active"
    assert audit["request_id"] == "audit-request-001"
    assert audit["lifecycle"] == "completed"
    assert audit["phase"] == "generate_code"
    assert audit["result"] == "in_progress"
    assert llm["requested"] is True
    assert llm["inflight"] is False
    assert llm["call_count"] == 2
    assert llm["success_count"] == 2
    assert llm["failure_count"] == 0
    assert llm["last_status"] == "succeeded"
    assert llm["last_stage"] == "generate_code"
    assert llm["model"] == "audit-test-model"


class TestGreenPhase:
    """Green 阶段测试."""

    @pytest.fixture
    def service(self):
        """创建 EvolutionService 实例."""
        return EvolutionService()

    @pytest.mark.asyncio
    async def test_code_passing_tests(self, service, evolution_setup):
        """测试代码通过 pytest."""
        holon_id = "green_test_holon"
        # 先创建 Holon 目录
        guard = HolonPathGuard(holon_id)
        guard.ensure_directory("temp")

        code = '''
def add(a, b):
    return a + b
'''
        tests = '''
from skill_module import add

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
'''
        result = await service._phase_green(code, tests, holon_id)

        if not result["passed"]:
            print(f"Green phase failed: {result.get('error')}")
            print(f"Details stdout: {result.get('details', {}).get('stdout')}")
            print(f"Details stderr: {result.get('details', {}).get('stderr')}")
        assert result["passed"] is True
        assert result["details"]["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_code_failing_tests(self, service, evolution_setup):
        """测试代码失败 pytest."""
        holon_id = "green_test_holon_fail"
        guard = HolonPathGuard(holon_id)
        guard.ensure_directory("temp")

        code = '''
def add(a, b):
    return a - b  # Bug: subtraction instead of addition
'''
        tests = '''
from skill_module import add

def test_add():
    assert add(1, 2) == 3
'''
        result = await service._phase_green(code, tests, holon_id)

        assert result["passed"] is False
        assert result["details"]["exit_code"] != 0


class TestVerifyPhase:
    """Verify 阶段测试."""

    @pytest.fixture
    def service(self):
        """创建 EvolutionService 实例."""
        return EvolutionService()

    def test_safe_code_passes_verify(self, service):
        """测试安全代码通过 Verify."""
        code = '''
def helper(data):
    return sorted(data)
'''
        result = service._phase_verify(code)

        assert result["passed"] is True
        assert len(result["violations"]) == 0

    def test_unsafe_code_fails_verify(self, service):
        """测试不安全代码失败 Verify."""
        code = '''
import os

def dangerous(path):
    os.system(f"rm -rf {path}")
'''
        result = service._phase_verify(code)

        assert result["passed"] is False
        assert len(result["violations"]) > 0


class TestPersistPhase:
    """Persist 阶段测试."""

    @pytest.fixture
    def service(self):
        """创建 EvolutionService 实例."""
        return EvolutionService()

    @pytest.fixture
    def tool_schema(self):
        """创建测试用的 ToolSchema."""
        return ToolSchema(
            name="test_skill",
            description="A test skill",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            },
            required=["input"],
        )

    @pytest.mark.asyncio
    async def test_skill_persisted_to_disk(self, evolution_setup, service, tool_schema):
        """测试技能被持久化到磁盘."""
        holon_id = "test_holon_001"

        # 准备测试数据
        code = '''
def process(input_data):
    return input_data.upper()
'''
        tests = '''
from skill_module import process

def test_process():
    assert process("hello") == "HELLO"
'''

        green_result = {
            "passed": True,
            "details": {"exit_code": 0, "duration_ms": 100},
        }
        verify_result = {"passed": True, "violations": [], "complexity": 1.0}

        result = await service._phase_persist(
            holon_id=holon_id,
            skill_name="UpperCase Skill",
            code=code,
            tests=tests,
            description="Convert text to uppercase",
            tool_schema=tool_schema,
            version="0.1.0",
            green_result=green_result,
            verify_result=verify_result,
        )

        assert result["success"] is True
        assert result["skill_id"] is not None
        assert result["attestation"] is not None
        assert Path(result["code_path"]).exists()
        assert Path(result["test_path"]).exists()
        assert Path(result["manifest_path"]).exists()

        # 验证 Attestation 结构
        att = result["attestation"]
        assert att.red_phase_passed is True
        assert att.green_phase_passed is True
        assert att.verify_phase_passed is True
        assert att.code_hash is not None
        assert att.test_hash is not None

    @pytest.mark.asyncio
    async def test_attestation_file_created(self, evolution_setup, service, tool_schema):
        """测试 Attestation 文件被创建."""
        holon_id = "test_holon_002"

        code = "def func(): pass"
        tests = "def test_func(): pass"

        result = await service._phase_persist(
            holon_id=holon_id,
            skill_name="Test Skill",
            code=code,
            tests=tests,
            description="Test",
            tool_schema=tool_schema,
            version="1.0.0",
            green_result={"passed": True, "details": {}},
            verify_result={"passed": True, "violations": []},
        )

        # 检查 attestation.json 文件存在
        skill_dir = Path(result["code_path"]).parent
        att_file = skill_dir / "attestation.json"
        assert att_file.exists()


class TestFullRGVWorkflow:
    """完整 RGV 工作流测试."""

    @pytest.fixture
    def service(self):
        """创建 EvolutionService 实例."""
        return EvolutionService()

    @pytest.mark.asyncio
    async def test_successful_evolution(self, evolution_setup, service, tool_schema):
        """测试成功的技能演化."""
        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b
'''
        tests = '''
from skill_module import add, multiply

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
'''

        result = await service.evolve_skill(
            holon_id="evo_holon_001",
            skill_name="Calculator",
            code=code,
            tests=tests,
            description="Basic calculator operations",
            tool_schema=tool_schema,
            version="0.1.0",
        )

        assert result.success is True
        assert result.phase == "complete"
        assert result.attestation is not None
        assert result.skill_id is not None

    @pytest.mark.asyncio
    async def test_evolution_fails_at_verify(self, evolution_setup, service, tool_schema):
        """测试演化在 Verify 阶段失败."""
        # 使用安全的测试代码，但危险的功能代码
        code = '''
import os

def dangerous_delete(path):
    """Dangerous function using os.system."""
    os.system(f"rm -rf {path}")
    return True
'''
        # 测试验证函数契约，避免在 Red 基线实现下误通过
        tests = '''
import skill_module

def test_function_contract():
    fn = getattr(skill_module, "dangerous_delete")
    assert callable(fn)
    assert fn.__name__ == "dangerous_delete"
'''

        result = await service.evolve_skill(
            holon_id="evo_holon_002",
            skill_name="DangerousSkill",
            code=code,
            tests=tests,
            description="Should fail security scan",
            tool_schema=tool_schema,
            version="0.1.0",
        )

        assert result.success is False
        assert result.phase == "verify"

    @pytest.mark.asyncio
    async def test_evolution_fails_at_green(self, evolution_setup, service, tool_schema):
        """测试演化在 Green 阶段失败."""
        code = '''
def broken_add(a, b):
    """Broken addition."""
    return a - b  # Bug!
'''
        tests = '''
from skill_module import broken_add

def test_add():
    assert broken_add(2, 3) == 5  # This will fail
'''

        result = await service.evolve_skill(
            holon_id="evo_holon_003",
            skill_name="BrokenSkill",
            code=code,
            tests=tests,
            description="Has a bug",
            tool_schema=tool_schema,
            version="0.1.0",
        )

        assert result.success is False
        assert result.phase == "green"

    @pytest.mark.asyncio
    async def test_evolution_fails_when_tests_do_not_define_behavior(
        self,
        evolution_setup,
        service,
        tool_schema,
    ):
        """测试弱测试用例在 Red 语义检查阶段被拒绝."""
        code = '''
def execute():
    return "ok"
'''
        tests = '''
import skill_module

def test_callable_exists_only():
    assert callable(skill_module.execute)
'''
        result = await service.evolve_skill(
            holon_id="evo_holon_004",
            skill_name="WeakTestSkill",
            code=code,
            tests=tests,
            description="Should fail at red due to weak tests",
            tool_schema=tool_schema,
            version="0.1.0",
        )

        assert result.success is False
        assert result.phase == "red"
        assert "too weak" in (result.error_message or "")


class TestEvolutionFromTestCases:
    """给定语义测试用例的演化测试."""

    @pytest.mark.asyncio
    async def test_evolve_skill_with_explicit_test_cases(self, evolution_setup, tool_schema, monkeypatch):
        service = EvolutionService()

        async def fake_generate_code(
            holon_id,
            skill_name,
            description,
            requirements,
            tests,
            pending_token=None,
        ):
            assert "test_case_1" in tests
            return """
def execute(a, b):
    return a + b
"""

        monkeypatch.setattr(service, "_generate_code_via_llm", fake_generate_code)

        result = await service.evolve_skill_with_test_cases(
            holon_id="case_holon_001",
            skill_name="CaseAdder",
            description="Add two numbers from explicit test cases",
            requirements=["Support numeric addition"],
            test_cases=[
                {
                    "description": "add integers",
                    "function": "execute",
                    "input": {"a": 2, "b": 3},
                    "expected": 5,
                }
            ],
            tool_schema=tool_schema,
            version="0.1.0",
        )

        assert result.success is True
        assert result.phase == "complete"
        assert result.skill_id == "caseadder"

    @pytest.mark.asyncio
    async def test_evolve_skill_autonomous_repairs_after_green_failure(
        self,
        evolution_setup,
        tool_schema,
        monkeypatch,
    ):
        service = EvolutionService()

        async def fake_generate_tests(
            holon_id,
            skill_name,
            description,
            requirements,
            pending_token=None,
        ):
            return """
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
"""

        async def fake_generate_code(
            holon_id,
            skill_name,
            description,
            requirements,
            tests,
            pending_token=None,
        ):
            return """
def execute(a, b):
    return a - b
"""

        async def fake_repair(
            holon_id,
            skill_name,
            description,
            requirements,
            tests,
            previous_code,
            failure_phase,
            failure_message,
            pending_token=None,
        ):
            assert failure_phase == "green"
            assert "pytest failed" in failure_message
            assert "return a - b" in previous_code
            return """
def execute(a, b):
    return a + b
"""

        monkeypatch.setattr(service, "_generate_tests_via_llm", fake_generate_tests)
        monkeypatch.setattr(service, "_generate_code_via_llm", fake_generate_code)
        monkeypatch.setattr(service, "_repair_code_via_llm", fake_repair)

        result = await service.evolve_skill_autonomous(
            holon_id="auto_retry_holon",
            skill_name="AutoRetryAdder",
            description="Add two numbers reliably",
            requirements=["Accept numbers a and b", "Return a + b"],
            tool_schema=tool_schema,
            max_attempts=2,
        )

        assert result.success is True
        assert result.phase == "complete"
        assert result.skill_id == "autoretryadder"

    @pytest.mark.asyncio
    async def test_evolve_skill_autonomous_repairs_invalid_tests_before_code_generation(
        self,
        evolution_setup,
        tool_schema,
        monkeypatch,
    ):
        service = EvolutionService()
        call_state = {"tests_generated": 0, "tests_repaired": 0}

        async def fake_generate_tests(
            holon_id,
            skill_name,
            description,
            requirements,
            pending_token=None,
        ):
            call_state["tests_generated"] += 1
            return """
def execute(a, b):
    return a + b

def test_weak():
    assert execute(2, 3) == 5
"""

        async def fake_repair_tests(
            holon_id,
            skill_name,
            description,
            requirements,
            previous_tests,
            failure_message,
            pending_token=None,
        ):
            call_state["tests_repaired"] += 1
            assert "import from skill_module" in failure_message
            return """
from skill_module import execute

def test_execute():
    assert execute(2, 3) == 5
"""

        async def fake_generate_code(
            holon_id,
            skill_name,
            description,
            requirements,
            tests,
            pending_token=None,
        ):
            assert "from skill_module import execute" in tests
            return """
def execute(a, b):
    return a + b
"""

        monkeypatch.setattr(service, "_generate_tests_via_llm", fake_generate_tests)
        monkeypatch.setattr(service, "_repair_tests_via_llm", fake_repair_tests)
        monkeypatch.setattr(service, "_generate_code_via_llm", fake_generate_code)

        result = await service.evolve_skill_autonomous(
            holon_id="auto_fix_tests_holon",
            skill_name="AutoFixTestsAdder",
            description="Add two numbers with valid tests",
            requirements=["Accept numbers a and b", "Return a + b"],
            tool_schema=tool_schema,
            max_attempts=2,
        )

        assert result.success is True
        assert result.phase == "complete"
        assert call_state["tests_generated"] == 1
        assert call_state["tests_repaired"] == 1


class TestProjectContractTests:
    """项目契约测试模板保障."""

    def test_project_contract_template_includes_runtime_guards(self):
        requirements = [
            "execute(...) must return a dict with keys: project_name, project_slug, files, run_instructions.",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
            "Required output file paths: README.md, package.json",
        ]

        template = EvolutionService._build_project_contract_tests(requirements)

        assert "test_server_third_party_imports_declared_in_package" in template
        assert 'assert "ws" in dependencies or "ws" in dev_dependencies' in template
        assert 'assert "process.env.port" in server' in template
        assert 'assert "/ws" in server' in template
        assert 'assert "process.env.port" in smoke' in template
        assert "assert \".on('/healthz'\" not in server" in template
        assert "MIN_FILE_COUNT" in template
        assert "test_skill_source_avoids_multiline_fstring_templates" in template
        assert "test_execute_signature_contract" in template
        assert "test_skill_source_avoids_format_template_brace_risk" in template
        assert "test_no_placeholder_tokens" in template
        assert "test_domain_modules_present_for_large_game" in template
        assert "test_gameplay_semantics_present" in template
        assert "test_gameplay_modules_have_substantive_logic" in template

    def test_project_contract_template_encodes_goal_keywords_without_preset_business_logic(self):
        requirements = [
            "execute(...) must return a dict with keys: project_name, project_slug, files, run_instructions.",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
            "Project goal: Build a large multiplayer alchemy guild simulation with faction diplomacy.",
        ]
        template = EvolutionService._build_project_contract_tests(requirements)
        lowered = template.lower()
        assert "project_goal =" in lowered
        assert "goal_keywords =" in lowered
        assert "alchemy" in lowered
        assert "diplomacy" in lowered

    def test_project_contract_template_passes_contract_validation(self):
        requirements = [
            "execute(...) must return a dict with keys: project_name, project_slug, files, run_instructions.",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
            "Required output file paths: README.md, package.json",
        ]

        template = EvolutionService._build_project_contract_tests(requirements)
        validation = EvolutionService._validate_generated_tests_contract(
            template,
            requirements=requirements,
        )
        assert validation["passed"] is True

    def test_project_min_file_count_infers_large_project_floor(self):
        requirements = [
            "Project goal: 构建大型多人在线 WebSocket MMO",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
        ]
        assert EvolutionService._infer_project_min_file_count(requirements) >= 12

    def test_project_generation_rules_include_runtime_constraints(self):
        requirements = [
            "execute(...) must return a dict with keys: project_name, project_slug, files, run_instructions.",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
            "Project goal: Build large multiplayer MMO game with websocket runtime.",
        ]
        rules = EvolutionService._project_generation_rules(requirements)
        merged = "\n".join(rules).lower()
        assert "at least" in merged and "files" in merged
        assert "/healthz" in merged and "/ws" in merged
        assert "new websocketserver" in merged or "new server" in merged
        assert "process.exit(0)" in merged

    def test_failure_repair_hints_cover_common_project_failures(self):
        requirements = [
            "execute(...) must return a dict with keys: project_name, project_slug, files, run_instructions.",
            "files must be Dict[str, str] mapping relative file paths to UTF-8 text content.",
            "Project goal: Build large multiplayer MMO game with websocket runtime.",
        ]
        failure_message = """
pytest failed
test_skill.py::test_no_placeholder_tokens FAILED
AssertionError: placeholder token in apps/client/src/render.js: render logic
test_skill.py::test_file_count_meets_scale_floor FAILED
assert len(files) >= 18
test_skill.py::test_server_contains_health_and_ws_reply_logic FAILED
assert '/ws' in server
test_skill.py::test_ws_smoke_has_timeout_and_failure_exit FAILED
assert "process.exit(0)" in smoke_raw
E   KeyError: '\\n  "name"'
"""
        hints = EvolutionService._derive_failure_repair_hints(
            failure_message=failure_message,
            requirements=requirements,
        )
        merged = "\n".join(hints).lower()
        assert "placeholder" in merged
        assert "at least 18" in merged
        assert "/ws" in merged
        assert "process.exit(0)" in merged
        assert ".format" in merged or "f-string" in merged


class TestAttestation:
    """Attestation 测试."""

    def test_attestation_creation(self):
        """测试 Attestation 创建."""
        att = Attestation(
            attestation_id="att_001",
            holon_id="holon_001",
            skill_name="TestSkill",
            version="1.0.0",
            red_phase_passed=True,
            green_phase_passed=True,
            verify_phase_passed=True,
            code_hash="abc123",
            test_hash="def456",
        )

        assert att.attestation_id == "att_001"
        assert att.red_phase_passed is True

    def test_attestation_to_dict(self):
        """测试 Attestation 序列化."""
        att = Attestation(
            attestation_id="att_002",
            holon_id="holon_002",
            skill_name="AnotherSkill",
            version="0.5.0",
            red_phase_passed=True,
            green_phase_passed=True,
            verify_phase_passed=True,
        )

        data = att.to_dict()

        assert data["attestation_id"] == "att_002"
        assert data["red_phase_passed"] is True
        assert data["code_hash"] == att.code_hash


class TestEvolutionResult:
    """EvolutionResult 测试."""

    def test_success_result(self):
        """测试成功结果."""
        result = EvolutionResult(
            success=True,
            skill_id="skill_001",
            phase="complete",
            code_path="/path/to/code.py",
        )

        assert result.success is True
        assert result.skill_id == "skill_001"

    def test_failure_result(self):
        """测试失败结果."""
        result = EvolutionResult(
            success=False,
            phase="verify",
            error_message="Security violation found",
        )

        assert result.success is False
        assert result.phase == "verify"
        assert "Security violation" in result.error_message


@pytest.fixture
def tool_schema():
    """创建测试用的 ToolSchema."""
    return ToolSchema(
        name="calculator",
        description="A calculator skill",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            }
        },
        required=["a", "b"],
    )


class TestValidateExistingSkill:
    """验证已有技能测试."""

    @pytest.mark.asyncio
    async def test_validate_missing_skill(self, evolution_setup):
        """测试验证不存在的技能."""
        service = EvolutionService()

        result = await service.validate_existing_skill(
            holon_id="missing_holon",
            skill_name="missing_skill",
        )

        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_validate_existing_skill(self, evolution_setup, tool_schema):
        """测试验证已存在的技能."""
        service = EvolutionService()
        holon_id = "validate_holon"

        # 先创建一个技能
        code = '''
def helper(x):
    return x * 2
'''
        tests = '''
from skill_module import helper

def test_helper():
    assert helper(5) == 10
'''

        await service._phase_persist(
            holon_id=holon_id,
            skill_name="HelperSkill",
            code=code,
            tests=tests,
            description="A helper skill",
            tool_schema=tool_schema,
            version="1.0.0",
            green_result={"passed": True, "details": {}},
            verify_result={"passed": True, "violations": []},
        )

        # 然后验证它
        result = await service.validate_existing_skill(
            holon_id=holon_id,
            skill_name="HelperSkill",
        )

        assert result["valid"] is True
        assert result["green_passed"] is True
        assert result["verify_passed"] is True
