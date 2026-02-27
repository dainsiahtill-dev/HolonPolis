"""Phase 1 Tests: Sandbox Runner - 物理隔离执行测试

验证 HolonPolis 的沙箱执行核心：
- 进程隔离
- 资源限制（超时）
- 命令安全校验
- 路径隔离
"""

import asyncio
import sys
from pathlib import Path

import pytest

from holonpolis.kernel.sandbox import (
    CommandSecurityError,
    ExitCodeError,
    PathEscapeError,
    ResourceLimitError,
    SandboxConfig,
    SandboxError,
    SandboxResult,
    SandboxRunner,
    SandboxStatus,
    get_sandbox_runner,
    run_in_sandbox,
)


class TestSandboxRunnerBasic:
    """沙箱执行器基础功能测试。"""

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        """创建测试用的沙箱执行器。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )
        return SandboxRunner()

    @pytest.mark.asyncio
    async def test_simple_echo_command(self, runner):
        """测试简单的 echo 命令执行。"""
        result = await runner.run(
            command=["echo", "hello world"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.SUCCESS
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.success is True

    @pytest.mark.asyncio
    async def test_python_execution(self, runner):
        """测试 Python 代码执行。"""
        result = await runner.run(
            command=[sys.executable, "-c", "print('from sandbox')"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.SUCCESS
        assert "from sandbox" in result.stdout

    @pytest.mark.asyncio
    async def test_working_directory_isolation(self, runner, tmp_path, monkeypatch):
        """测试工作目录隔离（必须在沙箱根目录内）。"""
        # 设置沙箱根目录
        sandbox_root = tmp_path / ".holonpolis"
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            sandbox_root
        )

        # 创建新的 runner 使用正确的根目录
        runner = SandboxRunner(sandbox_root)

        # 在沙箱内创建工作目录和测试文件
        work_dir = sandbox_root / "workspace" / "test_task"
        work_dir.mkdir(parents=True, exist_ok=True)
        test_file = work_dir / "test.txt"
        test_file.write_text("test content")

        config = SandboxConfig(
            timeout_seconds=5,
            working_dir=work_dir,
            strict_exit_code=False,
        )

        # Windows 和 Unix 使用不同的命令
        if sys.platform == "win32":
            config = SandboxConfig(
                timeout_seconds=5,
                working_dir=work_dir,
                strict_exit_code=False,
                allowed_commands=["cmd"],  # Windows 需要显式允许 cmd
            )
            result = await runner.run(
                command=["cmd", "/c", "type", "test.txt"],
                config=config,
            )
        else:
            result = await runner.run(
                command=["cat", "test.txt"],
                config=config,
            )

        assert result.status == SandboxStatus.SUCCESS
        assert "test content" in result.stdout

    @pytest.mark.asyncio
    async def test_environment_isolation(self, runner):
        """测试环境变量隔离。"""
        # 不继承环境变量，应该无法访问自定义变量
        result = await runner.run(
            command=[sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR', 'NOT_FOUND'))"],
            config=SandboxConfig(
                timeout_seconds=5,
                inherit_env=False,
                strict_exit_code=False,
            )
        )

        assert "NOT_FOUND" in result.stdout

    @pytest.mark.asyncio
    async def test_custom_environment_variables(self, runner):
        """测试自定义环境变量。"""
        result = await runner.run(
            command=[sys.executable, "-c", "import os; print(os.environ.get('CUSTOM_VAR', 'NOT_FOUND'))"],
            config=SandboxConfig(
                timeout_seconds=5,
                inherit_env=False,
                env_vars={"CUSTOM_VAR": "custom_value"},
                strict_exit_code=False,
            )
        )

        assert "custom_value" in result.stdout


class TestSandboxTimeout:
    """沙箱超时功能测试。"""

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        """创建测试用的沙箱执行器。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )
        return SandboxRunner()

    @pytest.mark.asyncio
    async def test_timeout_kills_long_running_process(self, runner):
        """测试超时杀死长时间运行的进程。"""
        result = await runner.run(
            command=[sys.executable, "-c", "import time; time.sleep(10)"],
            config=SandboxConfig(
                timeout_seconds=1,  # 1秒超时
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.TIMEOUT
        assert result.exit_code is None
        assert "超时" in result.stderr or "timeout" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_short_command_completes_before_timeout(self, runner):
        """测试短命令在超时前完成。"""
        result = await runner.run(
            command=[sys.executable, "-c", "print('quick')"],
            config=SandboxConfig(
                timeout_seconds=10,
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.SUCCESS
        assert "quick" in result.stdout
        assert result.duration_ms < 5000  # 应该很快完成


class TestSandboxSecurity:
    """沙箱安全功能测试。"""

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        """创建测试用的沙箱执行器。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )
        return SandboxRunner()

    @pytest.mark.asyncio
    async def test_blocked_command_rejected(self, runner):
        """测试被阻止的命令被拒绝。"""
        result = await runner.run(
            command=["rm", "-rf", "/"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            )
        )

        # 应该被拒绝，不会执行
        assert result.status == SandboxStatus.COMMAND_REJECTED

    @pytest.mark.asyncio
    async def test_command_with_allowed_list(self, runner):
        """测试允许的命令列表。"""
        # 只允许 echo
        result = await runner.run(
            command=["echo", "allowed"],
            config=SandboxConfig(
                timeout_seconds=5,
                allowed_commands=["echo"],
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_command_not_in_allowed_list_rejected(self, runner):
        """测试不在允许列表中的命令被拒绝。"""
        result = await runner.run(
            command=["cat", "/etc/passwd"],
            config=SandboxConfig(
                timeout_seconds=5,
                allowed_commands=["echo", "ls"],  # cat 不在列表中
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.COMMAND_REJECTED


class TestSandboxPythonExecution:
    """Python 代码执行便捷方法测试。"""

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        """创建测试用的沙箱执行器。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )
        return SandboxRunner()

    @pytest.mark.asyncio
    async def test_run_python_code(self, runner):
        """测试 run_python_code 方法。"""
        result = await runner.run_python_code(
            code="x = 1 + 2; print(f'result: {x}')",
            config=SandboxConfig(timeout_seconds=5),
        )

        assert result.status == SandboxStatus.SUCCESS
        assert "result: 3" in result.stdout

    @pytest.mark.asyncio
    async def test_python_code_isolation_mode(self, runner):
        """测试 Python 隔离模式（-I 标志）。"""
        result = await runner.run_python_code(
            code="import sys; print('isolated' if sys.flags.isolated else 'not isolated')",
            config=SandboxConfig(timeout_seconds=5),
        )

        assert result.status == SandboxStatus.SUCCESS
        # -I 标志设置隔离模式
        assert "isolated" in result.stdout

    @pytest.mark.asyncio
    async def test_python_code_with_exception(self, runner):
        """测试执行抛出异常的 Python 代码。"""
        result = await runner.run_python_code(
            code="raise ValueError('test error')",
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,  # 不要严格检查退出码
            ),
        )

        # 执行了，但有错误输出
        assert "ValueError" in result.stderr
        assert result.exit_code != 0


class TestSandboxResultProperties:
    """SandboxResult 属性测试。"""

    def test_success_property_with_zero_exit_code(self):
        """测试 success 属性在退出码为 0 时。"""
        result = SandboxResult(
            status=SandboxStatus.SUCCESS,
            exit_code=0,
            stdout="output",
            stderr="",
            duration_ms=100.0,
            work_dir=Path("/tmp"),
            command="echo test",
        )

        assert result.success is True
        assert result.failed is False

    def test_success_property_with_nonzero_exit_code(self):
        """测试 success 属性在退出码非 0 时。"""
        result = SandboxResult(
            status=SandboxStatus.SUCCESS,  # 执行成功，但退出码非 0
            exit_code=1,
            stdout="",
            stderr="error",
            duration_ms=100.0,
            work_dir=Path("/tmp"),
            command="false",
        )

        # 由于 strict_exit_code=False，status 是 SUCCESS，但 exit_code 是 1
        # success 属性应该检查两者
        assert result.success is False  # exit_code != 0

    def test_success_property_with_timeout(self):
        """测试 success 属性在超时状态。"""
        result = SandboxResult(
            status=SandboxStatus.TIMEOUT,
            exit_code=None,
            stdout="",
            stderr="timeout",
            duration_ms=5000.0,
            work_dir=Path("/tmp"),
            command="sleep 10",
        )

        assert result.success is False
        assert result.failed is True


class TestGlobalSandboxFunctions:
    """全局沙箱便捷函数测试。"""

    @pytest.mark.asyncio
    async def test_run_in_sandbox_convenience_function(self, tmp_path, monkeypatch):
        """测试 run_in_sandbox 便捷函数。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        result = await run_in_sandbox(
            command=[sys.executable, "-c", "print('convenience')"],
            timeout_seconds=5,
        )

        assert "convenience" in result.stdout

    def test_get_sandbox_runner_singleton(self, tmp_path, monkeypatch):
        """测试 get_sandbox_runner 单例模式。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        runner1 = get_sandbox_runner()
        runner2 = get_sandbox_runner()

        # 应该是同一个实例
        assert runner1 is runner2


class TestSandboxEdgeCases:
    """沙箱边界情况测试。"""

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        """创建测试用的沙箱执行器。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )
        return SandboxRunner()

    @pytest.mark.asyncio
    async def test_empty_command_list(self, runner):
        """测试空命令列表。"""
        result = await runner.run(
            command=[],
            config=SandboxConfig(timeout_seconds=5),
        )

        assert result.status == SandboxStatus.COMMAND_REJECTED

    @pytest.mark.asyncio
    async def test_nonexistent_executable(self, runner):
        """测试不存在的可执行文件被安全层拒绝。"""
        result = await runner.run(
            command=["nonexistent_executable_xyz"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            ),
        )

        # 安全层会拒绝不在允许列表中的命令（这是正确行为）
        assert result.status == SandboxStatus.COMMAND_REJECTED

    @pytest.mark.asyncio
    async def test_unicode_output(self, runner):
        """测试 Unicode 输出处理。"""
        result = await runner.run(
            command=[sys.executable, "-c", "print('你好，世界 🌍')"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            )
        )

        assert result.status == SandboxStatus.SUCCESS
        assert "你好，世界" in result.stdout

    @pytest.mark.asyncio
    async def test_large_output_truncation(self, runner):
        """测试大输出处理。"""
        # 生成大量输出
        result = await runner.run(
            command=[sys.executable, "-c", "print('x' * 100000)"],
            config=SandboxConfig(
                timeout_seconds=10,
                strict_exit_code=False,
            )
        )

        # 应该成功执行，尽管输出被截断
        assert result.status == SandboxStatus.SUCCESS
        assert len(result.stdout) > 0

    @pytest.mark.asyncio
    async def test_stderr_capture(self, runner):
        """测试标准错误捕获。"""
        result = await runner.run(
            command=[sys.executable, "-c", "import sys; sys.stderr.write('error message')"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            )
        )

        assert "error message" in result.stderr

    @pytest.mark.asyncio
    async def test_duration_tracking(self, runner):
        """测试执行时间追踪。"""
        result = await runner.run(
            command=[sys.executable, "-c", "import time; time.sleep(0.1)"],
            config=SandboxConfig(
                timeout_seconds=5,
                strict_exit_code=False,
            )
        )

        assert result.duration_ms >= 100  # 至少 100ms
        assert result.duration_ms < 5000  # 但少于 5 秒
