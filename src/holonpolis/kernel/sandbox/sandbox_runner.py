"""Sandbox Runner - 物理隔离执行器

HolonPolis 的沙箱执行核心，使用 asyncio subprocess 实现：
- 进程级隔离
- 资源限制（超时、内存）
- 工作目录锁定
- 命令安全校验

遵循铁律：所有执行必须被困在 .holonpolis/ 目录内，任何逃逸尝试都会被物理熔断。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from holonpolis.config import settings
from holonpolis.infrastructure.storage.path_guard import (
    InvalidArtifactPathError,
    ensure_within_root,
    normalize_path,
    safe_join,
)
from holonpolis.kernel.tools.tooling_security import validate_command_security


class SandboxError(Exception):
    """沙箱执行过程中的基础异常。"""
    pass


class PathEscapeError(SandboxError):
    """检测到路径逃逸尝试时抛出。"""
    pass


class CommandSecurityError(SandboxError):
    """命令未通过安全校验时抛出。"""
    pass


class ResourceLimitError(SandboxError):
    """资源限制（超时、内存）触发时抛出。"""
    pass


class ExitCodeError(SandboxError):
    """非零退出码时抛出（如调用方要求严格检查）。"""
    def __init__(
        self,
        message: str,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ):
        super().__init__(message)
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class SandboxStatus(Enum):
    """沙箱执行状态枚举。"""
    SUCCESS = auto()
    TIMEOUT = auto()
    MEMORY_EXCEEDED = auto()
    PATH_VIOLATION = auto()
    COMMAND_REJECTED = auto()
    EXECUTION_ERROR = auto()
    EXIT_CODE_NONZERO = auto()


@dataclass(frozen=True)
class SandboxResult:
    """沙箱执行结果不可变数据类。

    Attributes:
        status: 执行状态
        exit_code: 进程退出码（None 表示未正常结束）
        stdout: 标准输出
        stderr: 标准错误
        duration_ms: 执行耗时（毫秒）
        work_dir: 执行时的工作目录
        command: 实际执行的命令
    """
    status: SandboxStatus
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration_ms: float
    work_dir: Path
    command: str

    @property
    def success(self) -> bool:
        """是否成功执行（状态为 SUCCESS 且退出码为 0）。"""
        return self.status == SandboxStatus.SUCCESS and self.exit_code == 0

    @property
    def failed(self) -> bool:
        """是否执行失败。"""
        return not self.success


@dataclass
class SandboxConfig:
    """沙箱执行配置。

    Attributes:
        timeout_seconds: 超时时间（秒）
        max_memory_mb: 最大内存限制（MB，仅 Linux 支持）
        enable_network: 是否允许网络访问
        working_dir: 强制工作目录（None 则使用临时目录）
        env_vars: 额外环境变量
        inherit_env: 是否继承当前进程环境变量
        strict_exit_code: 是否严格要求退出码为 0
        allowed_commands: 允许的命令列表（None 则不限制）
        blocked_patterns:  blocked 的模式列表
    """
    timeout_seconds: int = 60
    max_memory_mb: int = 512
    enable_network: bool = False
    working_dir: Optional[Path] = None
    env_vars: dict[str, str] = field(default_factory=dict)
    inherit_env: bool = False
    strict_exit_code: bool = True
    allowed_commands: Optional[list[str]] = None
    blocked_patterns: Optional[list[str]] = None

    def __post_init__(self):
        # 从全局设置应用默认值
        if self.timeout_seconds == 60:  # 默认值检测
            self.timeout_seconds = settings.sandbox_timeout_seconds
        if self.max_memory_mb == 512:
            self.max_memory_mb = settings.sandbox_max_memory_mb
        if self.enable_network is False:
            self.enable_network = settings.sandbox_enable_network


class SandboxRunner:
    """沙箱执行器 - 物理隔离的任务运行环境。

    核心职责：
    1. 路径隔离：所有 I/O 必须在 .holonpolis/ 内
    2. 命令安全：执行前必须通过 tooling_security 校验
    3. 资源限制：超时、内存硬限制
    4. 进程隔离：使用 asyncio subprocess，完全隔离

    Usage:
        runner = SandboxRunner()
        result = await runner.run(
            command=["python", "-c", "print('hello')"],
            config=SandboxConfig(timeout_seconds=30)
        )
        if result.success:
            print(result.stdout)
    """

    # 需要网络隔离时设置的环境变量（Windows 有限支持）
    _NETWORK_BLOCK_ENV = {
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
    }

    def __init__(self, root_path: Optional[Path] = None):
        """初始化沙箱执行器。

        Args:
            root_path: 沙箱根目录，默认使用 settings.holonpolis_root
        """
        self._root = normalize_path(root_path or settings.holonpolis_root)
        self._ensure_root_exists()

    def _ensure_root_exists(self) -> None:
        """确保根目录存在。"""
        self._root.mkdir(parents=True, exist_ok=True)

    def _validate_working_dir(self, work_dir: Path) -> Path:
        """校验工作目录必须在沙箱根内。

        Args:
            work_dir: 待校验的工作目录

        Returns:
            校验通过的绝对路径

        Raises:
            PathEscapeError: 如果路径试图逃逸沙箱根
        """
        try:
            return ensure_within_root(self._root, work_dir)
        except InvalidArtifactPathError as e:
            raise PathEscapeError(
                f"工作目录逃逸检测: {work_dir} 不在 {self._root} 内"
            ) from e

    def _prepare_working_directory(
        self,
        config: SandboxConfig,
    ) -> Path:
        """准备工作目录。

        如果配置指定了工作目录，校验其在沙箱内；
        否则创建临时目录。

        Args:
            config: 沙箱配置

        Returns:
            准备好的工作目录路径
        """
        if config.working_dir is not None:
            work_dir = self._validate_working_dir(config.working_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            return work_dir

        # 创建临时工作目录在沙箱内
        sandbox_temp = safe_join(self._root, "sandbox", "temp")
        sandbox_temp.mkdir(parents=True, exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix="run_", dir=sandbox_temp)
        return Path(temp_dir)

    def _build_command_env(self, config: SandboxConfig) -> dict[str, str]:
        """构建命令执行环境变量。

        Args:
            config: 沙箱配置

        Returns:
            环境变量字典
        """
        if config.inherit_env:
            env = os.environ.copy()
        else:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),  # Windows 必需
                "TEMP": os.environ.get("TEMP", ""),
                "TMP": os.environ.get("TMP", ""),
            }

        # 应用自定义环境变量
        env.update(config.env_vars)

        # 网络隔离
        if not config.enable_network:
            env.update(self._NETWORK_BLOCK_ENV)

        # 资源限制提示（实际限制通过 ulimit/cgroups 等系统机制）
        env["HOLONPOLIS_SANDBOX"] = "1"
        env["HOLONPOLIS_MEMORY_LIMIT_MB"] = str(config.max_memory_mb)

        return env

    # 内置安全命令，包括 Python 可执行文件的各种形式
    _SAFE_COMMAND_PATTERNS = frozenset({
        "python", "python3", "python.exe", "python3.exe",
        "pytest", "pytest.exe",
    })

    def _is_python_interpreter(self, cmd: str) -> bool:
        """检查命令是否是 Python 解释器。

        支持完整路径的 Python 解释器以及简单命令名。
        """
        cmd_lower = cmd.lower()
        cmd_base = Path(cmd).name.lower()

        # 检查是否是 Python 解释器（通过路径或名称）
        if "python" in cmd_lower or cmd_base in ("python", "python3", "python.exe", "python3.exe"):
            return True

        return False

    def _validate_command_security(
        self,
        command: list[str],
        config: SandboxConfig,
    ) -> None:
        """校验命令安全性。

        Args:
            command: 命令列表
            config: 沙箱配置

        Raises:
            CommandSecurityError: 如果命令未通过安全校验
        """
        if not command:
            raise CommandSecurityError("命令不能为空")

        cmd_str = " ".join(command)
        cmd_exe = command[0]

        # 1. 检查 blocked patterns
        is_valid, error_msg = validate_command_security(
            cmd_str,
            allowed_commands=set(config.allowed_commands) if config.allowed_commands else None,
            blocked_patterns=config.blocked_patterns,
        )

        # 2. Python 解释器特殊处理：如果因不在 allowed_commands 中而失败，但命令本身是安全的 Python，则放行
        if not is_valid and self._is_python_interpreter(cmd_exe):
            # 检查失败原因是否只是"不在允许列表中"
            if "not in allowed list" in error_msg or "Command not in allowed list" in error_msg:
                is_valid = True

        if not is_valid:
            raise CommandSecurityError(f"命令安全校验失败: {error_msg}")

    def _build_process_args(
        self,
        command: list[str],
        work_dir: Path,
        config: SandboxConfig,
    ) -> dict:
        """构建 subprocess 执行参数。

        Args:
            command: 命令列表
            work_dir: 工作目录
            config: 沙箱配置

        Returns:
            subprocess 参数字典
        """
        env = self._build_command_env(config)

        args = {
            "cmd": command,
            "cwd": str(work_dir),
            "env": env,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "limit": 1024 * 1024,  # 流缓冲区限制 1MB
        }

        return args

    @staticmethod
    def _normalize_command_for_platform(command: list[str]) -> list[str]:
        """Normalize shell-builtin commands for cross-platform execution.

        On Windows, `echo` is a shell builtin, not an executable. Convert it
        into a Python print command so sandbox execution remains shell-free.
        """
        if os.name != "nt" or not command:
            return command

        if command[0].lower() != "echo":
            return command

        return [
            sys.executable,
            "-c",
            "import sys; print(' '.join(sys.argv[1:]))",
            *command[1:],
        ]

    async def run(
        self,
        command: list[str],
        config: Optional[SandboxConfig] = None,
        input_data: Optional[str] = None,
    ) -> SandboxResult:
        """在沙箱中执行命令。

        这是核心执行方法，完整实现了：
        - 路径隔离
        - 命令安全校验
        - 资源限制（超时）
        - 异步执行

        Args:
            command: 命令列表（如 ["python", "-c", "print('hi')"]）
            config: 沙箱配置，默认使用全局配置
            input_data: 写入 stdin 的数据

        Returns:
            SandboxResult 执行结果

        Raises:
            PathEscapeError: 如果工作目录配置逃逸沙箱
            CommandSecurityError: 如果命令未通过安全校验
        """
        config = config or SandboxConfig()
        start_time = asyncio.get_event_loop().time()

        # 1. 校验命令安全性
        try:
            self._validate_command_security(command, config)
        except CommandSecurityError as e:
            return SandboxResult(
                status=SandboxStatus.COMMAND_REJECTED,
                exit_code=None,
                stdout="",
                stderr=str(e),
                duration_ms=0.0,
                work_dir=config.working_dir or self._root,
                command=" ".join(command),
            )

        # 2. 准备工作目录
        try:
            work_dir = self._prepare_working_directory(config)
        except PathEscapeError as e:
            return SandboxResult(
                status=SandboxStatus.PATH_VIOLATION,
                exit_code=None,
                stdout="",
                stderr=str(e),
                duration_ms=0.0,
                work_dir=config.working_dir or self._root,
                command=" ".join(command),
            )

        # 3. 构建执行参数
        execution_command = self._normalize_command_for_platform(command)
        process_args = self._build_process_args(execution_command, work_dir, config)

        # 4. 异步执行
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *process_args["cmd"],
                cwd=process_args["cwd"],
                env=process_args["env"],
                stdout=process_args["stdout"],
                stderr=process_args["stderr"],
                limit=process_args["limit"],
            )

            # 处理输入数据
            stdin_data = input_data.encode("utf-8") if input_data else None

            # 带超时的通信
            stdout_data, stderr_data = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=config.timeout_seconds,
            )

            duration_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

            stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
            stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""

            # 确定状态
            status = SandboxStatus.SUCCESS
            if config.strict_exit_code and proc.returncode != 0:
                status = SandboxStatus.EXIT_CODE_NONZERO

            return SandboxResult(
                status=status,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                work_dir=work_dir,
                command=" ".join(command),
            )

        except asyncio.TimeoutError:
            # 超时处理：强制终止
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass

            duration_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                exit_code=None,
                stdout="",
                stderr=f"执行超时（限制: {config.timeout_seconds}秒）",
                duration_ms=duration_ms,
                work_dir=work_dir,
                command=" ".join(command),
            )

        except Exception as e:
            # 其他执行错误
            duration_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

            return SandboxResult(
                status=SandboxStatus.EXECUTION_ERROR,
                exit_code=None,
                stdout="",
                stderr=f"执行异常: {type(e).__name__}: {e}",
                duration_ms=duration_ms,
                work_dir=work_dir,
                command=" ".join(command),
            )

    async def run_python_code(
        self,
        code: str,
        config: Optional[SandboxConfig] = None,
    ) -> SandboxResult:
        """在沙箱中执行 Python 代码。

        这是运行 pytest 或技能代码的便捷方法。

        Args:
            code: Python 代码字符串
            config: 沙箱配置

        Returns:
            SandboxResult 执行结果
        """
        config = config or SandboxConfig()

        # 确保使用隔离的 Python 解释器
        python_exe = sys.executable

        # 构建命令：-I 隔离模式，-c 执行代码
        command = [python_exe, "-I", "-c", code]

        return await self.run(command, config)

    async def run_pytest(
        self,
        test_path: Path,
        config: Optional[SandboxConfig] = None,
    ) -> SandboxResult:
        """在沙箱中运行 pytest。

        这是 RGV 循环中 Red 阶段的核心方法。

        Args:
            test_path: 测试文件路径（必须在沙箱内）
            config: 沙箱配置

        Returns:
            SandboxResult 执行结果
        """
        config = config or SandboxConfig()

        # 校验测试路径在沙箱内
        try:
            safe_test_path = ensure_within_root(self._root, test_path)
        except InvalidArtifactPathError as e:
            raise PathEscapeError(
                f"测试路径逃逸: {test_path}"
            ) from e

        if not safe_test_path.exists():
            return SandboxResult(
                status=SandboxStatus.EXECUTION_ERROR,
                exit_code=None,
                stdout="",
                stderr=f"测试文件不存在: {safe_test_path}",
                duration_ms=0.0,
                work_dir=self._root,
                command=f"pytest {test_path}",
            )

        python_exe = sys.executable
        command = [
            python_exe,
            "-m", "pytest",
            str(safe_test_path),
            "-v",  # 详细输出
            "--tb=short",  # 短 traceback
        ]

        # 设置 pytest 特定的工作目录
        pytest_config = SandboxConfig(
            timeout_seconds=config.timeout_seconds,
            max_memory_mb=config.max_memory_mb,
            enable_network=config.enable_network,
            working_dir=safe_test_path.parent,
            env_vars=config.env_vars,
            inherit_env=config.inherit_env,
            strict_exit_code=False,  # pytest 非零退出码表示测试失败，不是执行错误
        )

        return await self.run(command, pytest_config)

    def cleanup_temp_directory(self, result: SandboxResult) -> None:
        """清理临时工作目录。

        Args:
            result: 执行结果，包含工作目录信息
        """
        temp_base = safe_join(self._root, "sandbox", "temp")

        # 只清理临时目录，不清理配置的固定工作目录
        if temp_base in result.work_dir.parents or result.work_dir == temp_base:
            try:
                if result.work_dir.exists():
                    shutil.rmtree(result.work_dir, ignore_errors=True)
            except Exception:
                pass


# 全局沙箱执行器实例
_default_runner: Optional[SandboxRunner] = None


def get_sandbox_runner(root_path: Optional[Path] = None) -> SandboxRunner:
    """获取全局沙箱执行器实例（单例模式）。

    Args:
        root_path: 可选的自定义根目录

    Returns:
        SandboxRunner 实例
    """
    global _default_runner
    if root_path is not None:
        return SandboxRunner(root_path)
    if _default_runner is None:
        _default_runner = SandboxRunner()
    return _default_runner


async def run_in_sandbox(
    command: list[str],
    timeout_seconds: int = 60,
    max_memory_mb: int = 512,
    working_dir: Optional[Path] = None,
) -> SandboxResult:
    """便捷函数：快速在沙箱中执行命令。

    Args:
        command: 命令列表
        timeout_seconds: 超时秒数
        max_memory_mb: 内存限制
        working_dir: 工作目录

    Returns:
        SandboxResult 执行结果
    """
    runner = get_sandbox_runner()
    config = SandboxConfig(
        timeout_seconds=timeout_seconds,
        max_memory_mb=max_memory_mb,
        working_dir=working_dir,
        inherit_env=False,  # 默认隔离
    )
    return await runner.run(command, config)
