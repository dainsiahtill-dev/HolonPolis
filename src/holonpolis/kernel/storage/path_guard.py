"""Kernel Path Guard - 内核级路径熔断器

这是 HolonPolis 的"物理法则"实现之一：绝对路径隔离。
所有试图逃离 .holonpolis/ 目录的 I/O 请求都将被物理熔断。

与 infrastructure/storage/path_guard 的区别：
- 基础设施层：通用路径工具
- 内核层：物理熔断器，专为 Holon 隔离设计

铁律：
1. 所有 Holon 数据必须在 .holonpolis/holons/<agent_id>/ 内
2. Genesis 数据必须在 .holonpolis/genesis/ 内
3. 任何 ../ 或绝对路径注入都会被立即拒绝
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from holonpolis.config import settings
from holonpolis.infrastructure.storage.path_guard import (
    InvalidArtifactPathError,
    ensure_within_root,
    normalize_path,
    safe_join,
    validate_holon_id,
)


class PathIsolationError(InvalidArtifactPathError):
    """路径隔离违规异常。

    当任何代码试图访问其被授权范围之外的路径时抛出。
    这是物理熔断机制，不可被上层捕获后忽略。
    """
    pass


class HolonPathGuard:
    """Holon 路径守卫 - 每个 Holon 的物理边界。

    每个 Holon 实例必须持有自己的 PathGuard，
    这是实现"物理级记忆隔离"的关键。

    Usage:
        guard = HolonPathGuard("agent_abc123")
        safe_path = guard.resolve("memory/lancedb")  # 自动解析到正确位置
        unsafe = guard.resolve("../../../etc/passwd")  # 抛出 PathIsolationError
    """

    # 允许的相对路径前缀
    ALLOWED_PREFIXES = frozenset({
        "memory",      # LanceDB 记忆存储
        "skills",      # 演化出的技能
        "workspace",   # 工作空间
        "logs",        # 执行日志
        "temp",        # 临时文件
        "blueprint",   # 蓝图缓存
    })

    # 危险路径模式（正则）
    DANGEROUS_PATTERNS = [
        re.compile(r"\.\.[\\/]"),  # ../ 或 ..\
        re.compile(r"^[\\/]"),      # 绝对路径（Unix/Windows）
        re.compile(r"^[A-Za-z]:"),   # Windows 盘符（如 C:）
        re.compile(r"[~]"),          # 用户 home 扩展
        re.compile(r"[%$]"),         # 环境变量扩展
    ]

    def __init__(self, holon_id: str):
        """初始化 Holon 路径守卫。

        Args:
            holon_id: Holon 唯一标识符

        Raises:
            ValueError: 如果 holon_id 格式无效
        """
        self.holon_id = validate_holon_id(holon_id)
        self._root = normalize_path(settings.holonpolis_root)
        self._holon_base = safe_join(
            self._root,
            "holons",
            self.holon_id,
        )

    @property
    def holon_base(self) -> Path:
        """此 Holon 的根目录路径。"""
        return self._holon_base

    @property
    def memory_path(self) -> Path:
        """LanceDB 记忆存储路径。"""
        return safe_join(self._holon_base, "memory", "lancedb")

    @property
    def skills_path(self) -> Path:
        """演化技能存储路径。"""
        return safe_join(self._holon_base, "skills")

    @property
    def workspace_path(self) -> Path:
        """工作空间路径。"""
        return safe_join(self._holon_base, "workspace")

    @property
    def logs_path(self) -> Path:
        """日志存储路径。"""
        return safe_join(self._holon_base, "logs")

    def _detect_path_injection(self, rel_path: str) -> Optional[str]:
        """检测路径注入攻击。

        Args:
            rel_path: 相对路径字符串

        Returns:
            如果检测到攻击，返回原因；否则 None
        """
        # 检查危险模式
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(rel_path):
                return f"检测到路径注入模式: {pattern.pattern}"

        # 规范化后检查
        normalized = rel_path.replace("\\", "/").strip("/")

        # 检查是否以允许的前缀开头
        parts = normalized.split("/")
        if not parts or parts[0] not in self.ALLOWED_PREFIXES:
            # 允许直接访问 Holon 根目录
            if normalized == "." or normalized == "":
                return None
            return f"不允许的路径前缀: {parts[0] if parts else 'empty'}"

        return None

    def resolve(
        self,
        rel_path: str | Path,
        must_exist: bool = False,
    ) -> Path:
        """解析并校验相对路径。

        这是核心方法，将相对路径解析为绝对路径，
        同时执行多层安全校验。

        Args:
            rel_path: 相对路径（如 "memory/lancedb"）
            must_exist: 是否要求路径必须存在

        Returns:
            解析后的绝对路径

        Raises:
            PathIsolationError: 如果路径试图逃逸沙箱
        """
        rel_str = str(rel_path)

        # 1. 注入检测
        injection = self._detect_path_injection(rel_str)
        if injection:
            raise PathIsolationError(
                f"Holon({self.holon_id}) 路径注入检测: {injection} "
                f"in path: {rel_str}"
            )

        # 2. 安全拼接
        try:
            resolved = safe_join(self._holon_base, rel_str)
        except InvalidArtifactPathError as e:
            raise PathIsolationError(
                f"Holon({self.holon_id}) 路径解析失败: {e}"
            ) from e

        # 3. 双重校验（确保在 Holon 目录内）
        try:
            ensure_within_root(self._holon_base, resolved)
        except InvalidArtifactPathError as e:
            raise PathIsolationError(
                f"Holon({self.holon_id}) 路径逃逸检测: {resolved} "
                f"不在 {self._holon_base} 内"
            ) from e

        # 4. 存在性检查
        if must_exist and not resolved.exists():
            raise PathIsolationError(
                f"Holon({self.holon_id}) 路径不存在: {resolved}"
            )

        return resolved

    def ensure_directory(self, rel_path: str | Path) -> Path:
        """确保目录存在，自动创建。

        Args:
            rel_path: 相对路径

        Returns:
            目录的绝对路径
        """
        path = self.resolve(rel_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def is_safe_path(self, path: str | Path) -> bool:
        """快速检查路径是否安全（不抛出异常）。

        Args:
            path: 待检查的路径

        Returns:
            是否安全
        """
        try:
            self.resolve(path)
            return True
        except PathIsolationError:
            return False

    def __repr__(self) -> str:
        return f"HolonPathGuard(holon_id={self.holon_id!r}, base={self._holon_base})"


class GenesisPathGuard:
    """Genesis 路径守卫 - 创世主的物理边界。

    Genesis 作为特殊 Holon，拥有独立的路径空间：
    .holonpolis/genesis/ 而非 .holonpolis/holons/genesis/
    """

    ALLOWED_PREFIXES = frozenset({
        "memory",           # Genesis 记忆库
        "blueprint_cache",  # 蓝图缓存
        "logs",             # 创世日志
        "species",          # 物种定义
    })

    DANGEROUS_PATTERNS = HolonPathGuard.DANGEROUS_PATTERNS

    def __init__(self):
        """初始化 Genesis 路径守卫。"""
        self._root = normalize_path(settings.holonpolis_root)
        self._genesis_base = safe_join(self._root, "genesis")

    @property
    def genesis_base(self) -> Path:
        """Genesis 根目录。"""
        return self._genesis_base

    @property
    def memory_path(self) -> Path:
        """Genesis 记忆库路径（独立的 LanceDB）。"""
        return safe_join(self._genesis_base, "memory", "lancedb")

    @property
    def blueprint_cache_path(self) -> Path:
        """蓝图缓存路径。"""
        return safe_join(self._genesis_base, "blueprint_cache")

    @property
    def species_path(self) -> Path:
        """物种定义存储路径。"""
        return safe_join(self._genesis_base, "species")

    def _detect_path_injection(self, rel_path: str) -> Optional[str]:
        """检测路径注入攻击。"""
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(rel_path):
                return f"检测到路径注入模式: {pattern.pattern}"

        normalized = rel_path.replace("\\", "/").strip("/")
        parts = normalized.split("/")

        if not parts or parts[0] not in self.ALLOWED_PREFIXES:
            if normalized in (".", ""):
                return None
            return f"不允许的路径前缀: {parts[0] if parts else 'empty'}"

        return None

    def resolve(
        self,
        rel_path: str | Path,
        must_exist: bool = False,
    ) -> Path:
        """解析并校验 Genesis 的相对路径。

        Args:
            rel_path: 相对路径
            must_exist: 是否要求存在

        Returns:
            解析后的绝对路径

        Raises:
            PathIsolationError: 如果路径违规
        """
        rel_str = str(rel_path)

        injection = self._detect_path_injection(rel_str)
        if injection:
            raise PathIsolationError(
                f"Genesis 路径注入检测: {injection} in path: {rel_str}"
            )

        try:
            resolved = safe_join(self._genesis_base, rel_str)
        except InvalidArtifactPathError as e:
            raise PathIsolationError(f"Genesis 路径解析失败: {e}") from e

        try:
            ensure_within_root(self._genesis_base, resolved)
        except InvalidArtifactPathError as e:
            raise PathIsolationError(
                f"Genesis 路径逃逸检测: {resolved} 不在 {self._genesis_base} 内"
            ) from e

        if must_exist and not resolved.exists():
            raise PathIsolationError(f"Genesis 路径不存在: {resolved}")

        return resolved

    def ensure_directory(self, rel_path: str | Path) -> Path:
        """确保 Genesis 目录存在。"""
        path = self.resolve(rel_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True)
class ResolvedPath:
    """解析后的路径结果。

    包含完整上下文信息，便于审计和调试。
    """
    absolute: Path
    relative: str
    owner_type: str  # "genesis" | "holon"
    owner_id: Optional[str]  # None for genesis
    is_allowed: bool


def resolve_holon_path(
    holon_id: str,
    rel_path: str | Path,
    must_exist: bool = False,
) -> ResolvedPath:
    """全局便捷函数：解析 Holon 路径。

    Args:
        holon_id: Holon ID
        rel_path: 相对路径
        must_exist: 是否要求存在

    Returns:
        ResolvedPath 结果

    Raises:
        PathIsolationError: 如果路径违规
    """
    guard = HolonPathGuard(holon_id)
    absolute = guard.resolve(rel_path, must_exist=must_exist)

    return ResolvedPath(
        absolute=absolute,
        relative=str(rel_path),
        owner_type="holon",
        owner_id=holon_id,
        is_allowed=True,
    )


def resolve_genesis_path(
    rel_path: str | Path,
    must_exist: bool = False,
) -> ResolvedPath:
    """全局便捷函数：解析 Genesis 路径。

    Args:
        rel_path: 相对路径
        must_exist: 是否要求存在

    Returns:
        ResolvedPath 结果
    """
    guard = GenesisPathGuard()
    absolute = guard.resolve(rel_path, must_exist=must_exist)

    return ResolvedPath(
        absolute=absolute,
        relative=str(rel_path),
        owner_type="genesis",
        owner_id=None,
        is_allowed=True,
    )


def create_holon_directories(holon_id: str) -> Path:
    """为新 Holon 创建完整的目录结构。

    这是 Spawn 操作的核心步骤之一。

    Args:
        holon_id: 新 Holon 的 ID

    Returns:
        Holon 根目录路径
    """
    guard = HolonPathGuard(holon_id)

    # 创建所有标准目录
    guard.ensure_directory("memory/lancedb")
    guard.ensure_directory("skills")
    guard.ensure_directory("workspace")
    guard.ensure_directory("logs")
    guard.ensure_directory("temp")
    guard.ensure_directory("blueprint")

    return guard.holon_base


def create_genesis_directories() -> Path:
    """为 Genesis 创建完整目录结构。

    Returns:
        Genesis 根目录路径
    """
    guard = GenesisPathGuard()

    guard.ensure_directory("memory/lancedb")
    guard.ensure_directory("blueprint_cache")
    guard.ensure_directory("species")
    guard.ensure_directory("logs")

    return guard.genesis_base
