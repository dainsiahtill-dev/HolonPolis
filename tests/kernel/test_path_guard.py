"""Phase 1 Tests: Kernel Path Guard - 物理路径隔离测试

验证 HolonPolis 的铁律之一：绝对路径隔离。
任何试图逃逸 .holonpolis/ 目录的请求都必须被物理熔断。
"""

import os
import tempfile
from pathlib import Path

import pytest

from holonpolis.kernel.storage.path_guard import (
    GenesisPathGuard,
    HolonPathGuard,
    PathIsolationError,
    ResolvedPath,
    create_genesis_directories,
    create_holon_directories,
    resolve_genesis_path,
    resolve_holon_path,
)


class TestHolonPathGuard:
    """Holon 路径守卫测试套件。"""

    def test_holon_path_guard_initialization(self, tmp_path):
        """测试 HolonPathGuard 正确初始化。"""
        guard = HolonPathGuard("test_agent_001")

        assert guard.holon_id == "test_agent_001"
        assert guard.holon_base.name == "test_agent_001"
        assert guard.holon_base.parent.name == "holons"

    def test_holon_standard_paths(self, tmp_path, monkeypatch):
        """测试标准路径属性正确解析。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # 验证路径结构（适配 Windows 和 Unix 路径分隔符）
        assert "memory" in str(guard.memory_path)
        assert "lancedb" in str(guard.memory_path)
        assert "skills" in str(guard.skills_path)
        assert "workspace" in str(guard.workspace_path)
        assert "logs" in str(guard.logs_path)

        # 验证都在 Holon 目录内
        assert guard.memory_path.is_relative_to(guard.holon_base)
        assert guard.skills_path.is_relative_to(guard.holon_base)

    def test_resolve_safe_relative_path(self, tmp_path, monkeypatch):
        """测试解析安全的相对路径。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")
        path = guard.resolve("memory/data.txt")

        assert path.is_absolute()
        assert path.name == "data.txt"
        assert "test_agent" in str(path)

    def test_detect_path_traversal_attack(self, tmp_path, monkeypatch):
        """测试检测路径遍历攻击（../）。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # 各种逃逸尝试
        escape_attempts = [
            "../etc/passwd",
            "memory/../../secret",
            "skills/../../../etc",
            "../" * 10 + "windows/system32",
        ]

        for attempt in escape_attempts:
            with pytest.raises(PathIsolationError) as exc_info:
                guard.resolve(attempt)
            assert "注入检测" in str(exc_info.value) or "逃逸" in str(exc_info.value)

    def test_detect_absolute_path_injection(self, tmp_path, monkeypatch):
        """测试检测绝对路径注入。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # Unix 绝对路径
        with pytest.raises(PathIsolationError):
            guard.resolve("/etc/passwd")

        # Windows 绝对路径
        with pytest.raises(PathIsolationError):
            guard.resolve("C:\\Windows\\System32")

        with pytest.raises(PathIsolationError):
            guard.resolve("D:/data/file.txt")

    def test_detect_home_directory_expansion(self, tmp_path, monkeypatch):
        """测试检测 home 目录扩展符。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        with pytest.raises(PathIsolationError):
            guard.resolve("~/.bashrc")

        with pytest.raises(PathIsolationError):
            guard.resolve("memory/~/data")

    def test_invalid_path_prefix(self, tmp_path, monkeypatch):
        """测试无效路径前缀被拒绝。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # 不允许直接访问 Holon 目录之外的子目录
        with pytest.raises(PathIsolationError) as exc_info:
            guard.resolve("forbidden/data.txt")
        assert "不允许的路径前缀" in str(exc_info.value)

    def test_ensure_directory_creates_structure(self, tmp_path, monkeypatch):
        """测试 ensure_directory 创建完整目录结构。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")
        path = guard.ensure_directory("workspace/nested/deep")

        assert path.exists()
        assert path.is_dir()
        assert "nested" in str(path)

    def test_is_safe_path_quick_check(self, tmp_path, monkeypatch):
        """测试 is_safe_path 快速检查。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        assert guard.is_safe_path("memory/data.txt") is True
        assert guard.is_safe_path("skills/test.py") is True
        assert guard.is_safe_path("../../../etc/passwd") is False
        assert guard.is_safe_path("/absolute/path") is False

    def test_must_exist_validation(self, tmp_path, monkeypatch):
        """测试 must_exist 参数。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # 不存在的路径，must_exist=False 应该成功
        path = guard.resolve("memory/nonexistent.txt", must_exist=False)
        assert path is not None

        # 不存在的路径，must_exist=True 应该失败
        with pytest.raises(PathIsolationError) as exc_info:
            guard.resolve("memory/nonexistent.txt", must_exist=True)
        assert "不存在" in str(exc_info.value)


class TestGenesisPathGuard:
    """Genesis 路径守卫测试套件。"""

    def test_genesis_path_guard_initialization(self, tmp_path, monkeypatch):
        """测试 GenesisPathGuard 初始化。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = GenesisPathGuard()

        assert "genesis" in str(guard.genesis_base)
        assert guard.genesis_base.name == "genesis"

    def test_genesis_separate_from_holons(self, tmp_path, monkeypatch):
        """测试 Genesis 路径与 Holons 路径物理隔离。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        genesis_guard = GenesisPathGuard()
        holon_guard = HolonPathGuard("genesis")  # 名为 genesis 的 Holon

        # 两者路径必须不同
        assert genesis_guard.genesis_base != holon_guard.holon_base
        assert genesis_guard.memory_path != holon_guard.memory_path

    def test_genesis_standard_paths(self, tmp_path, monkeypatch):
        """测试 Genesis 标准路径。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = GenesisPathGuard()

        assert "memory" in str(guard.memory_path)
        assert "lancedb" in str(guard.memory_path)
        assert "blueprint_cache" in str(guard.blueprint_cache_path)
        assert "species" in str(guard.species_path)

    def test_genesis_path_security(self, tmp_path, monkeypatch):
        """测试 Genesis 路径安全。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = GenesisPathGuard()

        with pytest.raises(PathIsolationError):
            guard.resolve("../escape")

        with pytest.raises(PathIsolationError):
            guard.resolve("/etc/passwd")


class TestPathResolutionUtilities:
    """路径解析便捷函数测试。"""

    def test_resolve_holon_path(self, tmp_path, monkeypatch):
        """测试 resolve_holon_path 便捷函数。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        result = resolve_holon_path("agent_001", "memory/data.db")

        assert isinstance(result, ResolvedPath)
        assert result.owner_type == "holon"
        assert result.owner_id == "agent_001"
        assert result.relative == "memory/data.db"
        assert result.is_allowed is True
        assert result.absolute.is_absolute()

    def test_resolve_genesis_path(self, tmp_path, monkeypatch):
        """测试 resolve_genesis_path 便捷函数。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        result = resolve_genesis_path("blueprint_cache/test.json")

        assert isinstance(result, ResolvedPath)
        assert result.owner_type == "genesis"
        assert result.owner_id is None
        assert result.relative == "blueprint_cache/test.json"


class TestDirectoryCreation:
    """目录创建功能测试。"""

    def test_create_holon_directories(self, tmp_path, monkeypatch):
        """测试为新 Holon 创建完整目录结构。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        base = create_holon_directories("new_agent_123")

        assert base.exists()
        assert (base / "memory" / "lancedb").exists()
        assert (base / "skills").exists()
        assert (base / "workspace").exists()
        assert (base / "logs").exists()
        assert (base / "temp").exists()
        assert (base / "blueprint").exists()

    def test_create_genesis_directories(self, tmp_path, monkeypatch):
        """测试为 Genesis 创建完整目录结构。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        base = create_genesis_directories()

        assert base.exists()
        assert (base / "memory" / "lancedb").exists()
        assert (base / "blueprint_cache").exists()
        assert (base / "species").exists()
        assert (base / "logs").exists()


class TestEdgeCases:
    """边界情况测试。"""

    def test_empty_path_handling(self, tmp_path, monkeypatch):
        """测试空路径处理。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # 空字符串应该解析为 Holon 根目录
        path = guard.resolve("")
        assert path == guard.holon_base

    def test_dot_path_handling(self, tmp_path, monkeypatch):
        """测试 '.' 路径处理。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        path = guard.resolve(".")
        assert path == guard.holon_base

    def test_backslash_normalization(self, tmp_path, monkeypatch):
        """测试反斜杠规范化（Windows）。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # Windows 风格路径应该被正确处理
        path = guard.resolve("memory\\data\\file.txt")
        assert "memory" in str(path)
        assert "data" in str(path)

    def test_unicode_path_handling(self, tmp_path, monkeypatch):
        """测试 Unicode 路径处理。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        # 包含 Unicode 的安全路径
        path = guard.resolve("memory/中文数据/文件.txt")
        assert "中文数据" in str(path)

    def test_deeply_nested_path(self, tmp_path, monkeypatch):
        """测试深层嵌套路径。"""
        monkeypatch.setattr(
            "holonpolis.config.settings.holonpolis_root",
            tmp_path / ".holonpolis"
        )

        guard = HolonPathGuard("test_agent")

        deep_path = "memory/" + "/".join([f"level{i}" for i in range(20)])
        path = guard.resolve(deep_path)

        assert path.is_absolute()
        assert all(f"level{i}" in str(path) for i in range(20))
