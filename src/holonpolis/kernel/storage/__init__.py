"""Kernel Storage - 物理级路径隔离。

实现 HolonPolis 的铁律之一：绝对路径隔离。
每个 Holon 拥有独立的物理存储边界。
"""

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

__all__ = [
    "HolonPathGuard",
    "GenesisPathGuard",
    "PathIsolationError",
    "ResolvedPath",
    "resolve_holon_path",
    "resolve_genesis_path",
    "create_holon_directories",
    "create_genesis_directories",
]
