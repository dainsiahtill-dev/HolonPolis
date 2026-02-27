"""Unified storage layout for HolonPolis.

This module is the single source of truth for internal storage paths.
Logical path prefixes:
- runtime/*
- workspace/*
- config/*
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

UNSUPPORTED_PATH_PREFIX = "UNSUPPORTED_PATH_PREFIX"
_ALLOWED_PREFIXES = ("runtime", "workspace", "config")
_RAMDISK_ENV = "HOLONPOLIS_RAMDISK_ROOT"
_STATE_TO_RAMDISK_ENV = "HOLONPOLIS_STATE_TO_RAMDISK"


@dataclass(frozen=True)
class StorageRoots:
    workspace_abs: str
    workspace_key: str
    home_root: str
    config_root: str
    workspaces_root: str
    workspace_persistent_root: str
    runtime_base: str
    runtime_root: str
    runtime_mode: str


def _truthy_env(value: str, default: bool = True) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def state_to_ramdisk_enabled() -> bool:
    return _truthy_env(os.environ.get(_STATE_TO_RAMDISK_ENV, "1"), default=True)


def normalize_ramdisk_root(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = os.path.expandvars(os.path.expanduser(raw))
    if not os.path.isabs(raw):
        return ""
    if re.match(r"^[A-Za-z]:$", raw):
        raw = raw + "\\"
    raw = os.path.abspath(raw).rstrip("\\/")
    if re.match(r"^[A-Za-z]:$", raw):
        raw = raw + "\\"
    return raw


def default_ramdisk_root() -> str:
    if os.name != "nt":
        return ""
    if os.path.exists("X:\\"):
        return "X:\\"
    return ""


def resolve_ramdisk_root(cli_value: Optional[str] = None) -> str:
    if cli_value is not None and str(cli_value).strip():
        return normalize_ramdisk_root(str(cli_value))
    env_value = str(os.environ.get(_RAMDISK_ENV) or "").strip()
    if env_value:
        return normalize_ramdisk_root(env_value)
    return normalize_ramdisk_root(default_ramdisk_root())


def holonpolis_home() -> str:
    raw = str(os.environ.get("HOLONPOLIS_HOME") or "").strip()
    if raw:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
    return os.path.abspath(os.path.expanduser("~/.holonpolis"))


def default_system_cache_base() -> str:
    if os.name == "nt":
        local_app_data = str(os.environ.get("LOCALAPPDATA") or "").strip()
        if local_app_data:
            return os.path.abspath(os.path.join(local_app_data, "HolonPolis", "cache"))
        return os.path.abspath(os.path.expanduser("~\\AppData\\Local\\HolonPolis\\cache"))
    if _sys_platform_is_macos():
        return os.path.abspath(os.path.expanduser("~/Library/Caches/HolonPolis"))
    xdg = str(os.environ.get("XDG_CACHE_HOME") or "").strip()
    if xdg:
        return os.path.abspath(os.path.join(os.path.expanduser(xdg), "holonpolis"))
    return os.path.abspath(os.path.expanduser("~/.cache/holonpolis"))


def _sys_platform_is_macos() -> bool:
    return os.name == "posix" and os.uname().sysname.lower() == "darwin"


def workspace_key(workspace: str) -> str:
    workspace_abs = os.path.abspath(os.path.expanduser(workspace or os.getcwd()))
    base = os.path.basename(workspace_abs.rstrip("\\/")) or "workspace"
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    slug = slug or "workspace"
    digest = hashlib.sha1(
        workspace_abs.encode("utf-8", errors="ignore")
    ).hexdigest()[:12]
    return f"{slug}-{digest}"


def _is_within_path(parent: str, child: str) -> bool:
    try:
        parent_abs = os.path.abspath(parent)
        child_abs = os.path.abspath(child)
        return os.path.commonpath([parent_abs, child_abs]) == parent_abs
    except Exception:
        return False


def _is_runtime_base_writable(base: str) -> bool:
    try:
        candidate = os.path.abspath(base)
        os.makedirs(candidate, exist_ok=True)
        probe_dir = os.path.join(candidate, ".holonpolis-probe")
        os.makedirs(probe_dir, exist_ok=True)
        probe_file = os.path.join(probe_dir, ".write")
        with open(probe_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe_file)
        os.rmdir(probe_dir)
        return True
    except Exception:
        return False


def _runtime_base_and_mode(workspace_abs: str, ramdisk_root: Optional[str]) -> tuple[str, str]:
    explicit_runtime_root = str(os.environ.get("HOLONPOLIS_RUNTIME_ROOT") or "").strip()
    if explicit_runtime_root:
        base = os.path.abspath(os.path.expanduser(os.path.expandvars(explicit_runtime_root)))
        if not _is_within_path(workspace_abs, base) and _is_runtime_base_writable(base):
            return base, "explicit_runtime_root"

    if state_to_ramdisk_enabled():
        ramdisk = resolve_ramdisk_root(ramdisk_root)
        if ramdisk and os.path.exists(ramdisk):
            if not _is_within_path(workspace_abs, ramdisk) and _is_runtime_base_writable(ramdisk):
                return ramdisk, "ramdisk"

    explicit_cache_root = str(os.environ.get("HOLONPOLIS_RUNTIME_CACHE_ROOT") or "").strip()
    if explicit_cache_root:
        base = os.path.abspath(os.path.expanduser(os.path.expandvars(explicit_cache_root)))
        if not _is_within_path(workspace_abs, base) and _is_runtime_base_writable(base):
            return base, "explicit_runtime_cache"

    system_cache_base = default_system_cache_base()
    if not _is_within_path(workspace_abs, system_cache_base) and _is_runtime_base_writable(system_cache_base):
        return system_cache_base, "system_cache"

    # Last-resort fallback for restricted environments (e.g. sandboxed tests).
    workspace_fallback = os.path.join(workspace_abs, ".holonpolis-cache")
    os.makedirs(workspace_fallback, exist_ok=True)
    return workspace_fallback, "workspace_fallback"


def resolve_storage_roots(workspace: str, ramdisk_root: Optional[str] = None) -> StorageRoots:
    workspace_abs = os.path.abspath(os.path.expanduser(workspace or os.getcwd()))
    key = workspace_key(workspace_abs)
    home_root = holonpolis_home()
    config_root = os.path.join(home_root, "config")
    workspaces_root = os.path.join(home_root, "workspaces")
    workspace_persistent_root = os.path.join(workspaces_root, key)
    runtime_base, runtime_mode = _runtime_base_and_mode(workspace_abs, ramdisk_root)
    runtime_root = os.path.join(runtime_base, ".holonpolis", "runtime", key)
    return StorageRoots(
        workspace_abs=workspace_abs,
        workspace_key=key,
        home_root=home_root,
        config_root=config_root,
        workspaces_root=workspaces_root,
        workspace_persistent_root=workspace_persistent_root,
        runtime_base=runtime_base,
        runtime_root=runtime_root,
        runtime_mode=runtime_mode,
    )


def normalize_logical_rel_path(rel_path: str) -> str:
    raw = str(rel_path or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    p = raw.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if p.startswith(".holonpolis/"):
        legacy = p.split("/", 1)[1] if "/" in p else ""
        if legacy == "runtime" or legacy.startswith("runtime/"):
            p = legacy
        elif legacy == "workspace" or legacy.startswith("workspace/"):
            p = legacy
        elif legacy == "config" or legacy.startswith("config/"):
            p = legacy
        else:
            raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {raw}")
    p = os.path.normpath(p).replace("\\", "/")
    if p == ".":
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {raw}")
    if p.startswith("../") or p == "..":
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {raw}")
    prefix = p.split("/", 1)[0]
    if prefix not in _ALLOWED_PREFIXES:
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {raw}")
    return p


def _join_under(root: str, rel_path: str) -> str:
    abs_root = os.path.abspath(root)
    full = os.path.abspath(os.path.join(abs_root, rel_path))
    try:
        if os.path.commonpath([abs_root, full]) != abs_root:
            raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
    except ValueError:
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
    return full


def resolve_global_path(rel_path: str) -> str:
    normalized = normalize_logical_rel_path(rel_path)
    if normalized == "config":
        suffix = ""
    elif normalized.startswith("config/"):
        suffix = normalized[len("config/"):]
    else:
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
    root = os.path.join(holonpolis_home(), "config")
    return _join_under(root, suffix)


def resolve_workspace_persistent_path(workspace: str, rel_path: str) -> str:
    normalized = normalize_logical_rel_path(rel_path)
    if normalized == "workspace":
        suffix = ""
    elif normalized.startswith("workspace/"):
        suffix = normalized[len("workspace/"):]
    else:
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
    roots = resolve_storage_roots(workspace)
    return _join_under(roots.workspace_persistent_root, suffix)


def resolve_runtime_path(
    workspace: str,
    rel_path: str,
    *,
    ramdisk_root: Optional[str] = None,
) -> str:
    normalized = normalize_logical_rel_path(rel_path)
    if normalized == "runtime":
        suffix = ""
    elif normalized.startswith("runtime/"):
        suffix = normalized[len("runtime/"):]
    else:
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
    roots = resolve_storage_roots(workspace, ramdisk_root=ramdisk_root)
    return _join_under(roots.runtime_root, suffix)


def resolve_logical_path(
    workspace: str,
    rel_path: str,
    *,
    ramdisk_root: Optional[str] = None,
) -> str:
    normalized = normalize_logical_rel_path(rel_path)
    if normalized == "runtime" or normalized.startswith("runtime/"):
        return resolve_runtime_path(
            workspace,
            normalized,
            ramdisk_root=ramdisk_root,
        )
    if normalized == "workspace" or normalized.startswith("workspace/"):
        return resolve_workspace_persistent_path(workspace, normalized)
    if normalized == "config" or normalized.startswith("config/"):
        return resolve_global_path(normalized)
    raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
