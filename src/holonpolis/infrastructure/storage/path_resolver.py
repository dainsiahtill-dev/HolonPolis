"""Path utilities for HolonPolis.

Hard-cut storage layout:
- runtime/*
- workspace/*
- config/*
"""

from __future__ import annotations

import os
import re
from typing import Optional

from holonpolis.infrastructure.storage.storage_layout import (
    UNSUPPORTED_PATH_PREFIX,
    normalize_logical_rel_path,
    resolve_logical_path,
    resolve_storage_roots,
    resolve_ramdisk_root,
    state_to_ramdisk_enabled,
    default_ramdisk_root,
)


ARTIFACT_ROOT = "runtime"
LEGACY_ARTIFACT_ROOT = ""
ARTIFACT_NAMESPACE = ""
LEGACY_ARTIFACT_NAMESPACE = ""


def find_workspace_root(start: str) -> str:
    """Find workspace root by looking for docs/ directory or .git/."""
    current = os.path.abspath(start)
    while True:
        # Check for common workspace markers
        if os.path.isdir(os.path.join(current, "docs")) or os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return ""


def resolve_workspace_path(path: str, *, require_docs: bool = False) -> str:
    """Resolve and validate workspace path."""
    start = (path or "").strip()
    if not start:
        start = os.getcwd()
    start = os.path.abspath(start)
    if not os.path.isdir(start):
        raise ValueError(f"Workspace path does not exist: {start}")
    root = find_workspace_root(start)
    if not root:
        if require_docs:
            raise ValueError(f"No docs/ or .git/ directory found at or above workspace: {start}")
        return start
    if os.path.abspath(root) != start:
        print(f"[workspace] Using '{root}' (found workspace marker above '{start}').")
    return root


def workspace_has_docs(workspace: str) -> bool:
    """Check if workspace has docs directory."""
    if not workspace:
        return False
    if os.path.isdir(os.path.join(workspace, "docs")):
        return True
    try:
        roots = resolve_storage_roots(workspace)
        return os.path.isdir(os.path.join(roots.workspace_persistent_root, "docs"))
    except Exception:
        return False


def normalize_artifact_rel_path(rel_path: str) -> str:
    """Normalize artifact relative path."""
    raw = str(rel_path or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return normalize_logical_rel_path(raw)


def _artifact_base_dir(workspace_full: str, cache_root_full: str) -> str:
    """Get artifact base directory."""
    if cache_root_full:
        return cache_root_full
    return resolve_storage_roots(workspace_full).runtime_root


def _strip_artifact_root_prefix(rel_path: str) -> str:
    """Strip artifact root prefix from path."""
    p = normalize_artifact_rel_path(rel_path)
    if p.startswith("runtime/"):
        return p[len("runtime/"):]
    if p.startswith("workspace/"):
        return p[len("workspace/"):]
    if p.startswith("config/"):
        return p[len("config/"):]
    return p


def build_cache_root(ramdisk_root: str, workspace_full: str) -> str:
    """Build cache root path."""
    roots = resolve_storage_roots(workspace_full, ramdisk_root=ramdisk_root or None)
    return roots.runtime_root


def is_hot_artifact_path(rel_path: str) -> bool:
    """Check if path is a hot (runtime) artifact path."""
    p = normalize_artifact_rel_path(rel_path)
    return p == "runtime" or p.startswith("runtime/")


def resolve_run_dir(workspace_full: str, cache_root_full: str, run_id: str) -> str:
    """Resolve run directory path."""
    if not run_id:
        return ""
    runtime_root = cache_root_full or resolve_storage_roots(workspace_full).runtime_root
    return os.path.join(runtime_root, "runs", run_id)


def update_latest_pointer(
    workspace_full: str, cache_root_full: str, run_id: str
) -> None:
    """Update latest run pointer."""
    from holonpolis.infrastructure.storage.io_text import write_json_atomic

    if not run_id:
        return
    runtime_root = cache_root_full or resolve_storage_roots(workspace_full).runtime_root
    latest_dir = os.path.join(runtime_root, "runs", "latest")
    run_dir = resolve_run_dir(workspace_full, cache_root_full, run_id)
    pointer_path = os.path.join(runtime_root, "latest_run.json")
    write_json_atomic(pointer_path, {"run_id": run_id, "path": run_dir})
    if os.path.exists(latest_dir):
        try:
            if os.path.islink(latest_dir):
                os.remove(latest_dir)
        except Exception:
            pass
    try:
        os.symlink(run_dir, latest_dir, target_is_directory=True)
    except Exception:
        pass


def is_run_artifact(rel_path: str) -> bool:
    """Check if path is a run artifact."""
    lowered = rel_path.lower().replace("\\", "/")
    if lowered.endswith("director_result.json") or lowered.endswith("director.result.json"):
        return True
    if lowered.endswith("events.jsonl") or lowered.endswith("runtime.events.jsonl"):
        return True
    if lowered.endswith("trajectory.json"):
        return True
    if lowered.endswith("qa_response.md") or lowered.endswith("qa.review.md"):
        return True
    if lowered.endswith("planner_response.md") or lowered.endswith("planner.output.md"):
        return True
    if lowered.endswith("ollama_response.md") or lowered.endswith("director_llm.output.md"):
        return True
    if lowered.endswith("reviewer_response.md") or lowered.endswith("auditor.review.md"):
        return True
    if lowered.endswith("runlog.md") or lowered.endswith("director.runlog.md"):
        return True
    return False


def resolve_artifact_path(
    workspace_full: str,
    cache_root_full: str,
    rel_path: str,
    run_id: Optional[str] = None,
) -> str:
    """Resolve artifact path with workspace safety guards."""
    if not rel_path:
        return ""
    raw = str(rel_path).strip()
    if os.path.isabs(raw):
        absolute = os.path.abspath(raw)
        roots = resolve_storage_roots(workspace_full)
        allowed_roots = [
            os.path.abspath(roots.runtime_root),
            os.path.abspath(roots.workspace_persistent_root),
            os.path.abspath(roots.config_root),
            os.path.abspath(workspace_full),
        ]
        for root in allowed_roots:
            try:
                if os.path.commonpath([root, absolute]) == root:
                    return absolute
            except Exception:
                continue
        raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")

    # Explicit legacy paths under ".holonpolis/" are treated as workspace-local
    # to preserve compatibility with existing CLI arguments and tests.
    legacy = raw.replace("\\", "/")
    while legacy.startswith("./"):
        legacy = legacy[2:]
    legacy = legacy.lstrip("/")
    if legacy.startswith(".holonpolis/"):
        absolute = os.path.abspath(os.path.join(workspace_full, legacy))
        workspace_abs = os.path.abspath(workspace_full)
        try:
            if os.path.commonpath([workspace_abs, absolute]) != workspace_abs:
                raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
        except Exception:
            raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
        return absolute

    normalized = normalize_artifact_rel_path(raw)
    if run_id and is_run_artifact(normalized):
        run_dir = resolve_run_dir(workspace_full, cache_root_full, run_id)
        basename = os.path.basename(normalized)
        return os.path.join(run_dir, basename)
    if normalized == "runtime" or normalized.startswith("runtime/"):
        runtime_root = cache_root_full or resolve_storage_roots(workspace_full).runtime_root
        if normalized == "runtime":
            return runtime_root
        return os.path.join(runtime_root, normalized[len("runtime/"):])
    if normalized == "workspace" or normalized.startswith("workspace/"):
        # workspace/ prefix maps directly to workspace directory
        suffix = "" if normalized == "workspace" else normalized[len("workspace/"):]
        # Security: reject paths that look like Windows drive letters (e.g., C:/)
        if _looks_like_windows_drive(suffix):
            raise ValueError(f"{UNSUPPORTED_PATH_PREFIX}: {rel_path}")
        return os.path.abspath(os.path.join(workspace_full, suffix))
    return resolve_logical_path(workspace_full, normalized)


def resolve_safe_path(workspace_full: str, cache_root_full: str, rel_path: str) -> str:
    """Resolve path with validation, raising HTTPException on invalid."""
    try:
        full = resolve_artifact_path(workspace_full, cache_root_full, rel_path)
    except ValueError as e:
        raise ValueError(f"Invalid path: {e}")
    if not full:
        raise ValueError("Path is required")
    return full


def _looks_like_windows_drive(path: str) -> bool:
    """Check if path starts with a Windows drive letter (e.g., C:/ or D:\\)."""
    if not path:
        return False
    # Match patterns like C:/, C:\, c:/, c:\ at the start of path
    return bool(re.match(r'^[A-Za-z]:[/\\]', path))


def select_latest_artifact(workspace: str, cache_root: str, rel_path: str) -> str:
    """Select latest artifact path if it exists."""
    try:
        path = resolve_artifact_path(workspace, cache_root, rel_path)
    except Exception:
        return ""
    if path and os.path.isfile(path):
        return path
    return ""
