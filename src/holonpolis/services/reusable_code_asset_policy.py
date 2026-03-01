"""Shared reusable code asset selection rules.

This module centralizes which source files are considered reusable text assets.
Both indexing and deterministic scaffold materialization must use the same rules
so that "learned" and "deliverable" stay aligned.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_CODE_EXTENSIONS = frozenset(
    {
        ".cjs",
        ".css",
        ".go",
        ".htm",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".less",
        ".md",
        ".mjs",
        ".py",
        ".rs",
        ".sass",
        ".scss",
        ".svelte",
        ".svg",
        ".swift",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)
UI_COMPONENT_EXTENSIONS = frozenset(
    {
        ".css",
        ".htm",
        ".html",
        ".jsx",
        ".less",
        ".sass",
        ".scss",
        ".svg",
        ".svelte",
        ".tsx",
        ".vue",
    }
)
TEXT_ASSET_FILENAMES = frozenset(
    {
        ".editorconfig",
        ".eslintignore",
        ".eslintrc",
        ".eslintrc.json",
        ".gitignore",
        ".npmrc",
        ".nvmrc",
        ".prettierignore",
        ".prettierrc",
        ".prettierrc.json",
        "dockerfile",
        "license",
        "license.md",
        "makefile",
        "readme",
        "readme.md",
    }
)
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".next",
        ".turbo",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
    }
)
STORY_OR_TEST_MARKERS = (".stories.", ".story.", ".spec.", ".test.")


def normalize_library_kind(library_kind: str) -> str:
    """Normalize a library kind into a stable slug."""
    normalized = str(library_kind or "code_asset").strip().lower()
    if not normalized:
        return "code_asset"
    slug = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")
    return slug or "code_asset"


def normalize_extensions(
    include_extensions: Optional[Iterable[str]],
    library_kind: str,
) -> set[str]:
    """Normalize caller-supplied extensions or return sensible defaults."""
    values = include_extensions or []
    normalized: set[str] = set()
    for item in values:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = f".{text}"
        normalized.add(text)
    if normalized:
        return normalized
    if normalize_library_kind(library_kind) == "ui_component":
        return set(UI_COMPONENT_EXTENSIONS)
    return set(DEFAULT_CODE_EXTENSIONS)


def should_skip_relative_path(relative_path: str) -> bool:
    """Return True when a path should be excluded from reuse/indexing."""
    lowered = str(relative_path or "").replace("\\", "/").lower()
    if not lowered:
        return True
    parts = [part for part in lowered.split("/") if part]
    if any(part in DEFAULT_EXCLUDED_DIRS for part in parts):
        return True
    name = parts[-1] if parts else lowered
    if any(marker in name for marker in STORY_OR_TEST_MARKERS):
        return True
    if name.endswith(".d.ts") or name.endswith(".lock"):
        return True
    return False


def is_supported_text_asset_path(
    relative_path: str,
    *,
    library_kind: str,
    include_extensions: Optional[Iterable[str]] = None,
) -> bool:
    """Return True when the relative path should be treated as a reusable text asset."""
    if should_skip_relative_path(relative_path):
        return False

    normalized_kind = normalize_library_kind(library_kind)
    allowed_extensions = normalize_extensions(include_extensions, normalized_kind)
    normalized_path = str(relative_path or "").replace("\\", "/").strip()
    if not normalized_path:
        return False

    name = Path(normalized_path).name.lower()
    if name in TEXT_ASSET_FILENAMES:
        return True
    if name.startswith(".env.") and (
        name.endswith(".example") or name.endswith(".sample")
    ):
        return True

    extension = Path(normalized_path).suffix.lower()
    if extension in allowed_extensions:
        return True

    # Support nested filenames such as ".eslintrc.cjs" via suffixes.
    for suffix in Path(normalized_path).suffixes:
        if suffix.lower() in allowed_extensions:
            return True
    return False
