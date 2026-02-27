"""Path guardrails adapted from Harborpilot storage layout rules."""

from __future__ import annotations

import os
import re
from pathlib import Path


_HOLON_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{2,127}$")


class InvalidArtifactPathError(ValueError):
    """Raised when a path escapes the configured HolonPolis root."""


def normalize_path(path: str | Path) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise InvalidArtifactPathError("path is required")
    expanded = os.path.expandvars(os.path.expanduser(raw))
    return Path(expanded).resolve(strict=False)


def ensure_within_root(root: str | Path, path: str | Path) -> Path:
    """Ensure `path` is under `root` (inclusive)."""
    root_path = normalize_path(root)
    candidate = normalize_path(path)

    try:
        common = os.path.commonpath([str(root_path), str(candidate)])
    except ValueError as exc:
        raise InvalidArtifactPathError(str(exc)) from exc

    if common != str(root_path):
        raise InvalidArtifactPathError(f"path escapes root: {candidate}")
    return candidate


def safe_join(root: str | Path, *parts: str) -> Path:
    base = normalize_path(root)
    candidate = (base.joinpath(*parts)).resolve(strict=False)
    return ensure_within_root(base, candidate)


def validate_holon_id(holon_id: str) -> str:
    value = str(holon_id or "").strip()
    if value == "genesis":
        return value
    if not _HOLON_ID_PATTERN.fullmatch(value):
        raise ValueError(f"invalid holon_id: {holon_id!r}")
    return value

