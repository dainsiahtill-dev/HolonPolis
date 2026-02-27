"""Tests for storage path guardrails."""

from pathlib import Path

import pytest

from holonpolis.infrastructure.storage.path_guard import (
    InvalidArtifactPathError,
    ensure_within_root,
    safe_join,
    validate_holon_id,
)


def test_ensure_within_root_accepts_child(tmp_path):
    root = tmp_path / ".holonpolis"
    child = root / "holons" / "holon_abc"
    resolved = ensure_within_root(root, child)
    assert str(resolved).startswith(str(root))


def test_safe_join_blocks_escape(tmp_path):
    root = tmp_path / ".holonpolis"
    with pytest.raises(InvalidArtifactPathError):
        safe_join(root, "..", "outside")


def test_validate_holon_id():
    assert validate_holon_id("holon_abc123") == "holon_abc123"
    assert validate_holon_id("genesis") == "genesis"
    with pytest.raises(ValueError):
        validate_holon_id("../bad")

