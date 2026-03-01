"""Project delivery/export utilities.

This service exports already-generated project artifacts from a Holon's
workspace to a user-requested destination. It never generates or edits
business code itself; it only copies the finalized output after incubation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from holonpolis.infrastructure.storage import ensure_within_root, normalize_path
from holonpolis.infrastructure.storage.file_io import list_directory
from holonpolis.infrastructure.storage.io_text import read_text_utf8, write_text_atomic
from holonpolis.kernel.storage import HolonPathGuard


class ProjectDeliveryService:
    """Export project artifacts from a Holon workspace to an external directory."""

    def export_project(
        self,
        *,
        holon_id: str,
        source_dir: str | Path,
        target_dir: str | Path,
        replace_existing: bool = False,
    ) -> Path:
        """Copy a generated project from `.holonpolis` to a user-visible location."""
        guard = HolonPathGuard(holon_id)
        source_root = normalize_path(source_dir)
        ensure_within_root(guard.workspace_path, source_root)
        if not source_root.exists() or not source_root.is_dir():
            raise ValueError(f"Generated project directory does not exist: {source_root}")

        target_root = normalize_path(target_dir)
        self._validate_target(target_root)

        if target_root.exists():
            if not replace_existing:
                raise ValueError(f"Export target already exists: {target_root}")
            if target_root.is_dir():
                shutil.rmtree(target_root, ignore_errors=True)
            else:
                target_root.unlink(missing_ok=True)

        target_root.mkdir(parents=True, exist_ok=True)

        listing = list_directory(str(source_root), "", recursive=True)
        entries = listing.get("entries")
        if not isinstance(entries, list):
            entries = []

        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue

            relative_path = str(entry.get("path") or "").replace("\\", "/").strip()
            full_path = str(entry.get("full_path") or "").strip()
            if not relative_path or not full_path:
                continue

            target_path = target_root / Path(relative_path)
            ensure_within_root(target_root, target_path)

            content = self._read_text_asset(full_path)
            if content is None:
                raise ValueError(f"Export failed because a file is not readable as UTF-8 text: {relative_path}")
            write_text_atomic(str(target_path), content)

        return target_root

    @staticmethod
    def _validate_target(target_root: Path) -> None:
        if not target_root.is_absolute():
            raise ValueError("export_target_path must be an absolute path")
        parts = {part.lower() for part in target_root.parts}
        if ".holonpolis" in parts:
            raise ValueError("export_target_path must be outside .holonpolis runtime storage")

    @staticmethod
    def _read_text_asset(full_path: str) -> Optional[str]:
        try:
            return read_text_utf8(full_path, errors="strict")
        except UnicodeDecodeError:
            try:
                return read_text_utf8(full_path, errors="replace")
            except Exception:
                return None
        except Exception:
            return None
