"""Deterministic project scaffold materialization from indexed reusable libraries."""

from __future__ import annotations

import json
import posixpath
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from holonpolis.infrastructure.storage import ensure_within_root, normalize_path
from holonpolis.infrastructure.storage.file_io import list_directory
from holonpolis.infrastructure.storage.io_text import read_text_utf8, write_text_atomic
from holonpolis.infrastructure.storage.path_resolver import resolve_workspace_path
from holonpolis.kernel.storage import HolonPathGuard
from holonpolis.services.holon_service import HolonService
from holonpolis.services.reusable_code_asset_policy import is_supported_text_asset_path

_DEFAULT_MATERIALIZE_MAX_FILE_BYTES = 512_000
_ROOT_PRIORITY_FILES = (
    ".editorconfig",
    ".env.example",
    ".env.sample",
    ".gitignore",
    ".npmrc",
    "index.html",
    "jsconfig.json",
    "package.json",
    "README.md",
    "README.txt",
    "tailwind.config.cjs",
    "tailwind.config.js",
    "tailwind.config.ts",
    "tsconfig.app.json",
    "tsconfig.json",
    "tsconfig.node.json",
    "vite.config.cjs",
    "vite.config.js",
    "vite.config.mjs",
    "vite.config.ts",
)
_ENTRYPOINT_CANDIDATES = (
    "src/main.tsx",
    "src/main.jsx",
    "src/main.ts",
    "src/main.js",
    "src/app.tsx",
    "src/app.jsx",
    "src/app.ts",
    "src/app.js",
    "main.py",
)
_REFERENCE_CANDIDATE_EXTENSIONS = (
    ".tsx",
    ".jsx",
    ".ts",
    ".js",
    ".mjs",
    ".cjs",
    ".vue",
    ".svelte",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".json",
    ".svg",
    ".html",
    ".htm",
    ".md",
    ".py",
)


@dataclass(frozen=True)
class ReusableProjectScaffold:
    """A deterministic scaffold materialized from an indexed reusable code library."""

    library_key: str
    library_name: str
    library_kind: str
    source_path: str
    output_dir: Path
    project_slug: str
    copied_files: List[str]
    run_instructions: List[str]


class ReusableProjectScaffoldService:
    """Materialize a reusable code library into a runnable project scaffold."""

    def __init__(self, holon_service: Optional[HolonService] = None):
        self._holon_service = holon_service or HolonService()

    def try_materialize(
        self,
        *,
        holon_id: str,
        project_name: str,
        project_slug: str,
        project_goal: str,
        candidate_assets: List[Dict[str, Any]],
        required_files: List[str],
    ) -> Optional[ReusableProjectScaffold]:
        """Clone the best matching indexed library into the Holon sandbox when possible."""
        _ = project_goal
        snapshot = self._select_library_snapshot(holon_id, candidate_assets)
        if snapshot is None:
            return None

        source_root = self._resolve_source_root(snapshot.get("source_path"))
        if source_root is None:
            return None

        library_kind = str(snapshot.get("library_kind") or "code_asset").strip() or "code_asset"
        include_extensions = snapshot.get("allowed_extensions")
        max_file_bytes = self._resolve_max_file_bytes(snapshot)
        candidate_entries = self._collect_candidate_entries(
            source_root=source_root,
            library_kind=library_kind,
            include_extensions=include_extensions,
            max_file_bytes=max_file_bytes,
        )
        if not candidate_entries:
            return None
        candidate_entries = self._select_materialization_entries(
            candidate_entries=candidate_entries,
            snapshot=snapshot,
            candidate_assets=candidate_assets,
            required_files=required_files,
        )

        candidate_paths = {str(item["relative_path"]) for item in candidate_entries}
        missing_required = [path for path in required_files if path not in candidate_paths]
        if missing_required:
            return None

        output_dir = self._create_output_dir(holon_id, project_slug)
        key_file_contents: Dict[str, str] = {}
        copied_files: List[str] = []

        for item in candidate_entries:
            relative_path = str(item["relative_path"])
            content = self._read_asset_text(str(item["full_path"]))
            if content is None:
                continue

            try:
                target = output_dir / relative_path
                ensure_within_root(output_dir, target)
                write_text_atomic(str(target), content)
            except Exception:
                self._cleanup_output_dir(output_dir)
                return None
            copied_files.append(relative_path)
            if relative_path in {"package.json", "pyproject.toml", "requirements.txt"}:
                key_file_contents[relative_path] = content

        if not copied_files:
            self._cleanup_output_dir(output_dir)
            return None

        missing_after_copy = [path for path in required_files if path not in set(copied_files)]
        if missing_after_copy:
            self._cleanup_output_dir(output_dir)
            return None

        return ReusableProjectScaffold(
            library_key=str(snapshot.get("library_key") or "reusable-library").strip() or "reusable-library",
            library_name=str(snapshot.get("library_name") or source_root.name).strip() or source_root.name,
            library_kind=library_kind,
            source_path=str(source_root),
            output_dir=output_dir,
            project_slug=project_slug,
            copied_files=sorted(copied_files),
            run_instructions=self._infer_run_instructions(key_file_contents),
        )

    def _select_library_snapshot(
        self,
        holon_id: str,
        candidate_assets: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        state = self._holon_service.get_holon_state(holon_id)
        libraries = state.get("reusable_code_libraries")
        if not isinstance(libraries, dict) or not libraries:
            return None

        scores: Dict[str, float] = {}
        for index, item in enumerate(candidate_assets):
            if not isinstance(item, dict):
                continue

            library_key = str(item.get("library_key") or "").strip()
            if not library_key:
                library_name = str(item.get("library_name") or "").strip()
                if library_name:
                    library_key = self._slugify(library_name)
            if not library_key or library_key not in libraries:
                continue

            weight = max(1.0, 6.0 - float(index))
            if str(item.get("asset_type") or "").strip() == "source":
                weight += 0.35
            if str(item.get("relative_path") or "").strip():
                weight += 0.15
            scores[library_key] = scores.get(library_key, 0.0) + weight

        if not scores and len(libraries) == 1:
            only_key = next(iter(libraries.keys()))
            snapshot = libraries.get(only_key)
            if isinstance(snapshot, dict):
                return dict(snapshot)
            return None

        ranked_keys = sorted(
            scores.keys(),
            key=lambda key: (
                float(scores.get(key) or 0.0),
                int((libraries.get(key) or {}).get("asset_count") or 0),
                str((libraries.get(key) or {}).get("updated_at") or ""),
            ),
            reverse=True,
        )
        for key in ranked_keys:
            snapshot = libraries.get(key)
            if isinstance(snapshot, dict):
                return dict(snapshot)
        return None

    def _resolve_source_root(self, source_path: Any) -> Optional[Path]:
        raw_path = str(source_path or "").strip()
        if not raw_path:
            return None
        try:
            normalized = normalize_path(raw_path)
            if not normalized.exists() or not normalized.is_dir():
                return None
            workspace_root = resolve_workspace_path(str(normalized))
            return ensure_within_root(workspace_root, normalized)
        except Exception:
            return None

    def _collect_candidate_entries(
        self,
        *,
        source_root: Path,
        library_kind: str,
        include_extensions: Any,
        max_file_bytes: int,
    ) -> List[Dict[str, Any]]:
        try:
            listing = list_directory(str(source_root), "", recursive=True)
        except Exception:
            return []

        entries = listing.get("entries")
        if not isinstance(entries, list):
            return []

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue

            relative_path = str(entry.get("path") or "").replace("\\", "/").strip()
            full_path = str(entry.get("full_path") or "").strip()
            if not relative_path or not full_path:
                continue
            if not is_supported_text_asset_path(
                relative_path,
                library_kind=library_kind,
                include_extensions=include_extensions,
            ):
                continue

            file_size = int(entry.get("size") or 0)
            if file_size < 0 or file_size > max_file_bytes:
                continue

            candidates.append(
                {
                    "relative_path": relative_path,
                    "full_path": full_path,
                }
            )
        return candidates

    def _select_materialization_entries(
        self,
        *,
        candidate_entries: List[Dict[str, str]],
        snapshot: Dict[str, Any],
        candidate_assets: List[Dict[str, Any]],
        required_files: List[str],
    ) -> List[Dict[str, str]]:
        entries_by_path = {
            str(item["relative_path"]): dict(item)
            for item in candidate_entries
            if isinstance(item, dict) and str(item.get("relative_path") or "").strip()
        }
        if not entries_by_path:
            return []

        selected_paths: set[str] = set()
        selected_paths.update(self._select_priority_paths(entries_by_path))
        selected_paths.update(self._select_asset_seed_paths(snapshot, candidate_assets, entries_by_path))
        selected_paths.update(
            path for path in required_files if isinstance(path, str) and path in entries_by_path
        )

        if not selected_paths:
            return [
                entries_by_path[path]
                for path in sorted(entries_by_path.keys(), key=self._materialization_sort_key)
            ]

        expanded_paths = self._expand_dependency_closure(selected_paths, entries_by_path)
        ordered_paths = sorted(expanded_paths, key=self._materialization_sort_key)
        return [entries_by_path[path] for path in ordered_paths if path in entries_by_path]

    @staticmethod
    def _select_priority_paths(entries_by_path: Dict[str, Dict[str, str]]) -> set[str]:
        selected: set[str] = set()
        for rel_path in _ROOT_PRIORITY_FILES:
            if rel_path in entries_by_path:
                selected.add(rel_path)
        for rel_path in _ENTRYPOINT_CANDIDATES:
            if rel_path in entries_by_path:
                selected.add(rel_path)
        return selected

    def _select_asset_seed_paths(
        self,
        snapshot: Dict[str, Any],
        candidate_assets: List[Dict[str, Any]],
        entries_by_path: Dict[str, Dict[str, str]],
    ) -> set[str]:
        selected: set[str] = set()
        target_key = str(snapshot.get("library_key") or "").strip()
        target_name = str(snapshot.get("library_name") or "").strip()

        for item in candidate_assets:
            if not isinstance(item, dict):
                continue
            library_key = str(item.get("library_key") or "").strip()
            library_name = str(item.get("library_name") or "").strip()
            if target_key and library_key and library_key != target_key:
                continue
            if target_name and library_name and library_name != target_name:
                continue

            relative_path = str(item.get("relative_path") or "").replace("\\", "/").strip()
            if relative_path and relative_path in entries_by_path:
                selected.add(relative_path)
        return selected

    def _expand_dependency_closure(
        self,
        seed_paths: set[str],
        entries_by_path: Dict[str, Dict[str, str]],
    ) -> set[str]:
        selected: set[str] = set(path for path in seed_paths if path in entries_by_path)
        pending = list(sorted(selected))
        visited: set[str] = set()

        while pending:
            current = pending.pop(0)
            if current in visited:
                continue
            visited.add(current)

            entry = entries_by_path.get(current)
            if not isinstance(entry, dict):
                continue

            content = self._read_asset_text(str(entry.get("full_path") or ""))
            if content is None:
                continue

            for reference in self._extract_local_references(current, content):
                resolved = self._resolve_reference_path(current, reference, entries_by_path)
                if not resolved or resolved in selected:
                    continue
                selected.add(resolved)
                pending.append(resolved)
        return selected

    @staticmethod
    def _extract_local_references(relative_path: str, content: str) -> List[str]:
        extension = Path(relative_path).suffix.lower()
        patterns: List[str] = []

        if extension in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte", ".py"}:
            patterns.extend(
                [
                    r"""(?m)(?:import|export)\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]""",
                    r"""require\(\s*['"]([^'"]+)['"]\s*\)""",
                    r"""new\s+URL\(\s*['"]([^'"]+)['"]\s*,""",
                ]
            )
        if extension in {".css", ".scss", ".sass", ".less"}:
            patterns.extend(
                [
                    r"""@import\s+(?:url\()?['"]?([^'")]+)['"]?\)?""",
                    r"""url\(\s*['"]?([^'")]+)['"]?\s*\)""",
                ]
            )
        if extension in {".html", ".htm"}:
            patterns.append(r"""(?i)(?:src|href)=["']([^"']+)["']""")

        references: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                value = str(match.group(1) or "").strip()
                if not value:
                    continue
                if value.startswith(("//", "http://", "https://", "data:", "#", "mailto:")):
                    continue
                if value.startswith(("./", "../", "/")):
                    references.append(value)

        deduped: List[str] = []
        for item in references:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _resolve_reference_path(
        current_path: str,
        reference: str,
        entries_by_path: Dict[str, Dict[str, str]],
    ) -> Optional[str]:
        normalized_reference = str(reference or "").strip()
        if not normalized_reference:
            return None

        normalized_reference = normalized_reference.split("?", 1)[0].split("#", 1)[0].strip()
        if not normalized_reference:
            return None

        if normalized_reference.startswith("/"):
            base_candidate = posixpath.normpath(normalized_reference.lstrip("/"))
        else:
            current_dir = str(PurePosixPath(current_path).parent).replace("\\", "/")
            if current_dir == ".":
                current_dir = ""
            base_candidate = posixpath.normpath(
                f"{current_dir}/{normalized_reference}" if current_dir else normalized_reference
            )

        if not base_candidate or base_candidate == "." or base_candidate.startswith("../"):
            return None

        if base_candidate in entries_by_path:
            return base_candidate

        candidate_obj = PurePosixPath(base_candidate)
        if candidate_obj.suffix:
            return None

        attempts: List[str] = []
        for extension in _REFERENCE_CANDIDATE_EXTENSIONS:
            attempts.append(f"{base_candidate}{extension}")
        for extension in _REFERENCE_CANDIDATE_EXTENSIONS:
            attempts.append(f"{base_candidate}/index{extension}")

        for attempt in attempts:
            if attempt in entries_by_path:
                return attempt
        return None

    @staticmethod
    def _materialization_sort_key(relative_path: str) -> tuple[int, int, str]:
        normalized = str(relative_path or "").replace("\\", "/")
        if normalized in _ROOT_PRIORITY_FILES:
            return (0, normalized.count("/"), normalized)
        if normalized in _ENTRYPOINT_CANDIDATES:
            return (1, normalized.count("/"), normalized)
        return (2, normalized.count("/"), normalized)

    @staticmethod
    def _resolve_max_file_bytes(snapshot: Dict[str, Any]) -> int:
        configured = int(snapshot.get("max_file_bytes") or 0)
        return max(configured, _DEFAULT_MATERIALIZE_MAX_FILE_BYTES)

    @staticmethod
    def _read_asset_text(full_path: str) -> Optional[str]:
        try:
            return read_text_utf8(full_path, errors="strict")
        except UnicodeDecodeError:
            try:
                return read_text_utf8(full_path, errors="replace")
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _create_output_dir(holon_id: str, project_slug: str) -> Path:
        guard = HolonPathGuard(holon_id)
        run_suffix = uuid.uuid4().hex[:8]
        return guard.ensure_directory(f"workspace/incubations/{project_slug}_{run_suffix}")

    @staticmethod
    def _cleanup_output_dir(output_dir: Path) -> None:
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            return

    @staticmethod
    def _infer_run_instructions(key_file_contents: Dict[str, str]) -> List[str]:
        package_json = key_file_contents.get("package.json")
        if package_json:
            install = "npm install"
            try:
                payload = json.loads(package_json)
            except Exception:
                payload = {}
            scripts = payload.get("scripts")
            if not isinstance(scripts, dict):
                scripts = {}
            if "dev" in scripts:
                return [install, "npm run dev"]
            if "start" in scripts:
                return [install, "npm start"]
            if "build" in scripts and "preview" in scripts:
                return [install, "npm run build", "npm run preview"]
            return [install]

        if "requirements.txt" in key_file_contents:
            return [
                "python -m pip install -r requirements.txt",
                "python main.py",
            ]
        if "pyproject.toml" in key_file_contents:
            return ["python -m pip install -e ."]
        return []

    @staticmethod
    def _slugify(value: str) -> str:
        import re

        slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
        return slug or "reusable-library"
