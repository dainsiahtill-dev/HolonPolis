"""UI component library indexing and retrieval for per-Holon memory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from holonpolis.domain.memory import MemoryKind
from holonpolis.infrastructure.storage import ensure_within_root, normalize_path
from holonpolis.infrastructure.storage.file_io import list_directory
from holonpolis.infrastructure.storage.io_text import read_text_utf8
from holonpolis.infrastructure.storage.path_resolver import resolve_workspace_path
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.services.holon_service import HolonService
from holonpolis.services.memory_service import HybridSearchResult, MemoryService

logger = structlog.get_logger()


_DEFAULT_COMPONENT_EXTENSIONS = (
    ".tsx",
    ".jsx",
    ".vue",
    ".svelte",
    ".css",
    ".scss",
    ".sass",
    ".less",
)
_STYLE_EXTENSIONS = {".css", ".scss", ".sass", ".less"}
_DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".next",
    ".turbo",
    "build",
    "coverage",
    "dist",
    "node_modules",
}
_STORY_OR_TEST_MARKERS = (".stories.", ".story.", ".spec.", ".test.")


class UIComponentLibraryService:
    """Index local UI component assets into per-Holon LanceDB memory."""

    def __init__(self, holon_id: str, memory_service: Optional[MemoryService] = None):
        self.holon_id = holon_id
        self.memory = memory_service or MemoryService(holon_id)
        self._holon_service = HolonService()

    async def index_local_library(
        self,
        source_path: str,
        *,
        library_name: str,
        framework: str = "react",
        store_mode: str = "full",
        include_extensions: Optional[List[str]] = None,
        max_file_bytes: int = 60000,
    ) -> Dict[str, Any]:
        """Index a local UI component library directory into LanceDB-backed memory."""
        normalized_mode = self._normalize_store_mode(store_mode)
        normalized_framework = str(framework or "unknown").strip().lower() or "unknown"
        source_root = self._resolve_source_root(source_path)
        normalized_library_name = str(library_name or source_root.name).strip() or source_root.name
        library_key = self._slugify(normalized_library_name)
        extensions = self._normalize_extensions(include_extensions)
        max_bytes = max(512, int(max_file_bytes))

        state = self._holon_service.get_holon_state(self.holon_id)
        existing_libraries = state.get("ui_component_libraries")
        if not isinstance(existing_libraries, dict):
            existing_libraries = {}
        prior_snapshot = existing_libraries.get(library_key)
        if not isinstance(prior_snapshot, dict):
            prior_snapshot = {}
        prior_fingerprints = prior_snapshot.get("fingerprints")
        if not isinstance(prior_fingerprints, dict):
            prior_fingerprints = {}

        listing = list_directory(str(source_root), "", recursive=True)
        entries = listing.get("entries")
        if not isinstance(entries, list):
            entries = []

        indexed_components: List[Dict[str, Any]] = []
        scanned_files = 0
        indexed_count = 0
        reused_count = 0
        skipped_files = 0
        active_fingerprints: Dict[str, str] = {}

        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue

            relative_path = str(entry.get("path") or "").replace("\\", "/").strip()
            full_path = str(entry.get("full_path") or "").strip()
            if not relative_path or not full_path:
                skipped_files += 1
                continue
            if self._should_skip_path(relative_path):
                skipped_files += 1
                continue

            file_path = Path(relative_path)
            extension = file_path.suffix.lower()
            if extension not in extensions:
                skipped_files += 1
                continue

            file_size = int(entry.get("size") or 0)
            if file_size <= 0 or file_size > max_bytes:
                skipped_files += 1
                continue

            scanned_files += 1

            try:
                code_content = read_text_utf8(full_path, errors="replace").strip()
            except Exception as exc:
                logger.warning(
                    "ui_component_library_read_failed",
                    holon_id=self.holon_id,
                    source_path=full_path,
                    error=str(exc),
                )
                skipped_files += 1
                continue

            if not code_content:
                skipped_files += 1
                continue

            content_hash = hashlib.sha1(code_content.encode("utf-8")).hexdigest()
            active_fingerprints[relative_path] = content_hash
            if prior_fingerprints.get(relative_path) == content_hash:
                reused_count += 1
                continue

            metadata = self._build_component_metadata(
                library_name=normalized_library_name,
                library_key=library_key,
                framework=normalized_framework,
                relative_path=relative_path,
                extension=extension,
                code_content=code_content,
                content_hash=content_hash,
                store_mode=normalized_mode,
            )
            memory_payload = self._build_memory_payload(metadata, code_content)

            await self.memory.remember(
                content=memory_payload,
                kind=MemoryKind.PATTERN,
                tags=self._build_tags(metadata),
                importance=1.2,
                source_skill=f"ui_library::{library_key}",
            )

            indexed_count += 1
            indexed_components.append(
                {
                    "component_name": metadata["component_name"],
                    "relative_path": metadata["relative_path"],
                    "asset_type": metadata["asset_type"],
                    "exports": metadata["exports"],
                }
            )

        snapshot = {
            "library_name": normalized_library_name,
            "library_key": library_key,
            "framework": normalized_framework,
            "source_path": str(source_root),
            "store_mode": normalized_mode,
            "max_file_bytes": max_bytes,
            "component_count": len(active_fingerprints),
            "indexed_components": indexed_count,
            "reused_components": reused_count,
            "skipped_files": skipped_files,
            "last_indexed_at": utc_now_iso(),
            "fingerprints": active_fingerprints,
        }
        self._holon_service.record_ui_component_library_index(
            self.holon_id,
            library_key,
            snapshot=snapshot,
        )

        return {
            "status": "indexed",
            "holon_id": self.holon_id,
            "library_name": normalized_library_name,
            "library_key": library_key,
            "framework": normalized_framework,
            "source_path": str(source_root),
            "store_mode": normalized_mode,
            "scanned_files": scanned_files,
            "indexed_components": indexed_count,
            "reused_components": reused_count,
            "skipped_files": skipped_files,
            "component_count": len(active_fingerprints),
            "components": indexed_components,
        }

    async def search_components(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search indexed UI components using hybrid LanceDB retrieval."""
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        candidate_limit = max(int(top_k), 1) * 4
        hits = await self.memory.hybrid_search(
            query=normalized_query,
            top_k=candidate_limit,
            vector_weight=0.55,
            text_weight=0.45,
            filters={"tags": ["ui-component-library"]},
            min_score=0.0,
        )

        active_registry = self._load_active_registry()
        best_hits: Dict[str, Dict[str, Any]] = {}

        for hit in hits:
            parsed = self._parse_search_hit(hit)
            if parsed is None:
                continue

            metadata = parsed[0]
            parsed[1]["score"] = self._score_result_against_query(parsed[1], normalized_query)
            library_key = str(metadata.get("library_key") or "").strip()
            relative_path = str(metadata.get("relative_path") or "").strip()
            content_hash = str(metadata.get("content_hash") or "").strip()

            if library_key and relative_path:
                current_hash = active_registry.get(library_key, {}).get(relative_path)
                if current_hash and current_hash != content_hash:
                    continue

            dedupe_key = f"{library_key}:{relative_path}" if library_key or relative_path else parsed[1]["memory_id"]
            existing = best_hits.get(dedupe_key)
            if existing is None:
                best_hits[dedupe_key] = parsed[1]
                continue

            existing_created_at = str(existing.get("created_at") or "")
            incoming_created_at = str(parsed[1].get("created_at") or "")
            if incoming_created_at > existing_created_at or (
                incoming_created_at == existing_created_at
                and float(parsed[1].get("score") or 0.0) > float(existing.get("score") or 0.0)
            ):
                best_hits[dedupe_key] = parsed[1]

        ordered = sorted(
            best_hits.values(),
            key=lambda item: (float(item.get("score") or 0.0), str(item.get("created_at") or "")),
            reverse=True,
        )
        return ordered[: max(int(top_k), 1)]

    async def build_prompt_context(
        self,
        query: str,
        *,
        top_k: int = 2,
        max_code_chars: int = 2500,
    ) -> str:
        """Build a prompt block with retrieved UI component source snippets."""
        hits = await self.search_components(query, top_k=top_k)
        if not hits:
            return ""

        bounded_hits = hits[: max(int(top_k), 1)]
        per_component_budget = max(240, max(int(max_code_chars), 240) // max(len(bounded_hits), 1))
        lines: List[str] = ["", "# Retrieved UI Components"]

        for item in bounded_hits:
            component_name = str(item.get("component_name") or "unknown_component").strip()
            framework = str(item.get("framework") or "unknown").strip()
            relative_path = str(item.get("relative_path") or "").strip()
            usage_example = str(item.get("usage_example") or "").strip()
            code_content = str(item.get("code_content") or "").strip()
            language_hint = self._language_hint(relative_path)

            descriptor = f"- {component_name} [{framework}]"
            if relative_path:
                descriptor += f" from {relative_path}"
            lines.append(descriptor)
            if usage_example:
                lines.append(f"Usage: {usage_example}")
            if code_content:
                trimmed = self._truncate_code(code_content, per_component_budget)
                lines.append(f"```{language_hint}")
                lines.append(trimmed)
                lines.append("```")

        return "\n".join(lines)

    def _resolve_source_root(self, source_path: str) -> Path:
        raw_path = str(source_path or "").strip()
        if not raw_path:
            raise ValueError("source_path is required")
        normalized = normalize_path(raw_path)
        if not normalized.exists() or not normalized.is_dir():
            raise ValueError(f"UI library path is not a directory: {raw_path}")
        workspace_root = resolve_workspace_path(str(normalized))
        return ensure_within_root(workspace_root, normalized)

    def _normalize_store_mode(self, store_mode: str) -> str:
        normalized = str(store_mode or "full").strip().lower()
        if normalized not in {"full", "snippet"}:
            raise ValueError("store_mode must be 'full' or 'snippet'")
        return normalized

    def _normalize_extensions(self, include_extensions: Optional[List[str]]) -> set[str]:
        values = include_extensions or []
        normalized: set[str] = set()
        for item in values:
            text = str(item or "").strip().lower()
            if not text:
                continue
            if not text.startswith("."):
                text = f".{text}"
            normalized.add(text)
        return normalized or set(_DEFAULT_COMPONENT_EXTENSIONS)

    def _should_skip_path(self, relative_path: str) -> bool:
        lowered = str(relative_path or "").replace("\\", "/").lower()
        if not lowered:
            return True
        parts = [part for part in lowered.split("/") if part]
        if any(part in _DEFAULT_EXCLUDED_DIRS for part in parts):
            return True
        name = parts[-1] if parts else lowered
        if any(marker in name for marker in _STORY_OR_TEST_MARKERS):
            return True
        if name.endswith(".d.ts"):
            return True
        return False

    def _build_component_metadata(
        self,
        *,
        library_name: str,
        library_key: str,
        framework: str,
        relative_path: str,
        extension: str,
        code_content: str,
        content_hash: str,
        store_mode: str,
    ) -> Dict[str, Any]:
        component_name = self._infer_component_name(relative_path, code_content)
        exports = self._extract_exports(code_content)
        dependencies = self._extract_dependencies(code_content)
        asset_type = "style" if extension in _STYLE_EXTENSIONS else "component"
        usage_example = self._build_usage_example(component_name, framework, asset_type)

        return {
            "component_name": component_name,
            "library_name": library_name,
            "library_key": library_key,
            "framework": framework,
            "relative_path": relative_path,
            "extension": extension,
            "asset_type": asset_type,
            "exports": exports,
            "dependencies": dependencies,
            "usage_example": usage_example,
            "content_hash": content_hash,
            "content_mode": store_mode,
        }

    def _build_tags(self, metadata: Dict[str, Any]) -> List[str]:
        tags = {
            "ui-component-library",
            "ui-component",
            str(metadata.get("framework") or "unknown"),
            str(metadata.get("library_key") or "ui-library"),
            str(metadata.get("asset_type") or "component"),
        }
        component_name = str(metadata.get("component_name") or "").strip()
        if component_name:
            tags.add(self._slugify(component_name))
        return sorted(tag for tag in tags if tag)

    def _build_memory_payload(self, metadata: Dict[str, Any], code_content: str) -> str:
        code = str(code_content or "")
        if metadata.get("content_mode") == "snippet":
            code = self._truncate_code(code, 4000)
        header = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        return f"UI_COMPONENT {header}\nCODE:\n{code}"

    def _parse_search_hit(
        self,
        hit: HybridSearchResult,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        content = str(getattr(hit, "content", "") or "").strip()
        prefix = "UI_COMPONENT "
        if not content.startswith(prefix):
            return None

        metadata_blob, separator, code_blob = content[len(prefix):].partition("\nCODE:\n")
        if not separator:
            return None

        try:
            metadata = json.loads(metadata_blob)
        except Exception:
            return None
        if not isinstance(metadata, dict):
            return None

        return (
            metadata,
            {
                "memory_id": hit.memory_id,
                "component_name": str(metadata.get("component_name") or "unknown_component").strip(),
                "library_name": str(metadata.get("library_name") or "").strip(),
                "library_key": str(metadata.get("library_key") or "").strip(),
                "framework": str(metadata.get("framework") or "").strip(),
                "relative_path": str(metadata.get("relative_path") or "").strip(),
                "asset_type": str(metadata.get("asset_type") or "").strip(),
                "usage_example": str(metadata.get("usage_example") or "").strip(),
                "exports": metadata.get("exports", []),
                "dependencies": metadata.get("dependencies", []),
                "content_hash": str(metadata.get("content_hash") or "").strip(),
                "content_mode": str(metadata.get("content_mode") or "full").strip(),
                "code_content": code_blob,
                "score": hit.hybrid_score,
                "created_at": hit.created_at,
                "tags": list(hit.tags or []),
            },
        )

    def _load_active_registry(self) -> Dict[str, Dict[str, str]]:
        state = self._holon_service.get_holon_state(self.holon_id)
        libraries = state.get("ui_component_libraries")
        if not isinstance(libraries, dict):
            return {}

        active: Dict[str, Dict[str, str]] = {}
        for library_key, snapshot in libraries.items():
            if not isinstance(snapshot, dict):
                continue
            fingerprints = snapshot.get("fingerprints")
            if not isinstance(fingerprints, dict):
                continue
            active[str(library_key)] = {
                str(path).replace("\\", "/"): str(value)
                for path, value in fingerprints.items()
                if str(path).strip() and str(value).strip()
            }
        return active

    @staticmethod
    def _score_result_against_query(result: Dict[str, Any], query: str) -> float:
        base_score = float(result.get("score") or 0.0)
        query_text = str(query or "").strip().lower()
        if not query_text:
            return base_score

        tokens = [token for token in re.findall(r"[a-z0-9_]+", query_text) if len(token) >= 2]
        if not tokens:
            tokens = [segment.strip().lower() for segment in query_text.split() if segment.strip()]
        if not tokens and query_text:
            tokens = [query_text]

        searchable = " ".join(
            [
                str(result.get("component_name") or ""),
                str(result.get("relative_path") or ""),
                str(result.get("usage_example") or ""),
                " ".join(str(tag) for tag in result.get("tags", [])),
                str(result.get("code_content") or "")[:400],
            ]
        ).lower()
        name = str(result.get("component_name") or "").lower()

        lexical_bonus = 0.0
        for token in tokens:
            if token in searchable:
                lexical_bonus += 0.08
            if token and token in name:
                lexical_bonus += 0.18

        return base_score + lexical_bonus

    @staticmethod
    def _infer_component_name(relative_path: str, code_content: str) -> str:
        stem = Path(relative_path).stem
        if stem.endswith(".module"):
            stem = stem[:-7]

        patterns = [
            r"export\s+default\s+function\s+([A-Z][A-Za-z0-9_]*)",
            r"export\s+function\s+([A-Z][A-Za-z0-9_]*)",
            r"export\s+const\s+([A-Z][A-Za-z0-9_]*)",
            r"export\s+class\s+([A-Z][A-Za-z0-9_]*)",
            r"const\s+([A-Z][A-Za-z0-9_]*)\s*=\s*\(",
        ]
        for pattern in patterns:
            match = re.search(pattern, code_content)
            if match:
                return match.group(1)

        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
        return sanitized or "ui_component"

    @staticmethod
    def _extract_exports(code_content: str) -> List[str]:
        names: List[str] = []
        patterns = [
            r"export\s+default\s+function\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+class\s+([A-Za-z_][A-Za-z0-9_]*)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, code_content):
                names.append(match.group(1))

        grouped = re.findall(r"export\s*\{\s*([^}]+)\s*\}", code_content)
        for group in grouped:
            for token in group.split(","):
                cleaned = token.strip().split(" as ")[-1].strip()
                if cleaned:
                    names.append(cleaned)

        deduped: List[str] = []
        for name in names:
            if name not in deduped:
                deduped.append(name)
        return deduped[:10]

    @staticmethod
    def _extract_dependencies(code_content: str) -> List[str]:
        deps = re.findall(r"(?:from|import)\s+['\"]([^'\"]+)['\"]", code_content)
        deduped: List[str] = []
        for item in deps:
            value = str(item).strip()
            if not value or value in deduped:
                continue
            deduped.append(value)
        return deduped[:12]

    @staticmethod
    def _build_usage_example(component_name: str, framework: str, asset_type: str) -> str:
        normalized_name = str(component_name or "Component").strip() or "Component"
        if asset_type == "style":
            return f"import './{normalized_name}.css';"
        _ = str(framework or "unknown").strip().lower()
        return f"<{normalized_name} />"

    @staticmethod
    def _truncate_code(code_content: str, max_chars: int) -> str:
        text = str(code_content or "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _language_hint(relative_path: str) -> str:
        extension = Path(relative_path).suffix.lower()
        mapping = {
            ".tsx": "tsx",
            ".jsx": "jsx",
            ".vue": "vue",
            ".svelte": "svelte",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
        }
        return mapping.get(extension, "text")

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
        return slug or "ui-library"
