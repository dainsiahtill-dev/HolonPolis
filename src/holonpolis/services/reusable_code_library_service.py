"""Reusable code library indexing and retrieval for per-Holon memory."""

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
from holonpolis.services.reusable_code_asset_policy import (
    is_supported_text_asset_path,
    normalize_extensions,
    normalize_library_kind,
)

logger = structlog.get_logger()


_STYLE_EXTENSIONS = {".css", ".scss", ".sass", ".less"}
_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml"}
_DOC_EXTENSIONS = {".md"}
_GENERIC_TAG = "reusable-code-library"
_LEGACY_UI_TAG = "ui-component-library"
_GENERIC_PREFIX = "CODE_ASSET "
_LEGACY_UI_PREFIX = "UI_COMPONENT "


class ReusableCodeLibraryService:
    """Index reusable source assets into per-Holon LanceDB memory."""

    def __init__(self, holon_id: str, memory_service: Optional[MemoryService] = None):
        self.holon_id = holon_id
        self.memory = memory_service or MemoryService(holon_id)
        self._holon_service = HolonService()

    async def index_local_library(
        self,
        source_path: str,
        *,
        library_name: str,
        library_kind: str = "code_asset",
        framework: str = "generic",
        store_mode: str = "full",
        include_extensions: Optional[List[str]] = None,
        max_file_bytes: int = 60000,
    ) -> Dict[str, Any]:
        """Index a local reusable code library into Holon memory."""
        normalized_mode = self._normalize_store_mode(store_mode)
        normalized_kind = normalize_library_kind(library_kind)
        normalized_framework = str(framework or "generic").strip().lower() or "generic"
        source_root = self._resolve_source_root(source_path)
        normalized_library_name = str(library_name or source_root.name).strip() or source_root.name
        library_key = self._slugify(normalized_library_name)
        extensions = normalize_extensions(include_extensions, normalized_kind)
        max_bytes = max(512, int(max_file_bytes))

        prior_fingerprints = self._load_prior_fingerprints(library_key)
        listing = list_directory(str(source_root), "", recursive=True)
        entries = listing.get("entries")
        if not isinstance(entries, list):
            entries = []

        indexed_assets: List[Dict[str, Any]] = []
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
            if not is_supported_text_asset_path(
                relative_path,
                library_kind=normalized_kind,
                include_extensions=extensions,
            ):
                skipped_files += 1
                continue

            extension = Path(relative_path).suffix.lower()
            file_size = int(entry.get("size") or 0)
            if file_size <= 0 or file_size > max_bytes:
                skipped_files += 1
                continue

            scanned_files += 1

            try:
                code_content = read_text_utf8(full_path, errors="replace").strip()
            except Exception as exc:
                logger.warning(
                    "reusable_code_library_read_failed",
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

            metadata = self._build_asset_metadata(
                library_name=normalized_library_name,
                library_key=library_key,
                library_kind=normalized_kind,
                framework=normalized_framework,
                relative_path=relative_path,
                extension=extension,
                code_content=code_content,
                content_hash=content_hash,
                store_mode=normalized_mode,
            )
            payload = self._build_memory_payload(metadata, code_content)
            await self.memory.remember(
                content=payload,
                kind=MemoryKind.PATTERN,
                tags=self._build_tags(metadata),
                importance=1.1,
                source_skill=f"code_library::{library_key}",
            )

            indexed_count += 1
            indexed_assets.append(
                {
                    "asset_name": metadata["asset_name"],
                    "relative_path": metadata["relative_path"],
                    "asset_type": metadata["asset_type"],
                    "exports": metadata["exports"],
                }
            )

        snapshot = {
            "library_name": normalized_library_name,
            "library_key": library_key,
            "library_kind": normalized_kind,
            "framework": normalized_framework,
            "source_path": str(source_root),
            "store_mode": normalized_mode,
            "max_file_bytes": max_bytes,
            "allowed_extensions": sorted(extensions),
            "asset_count": len(active_fingerprints),
            "indexed_assets": indexed_count,
            "reused_assets": reused_count,
            "skipped_files": skipped_files,
            "last_indexed_at": utc_now_iso(),
            "fingerprints": active_fingerprints,
        }
        self._holon_service.record_reusable_code_library_index(
            self.holon_id,
            library_key,
            snapshot=snapshot,
        )

        return {
            "status": "indexed",
            "holon_id": self.holon_id,
            "library_name": normalized_library_name,
            "library_key": library_key,
            "library_kind": normalized_kind,
            "framework": normalized_framework,
            "source_path": str(source_root),
            "store_mode": normalized_mode,
            "scanned_files": scanned_files,
            "indexed_assets": indexed_count,
            "reused_assets": reused_count,
            "skipped_files": skipped_files,
            "asset_count": len(active_fingerprints),
            "assets": indexed_assets,
        }

    async def search_assets(
        self,
        query: str,
        *,
        top_k: int = 3,
        library_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search reusable code assets using hybrid retrieval."""
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        filters: Dict[str, Any] = {"tags": [_GENERIC_TAG, _LEGACY_UI_TAG]}
        normalized_kind = ""
        if library_kind:
            normalized_kind = normalize_library_kind(library_kind)
            if normalized_kind == "ui_component":
                filters["tags"] = [_GENERIC_TAG, _LEGACY_UI_TAG, "ui-component"]
            else:
                filters["tags"] = [_GENERIC_TAG, normalized_kind]

        hits = await self.memory.hybrid_search(
            query=normalized_query,
            top_k=max(int(top_k), 1) * 4,
            vector_weight=0.55,
            text_weight=0.45,
            filters=filters,
            min_score=0.0,
        )

        active_registry = self._load_active_registry()
        best_hits: Dict[str, Dict[str, Any]] = {}

        for hit in hits:
            parsed = self._parse_search_hit(hit)
            if parsed is None:
                continue

            metadata = parsed[0]
            result = parsed[1]
            result["score"] = self._score_result_against_query(result, normalized_query)

            result_kind = str(result.get("library_kind") or "").strip()
            if normalized_kind and result_kind and result_kind != normalized_kind:
                continue

            library_key = str(metadata.get("library_key") or "").strip()
            relative_path = str(metadata.get("relative_path") or "").strip()
            content_hash = str(metadata.get("content_hash") or "").strip()

            current_hash = active_registry.get(library_key, {}).get(relative_path)
            if current_hash and current_hash != content_hash:
                continue

            dedupe_key = f"{library_key}:{relative_path}" if library_key or relative_path else result["memory_id"]
            existing = best_hits.get(dedupe_key)
            if existing is None:
                best_hits[dedupe_key] = result
                continue

            existing_created_at = str(existing.get("created_at") or "")
            incoming_created_at = str(result.get("created_at") or "")
            if incoming_created_at > existing_created_at or (
                incoming_created_at == existing_created_at
                and float(result.get("score") or 0.0) > float(existing.get("score") or 0.0)
            ):
                best_hits[dedupe_key] = result

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
        library_kind: Optional[str] = None,
        heading: str = "Retrieved Reusable Code Assets",
    ) -> str:
        """Build a prompt block with retrieved code assets."""
        hits = await self.search_assets(query, top_k=top_k, library_kind=library_kind)
        if not hits:
            return ""

        bounded_hits = hits[: max(int(top_k), 1)]
        per_asset_budget = max(240, max(int(max_code_chars), 240) // max(len(bounded_hits), 1))
        lines: List[str] = ["", f"# {heading}"]

        for item in bounded_hits:
            asset_name = str(item.get("asset_name") or "unknown_asset").strip()
            library_name = str(item.get("library_name") or "").strip()
            relative_path = str(item.get("relative_path") or "").strip()
            usage_example = str(item.get("usage_example") or "").strip()
            code_content = str(item.get("code_content") or "").strip()
            language_hint = self._language_hint(relative_path)

            descriptor = f"- {asset_name}"
            if library_name:
                descriptor += f" from {library_name}"
            if relative_path:
                descriptor += f" ({relative_path})"
            lines.append(descriptor)
            if usage_example:
                lines.append(f"Usage: {usage_example}")
            if code_content:
                lines.append(f"```{language_hint}")
                lines.append(self._truncate_code(code_content, per_asset_budget))
                lines.append("```")

        return "\n".join(lines)

    def _resolve_source_root(self, source_path: str) -> Path:
        raw_path = str(source_path or "").strip()
        if not raw_path:
            raise ValueError("source_path is required")
        normalized = normalize_path(raw_path)
        if not normalized.exists() or not normalized.is_dir():
            raise ValueError(f"Code library path is not a directory: {raw_path}")
        workspace_root = resolve_workspace_path(str(normalized))
        return ensure_within_root(workspace_root, normalized)

    @staticmethod
    def _normalize_store_mode(store_mode: str) -> str:
        normalized = str(store_mode or "full").strip().lower()
        if normalized not in {"full", "snippet"}:
            raise ValueError("store_mode must be 'full' or 'snippet'")
        return normalized

    def _load_prior_fingerprints(self, library_key: str) -> Dict[str, str]:
        state = self._holon_service.get_holon_state(self.holon_id)
        libraries = state.get("reusable_code_libraries")
        if not isinstance(libraries, dict):
            return {}
        snapshot = libraries.get(library_key)
        if not isinstance(snapshot, dict):
            return {}
        fingerprints = snapshot.get("fingerprints")
        if not isinstance(fingerprints, dict):
            return {}
        return {
            str(path).replace("\\", "/"): str(value)
            for path, value in fingerprints.items()
            if str(path).strip() and str(value).strip()
        }

    def _build_asset_metadata(
        self,
        *,
        library_name: str,
        library_key: str,
        library_kind: str,
        framework: str,
        relative_path: str,
        extension: str,
        code_content: str,
        content_hash: str,
        store_mode: str,
    ) -> Dict[str, Any]:
        asset_name = self._infer_asset_name(relative_path, code_content)
        exports = self._extract_exports(code_content)
        dependencies = self._extract_dependencies(code_content)
        asset_type = self._classify_asset_type(extension)
        usage_example = self._build_usage_example(asset_name, extension, library_kind, asset_type)
        return {
            "asset_name": asset_name,
            "library_name": library_name,
            "library_key": library_key,
            "library_kind": library_kind,
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
            _GENERIC_TAG,
            "reusable-code-asset",
            str(metadata.get("library_kind") or "code_asset"),
            str(metadata.get("framework") or "generic"),
            str(metadata.get("library_key") or "code-library"),
            str(metadata.get("asset_type") or "source"),
        }
        asset_name = str(metadata.get("asset_name") or "").strip()
        if asset_name:
            tags.add(self._slugify(asset_name))
        if str(metadata.get("library_kind") or "") == "ui_component":
            tags.add(_LEGACY_UI_TAG)
            tags.add("ui-component")
        return sorted(tag for tag in tags if tag)

    def _build_memory_payload(self, metadata: Dict[str, Any], code_content: str) -> str:
        code = str(code_content or "")
        if metadata.get("content_mode") == "snippet":
            code = self._truncate_code(code, 4000)
        header = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        return f"{_GENERIC_PREFIX}{header}\nCODE:\n{code}"

    def _parse_search_hit(
        self,
        hit: HybridSearchResult,
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        content = str(getattr(hit, "content", "") or "").strip()
        metadata: Dict[str, Any]
        code_blob: str

        if content.startswith(_GENERIC_PREFIX):
            metadata_blob, separator, code_blob = content[len(_GENERIC_PREFIX):].partition("\nCODE:\n")
            if not separator:
                return None
            try:
                metadata = json.loads(metadata_blob)
            except Exception:
                return None
        elif content.startswith(_LEGACY_UI_PREFIX):
            metadata_blob, separator, code_blob = content[len(_LEGACY_UI_PREFIX):].partition("\nCODE:\n")
            if not separator:
                return None
            try:
                raw_metadata = json.loads(metadata_blob)
            except Exception:
                return None
            if not isinstance(raw_metadata, dict):
                return None
            metadata = {
                "asset_name": raw_metadata.get("component_name"),
                "library_name": raw_metadata.get("library_name"),
                "library_key": raw_metadata.get("library_key"),
                "library_kind": "ui_component",
                "framework": raw_metadata.get("framework"),
                "relative_path": raw_metadata.get("relative_path"),
                "extension": raw_metadata.get("extension"),
                "asset_type": raw_metadata.get("asset_type"),
                "exports": raw_metadata.get("exports", []),
                "dependencies": raw_metadata.get("dependencies", []),
                "usage_example": raw_metadata.get("usage_example"),
                "content_hash": raw_metadata.get("content_hash"),
                "content_mode": raw_metadata.get("content_mode", "full"),
            }
        else:
            return None

        if not isinstance(metadata, dict):
            return None

        asset_name = str(metadata.get("asset_name") or metadata.get("component_name") or "unknown_asset").strip()
        return (
            metadata,
            {
                "memory_id": hit.memory_id,
                "asset_name": asset_name,
                "component_name": asset_name,
                "library_name": str(metadata.get("library_name") or "").strip(),
                "library_key": str(metadata.get("library_key") or "").strip(),
                "library_kind": str(metadata.get("library_kind") or "code_asset").strip(),
                "framework": str(metadata.get("framework") or "").strip(),
                "relative_path": str(metadata.get("relative_path") or "").strip(),
                "asset_type": str(metadata.get("asset_type") or "source").strip(),
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
        registries: Dict[str, Dict[str, str]] = {}

        for state_key in ("reusable_code_libraries", "ui_component_libraries"):
            libraries = state.get(state_key)
            if not isinstance(libraries, dict):
                continue
            for library_key, snapshot in libraries.items():
                if not isinstance(snapshot, dict):
                    continue
                fingerprints = snapshot.get("fingerprints")
                if not isinstance(fingerprints, dict):
                    continue
                registries[str(library_key)] = {
                    str(path).replace("\\", "/"): str(value)
                    for path, value in fingerprints.items()
                    if str(path).strip() and str(value).strip()
                }
        return registries

    @staticmethod
    def _infer_asset_name(relative_path: str, code_content: str) -> str:
        stem = Path(relative_path).stem
        if stem.endswith(".module"):
            stem = stem[:-7]

        patterns = [
            r"export\s+default\s+function\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+class\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"(?m)^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]",
            r"(?m)^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        ]
        for pattern in patterns:
            match = re.search(pattern, code_content)
            if match:
                return match.group(1)

        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
        return sanitized or "code_asset"

    @staticmethod
    def _extract_exports(code_content: str) -> List[str]:
        names: List[str] = []
        patterns = [
            r"export\s+default\s+function\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"export\s+class\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"(?m)^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:(]",
            r"(?m)^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
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
        return deduped[:12]

    @staticmethod
    def _extract_dependencies(code_content: str) -> List[str]:
        deps = re.findall(r"(?:from|import)\s+['\"]([^'\"]+)['\"]", code_content)
        dep_tokens = re.findall(r"^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)", code_content, flags=re.MULTILINE)
        combined = deps + dep_tokens
        deduped: List[str] = []
        for item in combined:
            value = str(item).strip()
            if not value or value in deduped:
                continue
            deduped.append(value)
        return deduped[:14]

    @staticmethod
    def _classify_asset_type(extension: str) -> str:
        if extension in _STYLE_EXTENSIONS:
            return "style"
        if extension in _CONFIG_EXTENSIONS:
            return "config"
        if extension in _DOC_EXTENSIONS:
            return "doc"
        return "source"

    @staticmethod
    def _build_usage_example(asset_name: str, extension: str, library_kind: str, asset_type: str) -> str:
        normalized_name = str(asset_name or "Asset").strip() or "Asset"
        if library_kind == "ui_component":
            if asset_type == "style":
                return f"import './{normalized_name}.css';"
            return f"<{normalized_name} />"
        if extension == ".py":
            return f"from module import {normalized_name}"
        if extension in {".ts", ".tsx", ".js", ".jsx"}:
            return f"import {{ {normalized_name} }} from './module';"
        return normalized_name

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
                str(result.get("asset_name") or ""),
                str(result.get("relative_path") or ""),
                str(result.get("usage_example") or ""),
                " ".join(str(tag) for tag in result.get("tags", [])),
                str(result.get("code_content") or "")[:400],
            ]
        ).lower()
        name = str(result.get("asset_name") or "").lower()

        lexical_bonus = 0.0
        for token in tokens:
            if token in searchable:
                lexical_bonus += 0.08
            if token and token in name:
                lexical_bonus += 0.18
        return base_score + lexical_bonus

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
            ".py": "python",
            ".ts": "ts",
            ".tsx": "tsx",
            ".js": "js",
            ".jsx": "jsx",
            ".json": "json",
            ".md": "md",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".vue": "vue",
            ".svelte": "svelte",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".kt": "kotlin",
            ".swift": "swift",
        }
        return mapping.get(extension, "text")

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
        return slug or "code-library"
