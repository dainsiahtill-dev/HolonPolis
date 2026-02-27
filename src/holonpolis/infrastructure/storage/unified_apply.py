"""Unified patch/file operation parser and applier for LLM responses.

Supports structured patch operations:
- SEARCH/REPLACE blocks
- Full file writes (FILE/CREATE)
- File deletion (DELETE)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class EditType(Enum):
    """Type of file edit operation."""
    SEARCH_REPLACE = auto()
    FULL_FILE = auto()
    CREATE = auto()
    DELETE = auto()


@dataclass
class FileOperation:
    """Unified file operation."""
    path: str
    edit_type: EditType
    search: Optional[str] = None
    replace: Optional[str] = None
    original_format: str = ""

    def is_valid(self) -> bool:
        """Check if operation is valid."""
        token = str(self.path or "").strip()
        if not token:
            return False
        if self.edit_type == EditType.SEARCH_REPLACE:
            return self.replace is not None
        if self.edit_type in (EditType.FULL_FILE, EditType.CREATE):
            return self.replace is not None
        return True


@dataclass
class ApplyResult:
    """Result of applying unified operations."""
    success: bool
    changed_files: List[str] = field(default_factory=list)
    failed_operations: List[Tuple[FileOperation, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _clean_code_content(content: str) -> str:
    """Strip common markdown wrappers."""
    if not content:
        return ""
    lines = content.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).rstrip("\n")


def _extract_path(header: str) -> str:
    """Extract file path from header."""
    raw = str(header or "").strip().strip("`")
    raw = re.sub(
        r"^(PATCH_FILE|FILE|CREATE|DELETE_FILE|DELETE)[:\s]+",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = raw.split("#", 1)[0].strip()
    raw = raw.split("//", 1)[0].strip()
    raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.strip()


def _normalize_empty_search(value: str) -> str:
    """Normalize empty search markers."""
    token = str(value or "").strip().lower()
    if token in ("<empty>", "<empty or missing>", "empty", "empty or missing"):
        return ""
    return value


def _header_requires_inner_file_path(header: str) -> bool:
    """Check if header requires inner file path."""
    normalized = str(header or "").strip().lower().replace(" ", "")
    if not normalized:
        return True
    return normalized in {
        "search/replace",
        "search_replace",
        "search-replace",
        "searchreplace",
        "sr",
    }


def _looks_like_search_replace_payload(text: str) -> bool:
    """Check if text looks like a search/replace payload."""
    payload = str(text or "")
    has_search = ("<<<<<<< SEARCH" in payload) or bool(
        re.search(r"(?im)^\s*SEARCH:?\s*$", payload)
    )
    has_replace = (">>>>>>> REPLACE" in payload) or bool(
        re.search(r"(?im)^\s*REPLACE:?\s*$", payload)
    )
    return has_search and has_replace


def parse_delete_operations(text: str) -> List[FileOperation]:
    """Parse DELETE_FILE/DELETE operations."""
    operations: List[FileOperation] = []
    if not text:
        return operations
    for match in re.finditer(r"(?m)^\s*(DELETE_FILE|DELETE)[:\s]+\s*([^\n]+?)\s*$", text):
        path = _extract_path(match.group(2))
        if not path:
            continue
        operations.append(
            FileOperation(
                path=path,
                edit_type=EditType.DELETE,
                original_format="DELETE_FILE",
            )
        )
    return operations


def parse_search_replace_blocks(text: str) -> List[FileOperation]:
    """Parse PATCH_FILE SEARCH/REPLACE blocks and compatible variants."""
    operations: List[FileOperation] = []
    if not text:
        return operations

    patch_sections = re.finditer(
        r"(?is)(?:^|\n)PATCH_FILE(?::|\s+)\s*([^\n]*)\n(.*?)(?:END PATCH_FILE|(?=\nPATCH_FILE(?::|\s+)|\Z))",
        text,
    )
    search_replace_pattern = re.compile(
        r"(?is)<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE"
    )
    search_replace_simple_pattern = re.compile(
        r"(?is)(?:^|\n)SEARCH:?\s*\n(.*?)\nREPLACE:?\s*\n(.*?)(?=\nSEARCH:?\s*\n|\nEND PATCH_FILE|\nEND FILE|\Z)"
    )
    inner_file_simple_pattern = re.compile(
        r"(?is)(?:^|\n)FILE:\s*([^\n]+?)\s*\nSEARCH:?\s*\n(.*?)\nREPLACE:?\s*\n(.*?)(?=\nEND FILE|\nFILE:\s*|\Z)"
    )

    for section in patch_sections:
        raw_header = str(section.group(1) or "")
        body = str(section.group(2) or "")
        file_path = _extract_path(raw_header)
        if _header_requires_inner_file_path(raw_header):
            inner_match = re.search(r"(?im)^\s*FILE:\s*(.+?)\s*$", body)
            if inner_match:
                file_path = _extract_path(inner_match.group(1))
        if not file_path:
            continue

        matched = False
        for sr in search_replace_pattern.finditer(body):
            matched = True
            operations.append(
                FileOperation(
                    path=file_path,
                    edit_type=EditType.SEARCH_REPLACE,
                    search=_normalize_empty_search(_clean_code_content(sr.group(1))),
                    replace=_clean_code_content(sr.group(2)),
                    original_format="PATCH_FILE+SEARCH_REPLACE",
                )
            )
        if matched:
            continue

        for sr in search_replace_simple_pattern.finditer(body):
            matched = True
            operations.append(
                FileOperation(
                    path=file_path,
                    edit_type=EditType.SEARCH_REPLACE,
                    search=_normalize_empty_search(_clean_code_content(sr.group(1))),
                    replace=_clean_code_content(sr.group(2)),
                    original_format="PATCH_FILE+SEARCH_REPLACE_SIMPLE",
                )
            )
        if matched:
            continue

        for sr in inner_file_simple_pattern.finditer(body):
            matched = True
            inner_path = _extract_path(sr.group(1)) or file_path
            operations.append(
                FileOperation(
                    path=inner_path,
                    edit_type=EditType.SEARCH_REPLACE,
                    search=_normalize_empty_search(_clean_code_content(sr.group(2))),
                    replace=_clean_code_content(sr.group(3)),
                    original_format="PATCH_FILE+FILE+SEARCH_REPLACE_SIMPLE",
                )
            )
        if matched:
            continue

        # Some weak models emit direct full-file content inside PATCH_FILE wrappers.
        full_content = _clean_code_content(body).strip()
        if full_content and not _looks_like_search_replace_payload(full_content):
            operations.append(
                FileOperation(
                    path=file_path,
                    edit_type=EditType.FULL_FILE,
                    replace=full_content,
                    original_format="PATCH_FILE_DIRECT_CONTENT",
                )
            )

    standalone_pattern = re.compile(
        r"(?is)(?:^|\n)([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)\s*\n<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE"
    )
    for match in standalone_pattern.finditer(text):
        path = _extract_path(match.group(1))
        if not path:
            continue
        operations.append(
            FileOperation(
                path=path,
                edit_type=EditType.SEARCH_REPLACE,
                search=_normalize_empty_search(_clean_code_content(match.group(2))),
                replace=_clean_code_content(match.group(3)),
                original_format="STANDALONE_SEARCH_REPLACE",
            )
        )

    return operations


def parse_full_file_blocks(text: str) -> List[FileOperation]:
    """Parse FILE/CREATE/PATCH_FILE full-file blocks."""
    operations: List[FileOperation] = []
    if not text:
        return operations

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        header = re.match(
            r"^\s*(FILE|CREATE|PATCH_FILE)[:\s]+\s*(.+?)\s*$",
            line,
            re.IGNORECASE,
        )
        if not header:
            i += 1
            continue

        marker = header.group(1).upper()
        file_path = _extract_path(header.group(2))
        i += 1
        if not file_path:
            continue

        content_lines: List[str] = []
        while i < len(lines):
            current = lines[i]
            if marker == "PATCH_FILE" and re.match(
                r"^\s*END PATCH_FILE\s*$",
                current,
                re.IGNORECASE,
            ):
                i += 1
                break
            if marker in ("FILE", "CREATE") and re.match(
                r"^\s*END\s+(FILE|CREATE)\s*$",
                current,
                re.IGNORECASE,
            ):
                i += 1
                break
            if re.match(
                r"^\s*(FILE|CREATE|PATCH_FILE|DELETE_FILE|DELETE)[:\s]+",
                current,
                re.IGNORECASE,
            ):
                break
            content_lines.append(current)
            i += 1

        content = _clean_code_content("\n".join(content_lines)).strip()
        if not content:
            continue
        if _looks_like_search_replace_payload(content):
            continue
        operations.append(
            FileOperation(
                path=file_path,
                edit_type=EditType.CREATE if marker == "CREATE" else EditType.FULL_FILE,
                replace=content,
                original_format=f"{marker}_BLOCK",
            )
        )

    return operations


def parse_all_operations(text: str) -> List[FileOperation]:
    """Parse operations from mixed LLM output."""
    if not text or text.strip() in ("", "NO_CHANGES"):
        return []

    operations: List[FileOperation] = []
    seen: set[tuple[str, EditType, str, str]] = set()
    parsers = (
        parse_delete_operations,
        parse_search_replace_blocks,
        parse_full_file_blocks,
    )
    for parser in parsers:
        try:
            parsed = parser(text)
        except Exception:
            parsed = []
        for op in parsed:
            if not op.is_valid():
                continue
            key = (
                op.path,
                op.edit_type,
                str(op.search or ""),
                str(op.replace or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            operations.append(op)
    return operations


def _resolve_workspace_file(workspace: str, rel_path: str) -> Tuple[bool, str]:
    """Resolve path with workspace safety check."""
    workspace_real = os.path.realpath(workspace)
    full_path = os.path.realpath(os.path.join(workspace_real, rel_path))
    try:
        safe = os.path.commonpath([workspace_real, full_path]) == workspace_real
    except Exception:
        safe = False
    return safe, full_path


def _find_fuzzy_match(content: str, search: str) -> Optional[str]:
    """Find a stable match when whitespace or blank lines drift."""
    search_lines = search.splitlines()
    content_lines = content.splitlines()
    if not search_lines:
        return None
    first = search_lines[0].strip()
    if not first:
        return None

    for idx, line in enumerate(content_lines):
        if line.strip() != first:
            continue
        if idx + len(search_lines) > len(content_lines):
            continue
        matched = True
        for offset, search_line in enumerate(search_lines):
            if content_lines[idx + offset].strip() != search_line.strip():
                matched = False
                break
        if matched:
            return "\n".join(content_lines[idx:idx + len(search_lines)])

    # Allow blank-line count drift while keeping non-empty lines in sequence.
    non_empty = [line.strip() for line in search_lines if line.strip()]
    if len(non_empty) < 2:
        return None
    pattern = re.escape(non_empty[0])
    for token in non_empty[1:]:
        pattern += r"(?:\r?\n[ \t]*)+" + re.escape(token)
    tolerant = re.search(pattern, content)
    if not tolerant:
        return None
    return str(tolerant.group(0))


def apply_operation(
    operation: FileOperation,
    workspace: str,
    *,
    fallback_to_full_file: bool = True,
) -> Tuple[bool, Optional[str], bool]:
    """Apply one operation.

    Returns:
        Tuple of (success, error_message, changed)
    """
    from holonpolis.infrastructure.storage.io_text import write_text_atomic

    safe, full_path = _resolve_workspace_file(workspace, operation.path)
    if not safe:
        return False, f"path outside workspace: {operation.path}", False

    try:
        if operation.edit_type == EditType.DELETE:
            if os.path.isfile(full_path):
                os.remove(full_path)
                return True, None, True
            return True, None, False

        os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)

        if operation.edit_type in (EditType.FULL_FILE, EditType.CREATE):
            next_content = str(operation.replace or "")
            old_content = ""
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8") as handle:
                    old_content = handle.read()
            if old_content == next_content:
                return True, None, False
            write_text_atomic(full_path, next_content)
            return True, None, True

        if operation.edit_type == EditType.SEARCH_REPLACE:
            current_content = ""
            if os.path.isfile(full_path):
                with open(full_path, "r", encoding="utf-8") as handle:
                    current_content = handle.read()
            search = str(operation.search or "")
            replace = str(operation.replace or "")
            if search == "":
                if current_content == replace:
                    return True, None, False
                write_text_atomic(full_path, replace)
                return True, None, True

            actual_search = search
            if actual_search not in current_content:
                fuzzy_match = _find_fuzzy_match(current_content, search)
                if fuzzy_match:
                    actual_search = fuzzy_match
                elif fallback_to_full_file:
                    if current_content == replace:
                        return True, None, False
                    write_text_atomic(full_path, replace)
                    return True, None, True
                else:
                    return False, f"search text not found: {operation.path}", False

            occurrences = current_content.count(actual_search)
            if occurrences > 1:
                return (
                    False,
                    f"ambiguous search text ({occurrences} matches): {operation.path}",
                    False,
                )
            next_content = current_content.replace(actual_search, replace, 1)
            if next_content == current_content:
                return True, None, False
            write_text_atomic(full_path, next_content)
            return True, None, True

        return False, f"unsupported edit type: {operation.edit_type}", False
    except Exception as exc:
        return False, str(exc), False


def apply_all_operations(
    text: str,
    workspace: str,
    *,
    fallback_to_full_file: bool = True,
    verbose: bool = False,
) -> ApplyResult:
    """Parse and apply all operations from response text.

    Args:
        text: LLM response text containing file operations
        workspace: Workspace root path
        fallback_to_full_file: Fallback to full file write on search failure
        verbose: Print verbose output

    Returns:
        ApplyResult with success status and details
    """
    operations = parse_all_operations(text)
    if not operations:
        return ApplyResult(success=False, errors=["no valid file operations found"])

    changed_files: List[str] = []
    failures: List[Tuple[FileOperation, str]] = []
    errors: List[str] = []

    for operation in operations:
        ok, error, changed = apply_operation(
            operation,
            workspace,
            fallback_to_full_file=fallback_to_full_file,
        )
        if ok:
            if changed and operation.path not in changed_files:
                changed_files.append(operation.path)
            continue
        message = error or "unknown apply error"
        failures.append((operation, message))
        errors.append(f"{operation.path}: {message}")
        if verbose:
            print(f"[unified_apply] failed {operation.path}: {message}", flush=True)

    if failures and changed_files:
        errors.append(
            f"partial_success: applied={len(changed_files)} failed={len(failures)}"
        )

    return ApplyResult(
        success=bool(changed_files),
        changed_files=changed_files,
        failed_operations=failures,
        errors=errors,
    )


__all__ = [
    "EditType",
    "FileOperation",
    "ApplyResult",
    "parse_delete_operations",
    "parse_search_replace_blocks",
    "parse_full_file_blocks",
    "parse_all_operations",
    "apply_operation",
    "apply_all_operations",
]
