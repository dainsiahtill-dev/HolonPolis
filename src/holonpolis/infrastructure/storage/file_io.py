"""File I/O operations for HolonPolis.

Provides safe file read/write/list/search/delete operations with path guards.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Iterator

from holonpolis.infrastructure.storage.io_text import (
    read_file_safe,
    write_text_atomic,
    write_json_atomic,
    read_json_safe,
    format_mtime,
    build_file_status,
)
from holonpolis.infrastructure.storage.path_resolver import (
    resolve_artifact_path,
    resolve_safe_path,
    resolve_workspace_path,
)


class FileIOError(Exception):
    """File I/O error with context."""

    def __init__(self, message: str, path: str = "", operation: str = ""):
        self.path = path
        self.operation = operation
        super().__init__(f"[{operation}] {path}: {message}" if operation else message)


class PathGuardError(FileIOError):
    """Path traversal or security error."""
    pass


def _is_path_allowed(workspace: str, path: str) -> bool:
    """Check if path is within allowed directories."""
    try:
        abs_path = os.path.abspath(path)
        abs_workspace = os.path.abspath(workspace)
        allowed_roots = [abs_workspace]

        # Also allow common temp/cache directories
        temp_dirs = [
            os.path.abspath(os.path.expanduser("~/.holonpolis")),
            os.path.abspath(os.path.join(os.environ.get("LOCALAPPDATA", ""), "HolonPolis")) if os.name == "nt" else "",
            os.path.abspath(os.path.expanduser("~/.cache/holonpolis")),
        ]
        allowed_roots.extend([d for d in temp_dirs if d])

        for root in allowed_roots:
            try:
                if os.path.commonpath([root, abs_path]) == root:
                    return True
            except ValueError:
                continue
        return False
    except Exception:
        return False


def read_file(
    workspace: str,
    rel_path: str,
    *,
    offset: int = 0,
    limit: int = 0,
) -> Dict[str, Any]:
    """Read file contents with optional offset/limit.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path (workspace/*, runtime/*, config/*)
        offset: Line offset to start reading (0-based)
        limit: Max lines to read (0 = unlimited)

    Returns:
        Dict with content, path, size, and mtime info
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "read")

    if not os.path.isfile(full_path):
        raise FileIOError("File not found", rel_path, "read")

    content = read_file_safe(full_path)
    lines = content.splitlines()
    total_lines = len(lines)

    if offset > 0:
        lines = lines[offset:]
    if limit > 0:
        lines = lines[:limit]

    return {
        "path": rel_path,
        "full_path": full_path,
        "content": "\n".join(lines),
        "total_lines": total_lines,
        "offset": offset,
        "limit": limit,
        "size": os.path.getsize(full_path),
        "mtime": format_mtime(full_path),
    }


def read_file_head(
    workspace: str,
    rel_path: str,
    max_chars: int = 20000,
) -> Dict[str, Any]:
    """Read first N characters of a file.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        max_chars: Maximum characters to read

    Returns:
        Dict with content preview and file info
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "read_head")

    if not os.path.isfile(full_path):
        raise FileIOError("File not found", rel_path, "read_head")

    try:
        with open(full_path, "rb") as handle:
            data = handle.read(max_chars if max_chars and max_chars > 0 else 20000)
        from holonpolis.infrastructure.storage.io_text import _decode_text_bytes
        content = _decode_text_bytes(data)
    except Exception as e:
        raise FileIOError(str(e), rel_path, "read_head")

    return {
        "path": rel_path,
        "full_path": full_path,
        "content": content,
        "size": os.path.getsize(full_path),
        "preview_chars": len(content),
        "mtime": format_mtime(full_path),
    }


def read_file_tail(
    workspace: str,
    rel_path: str,
    max_lines: int = 400,
    max_chars: int = 20000,
) -> Dict[str, Any]:
    """Read last N lines of a file (efficient for large files).

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        max_lines: Maximum lines to read
        max_chars: Maximum characters to read

    Returns:
        Dict with tail content and file info
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "read_tail")

    if not os.path.isfile(full_path):
        raise FileIOError("File not found", rel_path, "read_tail")

    try:
        with open(full_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if file_size == 0:
                return {
                    "path": rel_path,
                    "full_path": full_path,
                    "content": "",
                    "size": 0,
                    "lines_read": 0,
                    "mtime": format_mtime(full_path),
                }

            pos = file_size
            block_size = 4096
            chunks = []
            lines_found = 0
            chars_read = 0

            target_lines = max_lines if max_lines and max_lines > 0 else None

            while pos > 0:
                read_size = block_size if pos >= block_size else pos
                pos -= read_size
                handle.seek(pos)
                chunk = handle.read(read_size)
                if not chunk:
                    break

                chunks.append(chunk)
                chars_read += len(chunk)
                lines_found += chunk.count(b"\n")

                if target_lines is not None and lines_found >= target_lines + 1:
                    if max_chars <= 0 or chars_read >= max_chars:
                        break

                if max_chars > 0 and chars_read >= max_chars * 2:
                    break

            data = b"".join(reversed(chunks))
            from holonpolis.infrastructure.storage.io_text import _decode_text_bytes
            text = _decode_text_bytes(data)

            lines = text.splitlines()
            if max_lines > 0 and len(lines) > max_lines:
                lines = lines[-max_lines:]
            content = "\n".join(lines)

            if max_chars > 0 and len(content) > max_chars:
                content = content[-max_chars:]

        return {
            "path": rel_path,
            "full_path": full_path,
            "content": content,
            "size": file_size,
            "lines_read": len(lines),
            "mtime": format_mtime(full_path),
        }
    except Exception as e:
        raise FileIOError(str(e), rel_path, "read_tail")


def write_file(
    workspace: str,
    rel_path: str,
    content: str,
    *,
    atomic: bool = True,
) -> Dict[str, Any]:
    """Write content to file.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        content: File content to write
        atomic: Use atomic write (temp file + rename)

    Returns:
        Dict with operation result
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "write")

    if not _is_path_allowed(workspace, full_path):
        raise PathGuardError("Path outside allowed directories", rel_path, "write")

    try:
        # Ensure parent directory exists
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if atomic:
            write_text_atomic(full_path, content)
        else:
            with open(full_path, "w", encoding="utf-8") as handle:
                handle.write(content or "")

        return {
            "success": True,
            "path": rel_path,
            "full_path": full_path,
            "bytes_written": len(content.encode("utf-8")),
            "mtime": format_mtime(full_path),
        }
    except Exception as e:
        raise FileIOError(str(e), rel_path, "write")


def write_json_file(
    workspace: str,
    rel_path: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Write JSON data to file.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        data: JSON-serializable data

    Returns:
        Dict with operation result
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "write_json")

    if not _is_path_allowed(workspace, full_path):
        raise PathGuardError("Path outside allowed directories", rel_path, "write_json")

    try:
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        write_json_atomic(full_path, data)

        return {
            "success": True,
            "path": rel_path,
            "full_path": full_path,
            "mtime": format_mtime(full_path),
        }
    except Exception as e:
        raise FileIOError(str(e), rel_path, "write_json")


def delete_file(
    workspace: str,
    rel_path: str,
    *,
    must_exist: bool = False,
) -> Dict[str, Any]:
    """Delete a file.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        must_exist: Raise error if file doesn't exist

    Returns:
        Dict with operation result
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "delete")

    if not _is_path_allowed(workspace, full_path):
        raise PathGuardError("Path outside allowed directories", rel_path, "delete")

    if not os.path.exists(full_path):
        if must_exist:
            raise FileIOError("File not found", rel_path, "delete")
        return {
            "success": True,
            "path": rel_path,
            "deleted": False,
            "reason": "not_found",
        }

    if os.path.isdir(full_path):
        raise FileIOError("Path is a directory, use delete_directory", rel_path, "delete")

    try:
        os.remove(full_path)
        return {
            "success": True,
            "path": rel_path,
            "full_path": full_path,
            "deleted": True,
        }
    except Exception as e:
        raise FileIOError(str(e), rel_path, "delete")


def list_directory(
    workspace: str,
    rel_path: str = "",
    *,
    recursive: bool = False,
    pattern: str = "",
) -> Dict[str, Any]:
    """List directory contents.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path (empty = workspace root)
        recursive: List recursively
        pattern: Glob pattern to filter (e.g., "*.py")

    Returns:
        Dict with entries (files and directories)
    """
    if not rel_path:
        full_path = os.path.abspath(workspace)
    else:
        try:
            full_path = resolve_safe_path(workspace, "", rel_path)
        except ValueError as e:
            raise PathGuardError(str(e), rel_path, "list")

    if not os.path.isdir(full_path):
        raise FileIOError("Directory not found", rel_path, "list")

    entries = []

    try:
        if recursive:
            for root, dirs, files in os.walk(full_path):
                rel_root = os.path.relpath(root, full_path)
                if rel_root == ".":
                    rel_root = ""

                for d in dirs:
                    dir_rel = os.path.join(rel_root, d) if rel_root else d
                    dir_full = os.path.join(root, d)
                    entries.append({
                        "name": d,
                        "path": dir_rel,
                        "full_path": dir_full,
                        "type": "directory",
                        "size": 0,
                    })

                for f in files:
                    if pattern and not _matches_pattern(f, pattern):
                        continue
                    file_rel = os.path.join(rel_root, f) if rel_root else f
                    file_full = os.path.join(root, f)
                    try:
                        size = os.path.getsize(file_full)
                    except Exception:
                        size = 0
                    entries.append({
                        "name": f,
                        "path": file_rel,
                        "full_path": file_full,
                        "type": "file",
                        "size": size,
                        "mtime": format_mtime(file_full),
                    })
        else:
            for entry in os.listdir(full_path):
                entry_full = os.path.join(full_path, entry)
                entry_rel = os.path.join(rel_path, entry) if rel_path else entry
                is_dir = os.path.isdir(entry_full)

                if pattern and not is_dir and not _matches_pattern(entry, pattern):
                    continue

                item = {
                    "name": entry,
                    "path": entry_rel,
                    "full_path": entry_full,
                    "type": "directory" if is_dir else "file",
                }

                if is_dir:
                    item["size"] = 0
                else:
                    try:
                        item["size"] = os.path.getsize(entry_full)
                        item["mtime"] = format_mtime(entry_full)
                    except Exception:
                        item["size"] = 0

                entries.append(item)

        return {
            "path": rel_path or ".",
            "full_path": full_path,
            "entries": entries,
            "total": len(entries),
        }
    except Exception as e:
        raise FileIOError(str(e), rel_path, "list")


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Check if filename matches glob pattern."""
    import fnmatch
    return fnmatch.fnmatch(filename.lower(), pattern.lower())


def search_files(
    workspace: str,
    query: str,
    *,
    path: str = "",
    recursive: bool = True,
    pattern: str = "*",
    case_sensitive: bool = False,
    max_results: int = 100,
) -> Dict[str, Any]:
    """Search for text in files.

    Args:
        workspace: Workspace root path
        query: Search query string
        path: Subdirectory to search (empty = workspace root)
        recursive: Search recursively
        pattern: File pattern to filter (e.g., "*.py")
        case_sensitive: Case-sensitive search
        max_results: Maximum results to return

    Returns:
        Dict with matches found
    """
    if not query:
        raise FileIOError("Search query is required", "", "search")

    if path:
        try:
            search_path = resolve_safe_path(workspace, "", path)
        except ValueError as e:
            raise PathGuardError(str(e), path, "search")
    else:
        search_path = os.path.abspath(workspace)

    if not os.path.isdir(search_path):
        raise FileIOError("Search path not found", path, "search")

    matches = []
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        if recursive:
            iterator = os.walk(search_path)
        else:
            iterator = [(search_path, [], [f for f in os.listdir(search_path)
                                           if os.path.isfile(os.path.join(search_path, f))])]

        for root, dirs, files in iterator:
            for filename in files:
                if not _matches_pattern(filename, pattern):
                    continue

                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, workspace)

                try:
                    content = read_file_safe(file_path)
                    if not content:
                        continue

                    # Simple text search
                    search_content = content if case_sensitive else content.lower()
                    search_query = query if case_sensitive else query.lower()

                    if search_query in search_content:
                        # Find line numbers
                        lines = content.splitlines()
                        for i, line in enumerate(lines, 1):
                            line_check = line if case_sensitive else line.lower()
                            if search_query in line_check:
                                matches.append({
                                    "path": rel_path,
                                    "full_path": file_path,
                                    "line": i,
                                    "content": line.strip(),
                                })
                                if len(matches) >= max_results:
                                    break

                    if len(matches) >= max_results:
                        break

                except Exception:
                    continue

            if len(matches) >= max_results:
                break

        return {
            "query": query,
            "path": path or ".",
            "pattern": pattern,
            "matches": matches,
            "total": len(matches),
            "truncated": len(matches) >= max_results,
        }

    except Exception as e:
        raise FileIOError(str(e), path, "search")


def create_directory(
    workspace: str,
    rel_path: str,
    *,
    parents: bool = True,
    exist_ok: bool = True,
) -> Dict[str, Any]:
    """Create a directory.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        parents: Create parent directories
        exist_ok: Don't error if directory exists

    Returns:
        Dict with operation result
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "mkdir")

    if not _is_path_allowed(workspace, full_path):
        raise PathGuardError("Path outside allowed directories", rel_path, "mkdir")

    try:
        if parents:
            os.makedirs(full_path, exist_ok=exist_ok)
        else:
            os.mkdir(full_path)

        return {
            "success": True,
            "path": rel_path,
            "full_path": full_path,
            "created": os.path.isdir(full_path),
        }
    except FileExistsError:
        if not exist_ok:
            raise FileIOError("Directory already exists", rel_path, "mkdir")
        return {
            "success": True,
            "path": rel_path,
            "full_path": full_path,
            "created": False,
            "reason": "already_exists",
        }
    except Exception as e:
        raise FileIOError(str(e), rel_path, "mkdir")


def delete_directory(
    workspace: str,
    rel_path: str,
    *,
    recursive: bool = False,
    must_exist: bool = False,
) -> Dict[str, Any]:
    """Delete a directory.

    Args:
        workspace: Workspace root path
        rel_path: Relative or logical path
        recursive: Remove recursively (required for non-empty dirs)
        must_exist: Raise error if directory doesn't exist

    Returns:
        Dict with operation result
    """
    try:
        full_path = resolve_safe_path(workspace, "", rel_path)
    except ValueError as e:
        raise PathGuardError(str(e), rel_path, "rmdir")

    if not _is_path_allowed(workspace, full_path):
        raise PathGuardError("Path outside allowed directories", rel_path, "rmdir")

    # Prevent deleting workspace root
    if os.path.abspath(full_path) == os.path.abspath(workspace):
        raise PathGuardError("Cannot delete workspace root", rel_path, "rmdir")

    if not os.path.exists(full_path):
        if must_exist:
            raise FileIOError("Directory not found", rel_path, "rmdir")
        return {
            "success": True,
            "path": rel_path,
            "deleted": False,
            "reason": "not_found",
        }

    if not os.path.isdir(full_path):
        raise FileIOError("Path is not a directory", rel_path, "rmdir")

    try:
        if recursive:
            import shutil
            shutil.rmtree(full_path)
        else:
            os.rmdir(full_path)

        return {
            "success": True,
            "path": rel_path,
            "full_path": full_path,
            "deleted": True,
        }
    except OSError as e:
        raise FileIOError(str(e), rel_path, "rmdir")
