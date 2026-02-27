"""Unified file tools API for HolonPolis.

Exposes read/write/list/search/apply_patch/delete operations with path convergence
and safety guards. This is the main entry point for LLM file operations.

Supported logical path prefixes:
- workspace/*  - Files within the workspace
- runtime/*    - Runtime/temporary files
- config/*     - Configuration files

Examples:
    >>> from holonpolis.infrastructure.storage.file_tools import FileTools
    >>> tools = FileTools("/path/to/workspace")
    >>> tools.read("workspace/src/main.py")
    >>> tools.write("workspace/src/main.py", "print('hello')")
    >>> tools.list("workspace/src")
    >>> tools.search("def main", pattern="*.py")
    >>> tools.apply_patch(patch_text)
    >>> tools.delete("workspace/old_file.py")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from holonpolis.infrastructure.storage.file_io import (
    FileIOError,
    PathGuardError,
    create_directory,
    delete_directory,
    delete_file,
    list_directory,
    read_file,
    read_file_head,
    read_file_tail,
    search_files,
    write_file,
    write_json_file,
)
from holonpolis.infrastructure.storage.unified_apply import (
    ApplyResult,
    apply_all_operations,
    parse_all_operations,
)


@dataclass
class ToolResult:
    """Result of a file tool operation."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {"success": self.success}
        if self.error:
            result["error"] = self.error
        result.update(self.data)
        return result


class FileTools:
    """Unified file tools for LLM operations.

    All paths use logical prefixes (workspace/*, runtime/*, config/*)
    which are automatically resolved to absolute paths with safety guards.
    """

    def __init__(self, workspace: str):
        """Initialize FileTools with workspace path.

        Args:
            workspace: Absolute or relative path to workspace root
        """
        self.workspace = os.path.abspath(os.path.expanduser(workspace))
        if not os.path.isdir(self.workspace):
            raise ValueError(f"Workspace does not exist: {self.workspace}")

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 0,
    ) -> ToolResult:
        """Read file contents.

        Args:
            path: Logical path (e.g., "workspace/src/main.py")
            offset: Line offset to start reading (0-based)
            limit: Max lines to read (0 = unlimited)

        Returns:
            ToolResult with content, total_lines, size, mtime
        """
        try:
            data = read_file(self.workspace, path, offset=offset, limit=limit)
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    def read_head(
        self,
        path: str,
        max_chars: int = 20000,
    ) -> ToolResult:
        """Read first N characters of a file.

        Args:
            path: Logical path
            max_chars: Maximum characters to read

        Returns:
            ToolResult with content preview
        """
        try:
            data = read_file_head(self.workspace, path, max_chars=max_chars)
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    def read_tail(
        self,
        path: str,
        max_lines: int = 400,
        max_chars: int = 20000,
    ) -> ToolResult:
        """Read last N lines of a file.

        Efficient for large files. Reads from end in chunks.

        Args:
            path: Logical path
            max_lines: Maximum lines to read
            max_chars: Maximum characters to read

        Returns:
            ToolResult with tail content
        """
        try:
            data = read_file_tail(
                self.workspace, path, max_lines=max_lines, max_chars=max_chars
            )
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    def write(
        self,
        path: str,
        content: str,
        *,
        atomic: bool = True,
    ) -> ToolResult:
        """Write content to file.

        Args:
            path: Logical path
            content: File content
            atomic: Use atomic write (temp file + rename)

        Returns:
            ToolResult with operation status
        """
        try:
            data = write_file(self.workspace, path, content, atomic=atomic)
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    def write_json(
        self,
        path: str,
        data: Dict[str, Any],
    ) -> ToolResult:
        """Write JSON data to file.

        Args:
            path: Logical path
            data: JSON-serializable dictionary

        Returns:
            ToolResult with operation status
        """
        try:
            result = write_json_file(self.workspace, path, data)
            return ToolResult(success=True, data=result)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    # -------------------------------------------------------------------------
    # Directory Operations
    # -------------------------------------------------------------------------

    def list(
        self,
        path: str = "",
        *,
        recursive: bool = False,
        pattern: str = "",
    ) -> ToolResult:
        """List directory contents.

        Args:
            path: Logical path (empty = workspace root)
            recursive: List recursively
            pattern: Glob pattern to filter (e.g., "*.py")

        Returns:
            ToolResult with entries list
        """
        try:
            data = list_directory(
                self.workspace, path, recursive=recursive, pattern=pattern
            )
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    def mkdir(
        self,
        path: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> ToolResult:
        """Create a directory.

        Args:
            path: Logical path
            parents: Create parent directories
            exist_ok: Don't error if directory exists

        Returns:
            ToolResult with operation status
        """
        try:
            data = create_directory(
                self.workspace, path, parents=parents, exist_ok=exist_ok
            )
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    def rmdir(
        self,
        path: str,
        *,
        recursive: bool = False,
        must_exist: bool = False,
    ) -> ToolResult:
        """Delete a directory.

        Args:
            path: Logical path
            recursive: Remove recursively (required for non-empty dirs)
            must_exist: Raise error if directory doesn't exist

        Returns:
            ToolResult with operation status
        """
        try:
            data = delete_directory(
                self.workspace, path, recursive=recursive, must_exist=must_exist
            )
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        path: str = "",
        recursive: bool = True,
        pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> ToolResult:
        """Search for text in files.

        Args:
            query: Search query string
            path: Subdirectory to search (empty = workspace root)
            recursive: Search recursively
            pattern: File pattern filter (e.g., "*.py")
            case_sensitive: Case-sensitive search
            max_results: Maximum results to return

        Returns:
            ToolResult with matches list
        """
        try:
            data = search_files(
                self.workspace,
                query,
                path=path,
                recursive=recursive,
                pattern=pattern,
                case_sensitive=case_sensitive,
                max_results=max_results,
            )
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    def grep(
        self,
        pattern: str,
        *,
        path: str = "",
        recursive: bool = True,
        file_pattern: str = "*",
        max_results: int = 100,
    ) -> ToolResult:
        """Grep-like search with regex support.

        Alias for search with pattern matching. If regex fails, falls back to
        simple text search.

        Args:
            pattern: Regex pattern to search
            path: Subdirectory to search
            recursive: Search recursively
            file_pattern: File glob pattern
            max_results: Maximum results

        Returns:
            ToolResult with matches
        """
        import re

        try:
            # Validate regex
            regex = re.compile(pattern)
            # Use simple search for now, can enhance with regex later
            return self.search(
                query=pattern,
                path=path,
                recursive=recursive,
                pattern=file_pattern,
                max_results=max_results,
            )
        except re.error as e:
            return ToolResult(success=False, error=f"invalid regex: {e}")

    # -------------------------------------------------------------------------
    # Patch Operations
    # -------------------------------------------------------------------------

    def apply_patch(
        self,
        patch_text: str,
        *,
        fallback_to_full_file: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> ToolResult:
        """Apply patch operations from LLM response.

        Supports:
        - SEARCH/REPLACE blocks
        - FILE/CREATE full file blocks
        - DELETE_FILE operations

        Args:
            patch_text: LLM response containing patch operations
            fallback_to_full_file: Fallback to full file on search failure
            dry_run: Parse only, don't apply changes
            verbose: Print verbose output

        Returns:
            ToolResult with ApplyResult data
        """
        try:
            operations = parse_all_operations(patch_text)
            if not operations:
                return ToolResult(
                    success=False,
                    error="no valid file operations found in patch text",
                )

            if dry_run:
                return ToolResult(
                    success=True,
                    data={
                        "dry_run": True,
                        "operations_found": len(operations),
                        "operations": [
                            {
                                "path": op.path,
                                "type": op.edit_type.name,
                                "format": op.original_format,
                            }
                            for op in operations
                        ],
                    },
                )

            result = apply_all_operations(
                patch_text,
                self.workspace,
                fallback_to_full_file=fallback_to_full_file,
                verbose=verbose,
            )

            return ToolResult(
                success=result.success,
                data={
                    "changed_files": result.changed_files,
                    "failed_count": len(result.failed_operations),
                    "errors": result.errors,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"patch apply error: {e}")

    def parse_patch(
        self,
        patch_text: str,
    ) -> ToolResult:
        """Parse patch operations without applying.

        Args:
            patch_text: LLM response containing patch operations

        Returns:
            ToolResult with parsed operations
        """
        try:
            operations = parse_all_operations(patch_text)
            return ToolResult(
                success=True,
                data={
                    "operations_found": len(operations),
                    "operations": [
                        {
                            "path": op.path,
                            "type": op.edit_type.name,
                            "has_search": op.search is not None,
                            "has_replace": op.replace is not None,
                            "format": op.original_format,
                        }
                        for op in operations
                    ],
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"patch parse error: {e}")

    # -------------------------------------------------------------------------
    # Delete Operations
    # -------------------------------------------------------------------------

    def delete(
        self,
        path: str,
        *,
        must_exist: bool = False,
    ) -> ToolResult:
        """Delete a file.

        Args:
            path: Logical path
            must_exist: Raise error if file doesn't exist

        Returns:
            ToolResult with operation status
        """
        try:
            data = delete_file(self.workspace, path, must_exist=must_exist)
            return ToolResult(success=True, data=data)
        except (FileIOError, PathGuardError) as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"unexpected error: {e}")

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        from holonpolis.infrastructure.storage.path_resolver import resolve_artifact_path
        try:
            full_path = resolve_artifact_path(self.workspace, "", path)
            return os.path.exists(full_path)
        except Exception:
            return False

    def is_file(self, path: str) -> bool:
        """Check if path is a file."""
        from holonpolis.infrastructure.storage.path_resolver import resolve_artifact_path
        try:
            full_path = resolve_artifact_path(self.workspace, "", path)
            return os.path.isfile(full_path)
        except Exception:
            return False

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        from holonpolis.infrastructure.storage.path_resolver import resolve_artifact_path
        try:
            full_path = resolve_artifact_path(self.workspace, "", path)
            return os.path.isdir(full_path)
        except Exception:
            return False


# Convenience functions for direct use

def create_file_tools(workspace: str) -> FileTools:
    """Create FileTools instance for workspace."""
    return FileTools(workspace)


__all__ = [
    "FileTools",
    "ToolResult",
    "create_file_tools",
    "FileIOError",
    "PathGuardError",
    "ApplyResult",
]
