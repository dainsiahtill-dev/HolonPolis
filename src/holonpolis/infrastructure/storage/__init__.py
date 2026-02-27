"""Storage infrastructure for HolonPolis.

Provides file I/O, path resolution, and patch operations with safety guards.
"""

from .path_guard import (
    InvalidArtifactPathError,
    ensure_within_root,
    normalize_path,
    safe_join,
    validate_holon_id,
)
from .file_tools import (
    FileTools,
    ToolResult,
    create_file_tools,
    FileIOError,
    PathGuardError,
)
from .unified_apply import (
    ApplyResult,
    FileOperation,
    EditType,
    apply_all_operations,
    parse_all_operations,
)
from .file_io import (
    read_file,
    write_file,
    delete_file,
    list_directory,
    search_files,
    create_directory,
)

__all__ = [
    # Path guard
    "InvalidArtifactPathError",
    "ensure_within_root",
    "normalize_path",
    "safe_join",
    "validate_holon_id",
    # File tools (main API)
    "FileTools",
    "ToolResult",
    "create_file_tools",
    "FileIOError",
    "PathGuardError",
    # Unified apply
    "ApplyResult",
    "FileOperation",
    "EditType",
    "apply_all_operations",
    "parse_all_operations",
    # File I/O
    "read_file",
    "write_file",
    "delete_file",
    "list_directory",
    "search_files",
    "create_directory",
]

