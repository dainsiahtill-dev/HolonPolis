"""Tool contract definitions for HolonPolis.

Defines tool specifications, aliases, argument normalization, and validation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


ToolSpec = Dict[str, Any]


# Tool specifications with categories: read, write, exec
_TOOL_SPECS: Dict[str, ToolSpec] = {
    # Read tools
    "repo_tree": {
        "category": "read",
        "aliases": ["repo_ls", "repo_list_dir", "list_dir", "ls_tree"],
        "arg_aliases": {
            "root": "path",
            "dir": "path",
            "directory": "path",
            "max": "max_entries",
            "limit": "max_entries",
        },
        "required_any": [],
        "required_doc": "args.path optional (default '.')",
    },
    "repo_rg": {
        "category": "read",
        "aliases": ["repo_search", "repo_grep", "search", "grep", "rg"],
        "arg_aliases": {
            "query": "pattern",
            "text": "pattern",
            "search": "pattern",
            "keyword": "pattern",
            "file": "path",
            "file_path": "path",
            "max": "max_results",
            "limit": "max_results",
            "n": "max_results",
            "g": "glob",
        },
        "required_any": [("pattern",)],
        "required_doc": "args.pattern required",
    },
    "repo_read": {
        "category": "read",
        "aliases": ["read_file", "cat", "read"],
        "arg_aliases": {
            "file_path": "file",
            "path": "file",
            "offset": "offset",
            "limit": "limit",
        },
        "required_any": [("file",)],
        "required_doc": "args.file required",
    },
    "repo_read_head": {
        "category": "read",
        "aliases": ["read_head", "head"],
        "arg_aliases": {
            "file_path": "file",
            "count": "n",
            "lines": "n",
            "max_lines": "n",
        },
        "required_any": [("file",)],
        "required_doc": "args.file required",
    },
    "repo_read_tail": {
        "category": "read",
        "aliases": ["read_tail", "tail"],
        "arg_aliases": {
            "file_path": "file",
            "count": "n",
            "lines": "n",
            "max_lines": "n",
        },
        "required_any": [("file",)],
        "required_doc": "args.file required",
    },
    "repo_search": {
        "category": "read",
        "aliases": ["search_files", "find", "grep_files"],
        "arg_aliases": {
            "query": "query",
            "text": "query",
            "pattern": "pattern",
            "path": "path",
            "recursive": "recursive",
        },
        "required_any": [("query",)],
        "required_doc": "args.query required",
    },
    # Write tools
    "precision_edit": {
        "category": "write",
        "aliases": ["apply_search_replace", "search_replace", "replace_text", "edit"],
        "arg_aliases": {"path": "file"},
        "required_any": [("file",), ("search",), ("replace",)],
        "required_doc": "args.file + args.search + args.replace",
    },
    "repo_apply_patch": {
        "category": "write",
        "aliases": ["apply_patch", "patch_apply", "patch"],
        "arg_aliases": {"patch_text": "patch", "patch": "patch"},
        "required_any": [("patch",)],
        "required_doc": "args.patch required",
    },
    "repo_write": {
        "category": "write",
        "aliases": ["write_file", "create_file"],
        "arg_aliases": {
            "path": "file",
            "content": "content",
        },
        "required_any": [("file",), ("content",)],
        "required_doc": "args.file + args.content",
    },
    "repo_delete": {
        "category": "write",
        "aliases": ["delete_file", "remove", "rm"],
        "arg_aliases": {"path": "file"},
        "required_any": [("file",)],
        "required_doc": "args.file required",
    },
    "repo_mkdir": {
        "category": "write",
        "aliases": ["create_dir", "mkdir"],
        "arg_aliases": {"path": "dir"},
        "required_any": [("dir",)],
        "required_doc": "args.dir required",
    },
    # Exec tools
    "background_run": {
        "category": "exec",
        "aliases": ["bg_run", "run", "exec"],
        "arg_aliases": {
            "cmd": "command",
            "working_dir": "cwd",
            "workdir": "cwd",
            "max_seconds": "timeout",
            "timeout_seconds": "timeout",
        },
        "required_any": [("command",)],
        "required_doc": "args.command required; args.timeout <= 3600",
    },
    "todo_read": {
        "category": "read",
        "aliases": ["todo_list", "list_todos"],
        "arg_aliases": {},
        "required_any": [],
        "required_doc": "no args required",
    },
    "todo_write": {
        "category": "exec",
        "aliases": ["todo_update", "set_todos"],
        "arg_aliases": {"tasks": "items"},
        "required_any": [("items",)],
        "required_doc": "args.items required",
    },
    "task_create": {
        "category": "exec",
        "aliases": ["create_task", "add_task"],
        "arg_aliases": {
            "title": "subject",
            "name": "subject",
        },
        "required_any": [("subject",)],
        "required_doc": "args.subject required",
    },
    "task_update": {
        "category": "exec",
        "aliases": ["update_task", "set_task_status"],
        "arg_aliases": {
            "id": "task_id",
            "status": "status",
        },
        "required_any": [("task_id",), ("status",)],
        "required_doc": "args.task_id + args.status required",
    },
}


def _build_alias_index() -> Dict[str, str]:
    """Build index of tool aliases to canonical names."""
    index: Dict[str, str] = {}
    for canonical, spec in _TOOL_SPECS.items():
        index[canonical.lower()] = canonical
        for alias in spec.get("aliases", []):
            alias_name = str(alias or "").strip().lower()
            if alias_name:
                index[alias_name] = canonical
    return index


_TOOL_ALIAS_INDEX = _build_alias_index()


def canonicalize_tool_name(name: str, *, keep_unknown: bool = True) -> str:
    """Convert alias to canonical tool name.

    Args:
        name: Tool name or alias
        keep_unknown: If True, return input if not found; else return empty

    Returns:
        Canonical tool name or original/empty depending on keep_unknown
    """
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    canonical = _TOOL_ALIAS_INDEX.get(cleaned.lower())
    if canonical:
        return canonical
    return cleaned if keep_unknown else ""


def _has_value(value: Any) -> bool:
    """Check if value is non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _normalize_rg_pattern(raw: Any) -> Any:
    """Normalize ripgrep pattern - convert space-separated to OR."""
    if not isinstance(raw, str):
        return raw
    pattern = raw.strip()
    if not pattern:
        return pattern
    # If query appears as space-separated keywords, convert to OR for repo_rg
    if " " not in pattern:
        return pattern
    if re.search(r"[|()[\]{}*+?\\]", pattern):
        return pattern
    terms = [part.strip() for part in pattern.split() if part.strip()]
    if len(terms) <= 1:
        return pattern
    return "|".join(re.escape(term) for term in terms)


def normalize_tool_args(tool: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize tool arguments using alias mapping.

    Args:
        tool: Tool name (canonical or alias)
        args: Raw arguments dict

    Returns:
        Normalized arguments dict
    """
    canonical_tool = canonicalize_tool_name(tool)
    if not isinstance(args, dict):
        args = {}
    spec = _TOOL_SPECS.get(canonical_tool, {})
    aliases = spec.get("arg_aliases", {}) if isinstance(spec.get("arg_aliases"), dict) else {}

    normalized: Dict[str, Any] = {}
    for key, value in args.items():
        raw_key = str(key or "").strip()
        if not raw_key:
            continue
        canonical_key = aliases.get(raw_key) or aliases.get(raw_key.lower()) or raw_key
        if canonical_key in normalized:
            continue
        normalized[canonical_key] = value

    # Special handling for specific tools
    if canonical_tool == "repo_rg":
        if not _has_value(normalized.get("pattern")) and _has_value(normalized.get("query")):
            normalized["pattern"] = normalized.get("query")
        if _has_value(normalized.get("pattern")):
            normalized["pattern"] = _normalize_rg_pattern(normalized.get("pattern"))
        if _has_value(normalized.get("path")) and not _has_value(normalized.get("paths")):
            normalized["paths"] = normalized.get("path")

    if canonical_tool in ("repo_read_head", "repo_read_tail"):
        if _has_value(normalized.get("lines")) and not _has_value(normalized.get("n")):
            normalized["n"] = normalized.get("lines")

    if canonical_tool == "background_run":
        timeout_value = normalized.get("timeout")
        if timeout_value is not None:
            try:
                timeout_int = int(timeout_value)
            except Exception:
                timeout_int = 0
            normalized["timeout"] = timeout_int
        elif _has_value(normalized.get("max_seconds")):
            try:
                timeout_int = int(normalized.get("max_seconds"))
            except Exception:
                timeout_int = 0
            normalized["timeout"] = timeout_int
        else:
            normalized["timeout"] = 300

    return normalized


def validate_tool_step(tool: str, args: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str], str]:
    """Validate a tool step.

    Args:
        tool: Tool name
        args: Tool arguments

    Returns:
        Tuple of (valid, error_code, error_message)
    """
    canonical_tool = canonicalize_tool_name(tool)
    if canonical_tool not in _TOOL_SPECS:
        return (
            False,
            "UNKNOWN_TOOL",
            f"Unsupported tool '{tool}'. Allowed: {', '.join(supported_tool_names())}",
        )

    normalized = normalize_tool_args(canonical_tool, args if isinstance(args, dict) else {})
    spec = _TOOL_SPECS.get(canonical_tool, {})
    required_any = spec.get("required_any", [])

    if isinstance(required_any, list):
        for group in required_any:
            group_values = group if isinstance(group, (list, tuple)) else [group]
            if not any(_has_value(normalized.get(str(key))) for key in group_values):
                required_text = " or ".join(str(key) for key in group_values)
                return (
                    False,
                    "INVALID_TOOL_ARGS",
                    f"{canonical_tool} missing required args: {required_text}",
                )

    if canonical_tool == "background_run":
        timeout_value = normalized.get("timeout")
        try:
            timeout_int = int(timeout_value) if timeout_value is not None else 300
        except Exception:
            timeout_int = 0
        if timeout_int <= 0:
            return (
                False,
                "INVALID_TOOL_ARGS",
                "background_run missing valid args: timeout must be > 0",
            )
        if timeout_int > 3600:
            return (
                False,
                "INVALID_TOOL_ARGS",
                "background_run missing valid args: timeout must be <= 3600",
            )

    return True, None, ""


def read_tool_names() -> List[str]:
    """Get list of read-only tool names."""
    return sorted(
        [name for name, spec in _TOOL_SPECS.items() if str(spec.get("category") or "") == "read"]
    )


def write_tool_names() -> List[str]:
    """Get list of write tool names."""
    return sorted(
        [name for name, spec in _TOOL_SPECS.items() if str(spec.get("category") or "") == "write"]
    )


def exec_tool_names() -> List[str]:
    """Get list of exec tool names."""
    return sorted(
        [name for name, spec in _TOOL_SPECS.items() if str(spec.get("category") or "") == "exec"]
    )


def supported_tool_names() -> List[str]:
    """Get all supported tool names."""
    return sorted(_TOOL_SPECS.keys())


def list_tool_contracts(categories: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """List tool contracts filtered by category.

    Args:
        categories: Optional categories to filter ("read", "write", "exec")

    Returns:
        List of tool contract dicts
    """
    if categories is None:
        allowed = {"read", "write", "exec"}
    else:
        allowed = {str(item).strip().lower() for item in categories if str(item or "").strip()}
    contracts: List[Dict[str, Any]] = []
    for name in sorted(_TOOL_SPECS.keys()):
        spec = _TOOL_SPECS[name]
        category = str(spec.get("category") or "").strip().lower()
        if category not in allowed:
            continue
        contracts.append(
            {
                "name": name,
                "category": category,
                "aliases": list(spec.get("aliases", [])),
                "required_doc": str(spec.get("required_doc") or ""),
            }
        )
    return contracts


def render_tool_contract_for_prompt(
    *,
    include_write_tools: bool = False,
    include_exec_tools: bool = True,
) -> str:
    """Render tool contracts as prompt text.

    Args:
        include_write_tools: Whether to include write tools
        include_exec_tools: Whether to include exec tools

    Returns:
        Formatted tool contract string
    """
    categories = ["read"]
    if include_write_tools:
        categories.append("write")
    if include_exec_tools:
        categories.append("exec")
    contracts = list_tool_contracts(categories)
    lines: List[str] = []
    lines.append("Tool Contract (authoritative):")
    lines.append("- Prefer canonical tool names exactly as listed below.")
    lines.append("- Runtime accepts aliases, but planner output SHOULD use canonical names.")
    for item in contracts:
        aliases = item.get("aliases") or []
        alias_text = ", ".join(str(alias) for alias in aliases[:4]) if aliases else "none"
        if len(aliases) > 4:
            alias_text += ", ..."
        lines.append(
            f"- {item['name']}: {item['required_doc']}; aliases={alias_text}"
        )
    return "\n".join(lines)
