"""Tool executor for HolonPolis.

Integrates file tools with security gates and provides unified tool execution.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Callable

from holonpolis.infrastructure.storage import FileTools, create_file_tools
from holonpolis.kernel.tools.tool_contract import (
    canonicalize_tool_name,
    normalize_tool_args,
    validate_tool_step,
    read_tool_names,
    write_tool_names,
)
from holonpolis.kernel.tools.tooling_security import (
    validate_command_security,
    validate_write_target,
)
from holonpolis.kernel.tools.write_gate import WriteGate
from holonpolis.kernel.tools.existence_gate import ExistenceGate


class ToolExecutorError(Exception):
    """Error during tool execution."""
    pass


class ToolSecurityError(ToolExecutorError):
    """Security policy violation."""
    pass


class ToolExecutor:
    """Execute tools with security validation and workspace isolation."""

    def __init__(
        self,
        workspace: str,
        *,
        allow_write: bool = False,
        allow_exec: bool = False,
        allowed_write_extensions: Optional[set] = None,
    ):
        """Initialize tool executor.

        Args:
            workspace: Workspace root directory
            allow_write: Whether write tools are allowed
            allow_exec: Whether exec tools are allowed
            allowed_write_extensions: Set of allowed file extensions for writes
        """
        self.workspace = os.path.abspath(workspace)
        self.allow_write = allow_write
        self.allow_exec = allow_exec
        self.allowed_extensions = allowed_write_extensions or {
            ".py", ".js", ".ts", ".tsx", ".jsx",
            ".md", ".txt", ".json", ".yaml", ".yml",
            ".html", ".css", ".scss", ".less",
            ".rs", ".go", ".java", ".kt", ".scala",
            ".c", ".cpp", ".h", ".hpp",
            ".rb", ".php", ".swift", ".m", ".mm",
        }
        self._file_tools: Optional[FileTools] = None

    @property
    def file_tools(self) -> FileTools:
        """Lazy-initialized file tools."""
        if self._file_tools is None:
            self._file_tools = create_file_tools(self.workspace)
        return self._file_tools

    def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        *,
        act_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool with security validation.

        Args:
            tool: Tool name
            args: Tool arguments
            act_files: Declared scope files for write validation

        Returns:
            Tool execution result
        """
        canonical = canonicalize_tool_name(tool)
        if not canonical:
            return {"ok": False, "error": f"Unknown tool: {tool}"}

        # Validate tool step
        valid, error_code, error_msg = validate_tool_step(canonical, args)
        if not valid:
            return {"ok": False, "error": f"[{error_code}] {error_msg}"}

        # Normalize args
        normalized = normalize_tool_args(canonical, args)

        # Check permissions
        category = self._get_tool_category(canonical)
        if category == "write" and not self.allow_write:
            return {"ok": False, "error": "Write tools not allowed for this holon"}
        if category == "exec" and not self.allow_exec:
            return {"ok": False, "error": "Exec tools not allowed for this holon"}

        # Execute based on tool type
        try:
            if category == "read":
                return self._execute_read_tool(canonical, normalized)
            elif category == "write":
                return self._execute_write_tool(canonical, normalized, act_files)
            elif category == "exec":
                return self._execute_exec_tool(canonical, normalized)
            else:
                return {"ok": False, "error": f"Unknown tool category: {category}"}
        except ToolSecurityError as e:
            return {"ok": False, "error": f"Security violation: {e}"}
        except Exception as e:
            return {"ok": False, "error": f"Execution error: {e}"}

    def _get_tool_category(self, tool: str) -> str:
        """Get tool category (read/write/exec)."""
        read_tools = set(read_tool_names())
        write_tools = set(write_tool_names())

        if tool in read_tools:
            return "read"
        elif tool in write_tools:
            return "write"
        else:
            return "exec"

    def _execute_read_tool(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a read tool."""
        if tool == "repo_tree":
            return self._exec_repo_tree(args)
        elif tool == "repo_rg":
            return self._exec_repo_rg(args)
        elif tool == "repo_read":
            return self._exec_repo_read(args)
        elif tool == "repo_read_head":
            return self._exec_repo_read_head(args)
        elif tool == "repo_read_tail":
            return self._exec_repo_read_tail(args)
        elif tool == "repo_search":
            return self._exec_repo_search(args)
        else:
            return {"ok": False, "error": f"Read tool not implemented: {tool}"}

    def _execute_write_tool(
        self,
        tool: str,
        args: Dict[str, Any],
        act_files: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Execute a write tool with validation."""
        # Validate target path
        file_path = args.get("file", "")
        if file_path:
            valid, error = validate_write_target(
                file_path, self.workspace, self.allowed_extensions
            )
            if not valid:
                raise ToolSecurityError(error)

        # Execute the tool
        if tool == "precision_edit":
            result = self._exec_precision_edit(args)
        elif tool == "repo_apply_patch":
            result = self._exec_apply_patch(args)
        elif tool == "repo_write":
            result = self._exec_repo_write(args)
        elif tool == "repo_delete":
            result = self._exec_repo_delete(args)
        elif tool == "repo_mkdir":
            result = self._exec_repo_mkdir(args)
        else:
            return {"ok": False, "error": f"Write tool not implemented: {tool}"}

        # Validate against act_files scope
        if result.get("ok") and act_files:
            changed = result.get("changed_files", [])
            gate_result = WriteGate.validate(changed, act_files)
            if not gate_result.allowed:
                return {
                    "ok": False,
                    "error": f"Write scope violation: {gate_result.reason}",
                    "changed_files": changed,
                }

        return result

    def _execute_exec_tool(self, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an exec tool with security validation."""
        if tool == "background_run":
            return self._exec_background_run(args)
        elif tool == "todo_read":
            return {"ok": True, "todos": []}  # Placeholder
        elif tool == "todo_write":
            return {"ok": True, "message": "Todos updated"}  # Placeholder
        elif tool == "task_create":
            return {"ok": True, "task_id": f"task_{id(args)}", "subject": args.get("subject")}
        elif tool == "task_update":
            return {"ok": True, "task_id": args.get("task_id"), "status": args.get("status")}
        else:
            return {"ok": False, "error": f"Exec tool not implemented: {tool}"}

    # -------------------------------------------------------------------------
    # Read Tool Implementations
    # -------------------------------------------------------------------------

    def _exec_repo_tree(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_tree (list directory)."""
        path = args.get("path", ".")
        result = self.file_tools.list(f"workspace/{path}", recursive=args.get("recursive", False))
        if result.success:
            return {"ok": True, "entries": result.data.get("entries", []), "total": result.data.get("total", 0)}
        return {"ok": False, "error": result.error}

    def _exec_repo_rg(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_rg (grep search)."""
        pattern = args.get("pattern", "")
        paths = args.get("paths", ["."])
        if isinstance(paths, str):
            paths = [paths]

        all_matches = []
        for p in paths:
            result = self.file_tools.grep(
                pattern,
                path=f"workspace/{p}",
                max_results=args.get("max_results", 100),
            )
            if result.success:
                all_matches.extend(result.data.get("matches", []))

        return {"ok": True, "matches": all_matches, "total": len(all_matches)}

    def _exec_repo_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_read (read file)."""
        file_path = args.get("file", "")
        result = self.file_tools.read(
            f"workspace/{file_path}",
            offset=args.get("offset", 0),
            limit=args.get("limit", 0),
        )
        if result.success:
            return {
                "ok": True,
                "content": result.data.get("content"),
                "total_lines": result.data.get("total_lines"),
                "path": file_path,
            }
        return {"ok": False, "error": result.error}

    def _exec_repo_read_head(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_read_head."""
        file_path = args.get("file", "")
        n = args.get("n", 50)
        result = self.file_tools.read_head(f"workspace/{file_path}", max_chars=n * 100)
        if result.success:
            lines = result.data.get("content", "").splitlines()[:n]
            return {"ok": True, "content": "\n".join(lines), "lines": len(lines), "path": file_path}
        return {"ok": False, "error": result.error}

    def _exec_repo_read_tail(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_read_tail."""
        file_path = args.get("file", "")
        n = args.get("n", 50)
        result = self.file_tools.read_tail(f"workspace/{file_path}", max_lines=n)
        if result.success:
            return {"ok": True, "content": result.data.get("content"), "lines": result.data.get("lines_read"), "path": file_path}
        return {"ok": False, "error": result.error}

    def _exec_repo_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_search."""
        query = args.get("query", "")
        path = args.get("path", "")
        result = self.file_tools.search(
            query,
            path=f"workspace/{path}" if path else "",
            pattern=args.get("pattern", "*"),
            max_results=args.get("max_results", 100),
        )
        if result.success:
            return {"ok": True, "matches": result.data.get("matches", []), "total": result.data.get("total", 0)}
        return {"ok": False, "error": result.error}

    # -------------------------------------------------------------------------
    # Write Tool Implementations
    # -------------------------------------------------------------------------

    def _exec_precision_edit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute precision_edit (search/replace)."""
        file_path = args.get("file", "")
        search = args.get("search", "")
        replace = args.get("replace", "")

        # Build patch
        patch = f"""
PATCH_FILE: {file_path}
<<<<<<< SEARCH
{search}
=======
{replace}
>>>>>>> REPLACE
END PATCH_FILE
"""
        result = self.file_tools.apply_patch(patch)
        if result.success:
            return {
                "ok": True,
                "changed_files": result.data.get("changed_files", []),
                "file": file_path,
            }
        return {"ok": False, "error": result.error}

    def _exec_apply_patch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_apply_patch."""
        patch = args.get("patch", "")
        result = self.file_tools.apply_patch(patch)
        if result.success:
            return {"ok": True, "changed_files": result.data.get("changed_files", [])}
        return {"ok": False, "error": result.error}

    def _exec_repo_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_write."""
        file_path = args.get("file", "")
        content = args.get("content", "")

        result = self.file_tools.write(f"workspace/{file_path}", content)
        if result.success:
            return {"ok": True, "changed_files": [file_path], "file": file_path}
        return {"ok": False, "error": result.error}

    def _exec_repo_delete(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_delete."""
        file_path = args.get("file", "")
        result = self.file_tools.delete(f"workspace/{file_path}")
        if result.success:
            return {"ok": True, "deleted": result.data.get("deleted", False), "file": file_path}
        return {"ok": False, "error": result.error}

    def _exec_repo_mkdir(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute repo_mkdir."""
        dir_path = args.get("dir", "")
        result = self.file_tools.mkdir(f"workspace/{dir_path}")
        if result.success:
            return {"ok": True, "dir": dir_path}
        return {"ok": False, "error": result.error}

    # -------------------------------------------------------------------------
    # Exec Tool Implementations
    # -------------------------------------------------------------------------

    def _exec_background_run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute background_run with command validation."""
        command = args.get("command", "")
        timeout = args.get("timeout", 300)

        # Validate command security
        valid, error = validate_command_security(command)
        if not valid:
            raise ToolSecurityError(error)

        # For now, return a placeholder - full implementation would use subprocess
        return {
            "ok": True,
            "task_id": f"bg_{hash(command) & 0xFFFFFFFF:08x}",
            "command": command,
            "status": "started",
            "timeout": timeout,
        }


# Convenience function for direct execution

def execute_tool(
    tool: str,
    args: Dict[str, Any],
    workspace: str,
    *,
    allow_write: bool = False,
    allow_exec: bool = False,
    act_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute a tool with the given configuration."""
    executor = ToolExecutor(
        workspace=workspace,
        allow_write=allow_write,
        allow_exec=allow_exec,
    )
    return executor.execute(tool, args, act_files=act_files)
