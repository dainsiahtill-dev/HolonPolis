"""Change Verifier - Validate code changes and their effects.

Provides verification that:
- Files were actually changed (what changed)
- Changes are syntactically valid (whether it took effect)
- Evidence is collected (where is the evidence)
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from holonpolis.infrastructure.storage import FileTools, create_file_tools


@dataclass
class VerificationResult:
    """Result of change verification."""

    passed: bool
    verifier: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "verifier": self.verifier,
            "message": self.message,
            "details": self.details,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ChangeVerifier:
    """Verifies code changes and their effects."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.file_tools = create_file_tools(workspace)

    def verify_changes(
        self,
        changed_files: List[str],
        *,
        check_syntax: bool = True,
        check_imports: bool = False,
    ) -> List[VerificationResult]:
        """Verify a list of changed files.

        Args:
            changed_files: List of file paths that were changed
            check_syntax: Whether to check Python syntax
            check_imports: Whether to check imports (expensive)

        Returns:
            List of verification results
        """
        results = []

        for file_path in changed_files:
            # Check file exists
            if not self.file_tools.exists(f"workspace/{file_path}"):
                results.append(
                    VerificationResult(
                        passed=False,
                        verifier="existence",
                        message=f"File not found: {file_path}",
                    )
                )
                continue

            # Check syntax for Python files
            if check_syntax and file_path.endswith(".py"):
                result = self._verify_python_syntax(file_path)
                results.append(result)

            # Check for common issues
            result = self._verify_common_issues(file_path)
            if not result.passed:
                results.append(result)

        return results

    def _verify_python_syntax(self, file_path: str) -> VerificationResult:
        """Verify Python file syntax."""
        result = self.file_tools.read(f"workspace/{file_path}")

        if not result.success:
            return VerificationResult(
                passed=False,
                verifier="python_syntax",
                message=f"Failed to read file: {result.error}",
                errors=[result.error],
            )

        content = result.data.get("content", "")

        try:
            ast.parse(content)
            return VerificationResult(
                passed=True,
                verifier="python_syntax",
                message=f"Syntax OK: {file_path}",
                details={"file": file_path, "lines": len(content.splitlines())},
            )
        except SyntaxError as e:
            return VerificationResult(
                passed=False,
                verifier="python_syntax",
                message=f"Syntax error in {file_path}: {e.msg}",
                errors=[f"Line {e.lineno}: {e.text}"],
                details={
                    "file": file_path,
                    "line": e.lineno,
                    "column": e.offset,
                    "text": e.text,
                },
            )

    def _verify_common_issues(self, file_path: str) -> VerificationResult:
        """Check for common issues in any file type."""
        result = self.file_tools.read(f"workspace/{file_path}")

        if not result.success:
            return VerificationResult(
                passed=True,  # Pass if we can't read (handled elsewhere)
                verifier="common_issues",
                message=f"Could not check: {file_path}",
            )

        content = result.data.get("content", "")
        warnings = []

        # Check for merge conflict markers
        if "<<<<<<<" in content and ">>>>>>>" in content:
            return VerificationResult(
                passed=False,
                verifier="merge_conflicts",
                message=f"Merge conflict markers found in {file_path}",
                errors=["File contains unresolved merge conflict markers"],
            )

        # Check for TODO/FIXME in new code (warning only)
        todo_count = len(re.findall(r"TODO|FIXME|XXX", content, re.IGNORECASE))
        if todo_count > 0:
            warnings.append(f"Found {todo_count} TODO/FIXME markers")

        # Check for trailing whitespace
        trailing_ws = sum(1 for line in content.splitlines() if line.rstrip() != line)
        if trailing_ws > 0:
            warnings.append(f"Found {trailing_ws} lines with trailing whitespace")

        return VerificationResult(
            passed=True,
            verifier="common_issues",
            message=f"No common issues in {file_path}",
            warnings=warnings,
            details={"todo_count": todo_count, "trailing_ws_lines": trailing_ws},
        )

    def verify_patch_application(
        self,
        file_path: str,
        expected_search: str,
    ) -> VerificationResult:
        """Verify that a search/replace patch was applied correctly.

        Args:
            file_path: Path to the file
            expected_search: The search text that should have been replaced

        Returns:
            Verification result
        """
        result = self.file_tools.read(f"workspace/{file_path}")

        if not result.success:
            return VerificationResult(
                passed=False,
                verifier="patch_verification",
                message=f"Failed to read file: {result.error}",
            )

        content = result.data.get("content", "")

        # If search text is still present, patch wasn't applied
        if expected_search in content:
            return VerificationResult(
                passed=False,
                verifier="patch_verification",
                message=f"Patch not applied: search text still present in {file_path}",
                errors=["Search text was found in file, indicating patch was not applied"],
            )

        return VerificationResult(
            passed=True,
            verifier="patch_verification",
            message=f"Patch verified: search text not found (replaced) in {file_path}",
        )


# Convenience functions


def verify_file_changes(
    workspace: str,
    changed_files: List[str],
    *,
    check_syntax: bool = True,
) -> Tuple[bool, List[VerificationResult]]:
    """Verify file changes in a workspace.

    Args:
        workspace: Workspace root path
        changed_files: List of changed file paths
        check_syntax: Whether to check syntax

    Returns:
        Tuple of (all_passed, results)
    """
    verifier = ChangeVerifier(workspace)
    results = verifier.verify_changes(changed_files, check_syntax=check_syntax)
    all_passed = all(r.passed for r in results)
    return all_passed, results


def verify_syntax(file_path: str, content: Optional[str] = None) -> VerificationResult:
    """Verify Python syntax of a file or content.

    Args:
        file_path: Path to file (for reporting)
        content: File content (if None, reads from file_path)

    Returns:
        Verification result
    """
    if content is None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return VerificationResult(
                passed=False,
                verifier="python_syntax",
                message=f"Failed to read file: {e}",
                errors=[str(e)],
            )

    try:
        ast.parse(content)
        return VerificationResult(
            passed=True,
            verifier="python_syntax",
            message=f"Syntax OK: {file_path}",
            details={"file": file_path, "lines": len(content.splitlines())},
        )
    except SyntaxError as e:
        return VerificationResult(
            passed=False,
            verifier="python_syntax",
            message=f"Syntax error in {file_path}: {e.msg}",
            errors=[f"Line {e.lineno}: {e.text}"],
            details={
                "file": file_path,
                "line": e.lineno,
                "column": e.offset,
                "text": e.text,
            },
        )
