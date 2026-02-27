"""Security validation for tool execution.

Provides command blocking, pattern matching, and allowlist enforcement.
"""

import re
from typing import Optional, Set


# Default blocked command patterns (security sensitive)
DEFAULT_BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+~",
    r">\s*/",
    r"dd\s+if=.*of=/dev/",
    r"mkfs\.",
    r"fdisk",
    r":(){ :|:& };:",  # Fork bomb
    r"chmod\s+-R\s+777\s+/",
    r"chmod\s+-R\s+000\s+/",
    r"curl.*\|.*bash",
    r"wget.*\|.*bash",
    r"eval\s*\$",
    r"exec\s*\$",
    r"source\s*<\s*\(",
]

# Default allowed command prefixes for safe execution
DEFAULT_ALLOWED_COMMANDS = {
    "python",
    "python3",
    "pytest",
    "pip",
    "npm",
    "node",
    "cargo",
    "rustc",
    "go",
    "javac",
    "java",
    "git",
    "ls",
    "cat",
    "grep",
    "rg",
    "find",
    "head",
    "tail",
    "wc",
    "echo",
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
}


def is_command_blocked(command: str, blocked_patterns: Optional[list] = None) -> bool:
    """Check if a command matches blocked patterns.

    Args:
        command: Command string to check
        blocked_patterns: Optional list of regex patterns to check against

    Returns:
        True if command should be blocked
    """
    if not command:
        return True

    patterns = blocked_patterns or DEFAULT_BLOCKED_PATTERNS
    command_lower = command.lower()

    for pattern in patterns:
        try:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def is_command_allowed(
    command: str, allowed_commands: Optional[Set[str]] = None
) -> bool:
    """Check if a command is in the allowed whitelist.

    Args:
        command: Command string to check
        allowed_commands: Optional set of allowed command prefixes

    Returns:
        True if command is allowed
    """
    if not command:
        return False

    if is_command_blocked(command):
        return False

    allowed = allowed_commands or DEFAULT_ALLOWED_COMMANDS
    command_stripped = command.strip()

    for allowed_prefix in allowed:
        if command_stripped.startswith(allowed_prefix):
            return True

    return False


def validate_command_security(
    command: str,
    *,
    allowed_commands: Optional[Set[str]] = None,
    blocked_patterns: Optional[list] = None,
    require_allowed: bool = True,
) -> tuple[bool, str]:
    """Validate command against security policies.

    Args:
        command: Command to validate
        allowed_commands: Set of allowed command prefixes
        blocked_patterns: List of blocked regex patterns
        require_allowed: Whether to require allowlist membership

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not command or not command.strip():
        return False, "Command is empty"

    # Check blocked patterns first
    if is_command_blocked(command, blocked_patterns):
        return False, "Command matches blocked security pattern"

    # Check allowlist if required
    if require_allowed and not is_command_allowed(command, allowed_commands):
        return False, f"Command not in allowed list: {command.split()[0] if command else 'unknown'}"

    return True, ""


# File path patterns that are sensitive and should not be modified
SENSITIVE_PATH_PATTERNS = [
    r"^/etc/passwd",
    r"^/etc/shadow",
    r"^/etc/sudoers",
    r"^/etc/ssh/",
    r"^~/.ssh/",
    r"^/root/",
    r"\.env$",
    r"\.env\.local$",
    r"\.env\.production$",
    r"id_rsa",
    r"id_dsa",
    r"id_ecdsa",
    r"id_ed25519",
    r"\.pem$",
    r"\.key$",
    r"credentials",
    r"secret",
    r"password",
]


def is_path_sensitive(path: str) -> bool:
    """Check if a path is sensitive and should not be modified.

    Args:
        path: File path to check

    Returns:
        True if path is sensitive
    """
    if not path:
        return False

    path_lower = path.lower()

    for pattern in SENSITIVE_PATH_PATTERNS:
        try:
            if re.search(pattern, path_lower, re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def validate_write_target(
    path: str,
    workspace: str,
    allowed_extensions: Optional[Set[str]] = None,
) -> tuple[bool, str]:
    """Validate a write target path.

    Args:
        path: Target path
        workspace: Workspace root
        allowed_extensions: Optional set of allowed file extensions

    Returns:
        Tuple of (is_valid, error_message)
    """
    import os

    if not path:
        return False, "Path is empty"

    # Check sensitive paths
    if is_path_sensitive(path):
        return False, "Path is sensitive and cannot be modified"

    # Check extension if specified
    if allowed_extensions:
        ext = os.path.splitext(path)[1].lower()
        if ext not in allowed_extensions:
            return False, f"File extension '{ext}' not allowed"

    return True, ""
