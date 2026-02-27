"""Tool orchestration layer for HolonPolis.

Provides tool contract definitions, parsing, security validation, and execution.
"""

from holonpolis.kernel.tools.tool_contract import (
    ToolSpec,
    canonicalize_tool_name,
    normalize_tool_args,
    validate_tool_step,
    supported_tool_names,
    read_tool_names,
    write_tool_names,
    exec_tool_names,
    list_tool_contracts,
    render_tool_contract_for_prompt,
)
from holonpolis.kernel.tools.tooling_models import (
    ToolChainStep,
    ToolChainResult,
)
from holonpolis.kernel.tools.tooling_security import (
    is_command_blocked,
    is_command_allowed,
    validate_command_security,
    validate_write_target,
    is_path_sensitive,
)
from holonpolis.kernel.tools.write_gate import (
    WriteGate,
    WriteGateResult,
    validate_write_scope,
)
from holonpolis.kernel.tools.existence_gate import (
    ExistenceGate,
    GateResult,
    check_mode,
)
from holonpolis.kernel.tools.tool_executor import (
    ToolExecutor,
    ToolExecutorError,
    ToolSecurityError,
    execute_tool,
)

__all__ = [
    # Tool contract
    "ToolSpec",
    "canonicalize_tool_name",
    "normalize_tool_args",
    "validate_tool_step",
    "supported_tool_names",
    "read_tool_names",
    "write_tool_names",
    "exec_tool_names",
    "list_tool_contracts",
    "render_tool_contract_for_prompt",
    # Tooling models
    "ToolChainStep",
    "ToolChainResult",
    # Security
    "is_command_blocked",
    "is_command_allowed",
    "validate_command_security",
    "validate_write_target",
    "is_path_sensitive",
    # Write gate
    "WriteGate",
    "WriteGateResult",
    "validate_write_scope",
    # Existence gate
    "ExistenceGate",
    "GateResult",
    "check_mode",
    # Tool executor
    "ToolExecutor",
    "ToolExecutorError",
    "ToolSecurityError",
    "execute_tool",
]
