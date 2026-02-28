"""Import-cycle regression tests."""

from __future__ import annotations

import subprocess
import sys


def test_runtime_module_imports_in_clean_interpreter():
    """Runtime import should not depend on prior services import order."""
    process = subprocess.run(
        [sys.executable, "-c", "from holonpolis.runtime.holon_runtime import HolonRuntime; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "ok" in process.stdout
