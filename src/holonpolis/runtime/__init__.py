"""Runtime layer - generic Holon execution.

No specific Agent classes here. All Holons are runtime instances of Blueprints.
"""

from .holon_runtime import HolonRuntime
from .holon_manager import HolonManager

__all__ = ["HolonRuntime", "HolonManager"]
