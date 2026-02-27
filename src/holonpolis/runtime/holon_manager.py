"""Holon Manager - manages multiple Holon runtimes.

Caches runtimes for efficiency and handles lifecycle.
"""

from typing import Dict, List, Optional

import structlog

from holonpolis.domain import Blueprint
from holonpolis.runtime.holon_runtime import HolonRuntime
from holonpolis.services.holon_service import HolonService

logger = structlog.get_logger()


class HolonManager:
    """Manages Holon runtime instances.

    - Caches runtimes for reuse
    - Handles lifecycle (create, get, release)
    - Ensures proper resource cleanup
    """

    def __init__(self):
        self._cache: Dict[str, HolonRuntime] = {}
        self.holon_service = HolonService()

    def get_runtime(self, holon_id: str) -> HolonRuntime:
        """Get or create a runtime for a Holon.

        Args:
            holon_id: The Holon ID

        Returns:
            HolonRuntime instance
        """
        if holon_id in self._cache:
            logger.debug("runtime_cache_hit", holon_id=holon_id)
            return self._cache[holon_id]

        # Create new runtime
        runtime = HolonRuntime(holon_id)
        self._cache[holon_id] = runtime

        logger.debug("runtime_created", holon_id=holon_id)
        return runtime

    def release_runtime(self, holon_id: str) -> None:
        """Release a runtime from cache.

        This doesn't delete the Holon, just removes from runtime cache.
        """
        if holon_id in self._cache:
            del self._cache[holon_id]
            logger.debug("runtime_released", holon_id=holon_id)

    def clear_cache(self) -> None:
        """Clear all cached runtimes."""
        self._cache.clear()
        logger.debug("runtime_cache_cleared")

    def list_cached(self) -> List[str]:
        """List holon_ids with cached runtimes."""
        return list(self._cache.keys())

    async def chat(
        self,
        holon_id: str,
        user_message: str,
    ) -> Dict:
        """Convenience method: chat with a Holon.

        Args:
            holon_id: The Holon ID
            user_message: User's message

        Returns:
            Response dict
        """
        runtime = self.get_runtime(holon_id)
        return await runtime.chat(user_message)


# Global manager instance
_manager: Optional[HolonManager] = None


def get_holon_manager() -> HolonManager:
    """Get the global Holon manager."""
    global _manager
    if _manager is None:
        _manager = HolonManager()
    return _manager
