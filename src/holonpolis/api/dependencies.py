"""FastAPI dependencies."""

from typing import AsyncGenerator

from fastapi import Request

from holonpolis.services.genesis_service import GenesisService
from holonpolis.runtime.holon_manager import HolonManager, get_holon_manager


async def get_genesis_service() -> AsyncGenerator[GenesisService, None]:
    """Dependency for GenesisService."""
    service = GenesisService()
    yield service


async def get_holon_manager_dep() -> AsyncGenerator[HolonManager, None]:
    """Dependency for HolonManager."""
    yield get_holon_manager()
