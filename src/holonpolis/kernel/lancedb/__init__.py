"""LanceDB kernel module - physical memory isolation with Hybrid Search.

This module provides:
- Table schemas optimized for FTS + Vector hybrid retrieval
- Factory for creating isolated LanceDB instances per Holon
- Physical separation enforcement via path guards
"""

from holonpolis.kernel.lancedb.lancedb_factory import (
    LanceDBConnection,
    LanceDBFactory,
    get_lancedb_factory,
    reset_factory,
)
from holonpolis.kernel.lancedb.schemas import (
    DEFAULT_EMBEDDING_DIMENSION,
    EPISODES_SCHEMA,
    GENESIS_EVOLUTIONS_SCHEMA,
    GENESIS_HOLONS_SCHEMA,
    GENESIS_ROUTES_SCHEMA,
    MEMORIES_SCHEMA,
    MEMORIES_FTS_INDEX_CONFIG,
    build_genesis_holons_schema,
    build_genesis_routes_schema,
    build_memories_schema,
)

__all__ = [
    # Factory
    "LanceDBFactory",
    "LanceDBConnection",
    "get_lancedb_factory",
    "reset_factory",
    # Schemas
    "build_memories_schema",
    "build_genesis_holons_schema",
    "build_genesis_routes_schema",
    "EPISODES_SCHEMA",
    "GENESIS_EVOLUTIONS_SCHEMA",
    "MEMORIES_SCHEMA",
    "GENESIS_HOLONS_SCHEMA",
    "GENESIS_ROUTES_SCHEMA",
    "DEFAULT_EMBEDDING_DIMENSION",
    "MEMORIES_FTS_INDEX_CONFIG",
]
