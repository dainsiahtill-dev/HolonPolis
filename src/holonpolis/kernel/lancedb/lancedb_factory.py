"""LanceDB factory - unified database opening with physical isolation.

Key constraint: Each Holon has its own independent LanceDB directory.
Genesis has its own separate LanceDB directory.
"""

import asyncio
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from contextlib import contextmanager

import lancedb
import structlog

from .schemas import (
    EPISODES_SCHEMA,
    GENESIS_EVOLUTIONS_SCHEMA,
    GENESIS_HOLONS_SCHEMA,
    GENESIS_ROUTES_SCHEMA,
    MEMORIES_SCHEMA,
)

logger = structlog.get_logger()


class LanceDBConnection:
    """Wrapper around a LanceDB connection with table management."""

    def __init__(self, db_path: Path, connection: lancedb.DBConnection):
        self.db_path = db_path
        self.connection = connection
        self._tables: Dict[str, Any] = {}

    def get_table(self, name: str):
        """Get a table by name."""
        if name not in self._tables:
            try:
                self._tables[name] = self.connection.open_table(name)
            except Exception as e:
                logger.error("table_open_failed", table=name, error=str(e))
                raise
        return self._tables[name]

    def create_table(self, name: str, schema, exist_ok: bool = True):
        """Create a table with the given schema."""
        try:
            table = self.connection.create_table(name, schema=schema, exist_ok=exist_ok)
            self._tables[name] = table
            logger.debug("table_created", table=name, path=str(self.db_path))
            return table
        except Exception as e:
            if "already exists" in str(e).lower() and exist_ok:
                return self.get_table(name)
            raise

    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""
        return name in self.connection.table_names()

    def list_tables(self) -> list:
        """List all table names."""
        return self.connection.table_names()


class LanceDBFactory:
    """Factory for opening LanceDB connections with proper isolation.

    Enforces the rule: Each holon_id maps to exactly one independent database.
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._cache: Dict[str, LanceDBConnection] = {}
        self._lock = threading.RLock()

    def _get_db_path(self, holon_id: str) -> Path:
        """Get the database path for a holon.

        Special cases:
        - "genesis" -> .holonpolis/genesis/memory/lancedb/
        - regular holon -> .holonpolis/holons/{holon_id}/memory/lancedb/
        """
        if holon_id == "genesis":
            return self.base_path / "genesis" / "memory" / "lancedb"
        else:
            return self.base_path / "holons" / holon_id / "memory" / "lancedb"

    def open(self, holon_id: str) -> LanceDBConnection:
        """Open a LanceDB connection for the given holon.

        This is the ONLY way to access a holon's database.
        Each holon gets physically isolated storage.
        """
        with self._lock:
            if holon_id in self._cache:
                return self._cache[holon_id]

            db_path = self._get_db_path(holon_id)
            db_path.mkdir(parents=True, exist_ok=True)

            try:
                connection = lancedb.connect(str(db_path))
                wrapper = LanceDBConnection(db_path, connection)
                self._cache[holon_id] = wrapper

                logger.info(
                    "lancedb_opened",
                    holon_id=holon_id,
                    path=str(db_path),
                )
                return wrapper
            except Exception as e:
                logger.error(
                    "lancedb_open_failed",
                    holon_id=holon_id,
                    path=str(db_path),
                    error=str(e),
                )
                raise

    def close(self, holon_id: str) -> None:
        """Close a connection and remove from cache."""
        with self._lock:
            if holon_id in self._cache:
                del self._cache[holon_id]
                logger.info("lancedb_closed", holon_id=holon_id)

    def close_all(self) -> None:
        """Close all connections."""
        with self._lock:
            for holon_id in list(self._cache.keys()):
                del self._cache[holon_id]
            logger.info("lancedb_all_closed")

    def init_holon_tables(self, holon_id: str) -> None:
        """Initialize standard tables for a holon.

        Creates:
        - memories: Retrievable condensed memories
        - episodes: Full interaction records
        """
        conn = self.open(holon_id)

        # Create memories table
        conn.create_table("memories", MEMORIES_SCHEMA, exist_ok=True)

        # Create episodes table
        conn.create_table("episodes", EPISODES_SCHEMA, exist_ok=True)

        logger.info("holon_tables_initialized", holon_id=holon_id)

    def init_genesis_tables(self) -> None:
        """Initialize Genesis-specific tables.

        Creates:
        - holons: Index of all holons
        - routes: Routing decision history
        - evolutions: Evolution tracking
        """
        conn = self.open("genesis")

        # Create holons registry table
        conn.create_table("holons", GENESIS_HOLONS_SCHEMA, exist_ok=True)

        # Create routes table
        conn.create_table("routes", GENESIS_ROUTES_SCHEMA, exist_ok=True)

        # Create evolutions table
        conn.create_table("evolutions", GENESIS_EVOLUTIONS_SCHEMA, exist_ok=True)

        logger.info("genesis_tables_initialized")

    def delete_holon_db(self, holon_id: str) -> None:
        """Delete a holon's entire database.

        WARNING: This is irreversible. All memories and episodes will be lost.
        """
        with self._lock:
            self.close(holon_id)
            db_path = self._get_db_path(holon_id)
            if db_path.exists():
                import shutil
                shutil.rmtree(db_path)
                logger.warning("holon_db_deleted", holon_id=holon_id, path=str(db_path))


# Global factory instance
_factory: Optional[LanceDBFactory] = None
_factory_lock = threading.Lock()


def get_lancedb_factory(base_path: Optional[Path] = None) -> LanceDBFactory:
    """Get or create the global LanceDB factory."""
    global _factory

    if _factory is not None:
        return _factory

    with _factory_lock:
        if _factory is not None:
            return _factory

        if base_path is None:
            from holonpolis.config import settings
            base_path = settings.holonpolis_root

        _factory = LanceDBFactory(base_path)
        return _factory


def reset_factory() -> None:
    """Reset the global factory (mainly for testing)."""
    global _factory
    with _factory_lock:
        if _factory is not None:
            _factory.close_all()
        _factory = None
