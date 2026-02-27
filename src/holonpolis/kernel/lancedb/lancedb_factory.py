"""LanceDB factory - unified database opening with physical isolation.

Key constraint: Each Holon has its own independent LanceDB directory.
Genesis has its own separate LanceDB directory.
"""

import threading
from pathlib import Path
from typing import Any, Dict, Optional

import lancedb
import structlog

from holonpolis.infrastructure.storage.path_guard import safe_join, validate_holon_id

from .schemas import (
    EPISODES_SCHEMA,
    DEFAULT_EMBEDDING_DIMENSION,
    GENESIS_EVOLUTIONS_SCHEMA,
    build_genesis_holons_schema,
    build_genesis_routes_schema,
    build_memories_schema,
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

    def create_table(
        self,
        name: str,
        schema,
        exist_ok: bool = True,
        enable_fts: bool = False,
        fts_columns: Optional[list] = None,
    ):
        """Create a table with the given schema.

        Args:
            name: 表名
            schema: PyArrow schema
            exist_ok: 如果表已存在是否忽略
            enable_fts: 是否启用全文搜索 (Full-Text Search)
            fts_columns: 要建立 FTS 索引的列名列表
        """
        try:
            table = self.connection.create_table(name, schema=schema, exist_ok=exist_ok)
            self._tables[name] = table

            # 创建 FTS 索引
            if enable_fts and fts_columns:
                try:
                    # 检查是否已有 FTS 索引
                    existing_indices = table.list_indices() if hasattr(table, "list_indices") else []
                    has_fts = any(
                        hasattr(idx, "index_type") and idx.index_type == "FTS"
                        for idx in existing_indices
                    )

                    if not has_fts:
                        # LanceDB FTS API: 使用字段名列表
                        table.create_fts_index(fts_columns)
                        logger.debug("fts_index_created", table=name, columns=fts_columns)
                except Exception as e:
                    # FTS 索引创建失败不阻塞表创建
                    logger.warning("fts_index_creation_failed", table=name, error=str(e))

            logger.debug("table_created", table=name, path=str(self.db_path), fts=enable_fts)
            return table
        except Exception as e:
            if "already exists" in str(e).lower() and exist_ok:
                return self.get_table(name)
            raise

    def table_exists(self, name: str) -> bool:
        """Check if a table exists."""
        return name in self.list_tables()

    def list_tables(self) -> list:
        """List all table names."""
        response = self.connection.list_tables()
        if hasattr(response, "tables"):
            return list(response.tables)
        return list(response)


class LanceDBFactory:
    """Factory for opening LanceDB connections with proper isolation.

    Enforces the rule: Each holon_id maps to exactly one independent database.
    """

    def __init__(self, base_path: Path, embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION):
        self.base_path = Path(base_path)
        self.embedding_dimension = int(embedding_dimension)
        self._cache: Dict[str, LanceDBConnection] = {}
        self._lock = threading.RLock()

    def _get_db_path(self, holon_id: str) -> Path:
        """Get the database path for a holon.

        Special cases:
        - "genesis" -> .holonpolis/genesis/memory/lancedb/
        - regular holon -> .holonpolis/holons/{holon_id}/memory/lancedb/
        """
        safe_holon_id = validate_holon_id(holon_id)
        if safe_holon_id == "genesis":
            return safe_join(self.base_path, "genesis", "memory", "lancedb")
        return safe_join(self.base_path, "holons", safe_holon_id, "memory", "lancedb")

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
        - memories: Retrievable condensed memories (with FTS index)
        - episodes: Full interaction records
        """
        conn = self.open(holon_id)

        # Create memories table with FTS 索引
        conn.create_table(
            "memories",
            build_memories_schema(self.embedding_dimension),
            exist_ok=True,
            enable_fts=True,
            fts_columns=["content"],  # 对 content 字段建立 FTS 索引
        )

        # Create episodes table
        conn.create_table("episodes", EPISODES_SCHEMA, exist_ok=True)

        logger.info("holon_tables_initialized", holon_id=holon_id, fts_enabled=True)

    def init_genesis_tables(self) -> None:
        """Initialize Genesis-specific tables.

        Creates:
        - holons: Index of all holons
        - routes: Routing decision history
        - evolutions: Evolution tracking
        """
        conn = self.open("genesis")

        # Create holons registry table
        conn.create_table(
            "holons",
            build_genesis_holons_schema(self.embedding_dimension),
            exist_ok=True,
        )

        # Create routes table
        conn.create_table(
            "routes",
            build_genesis_routes_schema(self.embedding_dimension),
            exist_ok=True,
        )

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

        from holonpolis.kernel.embeddings.default_embedder import get_embedder

        _factory = LanceDBFactory(base_path, embedding_dimension=get_embedder().dimension)
        return _factory


def reset_factory() -> None:
    """Reset the global factory (mainly for testing)."""
    global _factory
    with _factory_lock:
        if _factory is not None:
            _factory.close_all()
        _factory = None
