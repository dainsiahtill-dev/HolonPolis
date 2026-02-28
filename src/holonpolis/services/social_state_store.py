"""Persistent state store for social-layer services.

This module provides transactional SQLite persistence for:
- market state
- collaboration state

All artifacts are kept under `.holonpolis/genesis/social_state/`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict

from holonpolis.config import settings
from holonpolis.infrastructure.storage.path_guard import safe_join
from holonpolis.infrastructure.time_utils import utc_now_iso


class SocialStateStore:
    """SQLite-backed state store with atomic commits."""

    _MAX_SNAPSHOTS_PER_SCOPE = 100
    _SCOPES = ("market", "collaboration")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state_dir = safe_join(settings.holonpolis_root, "genesis", "social_state")
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._state_dir / "state.db"
        self._init_db()

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def db_path(self) -> Path:
        return self._db_path

    def load_market_state(self) -> Dict[str, Any]:
        return self._load_scope("market")

    def save_market_state(self, payload: Dict[str, Any]) -> None:
        self._save_scope("market", payload)

    def load_collaboration_state(self) -> Dict[str, Any]:
        return self._load_scope("collaboration")

    def save_collaboration_state(self, payload: Dict[str, Any]) -> None:
        self._save_scope("collaboration", payload)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS social_state_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_social_state_scope_created "
                "ON social_state_snapshots(scope, created_at DESC, id DESC)"
            )

    def _load_scope(self, scope: str) -> Dict[str, Any]:
        if scope not in self._SCOPES:
            return {}

        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM social_state_snapshots "
                "WHERE scope = ? ORDER BY id DESC LIMIT 1",
                (scope,),
            ).fetchone()
            if row is None:
                return {}
            try:
                data = json.loads(str(row[0]))
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    def _save_scope(self, scope: str, payload: Dict[str, Any]) -> None:
        if scope not in self._SCOPES:
            return
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        created_at = utc_now_iso()

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO social_state_snapshots(scope, created_at, payload_json) VALUES (?, ?, ?)",
                    (scope, created_at, payload_json),
                )
                self._compact_scope(conn, scope)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _compact_scope(self, conn: sqlite3.Connection, scope: str) -> None:
        """Keep only the latest snapshots per scope."""
        conn.execute(
            """
            DELETE FROM social_state_snapshots
            WHERE scope = ?
              AND id NOT IN (
                  SELECT id FROM social_state_snapshots
                  WHERE scope = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (scope, scope, self._MAX_SNAPSHOTS_PER_SCOPE),
        )
