"""Persistent state store for social-layer services.

This module provides atomic JSON persistence for:
- market state
- collaboration state

All files are kept under `.holonpolis/genesis/social_state/`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from holonpolis.config import settings
from holonpolis.infrastructure.storage.path_guard import safe_join


class SocialStateStore:
    """JSON-backed state store with atomic writes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state_dir = safe_join(settings.holonpolis_root, "genesis", "social_state")
        self._state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def load_market_state(self) -> Dict[str, Any]:
        return self._load_json(self._state_dir / "market_state.json")

    def save_market_state(self, payload: Dict[str, Any]) -> None:
        self._save_json(self._state_dir / "market_state.json", payload)

    def load_collaboration_state(self) -> Dict[str, Any]:
        return self._load_json(self._state_dir / "collaboration_state.json")

    def save_collaboration_state(self, payload: Dict[str, Any]) -> None:
        self._save_json(self._state_dir / "collaboration_state.json", payload)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with self._lock:
            if not path.exists():
                return {}
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    def _save_json(self, path: Path, payload: Dict[str, Any]) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(path)
