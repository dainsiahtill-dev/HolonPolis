"""Holon Service - lifecycle management for Holons.

Creates and manages Holon directories, blueprints, and databases.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.config import settings
from holonpolis.domain import Blueprint
from holonpolis.domain.errors import HolonNotFoundError
from holonpolis.infrastructure.time_utils import utc_now_iso
from holonpolis.infrastructure.storage.path_guard import safe_join, validate_holon_id
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory

logger = structlog.get_logger()


_STATUS_ACTIVE = "active"
_STATUS_PENDING = "pending"
_STATUS_FROZEN = "frozen"
_VALID_STATUSES = frozenset({_STATUS_ACTIVE, _STATUS_PENDING, _STATUS_FROZEN})


def _deep_merge_dicts(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict patches while preserving unrelated keys."""
    result = dict(base)
    for key, value in patch.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(current, value)
        else:
            result[key] = value
    return result


class HolonUnavailableError(RuntimeError):
    """Raised when a Holon cannot accept runtime work in its current state."""

    def __init__(self, holon_id: str, status: str, action: str):
        self.holon_id = str(holon_id)
        self.status = str(status or _STATUS_ACTIVE)
        self.action = str(action or "run")
        super().__init__(
            f"Holon '{self.holon_id}' is {self.status} and cannot {self.action} until it returns to active."
        )


class HolonService:
    """Service for managing Holon lifecycle.

    Responsibilities:
    - Create Holon directories
    - Initialize per-Holon LanceDB
    - Store/load blueprints
    - Freeze/resume Holons
    """

    def __init__(self):
        self.factory = get_lancedb_factory()
        self.holons_path = settings.holons_path

    def _get_holon_path(self, holon_id: str) -> Path:
        """Get the base path for a Holon."""
        safe_holon_id = validate_holon_id(holon_id)
        return safe_join(self.holons_path, safe_holon_id)

    def _get_blueprint_path(self, holon_id: str) -> Path:
        """Get the blueprint.json path for a Holon."""
        return self._get_holon_path(holon_id) / "blueprint.json"

    def _get_state_path(self, holon_id: str) -> Path:
        """Get the state.json path for a Holon."""
        return self._get_holon_path(holon_id) / "state.json"

    def _read_state(self, holon_id: str) -> Dict[str, Any]:
        """Read state.json, defaulting to active if missing/corrupt."""
        state_path = self._get_state_path(holon_id)
        if not state_path.exists():
            return {"status": _STATUS_ACTIVE}
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"status": _STATUS_ACTIVE}
        if not isinstance(payload, dict):
            return {"status": _STATUS_ACTIVE}
        status = str(payload.get("status") or _STATUS_ACTIVE).strip().lower()
        if status not in _VALID_STATUSES:
            status = _STATUS_ACTIVE
        payload["status"] = status
        return payload

    def _write_state(self, holon_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a sanitized state payload."""
        payload = dict(state or {})
        status = str(payload.get("status") or _STATUS_ACTIVE).strip().lower()
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid Holon status: {status!r}")
        payload["status"] = status
        payload["updated_at"] = utc_now_iso()
        state_path = self._get_state_path(holon_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def get_holon_status(self, holon_id: str) -> str:
        """Return the Holon's runtime status."""
        return str(self._read_state(holon_id).get("status") or _STATUS_ACTIVE)

    def get_holon_state(self, holon_id: str) -> Dict[str, Any]:
        """Return the persisted Holon state payload."""
        return dict(self._read_state(holon_id))

    def set_holon_status(
        self,
        holon_id: str,
        status: str,
        *,
        reason: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a Holon runtime status transition."""
        normalized = str(status or _STATUS_ACTIVE).strip().lower()
        existing = self._read_state(holon_id)
        payload: Dict[str, Any] = dict(existing)
        payload["status"] = normalized
        if reason:
            payload["reason"] = str(reason)
        if isinstance(details, dict) and details:
            payload["details"] = dict(details)

        if normalized == _STATUS_PENDING:
            payload["pending_at"] = utc_now_iso()
            payload.pop("resumed_at", None)
            payload.pop("frozen_at", None)
        elif normalized == _STATUS_FROZEN:
            payload["frozen_at"] = utc_now_iso()
            payload.pop("pending_at", None)
            payload.pop("resumed_at", None)
        else:
            payload["resumed_at"] = utc_now_iso()
            payload.pop("pending_at", None)
            payload.pop("frozen_at", None)

        saved = self._write_state(holon_id, payload)
        logger.info("holon_status_changed", holon_id=holon_id, status=normalized, reason=reason or None)
        return saved

    def mark_pending(
        self,
        holon_id: str,
        *,
        reason: str = "evolving",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mark a Holon as pending while it upgrades."""
        return self.set_holon_status(holon_id, _STATUS_PENDING, reason=reason, details=details)

    def mark_active(
        self,
        holon_id: str,
        *,
        reason: str = "ready",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mark a Holon as active/runnable."""
        return self.set_holon_status(holon_id, _STATUS_ACTIVE, reason=reason, details=details)

    def update_evolution_audit(
        self,
        holon_id: str,
        patch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge audit metadata into state.json without disturbing status fields."""
        existing = self._read_state(holon_id)
        audit = existing.get("evolution_audit")
        if not isinstance(audit, dict):
            audit = {}
        if isinstance(patch, dict) and patch:
            audit = _deep_merge_dicts(audit, patch)
        audit["updated_at"] = utc_now_iso()
        existing["evolution_audit"] = audit
        return self._write_state(holon_id, existing)

    def record_self_reflection(
        self,
        holon_id: str,
        snapshot: Optional[Dict[str, Any]] = None,
        *,
        max_history: int = 10,
    ) -> Dict[str, Any]:
        """Persist the latest self-reflection snapshot with bounded history."""
        existing = self._read_state(holon_id)
        reflection = existing.get("self_reflection")
        if not isinstance(reflection, dict):
            reflection = {}

        history = reflection.get("history")
        if not isinstance(history, list):
            history = []

        snapshot_payload = dict(snapshot or {})
        reflection_id = str(snapshot_payload.get("reflection_id") or "").strip()
        created_at = str(snapshot_payload.get("created_at") or utc_now_iso()).strip()

        history_entry = {
            "reflection_id": reflection_id,
            "created_at": created_at,
            "summary": str(snapshot_payload.get("summary") or "").strip(),
            "metrics": snapshot_payload.get("metrics", {}),
            "capability_gaps": snapshot_payload.get("capability_gaps", []),
            "suggestions": snapshot_payload.get("suggestions", []),
            "auto_evolution": snapshot_payload.get("auto_evolution", {}),
        }

        bounded_history = [history_entry]
        for item in history:
            if not isinstance(item, dict):
                continue
            if item.get("reflection_id") == reflection_id and reflection_id:
                continue
            bounded_history.append(item)
            if len(bounded_history) >= max(1, int(max_history)):
                break

        reflection = _deep_merge_dicts(reflection, snapshot_payload)
        reflection["latest_reflection_id"] = reflection_id
        reflection["latest_created_at"] = created_at
        reflection["history"] = bounded_history
        reflection["updated_at"] = utc_now_iso()

        existing["self_reflection"] = reflection
        return self._write_state(holon_id, existing)

    def is_pending(self, holon_id: str) -> bool:
        """Check if a Holon is pending."""
        return self.get_holon_status(holon_id) == _STATUS_PENDING

    def is_active(self, holon_id: str) -> bool:
        """Check if a Holon is active/runnable."""
        return self.get_holon_status(holon_id) == _STATUS_ACTIVE

    def assert_runnable(self, holon_id: str, *, action: str = "run") -> None:
        """Raise if the Holon is not active and therefore not runnable."""
        status = self.get_holon_status(holon_id)
        if status != _STATUS_ACTIVE:
            raise HolonUnavailableError(holon_id=holon_id, status=status, action=action)

    async def create_holon(
        self,
        blueprint: Blueprint,
        initial_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Create a new Holon from a blueprint.

        Args:
            blueprint: The blueprint defining the Holon
            initial_memories: Optional initial memories to seed

        Returns:
            holon_id: The ID of the created Holon
        """
        holon_id = blueprint.holon_id
        holon_path = self._get_holon_path(holon_id)

        # Create directory structure
        (holon_path / "workspace").mkdir(parents=True, exist_ok=True)
        (holon_path / "skills").mkdir(parents=True, exist_ok=True)
        (holon_path / "skills_local").mkdir(parents=True, exist_ok=True)
        (holon_path / "memory" / "lancedb").mkdir(parents=True, exist_ok=True)

        # Write blueprint
        blueprint_path = holon_path / "blueprint.json"
        blueprint_path.write_text(
            json.dumps(blueprint.to_dict(), indent=2),
            encoding="utf-8"
        )
        state_path = holon_path / "state.json"
        if not state_path.exists():
            self.mark_active(holon_id, reason="created")

        # Initialize LanceDB for this Holon
        self.factory.init_holon_tables(holon_id)

        # Seed initial memories if provided
        if initial_memories:
            from .memory_service import MemoryService
            memory_svc = MemoryService(holon_id)

            for mem in initial_memories:
                await memory_svc.remember(
                    content=mem["content"],
                    kind=mem.get("kind", "fact"),
                    tags=mem.get("tags", []),
                    importance=mem.get("importance", 1.0),
                )

        logger.info(
            "holon_created",
            holon_id=holon_id,
            name=blueprint.name,
            species=blueprint.species_id,
        )

        return holon_id

    def get_blueprint(self, holon_id: str) -> Blueprint:
        """Load a Holon's blueprint.

        Args:
            holon_id: The Holon ID

        Returns:
            Blueprint object

        Raises:
            HolonNotFoundError: If Holon doesn't exist
        """
        blueprint_path = self._get_blueprint_path(holon_id)

        if not blueprint_path.exists():
            raise HolonNotFoundError(f"Holon not found: {holon_id}")

        data = json.loads(blueprint_path.read_text(encoding="utf-8"))
        return Blueprint.from_dict(data)

    def holon_exists(self, holon_id: str) -> bool:
        """Check if a Holon exists."""
        return self._get_blueprint_path(holon_id).exists()

    def list_holons(self) -> List[Dict[str, Any]]:
        """List all Holons."""
        holons = []

        if not self.holons_path.exists():
            return holons

        for holon_dir in self.holons_path.iterdir():
            if holon_dir.is_dir():
                blueprint_path = holon_dir / "blueprint.json"
                if blueprint_path.exists():
                    try:
                        data = json.loads(blueprint_path.read_text(encoding="utf-8"))
                        holons.append({
                            "holon_id": data["holon_id"],
                            "name": data["name"],
                            "species_id": data["species_id"],
                            "purpose": data["purpose"],
                            "created_at": data.get("created_at"),
                            "status": self.get_holon_status(data["holon_id"]),
                        })
                    except Exception as e:
                        logger.warning("failed_to_load_blueprint", path=str(blueprint_path), error=str(e))

        return holons

    def freeze_holon(self, holon_id: str) -> None:
        """Freeze a Holon (mark as inactive but keep data).

        This prevents new episodes but preserves memory for potential resume.
        """
        self.set_holon_status(holon_id, _STATUS_FROZEN, reason="manual_freeze")
        logger.info("holon_frozen", holon_id=holon_id)

    def resume_holon(self, holon_id: str) -> None:
        """Resume a frozen Holon."""
        self.mark_active(holon_id, reason="manual_resume")
        logger.info("holon_resumed", holon_id=holon_id)

    def is_frozen(self, holon_id: str) -> bool:
        """Check if a Holon is frozen."""
        return self.get_holon_status(holon_id) == _STATUS_FROZEN

    def delete_holon(self, holon_id: str) -> None:
        """Permanently delete a Holon and all its data.

        WARNING: This is irreversible. All memories will be lost.
        """
        holon_path = self._get_holon_path(holon_id)

        if holon_path.exists():
            # Delete the database first (proper cleanup)
            self.factory.delete_holon_db(holon_id)

            # Delete the entire directory
            shutil.rmtree(holon_path)

            logger.warning("holon_deleted", holon_id=holon_id)
