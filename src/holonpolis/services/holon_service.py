"""Holon Service - lifecycle management for Holons.

Creates and manages Holon directories, blueprints, and databases.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from holonpolis.config import settings
from holonpolis.domain import Blueprint
from holonpolis.domain.errors import HolonNotFoundError
from holonpolis.infrastructure.storage.path_guard import safe_join, validate_holon_id
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory

logger = structlog.get_logger()


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
        (holon_path / "skills_local").mkdir(parents=True, exist_ok=True)
        (holon_path / "memory" / "lancedb").mkdir(parents=True, exist_ok=True)

        # Write blueprint
        blueprint_path = holon_path / "blueprint.json"
        blueprint_path.write_text(
            json.dumps(blueprint.to_dict(), indent=2),
            encoding="utf-8"
        )

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
                        })
                    except Exception as e:
                        logger.warning("failed_to_load_blueprint", path=str(blueprint_path), error=str(e))

        return holons

    def freeze_holon(self, holon_id: str) -> None:
        """Freeze a Holon (mark as inactive but keep data).

        This prevents new episodes but preserves memory for potential resume.
        """
        holon_path = self._get_holon_path(holon_id)
        state_path = holon_path / "state.json"

        state = {
            "status": "frozen",
            "frozen_at": datetime.utcnow().isoformat(),
        }

        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        logger.info("holon_frozen", holon_id=holon_id)

    def resume_holon(self, holon_id: str) -> None:
        """Resume a frozen Holon."""
        holon_path = self._get_holon_path(holon_id)
        state_path = holon_path / "state.json"

        if state_path.exists():
            state = {
                "status": "active",
                "resumed_at": datetime.utcnow().isoformat(),
            }
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        logger.info("holon_resumed", holon_id=holon_id)

    def is_frozen(self, holon_id: str) -> bool:
        """Check if a Holon is frozen."""
        holon_path = self._get_holon_path(holon_id)
        state_path = holon_path / "state.json"

        if not state_path.exists():
            return False

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            return state.get("status") == "frozen"
        except Exception:
            return False

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
