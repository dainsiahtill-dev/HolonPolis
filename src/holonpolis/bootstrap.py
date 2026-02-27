"""Bootstrap - initialize the HolonPolis environment.

Creates:
- .holonpolis/ directory structure
- Genesis LanceDB with tables
- Initial species definitions
"""

import json

import structlog

from holonpolis.config import settings
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory

logger = structlog.get_logger()


def bootstrap():
    """Initialize the HolonPolis environment.

    This should be called once at application startup.
    It's safe to call multiple times (idempotent).
    """
    settings.setup_logging()
    logger.info("bootstrapping_holonpolis", root=str(settings.holonpolis_root))

    # 1. Ensure directory structure
    settings.ensure_directories()

    # 2. Initialize Genesis LanceDB
    factory = get_lancedb_factory()
    factory.init_genesis_tables()

    # 3. Initialize default species if not exists
    _init_default_species()

    logger.info("bootstrap_complete")


def _init_default_species():
    """Initialize default species definitions."""
    species_path = settings.species_path
    species_path.mkdir(exist_ok=True)

    # Generalist species
    generalist_path = species_path / "generalist.json"
    if not generalist_path.exists():
        generalist = {
            "species_id": "generalist",
            "name": "General Assistant",
            "description": "A balanced general-purpose assistant",
            "system_prompt_template": "You are a helpful AI assistant.",
            "default_boundary": {
                "allowed_tools": ["search", "calculate", "remember", "recall"],
                "denied_tools": ["execute_code", "file_delete"],
                "max_episodes_per_hour": 100,
                "max_tokens_per_episode": 10000,
                "allow_file_write": False,
                "allow_network": False,
                "allow_subprocess": False,
            },
            "default_evolution_policy": {
                "strategy": "balanced",
                "auto_promote_to_global": False,
                "require_tests": True,
                "max_evolution_attempts": 3,
            },
            "initial_tools": ["search", "calculate"],
        }
        generalist_path.write_text(json.dumps(generalist, indent=2))
        logger.debug("created_species", species="generalist")

    # Specialist species
    specialist_path = species_path / "specialist.json"
    if not specialist_path.exists():
        specialist = {
            "species_id": "specialist",
            "name": "Domain Specialist",
            "description": "Deep expertise in a specific domain",
            "system_prompt_template": "You are a specialized expert.",
            "default_boundary": {
                "allowed_tools": ["search", "calculate", "remember", "recall", "analyze"],
                "denied_tools": [],
                "max_episodes_per_hour": 50,
                "max_tokens_per_episode": 20000,
                "allow_file_write": True,
                "allow_network": True,
                "allow_subprocess": False,
            },
            "default_evolution_policy": {
                "strategy": "conservative",
                "auto_promote_to_global": False,
                "require_tests": True,
                "max_evolution_attempts": 5,
            },
            "initial_tools": ["search", "calculate", "analyze"],
        }
        specialist_path.write_text(json.dumps(specialist, indent=2))
        logger.debug("created_species", species="specialist")

    # Worker species
    worker_path = species_path / "worker.json"
    if not worker_path.exists():
        worker = {
            "species_id": "worker",
            "name": "Task Worker",
            "description": "High-throughput task executor",
            "system_prompt_template": "You are an efficient task executor.",
            "default_boundary": {
                "allowed_tools": ["execute"],
                "denied_tools": ["network", "file_write"],
                "max_episodes_per_hour": 500,
                "max_tokens_per_episode": 5000,
                "allow_file_write": False,
                "allow_network": False,
                "allow_subprocess": False,
            },
            "default_evolution_policy": {
                "strategy": "aggressive",
                "auto_promote_to_global": False,
                "require_tests": True,
                "max_evolution_attempts": 2,
            },
            "initial_tools": ["execute"],
        }
        worker_path.write_text(json.dumps(worker, indent=2))
        logger.debug("created_species", species="worker")


if __name__ == "__main__":
    bootstrap()
