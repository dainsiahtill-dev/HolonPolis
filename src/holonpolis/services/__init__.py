"""Services layer - domain orchestration.

This module intentionally uses lazy exports to avoid circular imports between
`runtime` and `services` modules during package initialization.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MemoryService": ("holonpolis.services.memory_service", "MemoryService"),
    "HybridSearchResult": ("holonpolis.services.memory_service", "HybridSearchResult"),
    "HolonService": ("holonpolis.services.holon_service", "HolonService"),
    "GenesisService": ("holonpolis.services.genesis_service", "GenesisService"),
    "EvolutionService": ("holonpolis.services.evolution_service", "EvolutionService"),
    "SecurityScanner": ("holonpolis.services.evolution_service", "SecurityScanner"),
    "Attestation": ("holonpolis.services.evolution_service", "Attestation"),
    "EvolutionResult": ("holonpolis.services.evolution_service", "EvolutionResult"),
    "RepositoryLearner": ("holonpolis.services.repository_learner", "RepositoryLearner"),
    "RepositoryLearningService": (
        "holonpolis.services.repository_learner",
        "RepositoryLearningService",
    ),
    "LearningResult": ("holonpolis.services.repository_learner", "LearningResult"),
    "RepositoryAnalysis": ("holonpolis.services.repository_learner", "RepositoryAnalysis"),
    "ProjectIncubationService": (
        "holonpolis.services.project_incubation_service",
        "ProjectIncubationService",
    ),
    "ProjectIncubationSpec": (
        "holonpolis.services.project_incubation_service",
        "ProjectIncubationSpec",
    ),
    "ProjectIncubationResult": (
        "holonpolis.services.project_incubation_service",
        "ProjectIncubationResult",
    ),
    "ProjectDeliveryService": (
        "holonpolis.services.project_delivery_service",
        "ProjectDeliveryService",
    ),
    "ReusableProjectScaffoldService": (
        "holonpolis.services.reusable_project_scaffold_service",
        "ReusableProjectScaffoldService",
    ),
    "ReusableProjectScaffold": (
        "holonpolis.services.reusable_project_scaffold_service",
        "ReusableProjectScaffold",
    ),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'holonpolis.services' has no attribute {name!r}")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
