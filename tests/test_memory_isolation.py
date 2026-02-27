"""Test memory isolation between Holons.

Key assertion: Each Holon has its own independent LanceDB.
No Holon can access another's memories.
"""

import asyncio
import shutil
import uuid
from pathlib import Path

import pytest

from holonpolis.bootstrap import bootstrap
from holonpolis.config import settings
from holonpolis.domain import Blueprint, Boundary, EvolutionPolicy
from holonpolis.domain.blueprints import EvolutionStrategy
from holonpolis.domain.memory import MemoryKind
from holonpolis.kernel.lancedb.lancedb_factory import get_lancedb_factory, reset_factory
from holonpolis.services.holon_service import HolonService
from holonpolis.services.memory_service import MemoryService


@pytest.fixture(autouse=True)
def clean_test_env(tmp_path):
    """Clean up test environment before and after each test."""
    # Use temp directory for tests
    original_root = settings.holonpolis_root
    settings.holonpolis_root = tmp_path / ".holonpolis"
    settings.holons_path = settings.holonpolis_root / "holons"
    settings.genesis_memory_path = settings.holonpolis_root / "genesis" / "memory" / "lancedb"

    # Reset factory cache
    reset_factory()

    # Bootstrap
    bootstrap()

    yield

    # Cleanup
    if settings.holonpolis_root.exists():
        shutil.rmtree(settings.holonpolis_root)
    settings.holonpolis_root = original_root
    reset_factory()


@pytest.mark.asyncio
async def test_holon_memory_isolation():
    """Test that Holons cannot access each other's memories."""

    # Create two Holons
    holon_a_id = await _create_test_holon("Holon A", "I handle math problems")
    holon_b_id = await _create_test_holon("Holon B", "I handle writing tasks")

    # Store memory in Holon A
    memory_a = MemoryService(holon_a_id)
    await memory_a.remember(
        content="Secret math formula: E=mc^2",
        kind=MemoryKind.FACT,
        tags=["math", "secret"],
        importance=1.5,
    )

    # Store memory in Holon B
    memory_b = MemoryService(holon_b_id)
    await memory_b.remember(
        content="Writing tip: Show don't tell",
        kind=MemoryKind.PROCEDURE,
        tags=["writing"],
        importance=1.0,
    )

    # Verify Holon A can recall its own memory
    results_a = await memory_a.recall("math formula", top_k=5)
    assert len(results_a) > 0
    assert "E=mc^2" in results_a[0]["content"]

    # Verify Holon B CANNOT recall Holon A's memory
    results_b_from_a = await memory_b.recall("math formula", top_k=5)
    # Should be empty or not contain the secret
    secret_contents = [r["content"] for r in results_b_from_a if "E=mc^2" in r["content"]]
    assert len(secret_contents) == 0, "Holon B should not access Holon A's memories"

    # Verify Holon B can recall its own memory
    results_b = await memory_b.recall("writing tip", top_k=5)
    assert len(results_b) > 0
    assert "Show don't tell" in results_b[0]["content"]


@pytest.mark.asyncio
async def test_physical_db_separation():
    """Test that each Holon has its own physical database files."""

    holon_a_id = await _create_test_holon("Holon A", "Test A")
    holon_b_id = await _create_test_holon("Holon B", "Test B")

    # Get DB paths
    factory = get_lancedb_factory()
    db_path_a = factory._get_db_path(holon_a_id)
    db_path_b = factory._get_db_path(holon_b_id)

    # Verify they are different paths
    assert db_path_a != db_path_b
    assert db_path_a.exists()
    assert db_path_b.exists()

    # Verify they are in different parent directories
    assert db_path_a.parent.parent.name == holon_a_id
    assert db_path_b.parent.parent.name == holon_b_id


@pytest.mark.asyncio
async def test_genesis_has_separate_db():
    """Test that Genesis has its own separate database."""

    factory = get_lancedb_factory()

    # Get Genesis DB path
    genesis_path = factory._get_db_path("genesis")

    # Verify it's in the genesis directory, not mixed with Holons
    assert "genesis" in str(genesis_path)
    assert "holons" not in str(genesis_path)

    # Verify tables exist
    conn = factory.open("genesis")
    tables = conn.list_tables()

    assert "holons" in tables
    assert "routes" in tables
    assert "evolutions" in tables


@pytest.mark.asyncio
async def test_memory_persistence():
    """Test that memories persist across service instances."""

    holon_id = await _create_test_holon("Persistent Holon", "Testing persistence")

    # Store memory
    memory1 = MemoryService(holon_id)
    await memory1.remember(
        content="Important fact to remember",
        kind=MemoryKind.FACT,
        tags=["test"],
    )

    # Create new service instance (simulating restart)
    memory2 = MemoryService(holon_id)
    results = await memory2.recall("important fact", top_k=5)

    assert len(results) > 0
    assert "Important fact" in results[0]["content"]


async def _create_test_holon(name: str, purpose: str) -> str:
    """Helper to create a test Holon."""

    holon_id = f"holon_{uuid.uuid4().hex[:12]}"
    blueprint_id = f"blueprint_{uuid.uuid4().hex[:12]}"

    blueprint = Blueprint(
        blueprint_id=blueprint_id,
        holon_id=holon_id,
        species_id="generalist",
        name=name,
        purpose=purpose,
        boundary=Boundary(),
        evolution_policy=EvolutionPolicy(strategy=EvolutionStrategy.BALANCED),
    )

    service = HolonService()
    return await service.create_holon(blueprint)
