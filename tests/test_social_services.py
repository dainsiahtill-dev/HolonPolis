"""Social layer service tests."""

import sqlite3

import pytest

from holonpolis.services.collaboration_service import CollaborationService
from holonpolis.services.market_service import MarketService
from holonpolis.services.social_state_store import SocialStateStore


@pytest.fixture
def social_setup(tmp_path, monkeypatch):
    """Isolate social state storage for each test."""
    root = tmp_path / ".holonpolis"
    monkeypatch.setattr("holonpolis.config.settings.holonpolis_root", root)
    monkeypatch.setattr("holonpolis.config.settings.holons_path", root / "holons")
    MarketService.reset_in_memory_cache()
    CollaborationService.reset_in_memory_cache()
    return root


def test_market_service_state_shared_across_instances(social_setup):
    """Market offers should persist across instances and reload after cache reset."""
    service_a = MarketService()
    offer = service_a.register_offer(
        holon_id="holon_market_a",
        skill_name="Code Review",
        skill_description="Review pull requests for quality issues",
        price_per_use=100.0,
        success_rate=0.9,
    )

    service_b = MarketService()
    assert offer.offer_id in service_b.offers
    db_path = social_setup / "genesis" / "social_state" / "state.db"
    assert db_path.exists()

    # Simulate process restart by clearing in-memory cache and reloading.
    MarketService.reset_in_memory_cache()
    service_c = MarketService()
    assert offer.offer_id in service_c.offers


@pytest.mark.asyncio
async def test_collaboration_state_shared_across_instances(social_setup):
    """Collaboration tasks should persist and survive cache reset."""
    service_a = CollaborationService()
    task = await service_a.create_collaboration(
        name="Shared Collaboration",
        description="Verify collaboration state continuity",
        coordinator_id="holon_coord",
        participant_ids=["holon_coord", "holon_worker_1"],
        task_structure={
            "subtasks": [
                {"name": "Plan", "description": "Create initial plan"},
            ],
            "dependencies": {},
        },
    )

    service_b = CollaborationService()
    assert task.task_id in service_b.active_collaborations
    db_path = social_setup / "genesis" / "social_state" / "state.db"
    assert db_path.exists()

    # Simulate process restart by clearing in-memory cache and reloading.
    CollaborationService.reset_in_memory_cache()
    service_c = CollaborationService()
    assert task.task_id in service_c.active_collaborations


def test_social_state_snapshot_compaction(social_setup):
    """Only latest snapshots should be retained per scope."""
    store = SocialStateStore()
    for i in range(130):
        store.save_market_state({"i": i})

    with sqlite3.connect(store.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM social_state_snapshots WHERE scope = 'market'"
        ).fetchone()[0]
    assert count <= 100
