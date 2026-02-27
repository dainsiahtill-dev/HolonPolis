"""LanceDB table schemas using PyArrow."""

import pyarrow as pa


# Schema for memories table (retrievable condensed memories)
MEMORIES_SCHEMA = pa.schema([
    pa.field("memory_id", pa.string(), nullable=False),
    pa.field("holon_id", pa.string(), nullable=False),
    pa.field("kind", pa.string(), nullable=False),  # MemoryKind value
    pa.field("content", pa.string(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    pa.field("tags", pa.list_(pa.string()), nullable=False),
    pa.field("importance", pa.float32(), nullable=False),
    pa.field("success_score", pa.float32(), nullable=True),
    pa.field("source_episode_id", pa.string(), nullable=True),
    pa.field("source_skill", pa.string(), nullable=True),
    pa.field("created_at", pa.string(), nullable=False),
    pa.field("last_accessed_at", pa.string(), nullable=True),
    pa.field("access_count", pa.int32(), nullable=False),
    pa.field("decay_factor", pa.float32(), nullable=False),
])


# Schema for episodes table (full interaction records)
EPISODES_SCHEMA = pa.schema([
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("holon_id", pa.string(), nullable=False),
    pa.field("transcript", pa.string(), nullable=False),  # JSON string
    pa.field("tool_chain", pa.string(), nullable=False),  # JSON string
    pa.field("outcome", pa.string(), nullable=False),
    pa.field("outcome_details", pa.string(), nullable=False),  # JSON string
    pa.field("cost", pa.float32(), nullable=False),
    pa.field("latency_ms", pa.int32(), nullable=False),
    pa.field("evolution_id", pa.string(), nullable=True),
    pa.field("started_at", pa.string(), nullable=False),
    pa.field("ended_at", pa.string(), nullable=True),
])


# Schema for Genesis holons table (index of all holons)
GENESIS_HOLONS_SCHEMA = pa.schema([
    pa.field("holon_id", pa.string(), nullable=False),
    pa.field("blueprint_id", pa.string(), nullable=False),
    pa.field("species_id", pa.string(), nullable=False),
    pa.field("name", pa.string(), nullable=False),
    pa.field("purpose", pa.string(), nullable=False),
    pa.field("status", pa.string(), nullable=False),  # active, frozen, archived
    pa.field("capabilities", pa.list_(pa.string()), nullable=False),
    pa.field("skills", pa.list_(pa.string()), nullable=False),
    pa.field("total_episodes", pa.int32(), nullable=False),
    pa.field("success_rate", pa.float32(), nullable=False),
    pa.field("last_active_at", pa.string(), nullable=True),
    pa.field("created_at", pa.string(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32()), nullable=False),  # Purpose embedding
])


# Schema for Genesis routes table (routing history)
GENESIS_ROUTES_SCHEMA = pa.schema([
    pa.field("route_id", pa.string(), nullable=False),
    pa.field("query", pa.string(), nullable=False),
    pa.field("query_embedding", pa.list_(pa.float32()), nullable=False),
    pa.field("decision", pa.string(), nullable=False),  # route_to, spawn, deny, clarify
    pa.field("target_holon_id", pa.string(), nullable=True),
    pa.field("spawned_blueprint_id", pa.string(), nullable=True),
    pa.field("reasoning", pa.string(), nullable=False),
    pa.field("outcome", pa.string(), nullable=True),  # success, failure, pending
    pa.field("created_at", pa.string(), nullable=False),
])


# Schema for Genesis evolutions table (evolution tracking)
GENESIS_EVOLUTIONS_SCHEMA = pa.schema([
    pa.field("evolution_id", pa.string(), nullable=False),
    pa.field("holon_id", pa.string(), nullable=False),
    pa.field("skill_name", pa.string(), nullable=False),
    pa.field("status", pa.string(), nullable=False),  # pending, success, failed
    pa.field("attempt_count", pa.int32(), nullable=False),
    pa.field("test_passed", pa.bool_(), nullable=False),
    pa.field("static_scan_passed", pa.bool_(), nullable=False),
    pa.field("attestation_id", pa.string(), nullable=True),
    pa.field("error_message", pa.string(), nullable=True),
    pa.field("created_at", pa.string(), nullable=False),
    pa.field("completed_at", pa.string(), nullable=True),
])
