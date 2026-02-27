"""Tests for LLM provider bundle loading and overrides."""

from holonpolis.kernel.llm.provider_config import build_default_provider_bundle, load_provider_bundle


def test_default_provider_bundle_contains_common_providers():
    bundle = build_default_provider_bundle()
    providers = bundle.get("providers", {})

    assert "openai" in providers
    assert "ollama" in providers
    assert "anthropic" in providers
    assert "kimi" in providers
    assert "gemini" in providers


def test_provider_bundle_accepts_env_json_override(monkeypatch):
    monkeypatch.setenv(
        "HOLONPOLIS_LLM_PROVIDERS_JSON",
        '{"default_provider_id":"custom","providers":{"custom":{"type":"openai_compat","model":"custom-model","base_url":"https://example.com/v1","api_key":"k"}}}',
    )
    bundle = load_provider_bundle()

    assert bundle["default_provider_id"] == "custom"
    assert "custom" in bundle["providers"]
    assert bundle["providers"]["custom"]["model"] == "custom-model"

