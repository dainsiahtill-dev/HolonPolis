"""Tests for LLM provider bundle loading and overrides."""

import json

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


def test_provider_bundle_loads_legacy_role_config(monkeypatch, tmp_path):
    runtime_root = tmp_path / ".holonpolis"
    config_dir = runtime_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = config_dir / "llm_config.json"
    legacy_file.write_text(
        json.dumps(
            {
                "providers": {
                    "chief": {
                        "type": "anthropic_compat",
                        "base_url": "https://kimi.example.test",
                        "api_key": "test-chief-key",
                        "model": "kimi-for-coding",
                        "headers": {"User-Agent": "KimiCLI/0.2.0"},
                        "max_output_tokens": 8192,
                    },
                    "architect": {
                        "type": "minimax",
                        "base_url": "https://minimax.example.test/v1",
                        "api_key": "test-architect-key",
                        "model": "MiniMax-M2.5",
                    },
                    "qa": {
                        "type": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "model": "qwen3-coder",
                    },
                },
                "roles": {
                    "chief_engineer": {"provider_id": "chief", "model": "kimi-for-coding"},
                    "architect": {"provider_id": "architect", "model": "MiniMax-M2.5"},
                    "qa": {"provider_id": "qa", "model": "qwen3-coder"},
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("holonpolis.config.settings.holonpolis_root", runtime_root)

    bundle = load_provider_bundle()

    assert bundle["default_provider_id"] == "chief"
    assert bundle["providers"]["chief"]["type"] == "anthropic_compat"
    assert bundle["providers"]["chief"]["extra_headers"]["User-Agent"] == "KimiCLI/0.2.0"
    assert bundle["providers"]["chief"]["max_tokens"] == 8192
    assert bundle["providers"]["architect"]["type"] == "openai_compat"
    assert bundle["roles"]["chief_engineer"]["provider_id"] == "chief"
    assert bundle["roles"]["qa"]["provider_id"] == "qa"
