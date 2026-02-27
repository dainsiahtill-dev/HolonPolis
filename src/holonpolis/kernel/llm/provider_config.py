"""LLM provider configuration loader inspired by Harborpilot config patterns."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from holonpolis.config import settings


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def build_default_provider_bundle() -> Dict[str, Any]:
    """Build default provider bundle from settings + environment."""
    providers: Dict[str, Dict[str, Any]] = {
        "openai": {
            "type": "openai_compat",
            "name": "OpenAI",
            "base_url": settings.openai_base_url or _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": settings.openai_api_key or _env("OPENAI_API_KEY"),
            "model": settings.openai_model,
            "api_path": "/v1/chat/completions",
            "timeout": 120,
        },
        "ollama": {
            "type": "ollama",
            "name": "Ollama",
            "base_url": settings.ollama_base_url,
            "model": settings.ollama_model,
            "api_path": "/api/chat",
            "timeout": 120,
        },
        "anthropic": {
            "type": "anthropic_compat",
            "name": "Anthropic",
            "base_url": settings.anthropic_base_url,
            "api_key": settings.anthropic_api_key or _env("ANTHROPIC_API_KEY"),
            "model": settings.anthropic_model,
            "api_path": "/v1/messages",
            "anthropic_version": _env("ANTHROPIC_VERSION", "2023-06-01"),
            "timeout": 120,
        },
        "kimi": {
            "type": "openai_compat",
            "name": "Kimi",
            "base_url": settings.kimi_base_url,
            "api_key": settings.kimi_api_key or _env("KIMI_API_KEY") or _env("MOONSHOT_API_KEY"),
            "model": settings.kimi_model,
            "api_path": "/v1/chat/completions",
            "timeout": 120,
        },
        "gemini": {
            "type": "gemini_api",
            "name": "Gemini API",
            "base_url": settings.gemini_base_url,
            "api_key": settings.gemini_api_key or _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            "model": settings.gemini_model,
            "api_path": "/v1beta/models/{model}:generateContent",
            "timeout": 120,
        },
    }

    default_provider_id = _env("HOLONPOLIS_LLM_DEFAULT_PROVIDER", settings.llm_provider or "openai")
    default_provider_id = {
        "local": "ollama",
        "openai": "openai",
        "ollama": "ollama",
        "anthropic": "anthropic",
        "kimi": "kimi",
        "gemini": "gemini",
        "gemini_api": "gemini",
    }.get(default_provider_id, default_provider_id)
    if default_provider_id not in providers:
        default_provider_id = "openai"

    return {
        "default_provider_id": default_provider_id,
        "providers": providers,
    }


def _config_file_path() -> Path:
    return settings.holonpolis_root / "config" / "llm" / "providers.json"


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_env_json(var_name: str) -> Dict[str, Any]:
    raw = _env(var_name)
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    providers = bundle.get("providers")
    if not isinstance(providers, dict):
        providers = {}

    normalized_providers: Dict[str, Dict[str, Any]] = {}
    for provider_id, provider_cfg in providers.items():
        if not isinstance(provider_cfg, dict):
            continue
        cfg = dict(provider_cfg)
        provider_type = str(cfg.get("type") or "").strip()
        if not provider_type:
            continue
        cfg["type"] = provider_type
        normalized_providers[str(provider_id)] = cfg

    default_provider_id = str(bundle.get("default_provider_id") or "").strip()
    if default_provider_id not in normalized_providers and normalized_providers:
        default_provider_id = next(iter(normalized_providers.keys()))
    if not default_provider_id:
        default_provider_id = "openai"

    return {
        "default_provider_id": default_provider_id,
        "providers": normalized_providers,
    }


def load_provider_bundle() -> Dict[str, Any]:
    """Load provider bundle from defaults + file + env JSON overrides."""
    bundle = build_default_provider_bundle()

    file_payload = _load_json_dict(_config_file_path())
    if file_payload:
        bundle = _deep_merge(bundle, file_payload)

    env_payload = _parse_env_json("HOLONPOLIS_LLM_PROVIDERS_JSON")
    if env_payload:
        bundle = _deep_merge(bundle, env_payload)

    return _normalize_bundle(bundle)
