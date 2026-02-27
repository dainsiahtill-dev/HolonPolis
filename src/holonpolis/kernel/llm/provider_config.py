"""LLM provider configuration loader inspired by Harborpilot config patterns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from holonpolis.config import settings
from holonpolis.infrastructure.storage.io_text import write_json_atomic, read_json_safe


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
        "kimi-coding": {
            "type": "anthropic_compat",
            "name": "Kimi Coding",
            "base_url": "https://api.kimi.com/coding",
            "api_key": settings.kimi_api_key or _env("KIMI_API_KEY") or _env("MOONSHOT_API_KEY"),
            "model": "kimi-for-coding",
            "api_path": "/v1/chat/completions",
            "timeout": 360,
        },
        "minimax": {
            "type": "openai_compat",
            "name": "MiniMax",
            "base_url": settings.minimax_base_url or _env("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
            "api_key": settings.minimax_api_key or _env("MINIMAX_API_KEY"),
            "model": settings.minimax_model or _env("MINIMAX_MODEL", "MiniMax-M2.5"),
            "api_path": "/v1/chat/completions",
            "timeout": 360,
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
        provider_type = str(cfg.get("type") or cfg.get("provider_type") or "").strip()
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


MASKED_SECRET = "********"


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    provider_id: str
    provider_type: str
    name: str
    base_url: str = ""
    api_key: str = ""
    api_path: str = "/v1/chat/completions"
    models_path: str = "/v1/models"
    timeout: int = 60
    retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 8192
    model: str = ""  # Default model for this provider
    extra_headers: Dict[str, str] = field(default_factory=dict)
    model_specific: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, mask_secrets: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally masking secrets."""
        result = {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
            "name": self.name,
            "base_url": self.base_url,
            "api_path": self.api_path,
            "models_path": self.models_path,
            "timeout": self.timeout,
            "retries": self.retries,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "extra_headers": self.extra_headers,
            "model_specific": self.model_specific,
        }
        if mask_secrets:
            result["api_key"] = MASKED_SECRET if self.api_key else ""
        else:
            result["api_key"] = self.api_key
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderConfig":
        """Create from dictionary."""
        return cls(
            provider_id=data.get("provider_id", ""),
            provider_type=data.get("provider_type", data.get("type", "openai_compat")),
            name=data.get("name", "Unnamed Provider"),
            base_url=data.get("base_url", ""),
            api_key=data.get("api_key", ""),
            api_path=data.get("api_path", "/v1/chat/completions"),
            models_path=data.get("models_path", "/v1/models"),
            timeout=data.get("timeout", 60),
            retries=data.get("retries", 3),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 8192),
            model=data.get("model", ""),
            extra_headers=data.get("extra_headers", {}),
            model_specific=data.get("model_specific", {}),
        )


class ProviderConfigManager:
    """Manages LLM provider configurations with persistence."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize with config directory.

        Args:
            config_dir: Directory to store configs (default: ~/.holonpolis/config)
        """
        if config_dir is None:
            config_dir = str(settings.holonpolis_root / "config")
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "llm", "providers.json")
        self._providers: Dict[str, ProviderConfig] = {}
        self._load()

    def _load(self) -> None:
        """Load providers from disk."""
        data = read_json_safe(self.config_file)
        if not data:
            # Initialize from defaults
            bundle = load_provider_bundle()
            self._providers = {
                pid: ProviderConfig.from_dict({"provider_id": pid, **pcfg})
                for pid, pcfg in bundle.get("providers", {}).items()
            }
            self._save()
            return

        providers_data = data.get("providers", {})
        self._providers = {
            pid: ProviderConfig.from_dict(pcfg)
            for pid, pcfg in providers_data.items()
        }

    def _save(self) -> None:
        """Save providers to disk."""
        data = {
            "version": 1,
            "providers": {
                pid: pcfg.to_dict(mask_secrets=False)
                for pid, pcfg in self._providers.items()
            },
        }
        write_json_atomic(self.config_file, data)

    def list_providers(self, mask_secrets: bool = True) -> List[Dict[str, Any]]:
        """List all providers.

        Args:
            mask_secrets: Whether to mask API keys

        Returns:
            List of provider configs as dicts
        """
        return [
            pcfg.to_dict(mask_secrets=mask_secrets)
            for pcfg in self._providers.values()
        ]

    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get a provider by ID."""
        return self._providers.get(provider_id)

    def add_provider(self, config: ProviderConfig) -> Tuple[bool, str]:
        """Add or update a provider.

        Args:
            config: Provider configuration

        Returns:
            Tuple of (success, message)
        """
        # Validate
        valid, errors = self._validate_config(config)
        if not valid:
            return False, f"Validation failed: {', '.join(errors)}"

        self._providers[config.provider_id] = config
        self._save()
        return True, f"Provider '{config.provider_id}' saved"

    def update_provider(
        self,
        provider_id: str,
        updates: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Update a provider configuration.

        Args:
            provider_id: Provider ID to update
            updates: Dictionary of updates

        Returns:
            Tuple of (success, message)
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return False, f"Provider '{provider_id}' not found"

        # Handle masked API key
        if "api_key" in updates and updates["api_key"] == MASKED_SECRET:
            updates.pop("api_key")  # Keep existing

        # Update fields
        for key, value in updates.items():
            if hasattr(provider, key):
                setattr(provider, key, value)

        self._save()
        return True, f"Provider '{provider_id}' updated"

    def delete_provider(self, provider_id: str) -> Tuple[bool, str]:
        """Delete a provider.

        Args:
            provider_id: Provider ID to delete

        Returns:
            Tuple of (success, message)
        """
        if provider_id not in self._providers:
            return False, f"Provider '{provider_id}' not found"

        del self._providers[provider_id]
        self._save()
        return True, f"Provider '{provider_id}' deleted"

    def _validate_config(self, config: ProviderConfig) -> Tuple[bool, List[str]]:
        """Validate provider configuration."""
        errors = []

        if not config.provider_id:
            errors.append("provider_id is required")
        if not config.provider_type:
            errors.append("provider_type is required")
        if not config.name:
            errors.append("name is required")

        # Validate URL format if provided
        if config.base_url:
            if not config.base_url.startswith(("http://", "https://")):
                errors.append("base_url must be a valid HTTP(S) URL")

        return len(errors) == 0, errors

    def health_check(self, provider_id: str) -> Dict[str, Any]:
        """Perform health check on a provider.

        Args:
            provider_id: Provider ID to check

        Returns:
            Health check result
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return {
                "healthy": False,
                "error": f"Provider '{provider_id}' not found",
            }

        # Simple connectivity check
        import urllib.request
        import urllib.error

        try:
            url = f"{provider.base_url}{provider.models_path}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            if provider.api_key:
                req.add_header("Authorization", f"Bearer {provider.api_key}")

            with urllib.request.urlopen(req, timeout=provider.timeout) as response:
                status = response.status
                if status == 200:
                    return {
                        "healthy": True,
                        "status": status,
                        "message": "Provider is responsive",
                    }
                else:
                    return {
                        "healthy": False,
                        "status": status,
                        "message": f"Unexpected status code: {status}",
                    }

        except urllib.error.HTTPError as e:
            # 401/403 is actually "healthy" - means endpoint exists
            if e.code in (401, 403):
                return {
                    "healthy": True,
                    "status": e.code,
                    "message": "Endpoint exists (auth required)",
                }
            return {
                "healthy": False,
                "status": e.code,
                "error": str(e.reason),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }

    def list_models(self, provider_id: str) -> Dict[str, Any]:
        """List available models from a provider.

        Args:
            provider_id: Provider ID

        Returns:
            Result with models list or error
        """
        provider = self._providers.get(provider_id)
        if not provider:
            return {
                "success": False,
                "error": f"Provider '{provider_id}' not found",
            }

        import urllib.request
        import urllib.error

        try:
            url = f"{provider.base_url}{provider.models_path}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Accept", "application/json")

            if provider.api_key:
                req.add_header("Authorization", f"Bearer {provider.api_key}")

            with urllib.request.urlopen(req, timeout=provider.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

                # Extract models (handle different response formats)
                models = []
                if "data" in data:
                    models = [m.get("id", m.get("name", "")) for m in data["data"]]
                elif "models" in data:
                    models = [m.get("name", m.get("id", "")) for m in data["models"]]

                return {
                    "success": True,
                    "models": models,
                    "count": len(models),
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# Global instance
_provider_manager: Optional[ProviderConfigManager] = None


def get_provider_manager() -> ProviderConfigManager:
    """Get the global provider config manager."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = ProviderConfigManager()
    return _provider_manager
