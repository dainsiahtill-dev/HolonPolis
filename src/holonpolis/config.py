"""HolonPolis configuration settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from holonpolis.infrastructure.config.settings_utils import (
    env_bool,
    env_int,
    env_list,
    env_str,
)
from holonpolis.infrastructure.logging_setup import configure_logging
from holonpolis.infrastructure.storage.path_guard import (
    ensure_within_root,
    normalize_path,
    safe_join,
)


class Settings(BaseSettings):
    """Application settings with env var support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Rooted runtime paths (all must remain inside holonpolis_root)
    holonpolis_root: Path = Field(default=Path(".holonpolis"))
    genesis_memory_path: Path = Field(default=Path("genesis/memory/lancedb"))
    holons_path: Path = Field(default=Path("holons"))
    global_skills_path: Path = Field(default=Path("skills_global"))
    attestations_path: Path = Field(default=Path("attestations"))
    runs_path: Path = Field(default=Path("runs"))
    blueprint_cache_path: Path = Field(default=Path("genesis/blueprint_cache"))
    index_path: Path = Field(default=Path("index"))
    species_path: Path = Field(default=Path("species"))

    # Server/observability
    api_host: str = Field(default_factory=lambda: env_str("HOLONPOLIS_HOST", "0.0.0.0"))
    api_port: int = Field(
        default_factory=lambda: env_int("HOLONPOLIS_PORT", 8000, minimum=1, maximum=65535)
    )
    api_reload: bool = Field(default_factory=lambda: env_bool("HOLONPOLIS_RELOAD", True))
    log_level: str = Field(default_factory=lambda: env_str("HOLONPOLIS_LOG_LEVEL", "INFO"))
    log_json: bool = Field(default_factory=lambda: env_bool("HOLONPOLIS_LOG_JSON", False))
    cors_origins: list[str] = Field(
        default_factory=lambda: env_list("HOLONPOLIS_CORS_ORIGINS", default=["*"])
    )

    # LLM configuration
    llm_provider: str = Field(
        default_factory=lambda: env_str("HOLONPOLIS_LLM_DEFAULT_PROVIDER", "openai")
    )
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = Field(
        default_factory=lambda: env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    ollama_model: str = Field(default_factory=lambda: env_str("OLLAMA_MODEL", "qwen2.5-coder:14b"))
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: str = Field(
        default_factory=lambda: env_str("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
    )
    anthropic_model: str = Field(
        default_factory=lambda: env_str("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    )
    kimi_api_key: Optional[str] = None
    kimi_base_url: str = Field(
        default_factory=lambda: env_str("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    )
    kimi_model: str = Field(default_factory=lambda: env_str("KIMI_MODEL", "moonshot-v1-8k"))
    gemini_api_key: Optional[str] = None
    gemini_base_url: str = Field(
        default_factory=lambda: env_str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    )
    gemini_model: str = Field(default_factory=lambda: env_str("GEMINI_MODEL", "gemini-1.5-pro"))
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Embedding configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Sandbox configuration
    sandbox_timeout_seconds: int = 60
    sandbox_max_memory_mb: int = 512
    sandbox_enable_network: bool = False

    # Memory configuration
    memory_default_top_k: int = 5
    memory_decay_enabled: bool = True
    memory_importance_threshold: float = 0.5

    # Evolution configuration
    evolution_max_attempts: int = 3
    evolution_pytest_timeout: int = 30

    def _resolve_under_root(self, value: Path) -> Path:
        root = normalize_path(self.holonpolis_root)
        raw = Path(value)
        if raw.is_absolute():
            try:
                return ensure_within_root(root, raw)
            except ValueError:
                parts = list(raw.parts)
                normalized_parts = [part.lower() for part in parts]
                if ".holonpolis" in normalized_parts:
                    idx = normalized_parts.index(".holonpolis")
                    suffix = parts[idx + 1 :]
                    if not suffix:
                        return root
                    return safe_join(root, *suffix)
                raise

        normalized = str(raw).replace("\\", "/")
        if normalized == ".holonpolis":
            return root
        if normalized.startswith(".holonpolis/"):
            raw = Path(normalized[len(".holonpolis/") :])

        if not raw.parts:
            return root
        return safe_join(root, *raw.parts)

    def _normalize_runtime_paths(self) -> None:
        self.holonpolis_root = normalize_path(self.holonpolis_root)

        self.genesis_memory_path = self._resolve_under_root(self.genesis_memory_path)
        self.holons_path = self._resolve_under_root(self.holons_path)
        self.global_skills_path = self._resolve_under_root(self.global_skills_path)
        self.attestations_path = self._resolve_under_root(self.attestations_path)
        self.runs_path = self._resolve_under_root(self.runs_path)
        self.blueprint_cache_path = self._resolve_under_root(self.blueprint_cache_path)
        self.index_path = self._resolve_under_root(self.index_path)
        self.species_path = self._resolve_under_root(self.species_path)

    @model_validator(mode="after")
    def _normalize_paths_validator(self) -> "Settings":
        self._normalize_runtime_paths()
        return self

    def setup_logging(self) -> None:
        configure_logging(level=self.log_level, json_logs=self.log_json)

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self._normalize_runtime_paths()

        required = [
            self.holonpolis_root,
            self.genesis_memory_path,
            self.holons_path,
            self.global_skills_path,
            self.attestations_path,
            self.runs_path,
            self.blueprint_cache_path,
            self.index_path,
            self.species_path,
        ]
        for path in required:
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
