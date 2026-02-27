"""HolonPolis configuration settings."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with env var support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    holonpolis_root: Path = Path(".holonpolis")
    genesis_memory_path: Path = Path(".holonpolis/genesis/memory/lancedb")
    holons_path: Path = Path(".holonpolis/holons")
    global_skills_path: Path = Path(".holonpolis/skills_global")
    attestations_path: Path = Path(".holonpolis/attestations")
    runs_path: Path = Path(".holonpolis/runs")

    # LLM Configuration
    llm_provider: str = "openai"  # openai, local, etc.
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Embedding Configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Sandbox Configuration
    sandbox_timeout_seconds: int = 60
    sandbox_max_memory_mb: int = 512
    sandbox_enable_network: bool = False

    # Memory Configuration
    memory_default_top_k: int = 5
    memory_decay_enabled: bool = True
    memory_importance_threshold: float = 0.5

    # Evolution Configuration
    evolution_max_attempts: int = 3
    evolution_pytest_timeout: int = 30

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        dirs = [
            self.holonpolis_root,
            self.genesis_memory_path,
            self.holons_path,
            self.global_skills_path,
            self.attestations_path,
            self.runs_path,
            Path(".holonpolis/genesis/blueprint_cache"),
            Path(".holonpolis/index"),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
