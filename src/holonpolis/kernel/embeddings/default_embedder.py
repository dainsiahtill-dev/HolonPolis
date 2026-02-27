"""Default embedding provider using OpenAI."""

import os
from typing import List, Optional

import structlog

from .embedder import EmbeddingProvider

logger = structlog.get_logger()


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

        # Dimensions for known models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions.get(self.model, 1536)

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise RuntimeError("openai package not installed")

            if not self.api_key:
                raise ValueError("OpenAI API key not provided")

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI API."""
        if not texts:
            return []

        client = self._get_client()

        try:
            response = await client.embeddings.create(
                model=self.model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            logger.debug("embeddings_created", count=len(embeddings), model=self.model)
            return embeddings

        except Exception as e:
            logger.error("embedding_failed", error=str(e), count=len(texts))
            raise


class SimpleEmbedder(EmbeddingProvider):
    """Simple deterministic embedder for testing (not for production).

    Generates pseudo-random embeddings based on text hash.
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic pseudo-embeddings."""
        import hashlib
        import random

        results = []
        for text in texts:
            # Seed random with text hash for determinism
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
            rng = random.Random(seed)

            # Generate normalized random vector
            vec = [rng.gauss(0, 1) for _ in range(self._dimension)]
            magnitude = sum(x ** 2 for x in vec) ** 0.5
            vec = [x / magnitude for x in vec]

            results.append(vec)

        return results


# Global embedder instance
_embedder: Optional[EmbeddingProvider] = None


def get_embedder() -> EmbeddingProvider:
    """Get the global embedder."""
    global _embedder
    if _embedder is None:
        from holonpolis.config import settings

        if settings.embedding_provider.lower() == "openai":
            api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                _embedder = OpenAIEmbedder(
                    model=settings.embedding_model,
                    api_key=api_key,
                    base_url=settings.openai_base_url,
                )
            else:
                logger.warning("openai_not_configured_using_simple_embedder")
                _embedder = SimpleEmbedder(dimension=settings.embedding_dimension)
        else:
            _embedder = SimpleEmbedder(dimension=settings.embedding_dimension)
    return _embedder


def set_embedder(embedder: EmbeddingProvider) -> None:
    """Set the global embedder."""
    global _embedder
    _embedder = embedder
