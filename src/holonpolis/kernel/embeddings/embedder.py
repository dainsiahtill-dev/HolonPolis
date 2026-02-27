"""Embedding provider interface."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts into vectors."""
        pass

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        results = await self.embed([text])
        return results[0]
