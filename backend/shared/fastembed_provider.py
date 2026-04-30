"""
FastEmbed embedding provider for generic mode.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)

FASTEMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


class FastEmbedProvider:
    """Lazy wrapper around the optional FastEmbed dependency."""

    def __init__(self, model_name: str = FASTEMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._init_lock = asyncio.Lock()

    def _create_model(self):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Generic mode requires the optional 'fastembed' dependency. "
                "Install requirements-generic.txt for hosted deployments."
            ) from exc

        logger.info("Initializing FastEmbed model '%s'", self.model_name)
        return TextEmbedding(model_name=self.model_name)

    async def _get_model(self):
        if self._model is None:
            async with self._init_lock:
                if self._model is None:
                    self._model = await asyncio.to_thread(self._create_model)
        return self._model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the hosted in-process model."""
        if not texts:
            return []

        model = await self._get_model()

        def _embed_sync() -> List[List[float]]:
            normalized = []
            for embedding in model.embed(texts):
                if hasattr(embedding, "tolist"):
                    normalized.append(embedding.tolist())
                else:
                    normalized.append(list(embedding))
            return normalized

        try:
            return await asyncio.to_thread(_embed_sync)
        except Exception as exc:
            raise RuntimeError(
                f"FastEmbed failed to generate embeddings with '{self.model_name}': {exc}"
            ) from exc
