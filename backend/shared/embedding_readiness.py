"""Shared preflight for RAG embedding availability."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.shared.config import rag_config, system_config
from backend.shared.lm_studio_client import lm_studio_client

logger = logging.getLogger(__name__)

EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE = (
    "RAG embeddings are unavailable. Configure an OpenRouter API key or run LM Studio "
    "with the nomic-ai/nomic-embed-text-v1.5 embedding model loaded. OAuth providers "
    "are supplementary chat/model providers and do not supply MOTO's RAG embeddings."
)


async def check_lm_studio_embedding_ready(timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Return whether LM Studio can serve the configured embedding model."""
    try:
        embeddings = await asyncio.wait_for(
            lm_studio_client.get_embeddings(
                ["MOTO embedding readiness check"],
                rag_config.embedding_model,
            ),
            timeout=timeout_seconds,
        )
        if embeddings and embeddings[0]:
            return {
                "ready": True,
                "provider": "lm_studio",
                "message": "LM Studio embeddings are available.",
            }
    except Exception as exc:
        logger.info("LM Studio embedding readiness check failed: %s", exc)

    return {
        "ready": False,
        "provider": "lm_studio",
        "message": EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE,
    }


async def check_embedding_provider_ready(timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Return whether at least one embedding provider is available for RAG."""
    if system_config.generic_mode:
        return {
            "ready": True,
            "provider": "fastembed",
            "message": "Generic mode uses in-process FastEmbed for RAG embeddings.",
        }

    if rag_config.openrouter_enabled and rag_config.openrouter_api_key:
        return {
            "ready": True,
            "provider": "openrouter",
            "message": "OpenRouter API key is configured for embedding fallback.",
        }

    lm_status = await check_lm_studio_embedding_ready(timeout_seconds=timeout_seconds)
    if lm_status.get("ready"):
        return lm_status

    return {
        "ready": False,
        "provider": None,
        "message": EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE,
    }


async def require_embedding_provider_ready() -> None:
    """Raise ValueError with user-facing guidance if RAG embeddings are unavailable."""
    status = await check_embedding_provider_ready()
    if not status.get("ready"):
        raise ValueError(str(status.get("message") or EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE))
