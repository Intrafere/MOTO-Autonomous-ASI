"""Shared preflight for RAG embedding availability."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from backend.shared.config import rag_config, system_config
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.subscription_providers.registry import vllm_client

logger = logging.getLogger(__name__)

_LM_STUDIO_EMBEDDING_STATUS_CACHE_TTL_SECONDS = 60.0
_lm_studio_embedding_status_cache: dict[str, Any] | None = None
_lm_studio_embedding_status_cache_at = 0.0
_lm_studio_embedding_status_lock = asyncio.Lock()
_VLLM_EMBEDDING_STATUS_CACHE_TTL_SECONDS = 60.0
_vllm_embedding_status_cache: tuple[str, dict[str, Any]] | None = None
_vllm_embedding_status_cache_at = 0.0
_vllm_embedding_status_lock = asyncio.Lock()

EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE = (
    "RAG embeddings are unavailable. Configure an OpenRouter API key, run LM Studio "
    "with the nomic-ai/nomic-embed-text-v1.5 embedding model loaded, or add a vLLM server "
    "that serves an embedding model. OAuth providers are supplementary chat/model providers "
    "and do not supply MOTO's RAG embeddings."
)


async def check_lm_studio_embedding_ready(
    timeout_seconds: float = 10.0,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return whether LM Studio can serve the configured embedding model."""
    global _lm_studio_embedding_status_cache, _lm_studio_embedding_status_cache_at

    now = time.monotonic()
    if (
        not force_refresh
        and _lm_studio_embedding_status_cache is not None
        and now - _lm_studio_embedding_status_cache_at < _LM_STUDIO_EMBEDDING_STATUS_CACHE_TTL_SECONDS
    ):
        return dict(_lm_studio_embedding_status_cache)

    async with _lm_studio_embedding_status_lock:
        now = time.monotonic()
        if (
            not force_refresh
            and _lm_studio_embedding_status_cache is not None
            and now - _lm_studio_embedding_status_cache_at < _LM_STUDIO_EMBEDDING_STATUS_CACHE_TTL_SECONDS
        ):
            return dict(_lm_studio_embedding_status_cache)

        try:
            embeddings = await asyncio.wait_for(
                lm_studio_client.get_embeddings(
                    ["MOTO embedding readiness check"],
                    rag_config.embedding_model,
                    quiet=True,
                ),
                timeout=timeout_seconds,
            )
            if embeddings and embeddings[0]:
                status = {
                    "ready": True,
                    "provider": "lm_studio",
                    "message": "LM Studio embeddings are available.",
                }
                _lm_studio_embedding_status_cache = status
                _lm_studio_embedding_status_cache_at = time.monotonic()
                return dict(status)
        except Exception as exc:
            logger.info("LM Studio embedding readiness check failed: %s", exc)

        status = {
            "ready": False,
            "provider": "lm_studio",
            "message": EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE,
        }
        _lm_studio_embedding_status_cache = status
        _lm_studio_embedding_status_cache_at = time.monotonic()
        return dict(status)


async def check_vllm_embedding_ready(
    timeout_seconds: float = 10.0,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Return whether the configured vLLM embedding model answers an embedding request."""
    global _vllm_embedding_status_cache, _vllm_embedding_status_cache_at

    space = vllm_client.embedding_space()
    if space is None:
        return {
            "ready": False,
            "provider": "vllm",
            "message": "No vLLM embedding model is set up.",
        }

    def cached() -> dict[str, Any] | None:
        if (
            not force_refresh
            and _vllm_embedding_status_cache is not None
            and _vllm_embedding_status_cache[0] == space
            and time.monotonic() - _vllm_embedding_status_cache_at < _VLLM_EMBEDDING_STATUS_CACHE_TTL_SECONDS
        ):
            return dict(_vllm_embedding_status_cache[1])
        return None

    hit = cached()
    if hit is not None:
        return hit
    async with _vllm_embedding_status_lock:
        hit = cached()
        if hit is not None:
            return hit
        status = {
            "ready": False,
            "provider": "vllm",
            "message": "The vLLM embedding server did not answer. Check that it is running.",
        }
        try:
            if await asyncio.wait_for(vllm_client.check_embedding(), timeout=timeout_seconds):
                status = {
                    "ready": True,
                    "provider": "vllm",
                    "message": "vLLM embeddings are available.",
                }
        except Exception as exc:
            logger.info("vLLM embedding readiness check failed: %s", exc)
        _vllm_embedding_status_cache = (space, status)
        _vllm_embedding_status_cache_at = time.monotonic()
        return dict(status)


async def check_vllm_availability(
    *,
    force_refresh: bool = False,
    embedding_timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Return live vLLM reachability, served chat models, and embedding readiness."""
    status = await vllm_client.status()
    result: dict[str, Any] = {
        **status,
        "available": False,
        "has_models": False,
        "has_usable_chat_model": False,
        "model_count": 0,
        "models": [],
        "has_embedding_model": False,
        "embedding_ready": False,
        "embedding_message": "No vLLM embedding model is set up.",
        "error": None,
    }
    if not status["configured"]:
        return result
    discovery = await vllm_client.discover(force=force_refresh)
    overview = vllm_client.cached_overview()
    errors = [probe.error for probe in discovery.values() if probe.error]
    embedding_status = await check_vllm_embedding_ready(
        timeout_seconds=embedding_timeout_seconds,
        force_refresh=force_refresh,
    )
    models = overview["chat_models"]
    result.update({
        "available": overview["reachable_servers"] > 0,
        "has_models": bool(models) or bool(embedding_status.get("ready")),
        "has_usable_chat_model": bool(models),
        "model_count": len(models),
        "models": models,
        "server_errors": overview["server_errors"],
        "has_embedding_model": bool(embedding_status.get("ready")),
        "embedding_ready": bool(embedding_status.get("ready")),
        "embedding_message": embedding_status.get("message"),
        "error": errors[0] if errors else None,
    })
    return result


def reset_vllm_embedding_status_cache() -> None:
    """Forget the cached vLLM embedding check after its configuration changes."""
    global _vllm_embedding_status_cache, _vllm_embedding_status_cache_at
    _vllm_embedding_status_cache = None
    _vllm_embedding_status_cache_at = 0.0


async def check_embedding_provider_ready(
    timeout_seconds: float = 10.0,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
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

    lm_status = await check_lm_studio_embedding_ready(
        timeout_seconds=timeout_seconds,
        force_refresh=force_refresh,
    )
    if lm_status.get("ready"):
        return lm_status

    vllm_status = await check_vllm_embedding_ready(
        timeout_seconds=timeout_seconds,
        force_refresh=force_refresh,
    )
    if vllm_status.get("ready"):
        return vllm_status

    return {
        "ready": False,
        "provider": None,
        "message": EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE,
    }


async def require_embedding_provider_ready() -> None:
    """Raise ValueError with user-facing guidance if RAG embeddings are unavailable."""
    status = await check_embedding_provider_ready(force_refresh=True)
    if not status.get("ready"):
        raise ValueError(str(status.get("message") or EMBEDDING_PROVIDER_UNAVAILABLE_MESSAGE))
