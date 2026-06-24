"""Non-secret connectivity status and feature-toggle routes."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.routes.cloud_access import get_cloud_access_status
from backend.shared.boost_manager import boost_manager
from backend.shared.config import rag_config, system_config
from backend.shared.embedding_readiness import check_lm_studio_embedding_ready
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.proof_search.search_service import proof_search_service
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.runtime_settings import (
    RuntimeSettingsError,
    save_connectivity_runtime_settings,
)
from backend.shared.syntheticlib4_client import syntheticlib4_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/connectivity", tags=["connectivity"])


class ConnectivityToggleRequest(BaseModel):
    syntheticlib4_enabled: bool | None = None
    agent_conversation_memory_enabled: bool | None = None
    wolfram_alpha_enabled: bool | None = None


async def _lm_studio_status() -> dict[str, Any]:
    if system_config.generic_mode:
        return {
            "status": "inactive",
            "active": False,
            "available": False,
            "model_count": 0,
            "models": [],
            "has_embedding_model": False,
            "message": "LM Studio is disabled in hosted/generic mode.",
        }
    try:
        availability = await lm_studio_client.check_availability()
        embedding_status = await check_lm_studio_embedding_ready(timeout_seconds=3.0)
    except Exception as exc:
        return {
            "status": "inactive",
            "active": False,
            "available": False,
            "model_count": 0,
            "models": [],
            "has_embedding_model": False,
            "message": f"LM Studio status unavailable: {exc}",
        }
    active = bool(availability.get("available") and availability.get("has_models"))
    return {
        "status": "ACTIVE" if active else "inactive",
        "active": active,
        **availability,
        "has_embedding_model": bool(embedding_status.get("ready")),
        "embedding_ready": bool(embedding_status.get("ready")),
        "embedding_message": embedding_status.get("message"),
    }


async def _openrouter_oauth_status() -> dict[str, Any]:
    try:
        cloud_status = await get_cloud_access_status()
    except Exception as exc:
        logger.warning("Connectivity cloud access status failed: %s", exc)
        cloud_status = {"providers": {}}
    providers = cloud_status.get("providers") or {}
    openrouter_configured = bool(rag_config.openrouter_api_key)
    oauth_configured = any(
        bool((providers.get(provider_id) or {}).get("configured"))
        for provider_id in ("openai_codex_oauth", "xai_grok_oauth", "sakana_fugu")
    )
    active = openrouter_configured or oauth_configured
    return {
        "status": "ACTIVE" if active else "inactive",
        "active": active,
        "openrouter_configured": openrouter_configured,
        "oauth_configured": oauth_configured,
        "providers": providers,
    }


async def _syntheticlib4_status() -> dict[str, Any]:
    enabled = bool(system_config.syntheticlib4_enabled)
    if not enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "ready": False,
            "message": "SyntheticLib4 proof-corpus retrieval is disabled for new runs.",
        }
    try:
        account_status, manifest, proof_count, validation, overview = await asyncio.gather(
            asyncio.to_thread(syntheticlib4_client.get_status),
            asyncio.to_thread(syntheticlib4_client.get_release_manifest),
            asyncio.to_thread(lambda: len(syntheticlib4_client.load_proof_metadata())),
            asyncio.to_thread(syntheticlib4_client.validate_local_snapshot),
            proof_search_service.overview(),
        )
    except Exception as exc:
        return {
            "status": "error",
            "enabled": True,
            "ready": False,
            "message": f"SyntheticLib4 status failed: {exc}",
        }
    synthetic_corpus = next(
        (corpus for corpus in overview.corpora if corpus.get("id") == "syntheticlib4"),
        {},
    )
    indexed_records = int(synthetic_corpus.get("count") or 0)
    validation_ok = bool(validation.get("valid", False))
    ready = proof_count > 0 and validation_ok and indexed_records > 0
    outdated = _syntheticlib4_is_outdated(account_status, ready=ready)
    status_label = "outdated" if outdated else ("ready" if ready else "not ready")
    return {
        "status": status_label,
        "enabled": True,
        "ready": ready,
        "outdated": outdated,
        "credential_configured": bool(account_status.get("credential_configured")),
        "auth_mode": account_status.get("auth_mode"),
        "release_id": manifest.get("release_id", ""),
        "proof_count": proof_count,
        "indexed_records": indexed_records,
        "validation": validation,
        "message": (
            "SyntheticLib4 is using a valid cached snapshot, but subscription/update access is unavailable."
            if outdated
            else (
                "SyntheticLib4 local proof corpus is ready."
                if ready
                else "SyntheticLib4 is enabled, but the snapshot/index is not ready."
            )
        ),
    }


def _syntheticlib4_is_outdated(account_status: dict[str, Any], *, ready: bool) -> bool:
    """Return True when a usable cached snapshot cannot refresh from live access."""
    if not ready:
        return False
    for field in ("authenticated", "membership_active", "subscription_active", "access_active"):
        if account_status.get(field) is False:
            return True
    if _syntheticlib4_access_expired(account_status.get("access_expires_at")):
        return True
    for field in ("last_refresh_error", "refresh_error", "update_error"):
        if str(account_status.get(field) or "").strip():
            return True
    status_text = str(account_status.get("status") or account_status.get("subscription_status") or "").strip().lower()
    if status_text in {
        "expired",
        "inactive",
        "unauthorized",
        "forbidden",
        "quota_exhausted",
        "refresh_failed",
        "update_failed",
    }:
        return True

    credential_configured = bool(account_status.get("credential_configured"))
    auth_mode = str(account_status.get("auth_mode") or "").strip().lower()
    has_live_access_intent = credential_configured or auth_mode in {
        "api_key",
        "oauth",
        "hosted_oauth",
        "subscription",
    }
    if not has_live_access_intent:
        return False
    return False


def _syntheticlib4_access_expired(value: Any) -> bool:
    raw_value = str(value or "").strip()
    if not raw_value:
        return False
    normalized = raw_value.replace("Z", "+00:00")
    try:
        expires_at = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= datetime.now(timezone.utc)


async def _agent_conversation_memory_status() -> dict[str, Any]:
    enabled = bool(system_config.agent_conversation_memory_enabled)
    if not enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "ready": False,
            "message": "Local agent proof/history memory is disabled for new runs.",
        }
    try:
        overview = await proof_search_service.overview()
    except Exception as exc:
        return {
            "status": "error",
            "enabled": True,
            "ready": False,
            "message": f"Local agent proof/history memory status failed: {exc}",
        }
    local_corpora = {"moto", "manual", "leanoj"}
    local_counts = {
        str(corpus.get("id")): int(corpus.get("count") or 0)
        for corpus in overview.corpora
        if corpus.get("id") in local_corpora
    }
    return {
        "status": "ready",
        "enabled": True,
        "ready": True,
        "local_records": sum(local_counts.values()),
        "local_corpora": local_counts,
        "message": "All stored proofs in memory are ready for AI proof-search retrieval.",
    }


def _wolfram_status() -> dict[str, Any]:
    has_key = bool(system_config.wolfram_alpha_api_key)
    active = bool(system_config.wolfram_alpha_enabled and has_key)
    return {
        "status": "ready" if active else "inactive",
        "enabled": bool(system_config.wolfram_alpha_enabled),
        "active": active,
        "has_key": has_key,
        "message": (
            "Wolfram Alpha tool calls are enabled."
            if active
            else "Wolfram Alpha is disabled or no App ID is configured."
        ),
    }


def _workflow_is_active() -> bool:
    """Return True when a top-level workflow is active."""
    try:
        from backend.aggregator.core.coordinator import coordinator
        from backend.autonomous.core.autonomous_coordinator import autonomous_coordinator
        from backend.compiler.core.compiler_coordinator import compiler_coordinator
        from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator

        return bool(
            coordinator.is_running
            or compiler_coordinator.is_running
            or autonomous_coordinator.is_active
            or leanoj_coordinator.is_active
        )
    except Exception as exc:
        logger.warning("Connectivity workflow activity check failed: %s", exc)
        return True


@router.get("/status")
async def get_connectivity_status() -> dict[str, Any]:
    """Return non-secret provider and optional-skill connectivity state."""
    openrouter_oauth, lm_studio, syntheticlib4, agent_memory = await asyncio.gather(
        _openrouter_oauth_status(),
        _lm_studio_status(),
        _syntheticlib4_status(),
        _agent_conversation_memory_status(),
    )
    return {
        "success": True,
        "generic_mode": system_config.generic_mode,
        "inference": {
            "openrouter_oauth": openrouter_oauth,
            "lm_studio": lm_studio,
        },
        "skills": {
            "syntheticlib4": syntheticlib4,
            "agent_conversation_memory": agent_memory,
            "wolfram_alpha": _wolfram_status(),
        },
        "boost": boost_manager.get_boost_status(),
    }


@router.post("/toggles")
async def update_connectivity_toggles(request: ConnectivityToggleRequest) -> dict[str, Any]:
    """Update non-secret optional-skill toggles without clearing credentials."""
    if _workflow_is_active():
        raise HTTPException(
            status_code=409,
            detail="Stop the active workflow before changing run-level connectivity toggles.",
        )

    if request.syntheticlib4_enabled is not None:
        system_config.syntheticlib4_enabled = bool(request.syntheticlib4_enabled)
    if request.agent_conversation_memory_enabled is not None:
        system_config.agent_conversation_memory_enabled = bool(request.agent_conversation_memory_enabled)
        if not system_config.agent_conversation_memory_enabled:
            await assistant_proof_search_coordinator.stop_all(
                clear_packs=True,
                broadcast=True,
                reason="agent_conversation_memory_disabled",
            )
    if request.wolfram_alpha_enabled is not None:
        system_config.wolfram_alpha_enabled = bool(request.wolfram_alpha_enabled)

    try:
        save_connectivity_runtime_settings()
    except RuntimeSettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return await get_connectivity_status()

