"""
API routes for OpenRouter configuration and model management.

This module handles:
- Global OpenRouter API key management (for per-role model selection)
- LM Studio availability checking
- OpenRouter model listing (using stored API key)
- Model provider listing

Note: Boost routes can reuse the active global key by default, while still allowing
an explicit boost-only override key when the user provides one.
"""
from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import json
import httpx
import hmac
import time
from pathlib import Path

from backend.shared.config import rag_config, system_config
from backend.shared.credential_locks import openrouter_credential_mutation
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import OpenRouterClient
from backend.shared.api_client_manager import api_client_manager
from backend.shared.embedding_readiness import check_lm_studio_embedding_ready
from backend.shared.free_model_manager import free_model_manager
from backend.shared.log_redaction import redact_log_text
from backend.shared.provider_pause import resume_provider_pauses
from backend.shared.runtime_settings import RuntimeSettingsError, save_free_model_runtime_settings
from backend.shared.secret_store import (
    SecretStoreError,
    clear_openrouter_api_key,
    load_openrouter_api_key,
    store_openrouter_api_key,
)
from backend.shared.models import FreeModelSettings

router = APIRouter()
logger = logging.getLogger(__name__)
_openrouter_validation_state = "unknown"
# The key the validation state was measured for, held in memory only like
# rag_config's copy, so a replaced key never inherits another key's result.
_validated_openrouter_key: Optional[str] = None
_validated_openrouter_key_at: Optional[float] = None
_OPENROUTER_VALIDATION_TTL_SECONDS = 5 * 60


def _normalized_key(api_key: Optional[str]) -> Optional[str]:
    return (api_key or "").strip() or None


def _validation_belongs_to(api_key: Optional[str]) -> bool:
    current = _normalized_key(api_key)
    if current is None or _validated_openrouter_key is None:
        return False
    return hmac.compare_digest(_validated_openrouter_key.encode("utf-8"), current.encode("utf-8"))


def _current_validation_state() -> str:
    if not rag_config.openrouter_api_key:
        return "missing"
    if not _validation_belongs_to(rag_config.openrouter_api_key):
        return "unknown"
    if (
        _openrouter_validation_state == "valid"
        and (
            _validated_openrouter_key_at is None
            or time.monotonic() - _validated_openrouter_key_at > _OPENROUTER_VALIDATION_TTL_SECONDS
        )
    ):
        return "unknown"
    return _openrouter_validation_state


class SetApiKeyRequest(BaseModel):
    """Request body for setting the global OpenRouter API key."""
    api_key: str


async def _validate_openrouter_key(api_key: str) -> Dict[str, Any]:
    client = OpenRouterClient(api_key)
    try:
        return await client.get_current_api_key()
    finally:
        await client.close()


def _raise_openrouter_validation_error(exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else 0
        if status_code in (401, 403):
            raise HTTPException(
                status_code=400,
                detail="OpenRouter rejected this API key. Double-check or replace it at https://openrouter.ai/keys.",
            ) from exc
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException)):
        raise HTTPException(
            status_code=502,
            detail="Could not reach OpenRouter to validate the key. The key was not saved.",
        ) from exc
    raise HTTPException(
        status_code=502,
        detail="OpenRouter could not validate the API key. The key was not saved.",
    ) from exc


@router.get("/api/openrouter/lm-studio-availability")
async def check_lm_studio_availability() -> Dict[str, Any]:
    """
    Check if LM Studio server is available and has models loaded.
    
    This endpoint is called on frontend startup to determine if the system
    should default to OpenRouter (when LM Studio is unavailable).
    
    Returns:
        Dict with availability status:
        - available: bool - True if LM Studio server is reachable
        - has_models: bool - True if at least one model is loaded
        - model_count: int - Number of loaded models
        - models: List[str] - List of loaded model IDs
        - error: Optional[str] - Error message if unavailable
    """
    if system_config.generic_mode:
        return {
            "success": True,
            "available": False,
            "has_models": False,
            "model_count": 0,
            "models": [],
            "has_embedding_model": False,
            "embedding_ready": False,
            "embedding_message": "LM Studio is disabled in generic mode.",
            "error": None,
            "generic_mode": True,
        }

    try:
        result = await lm_studio_client.check_availability()
        embedding_status = await check_lm_studio_embedding_ready(timeout_seconds=5.0)
        return {
            "success": True,
            **result,
            "has_embedding_model": bool(embedding_status.get("ready")),
            "embedding_ready": bool(embedding_status.get("ready")),
            "embedding_message": embedding_status.get("message"),
        }
    except Exception as e:
        logger.error(f"Error checking LM Studio availability: {e}")
        return {
            "success": True,  # Endpoint worked, but LM Studio may not be available
            "available": False,
            "has_models": False,
            "model_count": 0,
            "models": [],
            "has_embedding_model": False,
            "embedding_ready": False,
            "embedding_message": "Failed to check LM Studio embedding availability",
            "error": "Failed to check LM Studio availability"
        }


@router.post("/api/openrouter/set-api-key")
@openrouter_credential_mutation
async def set_api_key(request: SetApiKeyRequest) -> Dict[str, Any]:
    """
    Set the global OpenRouter API key for per-role model selection.
    
    This key is stored in memory and used by the API client manager for
    roles configured to use OpenRouter. It's separate from the boost API key.
    Also resets any credit exhaustion flags so roles can retry OpenRouter.
    
    Args:
        request: Request with api_key field
        
    Returns:
        Success status and validation result
    """
    global _openrouter_validation_state, _validated_openrouter_key, _validated_openrouter_key_at
    try:
        api_key = request.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        try:
            await _validate_openrouter_key(api_key)
        except HTTPException:
            raise
        except Exception as exc:
            _raise_openrouter_validation_error(exc)

        if not system_config.generic_mode:
            store_openrouter_api_key(api_key)
            if load_openrouter_api_key() != api_key:
                raise SecretStoreError("Stored OpenRouter credential could not be verified")
            
        rag_config.openrouter_api_key = api_key
        rag_config.openrouter_enabled = True
        api_client_manager.set_openrouter_api_key(api_key)
        _openrouter_validation_state = "valid"
        _validated_openrouter_key = _normalized_key(api_key)
        _validated_openrouter_key_at = time.monotonic()

        if system_config.generic_mode:
            logger.info("Generic mode active - keeping OpenRouter API key in runtime memory only")
            success_message = "OpenRouter API key validated and loaded into runtime memory"
        else:
            success_message = "OpenRouter API key validated and saved"
            
        free_model_manager.clear_account_exhaustion()
        reset_roles = await api_client_manager.reset_openrouter_fallbacks()
            
        logger.info("Global OpenRouter API key set successfully")
        if reset_roles:
            logger.info(f"Auto-reset {len(reset_roles)} role(s) back to OpenRouter after key update")
            
        return {
            "success": True,
            "message": success_message,
            "validation_state": "valid",
            "roles_reset": list(reset_roles.keys())
        }
            
    except SecretStoreError as e:
        logger.error(f"Failed to persist OpenRouter API key securely: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set OpenRouter API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate API key")


@router.delete("/api/openrouter/api-key")
@openrouter_credential_mutation
async def clear_api_key() -> Dict[str, Any]:
    """
    Clear the global OpenRouter API key.
    
    This disables OpenRouter for all roles that were configured to use it.
    Roles will need to be reconfigured to use LM Studio.
    
    Returns:
        Success status
    """
    global _openrouter_validation_state, _validated_openrouter_key, _validated_openrouter_key_at
    try:
        if system_config.generic_mode:
            success_message = "OpenRouter API key cleared from runtime memory"
        else:
            clear_openrouter_api_key()
            success_message = "OpenRouter API key cleared"

        rag_config.openrouter_api_key = None
        rag_config.openrouter_enabled = False
        api_client_manager.set_openrouter_api_key(None)
        _openrouter_validation_state = "unknown"
        _validated_openrouter_key = None
        _validated_openrouter_key_at = None

        logger.info("Global OpenRouter API key cleared")
        
        return {
            "success": True,
            "message": success_message
        }
    except SecretStoreError as e:
        logger.error(f"Failed to clear OpenRouter API key from secure storage: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.error(f"Failed to clear OpenRouter API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear API key")


@router.get("/api/openrouter/api-key-status")
async def get_api_key_status() -> Dict[str, Any]:
    """
    Get the current status of the global OpenRouter API key.
    
    Returns:
        Dict with:
        - has_key: bool - True if an API key is configured
        - enabled: bool - True if OpenRouter is enabled
    """
    return {
        "success": True,
        "has_key": bool(rag_config.openrouter_api_key),
        "enabled": rag_config.openrouter_enabled,
        "validation_state": _current_validation_state(),
    }


@router.post("/api/openrouter/validate-current-key")
@openrouter_credential_mutation
async def validate_current_api_key() -> Dict[str, Any]:
    """Validate the restored/runtime key without returning credential material."""
    global _openrouter_validation_state, _validated_openrouter_key, _validated_openrouter_key_at
    api_key = (rag_config.openrouter_api_key or "").strip()
    if not api_key:
        _openrouter_validation_state = "unknown"
        _validated_openrouter_key = None
        _validated_openrouter_key_at = None
        return {"success": True, "has_key": False, "validation_state": "missing"}
    try:
        await _validate_openrouter_key(api_key)
        _openrouter_validation_state = "valid"
        _validated_openrouter_key_at = time.monotonic()
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403):
            _openrouter_validation_state = "invalid"
        else:
            _openrouter_validation_state = "unreachable"
        _validated_openrouter_key_at = time.monotonic()
    except Exception:
        _openrouter_validation_state = "unreachable"
        _validated_openrouter_key_at = time.monotonic()
    _validated_openrouter_key = _normalized_key(api_key)
    return {
        "success": True,
        "has_key": True,
        "validation_state": _openrouter_validation_state,
    }


@router.get("/api/openrouter/models")
async def get_models(request: Request, free_only: bool = False, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Fetch available OpenRouter models.
    
    If Authorization is provided, uses that key. Otherwise uses the stored global key.
    
    Args:
        free_only: If True, only return models with $0 pricing (query parameter)
        authorization: Optional API key via Authorization header (Bearer token)
        
    Returns:
        List of available models with their details
    """
    try:
        if "api_key" in request.query_params:
            raise HTTPException(
                status_code=400,
                detail="OpenRouter API keys must be supplied via Authorization header, not URL query parameters.",
            )

        api_key = authorization.replace("Bearer ", "") if authorization and authorization.startswith("Bearer ") else authorization
        key_to_use = api_key or rag_config.openrouter_api_key
        
        if not key_to_use:
            raise HTTPException(
                status_code=400,
                detail="No OpenRouter API key available. Please set a global API key or provide one."
            )
        
        client = OpenRouterClient(key_to_use)
        try:
            models = await client.list_models(free_only=free_only)
            
            # Cache free models for rotation (filter extracts free models internally)
            free_model_manager.update_cached_models(models)
            
            return {
                "success": True,
                "models": models,
                "count": len(models),
                "free_only": free_only
            }
        finally:
            await client.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")


@router.get("/api/openrouter/providers/{model_id:path}")
async def get_model_providers(model_id: str, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Fetch available providers for a specific OpenRouter model.
    
    Args:
        model_id: The model ID to get providers for (path parameter)
        authorization: Optional API key via Authorization header (Bearer token).
                      If not provided, uses stored global key.
        
    Returns:
        List of available providers for the model
    """
    try:
        # Extract API key from Authorization header
        api_key = None
        if authorization:
            api_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
        
        # Use provided key or fall back to stored key
        key_to_use = api_key or rag_config.openrouter_api_key
        
        if not key_to_use:
            raise HTTPException(
                status_code=400,
                detail="No OpenRouter API key available. Please set a global API key or provide one."
            )
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        client = OpenRouterClient(key_to_use)
        try:
            endpoints = await client.get_model_endpoints(model_id)
            providers = sorted({
                endpoint.get("provider_name")
                for endpoint in endpoints
                if isinstance(endpoint.get("provider_name"), str) and endpoint.get("provider_name")
            })
            
            return {
                "success": True,
                "model_id": model_id,
                "providers": providers,
                "endpoints": endpoints
            }
        finally:
            await client.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to fetch providers for model %s: %s",
            redact_log_text(model_id, 160),
            redact_log_text(e, 240),
        )
        raise HTTPException(status_code=500, detail="Failed to fetch providers")


@router.get("/api/model-cache")
async def get_model_cache() -> Dict[str, str]:
    """
    Get cached model mappings (display name -> API ID).
    
    This endpoint serves pre-cached OpenRouter model mappings
    for the frontend profile system to convert display names to API IDs.
    
    Returns:
        Dict mapping model display names to OpenRouter API IDs
    """
    try:
        cache_file = Path(system_config.data_dir) / "model_cache.json"
        
        if not cache_file.exists():
            logger.warning(f"Model cache not found at {cache_file}")
            return {}
        
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        
        logger.debug(f"Serving {len(cache)} cached model mappings")
        return cache
        
    except Exception as e:
        logger.error(f"Failed to read model cache: {e}")
        return {}


@router.get("/api/openrouter/free-model-settings")
async def get_free_model_settings() -> Dict[str, Any]:
    """Get current free model looping and auto-selector settings."""
    return {
        "success": True,
        **free_model_manager.get_status()
    }


@router.post("/api/openrouter/free-model-settings")
async def set_free_model_settings(request: FreeModelSettings) -> Dict[str, Any]:
    """Update free model looping and auto-selector settings."""
    previous_status = free_model_manager.get_status()
    try:
        free_model_manager.configure(
            looping=request.looping_enabled,
            auto_selector=request.auto_selector_enabled
        )
        save_free_model_runtime_settings()
        return {
            "success": True,
            "message": "Free model settings updated",
            **free_model_manager.get_status()
        }
    except RuntimeSettingsError as e:
        free_model_manager.configure(
            looping=bool(previous_status.get("looping_enabled", False)),
            auto_selector=bool(previous_status.get("auto_selector_enabled", False)),
        )
        logger.error(f"Failed to persist free model settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist free model settings")
    except Exception as e:
        logger.error(f"Failed to update free model settings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/api/openrouter/test-connection")
async def test_connection(request: SetApiKeyRequest) -> Dict[str, Any]:
    """
    Test connection to OpenRouter with a provided API key.
    
    This endpoint validates an API key without storing it.
    
    Args:
        request: Request with api_key field
        
    Returns:
        Connection test result with model count
    """
    try:
        if not request.api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        try:
            await _validate_openrouter_key(request.api_key.strip())
            return {
                "success": True,
                "connected": True,
                "validation_state": "valid",
                "message": "Successfully authenticated with OpenRouter."
            }
        except Exception as exc:
            _raise_openrouter_validation_error(exc)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenRouter connection test failed: {e}")
        return {
            "success": True,  # Endpoint worked
            "connected": False,
            "model_count": 0,
            "message": "Failed to connect to OpenRouter"
        }


@router.post("/api/openrouter/reset-exhaustion")
async def reset_credit_exhaustion() -> Dict[str, Any]:
    """
    Reset all credit exhaustion flags and role fallback states.
    
    Call this after adding credits to OpenRouter so roles can retry
    without restarting the research mode.
    
    Resets:
    - Per-role permanent fallback states (roles that fell back to LM Studio)
    - Account-wide free model exhaustion flag
    
    Returns:
        Success status and list of roles that were reset
    """
    try:
        free_model_manager.clear_account_exhaustion()
        reset_roles = await api_client_manager.reset_openrouter_fallbacks()
        paused_workflows_resumed = resume_provider_pauses()
        
        roles_list = list(reset_roles.keys())
        logger.info(
            "Credit exhaustion reset: %s role(s) restored, account exhaustion flag cleared, %s paused workflow(s) resumed",
            len(roles_list),
            paused_workflows_resumed,
        )
        
        return {
            "success": True,
            "message": f"Reset {len(roles_list)} role(s) back to OpenRouter" if roles_list else "Exhaustion flags cleared (no roles needed reset)",
            "roles_reset": roles_list,
            "account_exhaustion_cleared": True,
            "paused_workflows_resumed": paused_workflows_resumed,
        }
    except Exception as e:
        logger.error(f"Failed to reset credit exhaustion: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset")
