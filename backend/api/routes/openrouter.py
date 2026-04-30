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
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import json
import httpx
from pathlib import Path

from backend.shared.config import rag_config, system_config
from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import OpenRouterClient
from backend.shared.api_client_manager import api_client_manager
from backend.shared.free_model_manager import free_model_manager
from backend.shared.secret_store import (
    SecretStoreError,
    clear_openrouter_api_key,
    store_openrouter_api_key,
)
from backend.shared.models import FreeModelSettings

router = APIRouter()
logger = logging.getLogger(__name__)


class SetApiKeyRequest(BaseModel):
    """Request body for setting the global OpenRouter API key."""
    api_key: str


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
            "error": None,
            "generic_mode": True,
        }

    try:
        result = await lm_studio_client.check_availability()
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Error checking LM Studio availability: {e}")
        return {
            "success": True,  # Endpoint worked, but LM Studio may not be available
            "available": False,
            "has_models": False,
            "model_count": 0,
            "models": [],
            "error": str(e)
        }


@router.post("/api/openrouter/set-api-key")
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
    try:
        api_key = request.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Validate API key by testing connection
        client = OpenRouterClient(api_key)
        try:
            try:
                models = await client.list_models(raise_on_error=True)
            except httpx.HTTPStatusError as http_exc:
                status_code = http_exc.response.status_code if http_exc.response is not None else 0
                if status_code in (401, 403):
                    raise HTTPException(
                        status_code=400,
                        detail="OpenRouter rejected this API key (unauthorized). Please double-check the key at https://openrouter.ai/keys."
                    ) from http_exc
                logger.warning(
                    "OpenRouter /models returned HTTP %s during key validation; treating as transient.",
                    status_code,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"OpenRouter is temporarily unreachable (HTTP {status_code}). "
                        "Your key was NOT saved. Please try again in a moment."
                    ),
                ) from http_exc
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as net_exc:
                logger.warning(
                    "Network error during OpenRouter key validation: %s", net_exc
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Could not reach OpenRouter to validate the key. "
                        "Your key was NOT saved. Please check your internet connection and try again."
                    ),
                ) from net_exc

            if not models:
                raise HTTPException(
                    status_code=400,
                    detail="OpenRouter returned no models for this key. Please verify the key has access."
                )
            
            # Store the API key globally
            rag_config.openrouter_api_key = api_key
            rag_config.openrouter_enabled = True
            
            # Also configure the API client manager
            api_client_manager.set_openrouter_api_key(api_key)

            if system_config.generic_mode:
                logger.info("Generic mode active - keeping OpenRouter API key in runtime memory only")
                success_message = "OpenRouter API key validated and loaded into runtime memory"
            else:
                # Persist to secure OS-backed storage so the key survives restarts.
                store_openrouter_api_key(api_key)
                success_message = "OpenRouter API key validated and saved"
            
            # Reset exhaustion flags so roles can retry OpenRouter
            free_model_manager.clear_account_exhaustion()
            reset_roles = await api_client_manager.reset_openrouter_fallbacks()
            
            logger.info(f"Global OpenRouter API key set successfully. {len(models)} models available.")
            if reset_roles:
                logger.info(f"Auto-reset {len(reset_roles)} role(s) back to OpenRouter after key update")
            
            return {
                "success": True,
                "message": success_message,
                "model_count": len(models),
                "roles_reset": list(reset_roles.keys())
            }
        finally:
            await client.close()
            
    except SecretStoreError as e:
        logger.error(f"Failed to persist OpenRouter API key securely: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set OpenRouter API key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate API key: {str(e)}")


@router.delete("/api/openrouter/api-key")
async def clear_api_key() -> Dict[str, Any]:
    """
    Clear the global OpenRouter API key.
    
    This disables OpenRouter for all roles that were configured to use it.
    Roles will need to be reconfigured to use LM Studio.
    
    Returns:
        Success status
    """
    try:
        rag_config.openrouter_api_key = None
        rag_config.openrouter_enabled = False
        api_client_manager.set_openrouter_api_key(None)

        if system_config.generic_mode:
            logger.info("Generic mode active - cleared in-memory OpenRouter API key")
            success_message = "OpenRouter API key cleared from runtime memory"
        else:
            clear_openrouter_api_key()
            success_message = "OpenRouter API key cleared"
        
        logger.info("Global OpenRouter API key cleared")
        
        return {
            "success": True,
            "message": success_message
        }
    except SecretStoreError as e:
        logger.error(f"Failed to clear OpenRouter API key from secure storage: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to clear OpenRouter API key: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear API key: {str(e)}")


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
        "enabled": rag_config.openrouter_enabled
    }


@router.get("/api/openrouter/models")
async def get_models(api_key: Optional[str] = None, free_only: bool = False) -> Dict[str, Any]:
    """
    Fetch available OpenRouter models.
    
    If api_key is provided, uses that key. Otherwise uses the stored global key.
    
    Args:
        api_key: Optional API key to use instead of stored key (query parameter)
        free_only: If True, only return models with $0 pricing (query parameter)
        
    Returns:
        List of available models with their details
    """
    try:
        # Use provided key or fall back to stored key
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
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")


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
        logger.error(f"Failed to fetch providers for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch providers: {str(e)}")


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
    try:
        free_model_manager.configure(
            looping=request.looping_enabled,
            auto_selector=request.auto_selector_enabled
        )
        return {
            "success": True,
            "message": "Free model settings updated",
            **free_model_manager.get_status()
        }
    except Exception as e:
        logger.error(f"Failed to update free model settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        
        client = OpenRouterClient(request.api_key)
        try:
            models = await client.list_models()
            
            return {
                "success": True,
                "connected": True,
                "model_count": len(models),
                "message": f"Successfully connected to OpenRouter. {len(models)} models available."
            }
        finally:
            await client.close()
            
    except Exception as e:
        logger.error(f"OpenRouter connection test failed: {e}")
        return {
            "success": True,  # Endpoint worked
            "connected": False,
            "model_count": 0,
            "message": f"Failed to connect: {str(e)}"
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
        
        roles_list = list(reset_roles.keys())
        logger.info(f"Credit exhaustion reset: {len(roles_list)} role(s) restored, account exhaustion flag cleared")
        
        return {
            "success": True,
            "message": f"Reset {len(roles_list)} role(s) back to OpenRouter" if roles_list else "Exhaustion flags cleared (no roles needed reset)",
            "roles_reset": roles_list,
            "account_exhaustion_cleared": True
        }
    except Exception as e:
        logger.error(f"Failed to reset credit exhaustion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")
