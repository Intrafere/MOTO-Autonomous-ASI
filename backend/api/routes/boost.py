"""
API routes for boost management.

Supports three boost modes:
1. Boost Next X Calls - Counter-based (/api/boost/set-next-count)
2. Category Boost - Role-based (/api/boost/toggle-category/{category})
3. Per-task Toggle - Task ID based (/api/boost/toggle-task/{task_id})

Plus boost logging endpoints for viewing API call history.
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from backend.shared.config import rag_config, system_config
from backend.shared.models import BoostConfig
from backend.shared.boost_manager import boost_manager
from backend.shared.boost_logger import boost_logger
from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import OpenRouterClient

router = APIRouter()
logger = logging.getLogger(__name__)


class BoostNextCountRequest(BaseModel):
    """Request body for setting boost next count."""
    count: int


def _resolve_boost_api_key(api_key: Optional[str], *, allow_current_override: bool = False) -> str:
    """Use an explicit/current boost key when provided, otherwise fall back to the active global key."""
    explicit_key = (api_key or "").strip()
    if explicit_key:
        return explicit_key

    if allow_current_override and boost_manager.boost_config:
        current_key = (boost_manager.boost_config.openrouter_api_key or "").strip()
        if current_key:
            return current_key

    global_key = (rag_config.openrouter_api_key or "").strip()
    if global_key:
        return global_key

    raise HTTPException(
        status_code=400,
        detail="No OpenRouter API key available. Use the active global key or provide one in the boost modal."
    )


@router.post("/api/boost/enable")
async def enable_boost(config: BoostConfig) -> Dict[str, Any]:
    """
    Enable API boost with OpenRouter.
    
    Args:
        config: Boost configuration with optional explicit API key and model
        
    Returns:
        Status and boost configuration
    """
    try:
        explicit_api_key = (config.openrouter_api_key or "").strip()
        effective_api_key = _resolve_boost_api_key(explicit_api_key)
        
        client = OpenRouterClient(effective_api_key)
        try:
            models = await client.list_models()
            
            if not models:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to connect to OpenRouter. Please check your API key."
                )
        finally:
            await client.close()

        # Keep explicit boost override keys in process memory only. When the
        # user relies on the global OpenRouter key, Boost stores no key at all.
        config.openrouter_api_key = explicit_api_key
        
        # Enable boost
        await boost_manager.set_boost_config(config)
        
        provider_info = (
            f", provider={redact_log_text(config.boost_provider, 120)}"
            if config.boost_provider
            else " (auto-routing)"
        )
        logger.info("Boost enabled: model=%s%s", redact_log_text(config.boost_model_id, 160), provider_info)
        
        return {
            "success": True,
            "message": "Boost enabled successfully",
            "config": {
                "model_id": config.boost_model_id,
                "provider": config.boost_provider,
                "reasoning_effort": config.boost_reasoning_effort,
                "context_window": config.boost_context_window,
                "max_output_tokens": config.boost_max_output_tokens
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable boost: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable boost")


@router.post("/api/boost/update-model")
async def update_boost_model(config: BoostConfig) -> Dict[str, Any]:
    """
    Update boost model/API key WITHOUT clearing boost state.
    
    This allows seamless model switching while preserving:
    - boost_next_count
    - boosted_categories
    - boosted_task_ids
    
    Args:
        config: New boost configuration with optional explicit API key and model
        
    Returns:
        Status and updated configuration
    """
    try:
        # Validate that boost is currently enabled
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(
                status_code=400, 
                detail="Boost must be enabled first. Use /api/boost/enable to enable boost."
            )
        
        explicit_api_key = (config.openrouter_api_key or "").strip()
        effective_api_key = _resolve_boost_api_key(
            explicit_api_key,
            allow_current_override=True,
        )
        
        client = OpenRouterClient(effective_api_key)
        try:
            models = await client.list_models()
            
            if not models:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to connect to OpenRouter. Please check your API key."
                )
        finally:
            await client.close()

        if explicit_api_key:
            config.openrouter_api_key = explicit_api_key
        elif boost_manager.boost_config and boost_manager.boost_config.openrouter_api_key:
            config.openrouter_api_key = boost_manager.boost_config.openrouter_api_key
        else:
            config.openrouter_api_key = ""
        
        # Store current boost state before update
        old_boost_next_count = boost_manager.boost_next_count
        old_boosted_categories = boost_manager.boosted_categories.copy()
        old_boosted_task_ids = boost_manager.boosted_task_ids.copy()
        
        # Update config (preserves boost state automatically)
        await boost_manager.set_boost_config(config)
        
        # Log the change
        provider_info = (
            f", provider={redact_log_text(config.boost_provider, 120)}"
            if config.boost_provider
            else " (auto-routing)"
        )
        logger.info(
            "Boost model updated: %s%s; preserved state: boost_next_count=%s, "
            "boosted_categories=%s, boosted_tasks=%s",
            redact_log_text(config.boost_model_id, 160),
            provider_info,
            old_boost_next_count,
            len(old_boosted_categories),
            len(old_boosted_task_ids),
        )
        
        return {
            "success": True,
            "message": "Boost model updated successfully (state preserved)",
            "config": {
                "model_id": config.boost_model_id,
                "provider": config.boost_provider,
                "reasoning_effort": config.boost_reasoning_effort,
                "context_window": config.boost_context_window,
                "max_output_tokens": config.boost_max_output_tokens
            },
            "preserved_state": {
                "boost_next_count": boost_manager.boost_next_count,
                "boosted_categories": list(boost_manager.boosted_categories),
                "boosted_task_count": len(boost_manager.boosted_task_ids)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update boost model: {e}")
        raise HTTPException(status_code=500, detail="Failed to update model")


@router.post("/api/boost/disable")
async def disable_boost() -> Dict[str, Any]:
    """
    Disable API boost.
    
    Returns:
        Status message
    """
    try:
        await boost_manager.clear_boost()
        logger.info("Boost disabled")
        
        return {
            "success": True,
            "message": "Boost disabled successfully"
        }
    except Exception as e:
        logger.error(f"Failed to disable boost: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable boost")


@router.get("/api/boost/status")
async def get_boost_status() -> Dict[str, Any]:
    """
    Get current boost status.
    
    Returns:
        Boost configuration and active tasks
    """
    try:
        status = boost_manager.get_boost_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        logger.error(f"Failed to get boost status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get boost status")


@router.post("/api/boost/toggle-task/{task_id}")
async def toggle_task_boost(task_id: str) -> Dict[str, Any]:
    """
    Toggle boost for a specific task.
    
    Args:
        task_id: Task ID to toggle
        
    Returns:
        New boost state for the task
    """
    try:
        boosted = await boost_manager.toggle_task_boost(task_id)
        
        return {
            "success": True,
            "task_id": task_id,
            "boosted": boosted
        }
    except Exception as e:
        logger.error(f"Failed to toggle task boost: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle task boost")


@router.get("/api/boost/openrouter-models")
async def get_openrouter_models(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Fetch available OpenRouter models.
    
    Args:
        authorization: Optional OpenRouter API key via Authorization header (Bearer token)
        
    Returns:
        List of available models
    """
    try:
        api_key = authorization.replace("Bearer ", "") if authorization and authorization.startswith("Bearer ") else authorization

        client = OpenRouterClient(_resolve_boost_api_key(api_key))
        try:
            models = await client.list_models()
        finally:
            await client.close()
        
        return {
            "success": True,
            "models": models
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch models")


@router.get("/api/boost/model-providers")
async def get_model_providers(model_id: str, authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Fetch available providers for a specific OpenRouter model.
    
    Args:
        model_id: The model ID to get providers for (query parameter)
        authorization: Optional OpenRouter API key via Authorization header (Bearer token)
        
    Returns:
        List of available providers for the model
    """
    try:
        api_key = authorization.replace("Bearer ", "") if authorization and authorization.startswith("Bearer ") else authorization
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        client = OpenRouterClient(_resolve_boost_api_key(api_key))
        try:
            endpoints = await client.get_model_endpoints(model_id)
            providers = sorted({
                endpoint.get("provider_name")
                for endpoint in endpoints
                if isinstance(endpoint.get("provider_name"), str) and endpoint.get("provider_name")
            })
        finally:
            await client.close()
        
        return {
            "success": True,
            "model_id": model_id,
            "providers": providers,
            "endpoints": endpoints
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to fetch providers for model %s: %s",
            redact_log_text(model_id, 160),
            redact_log_text(e, 240),
        )
        raise HTTPException(status_code=500, detail="Failed to fetch providers")


# ============================================================
# NEW: Boost Next X Calls (Counter-based mode)
# ============================================================

class BoostAlwaysPreferRequest(BaseModel):
    """Request body for toggling always-prefer-boost mode."""
    enabled: bool


@router.post("/api/boost/set-always-prefer")
async def set_boost_always_prefer(request: BoostAlwaysPreferRequest) -> Dict[str, Any]:
    """
    Enable or disable always-prefer-boost mode.
    
    When enabled, every API call attempts boost first and falls back to the
    primary model on any failure. Mutually exclusive with next_count and category
    modes — caller should clear those first.
    """
    try:
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(status_code=400, detail="Boost must be enabled first")
        
        await boost_manager.set_always_prefer(request.enabled)
        
        logger.info("Boost always-prefer set to %s", redact_log_text(request.enabled, 20))
        
        return {
            "success": True,
            "enabled": request.enabled,
            "message": "Boost will be attempted for every API call" if request.enabled else "Always-prefer boost disabled"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set always-prefer boost: {e}")
        raise HTTPException(status_code=500, detail="Failed to set always-prefer")


@router.post("/api/boost/set-next-count")
async def set_boost_next_count(request: BoostNextCountRequest) -> Dict[str, Any]:
    """
    Set the number of next API calls to boost.
    
    This mode boosts the next X API calls regardless of task ID or category.
    The counter decrements after each boosted call.
    
    Args:
        request: Request with count field
        
    Returns:
        Success status and new count
    """
    try:
        if request.count < 0:
            raise HTTPException(status_code=400, detail="Count must be non-negative")
        
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(status_code=400, detail="Boost must be enabled first")
        
        await boost_manager.set_boost_next_count(request.count)
        
        logger.info("Boost next count set to %s", redact_log_text(request.count, 20))
        
        return {
            "success": True,
            "message": f"Will boost the next {request.count} API calls",
            "count": request.count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set boost next count: {e}")
        raise HTTPException(status_code=500, detail="Failed to set count")


@router.post("/api/boost/toggle-category/{category}")
async def toggle_category_boost(category: str) -> Dict[str, Any]:
    """
    Toggle boost for an entire category (role prefix).
    
    When a category is boosted, ALL API calls for that role will use boost.
    
    Categories:
    - Aggregator: agg_sub1, agg_sub2, ..., agg_sub10, agg_val
    - Compiler: comp_writer, comp_hp, comp_val
    - Autonomous: auto_ts, auto_tv, auto_cr, auto_rs, auto_pt, auto_prc
    
    Args:
        category: Category prefix to toggle
        
    Returns:
        New boost state for the category
    """
    try:
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            raise HTTPException(status_code=400, detail="Boost must be enabled first")
        
        boosted = await boost_manager.toggle_category_boost(category)
        
        return {
            "success": True,
            "category": category,
            "boosted": boosted,
            "all_boosted_categories": list(boost_manager.boosted_categories)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle category boost: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle category")


@router.get("/api/boost/categories")
async def get_boost_categories(mode: Optional[str] = "all") -> Dict[str, Any]:
    """
    Get available boost categories for the current workflow mode.
    
    Args:
        mode: "aggregator", "compiler", "autonomous", or "all" (default)
        
    Returns:
        List of available categories with their current boost state
    """
    try:
        categories = boost_manager.get_available_categories(mode)
        
        # Add boosted state to each category
        for cat in categories:
            cat["boosted"] = cat["id"] in boost_manager.boosted_categories
        
        return {
            "success": True,
            "categories": categories,
            "boosted_categories": list(boost_manager.boosted_categories)
        }
    except Exception as e:
        logger.error(f"Failed to get boost categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get categories")


# ============================================================
# NEW: Boost Logs
# ============================================================

@router.get("/api/boost/logs")
async def get_boost_logs(limit: int = 100) -> Dict[str, Any]:
    """
    Get recent boost API call logs.
    
    Args:
        limit: Maximum number of log entries to return (default 100)
        
    Returns:
        List of log entries (most recent first)
    """
    try:
        logs = await boost_logger.get_logs(limit, include_full=False)
        stats = await boost_logger.get_stats()
        
        return {
            "success": True,
            "logs": logs,
            "stats": stats,
            "total": len(logs)
        }
    except Exception as e:
        logger.error(f"Failed to get boost logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get logs")


@router.get("/api/boost/logs/{index}")
async def get_boost_log_entry(index: int) -> Dict[str, Any]:
    """
    Get a specific log entry with full response content.
    
    Args:
        index: Index of the log entry (0 = most recent)
        
    Returns:
        Full log entry including complete response
    """
    try:
        entry = await boost_logger.get_log_entry(index, include_full=system_config.api_log_store_full_payloads)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Log entry not found")
        
        return {
            "success": True,
            "entry": entry
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get boost log entry: {e}")
        raise HTTPException(status_code=500, detail="Failed to get entry")


@router.post("/api/boost/clear-logs")
async def clear_boost_logs() -> Dict[str, Any]:
    """
    Clear all boost API logs.
    
    Returns:
        Success status
    """
    try:
        await boost_logger.clear_logs()
        
        logger.info("Boost logs cleared")
        
        return {
            "success": True,
            "message": "Boost logs cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear boost logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear logs")

