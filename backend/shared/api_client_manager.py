"""
API Client Manager - Unified manager for routing API calls to OpenRouter or LM Studio.
Handles fallback on credit exhaustion and boost integration.

Supports four boost modes:
1. Boost Next X Calls - Counter-based, applies to next X API calls
2. Category Boost - Role-based, boosts all calls for specific role categories
3. Always Prefer Boost - Tries boost for every call, falls back on failure
4. Per-task Toggle - Task ID based (legacy)
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import (
    OpenRouterClient, 
    CreditExhaustionError,
    OpenRouterPrivacyPolicyError,
    RateLimitError,
    FreeModelExhaustedError
)
from backend.shared.boost_manager import boost_manager
from backend.shared.boost_logger import boost_logger
from backend.shared.config import rag_config, system_config
from backend.shared.fastembed_provider import FASTEMBED_MODEL_NAME, FastEmbedProvider
from backend.shared.free_model_manager import free_model_manager
from backend.shared.models import ModelConfig
from backend.shared.token_tracker import token_tracker

logger = logging.getLogger(__name__)


class APIClientManager:
    """
    Central manager for routing API calls to OpenRouter or LM Studio.
    Handles fallback on credit exhaustion and boost integration.
    """
    CALL_METADATA_KEY = "_moto_call_metadata"
    
    def __init__(self):
        self._openrouter_client: Optional[OpenRouterClient] = None
        self._openrouter_api_key: Optional[str] = None
        self._fastembed_provider: Optional[FastEmbedProvider] = None
        
        # Track which roles have fallen back to LM Studio
        # Format: {role_id: "openrouter" | "lm_studio"}
        self._role_fallback_state: Dict[str, str] = {}
        
        # Track model configurations per role
        # Format: {role_id: ModelConfig}
        self._role_model_configs: Dict[str, ModelConfig] = {}
        
        # WebSocket broadcaster
        self._broadcast_callback: Optional[Callable] = None
        
        # Model tracking callback for Tier 3
        # Called after each successful API call with the model ID used
        # Signature: async callback(model_id: str)
        self._model_tracking_callback: Optional[Callable] = None
        
        # Autonomous API logger callback
        # Called after each API call (success or failure) with full details
        # Signature: async callback(task_id, role_id, model, provider, prompt, response, duration_ms, success, error, phase)
        self._autonomous_logger_callback: Optional[Callable] = None
        
        # Current autonomous phase (set by autonomous coordinator)
        self._current_autonomous_phase: str = "unknown"
        
        # Track roles that have already broadcast fallback_failed (prevent GUI log spam)
        self._fallback_failed_notified: set = set()
        
        # Lock for thread-safe state updates
        self._state_lock = asyncio.Lock()
    
    def set_broadcast_callback(self, callback: Callable) -> None:
        """Set callback for broadcasting WebSocket events."""
        self._broadcast_callback = callback
    
    async def _broadcast(self, event: str, data: Dict[str, Any] = None) -> None:
        """Broadcast an event through WebSocket."""
        if self._broadcast_callback:
            await self._broadcast_callback(event, data or {})
    
    async def _with_hung_connection_watchdog(
        self,
        coro,
        role_id: str,
        model: str,
        provider: str,
        timeout_seconds: int = 900
    ):
        """Wrap an API call coroutine with a watchdog that alerts after timeout_seconds (default 15 min)."""
        async def _watchdog():
            await asyncio.sleep(timeout_seconds)
            minutes = timeout_seconds // 60
            logger.warning(
                f"API call for role '{role_id}' using {model} via {provider} "
                f"has been running for {minutes}+ minutes — possible hung connection"
            )
            await self._broadcast("hung_connection_alert", {
                "role_id": role_id,
                "model": model,
                "provider": provider,
                "elapsed_minutes": minutes,
                "message": (
                    f"API call to {model} via {provider} has been running for {minutes}+ minutes. "
                    f"The connection may be hung. Consider stopping and trying a different host/provider."
                )
            })

        watchdog_task = asyncio.create_task(_watchdog())
        try:
            return await coro
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass

    def set_model_tracking_callback(self, callback: Optional[Callable]) -> None:
        """
        Set callback for model usage tracking during Tier 3 final answer generation.
        
        The callback is called after each successful API call with the model ID used.
        Used to track which models contribute to the final answer and tally API calls.
        
        Args:
            callback: Async function that takes model_id (str) as argument, or None to disable
        """
        self._model_tracking_callback = callback
        if callback:
            logger.info("Model tracking callback set for Tier 3")
        else:
            logger.info("Model tracking callback cleared")
    
    def set_autonomous_logger_callback(self, callback: Optional[Callable]) -> None:
        """
        Set callback for autonomous API logging.
        
        The callback is called after each API call with full details for logging.
        
        Args:
            callback: Async function with signature: 
                      callback(task_id, role_id, model, provider, prompt, response, 
                               duration_ms, success, error, phase)
                      or None to disable
        """
        self._autonomous_logger_callback = callback
        if callback:
            logger.info("Autonomous API logger callback set")
        else:
            logger.info("Autonomous API logger callback cleared")
    
    def set_autonomous_phase(self, phase: str) -> None:
        """
        Set the current autonomous research phase for logging context.
        
        Args:
            phase: Phase identifier ("topic_selection", "brainstorm", "paper_compilation", "tier3")
        """
        self._current_autonomous_phase = phase
    
    async def _track_model_usage(self, model_id: str) -> None:
        """
        Track model usage if tracking callback is set.
        
        Args:
            model_id: The model ID that was used for the API call
        """
        if self._model_tracking_callback:
            try:
                await self._model_tracking_callback(model_id)
            except Exception as e:
                logger.error(f"Error in model tracking callback: {e}")

    def _annotate_response_with_call_metadata(
        self,
        response: Dict[str, Any],
        *,
        task_id: str,
        role_id: str,
        configured_model: str,
        actual_model: str,
        configured_provider: Optional[str],
        actual_provider: str,
        boosted: bool,
        boost_mode: Optional[str] = None,
        openrouter_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach effective routing details to a successful API response."""
        if not isinstance(response, dict):
            return response

        response[self.CALL_METADATA_KEY] = {
            "task_id": task_id,
            "role_id": role_id,
            "configured_model": configured_model,
            "effective_model": actual_model,
            "configured_provider": configured_provider or actual_provider,
            "effective_provider": actual_provider,
            "provider": actual_provider,
            "boosted": boosted,
            "boost_mode": boost_mode,
            "openrouter_provider": openrouter_provider,
        }
        return response

    def extract_call_metadata(self, response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return routing metadata attached to a successful API response."""
        if not isinstance(response, dict):
            return {}

        metadata = response.get(self.CALL_METADATA_KEY)
        if isinstance(metadata, dict):
            return metadata.copy()
        return {}
    
    def set_openrouter_api_key(self, api_key: str) -> None:
        """
        Set OpenRouter API key and initialize client.
        
        Args:
            api_key: OpenRouter API key
        """
        self._openrouter_api_key = api_key
        if api_key:
            self._openrouter_client = OpenRouterClient(api_key)
            logger.info("OpenRouter client initialized")
        else:
            self._openrouter_client = None
            logger.info("OpenRouter client disabled (no API key)")

    def _get_fastembed_provider(self, model_name: Optional[str] = None) -> FastEmbedProvider:
        """Return the hosted in-process embedding provider for generic mode."""
        desired_model = model_name or FASTEMBED_MODEL_NAME
        if self._fastembed_provider is None or self._fastembed_provider.model_name != desired_model:
            self._fastembed_provider = FastEmbedProvider(model_name=desired_model)
        return self._fastembed_provider
    
    def configure_role(self, role_id: str, config: ModelConfig) -> None:
        """
        Configure a role with model settings.
        
        Args:
            role_id: Role identifier (e.g., "aggregator_submitter_1", "compiler_validator")
            config: Model configuration (includes provider, model_id, openrouter_model_id, 
                    lm_studio_fallback_id, and optionally openrouter_provider)
        """
        self._role_model_configs[role_id] = config
        
        # Set initial fallback state based on provider
        if config.provider == "openrouter":
            self._role_fallback_state[role_id] = "openrouter"
        else:
            self._role_fallback_state[role_id] = "lm_studio"
        
        # Log configuration with provider details if OpenRouter
        if config.provider == "openrouter":
            or_model = config.openrouter_model_id or config.model_id
            provider_str = f" via {config.openrouter_provider}" if config.openrouter_provider else ""
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.info(f"Configured role '{role_id}': provider=openrouter, model={or_model}{provider_str}{fallback_str}")
        else:
            logger.info(f"Configured role '{role_id}': provider=lm_studio, model={config.model_id}")
    
    def _determine_boost_mode(self, task_id: str) -> Optional[str]:
        """
        Determine which boost mode (if any) applies to this task.
        
        Returns:
            "next_count", "category", "task_id", or None
        """
        if not boost_manager.boost_config or not boost_manager.boost_config.enabled:
            return None
        
        # Check always-prefer mode (every call uses boost, fall back on failure)
        if boost_manager.boost_always_prefer:
            return "always_prefer"
        
        # Check boost_next_count first (counter-based mode)
        if boost_manager.boost_next_count > 0:
            return "next_count"
        
        # Check category boost (role-based mode)
        role_prefix = boost_manager._extract_role_prefix(task_id)
        if role_prefix in boost_manager.boosted_categories:
            return "category"
        
        # Check exact task ID (legacy per-task mode)
        if task_id in boost_manager.boosted_task_ids:
            return "task_id"
        
        return None
    
    async def generate_completion(
        self,
        task_id: str,
        role_id: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion using the appropriate API.
        
        Routing logic:
        1. Check if task should use boost (via should_use_boost) → Use boost OpenRouter model
        2. Check role fallback state:
           - If "openrouter" and not fallen back → Try OpenRouter
           - If "lm_studio" or fallen back → Use LM Studio
        3. On OpenRouter credit exhaustion → Fall back to LM Studio permanently
        
        Args:
            task_id: Task ID to check boost state
            role_id: Role identifier for fallback tracking
            model: Model identifier (LM Studio format)
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional response format
            **kwargs: Additional arguments
            
        Returns:
            API response dict
        """
        requested_model = model
        async with self._state_lock:
            initial_role_config = self._role_model_configs.get(role_id)
        configured_provider = initial_role_config.provider if initial_role_config else None

        # Check if task should use boost (unified check for all boost modes)
        boost_mode = self._determine_boost_mode(task_id)
        
        if boost_mode and boost_manager.boost_config:
            boost_model = boost_manager.boost_config.boost_model_id
            boost_provider = boost_manager.boost_config.boost_provider
            provider_info = f" via {boost_provider}" if boost_provider else " (auto-routing)"
            logger.info(f"Task {task_id} using boost ({boost_mode}): {boost_model}{provider_info}")
            
            # Get prompt preview for logging
            prompt_preview = ""
            if messages:
                last_message = messages[-1].get("content", "")
                prompt_preview = last_message[:500] if last_message else ""
            
            start_time = time.time()
            
            try:
                boost_api_key = (
                    boost_manager.boost_config.openrouter_api_key or
                    rag_config.openrouter_api_key
                )
                if not boost_api_key:
                    raise RuntimeError("Boost requested but no OpenRouter API key is available")

                # Create temporary client with boost API key
                boost_client = OpenRouterClient(boost_api_key)
                boost_provider = boost_manager.boost_config.boost_provider
                try:
                    result = await self._with_hung_connection_watchdog(
                        boost_client.generate_completion(
                            model=boost_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens or boost_manager.boost_config.boost_max_output_tokens,
                            response_format=response_format,
                            provider=boost_provider,
                            tools=tools,
                            tool_choice=tool_choice,
                        ),
                        role_id=role_id,
                        model=boost_model,
                        provider=boost_provider or "OpenRouter"
                    )
                    
                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Check for missing choices (upstream provider timeout/error)
                    if not result.get("choices"):
                        import json as _json
                        raw_response = _json.dumps(result)[:2000]
                        logger.error(f"OpenRouter boost response missing 'choices' after {duration_ms:.0f}ms - raw: {raw_response}")
                        
                        # Log as failure
                        await boost_logger.log_boost_call(
                            task_id=task_id,
                            role_id=role_id,
                            model=boost_model,
                            prompt_preview=prompt_preview,
                            response_content="",
                            tokens_used=None,
                            duration_ms=duration_ms,
                            success=False,
                            boost_mode=boost_mode,
                            error="Response missing 'choices' - upstream provider timeout or error"
                        )
                        
                        # Raise so retry/fallback logic can handle it
                        raise ValueError(f"OpenRouter response missing 'choices' after {duration_ms:.0f}ms (upstream provider timeout)")
                    
                    # Extract response content for logging
                    response_content = ""
                    tokens_used = None
                    
                    if result.get("choices"):
                        message = result["choices"][0].get("message", {})
                        response_content = message.get("content") or message.get("reasoning") or ""
                    if result.get("usage"):
                        tokens_used = result["usage"].get("total_tokens")
                        _pt = result["usage"].get("prompt_tokens")
                        _ct = result["usage"].get("completion_tokens")
                        if _pt is not None and _ct is not None:
                            token_tracker.track(boost_model, _pt, _ct)
                            await self._broadcast("token_usage_updated", token_tracker.get_stats())

                    result = self._annotate_response_with_call_metadata(
                        result,
                        task_id=task_id,
                        role_id=role_id,
                        configured_model=requested_model,
                        actual_model=boost_model,
                        configured_provider=configured_provider,
                        actual_provider="openrouter",
                        boosted=True,
                        boost_mode=boost_mode,
                        openrouter_provider=boost_provider,
                    )
                    
                    # Log the boost call
                    await boost_logger.log_boost_call(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        prompt_preview=prompt_preview,
                        response_content=response_content,
                        tokens_used=tokens_used,
                        duration_ms=duration_ms,
                        success=True,
                        boost_mode=boost_mode
                    )
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=boost_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response=response_content,
                            tokens_used=tokens_used,
                            duration_ms=duration_ms,
                            success=True,
                            error=None,
                            phase=self._current_autonomous_phase
                        )
                    
                    # Track model usage for Tier 3
                    await self._track_model_usage(boost_model)
                    
                    # Consume boost count if using next_count mode
                    if boost_mode == "next_count":
                        await boost_manager.consume_boost_count()
                    
                    return result
                finally:
                    await boost_client.close()
                    
            except RateLimitError as e:
                # Rate limit error - log and fall through to primary (boost has no fallback concept)
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    boost_mode=boost_mode
                )
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=f"Rate Limit: {str(e)}",
                        phase=self._current_autonomous_phase
                    )
                
                logger.warning(f"Boost model rate limited for task {task_id}: {e}")
                
                # Broadcast rate limit event to frontend
                await self._broadcast("openrouter_rate_limit", {
                    "model": boost_model,
                    "role_id": role_id,
                    "message": f"OpenRouter rate limit hit for '{boost_model}' after retries exhausted."
                })
                
                # Fall through to primary model (boost has no fallback concept)
                logger.info(f"Boost rate limited, using primary model for task {task_id}")
            
            except OpenRouterPrivacyPolicyError as e:
                # Privacy policy error - log and crash (boost has no fallback concept)
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    boost_mode=boost_mode
                )
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=f"Privacy Policy Error: {str(e)}",
                        phase=self._current_autonomous_phase
                    )
                
                logger.error(f"OpenRouter privacy policy error for boost task {task_id}: {e}")
                
                # Broadcast warning to frontend
                await self._broadcast("openrouter_privacy_error", {
                    "error_type": "privacy_policy",
                    "model": boost_model,
                    "role_id": role_id,
                    "message": str(e),
                    "solution_url": "https://openrouter.ai/settings/privacy",
                    "solution_text": (
                        "To use free models on OpenRouter:\n\n"
                        "1. Visit https://openrouter.ai/settings/privacy\n"
                        "2. Enable 'Allow my data to be used for model training'\n"
                        "3. Save your settings\n\n"
                        "Free models on OpenRouter require this setting because they are "
                        "subsidized through training data collection. Alternatively, you can:\n\n"
                        "• Use a paid OpenRouter model instead\n"
                        "• Configure an LM Studio fallback model in settings"
                    )
                })
                
                # Raise clear error - boost mode has no fallback concept
                raise RuntimeError(
                    f"Cannot use boost: OpenRouter privacy settings are blocking free models. "
                    f"Please visit https://openrouter.ai/settings/privacy and enable "
                    f"'Allow my data to be used for model training', OR use a paid OpenRouter model."
                )
                
            except CreditExhaustionError as e:
                # Log the failed boost call
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    boost_mode=boost_mode
                )
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase
                    )
                
                # Boost credits exhausted - fall back to primary for this task
                logger.warning(f"Boost credits exhausted for task {task_id}, using primary model")
                await self._broadcast("boost_credits_exhausted", {
                    "task_id": task_id,
                    "message": str(e)
                })
                # Continue to primary model routing below
                
            except Exception as e:
                # Log the failed boost call
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    boost_mode=boost_mode
                )
                
                # Log to autonomous API logger if callback set
                if self._autonomous_logger_callback:
                    full_prompt = messages[-1].get("content", "") if messages else ""
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=boost_model,
                        provider="openrouter",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase
                    )
                
                logger.error(f"Boost API error for task {task_id}: {e}, using primary model")
                # Fall through to primary model
        
        # Check role fallback state
        async with self._state_lock:
            fallback_state = self._role_fallback_state.get(role_id, "lm_studio")
            role_config = self._role_model_configs.get(role_id)
        
        # If OpenRouter configured and not fallen back, try OpenRouter
        if fallback_state == "openrouter" and role_config:
            # Lazy-initialize OpenRouter client if needed
            if not self._openrouter_client:
                # Check if API key is available in rag_config
                from backend.shared.config import rag_config
                if rag_config.openrouter_api_key:
                    logger.info(f"Lazy-initializing OpenRouter client for role {role_id}")
                    self.set_openrouter_api_key(rag_config.openrouter_api_key)
                elif not role_config.lm_studio_fallback_id:
                    # No API key AND no fallback - cannot proceed
                    error_msg = (
                        f"Role '{role_id}' is configured for OpenRouter but no API key is set "
                        f"and no LM Studio fallback is configured. Please set OpenRouter API key "
                        f"or configure an LM Studio fallback model."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                else:
                    # No API key but fallback exists - use fallback
                    logger.warning(f"Role '{role_id}' configured for OpenRouter but no API key set. Using LM Studio fallback: {role_config.lm_studio_fallback_id}")
                    model = role_config.lm_studio_fallback_id
                    # Skip OpenRouter block entirely, go to LM Studio
            
            if self._openrouter_client:
                openrouter_model = role_config.openrouter_model_id or role_config.model_id
                openrouter_provider = role_config.openrouter_provider
                
                # Account-wide free credit exhaustion pre-check
                is_free = ":free" in openrouter_model.lower()
                if is_free and free_model_manager.is_account_exhausted():
                    if role_config.lm_studio_fallback_id:
                        logger.warning(
                            f"Account free credits exhausted. Using LM Studio fallback for role '{role_id}': "
                            f"{role_config.lm_studio_fallback_id}"
                        )
                        model = role_config.lm_studio_fallback_id
                    else:
                        await self._broadcast("account_credits_exhausted", {
                            "message": "OpenRouter account free credits depleted. Add credits at openrouter.ai or configure LM Studio fallback."
                        })
                        raise FreeModelExhaustedError(
                            f"Account free credits exhausted and no LM Studio fallback for role '{role_id}'."
                        )
                
                provider_info = f" via {openrouter_provider}" if openrouter_provider else ""
                
                start_time = time.time()
                
                try:
                    logger.debug(f"Role {role_id} using OpenRouter: {openrouter_model}{provider_info}")
                    result = await self._with_hung_connection_watchdog(
                        self._openrouter_client.generate_completion(
                            model=openrouter_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens or role_config.max_output_tokens,
                            response_format=response_format,
                            provider=openrouter_provider,
                            tools=tools,
                            tool_choice=tool_choice,
                        ),
                        role_id=role_id,
                        model=openrouter_model,
                        provider=openrouter_provider or "OpenRouter"
                    )
                    
                    # Calculate duration and extract response
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Check for missing choices (upstream provider timeout/error)
                    if not result.get("choices"):
                        import json as _json
                        raw_response = _json.dumps(result)[:2000]
                        logger.error(f"OpenRouter response missing 'choices' after {duration_ms:.0f}ms - raw: {raw_response}")
                        raise ValueError(f"OpenRouter response missing 'choices' after {duration_ms:.0f}ms (upstream provider timeout)")
                    
                    response_content = ""
                    tokens_used = None
                    if result.get("choices"):
                        message = result["choices"][0].get("message", {})
                        response_content = message.get("content") or message.get("reasoning") or ""
                    if result.get("usage"):
                        tokens_used = result["usage"].get("total_tokens")
                        _pt = result["usage"].get("prompt_tokens")
                        _ct = result["usage"].get("completion_tokens")
                        if _pt is not None and _ct is not None:
                            token_tracker.track(openrouter_model, _pt, _ct)
                            await self._broadcast("token_usage_updated", token_tracker.get_stats())

                    result = self._annotate_response_with_call_metadata(
                        result,
                        task_id=task_id,
                        role_id=role_id,
                        configured_model=requested_model,
                        actual_model=openrouter_model,
                        configured_provider=role_config.provider if role_config else configured_provider or "openrouter",
                        actual_provider="openrouter",
                        boosted=False,
                        boost_mode=None,
                        openrouter_provider=openrouter_provider,
                    )
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response=response_content,
                            tokens_used=tokens_used,
                            duration_ms=duration_ms,
                            success=True,
                            error=None,
                            phase=self._current_autonomous_phase
                        )
                    
                    # Track model usage for Tier 3
                    await self._track_model_usage(openrouter_model)
                    
                    return result
                
                except RateLimitError as e:
                    # Rate limit error - attempt free model rotation chain before fallback
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response="",
                            tokens_used=None,
                            duration_ms=duration_ms,
                            success=False,
                            error=f"Rate Limit: {str(e)}",
                            phase=self._current_autonomous_phase
                        )
                    
                    logger.warning(f"OpenRouter rate limit for role {role_id}: {e}")
                    
                    await self._broadcast("openrouter_rate_limit", {
                        "model": openrouter_model,
                        "role_id": role_id,
                        "message": f"OpenRouter rate limit hit for '{openrouter_model}' after retries exhausted."
                    })
                    
                    # Mark this model as failed for rotation
                    free_model_manager.mark_model_failed(openrouter_model)
                    
                    # --- FREE MODEL ROTATION CHAIN ---
                    rotated_result = await self._try_free_model_rotation(
                        task_id=task_id,
                        role_id=role_id,
                        original_model=openrouter_model,
                        configured_model=requested_model,
                        configured_provider=role_config.provider if role_config else configured_provider or "openrouter",
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens or role_config.max_output_tokens,
                        response_format=response_format,
                    )
                    if rotated_result is not None:
                        free_model_manager.clear_failed_models()  # Success - clear failures
                        return rotated_result
                    
                    # Rotation chain exhausted — try LM Studio fallback
                    if not role_config.lm_studio_fallback_id:
                        raise FreeModelExhaustedError(
                            f"All free model options exhausted for role '{role_id}'. "
                            f"No LM Studio fallback configured."
                        )
                    
                    fallback_model = role_config.lm_studio_fallback_id
                    logger.info(
                        f"Free model rotation exhausted for role '{role_id}'. "
                        f"Temporarily using LM Studio fallback: {fallback_model}"
                    )
                    model = fallback_model
                
                except OpenRouterPrivacyPolicyError as e:
                    # Privacy policy error - try LM Studio fallback if configured
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response="",
                            tokens_used=None,
                            duration_ms=duration_ms,
                            success=False,
                            error=f"Privacy Policy Error: {str(e)}",
                            phase=self._current_autonomous_phase
                        )
                    
                    logger.error(f"OpenRouter privacy policy error for role {role_id}: {e}")
                    
                    # Broadcast warning to frontend
                    await self._broadcast("openrouter_privacy_error", {
                        "error_type": "privacy_policy",
                        "model": openrouter_model,
                        "role_id": role_id,
                        "message": str(e),
                        "solution_url": "https://openrouter.ai/settings/privacy",
                        "solution_text": (
                            "To use free models on OpenRouter:\n\n"
                            "1. Visit https://openrouter.ai/settings/privacy\n"
                            "2. Enable 'Allow my data to be used for model training'\n"
                            "3. Save your settings\n\n"
                            "Free models on OpenRouter require this setting because they are "
                            "subsidized through training data collection. Alternatively, you can:\n\n"
                            "• Use a paid OpenRouter model instead\n"
                            "• Configure an LM Studio fallback model in settings"
                        )
                    })
                    
                    # CHECK: Is fallback configured?
                    if not role_config.lm_studio_fallback_id:
                        # NO FALLBACK - raise clear error
                        error_msg = (
                            f"OpenRouter privacy settings are blocking free models for role '{role_id}' "
                            f"and no LM Studio fallback configured. "
                            f"Please visit https://openrouter.ai/settings/privacy and enable "
                            f"'Allow my data to be used for model training', OR configure an LM Studio "
                            f"fallback model in settings."
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    
                    # Fallback IS configured - use it
                    fallback_model = role_config.lm_studio_fallback_id
                    
                    logger.warning(
                        f"OpenRouter privacy policy blocking free models for role '{role_id}'. "
                        f"Falling back to LM Studio model: {fallback_model}"
                    )
                    
                    # Fall through to LM Studio (don't re-raise)
                    model = fallback_model
                
                except CreditExhaustionError as e:
                    # PERMANENT FALLBACK - OpenRouter credits exhausted for this role
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response="",
                            tokens_used=None,
                            duration_ms=duration_ms,
                            success=False,
                            error=f"Credit Exhaustion: {str(e)}",
                            phase=self._current_autonomous_phase
                        )
                    
                    # CHECK: Is fallback configured?
                    if not role_config.lm_studio_fallback_id:
                        # NO FALLBACK - raise clear error
                        error_msg = (
                            f"OpenRouter credits exhausted for role '{role_id}' "
                            f"and no LM Studio fallback configured. "
                            f"Please add credits to OpenRouter or configure an LM Studio "
                            f"fallback model in settings."
                        )
                        logger.error(error_msg)
                        if role_id not in self._fallback_failed_notified:
                            self._fallback_failed_notified.add(role_id)
                            await self._broadcast("openrouter_fallback_failed", {
                                "role_id": role_id,
                                "reason": "no_fallback_configured",
                                "message": error_msg
                            })
                        raise RuntimeError(error_msg)
                    
                    # Fallback IS configured - use it
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    
                    fallback_model = role_config.lm_studio_fallback_id
                    
                    logger.error(
                        f"OpenRouter credits exhausted for role '{role_id}'. "
                        f"Permanently falling back to LM Studio model: {fallback_model}"
                    )
                    
                    await self._broadcast("openrouter_fallback", {
                        "role_id": role_id,
                        "reason": "credit_exhaustion",
                        "message": str(e),
                        "fallback_model": fallback_model
                    })
                    
                    # Fall through to LM Studio
                    model = fallback_model
                
                except Exception as e:
                    # Other OpenRouter error - fall back for this call only (don't mark as permanent)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = messages[-1].get("content", "") if messages else ""
                        await self._autonomous_logger_callback(
                            task_id=task_id,
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            prompt=full_prompt,
                            response="",
                            tokens_used=None,
                            duration_ms=duration_ms,
                            success=False,
                            error=str(e),
                            phase=self._current_autonomous_phase
                        )
                    
                    # For non-credit errors, only fall back if fallback is configured
                    if role_config.lm_studio_fallback_id:
                        logger.error(
                            f"OpenRouter error for role '{role_id}': {e}, "
                            f"falling back to LM Studio model: {role_config.lm_studio_fallback_id}"
                        )
                        model = role_config.lm_studio_fallback_id
                        # Fall through to LM Studio
                    else:
                        # No fallback configured - re-raise the error
                        logger.error(
                            f"OpenRouter error for role '{role_id}': {e}, "
                            f"and no LM Studio fallback configured"
                        )
                        raise
        
        # Use LM Studio (either configured as primary or fallen back)
        logger.debug(f"Role {role_id} using LM Studio: {model}")
        start_time = time.time()
        
        try:
            result = await self._with_hung_connection_watchdog(
                lm_studio_client.generate_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs
                ),
                role_id=role_id,
                model=model,
                provider="LM Studio"
            )
            
            # Calculate duration and extract response
            duration_ms = (time.time() - start_time) * 1000
            
            # Check for missing choices
            if not result.get("choices"):
                import json as _json
                raw_response = _json.dumps(result)[:2000]
                logger.error(f"LM Studio response missing 'choices' after {duration_ms:.0f}ms - raw: {raw_response}")
                raise ValueError(f"LM Studio response missing 'choices' after {duration_ms:.0f}ms")
            
            response_content = ""
            tokens_used = None
            if result.get("choices"):
                message = result["choices"][0].get("message", {})
                response_content = message.get("content") or message.get("reasoning") or ""
            if result.get("usage"):
                tokens_used = result["usage"].get("total_tokens")
                _pt = result["usage"].get("prompt_tokens")
                _ct = result["usage"].get("completion_tokens")
                if _pt is not None and _ct is not None:
                    token_tracker.track(model, _pt, _ct)
                    await self._broadcast("token_usage_updated", token_tracker.get_stats())

            result = self._annotate_response_with_call_metadata(
                result,
                task_id=task_id,
                role_id=role_id,
                configured_model=requested_model,
                actual_model=model,
                configured_provider=role_config.provider if role_config else configured_provider or "lm_studio",
                actual_provider="lm_studio",
                boosted=False,
                boost_mode=None,
            )
            
            # Log to autonomous API logger if callback set
            if self._autonomous_logger_callback:
                full_prompt = messages[-1].get("content", "") if messages else ""
                await self._autonomous_logger_callback(
                    task_id=task_id,
                    role_id=role_id,
                    model=model,
                    provider="lm_studio",
                    prompt=full_prompt,
                    response=response_content,
                    tokens_used=tokens_used,
                    duration_ms=duration_ms,
                    success=True,
                    error=None,
                    phase=self._current_autonomous_phase
                )
            
            # Track model usage for Tier 3
            await self._track_model_usage(model)
            
            return result
            
        except Exception as e:
            # Log LM Studio error to autonomous logger if callback set
            duration_ms = (time.time() - start_time) * 1000
            if self._autonomous_logger_callback:
                full_prompt = messages[-1].get("content", "") if messages else ""
                await self._autonomous_logger_callback(
                    task_id=task_id,
                    role_id=role_id,
                    model=model,
                    provider="lm_studio",
                    prompt=full_prompt,
                    response="",
                    tokens_used=None,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                    phase=self._current_autonomous_phase
                )
            # Re-raise the exception
            raise
    
    async def _try_free_model_rotation(
        self,
        task_id: str,
        role_id: str,
        original_model: str,
        configured_model: str,
        configured_provider: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt free model rotation chain: looping -> auto-selector.
        Returns API result on success, None if all options exhausted.
        """
        if not self._openrouter_client:
            return None

        # Step 1: Free Model Looping — iterate through available free models
        if free_model_manager.looping_enabled:
            tried_models = {original_model}
            while True:
                alt_model = free_model_manager.get_alternative_free_model(
                    original_model, skip_models=tried_models
                )
                if not alt_model or alt_model in tried_models:
                    break
                tried_models.add(alt_model)
                logger.info(f"Free model rotation: {original_model} -> {alt_model} for role {role_id}")
                await self._broadcast("free_model_rotated", {
                    "role_id": role_id,
                    "from_model": original_model,
                    "to_model": alt_model,
                    "reason": "rate_limit",
                })
                try:
                    result = await self._with_hung_connection_watchdog(
                        self._openrouter_client.generate_completion(
                            model=alt_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                            tools=tools,
                            tool_choice=tool_choice,
                        ),
                        role_id=role_id,
                        model=alt_model,
                        provider="OpenRouter (free rotation)"
                    )
                    await self._track_model_usage(alt_model)
                    if result.get("usage"):
                        _pt = result["usage"].get("prompt_tokens")
                        _ct = result["usage"].get("completion_tokens")
                        if _pt is not None and _ct is not None:
                            token_tracker.track(alt_model, _pt, _ct)
                            await self._broadcast("token_usage_updated", token_tracker.get_stats())
                    result = self._annotate_response_with_call_metadata(
                        result,
                        task_id=task_id,
                        role_id=role_id,
                        configured_model=configured_model,
                        actual_model=alt_model,
                        configured_provider=configured_provider,
                        actual_provider="openrouter",
                        boosted=False,
                        boost_mode=None,
                    )
                    if free_model_manager.is_account_exhausted():
                        free_model_manager.clear_account_exhaustion()
                    return result
                except RateLimitError:
                    free_model_manager.mark_model_failed(alt_model)
                    logger.warning(f"Rotated model {alt_model} also rate-limited, trying next")
                except CreditExhaustionError as inner_e:
                    logger.warning(f"Rotated model {alt_model} credit exhaustion: {inner_e}")
                    break

        # Step 2: Auto-Selector Backup — try openrouter/free
        if free_model_manager.auto_selector_enabled:
            auto_model = free_model_manager.AUTO_SELECTOR_MODEL
            logger.info(f"Trying auto-selector '{auto_model}' for role {role_id}")
            await self._broadcast("free_model_auto_selector_used", {
                "role_id": role_id,
                "original_model": original_model,
            })
            try:
                result = await self._with_hung_connection_watchdog(
                    self._openrouter_client.generate_completion(
                        model=auto_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        tools=tools,
                        tool_choice=tool_choice,
                    ),
                    role_id=role_id,
                    model=auto_model,
                    provider="OpenRouter (auto-selector)"
                )
                await self._track_model_usage(auto_model)
                if result.get("usage"):
                    _pt = result["usage"].get("prompt_tokens")
                    _ct = result["usage"].get("completion_tokens")
                    if _pt is not None and _ct is not None:
                        token_tracker.track(auto_model, _pt, _ct)
                        await self._broadcast("token_usage_updated", token_tracker.get_stats())
                result = self._annotate_response_with_call_metadata(
                    result,
                    task_id=task_id,
                    role_id=role_id,
                    configured_model=configured_model,
                    actual_model=auto_model,
                    configured_provider=configured_provider,
                    actual_provider="openrouter",
                    boosted=False,
                    boost_mode=None,
                )
                if free_model_manager.is_account_exhausted():
                    free_model_manager.clear_account_exhaustion()
                return result
            except (RateLimitError, CreditExhaustionError) as inner_e:
                logger.warning(f"Auto-selector '{auto_model}' also failed: {inner_e}")

        return None

    def get_fallback_state(self, role_id: str) -> str:
        """
        Get current fallback state for a role.
        
        Args:
            role_id: Role identifier
            
        Returns:
            "openrouter" or "lm_studio"
        """
        return self._role_fallback_state.get(role_id, "lm_studio")
    
    def get_all_fallback_states(self) -> Dict[str, str]:
        """
        Get fallback states for all configured roles.
        
        Returns:
            Dict mapping role_id to fallback state
        """
        return self._role_fallback_state.copy()
    
    async def reset_openrouter_fallbacks(self) -> Dict[str, str]:
        """
        Reset all roles that were originally configured for OpenRouter back to 'openrouter' state.
        Called when user adds credits and wants to retry OpenRouter without restarting.
        
        Returns:
            Dict of role_id -> new_state for roles that were reset
        """
        reset_roles = {}
        async with self._state_lock:
            for role_id, config in self._role_model_configs.items():
                if config.provider == "openrouter" and self._role_fallback_state.get(role_id) == "lm_studio":
                    self._role_fallback_state[role_id] = "openrouter"
                    reset_roles[role_id] = "openrouter"
                    logger.info(f"Reset role '{role_id}' back to OpenRouter (was fallen back to LM Studio)")
        
        if reset_roles:
            self._fallback_failed_notified.difference_update(reset_roles.keys())
            await self._broadcast("openrouter_fallbacks_reset", {
                "reset_roles": list(reset_roles.keys()),
                "message": f"Reset {len(reset_roles)} role(s) back to OpenRouter"
            })
        
        return reset_roles
    
    async def get_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings, routing to LM Studio first, then OpenRouter fallback.
        
        This enables the system to work without LM Studio if OpenRouter is configured.
        LM Studio is tried first (local, free), then falls back to OpenRouter.
        
        Args:
            texts: Texts to embed
            model: Optional model override
        
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If both LM Studio and OpenRouter are unavailable
        """
        if not texts:
            return []

        if system_config.generic_mode:
            provider_model = None if model in (None, rag_config.embedding_model) else model
            logger.debug("Generic mode enabled - using FastEmbed for embeddings")
            return await self._get_fastembed_provider(provider_model).embed(texts)
        
        # Try LM Studio first (local, free)
        try:
            return await lm_studio_client.get_embeddings(texts, model)
        except Exception as lm_error:
            logger.warning(f"LM Studio embeddings unavailable: {lm_error}")
            
            # Fall back to OpenRouter if configured
            if self._openrouter_client:
                logger.info("Falling back to OpenRouter for embeddings")
                try:
                    return await self._openrouter_client.get_embeddings(texts, model)
                except Exception as or_error:
                    logger.error(f"OpenRouter embeddings also failed: {or_error}")
                    raise RuntimeError(
                        f"Embeddings unavailable: LM Studio error ({lm_error}), "
                        f"OpenRouter error ({or_error})"
                    )
            else:
                raise RuntimeError(
                    "Embeddings unavailable: LM Studio is down and OpenRouter is not configured. "
                    "Please start LM Studio or configure OpenRouter API key."
                )
    
    async def close(self):
        """Close all API clients."""
        if self._openrouter_client:
            await self._openrouter_client.close()
        # lm_studio_client is global singleton, don't close it here


# Global singleton instance
api_client_manager = APIClientManager()

