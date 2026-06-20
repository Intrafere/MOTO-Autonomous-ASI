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
import json
import logging
import re
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
from backend.shared.openai_codex_client import OpenAICodexError, openai_codex_client
from backend.shared.xai_grok_client import XAIGrokError, xai_grok_client
from backend.shared.boost_manager import boost_manager
from backend.shared.boost_logger import boost_logger
from backend.shared.config import rag_config, system_config
from backend.shared.fastembed_provider import FASTEMBED_MODEL_NAME, FastEmbedProvider
from backend.shared.free_model_manager import free_model_manager
from backend.shared.json_parser import sanitize_model_output_for_retry_context
from backend.shared.log_redaction import redact_log_text
from backend.shared.models import ModelConfig
from backend.shared.provider_notification_store import record_provider_notification
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.proof_search.assistant_models import AssistantTargetSnapshot
from backend.shared.response_extraction import extract_response_text
from backend.shared.token_tracker import token_tracker
from backend.shared.utils import count_tokens

logger = logging.getLogger(__name__)


OAUTH_LIVE_ERROR_MAX_CHARS = 250
_TRUNCATION_SUFFIX = "..."


def _cap_oauth_live_error_text(value: Any, max_chars: int = OAUTH_LIVE_ERROR_MAX_CHARS) -> str:
    """Return a redacted one-line provider error that is at most max_chars long."""
    text = redact_log_text(value).strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return text[:max_chars]
    return text[: max_chars - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


def _extract_error_message_from_json(value: Any) -> tuple[Optional[str], Optional[str]]:
    """Extract (code, message) from common provider error JSON shapes."""
    if not isinstance(value, dict):
        return None, None

    code = value.get("code")
    message = value.get("message")
    if isinstance(message, str) and message.strip():
        return (str(code).strip() if code is not None else None), message.strip()

    for key in ("error", "response"):
        nested = value.get(key)
        if isinstance(nested, dict):
            nested_code, nested_message = _extract_error_message_from_json(nested)
            if nested_message:
                return nested_code or (str(code).strip() if code is not None else None), nested_message

    return (str(code).strip() if code is not None else None), None


def oauth_live_activity_error_message(error: Exception) -> str:
    """Best-effort visible OAuth provider error summary for live activity."""
    raw = str(error)
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        code, message = _extract_error_message_from_json(parsed)
        if message:
            detail = f"{code}: {message}" if code else message
            return _cap_oauth_live_error_text(detail)

    for prefix in (
        "OpenAI Codex completion failed:",
        "OpenAI Codex failed:",
        "xAI Grok completion failed:",
        "xAI Grok failed:",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    return _cap_oauth_live_error_text(raw)


def _response_shape_for_logging(response: Any) -> str:
    """Summarize an upstream response shape without logging provider/model text."""
    if isinstance(response, dict):
        keys = sorted(str(key) for key in response.keys())
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        return (
            f"type=dict, keys={keys}, choices_present={bool(response.get('choices'))}, "
            f"error_present={'error' in response}, usage_keys={sorted(str(key) for key in usage.keys())}"
        )
    if isinstance(response, list):
        return f"type=list, length={len(response)}"
    return f"type={type(response).__name__}"


class APIClientManager:
    """
    Central manager for routing API calls to OpenRouter or LM Studio.
    Handles fallback on credit exhaustion and boost integration.
    """
    CALL_METADATA_KEY = "_moto_call_metadata"
    # Supercharge intentionally breaks the default 0.0 temperature policy for
    # candidate attempts so parallel completions produce meaningfully different answers.
    SUPERCHARGE_ATTEMPT_TEMPERATURES = (0.0, 0.2, 0.4, 0.8)
    SUPERCHARGE_CANDIDATE_MAX_CHARS = 20000
    # Parallel brainstorm submitters use a lane-based ladder: submitter 1 stays
    # deterministic, later lanes get increasing exploration pressure.
    PARALLEL_BRAINSTORM_SUBMITTER_TEMPERATURES = (
        0.0, 0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8, 0.9,
    )
    ASSISTANT_MEMORY_MAX_CODE_CHARS = 1200
    ASSISTANT_MEMORY_MAX_TARGET_CHARS = 8000
    ASSISTANT_MEMORY_SUMMARY_CHARS = 1800
    ASSISTANT_MEMORY_SECTION_BOUNDARIES = (
        "USER PROMPT",
        "USER'S RESEARCH PROMPT",
        "USER RESEARCH PROMPT",
        "ORIGINAL USER PROMPT",
        "RESEARCH PROMPT",
        "RESEARCH GOAL",
        "USER GOAL",
        "HIGH-LEVEL RESEARCH PROMPT",
        "CURRENT BRAINSTORM TOPIC",
        "BRAINSTORM TOPIC",
        "TOPIC PROMPT",
        "CURRENT TOPIC",
        "LEANOJ PROBLEM",
        "PROBLEM",
        "YOUR TASK",
        "WRITING GOAL",
        "CURRENT PHASE",
        "PAPER TITLE",
        "THEOREM CANDIDATE",
        "TARGET THEOREM",
        "CURRENT OUTLINE",
        "OUTLINE",
        "VOLUME ORGANIZATION",
        "CURRENT DOCUMENT PROGRESS",
        "CURRENT PAPER",
        "MASTER PROOF",
        "LEAN TEMPLATE",
        "CURRENT PROOF DRAFT",
        "SOURCE CONTENT",
        "CURRENT ACCEPTED SUBMISSIONS DATABASE",
        "ACCEPTED SUBMISSIONS",
        "BRAINSTORM SUMMARY",
        "VERIFIED PROOF SUMMARIES",
        "DIRECT PROOF CONTEXT",
        "REJECTION FEEDBACK",
        "RECENT REJECTIONS",
        "FAILED ATTEMPTS",
        "LEAN ERRORS",
        "EXECUTION FEEDBACK",
    )
    
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
        
        # API logger callback. Workflows can override this to add namespace-specific
        # metadata; otherwise the manager still logs every model call by default.
        # Signature: async callback(task_id, role_id, model, provider, prompt, response,
        #                           tokens_used, duration_ms, success, error, phase)
        self._autonomous_logger_callback: Optional[Callable] = self._default_api_logger_callback
        
        # Current autonomous phase (set by autonomous coordinator)
        self._current_autonomous_phase: str = "unknown"
        
        # Track roles that have already broadcast fallback_failed (prevent GUI log spam)
        self._fallback_failed_notified: set = set()
        
        # Lock for thread-safe state updates
        self._state_lock = asyncio.Lock()

    @classmethod
    def parallel_brainstorm_submitter_temperature(cls, submitter_index: int) -> float:
        """Return the deterministic temperature lane for a parallel brainstorm submitter."""
        try:
            index = int(submitter_index)
        except (TypeError, ValueError):
            index = 1
        index = max(1, index)
        ladder_index = min(index - 1, len(cls.PARALLEL_BRAINSTORM_SUBMITTER_TEMPERATURES) - 1)
        return cls.PARALLEL_BRAINSTORM_SUBMITTER_TEMPERATURES[ladder_index]
    
    def set_broadcast_callback(self, callback: Callable) -> None:
        """Set callback for broadcasting WebSocket events."""
        self._broadcast_callback = callback
    
    async def _broadcast(self, event: str, data: Dict[str, Any] = None) -> None:
        """Broadcast an event through WebSocket."""
        if self._broadcast_callback:
            await self._broadcast_callback(event, data or {})

    async def _broadcast_unrecoverable_codex_error(
        self,
        *,
        role_id: str,
        model: str,
        error: Exception,
    ) -> None:
        """Notify the UI when a Codex role cannot recover through fallback."""
        payload = {
            "role_id": role_id,
            "model": model,
            "provider": "openai_codex_oauth",
            "provider_label": "OpenAI Codex",
            "reason": "unrecoverable_codex_error",
            "recoverable": False,
            "message": (
                "OpenAI Codex failed and no LM Studio fallback is configured. "
                "Please check your OpenAI Codex OAuth connection in OpenRouter/OAuth, "
                "sign in again, and retry."
            ),
            "error_summary": redact_log_text(str(error), 700),
            "oauth_error_message": oauth_live_activity_error_message(error),
        }
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "openai_codex_oauth_error",
            payload,
        )
        await self._broadcast("openai_codex_oauth_error", stored_payload)

    async def _broadcast_unrecoverable_xai_grok_error(
        self,
        *,
        role_id: str,
        model: str,
        error: Exception,
    ) -> None:
        """Notify the UI when a Grok OAuth role cannot recover through fallback."""
        payload = {
            "role_id": role_id,
            "model": model,
            "provider": "xai_grok_oauth",
            "provider_label": "xAI Grok",
            "reason": "unrecoverable_xai_grok_error",
            "recoverable": False,
            "message": (
                "xAI Grok failed and no LM Studio fallback is configured. "
                "Please check your xAI Grok OAuth connection in OpenRouter/OAuth, "
                "sign in again, and retry. If xAI reports subscription or credit limits, "
                "check your SuperGrok/X Premium entitlement."
            ),
            "error_summary": redact_log_text(str(error), 700),
            "oauth_error_message": oauth_live_activity_error_message(error),
        }
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "oauth_provider_error",
            payload,
        )
        await self._broadcast("oauth_provider_error", stored_payload)
    
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
                "API call for role '%s' using %s via %s has been running for %s+ minutes - possible hung connection",
                redact_log_text(role_id, 120),
                redact_log_text(model, 160),
                redact_log_text(provider, 120),
                minutes,
            )
            await self._broadcast("hung_connection_alert", {
                "role_id": role_id,
                "model": model,
                "provider": provider,
                "elapsed_minutes": minutes,
                "message": (
                    "The model may still be thinking; you can keep waiting or lower reasoning effort "
                    "in Settings if this repeats."
                )
            })

        watchdog_task = asyncio.create_task(_watchdog())
        try:
            return await coro
        finally:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task, return_exceptions=True)

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
    
    @staticmethod
    def _infer_api_log_workflow(task_id: str, role_id: str) -> str:
        """Infer the API-log namespace used by the shared log tab."""
        task = (task_id or "").strip().lower()
        role = (role_id or "").strip().lower()
        if role.startswith("leanoj_") or task.startswith("leanoj_"):
            return "leanoj"
        return "autonomous"

    @staticmethod
    def _prompt_for_logging(messages: Optional[List[Dict[str, Any]]]) -> str:
        """Return a safe prompt preview source without raw tool-result content."""
        if not messages:
            return ""

        message = messages[-1]
        role = str(message.get("role") or "")
        content = message.get("content", "")

        if role == "tool":
            tool_name = str(message.get("name") or "")
            tool_call_id = str(message.get("tool_call_id") or "")
            content_len = len(content) if isinstance(content, str) else len(str(content or ""))
            return (
                "[tool message redacted for API logging; "
                f"name={tool_name or 'unknown'}, "
                f"tool_call_id_present={bool(tool_call_id)}, "
                f"content_length={content_len}]"
            )

        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content or "")

    async def _default_api_logger_callback(
        self,
        task_id,
        role_id,
        model,
        provider,
        prompt,
        response,
        tokens_used,
        duration_ms,
        success,
        error,
        phase,
    ) -> None:
        """Persist API calls even when no workflow-specific logger is active."""
        try:
            from backend.autonomous.memory.autonomous_api_logger import autonomous_api_logger

            await autonomous_api_logger.log_api_call(
                task_id=task_id,
                role_id=role_id,
                model=model,
                provider=provider,
                prompt=prompt,
                response_content=response,
                tokens_used=tokens_used,
                duration_ms=duration_ms,
                success=success,
                error=error,
                phase=phase or self._current_autonomous_phase,
                workflow=self._infer_api_log_workflow(task_id, role_id),
            )
        except Exception as e:
            logger.error(f"Failed to log API call in default logger: {e}")

    def set_autonomous_logger_callback(self, callback: Optional[Callable]) -> None:
        """
        Set callback for autonomous API logging.
        
        The callback is called after each API call with full details for logging.
        
        Args:
            callback: Async function with signature:
                      callback(task_id, role_id, model, provider, prompt, response, 
                               tokens_used, duration_ms, success, error, phase)
                      or None to restore default all-call logging
        """
        self._autonomous_logger_callback = callback or self._default_api_logger_callback
        if callback:
            logger.info("Autonomous API logger callback set")
        else:
            logger.info("Autonomous API logger callback restored to default")
    
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
        openrouter_reasoning_effort: Optional[str] = None,
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
            "openrouter_reasoning_effort": openrouter_reasoning_effort,
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

    @staticmethod
    def _effective_max_tokens(explicit_max_tokens: Optional[int], configured_max_tokens: Optional[int], role_id: str) -> int:
        """Use the configured role budget as the ceiling for every provider call."""
        try:
            configured = int(configured_max_tokens)
        except (TypeError, ValueError):
            configured = 0
        if configured <= 0:
            raise ValueError(f"Role '{role_id}' requires a positive max output token setting.")

        if explicit_max_tokens is None:
            return configured

        try:
            explicit = int(explicit_max_tokens)
        except (TypeError, ValueError):
            explicit = 0
        if explicit <= 0:
            raise ValueError(f"Role '{role_id}' received a non-positive max output token override.")
        return min(explicit, configured)
    
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
        if int(config.context_window or 0) <= 0 or int(config.max_output_tokens or 0) <= 0:
            raise ValueError(
                f"Role '{role_id}' requires explicit positive context_window and max_output_tokens settings."
            )
        if int(config.max_output_tokens) >= int(config.context_window):
            raise ValueError(
                f"Role '{role_id}' max_output_tokens must be smaller than context_window."
            )

        if system_config.generic_mode:
            if config.provider != "openrouter":
                logger.warning(
                    "Generic mode is OpenRouter-only. Normalizing role '%s' from provider=%s to OpenRouter.",
                    role_id,
                    config.provider,
                )
                config = config.model_copy(
                    update={
                        "provider": "openrouter",
                        "openrouter_model_id": config.openrouter_model_id or config.model_id,
                        "lm_studio_fallback_id": None,
                    }
                )
            elif config.lm_studio_fallback_id:
                logger.warning(
                    "Generic mode is OpenRouter-only. Dropping LM Studio fallback for role '%s'.",
                    role_id,
                )
                config = config.model_copy(update={"lm_studio_fallback_id": None})

        self._role_model_configs[role_id] = config
        
        # Set initial fallback state based on provider
        if config.provider in {"openrouter", "openai_codex_oauth", "xai_grok_oauth"}:
            self._role_fallback_state[role_id] = config.provider
        else:
            self._role_fallback_state[role_id] = "lm_studio"
        
        # Log configuration with provider details if OpenRouter
        if config.provider == "openrouter":
            or_model = config.openrouter_model_id or config.model_id
            provider_str = f" via {config.openrouter_provider}" if config.openrouter_provider else ""
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.info(f"Configured role '{role_id}': provider=openrouter, model={or_model}{provider_str}{fallback_str}")
        elif config.provider == "openai_codex_oauth":
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.info(f"Configured role '{role_id}': provider=openai_codex_oauth, model={config.model_id}{fallback_str}")
        elif config.provider == "xai_grok_oauth":
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.info(f"Configured role '{role_id}': provider=xai_grok_oauth, model={config.model_id}{fallback_str}")
        else:
            logger.info(f"Configured role '{role_id}': provider=lm_studio, model={config.model_id}")

    def get_role_config(self, role_id: str) -> Optional[ModelConfig]:
        """Return a configured role snapshot without exposing mutable internals."""
        config = self._role_model_configs.get(role_id)
        return config.model_copy() if config is not None else None

    @classmethod
    def _assistant_memory_role_is_excluded(cls, role_id: str, task_id: str, prompt: str) -> bool:
        """Return True for roles that must never receive Assistant memory context."""
        role_key = f"{role_id} {task_id}".lower()
        prompt_key = (prompt or "").lower()
        excluded_markers = (
            "assistant",
            "validator",
            "_val",
            "validation",
            "critique",
            "paper_critic",
            "redundancy",
            "checker",
            "integrity",
            "gate",
            "novelty",
        )
        if any(marker in role_key for marker in excluded_markers):
            return True
        if "self-validation" in prompt_key or "self validation" in prompt_key:
            return True
        user_prompt_key = cls._extract_assistant_goal_hint(prompt).lower()
        if (
            "topic exploration phase" in user_prompt_key
            or "paper title exploration phase" in user_prompt_key
        ):
            return True
        if '"critique_needed"' in prompt_key or "critique_needed" in prompt_key:
            return True
        if "validate the" in prompt_key and "respond as json" in prompt_key:
            return True
        return False

    @staticmethod
    def _assistant_workflow_mode_for_role(role_id: str) -> str:
        normalized = (role_id or "").lower()
        if "manual" in normalized or "compiler_aggregator" in normalized:
            return "manual_proof_check"
        if normalized.startswith("leanoj"):
            return "leanoj"
        if normalized.startswith("compiler") or normalized.startswith("comp_"):
            return "compiler"
        if normalized.startswith("agg") or normalized.startswith("aggregator"):
            return "aggregator"
        return "autonomous"

    @staticmethod
    def _assistant_target_kind_for_role(role_id: str, task_id: str, prompt: str) -> str:
        role_key = f"{role_id} {task_id}".lower()
        prompt_key = (prompt or "").lower()
        if role_id.lower().startswith("aggregator_submitter_"):
            return "brainstorm_context"
        if "reference" in role_key:
            return "reference_selection_context"
        if "title" in role_key:
            return "title_context"
        if "topic" in role_key:
            return "topic_context"
        if "completion" in role_key:
            return "completion_review_context"
        if "certainty" in role_key or "format_selector" in role_key or "volume_organizer" in role_key:
            return "final_answer_context"
        if "path" in role_key:
            return "path_context"
        if "final_review" in role_key or "semantic" in role_key:
            return "semantic_review_context"
        if "final" in role_key:
            return "final_solver"
        if "proof" in role_key or "rigor" in role_key or "high_param" in role_key:
            return "theorem_discovery"
        if "outline" in prompt_key or "outline_complete" in prompt_key:
            return "outline_context"
        if "current document progress" in prompt_key or "construction" in role_key or "writer" in role_key:
            return "writing_context"
        return "brainstorm_context"

    @staticmethod
    def _assistant_workflow_phase_for_role(role_id: str, task_id: str, prompt: str) -> str:
        role_key = f"{role_id} {task_id}".lower()
        prompt_key = (prompt or "").lower()
        if role_id.lower().startswith("aggregator_submitter_"):
            return "brainstorm"
        if "outline" in prompt_key or "outline" in role_key:
            return "outline"
        if "construction" in role_key or "current document progress" in prompt_key:
            return "construction"
        if "review" in role_key or "red-team" in prompt_key or "red team" in prompt_key:
            return "review"
        if "rigor" in role_key or "proof" in role_key or "lemma" in role_key:
            return "proof"
        if "reference" in role_key:
            return "reference_selection"
        if "title" in role_key:
            return "title_selection"
        if "topic" in role_key:
            return "topic"
        if "completion" in role_key:
            return "completion_review"
        if "final" in role_key or "certainty" in role_key or "format_selector" in role_key or "volume" in role_key:
            return "final_answer"
        if "leanoj" in role_key:
            return "leanoj"
        return "brainstorm"

    @classmethod
    def _build_assistant_target_snapshot(cls, role_id: str, task_id: str, prompt: str) -> AssistantTargetSnapshot:
        workflow_mode = cls._assistant_workflow_mode_for_role(role_id)
        return cls._build_assistant_target_snapshot_with_overrides(
            role_id,
            task_id,
            prompt,
            workflow_mode_override=workflow_mode,
        )

    @classmethod
    def _build_assistant_target_snapshot_with_overrides(
        cls,
        role_id: str,
        task_id: str,
        prompt: str,
        *,
        workflow_mode_override: Optional[str] = None,
    ) -> AssistantTargetSnapshot:
        workflow_mode = workflow_mode_override or cls._assistant_workflow_mode_for_role(role_id)
        target_kind = cls._assistant_target_kind_for_role(role_id, task_id, prompt)
        workflow_phase = cls._assistant_workflow_phase_for_role(role_id, task_id, prompt)
        compact_prompt = cls._compact_assistant_text(prompt, cls.ASSISTANT_MEMORY_MAX_TARGET_CHARS)
        goal_hint = cls._extract_assistant_goal_hint(prompt)
        topic_hint = cls._extract_assistant_section(
            prompt,
            (
                "CURRENT BRAINSTORM TOPIC",
                "BRAINSTORM TOPIC",
                "TOPIC PROMPT",
                "CURRENT TOPIC",
                "LEANOJ PROBLEM",
                "PROBLEM",
            ),
        )
        writing_goal = cls._extract_assistant_section(
            prompt,
            (
                "YOUR TASK",
                "WRITING GOAL",
                "CURRENT PHASE",
                "PAPER TITLE",
                "THEOREM CANDIDATE",
                "TARGET THEOREM",
            ),
        )
        outline_summary = cls._extract_assistant_section(
            prompt,
            ("CURRENT OUTLINE", "OUTLINE", "VOLUME ORGANIZATION"),
        )
        draft_summary = cls._extract_assistant_section(
            prompt,
            (
                "CURRENT DOCUMENT PROGRESS",
                "CURRENT PAPER",
                "MASTER PROOF",
                "LEAN TEMPLATE",
                "CURRENT PROOF DRAFT",
                "SOURCE CONTENT",
            ),
        )
        accepted_summary = cls._extract_assistant_section(
            prompt,
            (
                "CURRENT ACCEPTED SUBMISSIONS DATABASE",
                "ACCEPTED SUBMISSIONS",
                "BRAINSTORM SUMMARY",
                "VERIFIED PROOF SUMMARIES",
                "DIRECT PROOF CONTEXT",
            ),
        )
        rejection_feedback = cls._extract_assistant_section(
            prompt,
            (
                "REJECTION FEEDBACK",
                "RECENT REJECTIONS",
                "FAILED ATTEMPTS",
                "LEAN ERRORS",
                "EXECUTION FEEDBACK",
            ),
        )
        source_titles = cls._extract_assistant_source_titles(prompt)

        target_statement = goal_hint or topic_hint or writing_goal or f"{workflow_mode}:{target_kind}"
        is_aggregator_submitter = role_id.lower().startswith("aggregator_submitter_")
        if is_aggregator_submitter:
            # All parallel submitters in one brainstorm phase share one Assistant
            # memory target. Per-lane rejection logs and task IDs are intentionally
            # excluded so the pack refreshes for the brainstorm state, not each lane.
            compact_prompt = ""
            rejection_feedback = ""
            source_title = f"{workflow_mode}:brainstorm_submitter_pack"
            source_type = f"{workflow_mode}_brainstorm_submitters"
            source_id = "shared_brainstorm_pack"
        else:
            source_title = f"{role_id} {task_id}".strip()
            source_type = role_id
            source_id = task_id
        return AssistantTargetSnapshot(
            workflow_mode=workflow_mode,
            target_kind=target_kind,
            workflow_phase=workflow_phase,
            active_mode=workflow_mode,
            user_prompt=goal_hint or compact_prompt,
            current_prompt_or_topic=topic_hint,
            current_submission_or_draft=compact_prompt,
            accepted_memory_summary=accepted_summary,
            writing_goal=writing_goal,
            outline_summary=outline_summary,
            paper_or_proof_draft_summary=draft_summary,
            recent_activity_summary=rejection_feedback,
            rejection_feedback=rejection_feedback,
            target_statement=target_statement,
            formal_sketch=compact_prompt,
            source_title=source_title,
            source_type=source_type,
            source_id=source_id,
            source_titles=source_titles,
            imports=["Mathlib"],
        )

    @classmethod
    def _compact_assistant_text(cls, value: str, max_chars: int) -> str:
        text = " ".join((value or "").split())
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @classmethod
    def _extract_assistant_goal_hint(cls, prompt: str) -> str:
        return cls._extract_assistant_section(
            prompt,
            (
                "USER PROMPT",
                "USER COMPILER-DIRECTING PROMPT",
                "USER'S RESEARCH PROMPT",
                "USER RESEARCH PROMPT",
                "ORIGINAL USER PROMPT",
                "RESEARCH PROMPT",
                "RESEARCH GOAL",
                "USER GOAL",
                "HIGH-LEVEL RESEARCH PROMPT",
            ),
        )

    @classmethod
    def _extract_assistant_section(cls, prompt: str, headings: tuple[str, ...]) -> str:
        if not prompt:
            return ""
        lines = prompt.splitlines()
        capture: list[str] = []
        found = False
        for line in lines:
            stripped = line.strip()
            if not found:
                matched, remainder = cls._assistant_heading_match(stripped, headings)
                if not matched:
                    continue
                found = True
                if remainder:
                    capture.append(remainder)
                continue
            if cls._assistant_line_is_boundary(stripped):
                break
            capture.append(line)
        if not found:
            return ""
        text = " ".join("\n".join(capture).split())
        return cls._compact_assistant_text(text, cls.ASSISTANT_MEMORY_SUMMARY_CHARS)

    @classmethod
    def _assistant_heading_match(cls, line: str, headings: tuple[str, ...]) -> tuple[bool, str]:
        normalized_line = cls._normalize_assistant_heading(line)
        for heading in headings:
            normalized_heading = cls._normalize_assistant_heading(heading)
            if normalized_line == normalized_heading:
                return True, ""
            if normalized_line.startswith(f"{normalized_heading}:"):
                return True, line.split(":", 1)[1].strip()
        return False, ""

    @classmethod
    def _assistant_line_is_boundary(cls, line: str) -> bool:
        if not line:
            return False
        if set(line) == {"-"}:
            return True
        matched, _ = cls._assistant_heading_match(line, cls.ASSISTANT_MEMORY_SECTION_BOUNDARIES)
        return matched

    @staticmethod
    def _normalize_assistant_heading(value: str) -> str:
        text = re.sub(r"^\s*#+\s*", "", value or "").strip()
        text = text.rstrip(":").strip()
        return " ".join(text.upper().split())

    @classmethod
    def _extract_assistant_source_titles(cls, prompt: str) -> list[str]:
        if not prompt:
            return []
        titles: list[str] = []
        patterns = (
            r"(?im)^\s*(?:paper|source|reference)\s+title\s*:\s*(.+)$",
            r"(?im)^\s*title\s*:\s*(.+)$",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, prompt):
                title = " ".join(match.group(1).split())[:200]
                if title and title not in titles:
                    titles.append(title)
                if len(titles) >= 8:
                    return titles
        return titles

    async def _maybe_add_assistant_memory_context(
        self,
        *,
        task_id: str,
        role_id: str,
        role_config: Optional[ModelConfig],
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        workflow_mode_override: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Append non-blocking Assistant memory to eligible non-validator calls.

        Assistant memory is optional and last-drop. Validators, critique roles,
        multi-turn tool-call protocol conversations, and retry conversations are
        intentionally left untouched. Initial single-user messages may still
        receive memory before tools are offered to the model.
        """
        if not system_config.agent_conversation_memory_enabled:
            return messages, ""
        if role_config is None:
            return messages, ""
        if len(messages) != 1 or messages[0].get("role") != "user":
            return messages, ""

        prompt = str(messages[0].get("content") or "")
        if not prompt or "ASSISTANT RETRIEVED " in prompt:
            return messages, ""
        if self._assistant_memory_role_is_excluded(role_id, task_id, prompt):
            return messages, ""

        snapshot = self._build_assistant_target_snapshot_with_overrides(
            role_id,
            task_id,
            prompt,
            workflow_mode_override=workflow_mode_override,
        )
        target_hash = assistant_proof_search_coordinator.submit_target(snapshot)
        pack = assistant_proof_search_coordinator.get_latest_pack(target_hash)
        if not pack or not pack.results:
            return messages, ""

        assistant_context = pack.to_memory_prompt_context(
            max_code_chars_per_result=self.ASSISTANT_MEMORY_MAX_CODE_CHARS,
        )
        augmented_prompt = self._append_assistant_memory_block(prompt, assistant_context)
        if not self._prompt_fits_role_budget(
            augmented_prompt,
            role_config=role_config,
            explicit_max_tokens=max_tokens,
            role_id=role_id,
        ):
            metadata_only_context = pack.to_memory_prompt_context(max_code_chars_per_result=0)
            augmented_prompt = self._append_assistant_memory_block(prompt, metadata_only_context)
            if not self._prompt_fits_role_budget(
                augmented_prompt,
                role_config=role_config,
                explicit_max_tokens=max_tokens,
                role_id=role_id,
            ):
                return messages, ""

        return [{**messages[0], "content": augmented_prompt}], target_hash

    async def prewarm_assistant_memory_context(
        self,
        *,
        task_id: str,
        role_id: str,
        prompt: str,
        workflow_mode_override: Optional[str] = None,
    ) -> str:
        """Schedule Assistant memory for an eligible prompt before model-call preflight.

        Many workflows validate mandatory prompt size before calling
        `generate_completion()`. This helper gives those producer paths the same
        non-blocking Assistant lifecycle as normal completions, even if the
        prompt later overflows and no model call is made.
        """
        if not system_config.agent_conversation_memory_enabled:
            return ""
        async with self._state_lock:
            role_config = self._role_model_configs.get(role_id)
        if role_config is None:
            return ""
        prompt = str(prompt or "")
        if not prompt or "ASSISTANT RETRIEVED " in prompt:
            return ""
        if self._assistant_memory_role_is_excluded(role_id, task_id, prompt):
            return ""
        snapshot = self._build_assistant_target_snapshot_with_overrides(
            role_id,
            task_id,
            prompt,
            workflow_mode_override=workflow_mode_override,
        )
        target_hash = assistant_proof_search_coordinator.submit_target(snapshot)
        return target_hash

    @staticmethod
    def _append_assistant_memory_block(prompt: str, assistant_context: str) -> str:
        return (
            f"{prompt}\n\n---\n\n"
            "OPTIONAL ASSISTANT MEMORY CONTEXT:\n"
            f"{assistant_context}\n\n"
            "Use the Assistant memory only when it is relevant. It is supporting context, "
            "not validator feedback, not a requirement to cite, and not a replacement for the user prompt."
        )

    def _prompt_fits_role_budget(
        self,
        prompt: str,
        *,
        role_config: ModelConfig,
        explicit_max_tokens: Optional[int],
        role_id: str,
    ) -> bool:
        try:
            effective_max_tokens = self._effective_max_tokens(
                explicit_max_tokens,
                role_config.max_output_tokens,
                role_id,
            )
            max_input_tokens = rag_config.get_available_input_tokens(
                role_config.context_window,
                effective_max_tokens,
            )
        except Exception:
            return False
        return count_tokens(prompt) <= max_input_tokens
    
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
        """Generate a completion, optionally wrapping the role with Supercharge."""
        disable_supercharge = bool(kwargs.pop("_moto_disable_supercharge", False))
        assistant_workflow_mode_override = kwargs.pop("_moto_assistant_workflow_mode", None)
        async with self._state_lock:
            role_config = self._role_model_configs.get(role_id)

        messages, assistant_memory_target_hash = await self._maybe_add_assistant_memory_context(
            task_id=task_id,
            role_id=role_id,
            role_config=role_config,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            workflow_mode_override=assistant_workflow_mode_override,
        )

        supercharge_enabled = bool(getattr(role_config, "supercharge_enabled", False)) and not disable_supercharge
        # Tool-call conversations need exact assistant/tool turn pairing, so keep them single-shot.
        if not supercharge_enabled or tools or tool_choice is not None:
            response = await self._generate_completion_once(
                task_id=task_id,
                role_id=role_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs
            )
        else:
            response = await self._generate_supercharged_completion(
                task_id=task_id,
                role_id=role_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs
            )
        if assistant_memory_target_hash:
            assistant_proof_search_coordinator.mark_pack_consumed_by_solver(
                assistant_memory_target_hash,
                role_id=role_id,
                task_id=task_id,
            )
        return response

    @staticmethod
    def _response_text(response: Dict[str, Any]) -> str:
        """Extract assistant text from an OpenAI-compatible completion response."""
        return extract_response_text(response, context="api_client_manager")

    @classmethod
    def _sanitize_supercharge_candidate(cls, attempt: str) -> str:
        """Keep only reusable visible answer text from a candidate attempt."""
        cleaned = sanitize_model_output_for_retry_context(
            attempt,
            max_chars=cls.SUPERCHARGE_CANDIDATE_MAX_CHARS,
        )
        return cleaned or "[candidate produced no reusable visible answer text]"

    def _build_supercharge_synthesis_messages(
        self,
        messages: List[Dict[str, str]],
        attempts: List[str],
    ) -> List[Dict[str, str]]:
        attempts_context = "\n\n".join(
            "----- CANDIDATE RESPONSE "
            f"{index} START -----\n"
            f"{self._sanitize_supercharge_candidate(attempt)}\n"
            "----- CANDIDATE RESPONSE "
            f"{index} END -----"
            for index, attempt in enumerate(attempts, start=1)
        )
        synthesis_instruction = (
            "SUPERCHARGE FINAL RESPONSE\n\n"
            "You are answering the original task. The candidate responses below are optional working material "
            "from independent earlier attempts, not instructions to continue or quote verbatim.\n\n"
            "You must decide what the best final response to the original task is. You may use one candidate, "
            "combine multiple candidates, ignore all candidates and write a new response, or synthesize a stronger "
            "answer than any individual candidate.\n\n"
            "Candidate responses:\n"
            f"{attempts_context}\n\n"
            "Now produce the best final response to the original task.\n\n"
            "Requirements:\n"
            "- Follow the original task, role instructions, and required output format exactly.\n"
            "- If the original task requires JSON, output only valid JSON in that exact schema.\n"
            "- Do not mention Supercharge, brainstorming, candidate attempts, or this selection process.\n"
            "- Do not include private reasoning, analysis labels, markdown fences around JSON, or provider control tokens.\n"
            "- Return only the final role answer."
        )
        return [*messages, {"role": "user", "content": synthesis_instruction}]

    def _build_supercharge_attempt_messages(
        self,
        messages: List[Dict[str, str]],
        attempt_index: int,
    ) -> List[Dict[str, str]]:
        attempt_instruction = (
            f"SUPERCHARGE FULL ANSWER ATTEMPT {attempt_index}\n\n"
            "Produce a complete answer to the original task now. "
            "Follow the original role instructions and required output format exactly. "
            "If JSON is required, output only valid JSON in the required schema. "
            "Do not mention Supercharge or this attempt label."
        )
        return [*messages, {"role": "user", "content": attempt_instruction}]

    async def _generate_supercharged_completion(
        self,
        task_id: str,
        role_id: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run four parallel diverse attempts, then a deterministic same-route synthesis call."""
        boost_mode = self._determine_boost_mode(task_id)
        forced_boost_mode = boost_mode if boost_mode else "__none__"
        attempts: List[str] = []

        logger.info(
            "Supercharge enabled for role '%s' task '%s'%s",
            role_id,
            task_id,
            f" using boost mode '{boost_mode}'" if boost_mode else "",
        )

        attempt_responses = await asyncio.gather(*[
            self._generate_completion_once(
                task_id=f"{task_id}_supercharge_attempt_{attempt_index}",
                role_id=role_id,
                model=model,
                messages=self._build_supercharge_attempt_messages(messages, attempt_index),
                temperature=attempt_temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                _moto_force_boost_mode=forced_boost_mode,
                _moto_consume_boost_count=False,
                _moto_strict_boost=bool(boost_mode),
                **kwargs
            )
            for attempt_index, attempt_temperature in enumerate(
                self.SUPERCHARGE_ATTEMPT_TEMPERATURES,
                start=1,
            )
        ])
        attempts = [self._response_text(response) for response in attempt_responses]

        synthesis_response = await self._generate_completion_once(
            task_id=f"{task_id}_supercharge_final",
            role_id=role_id,
            model=model,
            messages=self._build_supercharge_synthesis_messages(messages, attempts),
            temperature=0.0,
            max_tokens=max_tokens,
            response_format=response_format,
            _moto_force_boost_mode=forced_boost_mode,
            _moto_consume_boost_count=False,
            _moto_strict_boost=bool(boost_mode),
            **kwargs
        )

        metadata = self.extract_call_metadata(synthesis_response)
        if boost_mode == "next_count" and metadata.get("boosted"):
            await boost_manager.consume_boost_count()

        if isinstance(synthesis_response, dict):
            synthesis_response[self.CALL_METADATA_KEY] = {
                **metadata,
                "supercharged": True,
                "supercharge_attempts": 4,
                "supercharge_attempt_temperatures": list(self.SUPERCHARGE_ATTEMPT_TEMPERATURES),
            }
        return synthesis_response

    async def _generate_completion_once(
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
        forced_boost_mode = kwargs.pop("_moto_force_boost_mode", None)
        consume_boost_count = kwargs.pop("_moto_consume_boost_count", True)
        strict_boost = kwargs.pop("_moto_strict_boost", False)
        reasoning_effort_override = kwargs.pop("_moto_reasoning_effort_override", None)
        requested_model = model
        async with self._state_lock:
            initial_role_config = self._role_model_configs.get(role_id)
        configured_provider = initial_role_config.provider if initial_role_config else None
        role_reasoning_effort = (
            reasoning_effort_override
            if reasoning_effort_override is not None
            else (initial_role_config.openrouter_reasoning_effort if initial_role_config else None)
        )

        # Check if task should use boost (unified check for all boost modes)
        if forced_boost_mode == "__none__":
            boost_mode = None
        elif forced_boost_mode is not None:
            boost_mode = forced_boost_mode
        else:
            boost_mode = self._determine_boost_mode(task_id)
        
        if boost_mode and boost_manager.boost_config:
            boost_model = boost_manager.boost_config.boost_model_id
            boost_provider = boost_manager.boost_config.boost_provider
            provider_info = f" via {boost_provider}" if boost_provider else " (auto-routing)"
            logger.info(f"Task {task_id} using boost ({boost_mode}): {boost_model}{provider_info}")
            
            # Get prompt preview for logging
            prompt_preview = ""
            if messages:
                last_message = self._prompt_for_logging(messages)
                prompt_preview = last_message or ""
            
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
                            max_tokens=self._effective_max_tokens(
                                max_tokens,
                                boost_manager.boost_config.boost_max_output_tokens,
                                role_id,
                            ),
                            response_format=response_format,
                            provider=boost_provider,
                            reasoning_effort=(
                                reasoning_effort_override
                                if reasoning_effort_override is not None
                                else boost_manager.boost_config.boost_reasoning_effort
                            ),
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
                        logger.error(
                            "OpenRouter boost response missing 'choices' after %.0fms - %s",
                            duration_ms,
                            _response_shape_for_logging(result),
                        )
                        
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
                        response_content = extract_response_text(result, context=task_id)
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
                        openrouter_reasoning_effort=(
                            reasoning_effort_override
                            if reasoning_effort_override is not None
                            else boost_manager.boost_config.boost_reasoning_effort
                        ),
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
                        full_prompt = self._prompt_for_logging(messages)
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
                    if boost_mode == "next_count" and consume_boost_count:
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
                    full_prompt = self._prompt_for_logging(messages)
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
                if strict_boost:
                    raise RuntimeError(f"Strict boost call failed for task {task_id}: {e}") from e
            
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
                    full_prompt = self._prompt_for_logging(messages)
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
                    "message": "Model requires privacy policy acceptance",
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
                    full_prompt = self._prompt_for_logging(messages)
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
                    "message": "Boost credits exhausted, falling back to primary model"
                })
                if strict_boost:
                    raise RuntimeError(f"Strict boost call credits exhausted for task {task_id}: {e}") from e
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
                    full_prompt = self._prompt_for_logging(messages)
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
                if strict_boost:
                    raise RuntimeError(f"Strict boost call failed for task {task_id}: {e}") from e
                # Fall through to primary model
        
        # Check role fallback state
        async with self._state_lock:
            fallback_state = self._role_fallback_state.get(role_id, "lm_studio")
            role_config = self._role_model_configs.get(role_id)

            if system_config.generic_mode and role_config and fallback_state != "openrouter":
                logger.warning(
                    "Generic mode reset role '%s' fallback state from %s to OpenRouter.",
                    role_id,
                    fallback_state,
                )
                fallback_state = "openrouter"
                self._role_fallback_state[role_id] = "openrouter"
        
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
                            max_tokens=self._effective_max_tokens(max_tokens, role_config.max_output_tokens, role_id),
                            response_format=response_format,
                            provider=openrouter_provider,
                            reasoning_effort=role_reasoning_effort,
                            tools=tools,
                            tool_choice=tool_choice,
                            allow_provider_auto_fallback=role_id.endswith("_assistant"),
                        ),
                        role_id=role_id,
                        model=openrouter_model,
                        provider=openrouter_provider or "OpenRouter"
                    )
                    
                    # Calculate duration and extract response
                    duration_ms = (time.time() - start_time) * 1000
                    provider_auto_fallback = None
                    if isinstance(result, dict):
                        provider_auto_fallback = result.pop("_moto_openrouter_provider_auto_fallback", None)
                    if provider_auto_fallback and openrouter_provider:
                        logger.warning(
                            "Clearing unavailable OpenRouter host provider '%s' for Assistant role '%s'; future calls will use Auto routing.",
                            redact_log_text(openrouter_provider, 120),
                            role_id,
                        )
                        role_config.openrouter_provider = None
                        openrouter_provider = None
                    
                    # Check for missing choices (upstream provider timeout/error)
                    if not result.get("choices"):
                        logger.error(
                            "OpenRouter response missing 'choices' after %.0fms - %s",
                            duration_ms,
                            _response_shape_for_logging(result),
                        )
                        raise ValueError(f"OpenRouter response missing 'choices' after {duration_ms:.0f}ms (upstream provider timeout)")
                    
                    response_content = ""
                    tokens_used = None
                    if result.get("choices"):
                        response_content = extract_response_text(result, context=task_id)
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
                        openrouter_reasoning_effort=role_reasoning_effort,
                    )
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = self._prompt_for_logging(messages)
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
                        full_prompt = self._prompt_for_logging(messages)
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
                        max_tokens=self._effective_max_tokens(max_tokens, role_config.max_output_tokens, role_id),
                        response_format=response_format,
                        reasoning_effort=role_reasoning_effort,
                        tools=tools,
                        tool_choice=tool_choice,
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
                        full_prompt = self._prompt_for_logging(messages)
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
                        "message": "Model requires privacy policy acceptance",
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
                        full_prompt = self._prompt_for_logging(messages)
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
                        "message": "Credits exhausted, falling back to alternative model",
                        "fallback_model": fallback_model
                    })
                    
                    # Fall through to LM Studio
                    model = fallback_model
                
                except Exception as e:
                    # Other OpenRouter error - fall back for this call only (don't mark as permanent)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log to autonomous API logger if callback set
                    if self._autonomous_logger_callback:
                        full_prompt = self._prompt_for_logging(messages)
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
        
        if fallback_state == "openai_codex_oauth" and role_config:
            codex_model = role_config.model_id
            start_time = time.time()
            try:
                logger.debug("Role %s using OpenAI Codex OAuth: %s", role_id, codex_model)
                result = await self._with_hung_connection_watchdog(
                    openai_codex_client.generate_completion(
                        model=codex_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self._effective_max_tokens(max_tokens, role_config.max_output_tokens, role_id),
                        response_format=response_format,
                        reasoning_effort=role_reasoning_effort,
                        tools=tools,
                        tool_choice=tool_choice,
                    ),
                    role_id=role_id,
                    model=codex_model,
                    provider="OpenAI Codex",
                )
                duration_ms = (time.time() - start_time) * 1000
                if not result.get("choices"):
                    logger.error(
                        "OpenAI Codex response missing 'choices' after %.0fms - %s",
                        duration_ms,
                        _response_shape_for_logging(result),
                    )
                    raise ValueError(f"OpenAI Codex response missing 'choices' after {duration_ms:.0f}ms")

                response_content = ""
                tokens_used = None
                if result.get("choices"):
                    response_content = extract_response_text(result, context=task_id)
                if result.get("usage"):
                    tokens_used = result["usage"].get("total_tokens")
                    _pt = result["usage"].get("prompt_tokens")
                    _ct = result["usage"].get("completion_tokens")
                    if _pt is not None and _ct is not None:
                        token_tracker.track(codex_model, _pt, _ct)
                        await self._broadcast("token_usage_updated", token_tracker.get_stats())

                result = self._annotate_response_with_call_metadata(
                    result,
                    task_id=task_id,
                    role_id=role_id,
                    configured_model=requested_model,
                    actual_model=codex_model,
                    configured_provider=role_config.provider,
                    actual_provider="openai_codex_oauth",
                    boosted=False,
                    boost_mode=None,
                    openrouter_reasoning_effort=role_reasoning_effort,
                )

                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=codex_model,
                        provider="openai_codex_oauth",
                        prompt=full_prompt,
                        response=response_content,
                        tokens_used=tokens_used,
                        duration_ms=duration_ms,
                        success=True,
                        error=None,
                        phase=self._current_autonomous_phase,
                    )

                await self._track_model_usage(codex_model)
                return result

            except OpenAICodexError as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=codex_model,
                        provider="openai_codex_oauth",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase,
                    )
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "OpenAI Codex failed for role '%s'; falling back to LM Studio model %s",
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_unrecoverable_codex_error(
                        role_id=role_id,
                        model=codex_model,
                        error=e,
                    )
                    raise RuntimeError(
                        f"OpenAI Codex failed for role '{role_id}' and no LM Studio fallback is configured: {e}"
                    ) from e
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=codex_model,
                        provider="openai_codex_oauth",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase,
                    )
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "OpenAI Codex error for role '%s': %s; falling back to LM Studio model %s",
                        role_id,
                        e,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_unrecoverable_codex_error(
                        role_id=role_id,
                        model=codex_model,
                        error=e,
                    )
                    raise

        if fallback_state == "xai_grok_oauth" and role_config:
            xai_model = role_config.model_id
            start_time = time.time()
            try:
                logger.debug("Role %s using xAI Grok OAuth: %s", role_id, xai_model)
                result = await self._with_hung_connection_watchdog(
                    xai_grok_client.generate_completion(
                        model=xai_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self._effective_max_tokens(max_tokens, role_config.max_output_tokens, role_id),
                        response_format=response_format,
                        reasoning_effort=role_reasoning_effort,
                        tools=tools,
                        tool_choice=tool_choice,
                    ),
                    role_id=role_id,
                    model=xai_model,
                    provider="xAI Grok",
                )
                duration_ms = (time.time() - start_time) * 1000
                if not result.get("choices"):
                    logger.error(
                        "xAI Grok response missing 'choices' after %.0fms - %s",
                        duration_ms,
                        _response_shape_for_logging(result),
                    )
                    raise ValueError(f"xAI Grok response missing 'choices' after {duration_ms:.0f}ms")

                response_content = ""
                tokens_used = None
                if result.get("choices"):
                    response_content = extract_response_text(result, context=task_id)
                if result.get("usage"):
                    tokens_used = result["usage"].get("total_tokens")
                    _pt = result["usage"].get("prompt_tokens")
                    _ct = result["usage"].get("completion_tokens")
                    if _pt is not None and _ct is not None:
                        token_tracker.track(xai_model, _pt, _ct)
                        await self._broadcast("token_usage_updated", token_tracker.get_stats())

                result = self._annotate_response_with_call_metadata(
                    result,
                    task_id=task_id,
                    role_id=role_id,
                    configured_model=requested_model,
                    actual_model=xai_model,
                    configured_provider=role_config.provider,
                    actual_provider="xai_grok_oauth",
                    boosted=False,
                    boost_mode=None,
                    openrouter_reasoning_effort=role_reasoning_effort,
                )

                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=xai_model,
                        provider="xai_grok_oauth",
                        prompt=full_prompt,
                        response=response_content,
                        tokens_used=tokens_used,
                        duration_ms=duration_ms,
                        success=True,
                        error=None,
                        phase=self._current_autonomous_phase,
                    )

                await self._track_model_usage(xai_model)
                return result

            except XAIGrokError as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=xai_model,
                        provider="xai_grok_oauth",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase,
                    )
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "xAI Grok failed for role '%s'; falling back to LM Studio model %s",
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_unrecoverable_xai_grok_error(
                        role_id=role_id,
                        model=xai_model,
                        error=e,
                    )
                    raise RuntimeError(
                        f"xAI Grok failed for role '{role_id}' and no LM Studio fallback is configured: {e}"
                    ) from e
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=xai_model,
                        provider="xai_grok_oauth",
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        phase=self._current_autonomous_phase,
                    )
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "xAI Grok error for role '%s': %s; falling back to LM Studio model %s",
                        role_id,
                        e,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_unrecoverable_xai_grok_error(
                        role_id=role_id,
                        model=xai_model,
                        error=e,
                    )
                    raise

        if system_config.generic_mode:
            raise RuntimeError(
                f"Generic mode is OpenRouter-only; role '{role_id}' cannot use LM Studio. "
                "Configure the role with provider='openrouter' and a valid OpenRouter model/key."
            )

        # Use LM Studio (either configured as primary or fallen back)
        logger.debug(
            "Role %s using LM Studio: %s",
            redact_log_text(role_id, 120),
            redact_log_text(model, 160),
        )
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
                logger.error(
                    "LM Studio response missing 'choices' after %.0fms - %s",
                    duration_ms,
                    _response_shape_for_logging(result),
                )
                raise ValueError(f"LM Studio response missing 'choices' after {duration_ms:.0f}ms")
            
            response_content = ""
            tokens_used = None
            lm_routing_metadata = lm_studio_client.extract_routing_metadata(result)
            actual_lm_studio_model = lm_routing_metadata.get("actual_model") or model
            if result.get("choices"):
                response_content = extract_response_text(result, context=task_id)
            if result.get("usage"):
                tokens_used = result["usage"].get("total_tokens")
                _pt = result["usage"].get("prompt_tokens")
                _ct = result["usage"].get("completion_tokens")
                if _pt is not None and _ct is not None:
                    token_tracker.track(actual_lm_studio_model, _pt, _ct)
                    await self._broadcast("token_usage_updated", token_tracker.get_stats())

            result = self._annotate_response_with_call_metadata(
                result,
                task_id=task_id,
                role_id=role_id,
                configured_model=requested_model,
                actual_model=actual_lm_studio_model,
                configured_provider=role_config.provider if role_config else configured_provider or "lm_studio",
                actual_provider="lm_studio",
                boosted=False,
                boost_mode=None,
            )
            
            # Log to autonomous API logger if callback set
            if self._autonomous_logger_callback:
                full_prompt = self._prompt_for_logging(messages)
                await self._autonomous_logger_callback(
                    task_id=task_id,
                    role_id=role_id,
                    model=actual_lm_studio_model,
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
            await self._track_model_usage(actual_lm_studio_model)
            
            return result
            
        except Exception as e:
            # Log LM Studio error to autonomous logger if callback set
            duration_ms = (time.time() - start_time) * 1000
            if self._autonomous_logger_callback:
                full_prompt = self._prompt_for_logging(messages)
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
        reasoning_effort: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
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
                            reasoning_effort=reasoning_effort,
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
                        openrouter_reasoning_effort=reasoning_effort,
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
                        reasoning_effort=reasoning_effort,
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
                    openrouter_reasoning_effort=reasoning_effort,
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

