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
import uuid
from typing import Dict, Any, Awaitable, List, Optional, Callable

import httpx

from backend.shared.lm_studio_client import lm_studio_client
from backend.shared.openrouter_client import (
    OpenRouterClient, 
    CreditExhaustionError,
    OpenRouterPrivacyPolicyError,
    RateLimitError,
    FreeModelExhaustedError,
    OpenRouterInvalidResponseError,
    OpenRouterNoEndpointsError,
    sanitize_provider_error_text,
)
from backend.shared.openai_codex_client import (
    OpenAICodexAuthError,
    OpenAICodexError,
    OpenAICodexRequestError,
    OpenAICodexUntimedUsageLimitError,
    OAuthUsageLimitError,
    openai_codex_client,
)
from backend.shared.subscription_providers.errors import (
    SubscriptionEntitlementError,
    SubscriptionProviderError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.metadata import CLOUD_PROVIDER_IDS, CLOUD_PROVIDERS
from backend.shared.subscription_providers.registry import get_api_key_subscription_client
from backend.shared.xai_grok_client import (
    XAIGrokError,
    XAIGrokSpendingLimitError,
    XAIGrokUsageLimitError,
    xai_grok_client,
)
from backend.shared.boost_manager import boost_manager
from backend.shared.boost_logger import boost_logger
from backend.shared.config import rag_config, system_config
from backend.shared.fastembed_provider import FASTEMBED_MODEL_NAME, FastEmbedProvider
from backend.shared.free_model_manager import free_model_manager
from backend.shared.json_parser import sanitize_model_output_for_retry_context
from backend.shared.log_redaction import redact_log_text
from backend.shared.models import ModelConfig
from backend.shared.model_error_utils import (
    is_provider_context_length_error,
    is_retryable_model_output_error,
    is_transient_model_call_error,
)
from backend.shared.provider_notification_store import record_provider_notification
from backend.shared.provider_recovery.events import canonical_recovery_event
from backend.shared.provider_recovery.runtime import provider_recovery_runtime
from backend.shared.provider_errors import (
    ProviderContextLengthError,
    ProviderRepairRequiredError,
    ProviderRouteError,
    ProviderRouteIdentity,
)
from backend.shared.provider_route_identity import (
    ROUTE_IDENTITY_FIELD_NAMES,
    VALID_WORKFLOW_MODES,
    is_model_provider,
    is_placeholder_provider,
    merge_route_identity,
    provider_label as provider_display_label,
    route_identity_fields,
    route_identity_from_error,
    with_route_identity,
)
from backend.shared.proof_search.assistant_coordinator import assistant_proof_search_coordinator
from backend.shared.proof_search.assistant_models import AssistantTargetSnapshot
from backend.shared.response_extraction import drop_unfinished_thinking, extract_response_text
from backend.shared.token_tracker import token_tracker
from backend.shared.utils import count_tokens
from backend.shared.workflow_start_guard import workflow_start_guard

logger = logging.getLogger(__name__)


OAUTH_LIVE_ERROR_MAX_CHARS = 1800
_TRUNCATION_SUFFIX = "..."
_HARD_CODEX_REQUEST_ERROR_MARKERS = (
    "authentication failed",
    "authorization failed",
    "entitlement",
    "forbidden",
    "invalid model",
    "invalid_model",
    "model is not available",
    "model not found",
    "no access to model",
    "not entitled",
    "permission denied",
    "subscription is required",
    "unsupported model",
)
_HARD_CODEX_ERROR_CODES = {
    "authentication_error",
    "authorization_error",
    "context_length_exceeded",
    "entitlement_required",
    "invalid_api_key",
    "invalid_model",
    "invalid_request_error",
    "model_not_found",
    "not_entitled",
    "permission_denied",
    "subscription_required",
    "unsupported_model",
}
_RETRYABLE_CODEX_FAILURE_KINDS = {
    "empty_response",
    "empty_stream",
    "invalid_terminal_state",
    "terminal_missing",
    "stream_rejected",
    "transient_http_exhausted",
    "transient_stream_exhausted",
    "transport_exhausted",
}
_RETRYABLE_CODEX_HTTP_STATUSES = {
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
    520,
    521,
    522,
    523,
    524,
}


def _is_retryable_codex_completion_error(error: Exception) -> bool:
    """Retry ambiguous Codex completion rejections unless they are definitively hard."""
    if isinstance(error, OpenAICodexAuthError):
        return False
    if not isinstance(error, OpenAICodexRequestError):
        return False
    if is_provider_context_length_error(error):
        return False
    error_code = str(getattr(error, "error_code", "") or "").strip().lower()
    if error_code in _HARD_CODEX_ERROR_CODES:
        return False
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        if int(status_code) in _RETRYABLE_CODEX_HTTP_STATUSES:
            return True
        if 400 <= int(status_code) < 500:
            return False
    failure_kind = str(getattr(error, "failure_kind", "") or "").strip().lower()
    if failure_kind in _RETRYABLE_CODEX_FAILURE_KINDS:
        return True
    message = str(error or "").lower()
    return not any(marker in message for marker in _HARD_CODEX_REQUEST_ERROR_MARKERS)


def _active_notification_workflow_mode() -> str:
    return {
        "manual_aggregator": "aggregator",
        "manual_compiler": "compiler",
        "manual_compiler_proof_only": "manual_proof_check",
        "autonomous": "autonomous",
        "leanoj": "leanoj",
    }.get(str(workflow_start_guard.active_owner or ""), "")


def _manual_proof_role_workflow_mode(role: str) -> str:
    """Return the workflow a manual proof-check role belongs to, or ``""``.

    Manual proof checks run beside whichever workflow owns the session, so their
    roles name their own workflow: the ``manual_proof_*`` pseudo-roles and the
    stage roles built with a ``manual_{source_type}`` suffix. The compiler
    saved-paper check (``..._compiler_manual_paper``) is labelled compiler, as
    ``ProofVerificationStage`` labels its ``manual_compiler_save`` trigger.
    """
    if role.startswith("manual_proof_"):
        return "manual_proof_check"
    if role.startswith("autonomous_proof_") and "_manual_" in role:
        return "compiler" if "_compiler_manual_paper" in role else "manual_proof_check"
    return ""


def _retry_workflow_mode(role_id: str) -> str:
    role = str(role_id or "").lower()
    manual_proof_mode = _manual_proof_role_workflow_mode(role)
    if manual_proof_mode:
        return manual_proof_mode
    active_mode = _active_notification_workflow_mode()
    if active_mode:
        return active_mode
    if "leanoj" in role:
        return "leanoj"
    if role.startswith(("comp_", "compiler_")) or "manual_compiler" in role:
        return "compiler"
    if role.startswith(("agg_", "aggregator_")) or "manual_aggregator" in role:
        return "aggregator"
    if "autonomous" in role:
        return "autonomous"
    return ""


# Raw route inputs accepted by route_identity_fields(); composed fields such as
# role_label, host_routing and route_identity are derived, never copied in.
_ROUTE_IDENTITY_INPUT_KEYS = (
    "model",
    "provider",
    "provider_label_text",
    "route_kind",
    "task_id",
    "configured_model",
    "configured_provider",
    "configured_host_provider",
    "upstream_host_provider",
    "auto_fallback_from_host",
    "provider_request_id",
)
# Identity fields that describe one role's call. A provider-wide cooldown payload
# reused for another role drops them before that role's identity is merged.
_ROLE_SCOPED_ROUTE_IDENTITY_KEYS = frozenset(ROUTE_IDENTITY_FIELD_NAMES) - {
    "workflow_mode",
    "role_id",
    "model",
    "provider",
    "provider_label",
}
# Per-event identity of the role that first hit a provider-wide cooldown; a
# reused cooldown payload rebuilds these for the role it is reported for.
_COOLDOWN_EVENT_IDENTITY_KEYS = frozenset({
    "event_id",
    "correlation_id",
    "activity_correlation_id",
    "activity_id",
    "notification_key",
    "id",
    "workflow_mode",
})


def _merge_route_fields(payload: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing route identity keys without overriding producer-supplied values.

    A placeholder provider yields to a real one together with its label, and
    ``configured_host_provider`` keeps ``""`` (OpenRouter auto routing) distinct
    from ``None`` (unknown host), so only an absent or ``None`` host is filled.
    """
    merged = merge_route_identity(
        payload,
        {
            key: value
            for key, value in fields.items()
            if key not in ("configured_host_provider", "provider_label")
        },
    )
    provider = fields.get("provider")
    label = fields.get("provider_label")
    if provider and merged.get("provider") == provider:
        if merged.get("provider") != payload.get("provider") or not merged.get("provider_label"):
            merged["provider_label"] = label or provider_display_label(provider)
    elif not provider and label and not merged.get("provider_label"):
        merged["provider_label"] = label
    host = fields.get("configured_host_provider")
    if host is not None and merged.get("configured_host_provider") is None:
        merged["configured_host_provider"] = host
    return merged


def _configured_role_model(config: ModelConfig) -> str:
    """Return the model id a role is configured to call on its own provider."""
    if config.provider == "openrouter":
        return str(config.openrouter_model_id or config.model_id or "")
    return str(config.model_id or "")


def _role_route_model(config: ModelConfig, provider: str) -> str:
    """Return the model a role calls on ``provider``: its own or its LM Studio fallback."""
    if provider == config.provider:
        return _configured_role_model(config)
    if provider == "lm_studio":
        return str(config.lm_studio_fallback_id or "")
    return ""


def _route_log_host(fields: Dict[str, Any]) -> str:
    """Describe the host of a route for plain-text logs."""
    routing = str(fields.get("host_routing") or "")
    upstream = str(fields.get("effective_host_provider") or "")
    if routing == "pinned":
        return str(fields.get("configured_host_provider") or upstream or "pinned")
    if routing in ("auto", "auto_fallback"):
        return f"{routing} -> {upstream}" if upstream else routing
    return routing or "n/a"


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


_PROVIDER_ERROR_MESSAGE_PREFIXES = tuple(
    prefix
    for provider in CLOUD_PROVIDERS
    for prefix in (f"{provider.label} completion failed:", f"{provider.label} failed:")
)


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

    for prefix in _PROVIDER_ERROR_MESSAGE_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    return _cap_oauth_live_error_text(raw)


def _is_provider_context_length_error(error: Exception) -> bool:
    """Return true when the provider rejected the request as too large."""
    detail = oauth_live_activity_error_message(error).lower()
    return is_provider_context_length_error(error) or is_provider_context_length_error(Exception(detail))


def _typed_provider_context_error(
    error: Exception,
    *,
    provider: str,
    model: str,
    route_kind: str = "primary",
) -> ProviderContextLengthError:
    """Create a redacted typed context rejection without replaying provider text."""
    if isinstance(error, ProviderContextLengthError):
        return error.with_route_context(
            role_id=error.route.role_id,
            task_id=error.route.task_id,
            route_kind=route_kind,
        )
    return ProviderContextLengthError(
        f"{provider} rejected the request because its input exceeded the provider context limit.",
        route=ProviderRouteIdentity(
            provider=provider,
            model=model,
            route_kind=route_kind,
        ),
        cause=error,
    )


class RetryableProviderError(RuntimeError):
    """Raised when a provider failure should be retried by the workflow."""

    def __init__(
        self,
        *,
        provider: str,
        provider_label: str,
        role_id: str,
        model: str,
        reason: str,
        message: str,
        retry_after_seconds: Optional[int] = None,
        error_summary: str = "",
        route_kind: str = "",
        task_id: str = "",
        configured_provider: str = "",
        configured_model: str = "",
        configured_host_provider: Optional[str] = None,
        upstream_host_provider: str = "",
        auto_fallback_from_host: str = "",
        provider_request_id: str = "",
    ) -> None:
        self.provider = provider
        self.provider_label = provider_label
        self.role_id = role_id
        self.model = model
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds
        self.error_summary = sanitize_provider_error_text(
            error_summary or message,
            max_chars=1000,
        )
        # Structured route identity for Live Activity only; never part of the
        # exception text, error_summary, or backoff keys.
        self.route_kind = route_kind
        self.task_id = task_id
        self.configured_provider = configured_provider
        self.configured_model = configured_model
        self.configured_host_provider = configured_host_provider
        self.upstream_host_provider = upstream_host_provider
        self.auto_fallback_from_host = auto_fallback_from_host
        self.provider_request_id = provider_request_id
        super().__init__(sanitize_provider_error_text(message, max_chars=1000))


class ProviderCooldownError(RetryableProviderError):
    """Raised when a provider is cooling down until a reported reset time."""

    def __init__(
        self,
        *,
        provider: str,
        provider_label: str,
        role_id: str,
        model: str,
        resets_at: Optional[int],
        resets_in_seconds: Optional[int],
        plan_type: str = "",
        message: str = "",
        **route_identity: Any,
    ) -> None:
        self.resets_at = resets_at
        self.resets_in_seconds = resets_in_seconds
        self.plan_type = plan_type
        base = message or f"{provider_label} usage limit reached"
        if resets_in_seconds is not None:
            base = f"{base}; resets in {resets_in_seconds} seconds"
        super().__init__(
            provider=provider,
            provider_label=provider_label,
            role_id=role_id,
            model=model,
            reason="usage_limit_reached",
            message=base,
            retry_after_seconds=resets_in_seconds,
            **route_identity,
        )


# Compatibility alias for callers that imported the original OAuth-specific name.
OAuthProviderCooldownError = ProviderCooldownError


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
        "SHARED TRAINING",
        "LOCAL TRAINING",
        "REJECTION LOG",
        "RETRIEVED EVIDENCE",
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

        # Track provider-wide usage-limit cooldowns reported by subscription providers.
        self._oauth_provider_cooldowns: Dict[str, Dict[str, Any]] = {}
        self._provider_recent_expired_cooldowns: Dict[str, Dict[str, Any]] = {}
        self._provider_resume_pending: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        self._oauth_cooldown_notified: set[str] = set()
        self._oauth_cooldown_fallback_roles: set[str] = set()
        self._oauth_error_notified: set[str] = set()
        self._retryable_provider_backoff_state: Dict[str, int] = {}

        # Top-level workflow owners may suppress proof-only Assistant memory for
        # a run without changing the user's persisted Session History setting.
        self._assistant_memory_suppression_owners: set[str] = set()

        # User-visible source of calls made by run-scoped roles (manual proof
        # checks), so their provider activity can be routed to the owning feed.
        self._role_activity_sources: Dict[str, Dict[str, str]] = {}
        
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

    async def _broadcast_provider_retry_cooldown(
        self,
        payload: Dict[str, Any],
        *,
        role_id: str = "",
    ) -> None:
        """Publish one additive, secret-safe activity item before a provider retry wait."""
        wait_seconds = max(0.0, float(payload.get("retry_after_seconds") or 0))
        provider_label = str(payload.get("provider_label") or payload.get("provider") or "Provider")
        retry_attempt = max(1, int(payload.get("retry_attempt") or 1))
        reason = str(payload.get("reason") or "transient_provider_error")
        error_summary = sanitize_provider_error_text(
            str(payload.get("error_summary") or ""),
            max_chars=1000,
        ).strip()
        supplied_message = sanitize_provider_error_text(
            str(payload.get("message") or ""),
            max_chars=1800,
        )
        error_label = (
            "connection error"
            if reason in {"connection_error", "stream_error"}
            else "server-side error"
            if reason in {
                "http_error",
                "invalid_response",
                "transient_http_error",
                "upstream_provider_error",
            }
            else "API error"
        )
        retry_message = (
            f"{provider_label} API call will retry after a {wait_seconds:g}-second cooldown "
            f"(retry {retry_attempt})."
        )
        display_role = str(payload.get("role_id") or role_id or "")
        supplied_mode = str(payload.get("workflow_mode") or "").strip().lower()
        workflow_mode = (
            supplied_mode
            if supplied_mode in VALID_WORKFLOW_MODES
            else _retry_workflow_mode(display_role)
        )
        event_payload = canonical_recovery_event("provider_retry_cooldown", {
            "workflow_mode": workflow_mode,
            "provider": str(payload.get("provider") or "unknown"),
            "provider_label": provider_label,
            "role_id": display_role,
            "model": str(payload.get("model") or ""),
            "retry_attempt": retry_attempt,
            "max_attempts": payload.get("max_attempts"),
            "retry_after_seconds": wait_seconds,
            "reason": reason,
            "error_summary": error_summary,
            "correlation_id": str(
                payload.get("correlation_id")
                or payload.get("activity_correlation_id")
                or ""
            ),
            "message": supplied_message or (
                f"{provider_label} {error_label}: {error_summary.rstrip('.')}."
                f" {retry_message}"
                if error_summary
                else retry_message
            ),
        }, recovery_action="retry", retry_layer=str(payload.get("retry_layer") or "adapter"))
        # Route identity joins the event after its correlation and event IDs are
        # fixed, and prefixes only the display message.
        if str(payload.get("route_identity") or "").strip():
            fields = {
                key: payload[key]
                for key in ROUTE_IDENTITY_FIELD_NAMES
                if payload.get(key) is not None
                and (payload[key] != "" or key == "configured_host_provider")
            }
        else:
            raw = {key: payload.get(key) for key in _ROUTE_IDENTITY_INPUT_KEYS}
            raw["provider"] = str(payload.get("provider") or "unknown")
            raw["provider_label_text"] = payload.get("provider_label")
            fields = self._resolve_route_identity(
                display_role,
                raw,
                workflow_mode=workflow_mode,
                model_role_id=str(payload.get("model_role_id") or ""),
            )
        event_payload = _merge_route_fields(event_payload, fields)
        event_payload["message"] = with_route_identity(event_payload["message"], fields)
        source_fields = {
            key: payload[key]
            for key in ("source_type", "source_id", "proof_run_id")
            if payload.get(key)
        } or self._role_activity_source_fields(
            str(payload.get("model_role_id") or ""),
            display_role,
        )
        for key, value in source_fields.items():
            event_payload.setdefault(key, value)
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "provider_retry_cooldown",
            event_payload,
        )
        await self._broadcast("provider_retry_cooldown", stored_payload)

    def _resolve_route_identity(
        self,
        role_id: str,
        raw: Dict[str, Any],
        *,
        workflow_mode: str = "",
        model_role_id: str = "",
        error: Optional[BaseException] = None,
    ) -> Dict[str, str]:
        """Compose route identity from producer keys, a typed error and role settings.

        Producer keys win over the error, and both win over role settings. Role
        settings only describe primary routes of model providers, so non-model
        workflow-loop or storage retries never gain a fabricated model or host.
        """
        display_role = str(role_id or "").strip()
        model_role = str(model_role_id or "").strip()
        identity: Dict[str, Any] = {}
        for key in _ROUTE_IDENTITY_INPUT_KEYS:
            value = raw.get(key)
            if key == "configured_host_provider":
                if value is not None:
                    identity[key] = str(value)
            elif value not in (None, "") and not (
                key == "provider" and is_placeholder_provider(value)
            ):
                identity[key] = value
        provider_hint = str(raw.get("provider") or "").strip()
        if error is not None:
            from_error = route_identity_from_error(error)
            error_role = str(from_error.pop("role_id", "") or "")
            error_label = from_error.pop("provider_label_text", "")
            if "provider" not in identity and from_error.get("provider"):
                # The producer's label described its placeholder provider.
                identity.pop("provider_label_text", None)
            for key, value in from_error.items():
                identity.setdefault(key, value)
            if (
                error_label
                and "provider_label_text" not in identity
                and identity.get("provider") == from_error.get("provider")
            ):
                identity["provider_label_text"] = error_label
            if error_role and not display_role:
                display_role = error_role
            elif error_role and error_role != display_role and not model_role:
                model_role = error_role

        config_role = model_role or display_role
        config = self._role_model_configs.get(config_role) if config_role else None
        provider = str(identity.get("provider") or "").strip().lower()
        route_kind = str(identity.get("route_kind") or "").strip().lower()
        if (
            config is not None
            and route_kind in ("", "primary")
            and (is_model_provider(provider) or not (provider or provider_hint))
        ):
            configured_provider = str(config.provider or "")
            identity.setdefault("configured_provider", configured_provider)
            configured_model = _configured_role_model(config)
            if configured_model:
                identity.setdefault("configured_model", configured_model)
            if not provider and is_model_provider(configured_provider):
                fallback_state = self._role_fallback_state.get(config_role, configured_provider)
                if (
                    fallback_state == "lm_studio"
                    and configured_provider != "lm_studio"
                    and config.lm_studio_fallback_id
                ):
                    provider = "lm_studio"
                    identity.setdefault("route_kind", "fallback")
                else:
                    provider = configured_provider
                identity["provider"] = provider
                identity.pop("provider_label_text", None)
            if "model" not in identity:
                route_model = _role_route_model(config, provider)
                if route_model:
                    identity["model"] = route_model
            if (
                "configured_host_provider" not in identity
                and provider == "openrouter"
                and configured_provider == "openrouter"
            ):
                identity["configured_host_provider"] = str(config.openrouter_provider or "")

        mode = str(workflow_mode or "").strip().lower()
        if mode not in VALID_WORKFLOW_MODES:
            mode = _retry_workflow_mode(display_role)
        return route_identity_fields(
            workflow_mode=mode,
            role_id=display_role,
            model_role_id=model_role,
            **identity,
        )

    def route_identity_for_role(
        self,
        role_id: str,
        *,
        workflow_mode: str = "",
        error: Optional[BaseException] = None,
        model: str = "",
        provider: str = "",
        route_kind: str = "",
        task_id: str = "",
        model_role_id: str = "",
        configured_host_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return Live Activity route identity fields for one role's provider error.

        Explicit arguments win over identity carried by ``error``, which wins over
        the role's configured route. Unknown or pseudo roles get no settings
        backfill, and a non-model provider such as ``unknown`` or ``chromadb``
        keeps the result to workflow mode and role only.
        """
        return self._resolve_route_identity(
            role_id,
            {
                "model": model,
                "provider": provider,
                "route_kind": route_kind,
                "task_id": task_id,
                "configured_host_provider": configured_host_provider,
            },
            workflow_mode=workflow_mode,
            model_role_id=model_role_id,
            error=error,
        )

    def _provider_retry_callback(
        self,
        role_id: str,
        *,
        route_kind: str = "primary",
        task_id: str = "",
        configured_model: str = "",
        configured_provider: str = "",
        configured_host_provider: Optional[str] = None,
    ) -> Callable[[Dict[str, Any]], Awaitable[None]]:
        correlation_id = f"provider-retry:{uuid.uuid4().hex}"
        route_context = {
            "route_kind": route_kind,
            "task_id": task_id,
            "configured_model": configured_model,
            "configured_provider": configured_provider,
            "configured_host_provider": configured_host_provider,
        }

        async def callback(payload: Dict[str, Any]) -> None:
            retry_task = asyncio.current_task()
            await self.emit_and_wait_for_retry(
                {
                    **_merge_route_fields(payload, route_context),
                    "correlation_id": (
                        payload.get("correlation_id")
                        or payload.get("activity_correlation_id")
                        or correlation_id
                    ),
                },
                role_id=role_id,
                should_stop=lambda: bool(
                    retry_task is not None
                    and retry_task.cancelling()
                ),
            )

        return callback

    async def emit_and_wait_for_retry(
        self,
        payload: Dict[str, Any],
        *,
        role_id: str = "",
        should_stop: Optional[Callable[[], bool]] = None,
        error: Optional[BaseException] = None,
    ) -> bool:
        """Atomically publish a retry wait and perform its stop-aware delay.

        ``error`` optionally supplies typed route identity for the activity item;
        it only fills keys the producer did not provide.
        """
        wait_seconds = max(0.0, float(payload.get("retry_after_seconds") or 0))
        event_payload = dict(payload)
        if error is not None:
            from_error = route_identity_from_error(error)
            error_role = str(from_error.pop("role_id", "") or "")
            error_label = from_error.pop("provider_label_text", "")
            if error_label:
                from_error["provider_label"] = error_label
            event_payload = _merge_route_fields(event_payload, from_error)
            display_role = str(event_payload.get("role_id") or role_id or "")
            if error_role and not display_role:
                event_payload["role_id"] = error_role
            elif error_role and error_role != display_role and not event_payload.get("model_role_id"):
                event_payload["model_role_id"] = error_role
        if not (
            event_payload.get("correlation_id")
            or event_payload.get("activity_correlation_id")
        ):
            event_payload["correlation_id"] = f"provider-retry:{uuid.uuid4().hex}"
        await self._broadcast_provider_retry_cooldown(event_payload, role_id=role_id)
        return await self._sleep_with_optional_stop(
            wait_seconds,
            should_stop,
        )

    async def _broadcast_unrecoverable_codex_error(
        self,
        *,
        role_id: str,
        model: str,
        error: Exception,
        task_id: str = "",
    ) -> None:
        """Notify the UI when a Codex role cannot recover through fallback."""
        notification_key = f"openai_codex_oauth:{role_id}:unrecoverable_codex_error:{model}"
        if notification_key in self._oauth_error_notified:
            return
        self._oauth_error_notified.add(notification_key)
        payload = {
            "role_id": role_id,
            "model": model,
            "provider": "openai_codex_oauth",
            "provider_label": "OpenAI Codex",
            "reason": "unrecoverable_codex_error",
            "recoverable": False,
            "workflow_mode": _retry_workflow_mode(role_id),
            "message": (
                "OpenAI Codex failed and no LM Studio fallback is configured. "
                "Please check your OpenAI Codex OAuth connection in OpenRouter/OAuth, "
                "sign in again, and retry."
            ),
            "error_summary": redact_log_text(str(error), 700),
            "oauth_error_message": oauth_live_activity_error_message(error),
        }
        payload = _merge_route_fields(payload, self.route_identity_for_role(
            role_id,
            workflow_mode=payload["workflow_mode"],
            error=error,
            model=model,
            provider="openai_codex_oauth",
            route_kind="primary",
            task_id=task_id,
        ))
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "openai_codex_oauth_error",
            payload,
        )
        await self._broadcast("openai_codex_oauth_error", stored_payload)

    async def _broadcast_unrecoverable_subscription_error(
        self,
        *,
        provider: str,
        role_id: str,
        model: str,
        error: Exception,
        task_id: str = "",
    ) -> None:
        """Notify the UI when an API-key subscription role cannot recover through fallback."""
        client = get_api_key_subscription_client(provider)
        if client is None:
            return
        label = client.provider_label
        reason = f"unrecoverable_{provider}_error"
        notification_key = f"{provider}:{role_id}:{reason}:{model}"
        if notification_key in self._oauth_error_notified:
            return
        self._oauth_error_notified.add(notification_key)
        payload = {
            "role_id": role_id,
            "model": model,
            "provider": provider,
            "provider_label": label,
            "reason": reason,
            "recoverable": False,
            "workflow_mode": _retry_workflow_mode(role_id),
            "message": (
                f"{label} failed and no LM Studio fallback is configured. "
                f"Please check your {label} API key in OpenRouter/OAuth and retry."
            ),
            "error_summary": redact_log_text(str(error), 700),
            "oauth_error_message": oauth_live_activity_error_message(error),
        }
        terminal_guidance = str(getattr(error, "terminal_guidance", "") or "")
        if terminal_guidance:
            payload["terminal_guidance"] = terminal_guidance
        payload = _merge_route_fields(payload, self.route_identity_for_role(
            role_id,
            workflow_mode=payload["workflow_mode"],
            error=error,
            model=model,
            provider=provider,
            route_kind="primary",
            task_id=task_id,
        ))
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "subscription_provider_error",
            payload,
        )
        await self._broadcast("subscription_provider_error", stored_payload)

    async def _broadcast_unrecoverable_xai_grok_error(
        self,
        *,
        role_id: str,
        model: str,
        error: Exception,
        task_id: str = "",
    ) -> None:
        """Notify the UI when a Grok OAuth role cannot recover through fallback."""
        reason = (
            error.reason
            if isinstance(error, ProviderRepairRequiredError)
            else "unrecoverable_xai_grok_error"
        )
        # The dedupe key keeps its original owner-mode composition so already
        # notified alerts are not replayed; the payload names the role's workflow.
        notification_key = (
            f"xai_grok_oauth:{_active_notification_workflow_mode() or 'unknown'}:"
            f"{role_id}:{reason}:{model}"
        )
        if notification_key in self._oauth_error_notified:
            return
        self._oauth_error_notified.add(notification_key)
        workflow_mode = _retry_workflow_mode(role_id)
        payload = {
            "role_id": role_id,
            "model": model,
            "provider": "xai_grok_oauth",
            "provider_label": "xAI Grok",
            "reason": reason,
            "recoverable": False,
            "workflow_mode": workflow_mode,
            "message": (
                error.safe_message
                if isinstance(error, ProviderRepairRequiredError)
                else (
                    "xAI Grok failed and no LM Studio fallback is configured. "
                    "Please check your xAI Grok OAuth connection in OpenRouter/OAuth, "
                    "sign in again, and retry. If xAI reports subscription or credit limits, "
                    "check your SuperGrok/X Premium entitlement."
                )
            ),
            "error_summary": redact_log_text(str(error), 700),
            "oauth_error_message": oauth_live_activity_error_message(error),
            "terminal_guidance": (
                error.terminal_guidance
                if isinstance(error, ProviderRepairRequiredError)
                else ""
            ),
        }
        # A repair error contributes its configured route, host and task fields.
        payload = _merge_route_fields(payload, self.route_identity_for_role(
            role_id,
            workflow_mode=workflow_mode,
            error=error,
            model=model,
            provider="xai_grok_oauth",
            route_kind="primary",
            task_id=task_id,
        ))
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "oauth_provider_error",
            payload,
        )
        await self._broadcast("oauth_provider_error", stored_payload)

    @staticmethod
    def _cooldown_until_from_error(error: Any) -> int:
        if error.resets_at:
            return int(error.resets_at)
        if error.resets_in_seconds:
            return int(time.time()) + int(error.resets_in_seconds)
        raise ValueError("Timed provider cooldown requires a credible reset time")

    def get_provider_cooldown(self, provider: str) -> Optional[Dict[str, Any]]:
        """Return active cooldown metadata for a provider, clearing expired entries."""
        provider_key = str(provider or "").strip()
        if not provider_key:
            return None
        cooldown = self._oauth_provider_cooldowns.get(provider_key)
        if not cooldown:
            return None
        cooldown_until = int(cooldown.get("cooldown_until") or cooldown.get("resets_at") or 0)
        if cooldown_until and cooldown_until <= int(time.time()):
            self._oauth_provider_cooldowns.pop(provider_key, None)
            self._provider_recent_expired_cooldowns[provider_key] = dict(cooldown)
            self._oauth_cooldown_notified = {
                key for key in self._oauth_cooldown_notified if not key.startswith(f"{provider_key}:")
            }
            return None
        resets_in_seconds = max(1, cooldown_until - int(time.time())) if cooldown_until else None
        return {**cooldown, "resets_in_seconds": resets_in_seconds}

    def is_provider_cooling_down(self, provider: str) -> bool:
        return self.get_provider_cooldown(provider) is not None

    @staticmethod
    def _retryable_provider_key(
        provider: str,
        role_id: str,
        model: str,
        reason: str = "",
    ) -> str:
        return "|".join(
            part.strip().lower()
            for part in (provider or "unknown", role_id or "unknown", model or "unknown", reason or "retryable")
        )

    @staticmethod
    async def _sleep_with_optional_stop(
        seconds: float,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Compatibility seam over the canonical stop-aware recovery wait."""
        return await provider_recovery_runtime.sleep_stop_aware(seconds, should_stop)

    def _clear_retryable_provider_backoff(self, provider: str, role_id: str, model: str) -> None:
        prefix = self._retryable_provider_key(provider, role_id, model).rsplit("|", 1)[0] + "|"
        self._retryable_provider_backoff_state = {
            key: value
            for key, value in self._retryable_provider_backoff_state.items()
            if not key.startswith(prefix)
        }

    @staticmethod
    def _as_retryable_provider_error(
        *,
        provider: str,
        provider_label: str,
        role_id: str,
        model: str,
        error: Exception,
        route_kind: str = "",
        task_id: str = "",
        configured_provider: str = "",
        configured_model: str = "",
        configured_host_provider: Optional[str] = None,
    ) -> RetryableProviderError:
        error_summary = sanitize_provider_error_text(
            getattr(error, "safe_message", "") or str(error),
            max_chars=1000,
        )
        message = (
            f"{provider_label} transient provider failure for role '{role_id}' "
            f"after internal retries: {error_summary}"
        )
        # Only the host-side facts the adapter observed come from the typed error.
        route_identity = route_identity_from_error(error)
        if configured_host_provider is None:
            configured_host_provider = route_identity.get("configured_host_provider")
        return RetryableProviderError(
            provider=provider,
            provider_label=provider_label,
            role_id=role_id,
            model=model,
            reason="transient_provider_error",
            message=message,
            error_summary=error_summary,
            route_kind=route_kind,
            task_id=task_id,
            configured_provider=configured_provider,
            configured_model=configured_model,
            configured_host_provider=configured_host_provider,
            upstream_host_provider=str(route_identity.get("upstream_host_provider") or ""),
            auto_fallback_from_host=str(route_identity.get("auto_fallback_from_host") or ""),
            provider_request_id=str(route_identity.get("provider_request_id") or ""),
        )

    @staticmethod
    def _as_provider_repair_error(
        *,
        provider: str,
        provider_label: str,
        role_id: str,
        model: str,
        error: Exception,
        reason: str = "provider_repair_required",
        terminal_guidance: str = "",
    ) -> ProviderRepairRequiredError:
        """Normalize a definitive route failure without replaying provider payloads."""
        if isinstance(error, ProviderRepairRequiredError):
            return error
        route = getattr(error, "route", None)
        error_detail = getattr(error, "safe_message", "") or str(error)
        return ProviderRepairRequiredError(
            provider=provider,
            provider_label=provider_label,
            role_id=role_id,
            model=model,
            reason=reason,
            message=(
                f"{provider_label} could not serve role '{role_id}'. "
                "The configured provider route requires operator repair."
            ),
            terminal_guidance=terminal_guidance or (
                f"Repair the {provider_label} credential, entitlement, model selection, "
                "or local service, then retry the workflow."
            ),
            configured_provider=getattr(route, "configured_provider", "") or provider,
            configured_model=getattr(route, "configured_model", "") or model,
            effective_host_provider=(
                getattr(route, "upstream_host_provider", "")
                or getattr(route, "host_provider", "")
            ),
            route_kind=getattr(route, "route_kind", ""),
            error_detail=error_detail,
            configured_host_provider=(
                (getattr(route, "host_provider", "") or "")
                if route is not None
                else None
            ),
            upstream_host_provider=getattr(route, "upstream_host_provider", "") or "",
            auto_fallback_from_host=getattr(route, "auto_fallback_from_host", "") or "",
            task_id=getattr(route, "task_id", "") or "",
        )

    @staticmethod
    def is_provider_failure(error: Exception) -> bool:
        """Return whether an exception belongs to provider recovery, not validation."""
        return isinstance(
            error,
            (
                RetryableProviderError,
                ProviderRepairRequiredError,
                ProviderContextLengthError,
                ProviderRouteError,
                CreditExhaustionError,
                FreeModelExhaustedError,
                OpenRouterPrivacyPolicyError,
            ),
        ) or is_transient_model_call_error(error)

    async def wait_for_retryable_provider_error(
        self,
        error: RetryableProviderError,
        *,
        role_id: str = "",
        should_stop: Optional[Callable[[], bool]] = None,
        activity_callback: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[None]]
        ] = None,
        workflow_mode: str = "",
    ) -> None:
        """Apply one standard workflow-level backoff for retryable provider failures.

        ``workflow_mode`` names the owning workflow when the caller knows it
        (for example a manual proof check started while another workflow owns
        the session); otherwise it is derived from the active owner and role.
        """
        if isinstance(error, ProviderCooldownError):
            await self.wait_for_provider_cooldown(
                error,
                role_id=role_id,
                should_stop=should_stop,
                workflow_mode=workflow_mode,
            )
            return

        provider = str(error.provider or "").strip() or "unknown"
        active_role = error.role_id or role_id or "unknown"
        display_role = role_id or active_role
        model = str(error.model or "").strip()
        key = self._retryable_provider_key(provider, active_role, model, error.reason)
        failure_count = self._retryable_provider_backoff_state.get(key, 0) + 1
        self._retryable_provider_backoff_state[key] = failure_count
        wait_seconds = int(error.retry_after_seconds or min(60 * (2 ** (failure_count - 1)), 900))
        safe_error_detail = sanitize_provider_error_text(
            error.error_summary or oauth_live_activity_error_message(error),
            max_chars=OAUTH_LIVE_ERROR_MAX_CHARS,
        )
        # Identity is built before the activity callback so callback consumers
        # (such as proof_retry_* events) and the broadcast share one payload.
        fields = self._resolve_route_identity(
            display_role,
            {"provider": provider},
            workflow_mode=workflow_mode,
            model_role_id=(
                error.role_id
                if error.role_id and error.role_id != display_role
                else ""
            ),
            error=error,
        )
        activity_payload = _merge_route_fields({
            "provider": provider,
            "provider_label": error.provider_label or provider,
            "role_id": display_role,
            "model": model,
            "retry_attempt": failure_count,
            "retry_after_seconds": wait_seconds,
            "reason": error.reason,
            "error_summary": safe_error_detail,
            "retry_layer": "workflow",
        }, fields)
        provider_label = activity_payload["provider_label"]
        logger.warning(
            "%s retryable provider failure for role '%s' (model %s, host %s; attempt %s after "
            "provider retries); waiting %s seconds before retry: %s",
            provider_label,
            display_role,
            activity_payload.get("model") or "unknown",
            _route_log_host(fields),
            failure_count,
            wait_seconds,
            error,
        )
        activity_payload["message"] = with_route_identity(
            f"{provider_label} API error: "
            f"{safe_error_detail.rstrip('.')}. "
            f"{provider_label} API call will retry after a "
            f"{wait_seconds:g}-second cooldown (retry {failure_count}).",
            fields,
        )
        if activity_callback is not None:
            await activity_callback("waiting", activity_payload)
        await self.emit_and_wait_for_retry(
            activity_payload,
            role_id=display_role,
            should_stop=should_stop,
        )
        if (
            activity_callback is not None
            and not (should_stop is not None and should_stop())
        ):
            await activity_callback("resuming", activity_payload)

    async def wait_for_provider_cooldown(
        self,
        error: ProviderCooldownError,
        *,
        role_id: str = "",
        should_stop: Optional[Callable[[], bool]] = None,
        workflow_mode: str = "",
    ) -> None:
        """Sleep until a provider usage-limit cooldown expires or stop is requested."""
        provider = str(error.provider or "").strip() or "unknown"
        active_role = role_id or error.role_id or "unknown"
        fields = self._resolve_route_identity(
            active_role,
            {"provider": provider},
            workflow_mode=workflow_mode,
            model_role_id=(
                error.role_id
                if error.role_id and error.role_id != active_role
                else ""
            ),
            error=error,
        )
        waited_for_cooldown = False
        cooldown_chunk_attempt = 0
        while self.is_provider_cooling_down(provider):
            if should_stop is not None and should_stop():
                return
            waited_for_cooldown = True
            cooldown = self.get_provider_cooldown(provider) or {}
            wait_seconds = cooldown.get("resets_in_seconds")
            if wait_seconds is None and error.resets_in_seconds is not None:
                wait_seconds = error.resets_in_seconds
            if wait_seconds is None and error.resets_at:
                wait_seconds = max(1, int(error.resets_at) - int(time.time()))
            wait_seconds = max(1, min(int(wait_seconds or 60), 300))
            cooldown_chunk_attempt += 1
            logger.warning(
                "%s usage-limit cooldown active for role '%s' (model %s, host %s); "
                "waiting %s seconds before retry",
                error.provider_label or provider,
                active_role,
                fields.get("model") or "unknown",
                _route_log_host(fields),
                wait_seconds,
            )
            await self.emit_and_wait_for_retry(
                _merge_route_fields({
                    "provider": provider,
                    "provider_label": error.provider_label or provider,
                    "role_id": active_role,
                    "model": error.model,
                    "retry_attempt": cooldown_chunk_attempt,
                    "retry_after_seconds": wait_seconds,
                    "reason": "usage_limit_cooldown",
                    "cooldown_until": (
                        cooldown.get("cooldown_until")
                        or cooldown.get("resets_at")
                        or error.resets_at
                    ),
                }, fields),
                role_id=active_role,
                should_stop=should_stop,
            )
            if should_stop is not None and should_stop():
                return
        if waited_for_cooldown and not (should_stop is not None and should_stop()):
            expired = self._provider_recent_expired_cooldowns.get(provider, {})
            self._provider_resume_pending[(provider, active_role, str(error.model or ""))] = {
                **expired,
                "provider": provider,
                "provider_label": error.provider_label or expired.get("provider_label") or provider,
                "role_id": active_role,
                "model": error.model,
                "resets_at": error.resets_at or expired.get("resets_at"),
                "cooldown_until": error.resets_at or expired.get("cooldown_until"),
                # The waiting role's workflow, not the role that first hit the limit.
                "workflow_mode": str(fields.get("workflow_mode") or ""),
            }

    async def _confirm_provider_usage_limit_resumed(
        self,
        *,
        provider: str,
        provider_label: str,
        role_id: str,
        model: str,
        task_id: str = "",
    ) -> None:
        """Emit one durable resume event only after a real provider success."""
        key = (provider, role_id, model)
        pending = self._provider_resume_pending.pop(key, None)
        if pending is None:
            return
        self._provider_recent_expired_cooldowns.pop(provider, None)
        self._provider_resume_pending = {
            pending_key: pending_value
            for pending_key, pending_value in self._provider_resume_pending.items()
            if pending_key[0] != provider
        }
        reset_at = int(pending.get("cooldown_until") or pending.get("resets_at") or 0)
        safe_model = str(model or "*").replace(":", "_")
        payload = canonical_recovery_event("provider_cooldown_resumed", {
            "notification_key": (
                f"{provider}:{role_id}:usage_limit_resumed:"
                f"{safe_model}@{reset_at or 'elapsed'}"
            ),
            "provider": provider,
            "provider_label": provider_label,
            "role_id": role_id,
            "model": model,
            "reason": "usage_limit_resumed",
            "recoverable": True,
            "workflow_mode": pending.get("workflow_mode") or _retry_workflow_mode(role_id),
            "resets_at": reset_at or None,
            "cooldown_until": reset_at or None,
            "message": (
                f"{provider_label} responded successfully for {role_id}; "
                "automatic provider work resumed."
            ),
        }, recovery_action="recovery_confirmed")
        payload = _merge_route_fields(payload, self.route_identity_for_role(
            role_id,
            workflow_mode=str(payload.get("workflow_mode") or ""),
            model=model,
            provider=provider,
            route_kind="primary",
            task_id=task_id,
        ))
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "provider_usage_limit_resumed",
            payload,
        )
        await self._broadcast("provider_usage_limit_resumed", stored_payload)

    async def wait_for_oauth_provider_cooldown(
        self,
        error: ProviderCooldownError,
        *,
        role_id: str = "",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Compatibility wrapper for the original OAuth-specific helper."""
        await self.wait_for_provider_cooldown(error, role_id=role_id, should_stop=should_stop)

    def _mark_provider_cooldown(
        self,
        error: Any,
        *,
        role_id: str,
        model: str,
        task_id: str = "",
    ) -> Dict[str, Any]:
        cooldown_until = self._cooldown_until_from_error(error)
        resets_in_seconds = max(1, cooldown_until - int(time.time()))
        provider = str(getattr(error, "provider", "") or "unknown")
        provider_label = str(getattr(error, "provider_label", "") or provider)
        payload = canonical_recovery_event("provider_cooldown_started", {
            "provider": provider,
            "provider_label": provider_label,
            "role_id": role_id,
            "model": model,
            "reason": "usage_limit_reached",
            "recoverable": True,
            "workflow_mode": _retry_workflow_mode(role_id),
            "plan_type": str(getattr(error, "plan_type", "") or ""),
            "resets_at": cooldown_until,
            "cooldown_until": cooldown_until,
            "resets_in_seconds": resets_in_seconds,
            "message": (
                f"{provider_label} usage limit reached for {role_id}. "
                f"Provider reports reset in {resets_in_seconds} seconds."
            ),
            "error_summary": redact_log_text(str(error), 700),
            "oauth_error_message": oauth_live_activity_error_message(error),
        }, recovery_action="cooldown")
        payload = _merge_route_fields(payload, self.route_identity_for_role(
            role_id,
            workflow_mode=str(payload.get("workflow_mode") or ""),
            error=error,
            model=model,
            provider=provider,
            route_kind="primary",
            task_id=task_id,
        ))
        self._oauth_provider_cooldowns[provider] = payload
        return payload

    def _mark_oauth_provider_cooldown(
        self,
        error: Any,
        *,
        role_id: str,
        model: str,
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Compatibility wrapper for the original OAuth-specific helper."""
        return self._mark_provider_cooldown(error, role_id=role_id, model=model, task_id=task_id)

    def _cooldown_payload_for_role(
        self,
        cooldown: Dict[str, Any],
        *,
        role_id: str,
        model: str,
        task_id: str = "",
    ) -> Dict[str, Any]:
        """Reuse a provider-wide cooldown payload for another role's blocked call.

        The cooldown is shared across roles, so the event identity, workflow mode
        and route identity derived for the role that first hit the limit are
        rebuilt for this role; its notification then never replaces that role's.
        """
        payload = {
            key: value
            for key, value in cooldown.items()
            if key not in _ROLE_SCOPED_ROUTE_IDENTITY_KEYS
            and key not in _COOLDOWN_EVENT_IDENTITY_KEYS
        }
        payload.update({
            "role_id": role_id,
            "model": model,
            "workflow_mode": _retry_workflow_mode(role_id),
        })
        payload = canonical_recovery_event(
            "provider_cooldown_started",
            payload,
            recovery_action="cooldown",
        )
        return _merge_route_fields(payload, self.route_identity_for_role(
            role_id,
            workflow_mode=str(payload.get("workflow_mode") or ""),
            model=model,
            provider=str(payload.get("provider") or ""),
            route_kind="primary",
            task_id=task_id,
        ))

    async def _broadcast_provider_usage_limit(
        self,
        payload: Dict[str, Any],
        *,
        fallback_model: str = "",
    ) -> None:
        notify_payload = dict(payload)
        provider_label = notify_payload.get("provider_label", "OAuth provider")
        role_id = notify_payload.get("role_id", "a role")
        resets_in_seconds = notify_payload.get("resets_in_seconds")
        if fallback_model:
            notify_payload["fallback_model"] = fallback_model
            notify_payload["message"] = (
                f"{provider_label} usage limit reached for "
                f"{role_id}. Using LM Studio fallback model {fallback_model} "
                f"until the provider reset."
            )
        elif notify_payload.get("reason") == "usage_limit_reached":
            reset_text = (
                f" Provider reports reset in {resets_in_seconds} seconds."
                if resets_in_seconds is not None
                else ""
            )
            notify_payload["message"] = (
                f"{provider_label} usage limit reached for {role_id}."
                " Roles without fallback will wait until the provider reset."
                f"{reset_text}"
            )
        cooldown_key = (
            f"{notify_payload.get('provider')}:{notify_payload.get('role_id')}:"
            f"{notify_payload.get('model')}:{notify_payload.get('cooldown_until')}:"
            f"{notify_payload.get('fallback_model', '')}"
        )
        if cooldown_key in self._oauth_cooldown_notified:
            return
        self._oauth_cooldown_notified.add(cooldown_key)
        stored_payload = await asyncio.to_thread(
            record_provider_notification,
            "oauth_provider_usage_limited",
            notify_payload,
        )
        await self._broadcast("oauth_provider_usage_limited", stored_payload)

    async def _broadcast_oauth_usage_limit(
        self,
        payload: Dict[str, Any],
        *,
        fallback_model: str = "",
    ) -> None:
        """Compatibility wrapper for the original OAuth-specific helper."""
        await self._broadcast_provider_usage_limit(payload, fallback_model=fallback_model)
    
    async def _with_hung_connection_watchdog(
        self,
        coro,
        role_id: str,
        model: str,
        provider: str,
        timeout_seconds: int = 1500,
        *,
        provider_id: str = "",
        route_kind: str = "",
        task_id: str = "",
        configured_host_provider: Optional[str] = None,
    ):
        """Wrap an API call with a non-canceling watchdog (default 25 minutes).

        ``provider`` is the display text the alert has always carried;
        ``provider_id`` and the route keywords only supply route identity.
        """
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
            fields = self.route_identity_for_role(
                role_id,
                model=model,
                provider=provider_id,
                route_kind=route_kind,
                task_id=task_id,
                configured_host_provider=configured_host_provider,
            )
            await self._broadcast("hung_connection_alert", _merge_route_fields({
                "role_id": role_id,
                "model": model,
                "provider": provider,
                "provider_label": fields.get("provider_label", ""),
                "elapsed_minutes": minutes,
                "message": (
                    "The model may still be thinking; you can keep waiting or lower reasoning effort "
                    "in Settings if this repeats."
                )
            }, fields))

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
    
    def set_role_activity_source(
        self,
        role_ids: List[str],
        *,
        source_type: str = "",
        source_id: str = "",
        proof_run_id: str = "",
    ) -> None:
        """Record which user-visible source a run-scoped role's calls belong to.

        The source travels on that role's provider activity as routing data only;
        it never enters correlation, notification, or backoff keys.
        """
        source = {
            key: redact_log_text(value, limit)
            for key, value, limit in (
                ("source_type", source_type, 40),
                ("source_id", source_id, 240),
                ("proof_run_id", proof_run_id, 120),
            )
            if value
        }
        for role_id in role_ids:
            if role_id and source:
                self._role_activity_sources[role_id] = dict(source)

    def clear_role_activity_source(self, role_ids: List[str]) -> None:
        for role_id in role_ids:
            self._role_activity_sources.pop(role_id, None)

    def _role_activity_source_fields(self, *role_ids: str) -> Dict[str, str]:
        """Return the registered source for a role, including run-suffixed variants."""
        for role_id in role_ids:
            if not role_id:
                continue
            source = self._role_activity_sources.get(role_id)
            if source is not None:
                return dict(source)
            for registered_role, registered_source in self._role_activity_sources.items():
                if role_id.startswith(f"{registered_role}_"):
                    return dict(registered_source)
        return {}

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
        self._oauth_cooldown_fallback_roles.discard(role_id)
        
        # Set initial fallback state based on provider
        if config.provider == "openrouter" or config.provider in CLOUD_PROVIDER_IDS:
            self._role_fallback_state[role_id] = config.provider
        else:
            self._role_fallback_state[role_id] = "lm_studio"

        # Routine role registration is frequent (especially when restoring workflows).
        # Keep the details available for diagnostics without flooding normal startup logs.
        if config.provider == "openrouter":
            or_model = config.openrouter_model_id or config.model_id
            provider_str = f" via {config.openrouter_provider}" if config.openrouter_provider else ""
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.debug(f"Configured role '{role_id}': provider=openrouter, model={or_model}{provider_str}{fallback_str}")
        elif config.provider in CLOUD_PROVIDER_IDS:
            fallback_str = f", fallback={config.lm_studio_fallback_id}" if config.lm_studio_fallback_id else ""
            logger.debug(f"Configured role '{role_id}': provider={config.provider}, model={config.model_id}{fallback_str}")
        else:
            logger.debug(f"Configured role '{role_id}': provider=lm_studio, model={config.model_id}")

    def get_role_config(self, role_id: str) -> Optional[ModelConfig]:
        """Return a configured role snapshot without exposing mutable internals."""
        config = self._role_model_configs.get(role_id)
        return config.model_copy() if config is not None else None

    def set_assistant_memory_suppressed(self, owner: str, suppressed: bool) -> None:
        """Set run-scoped Assistant proof-memory suppression for one owner."""
        owner_key = str(owner or "").strip()
        if not owner_key:
            return
        if suppressed:
            self._assistant_memory_suppression_owners.add(owner_key)
        else:
            self._assistant_memory_suppression_owners.discard(owner_key)

    def _assistant_memory_is_suppressed(self) -> bool:
        """Return whether an active workflow has disabled proof-memory context."""
        return bool(self._assistant_memory_suppression_owners)

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
            "formalization",
            "proof_form",
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
        run_id_override: str = "",
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
                "SHARED TRAINING",
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
        formal_target_kinds = {
            "proof_candidate",
            "lean_error",
            "theorem_discovery",
            "master_proof",
            "paper_claim",
        }
        uses_formal_proof_context = (
            target_kind in formal_target_kinds
            or workflow_mode == "manual_proof_check"
            or (workflow_mode == "leanoj" and target_kind == "final_solver")
        )
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
            run_id=run_id_override,
            source_titles=source_titles,
            imports=["Mathlib"] if uses_formal_proof_context else [],
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
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
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
        if (
            not system_config.agent_conversation_memory_enabled
            or self._assistant_memory_is_suppressed()
        ):
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
            run_id_override=await self._assistant_run_id_for_workflow(
                workflow_mode_override or self._assistant_workflow_mode_for_role(role_id)
            ),
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
        if (
            not system_config.agent_conversation_memory_enabled
            or self._assistant_memory_is_suppressed()
        ):
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
            run_id_override=await self._assistant_run_id_for_workflow(
                workflow_mode_override or self._assistant_workflow_mode_for_role(role_id)
            ),
        )
        target_hash = assistant_proof_search_coordinator.submit_target(snapshot)
        return target_hash

    @staticmethod
    async def _assistant_run_id_for_workflow(workflow_mode: str) -> str:
        """Resolve durable workflow identity without deriving it from model prompts."""
        mode = str(workflow_mode or "").strip().lower()
        if mode in {"aggregator", "compiler", "manual_proof_check"}:
            try:
                from backend.autonomous.memory.proof_database import manual_proof_database
                return await manual_proof_database.get_or_create_active_run_id()
            except Exception:
                logger.debug("Could not resolve active manual Assistant run ID", exc_info=True)
                return ""
        if mode == "autonomous":
            try:
                from backend.autonomous.memory.session_manager import session_manager
                return str(session_manager.session_id or "").strip() if session_manager.is_session_active else ""
            except Exception:
                logger.debug("Could not resolve autonomous Assistant run ID", exc_info=True)
                return ""
        if mode == "leanoj":
            try:
                from backend.leanoj.core.leanoj_coordinator import leanoj_coordinator
                state = getattr(leanoj_coordinator, "_state", None)
                return str(getattr(state, "session_id", "") or "").strip()
            except Exception:
                logger.debug("Could not resolve LeanOJ Assistant run ID", exc_info=True)
                return ""
        return ""

    @staticmethod
    def _append_assistant_memory_block(prompt: str, assistant_context: str) -> str:
        return (
            f"{prompt}\n\n---\n\n"
            "OPTIONAL ASSISTANT MEMORY CONTEXT:\n"
            f"{assistant_context}\n\n"
            "Use the Assistant memory only when it is relevant. It is supporting context, "
            "not validator feedback, not a requirement to cite, and not a replacement for the user prompt. "
            "These proof supports cannot redirect or mathematically reinterpret the user objective, "
            "and they do not require mathematics or formal proof."
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

        if role_config is None:
            raise ProviderRepairRequiredError(
                provider="unconfigured",
                provider_label="Model routing",
                role_id=role_id,
                model=model,
                reason="role_not_configured",
                message=(
                    f"Model role '{role_id}' is not configured. "
                    "MOTO refused to guess a provider route."
                ),
                terminal_guidance=(
                    "Configure this role with explicit provider, model, context-window, "
                    "and max-output settings before retrying the workflow."
                ),
                task_id=task_id,
            )

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
        try:
            response = await self._generate_completion_once_routed(
                task_id=task_id,
                role_id=role_id,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            drop_unfinished_thinking(response, context=task_id)
            return response
        except (ProviderRouteError, ProviderContextLengthError) as error:
            if error.route.role_id == role_id and error.route.task_id == task_id:
                raise
            route_kind = error.route.route_kind
            if error.route.provider == "lm_studio":
                configured = self._role_model_configs.get(role_id)
                if configured and configured.provider != "lm_studio":
                    route_kind = "fallback"
            configured = self._role_model_configs.get(role_id)
            raise error.with_route_context(
                role_id=role_id,
                task_id=task_id,
                route_kind=route_kind,
                configured_provider=configured.provider if configured else "",
                configured_model=configured.model_id if configured else model,
            ) from error

    async def _generate_completion_once_routed(
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
        """Execute one routed call; the public helper adds safe route context."""
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
                            retry_callback=self._provider_retry_callback(
                                role_id,
                                route_kind="boost",
                                task_id=task_id,
                                configured_model=(
                                    _configured_role_model(initial_role_config)
                                    if initial_role_config
                                    else requested_model
                                ),
                                configured_provider=configured_provider or "",
                                configured_host_provider=boost_provider or "",
                            ),
                        ),
                        role_id=role_id,
                        model=boost_model,
                        provider=boost_provider or "OpenRouter",
                        provider_id="openrouter",
                        route_kind="boost",
                        task_id=task_id,
                        configured_host_provider=boost_provider or "",
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
                await self._broadcast("openrouter_rate_limit", _merge_route_fields({
                    "model": boost_model,
                    "provider": "openrouter",
                    "role_id": role_id,
                    "message": f"OpenRouter rate limit hit for '{boost_model}' after retries exhausted."
                }, self.route_identity_for_role(
                    role_id,
                    error=e,
                    model=boost_model,
                    provider="openrouter",
                    route_kind="boost",
                    task_id=task_id,
                    configured_host_provider=boost_provider or "",
                )))
                
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
                await self._broadcast("openrouter_privacy_error", _merge_route_fields({
                    "error_type": "privacy_policy",
                    "model": boost_model,
                    "provider": "openrouter",
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
                }, self.route_identity_for_role(
                    role_id,
                    error=e,
                    model=boost_model,
                    provider="openrouter",
                    route_kind="boost",
                    task_id=task_id,
                    configured_host_provider=boost_provider or "",
                )))

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
                await self._broadcast("boost_credits_exhausted", _merge_route_fields({
                    "task_id": task_id,
                    "role_id": role_id,
                    "model": boost_model,
                    "provider": "openrouter",
                    "message": "Boost credits exhausted, falling back to primary model"
                }, self.route_identity_for_role(
                    role_id,
                    error=e,
                    model=boost_model,
                    provider="openrouter",
                    route_kind="boost",
                    task_id=task_id,
                    configured_host_provider=boost_provider or "",
                )))
                if strict_boost:
                    raise RuntimeError(f"Strict boost call credits exhausted for task {task_id}: {e}") from e
                # Continue to primary model routing below
                
            except ProviderContextLengthError as e:
                duration_ms = (time.time() - start_time) * 1000
                await boost_logger.log_boost_call(
                    task_id=task_id,
                    role_id=role_id,
                    model=boost_model,
                    prompt_preview=prompt_preview,
                    response_content="",
                    duration_ms=duration_ms,
                    success=False,
                    error=e.safe_message,
                    boost_mode=boost_mode,
                )
                typed_error = _typed_provider_context_error(
                    e,
                    provider="openrouter",
                    model=boost_model,
                    route_kind="boost",
                )
                if strict_boost:
                    raise typed_error from e
                logger.warning(
                    "Boost context limit rejected task %s; trying primary route",
                    task_id,
                )
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
            role_config = self._role_model_configs.get(role_id)
            if role_config is None:
                raise ProviderRepairRequiredError(
                    provider="unconfigured",
                    provider_label="Model routing",
                    role_id=role_id,
                    model=model,
                    reason="role_not_configured",
                    message=(
                        f"Model role '{role_id}' is not configured. "
                        "MOTO refused to guess a provider route."
                    ),
                    terminal_guidance=(
                        "Configure this role with explicit provider, model, context-window, "
                        "and max-output settings before retrying the workflow."
                    ),
                    task_id=task_id,
                )
            fallback_state = self._role_fallback_state.get(role_id, role_config.provider)

            if system_config.generic_mode and role_config and fallback_state != "openrouter":
                logger.warning(
                    "Generic mode reset role '%s' fallback state from %s to OpenRouter.",
                    role_id,
                    fallback_state,
                )
                fallback_state = "openrouter"
                self._role_fallback_state[role_id] = "openrouter"
            elif (
                role_config
                and role_config.provider in CLOUD_PROVIDER_IDS
                and fallback_state == "lm_studio"
                and role_id in self._oauth_cooldown_fallback_roles
                and not self.is_provider_cooling_down(role_config.provider)
            ):
                logger.info(
                    "%s cooldown expired for role '%s'; returning role to its configured provider.",
                    role_config.provider,
                    role_id,
                )
                fallback_state = role_config.provider
                self._role_fallback_state[role_id] = role_config.provider
                self._oauth_cooldown_fallback_roles.discard(role_id)
        
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
                    raise ProviderRepairRequiredError(
                        provider="openrouter",
                        provider_label="OpenRouter",
                        role_id=role_id,
                        model=role_config.openrouter_model_id or role_config.model_id,
                        reason="missing_api_key",
                        message=error_msg,
                        terminal_guidance=(
                            "Set a valid OpenRouter API key, change this role's provider, "
                            "or configure an LM Studio fallback."
                        ),
                        configured_provider=role_config.provider,
                        configured_model=role_config.openrouter_model_id or role_config.model_id,
                        configured_host_provider=role_config.openrouter_provider or "",
                        effective_host_provider=role_config.openrouter_provider or "",
                        route_kind="primary",
                        task_id=task_id,
                    )
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
                        await self._broadcast("account_credits_exhausted", _merge_route_fields({
                            "role_id": role_id,
                            "model": openrouter_model,
                            "provider": "openrouter",
                            "message": "OpenRouter account free credits depleted. Add credits at openrouter.ai or configure LM Studio fallback."
                        }, self.route_identity_for_role(
                            role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            route_kind="primary",
                            task_id=task_id,
                            configured_host_provider=openrouter_provider or "",
                        )))
                        raise FreeModelExhaustedError(
                            f"Account free credits exhausted and no LM Studio fallback for role '{role_id}'.",
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            route_kind="primary",
                            task_id=task_id,
                            configured_host_provider=openrouter_provider or "",
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
                            retry_callback=self._provider_retry_callback(
                                role_id,
                                task_id=task_id,
                                configured_model=openrouter_model,
                                configured_provider=role_config.provider,
                                configured_host_provider=openrouter_provider or "",
                            ),
                        ),
                        role_id=role_id,
                        model=openrouter_model,
                        provider=openrouter_provider or "OpenRouter",
                        provider_id="openrouter",
                        route_kind="primary",
                        task_id=task_id,
                        configured_host_provider=openrouter_provider or "",
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
                    self._clear_retryable_provider_backoff("openrouter", role_id, openrouter_model)
                    
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
                    
                    await self._broadcast("openrouter_rate_limit", _merge_route_fields({
                        "model": openrouter_model,
                        "provider": "openrouter",
                        "role_id": role_id,
                        "message": f"OpenRouter rate limit hit for '{openrouter_model}' after retries exhausted."
                    }, self.route_identity_for_role(
                        role_id,
                        error=e,
                        model=openrouter_model,
                        provider="openrouter",
                        route_kind="primary",
                        task_id=task_id,
                        configured_host_provider=openrouter_provider or "",
                    )))
                    
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
                        has_lm_studio_fallback=bool(role_config and role_config.lm_studio_fallback_id),
                    )
                    if rotated_result is not None:
                        free_model_manager.clear_failed_models()  # Success - clear failures
                        return rotated_result
                    
                    # Rotation chain exhausted — try LM Studio fallback
                    if not role_config.lm_studio_fallback_id:
                        rotation_ran = (
                            free_model_manager.looping_enabled
                            or free_model_manager.auto_selector_enabled
                        )
                        raise FreeModelExhaustedError(
                            f"All free model options exhausted for role '{role_id}'. "
                            f"No LM Studio fallback configured.",
                            role_id=role_id,
                            model=openrouter_model,
                            provider="openrouter",
                            route_kind=(
                                ("free_rotation" if free_model_manager.looping_enabled else "auto_selector")
                                if rotation_ran
                                else "primary"
                            ),
                            task_id=task_id,
                            configured_host_provider=(
                                "" if rotation_ran else openrouter_provider or ""
                            ),
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
                    await self._broadcast("openrouter_privacy_error", _merge_route_fields({
                        "error_type": "privacy_policy",
                        "model": openrouter_model,
                        "provider": "openrouter",
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
                    }, self.route_identity_for_role(
                        role_id,
                        error=e,
                        model=openrouter_model,
                        provider="openrouter",
                        route_kind="primary",
                        task_id=task_id,
                        configured_host_provider=openrouter_provider or "",
                    )))
                    
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
                        raise ProviderRepairRequiredError(
                            provider="openrouter",
                            provider_label="OpenRouter",
                            role_id=role_id,
                            model=openrouter_model,
                            reason="privacy_policy_required",
                            message=error_msg,
                            terminal_guidance=(
                                "Update OpenRouter privacy settings, select another provider, "
                                "or configure an LM Studio fallback."
                            ),
                            configured_provider=role_config.provider,
                            configured_model=openrouter_model,
                            configured_host_provider=openrouter_provider or "",
                            effective_host_provider=openrouter_provider or "",
                            route_kind="primary",
                            task_id=task_id,
                        ) from e
                    
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
                            await self._broadcast("openrouter_fallback_failed", _merge_route_fields({
                                "role_id": role_id,
                                "model": openrouter_model,
                                "provider": "openrouter",
                                "reason": "no_fallback_configured",
                                "message": error_msg
                            }, self.route_identity_for_role(
                                role_id,
                                error=e,
                                model=openrouter_model,
                                provider="openrouter",
                                route_kind="primary",
                                task_id=task_id,
                                configured_host_provider=openrouter_provider or "",
                            )))
                        raise ProviderRepairRequiredError(
                            provider="openrouter",
                            provider_label="OpenRouter",
                            role_id=role_id,
                            model=openrouter_model,
                            reason="credit_exhaustion",
                            message=error_msg,
                            terminal_guidance=(
                                "Add OpenRouter credits, select another provider, "
                                "or configure an LM Studio fallback."
                            ),
                            configured_provider=role_config.provider,
                            configured_model=openrouter_model,
                            configured_host_provider=openrouter_provider or "",
                            effective_host_provider=openrouter_provider or "",
                            route_kind="primary",
                            task_id=task_id,
                        ) from e
                    
                    # Fallback IS configured - use it
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    
                    fallback_model = role_config.lm_studio_fallback_id
                    
                    logger.error(
                        f"OpenRouter credits exhausted for role '{role_id}'. "
                        f"Permanently falling back to LM Studio model: {fallback_model}"
                    )
                    
                    await self._broadcast("openrouter_fallback", _merge_route_fields({
                        "role_id": role_id,
                        "model": openrouter_model,
                        "provider": "openrouter",
                        "reason": "credit_exhaustion",
                        "message": "Credits exhausted, falling back to alternative model",
                        "fallback_model": fallback_model
                    }, self.route_identity_for_role(
                        role_id,
                        error=e,
                        model=openrouter_model,
                        provider="openrouter",
                        route_kind="primary",
                        task_id=task_id,
                        configured_host_provider=openrouter_provider or "",
                    )))
                    
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
                        if is_transient_model_call_error(e):
                            raise self._as_retryable_provider_error(
                                provider="openrouter",
                                provider_label="OpenRouter",
                                role_id=role_id,
                                model=openrouter_model,
                                error=e,
                                route_kind="primary",
                                task_id=task_id,
                                configured_provider=role_config.provider,
                                configured_model=openrouter_model,
                                configured_host_provider=openrouter_provider or "",
                            ) from e
                        if isinstance(e, (ProviderRepairRequiredError, ProviderContextLengthError)):
                            raise
                        if isinstance(e, ProviderRouteError):
                            raise self._as_provider_repair_error(
                                provider="openrouter",
                                provider_label="OpenRouter",
                                role_id=role_id,
                                model=openrouter_model,
                                error=e,
                            ) from e
                        raise
        
        subscription_client = get_api_key_subscription_client(fallback_state) if role_config else None
        if subscription_client is not None and role_config:
            subscription_provider = subscription_client.provider_id
            subscription_label = subscription_client.provider_label
            subscription_settings = subscription_client.settings
            subscription_model = role_config.model_id
            active_cooldown = self.get_provider_cooldown(subscription_provider)
            if active_cooldown:
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                        self._oauth_cooldown_fallback_roles.add(role_id)
                    await self._broadcast_provider_usage_limit(
                        self._cooldown_payload_for_role(
                            active_cooldown,
                            role_id=role_id,
                            model=subscription_model,
                            task_id=task_id,
                        ),
                        fallback_model=role_config.lm_studio_fallback_id,
                    )
                    logger.warning(
                        "%s cooldown active for role '%s'; using LM Studio fallback model %s until reset",
                        subscription_label,
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_provider_usage_limit(
                        self._cooldown_payload_for_role(
                            active_cooldown,
                            role_id=role_id,
                            model=subscription_model,
                            task_id=task_id,
                        )
                    )
                    raise ProviderCooldownError(
                        provider=subscription_provider,
                        provider_label=subscription_label,
                        role_id=role_id,
                        model=subscription_model,
                        resets_at=active_cooldown.get("cooldown_until") or active_cooldown.get("resets_at"),
                        resets_in_seconds=active_cooldown.get("resets_in_seconds"),
                        plan_type=str(active_cooldown.get("plan_type") or ""),
                        message=str(active_cooldown.get("message") or ""),
                    )
            start_time = time.time()
            use_subscription = not (
                model == role_config.lm_studio_fallback_id
                and self._role_fallback_state.get(role_id) == "lm_studio"
            )
            if use_subscription:
              try:
                logger.debug("Role %s using %s: %s", role_id, subscription_label, subscription_model)
                result = await self._with_hung_connection_watchdog(
                    subscription_client.generate_completion(
                        model=subscription_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self._effective_max_tokens(max_tokens, role_config.max_output_tokens, role_id),
                        response_format=response_format,
                        reasoning_effort=role_reasoning_effort,
                        tools=tools,
                        tool_choice=tool_choice,
                        retry_callback=self._provider_retry_callback(
                            role_id,
                            task_id=task_id,
                            configured_model=subscription_model,
                            configured_provider=role_config.provider,
                        ),
                    ),
                    role_id=role_id,
                    model=subscription_model,
                    provider=subscription_label,
                    provider_id=subscription_provider,
                    route_kind="primary",
                    task_id=task_id,
                )
                duration_ms = (time.time() - start_time) * 1000
                if not result.get("choices"):
                    logger.error(
                        "%s response missing 'choices' after %.0fms - %s",
                        subscription_label,
                        duration_ms,
                        _response_shape_for_logging(result),
                    )
                    raise ValueError(f"{subscription_label} response missing 'choices' after {duration_ms:.0f}ms")

                response_content = extract_response_text(result, context=task_id)
                tokens_used = None
                if result.get("usage"):
                    tokens_used = result["usage"].get("total_tokens")
                    _pt = result["usage"].get("prompt_tokens")
                    _ct = result["usage"].get("completion_tokens")
                    if _pt is not None and _ct is not None:
                        token_tracker.track(subscription_model, _pt, _ct)
                        await self._broadcast("token_usage_updated", token_tracker.get_stats())

                result = self._annotate_response_with_call_metadata(
                    result,
                    task_id=task_id,
                    role_id=role_id,
                    configured_model=requested_model,
                    actual_model=subscription_model,
                    configured_provider=role_config.provider,
                    actual_provider=subscription_provider,
                    boosted=False,
                    boost_mode=None,
                    openrouter_reasoning_effort=role_reasoning_effort,
                )

                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=subscription_model,
                        provider=subscription_provider,
                        prompt=full_prompt,
                        response=response_content,
                        tokens_used=tokens_used,
                        duration_ms=duration_ms,
                        success=True,
                        error=None,
                        phase=self._current_autonomous_phase,
                    )

                await self._track_model_usage(subscription_model)
                self._clear_retryable_provider_backoff(subscription_provider, role_id, subscription_model)
                recent_cooldown = self._provider_recent_expired_cooldowns.get(subscription_provider)
                if recent_cooldown is not None:
                    self._provider_resume_pending.setdefault(
                        (subscription_provider, role_id, subscription_model),
                        {
                            **recent_cooldown,
                            "provider": subscription_provider,
                            "provider_label": subscription_label,
                            "role_id": role_id,
                            "model": subscription_model,
                            "workflow_mode": _retry_workflow_mode(role_id),
                        },
                    )
                await self._confirm_provider_usage_limit_resumed(
                    provider=subscription_provider,
                    provider_label=subscription_label,
                    role_id=role_id,
                    model=subscription_model,
                    task_id=task_id,
                )
                return result
              except SubscriptionUsageLimitError as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=subscription_model,
                        provider=subscription_provider,
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=oauth_live_activity_error_message(e),
                        phase=self._current_autonomous_phase,
                    )
                cooldown_payload = self._mark_provider_cooldown(
                    e,
                    role_id=role_id,
                    model=subscription_model,
                    task_id=task_id,
                )
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                        self._oauth_cooldown_fallback_roles.add(role_id)
                    await self._broadcast_provider_usage_limit(
                        cooldown_payload,
                        fallback_model=role_config.lm_studio_fallback_id,
                    )
                    logger.warning(
                        "%s usage limit reached for role '%s'; falling back to LM Studio model %s until provider reset",
                        subscription_label,
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_provider_usage_limit(cooldown_payload)
                    raise ProviderCooldownError(
                        provider=str(getattr(e, "provider", "") or subscription_provider),
                        provider_label=str(getattr(e, "provider_label", "") or subscription_label),
                        role_id=role_id,
                        model=subscription_model,
                        resets_at=cooldown_payload.get("cooldown_until") or e.resets_at,
                        resets_in_seconds=cooldown_payload.get("resets_in_seconds") or e.resets_in_seconds,
                        plan_type=e.plan_type,
                        message=str(cooldown_payload.get("message") or str(e)),
                    ) from e
              except SubscriptionEntitlementError as e:
                if (
                    role_config.lm_studio_fallback_id
                    and subscription_settings.entitlement_with_fallback == "fallback"
                ):
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "%s entitlement failed for role '%s'; falling back to LM Studio model %s",
                        subscription_label,
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_unrecoverable_subscription_error(
                        provider=subscription_provider,
                        role_id=role_id,
                        model=subscription_model,
                        error=e,
                        task_id=task_id,
                    )
                    raise self._as_provider_repair_error(
                        provider=subscription_provider,
                        provider_label=subscription_label,
                        role_id=role_id,
                        model=subscription_model,
                        error=e,
                        reason="entitlement_required",
                        terminal_guidance=e.terminal_guidance or (
                            f"Repair or upgrade the {subscription_label} subscription entitlement, "
                            "then retry the workflow."
                        ),
                    ) from e
              except SubscriptionProviderError as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=subscription_model,
                        provider=subscription_provider,
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=oauth_live_activity_error_message(e),
                        phase=self._current_autonomous_phase,
                    )
                if is_retryable_model_output_error(e):
                    raise
                if role_config.lm_studio_fallback_id:
                    if not is_transient_model_call_error(e):
                        async with self._state_lock:
                            self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "%s failed for role '%s'; falling back to LM Studio model %s",
                        subscription_label,
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    if _is_provider_context_length_error(e):
                        raise _typed_provider_context_error(
                            e,
                            provider=subscription_provider,
                            model=subscription_model,
                        ) from e
                    if is_transient_model_call_error(e):
                        raise self._as_retryable_provider_error(
                            provider=subscription_provider,
                            provider_label=subscription_label,
                            role_id=role_id,
                            model=subscription_model,
                            error=e,
                            route_kind="primary",
                            task_id=task_id,
                            configured_provider=role_config.provider,
                            configured_model=subscription_model,
                        ) from e
                    await self._broadcast_unrecoverable_subscription_error(
                        provider=subscription_provider,
                        role_id=role_id,
                        model=subscription_model,
                        error=e,
                        task_id=task_id,
                    )
                    raise self._as_provider_repair_error(
                        provider=subscription_provider,
                        provider_label=subscription_label,
                        role_id=role_id,
                        model=subscription_model,
                        error=e,
                    ) from e
              except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if self._autonomous_logger_callback:
                    full_prompt = self._prompt_for_logging(messages)
                    await self._autonomous_logger_callback(
                        task_id=task_id,
                        role_id=role_id,
                        model=subscription_model,
                        provider=subscription_provider,
                        prompt=full_prompt,
                        response="",
                        tokens_used=None,
                        duration_ms=duration_ms,
                        success=False,
                        error=oauth_live_activity_error_message(e),
                        phase=self._current_autonomous_phase,
                    )
                if is_retryable_model_output_error(e):
                    raise
                if role_config.lm_studio_fallback_id:
                    if (
                        subscription_settings.generic_error_fallback == "permanent"
                        or not is_transient_model_call_error(e)
                    ):
                        async with self._state_lock:
                            self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "%s error for role '%s': %s; falling back to LM Studio model %s",
                        subscription_label,
                        role_id,
                        e,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    if _is_provider_context_length_error(e):
                        raise _typed_provider_context_error(
                            e,
                            provider=subscription_provider,
                            model=subscription_model,
                        ) from e
                    if is_transient_model_call_error(e):
                        raise self._as_retryable_provider_error(
                            provider=subscription_provider,
                            provider_label=subscription_label,
                            role_id=role_id,
                            model=subscription_model,
                            error=e,
                            route_kind="primary",
                            task_id=task_id,
                            configured_provider=role_config.provider,
                            configured_model=subscription_model,
                        ) from e
                    await self._broadcast_unrecoverable_subscription_error(
                        provider=subscription_provider,
                        role_id=role_id,
                        model=subscription_model,
                        error=e,
                        task_id=task_id,
                    )
                    if isinstance(e, (ProviderRepairRequiredError, ProviderContextLengthError)):
                        raise
                    raise self._as_provider_repair_error(
                        provider=subscription_provider,
                        provider_label=subscription_label,
                        role_id=role_id,
                        model=subscription_model,
                        error=e,
                    ) from e

        if fallback_state == "openai_codex_oauth" and role_config:
            codex_model = role_config.model_id
            active_cooldown = self.get_provider_cooldown("openai_codex_oauth")
            if active_cooldown:
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                        self._oauth_cooldown_fallback_roles.add(role_id)
                    await self._broadcast_oauth_usage_limit(
                        self._cooldown_payload_for_role(
                            active_cooldown,
                            role_id=role_id,
                            model=codex_model,
                            task_id=task_id,
                        ),
                        fallback_model=role_config.lm_studio_fallback_id,
                    )
                    logger.warning(
                        "OpenAI Codex cooldown active for role '%s'; using LM Studio fallback model %s until reset",
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_oauth_usage_limit(
                        self._cooldown_payload_for_role(
                            active_cooldown,
                            role_id=role_id,
                            model=codex_model,
                            task_id=task_id,
                        )
                    )
                    raise OAuthProviderCooldownError(
                        provider="openai_codex_oauth",
                        provider_label="OpenAI Codex",
                        role_id=role_id,
                        model=codex_model,
                        resets_at=active_cooldown.get("cooldown_until") or active_cooldown.get("resets_at"),
                        resets_in_seconds=active_cooldown.get("resets_in_seconds"),
                        plan_type=str(active_cooldown.get("plan_type") or ""),
                        message=str(active_cooldown.get("message") or ""),
                    )
            start_time = time.time()
            use_codex = not (
                model == role_config.lm_studio_fallback_id
                and self._role_fallback_state.get(role_id) == "lm_studio"
            )
            if use_codex:
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
                            retry_callback=self._provider_retry_callback(
                                role_id,
                                task_id=task_id,
                                configured_model=codex_model,
                                configured_provider=role_config.provider,
                            ),
                        ),
                        role_id=role_id,
                        model=codex_model,
                        provider="OpenAI Codex",
                        provider_id="openai_codex_oauth",
                        route_kind="primary",
                        task_id=task_id,
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
                    self._clear_retryable_provider_backoff("openai_codex_oauth", role_id, codex_model)
                    recent_cooldown = self._provider_recent_expired_cooldowns.get(
                        "openai_codex_oauth"
                    )
                    if recent_cooldown is not None:
                        self._provider_resume_pending.setdefault(
                            ("openai_codex_oauth", role_id, codex_model),
                            {
                                **recent_cooldown,
                                "provider": "openai_codex_oauth",
                                "provider_label": "OpenAI Codex",
                                "role_id": role_id,
                                "model": codex_model,
                                "workflow_mode": _retry_workflow_mode(role_id),
                            },
                        )
                    await self._confirm_provider_usage_limit_resumed(
                        provider="openai_codex_oauth",
                        provider_label="OpenAI Codex",
                        role_id=role_id,
                        model=codex_model,
                        task_id=task_id,
                    )
                    return result

                except OAuthUsageLimitError as e:
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
                    cooldown_payload = self._mark_oauth_provider_cooldown(
                        e,
                        role_id=role_id,
                        model=codex_model,
                        task_id=task_id,
                    )
                    if role_config.lm_studio_fallback_id:
                        async with self._state_lock:
                            self._role_fallback_state[role_id] = "lm_studio"
                            self._oauth_cooldown_fallback_roles.add(role_id)
                        await self._broadcast_oauth_usage_limit(
                            cooldown_payload,
                            fallback_model=role_config.lm_studio_fallback_id,
                        )
                        logger.warning(
                            "OpenAI Codex usage limit reached for role '%s'; falling back to LM Studio model %s until provider reset",
                            role_id,
                            role_config.lm_studio_fallback_id,
                        )
                        model = role_config.lm_studio_fallback_id
                    else:
                        await self._broadcast_oauth_usage_limit(cooldown_payload)
                        raise OAuthProviderCooldownError(
                            provider=e.provider,
                            provider_label=e.provider_label,
                            role_id=role_id,
                            model=codex_model,
                            resets_at=cooldown_payload.get("cooldown_until") or e.resets_at,
                            resets_in_seconds=cooldown_payload.get("resets_in_seconds") or e.resets_in_seconds,
                            plan_type=e.plan_type,
                            message=str(cooldown_payload.get("message") or str(e)),
                        ) from e
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
                            error=oauth_live_activity_error_message(e),
                            phase=self._current_autonomous_phase,
                        )
                    if is_retryable_model_output_error(e):
                        raise
                    if isinstance(e, OpenAICodexUntimedUsageLimitError):
                        if role_config.lm_studio_fallback_id:
                            async with self._state_lock:
                                self._role_fallback_state[role_id] = "lm_studio"
                            logger.warning(
                                "OpenAI Codex usage limit has no reset for role '%s'; "
                                "using LM Studio model %s until manual provider reset",
                                role_id,
                                role_config.lm_studio_fallback_id,
                            )
                            model = role_config.lm_studio_fallback_id
                        else:
                            repair_error = self._as_provider_repair_error(
                                provider="openai_codex_oauth",
                                provider_label="OpenAI Codex",
                                role_id=role_id,
                                model=codex_model,
                                error=e,
                                reason="usage_limit_reached",
                                terminal_guidance=(
                                    "Wait for or restore the OpenAI Codex usage allowance, "
                                    "change this role's provider, or configure an LM Studio fallback."
                                ),
                            )
                            await self._broadcast_unrecoverable_codex_error(
                                role_id=role_id,
                                model=codex_model,
                                error=repair_error,
                                task_id=task_id,
                            )
                            raise repair_error from e
                    elif role_config.lm_studio_fallback_id:
                        if not (
                            is_transient_model_call_error(e)
                            or _is_retryable_codex_completion_error(e)
                        ):
                            async with self._state_lock:
                                self._role_fallback_state[role_id] = "lm_studio"
                        logger.warning(
                            "OpenAI Codex failed for role '%s'; falling back to LM Studio model %s",
                            role_id,
                            role_config.lm_studio_fallback_id,
                        )
                        model = role_config.lm_studio_fallback_id
                    else:
                        if _is_provider_context_length_error(e):
                            raise _typed_provider_context_error(
                                e,
                                provider="openai_codex_oauth",
                                model=codex_model,
                            ) from e
                        if (
                            is_transient_model_call_error(e)
                            or _is_retryable_codex_completion_error(e)
                        ):
                            raise self._as_retryable_provider_error(
                                provider="openai_codex_oauth",
                                provider_label="OpenAI Codex",
                                role_id=role_id,
                                model=codex_model,
                                error=e,
                                route_kind="primary",
                                task_id=task_id,
                                configured_provider=role_config.provider,
                                configured_model=codex_model,
                            ) from e
                        await self._broadcast_unrecoverable_codex_error(
                            role_id=role_id,
                            model=codex_model,
                            error=e,
                            task_id=task_id,
                        )
                        raise self._as_provider_repair_error(
                            provider="openai_codex_oauth",
                            provider_label="OpenAI Codex",
                            role_id=role_id,
                            model=codex_model,
                            error=e,
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
                            error=oauth_live_activity_error_message(e),
                            phase=self._current_autonomous_phase,
                        )
                    if is_retryable_model_output_error(e):
                        raise
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
                        if _is_provider_context_length_error(e):
                            raise _typed_provider_context_error(
                                e,
                                provider="openai_codex_oauth",
                                model=codex_model,
                            ) from e
                        if (
                            is_transient_model_call_error(e)
                            or _is_retryable_codex_completion_error(e)
                        ):
                            raise self._as_retryable_provider_error(
                                provider="openai_codex_oauth",
                                provider_label="OpenAI Codex",
                                role_id=role_id,
                                model=codex_model,
                                error=e,
                                route_kind="primary",
                                task_id=task_id,
                                configured_provider=role_config.provider,
                                configured_model=codex_model,
                            ) from e
                        await self._broadcast_unrecoverable_codex_error(
                            role_id=role_id,
                            model=codex_model,
                            error=e,
                            task_id=task_id,
                        )
                        if isinstance(e, (ProviderRepairRequiredError, ProviderContextLengthError)):
                            raise
                        raise self._as_provider_repair_error(
                            provider="openai_codex_oauth",
                            provider_label="OpenAI Codex",
                            role_id=role_id,
                            model=codex_model,
                            error=e,
                        ) from e

        if fallback_state == "xai_grok_oauth" and role_config:
            xai_model = role_config.model_id
            active_cooldown = self.get_provider_cooldown("xai_grok_oauth")
            if active_cooldown:
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                        self._oauth_cooldown_fallback_roles.add(role_id)
                    await self._broadcast_provider_usage_limit(
                        self._cooldown_payload_for_role(
                            active_cooldown,
                            role_id=role_id,
                            model=xai_model,
                            task_id=task_id,
                        ),
                        fallback_model=role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_provider_usage_limit(
                        self._cooldown_payload_for_role(
                            active_cooldown,
                            role_id=role_id,
                            model=xai_model,
                            task_id=task_id,
                        )
                    )
                    raise ProviderCooldownError(
                        provider="xai_grok_oauth",
                        provider_label="xAI Grok",
                        role_id=role_id,
                        model=xai_model,
                        resets_at=active_cooldown.get("cooldown_until") or active_cooldown.get("resets_at"),
                        resets_in_seconds=active_cooldown.get("resets_in_seconds"),
                        plan_type=str(active_cooldown.get("plan_type") or ""),
                        message=str(active_cooldown.get("message") or ""),
                    )
            start_time = time.time()
            use_xai = not (
                model == role_config.lm_studio_fallback_id
                and self._role_fallback_state.get(role_id) == "lm_studio"
            )
            if use_xai:
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
                        retry_callback=self._provider_retry_callback(
                            role_id,
                            task_id=task_id,
                            configured_model=xai_model,
                            configured_provider=role_config.provider,
                        ),
                    ),
                    role_id=role_id,
                    model=xai_model,
                    provider="xAI Grok",
                    provider_id="xai_grok_oauth",
                    route_kind="primary",
                    task_id=task_id,
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
                self._clear_retryable_provider_backoff("xai_grok_oauth", role_id, xai_model)
                recent_cooldown = self._provider_recent_expired_cooldowns.get("xai_grok_oauth")
                if recent_cooldown is not None:
                    self._provider_resume_pending.setdefault(
                        ("xai_grok_oauth", role_id, xai_model),
                        {
                            **recent_cooldown,
                            "provider": "xai_grok_oauth",
                            "provider_label": "xAI Grok",
                            "role_id": role_id,
                            "model": xai_model,
                            "workflow_mode": _retry_workflow_mode(role_id),
                        },
                    )
                await self._confirm_provider_usage_limit_resumed(
                    provider="xai_grok_oauth",
                    provider_label="xAI Grok",
                    role_id=role_id,
                    model=xai_model,
                    task_id=task_id,
                )
                return result

              except XAIGrokUsageLimitError as e:
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
                        error=oauth_live_activity_error_message(e),
                        phase=self._current_autonomous_phase,
                    )
                cooldown_payload = self._mark_provider_cooldown(
                    e,
                    role_id=role_id,
                    model=xai_model,
                    task_id=task_id,
                )
                if role_config.lm_studio_fallback_id:
                    async with self._state_lock:
                        self._role_fallback_state[role_id] = "lm_studio"
                        self._oauth_cooldown_fallback_roles.add(role_id)
                    await self._broadcast_provider_usage_limit(
                        cooldown_payload,
                        fallback_model=role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    await self._broadcast_provider_usage_limit(cooldown_payload)
                    raise ProviderCooldownError(
                        provider=e.provider,
                        provider_label=e.provider_label,
                        role_id=role_id,
                        model=xai_model,
                        resets_at=cooldown_payload.get("cooldown_until") or e.resets_at,
                        resets_in_seconds=(
                            cooldown_payload.get("resets_in_seconds")
                            or e.resets_in_seconds
                        ),
                        plan_type=e.plan_type,
                        message=str(cooldown_payload.get("message") or str(e)),
                    ) from e
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
                        error=oauth_live_activity_error_message(e),
                        phase=self._current_autonomous_phase,
                    )
                if is_retryable_model_output_error(e):
                    raise
                if role_config.lm_studio_fallback_id:
                    if not is_transient_model_call_error(e):
                        async with self._state_lock:
                            self._role_fallback_state[role_id] = "lm_studio"
                    logger.warning(
                        "xAI Grok failed for role '%s'; falling back to LM Studio model %s",
                        role_id,
                        role_config.lm_studio_fallback_id,
                    )
                    model = role_config.lm_studio_fallback_id
                else:
                    if isinstance(e, XAIGrokSpendingLimitError):
                        repair_error = ProviderRepairRequiredError(
                            provider="xai_grok_oauth",
                            provider_label="xAI Grok",
                            role_id=role_id,
                            model=xai_model,
                            reason="spending_limit_reached",
                            message=(
                                f"xAI Grok spending limit reached for role '{role_id}' "
                                "and no LM Studio fallback is configured."
                            ),
                            terminal_guidance=(
                                "Increase or restore the xAI Grok spending/subscription allowance, "
                                "change this role's provider, or configure an LM Studio fallback."
                            ),
                            configured_provider=role_config.provider,
                            configured_model=xai_model,
                            route_kind="primary",
                            task_id=task_id,
                        )
                        await self._broadcast_unrecoverable_xai_grok_error(
                            role_id=role_id,
                            model=xai_model,
                            error=repair_error,
                            task_id=task_id,
                        )
                        raise repair_error from e
                    if _is_provider_context_length_error(e):
                        raise _typed_provider_context_error(
                            e,
                            provider="xai_grok_oauth",
                            model=xai_model,
                        ) from e
                    if is_transient_model_call_error(e):
                        raise self._as_retryable_provider_error(
                            provider="xai_grok_oauth",
                            provider_label="xAI Grok",
                            role_id=role_id,
                            model=xai_model,
                            error=e,
                            route_kind="primary",
                            task_id=task_id,
                            configured_provider=role_config.provider,
                            configured_model=xai_model,
                        ) from e
                    await self._broadcast_unrecoverable_xai_grok_error(
                        role_id=role_id,
                        model=xai_model,
                        error=e,
                        task_id=task_id,
                    )
                    raise self._as_provider_repair_error(
                        provider="xai_grok_oauth",
                        provider_label="xAI Grok",
                        role_id=role_id,
                        model=xai_model,
                        error=e,
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
                        error=oauth_live_activity_error_message(e),
                        phase=self._current_autonomous_phase,
                    )
                if is_retryable_model_output_error(e):
                    raise
                if role_config.lm_studio_fallback_id:
                    if not is_transient_model_call_error(e):
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
                    if _is_provider_context_length_error(e):
                        raise _typed_provider_context_error(
                            e,
                            provider="xai_grok_oauth",
                            model=xai_model,
                        ) from e
                    if is_transient_model_call_error(e):
                        raise self._as_retryable_provider_error(
                            provider="xai_grok_oauth",
                            provider_label="xAI Grok",
                            role_id=role_id,
                            model=xai_model,
                            error=e,
                            route_kind="primary",
                            task_id=task_id,
                            configured_provider=role_config.provider,
                            configured_model=xai_model,
                        ) from e
                    await self._broadcast_unrecoverable_xai_grok_error(
                        role_id=role_id,
                        model=xai_model,
                        error=e,
                        task_id=task_id,
                    )
                    if isinstance(e, (ProviderRepairRequiredError, ProviderContextLengthError)):
                        raise
                    raise self._as_provider_repair_error(
                        provider="xai_grok_oauth",
                        provider_label="xAI Grok",
                        role_id=role_id,
                        model=xai_model,
                        error=e,
                    ) from e

        if (
            fallback_state == "lm_studio"
            and role_config.provider != "lm_studio"
            and role_config.lm_studio_fallback_id
        ):
            model = role_config.lm_studio_fallback_id

        lm_studio_route_kind = "fallback" if role_config.provider != "lm_studio" else "primary"
        if system_config.generic_mode:
            raise ProviderRepairRequiredError(
                provider="lm_studio",
                provider_label="LM Studio",
                role_id=role_id,
                model=model,
                reason="provider_unavailable_in_generic_mode",
                message=(
                    f"Generic mode is OpenRouter-only; role '{role_id}' cannot use LM Studio."
                ),
                terminal_guidance=(
                    "Configure the role with provider='openrouter' and a valid "
                    "OpenRouter model and key."
                ),
                configured_provider=role_config.provider,
                configured_model=_configured_role_model(role_config),
                route_kind=lm_studio_route_kind,
                task_id=task_id,
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
                    retry_callback=self._provider_retry_callback(
                        role_id,
                        route_kind=lm_studio_route_kind,
                        task_id=task_id,
                        configured_model=_configured_role_model(role_config),
                        configured_provider=role_config.provider,
                    ),
                    **kwargs
                ),
                role_id=role_id,
                model=model,
                provider="LM Studio",
                provider_id="lm_studio",
                route_kind=lm_studio_route_kind,
                task_id=task_id,
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
            if isinstance(e, (ProviderRepairRequiredError, ProviderContextLengthError)):
                raise
            if is_transient_model_call_error(e):
                raise self._as_retryable_provider_error(
                    provider="lm_studio",
                    provider_label="LM Studio",
                    role_id=role_id,
                    model=model,
                    error=e,
                    route_kind=lm_studio_route_kind,
                    task_id=task_id,
                    configured_provider=role_config.provider,
                    configured_model=_configured_role_model(role_config),
                ) from e
            raise self._as_provider_repair_error(
                provider="lm_studio",
                provider_label="LM Studio",
                role_id=role_id,
                model=model,
                error=e,
                reason="local_model_unavailable",
                terminal_guidance=(
                    "Start LM Studio, load the configured model with the configured "
                    "context window, or change this role's provider."
                ),
            ) from e
    
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
        has_lm_studio_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt free model rotation chain: looping -> auto-selector.
        Returns API result on success, None if all options exhausted.

        A transient failure of one rotation route moves on to the next route.
        If no route succeeds and the role has no LM Studio fallback, the last
        transient failure is raised as a route-identified retryable error.
        """
        if not self._openrouter_client:
            return None
        # (route-identified retryable error, the provider failure that caused it)
        last_transient_failure: Optional[tuple[Exception, BaseException]] = None

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
                await self._broadcast("free_model_rotated", _merge_route_fields({
                    "role_id": role_id,
                    "model": alt_model,
                    "provider": "openrouter",
                    "from_model": original_model,
                    "to_model": alt_model,
                    "reason": "rate_limit",
                    "configured_model": configured_model,
                    "configured_provider": configured_provider,
                }, self.route_identity_for_role(
                    role_id,
                    model=alt_model,
                    provider="openrouter",
                    route_kind="free_rotation",
                    task_id=task_id,
                    configured_host_provider="",
                )))
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
                            retry_callback=self._provider_retry_callback(
                                role_id,
                                route_kind="free_rotation",
                                task_id=task_id,
                                configured_model=configured_model,
                                configured_provider=configured_provider,
                                configured_host_provider="",
                            ),
                        ),
                        role_id=role_id,
                        model=alt_model,
                        provider="OpenRouter (free rotation)",
                        provider_id="openrouter",
                        route_kind="free_rotation",
                        task_id=task_id,
                        configured_host_provider="",
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
                    self._clear_retryable_provider_backoff("openrouter", role_id, alt_model)
                    return result
                except RateLimitError:
                    free_model_manager.mark_model_failed(alt_model)
                    logger.warning(f"Rotated model {alt_model} also rate-limited, trying next")
                except CreditExhaustionError as inner_e:
                    logger.warning(f"Rotated model {alt_model} credit exhaustion: {inner_e}")
                    break
                except OpenRouterPrivacyPolicyError:
                    raise
                except ProviderContextLengthError as inner_e:
                    raise inner_e.with_route_context(
                        role_id=role_id,
                        task_id=task_id,
                        route_kind="free_rotation",
                        configured_provider=configured_provider,
                        configured_model=configured_model,
                    ) from inner_e
                except ProviderRouteError as inner_e:
                    routed = self._free_model_route_failure(
                        inner_e,
                        role_id=role_id,
                        task_id=task_id,
                        route_kind="free_rotation",
                        configured_provider=configured_provider,
                        configured_model=configured_model,
                    )
                    if not isinstance(routed, RetryableProviderError):
                        raise routed from inner_e
                    free_model_manager.mark_model_failed(alt_model)
                    last_transient_failure = (routed, inner_e)
                    if isinstance(inner_e.cause, httpx.ConnectError):
                        # OpenRouter itself is unreachable, so every other free
                        # model would fail the same way.
                        logger.warning(
                            "Rotated model %s could not reach OpenRouter; ending free model looping",
                            alt_model,
                        )
                        break
                    logger.warning(
                        "Rotated model %s transient provider failure, trying next: %s",
                        alt_model,
                        inner_e,
                    )
                except OpenRouterNoEndpointsError as inner_e:
                    free_model_manager.mark_model_failed(alt_model)
                    logger.warning(
                        "Rotated model %s has no available endpoint, trying next: %s",
                        alt_model,
                        inner_e,
                    )
                except (OpenRouterInvalidResponseError, ValueError) as inner_e:
                    free_model_manager.mark_model_failed(alt_model)
                    logger.warning(
                        "Rotated model %s failed provider call, trying next: %s",
                        alt_model,
                        inner_e,
                    )

        # Step 2: Auto-Selector Backup — try openrouter/free
        if free_model_manager.auto_selector_enabled:
            auto_model = free_model_manager.AUTO_SELECTOR_MODEL
            logger.info(f"Trying auto-selector '{auto_model}' for role {role_id}")
            await self._broadcast("free_model_auto_selector_used", _merge_route_fields({
                "role_id": role_id,
                "model": auto_model,
                "provider": "openrouter",
                "original_model": original_model,
                "configured_model": configured_model,
                "configured_provider": configured_provider,
            }, self.route_identity_for_role(
                role_id,
                model=auto_model,
                provider="openrouter",
                route_kind="auto_selector",
                task_id=task_id,
                configured_host_provider="",
            )))
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
                        retry_callback=self._provider_retry_callback(
                            role_id,
                            route_kind="auto_selector",
                            task_id=task_id,
                            configured_model=configured_model,
                            configured_provider=configured_provider,
                            configured_host_provider="",
                        ),
                    ),
                    role_id=role_id,
                    model=auto_model,
                    provider="OpenRouter (auto-selector)",
                    provider_id="openrouter",
                    route_kind="auto_selector",
                    task_id=task_id,
                    configured_host_provider="",
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
                self._clear_retryable_provider_backoff("openrouter", role_id, auto_model)
                return result
            except ProviderContextLengthError as inner_e:
                raise inner_e.with_route_context(
                    role_id=role_id,
                    task_id=task_id,
                    route_kind="auto_selector",
                    configured_provider=configured_provider,
                    configured_model=configured_model,
                ) from inner_e
            except ProviderRouteError as inner_e:
                routed = self._free_model_route_failure(
                    inner_e,
                    role_id=role_id,
                    task_id=task_id,
                    route_kind="auto_selector",
                    configured_provider=configured_provider,
                    configured_model=configured_model,
                )
                if not isinstance(routed, RetryableProviderError):
                    raise routed from inner_e
                last_transient_failure = (routed, inner_e)
                logger.warning(
                    "Auto-selector '%s' transient provider failure: %s",
                    auto_model,
                    inner_e,
                )
            except (RateLimitError, CreditExhaustionError) as inner_e:
                logger.warning(f"Auto-selector '{auto_model}' also failed: {inner_e}")

        # With an LM Studio fallback the caller uses it; otherwise the last
        # transient route failure is retried by the workflow under its own route.
        if last_transient_failure is not None and not has_lm_studio_fallback:
            failure, cause = last_transient_failure
            raise failure from cause
        return None

    def _free_model_route_failure(
        self,
        error: ProviderRouteError,
        *,
        role_id: str,
        task_id: str,
        route_kind: str,
        configured_provider: str,
        configured_model: str,
    ) -> Exception:
        """Attach free-model route context; type transient failures for workflow retry."""
        routed = error.with_route_context(
            role_id=role_id,
            task_id=task_id,
            route_kind=route_kind,
            configured_provider=configured_provider,
            configured_model=configured_model,
        )
        if not is_transient_model_call_error(routed):
            return routed
        return self._as_retryable_provider_error(
            provider="openrouter",
            provider_label="OpenRouter",
            role_id=role_id,
            model=routed.route.model,
            error=routed,
            route_kind=route_kind,
            task_id=task_id,
            configured_provider=configured_provider,
            configured_model=configured_model,
            configured_host_provider="",
        )

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

    async def reset_provider_fallbacks(self, provider: str) -> Dict[str, str]:
        """Restore configured roles after a provider credential or service repair."""
        provider = str(provider or "").strip()
        if not provider:
            return {}
        reset_roles: Dict[str, str] = {}
        async with self._state_lock:
            self._oauth_provider_cooldowns.pop(provider, None)
            for role_id, config in self._role_model_configs.items():
                if config.provider != provider:
                    continue
                if self._role_fallback_state.get(role_id) != provider:
                    self._role_fallback_state[role_id] = provider
                    reset_roles[role_id] = provider
                self._oauth_cooldown_fallback_roles.discard(role_id)
        if reset_roles:
            self._fallback_failed_notified.difference_update(reset_roles.keys())
            self._oauth_error_notified = {
                key
                for key in self._oauth_error_notified
                if not key.startswith(f"{provider}:")
            }
            await self._broadcast(
                "provider_fallbacks_reset",
                {
                    "provider": provider,
                    "reset_roles": list(reset_roles),
                    "message": (
                        f"Restored {len(reset_roles)} role(s) to their configured provider."
                    ),
                },
            )
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
            return await lm_studio_client.get_embeddings(
                texts,
                model,
                retry_callback=self._provider_retry_callback(
                    "embeddings",
                    route_kind="embeddings",
                ),
            )
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

