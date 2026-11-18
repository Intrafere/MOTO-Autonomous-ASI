"""Claude Messages API wire shared by the Claude API and Amazon Bedrock providers.

Both providers serve Claude through the same Messages API (Bedrock on its
``bedrock-mantle`` endpoint), so one wire converts MOTO's Chat Completions-shaped
request, streams the answer with the official Anthropic SDK, and returns the Chat
Completions shape the rest of MOTO consumes.

- SDK retries are off. Every retry wait goes through the client's shared
  stop-aware wait seam, like every other provider.
- Sampling parameters are never sent: current Claude models reject them, so
  MOTO's temperature ladders do not apply to Claude.
- Thinking is adaptive wherever the model supports it; the role's reasoning
  effort maps to ``output_config.effort`` and is never raised above the request.
- A tool-call turn is sent back with the exact blocks Claude produced (thinking
  blocks included), kept in memory only.
- A refusal is a model outcome, not a provider failure: it returns whatever text
  exists with finish reason ``content_filter``. No fallback model is requested, so
  the configured model is always the one that answers.

The SDK is imported on first use, off the event loop, so a MOTO process that
never touches a Claude provider does not need it and the first Claude call does
not stall the API while the import runs.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import json
import logging
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.subscription_providers.errors import (
    SubscriptionAuthError,
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionParameterRejectedError,
    SubscriptionProviderError,
    SubscriptionRequestError,
)
from backend.shared.subscription_providers.profile import ApiKeyProviderProfile

logger = logging.getLogger(__name__)

WIRE_NAME = "anthropic.messages.stream"
EFFORT_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {
    "maximum": "max",
    "highest": "max",
    "ultra": "max",
    "minimal": "low",
    "minimum": "low",
    "light": "low",
    # Several current Claude models cannot turn thinking off; the lowest effort is
    # the closest setting every model accepts.
    "none": "low",
}
_CONNECT_TIMEOUT_SECONDS = 30.0
_TOOL_TURN_CACHE_SIZE = 512
_TOOL_DEFINITION_CACHE_SIZE = 128
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
}
_CONTEXT_LIMIT_MARKERS = (
    "prompt is too long",
    "context window",
    "context length",
    "context limit",
    "input is too long",
    "too many tokens",
    "too many input tokens",
)
_BILLING_MARKERS = ("credit balance is too low", "billing", "spend limit", "payment")
_FAMILY_SUFFIX = re.compile(r"(@.*|-v\d+(:\d+)?|-\d{8})$")
# An error reported inside a stream arrives after HTTP 200, so its documented
# status comes from the error type.
_ERROR_TYPE_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}
_UNREADABLE_STREAM = "the response stream contained unreadable data, so the answer could not be read"


@dataclass(frozen=True)
class ClaudeModelTraits:
    """What MOTO needs to know to shape a request for one Claude model."""

    name: str
    # None means the value is not published; MOTO never invents one.
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # Supported ``output_config.effort`` levels; empty means the model takes none.
    efforts: Tuple[str, ...] = ()
    adaptive_thinking: bool = False


_FULL = EFFORT_LEVELS
_NO_XHIGH = ("low", "medium", "high", "max")

# Documented limits and request surface for current Claude models, keyed by the
# first-party model id. The Claude API's own model metadata overrides these.
CLAUDE_MODEL_TRAITS: Dict[str, ClaudeModelTraits] = {
    "claude-fable-5-2": ClaudeModelTraits("Claude Fable 5.2", 1_000_000, 128_000, _FULL, True),
    "claude-mythos-5-2": ClaudeModelTraits("Claude Mythos 5.2", 1_000_000, 128_000, _FULL, True),
    "claude-fable-5-1": ClaudeModelTraits("Claude Fable 5.1", 1_000_000, 128_000, _FULL, True),
    "claude-mythos-5-1": ClaudeModelTraits("Claude Mythos 5.1", 1_000_000, 128_000, _FULL, True),
    "claude-fable-5": ClaudeModelTraits("Claude Fable 5", 1_000_000, 128_000, _FULL, True),
    "claude-mythos-5": ClaudeModelTraits("Claude Mythos 5", 1_000_000, 128_000, _FULL, True),
    "claude-opus-5-5": ClaudeModelTraits("Claude Opus 5.5", 1_000_000, 128_000, _FULL, True),
    "claude-opus-5": ClaudeModelTraits("Claude Opus 5", 1_000_000, 128_000, _FULL, True),
    "claude-opus-4-8": ClaudeModelTraits("Claude Opus 4.8", 1_000_000, 128_000, _FULL, True),
    "claude-opus-4-7": ClaudeModelTraits("Claude Opus 4.7", 1_000_000, 128_000, _FULL, True),
    "claude-opus-4-6": ClaudeModelTraits("Claude Opus 4.6", 1_000_000, 128_000, _NO_XHIGH, True),
    "claude-sonnet-5-5": ClaudeModelTraits("Claude Sonnet 5.5", 1_000_000, 128_000, _FULL, True),
    "claude-sonnet-5": ClaudeModelTraits("Claude Sonnet 5", 1_000_000, 128_000, _FULL, True),
    "claude-sonnet-4-6": ClaudeModelTraits("Claude Sonnet 4.6", 1_000_000, 128_000, _NO_XHIGH, True),
    "claude-haiku-5-5": ClaudeModelTraits("Claude Haiku 5.5", 1_000_000, 128_000, _FULL, True),
    # Haiku 4.5 has no effort setting and no adaptive thinking.
    "claude-haiku-4-5": ClaudeModelTraits("Claude Haiku 4.5", 200_000, 64_000, (), False),
}


def claude_family_id(model_id: str) -> str:
    """Return the first-party Claude id inside a provider model id, or ``""``.

    ``global.anthropic.claude-opus-5-5`` and ``claude-haiku-4-5-20251001`` both
    resolve to their bare first-party id.
    """
    lowered = str(model_id or "").strip().lower()
    start = lowered.find("claude-")
    if start < 0:
        return ""
    family = lowered[start:]
    while True:
        trimmed = _FAMILY_SUFFIX.sub("", family)
        if trimmed == family:
            return family
        family = trimmed


def is_claude_model(model_id: str) -> bool:
    return bool(claude_family_id(model_id))


def known_claude_traits(model_id: str) -> Optional[ClaudeModelTraits]:
    return CLAUDE_MODEL_TRAITS.get(claude_family_id(model_id))


def resolve_claude_effort(requested: Optional[str], traits: Optional[ClaudeModelTraits]) -> Optional[str]:
    """Return the effort to send, or None to use the model's own default.

    An explicit request is never raised: a level the model lacks falls to the
    highest supported level below it.
    """
    value = (requested or "").strip().lower()
    value = _EFFORT_ALIASES.get(value, value)
    if value not in EFFORT_LEVELS:
        return None
    if traits is None:
        return value
    if value in traits.efforts:
        return value
    lower = [level for level in EFFORT_LEVELS[: EFFORT_LEVELS.index(value)] if level in traits.efforts]
    return lower[-1] if lower else None


def claude_model_record(
    provider_id: str,
    model_id: str,
    traits: Optional[ClaudeModelTraits],
    *,
    name: str = "",
) -> Dict[str, Any]:
    """Build the model record the settings UI reads for one Claude model."""
    efforts = list(traits.efforts) if traits else []
    record: Dict[str, Any] = {
        "id": model_id,
        "name": name or (traits.name if traits else model_id),
        "pricing": {"prompt": "usage", "completion": "usage"},
        "provider_metadata": {
            "source": provider_id,
            "supports_reasoning_effort": bool(efforts),
        },
    }
    if efforts:
        record["provider_metadata"]["reasoning_efforts"] = ["auto", *reversed(efforts)]
    if traits and traits.context_length:
        record["context_length"] = traits.context_length
    if traits and traits.max_output_tokens:
        record["max_output_tokens"] = traits.max_output_tokens
    return record


# --- request conversion ---------------------------------------------------------
def _text_of(content: Any) -> str:
    """Return message content as text without dropping any part of it."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif part is not None:
                parts.append(part if isinstance(part, str) else json.dumps(part, ensure_ascii=False))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _text_blocks(content: Any) -> List[Dict[str, Any]]:
    text = _text_of(content)
    # The Messages API rejects empty and whitespace-only text blocks.
    return [{"type": "text", "text": text}] if text.strip() else []


def _tool_call_ids(message: Dict[str, Any]) -> List[str]:
    return [
        str(call["id"])
        for call in message.get("tool_calls") or []
        if isinstance(call, dict) and call.get("id")
    ]


def _tool_use_blocks(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            arguments = {}
        blocks.append({
            "type": "tool_use",
            "id": str(call.get("id") or ""),
            "name": str(function.get("name") or ""),
            "input": arguments if isinstance(arguments, dict) else {"value": arguments},
        })
    return blocks


def _append_turn(turns: List[Dict[str, Any]], role: str, blocks: List[Any]) -> None:
    if not blocks:
        return
    if turns and turns[-1]["role"] == role:
        turns[-1]["content"].extend(blocks)
    else:
        turns.append({"role": role, "content": list(blocks)})


def convert_messages(
    messages: List[Dict[str, Any]],
    replayed_turns: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert Chat Completions messages to a Messages API system prompt and turns.

    ``replayed_turns`` maps a tool-call id to the exact content blocks Claude
    produced for that assistant turn; a match is sent back unchanged.
    """
    system_parts: List[str] = []
    turns: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role in {"system", "developer"}:
            if not turns:
                text = _text_of(message.get("content"))
                if text.strip():
                    system_parts.append(text)
            else:
                # A later operator instruction stays in order as a user turn, which
                # every Claude model accepts.
                _append_turn(turns, "user", _text_blocks(message.get("content")))
        elif role == "tool":
            result: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": str(message.get("tool_call_id") or ""),
            }
            text = _text_of(message.get("content"))
            if text.strip():
                result["content"] = text
            _append_turn(turns, "user", [result])
        elif role == "assistant":
            replay = next(
                (replayed_turns[call_id] for call_id in _tool_call_ids(message) if call_id in replayed_turns),
                None,
            ) if replayed_turns else None
            if replay is not None:
                _append_turn(turns, "assistant", replay)
            else:
                _append_turn(turns, "assistant", [*_text_blocks(message.get("content")), *_tool_use_blocks(message)])
        else:
            _append_turn(turns, "user", _text_blocks(message.get("content")))
    return "\n\n".join(system_parts), turns


def convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        entry: Dict[str, Any] = {
            "name": str(function["name"]),
            "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
        }
        if function.get("description"):
            entry["description"] = str(function["description"])
        converted.append(entry)
    return converted


def _history_tool_names(turns: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for turn in turns:
        for block in turn["content"]:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
            if block_type != "tool_use":
                continue
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
            if name and name not in names:
                names.append(str(name))
    return names


def message_to_chat_completion(message: Any, model: str) -> Dict[str, Any]:
    """Normalize an SDK ``Message`` to the Chat Completions shape MOTO consumes."""
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in message.content or []:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            text_parts.append(block.text or "")
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input or {})},
            })
    assistant: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        assistant["tool_calls"] = tool_calls

    usage = message.usage
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_written
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    normalized_usage: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cache_read or cache_written:
        normalized_usage["prompt_tokens_details"] = {
            "cached_tokens": cache_read,
            "cache_write_tokens": cache_written,
        }
    return {
        "id": str(message.id or f"claude-{int(time.time() * 1000)}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(message.model or model),
        "choices": [{
            "index": 0,
            "message": assistant,
            "finish_reason": _FINISH_REASONS.get(str(message.stop_reason or ""), "stop"),
        }],
        "usage": normalized_usage,
        "_moto_wire_api": WIRE_NAME,
    }


# --- shared profile behavior ------------------------------------------------------
class ClaudeFamilyProfile(ApiKeyProviderProfile):
    """Request shaping and error typing shared by providers that serve Claude."""

    # Plain-language repair advice; subclasses name their own console and terms.
    auth_guidance = ""
    billing_guidance = ""
    permission_guidance = ""
    model_guidance = ""
    retention_guidance = (
        "This model requires data retention to be enabled for your organization or workspace. "
        "Select a different model for this role, or change the retention setting with the provider."
    )

    def __init__(self, settings: Any) -> None:
        super().__init__(settings)
        # Model metadata reported by the provider itself, keyed by exact model id.
        self.live_traits: Dict[str, ClaudeModelTraits] = {}

    def traits_for(self, model: str) -> Optional[ClaudeModelTraits]:
        return self.live_traits.get(str(model or "").strip()) or known_claude_traits(model)

    def unsupported_model_guidance(self, model: str) -> str:
        """Return repair advice when this provider cannot serve the model on this wire."""
        return ""

    def finalize_model_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill in documented Claude metadata the provider's own model list omits."""
        finalized: List[Dict[str, Any]] = []
        for record in records:
            model_id = str(record.get("id") or "")
            traits = self.traits_for(model_id) if is_claude_model(model_id) else None
            if traits is None:
                finalized.append({**record, "pricing": {"prompt": "usage", "completion": "usage"}})
                continue
            listed_name = str(record.get("name") or "")
            known = claude_model_record(
                self.provider_id,
                model_id,
                traits,
                name=listed_name if listed_name and listed_name != model_id else "",
            )
            merged = {**record, **known}
            # Limits the provider itself reported win over the documented table.
            for field_name in ("context_length", "max_output_tokens"):
                if record.get(field_name):
                    merged[field_name] = record[field_name]
            finalized.append(merged)
        return finalized

    def normalize_reasoning_effort(self, reasoning_effort: Optional[str], *, model: str) -> Optional[str]:
        if not is_claude_model(model):
            return None
        return resolve_claude_effort(reasoning_effort, self.traits_for(model))

    @property
    def parameter_guidance(self) -> str:
        return (
            f"MOTO sent a request setting that {self.label} does not accept for this model. This is a MOTO "
            f"compatibility problem, not a problem with your {self.label} API key or account. Switch this role "
            "to another model or provider in Settings for now, and report the setting named in the error so "
            "MOTO can be updated."
        )

    def classify_status(
        self,
        status: Optional[int],
        error_type: str,
        message: str,
    ) -> Optional[SubscriptionProviderError]:
        """Return a typed, non-retried error for a definitive rejection, else None."""
        lowered = (message or "").lower()
        kind = (error_type or "").strip().lower()
        safe = redact_log_text(sanitize_provider_error_text(message or ""))
        common = {"provider": self.provider_id, "provider_label": self.label, "status_code": status}

        if status == 401 or kind == "authentication_error":
            return SubscriptionAuthError(
                f"{self.label} rejected the API key: {safe}",
                error_code="authentication_error",
                terminal_guidance=self.auth_guidance,
                **common,
            )
        if status == 413 or (status == 400 and any(marker in lowered for marker in _CONTEXT_LIMIT_MARKERS)):
            return SubscriptionContextLimitError(
                f"{self.label} rejected the request: context_length_exceeded ({safe})",
                error_code="context_length_exceeded",
                **common,
            )
        if status == 402 or kind == "billing_error" or (
            status in {400, 403, 429} and any(marker in lowered for marker in _BILLING_MARKERS)
        ):
            return SubscriptionEntitlementError(
                f"{self.label} cannot serve this request until billing is fixed: {safe}",
                error_code="billing_error",
                terminal_guidance=self.billing_guidance,
                **common,
            )
        if status == 400 and "data retention" in lowered:
            return SubscriptionEntitlementError(
                f"{self.label} cannot serve this model: {safe}",
                error_code="data_retention_required",
                terminal_guidance=self.retention_guidance,
                **common,
            )
        if status == 403 or kind == "permission_error":
            return SubscriptionEntitlementError(
                f"{self.label} denied access to this request: {safe}",
                error_code="permission_denied",
                terminal_guidance=self.permission_guidance,
                **common,
            )
        if status == 404 or kind == "not_found_error":
            return SubscriptionEntitlementError(
                f"{self.label} cannot serve this model: {safe}",
                error_code="model_not_available",
                terminal_guidance=self.model_guidance,
                **common,
            )
        if status in {400, 422}:
            return SubscriptionParameterRejectedError(
                f"{self.label} rejected a request setting: {safe}",
                error_code="invalid_request_parameter",
                terminal_guidance=self.parameter_guidance,
                **common,
            )
        return None


# --- the wire ----------------------------------------------------------------------
def _load_sdk(label: str, provider_id: str) -> Tuple[Any, Any]:
    try:
        import anthropic
        import httpx2
    except ImportError as exc:
        raise SubscriptionRequestError(
            f"{label} needs the 'anthropic' Python package, which is not installed. "
            "Start MOTO with its launcher (or run pip install -r requirements.txt) and try again.",
            provider=provider_id,
            provider_label=label,
        ) from exc
    return anthropic, httpx2


async def _load_sdk_off_loop(label: str, provider_id: str) -> Tuple[Any, Any]:
    """Load the SDK without blocking the event loop; its first import takes seconds."""
    if "anthropic" in sys.modules and "httpx2" in sys.modules:
        return _load_sdk(label, provider_id)
    return await asyncio.to_thread(_load_sdk, label, provider_id)


def _error_fields(exc: Any) -> Tuple[Optional[int], str, str]:
    """Return (status, error type, provider message) from an SDK status error."""
    status = getattr(exc, "status_code", None)
    error_type = str(getattr(exc, "type", "") or "")
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        message = str(error.get("message") or "")
        error_type = error_type or str(error.get("type") or "")
    if status is None or status < 400:
        status = _ERROR_TYPE_STATUS.get(error_type.strip().lower(), status)
    return status, error_type, message or str(getattr(exc, "message", "") or exc)


class ClaudeMessagesWire:
    """Messages API transport for one ``SubscriptionApiKeyClient``."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._profile: ClaudeFamilyProfile = client.profile
        self._sdk: Any = None
        self._sdk_identity: Tuple[Any, ...] = ()
        # tool_use id -> the assistant turn's exact content blocks (memory only).
        self._tool_turns: "OrderedDict[str, List[Any]]" = OrderedDict()
        self._tool_definitions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    # --- SDK clients ---------------------------------------------------------------
    def _new_sdk(self, base_url: str, api_key: str) -> Any:
        from backend.shared.subscription_providers.api_key_client import moto_user_agent

        anthropic, _httpx2 = _load_sdk(self._client.provider_label, self._client.provider_id)
        self._client._remember_secret(api_key)
        return anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=self._profile.messages_base_url(base_url),
            # MOTO owns retries so every wait is visible and stop-aware.
            max_retries=0,
            # Long thinking phases are expected; only connecting is bounded.
            timeout=anthropic.Timeout(None, connect=_CONNECT_TIMEOUT_SECONDS),
            default_headers={"User-Agent": moto_user_agent()},
            http_client=self._client.messages_http_client,
        )

    async def _close_sdk(self, sdk: Any) -> None:
        # An injected HTTP client belongs to its owner and is never closed here.
        if sdk is not None and self._client.messages_http_client is None:
            await sdk.close()

    async def _session_sdk(self) -> Any:
        api_key = self._client._load_api_key()
        if not api_key:
            raise SubscriptionAuthError(
                f"{self._client.provider_label} API key is not configured.",
                provider=self._client.provider_id,
                provider_label=self._client.provider_label,
            )
        identity = (self._client.base_url, api_key, self._client.messages_http_client)
        if self._sdk is None or self._sdk_identity != identity:
            await self._close_sdk(self._sdk)
            self._sdk = self._new_sdk(self._client.base_url, api_key)
            self._sdk_identity = identity
        return self._sdk

    async def close(self) -> None:
        await self._close_sdk(self._sdk)
        self._sdk = None
        self._sdk_identity = ()

    # --- errors ----------------------------------------------------------------------
    def _classified(self, exc: Any) -> Optional[SubscriptionProviderError]:
        status, error_type, message = _error_fields(exc)
        classified = self._profile.classify_status(status, error_type, message)
        if classified is not None:
            classified.args = (self._client._scrub(str(classified)),)
        return classified

    def _status_error(self, exc: Any, action: str) -> SubscriptionProviderError:
        status, _error_type, message = _error_fields(exc)
        error_cls = SubscriptionAuthError if status in {401, 403} else SubscriptionRequestError
        return error_cls(
            f"{self._client.provider_label} {action} failed: {self._client._detail(message)}",
            provider=self._client.provider_id,
            provider_label=self._client.provider_label,
            status_code=status,
        )

    def _is_transient(self, exc: Any) -> bool:
        return _error_fields(exc)[0] in self._client.settings.transient_status_codes

    # --- model list --------------------------------------------------------------------
    def _traits_from_model_info(self, info: Any) -> ClaudeModelTraits:
        known = known_claude_traits(info.id)
        capabilities = getattr(info, "capabilities", None)
        effort = getattr(capabilities, "effort", None)
        thinking = getattr(capabilities, "thinking", None)
        if effort is not None:
            efforts = tuple(
                level for level in EFFORT_LEVELS
                if getattr(effort, "supported", False) and getattr(getattr(effort, level, None), "supported", False)
            )
        else:
            efforts = known.efforts if known else ()
        if thinking is not None:
            adaptive = bool(getattr(getattr(getattr(thinking, "types", None), "adaptive", None), "supported", False))
        else:
            adaptive = bool(known and known.adaptive_thinking)
        return ClaudeModelTraits(
            name=str(getattr(info, "display_name", "") or (known.name if known else info.id)),
            context_length=ApiKeyProviderProfile.coerce_positive_int(getattr(info, "max_input_tokens", None))
            or (known.context_length if known else None),
            max_output_tokens=ApiKeyProviderProfile.coerce_positive_int(getattr(info, "max_tokens", None))
            or (known.max_output_tokens if known else None),
            efforts=efforts,
            adaptive_thinking=adaptive,
        )

    async def list_models(self, base_url: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
        """List models through the Claude Models API, which also authenticates the key."""
        client = self._client
        if not api_key:
            raise SubscriptionAuthError(
                f"{client.provider_label} API key is not configured.",
                provider=client.provider_id,
                provider_label=client.provider_label,
            )
        anthropic, httpx2 = await _load_sdk_off_loop(client.provider_label, client.provider_id)
        max_retries = client.settings.max_retries
        for attempt in range(max_retries):
            sdk = self._new_sdk(base_url, api_key)
            try:
                traits_by_id: Dict[str, ClaudeModelTraits] = {}
                async for info in sdk.models.list(limit=100):
                    traits_by_id[str(info.id)] = self._traits_from_model_info(info)
            except (anthropic.APIConnectionError, httpx2.TransportError) as exc:
                detail = client._detail(str(exc) or repr(exc))
                if attempt >= max_retries - 1:
                    raise client._request_failed(detail) from exc
                await client._wait_before_retry(
                    attempt=attempt,
                    delay=client._retry_delay(attempt),
                    reason="connection_error",
                    error_summary=f"[{type(exc).__name__}] {detail}",
                    retry_model="",
                    retry_callback=None,
                )
                continue
            except anthropic.APIStatusError as exc:
                classified = self._classified(exc)
                if classified is not None:
                    raise classified from exc
                if not self._is_transient(exc):
                    raise self._status_error(exc, "model list") from exc
                detail = client._detail(_error_fields(exc)[2])
                if attempt >= max_retries - 1:
                    raise client._request_failed(f"HTTP {exc.status_code}: {detail}", status_code=exc.status_code) from exc
                await client._wait_before_retry(
                    attempt=attempt,
                    delay=client._retry_after_delay(exc.response, attempt),
                    reason="transient_http_error",
                    error_summary=detail,
                    retry_model="",
                    retry_callback=None,
                )
                continue
            except SubscriptionProviderError:
                raise
            except Exception as exc:
                # Anything else is a reply that was not a model list (for example a proxy page).
                raise SubscriptionRequestError(
                    f"{client.provider_label} model list failed before a valid model list was returned.",
                    provider=client.provider_id,
                    provider_label=client.provider_label,
                ) from exc
            finally:
                await self._close_sdk(sdk)
            if not traits_by_id:
                return client.known_model_records()
            if api_key == client._load_api_key():
                self._profile.live_traits = traits_by_id
            return [
                claude_model_record(client.provider_id, model_id, traits)
                for model_id, traits in traits_by_id.items()
            ]
        raise client._request_failed("no attempts remained")

    # --- completion ---------------------------------------------------------------------
    def _remember_tools(self, tools: List[Dict[str, Any]]) -> None:
        for tool in tools:
            self._tool_definitions[tool["name"]] = tool
            self._tool_definitions.move_to_end(tool["name"])
        while len(self._tool_definitions) > _TOOL_DEFINITION_CACHE_SIZE:
            self._tool_definitions.popitem(last=False)

    def _remember_tool_turn(self, message: Any) -> None:
        tool_ids = [block.id for block in message.content or [] if getattr(block, "type", "") == "tool_use"]
        for tool_id in tool_ids:
            self._tool_turns[tool_id] = list(message.content)
            self._tool_turns.move_to_end(tool_id)
        while len(self._tool_turns) > _TOOL_TURN_CACHE_SIZE:
            self._tool_turns.popitem(last=False)

    def build_request(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        reasoning_effort: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        replay_tool_turns: bool = True,
    ) -> Dict[str, Any]:
        """Return the ``messages.stream`` arguments for one MOTO completion request."""
        client = self._client
        traits = self._profile.traits_for(model)
        system, turns = convert_messages(messages, self._tool_turns if replay_tool_turns else None)

        output_cap = traits.max_output_tokens if traits else None
        limit = ApiKeyProviderProfile.coerce_positive_int(max_tokens) or output_cap
        if not limit:
            raise SubscriptionRequestError(
                f"{client.provider_label} requires a max output token setting for model {model}.",
                provider=client.provider_id,
                provider_label=client.provider_label,
            )
        request: Dict[str, Any] = {
            "model": model,
            # A configured value above the model's published cap would be rejected.
            "max_tokens": min(limit, output_cap) if output_cap else limit,
            "messages": turns,
        }
        if system:
            request["system"] = system
        if traits is not None and traits.adaptive_thinking:
            request["thinking"] = {"type": "adaptive"}

        output_config: Dict[str, Any] = {}
        effort = resolve_claude_effort(reasoning_effort, traits)
        if effort:
            output_config["effort"] = effort
        schema = (response_format or {}).get("json_schema") if isinstance(response_format, dict) else None
        if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
            output_config["format"] = {"type": "json_schema", "schema": schema["schema"]}
        if output_config:
            request["output_config"] = output_config

        claude_tools = convert_tools(tools)
        self._remember_tools(claude_tools)
        history_tools = _history_tool_names(turns)
        if claude_tools:
            request["tools"] = claude_tools
            if tool_choice == "none":
                request["tool_choice"] = {"type": "none"}
        elif history_tools:
            # The Messages API requires tool definitions whenever the history holds
            # tool turns; the caller offered no tools, so none may be called.
            request["tools"] = [
                self._tool_definitions.get(name) or {"name": name, "input_schema": {"type": "object"}}
                for name in history_tools
            ]
            request["tool_choice"] = {"type": "none"}
        if claude_tools or history_tools:
            # A tool loop resends a growing prefix; cache it instead of re-billing it.
            request["cache_control"] = {"type": "ephemeral"}
        return request

    async def generate(
        self,
        *,
        retry_callback: Any,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        reasoning_effort: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Any],
        temperature: float = 0.0,
        cache_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        client = self._client
        unsupported = self._profile.unsupported_model_guidance(model)
        if unsupported:
            raise SubscriptionEntitlementError(
                f"{client.provider_label} cannot serve model {model} through MOTO.",
                provider=client.provider_id,
                provider_label=client.provider_label,
                error_code="model_not_supported",
                terminal_guidance=unsupported,
            )
        anthropic, httpx2 = await _load_sdk_off_loop(client.provider_label, client.provider_id)
        build = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        request = self.build_request(**build)
        replaying = True
        max_retries = client.settings.max_retries
        for attempt in range(max_retries):
            sdk = await self._session_sdk()
            wait: Optional[Dict[str, Any]] = None
            message: Any = None
            # Built outside the guarded block so a request MOTO shaped wrongly is not
            # mistaken for a damaged reply.
            stream_manager = sdk.messages.stream(**request)
            try:
                async with stream_manager as stream:
                    message = await stream.get_final_message()
            except (anthropic.APIConnectionError, httpx2.TransportError) as exc:
                detail = client._detail(str(exc) or repr(exc))
                if attempt >= max_retries - 1:
                    raise client._request_failed(detail) from exc
                wait = {
                    "delay": client._retry_delay(attempt),
                    "reason": "connection_error",
                    "error_summary": f"[{type(exc).__name__}] {detail}",
                }
            except anthropic.APIStatusError as exc:
                status, _error_type, provider_message = _error_fields(exc)
                lowered = provider_message.lower()
                if replaying and status == 400 and "thinking" in lowered and "signature" in lowered:
                    # Replayed thinking no longer matches this conversation (for example
                    # after an earlier turn changed); answer without the saved blocks.
                    replaying = False
                    request = self.build_request(**build, replay_tool_turns=False)
                    continue
                classified = self._classified(exc)
                if classified is not None:
                    raise classified from exc
                if not self._is_transient(exc):
                    raise self._status_error(exc, "completion") from exc
                detail = client._detail(provider_message)
                if attempt >= max_retries - 1:
                    raise client._request_failed(f"HTTP {status}: {detail}", status_code=status) from exc
                wait = {
                    "delay": client._retry_after_delay(exc.response, attempt),
                    "reason": "transient_http_error",
                    "error_summary": detail,
                }
            except Exception as exc:
                # The reply was not a readable Messages stream (a damaged chunk, an
                # empty body, or a proxy page). Nothing usable arrived, so ask again.
                logger.warning(
                    "%s returned an unreadable response stream (%s); raw data omitted.",
                    client.provider_label,
                    type(exc).__name__,
                )
                if attempt >= max_retries - 1:
                    raise client._request_failed(_UNREADABLE_STREAM) from exc
                wait = {
                    "delay": client._retry_delay(attempt),
                    "reason": "stream_incomplete",
                    "error_summary": f"{_UNREADABLE_STREAM.capitalize()}.",
                }
            if wait is None and getattr(message, "stop_reason", None) is None:
                incomplete = "the response stream ended before the provider completed the answer"
                if attempt >= max_retries - 1:
                    raise client._request_failed(incomplete)
                wait = {
                    "delay": client._retry_delay(attempt),
                    "reason": "stream_incomplete",
                    "error_summary": f"{incomplete.capitalize()}.",
                }
            if wait is not None:
                await client._wait_before_retry(
                    attempt=attempt,
                    retry_model=model,
                    retry_callback=retry_callback,
                    **wait,
                )
                continue
            if message.stop_reason == "refusal":
                details = getattr(message, "stop_details", None)
                logger.warning(
                    "%s model %s declined a request (category: %s).",
                    client.provider_label,
                    model,
                    getattr(details, "category", None) or "unspecified",
                )
            self._remember_tool_turn(message)
            return message_to_chat_completion(message, model)
        raise client._request_failed("no attempts remained")
