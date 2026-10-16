"""Kimi Code membership profile (Moonshot AI's coding subscription).

Access method: third-party tools use an API key from the Kimi Code console and
must identify themselves with their real User-Agent; MOTO never presents itself
as another client. The key is not a Kimi Platform (pay-as-you-go) key.

Wire format: streamed OpenAI-compatible Chat Completions. Kimi documents
``reasoning_effort`` (low/high/max) and ``thinking: {type: disabled}`` there, and
requires assistant tool-call turns to carry back their ``reasoning_content``.

Errors are identified by message text. Usage-window errors (5-hour, weekly,
monthly) carry no reset time, so the reset is read from ``GET /usages``; without
a credible reset the failure requires operator repair instead of a guessed timer.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from backend.shared.log_redaction import redact_log_text
from backend.shared.openrouter_client import sanitize_provider_error_text
from backend.shared.subscription_providers.errors import (
    SubscriptionContextLimitError,
    SubscriptionEntitlementError,
    SubscriptionProviderError,
    SubscriptionUsageLimitError,
)
from backend.shared.subscription_providers.metadata import get_cloud_provider
from backend.shared.subscription_providers.profile import (
    ApiKeyProviderProfile,
    ApiKeyProviderSettings,
    EndpointRegion,
    KnownModel,
)

_EFFORTS = ("auto", "max", "high", "low", "none")
_HIGHSPEED_MODEL = "kimi-for-coding-highspeed"
_TOOL_REASONING_CACHE_SIZE = 512

KIMI_CODE_SETTINGS = ApiKeyProviderSettings(
    metadata=get_cloud_provider("kimi_code"),
    secret_name="kimi_code_api_key",
    # Deliberately not KIMI_API_KEY, which is the pay-as-you-go Kimi Platform key.
    env_var_names=("MOTO_KIMI_CODE_API_KEY", "KIMI_CODE_API_KEY"),
    regions=(
        EndpointRegion("international", "International (api.kimi.ai)", "https://api.kimi.ai/coding/v1"),
        EndpointRegion("china", "China (api.kimi.com)", "https://api.kimi.com/coding/v1"),
    ),
    base_url_env_var="MOTO_KIMI_CODE_BASE_URL",
    wire_api="chat_completions_stream",
    known_models=(
        KnownModel("k3", "Kimi K3", 1_048_576, None, _EFFORTS),
        KnownModel("k3-256k", "Kimi K3 (256K)", 262_144, None, _EFFORTS),
        KnownModel("kimi-for-coding", "Kimi for Coding (K2.8 Preview)", 1_048_576, None, _EFFORTS),
        KnownModel(_HIGHSPEED_MODEL, "Kimi for Coding HighSpeed (K2.7)", 262_144, None, ()),
    ),
    default_model="k3",
    console_url="https://www.kimi.com/code/",
    validation_paths=("/models", "/usages"),
    supports_usage=True,
    invalid_key_message=(
        "This key was not accepted by Kimi Code in either region. Use an API key from the "
        "Kimi Code console; a Kimi Platform (pay-as-you-go) key will not work here."
    ),
    # 402 is Kimi's temporary "unable to verify your membership benefits" response.
    transient_status_codes=frozenset({402, 408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}),
    entitlement_with_fallback="repair",
    generic_error_fallback="permanent",
)

# (message marker, usage window) for Kimi's 403 usage-limit responses.
_USAGE_WINDOW_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("5-hour usage limit", "five_hour"),
    ("weekly (7-day) usage limit", "weekly"),
    ("weekly usage limit", "weekly"),
    ("monthly usage limit", "monthly"),
)
_POOL_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    # (key fragment, window, label)
    ("5h", "five_hour", "5-hour window"),
    ("300", "five_hour", "5-hour window"),
    ("week", "weekly", "Weekly"),
    ("7d", "weekly", "Weekly"),
    ("month_code", "monthly", "Monthly (coding)"),
    ("month_total", "monthly", "Monthly (total)"),
    ("month", "monthly", "Monthly"),
)


class _KimiUsageWindowError(SubscriptionEntitlementError):
    """A usage-window rejection awaiting a reset time from ``/usages``."""

    def __init__(self, message: str, *, window: str, status_code: Optional[int]) -> None:
        self.window = window
        super().__init__(
            message,
            provider="kimi_code",
            provider_label="Kimi Code",
            status_code=status_code,
            error_code=f"usage_limit_{window}",
            terminal_guidance=(
                "Kimi Code reported a usage limit but did not report when it resets. "
                "Wait for the window to reset (check the Kimi Code console) or enable Extra Usage, "
                "then retry the workflow."
            ),
        )


class KimiCodeProfile(ApiKeyProviderProfile):
    def __init__(self, settings: ApiKeyProviderSettings) -> None:
        super().__init__(settings)
        # tool_call_id -> reasoning_content, kept in memory only so assistant
        # tool-call turns can be sent back exactly as Kimi requires.
        self._tool_reasoning: "OrderedDict[str, str]" = OrderedDict()

    # --- request shaping --------------------------------------------------------
    def normalize_reasoning_effort(self, reasoning_effort: Optional[str], *, model: str) -> Optional[str]:
        if model == _HIGHSPEED_MODEL:
            return None
        effort = (reasoning_effort or "").strip().lower()
        if not effort or effort == "auto":
            return None
        if effort in {"max", "xhigh", "ultra", "maximum", "highest"}:
            return "max"
        if effort in {"high", "medium"}:
            return "high"
        if effort in {"low", "minimal", "minimum", "light"}:
            return "low"
        if effort == "none":
            return "none"
        return None

    def apply_reasoning(self, payload: Dict[str, Any], effort: Optional[str], *, model: str, wire: str) -> None:
        if effort == "none":
            payload["thinking"] = {"type": "disabled"}
        elif effort:
            payload["reasoning_effort"] = effort

    def prepare_chat_messages(
        self,
        messages: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not any(self._is_tool_call_turn(message) for message in messages):
            return messages
        prepared: List[Dict[str, Any]] = []
        missing_reasoning = False
        for message in messages:
            if not self._is_tool_call_turn(message) or message.get("reasoning_content"):
                prepared.append(message)
                continue
            reasoning = next(
                (self._tool_reasoning[call_id] for call_id in self._tool_call_ids(message) if call_id in self._tool_reasoning),
                None,
            )
            if reasoning is None:
                missing_reasoning = True
                prepared.append(message)
            else:
                prepared.append({**message, "reasoning_content": reasoning})
        if missing_reasoning:
            # Kimi rejects thinking-enabled tool follow-ups that lack the original
            # reasoning (for example after a restart); answer this turn without thinking.
            payload.pop("reasoning_effort", None)
            payload["thinking"] = {"type": "disabled"}
        return prepared

    def observe_chat_message(self, message: Dict[str, Any]) -> None:
        reasoning = message.get("reasoning_content")
        if not self._is_tool_call_turn(message) or not isinstance(reasoning, str) or not reasoning:
            return
        for call_id in self._tool_call_ids(message):
            self._tool_reasoning[call_id] = reasoning
            self._tool_reasoning.move_to_end(call_id)
        while len(self._tool_reasoning) > _TOOL_REASONING_CACHE_SIZE:
            self._tool_reasoning.popitem(last=False)

    @staticmethod
    def _is_tool_call_turn(message: Dict[str, Any]) -> bool:
        return str(message.get("role") or "") == "assistant" and bool(message.get("tool_calls"))

    @staticmethod
    def _tool_call_ids(message: Dict[str, Any]) -> Iterable[str]:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                yield str(call["id"])

    # --- error classification -----------------------------------------------------
    def _error_message(self, response: httpx.Response) -> str:
        payload = self.response_payload(response)
        for item in self.iter_error_objects(payload):
            message = item.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        return response.text or ""

    def is_transient_response(self, response: httpx.Response) -> bool:
        if response.status_code in self.settings.transient_status_codes:
            return True
        # Kimi's risk control reports too many in-flight requests as a 403.
        return response.status_code == 403 and "concurrent request limit" in self._error_message(response).lower()

    def classify_error(self, response: httpx.Response) -> Optional[SubscriptionProviderError]:
        raw_message = self._error_message(response)
        lowered = raw_message.lower()
        safe = redact_log_text(sanitize_provider_error_text(raw_message))
        status = response.status_code
        label = self.label

        if status == 403:
            for marker, window in _USAGE_WINDOW_MARKERS:
                if marker in lowered:
                    return _KimiUsageWindowError(f"{label} usage limit reached: {safe}", window=window, status_code=status)
            return None

        if status in {400, 413} and (
            "exceeded model token limit" in lowered or ("message size" in lowered and "exceeds limit" in lowered)
        ):
            return SubscriptionContextLimitError(
                f"{label} rejected the request: context_length_exceeded ({safe})",
                provider=self.provider_id,
                provider_label=label,
                status_code=status,
                error_code="context_length_exceeded",
            )

        entitlement_guidance = ""
        if "up to 256k context" in lowered:
            entitlement_guidance = (
                "Your Kimi Code plan limits K3 to 256K context. Select k3-256k, or set this role's "
                "context window to 262,144 tokens or less, or upgrade the plan."
            )
        elif "does not have access to" in lowered:
            entitlement_guidance = (
                "Your Kimi Code plan does not include this model. Select a model your plan includes "
                "(for example kimi-for-coding or k3-256k) or upgrade the plan."
            )
        elif "model id does not exist" in lowered or "not found the model" in lowered:
            entitlement_guidance = "Kimi Code did not recognize this model id. Select a listed Kimi Code model."
        if entitlement_guidance:
            return SubscriptionEntitlementError(
                f"{label} cannot serve this request: {safe}",
                provider=self.provider_id,
                provider_label=label,
                status_code=status,
                error_code="model_not_entitled",
                terminal_guidance=entitlement_guidance,
            )
        return None

    async def resolve_usage_limit(self, error: SubscriptionProviderError, client: Any) -> SubscriptionProviderError:
        if not isinstance(error, _KimiUsageWindowError):
            return error
        try:
            usage = await client.usage()
        except Exception:
            return error
        reset_at = self._reset_for_window(usage.get("pools") or [], error.window)
        if reset_at is None:
            return error
        resets_in_seconds = max(1, reset_at - int(time.time()))
        return SubscriptionUsageLimitError(
            str(error),
            provider=self.provider_id,
            provider_label=self.label,
            resets_at=reset_at,
            resets_in_seconds=resets_in_seconds,
            plan_type=error.window,
            status_code=error.status_code,
            error_code=error.error_code,
        )

    @staticmethod
    def _reset_for_window(pools: List[Dict[str, Any]], window: str) -> Optional[int]:
        now = int(time.time())
        candidates = [
            pool for pool in pools
            if pool.get("window") == window and isinstance(pool.get("reset_at"), int) and pool["reset_at"] > now
        ]
        if not candidates:
            return None
        exhausted = [pool for pool in candidates if float(pool.get("used_ratio") or 0) >= 0.999]
        chosen = exhausted or candidates
        return min(int(pool["reset_at"]) for pool in chosen)

    # --- usage --------------------------------------------------------------------
    def parse_usage(self, payload: Any) -> Dict[str, Any]:
        pools: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for key, entry in self._usage_entries(payload):
            window, label = self._window_for_key(key)
            if window is None or key in seen:
                continue
            seen.add(key)
            pools.append({
                "id": key,
                "label": label,
                "window": window,
                "used_ratio": self._used_ratio(entry),
                "reset_at": self._reset_epoch(entry),
            })
        return {"pools": pools, "fetched_at": int(time.time())}

    @classmethod
    def _usage_entries(cls, payload: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for item in cls.iter_error_objects(payload, max_depth=6):
            for key, value in item.items():
                if isinstance(value, dict) and cls._looks_like_pool(value):
                    yield str(key), value

    @staticmethod
    def _looks_like_pool(value: Dict[str, Any]) -> bool:
        keys = {str(key).lower() for key in value}
        return bool(keys & {"used_ratio", "usedratio", "used", "remaining"}) and bool(
            keys & {"reset_time", "resettime", "reset_at", "resetat"}
        )

    @staticmethod
    def _window_for_key(key: str) -> Tuple[Optional[str], str]:
        lowered = key.lower()
        for fragment, window, label in _POOL_WINDOWS:
            if fragment in lowered:
                return window, label
        return None, ""

    @classmethod
    def _used_ratio(cls, entry: Dict[str, Any]) -> Optional[float]:
        for name in ("used_ratio", "usedRatio"):
            value = entry.get(name)
            if isinstance(value, (int, float)):
                ratio = float(value)
                return ratio / 100.0 if ratio > 1.5 else ratio
        used, limit = entry.get("used"), entry.get("limit")
        if isinstance(used, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
            return float(used) / float(limit)
        return None

    @classmethod
    def _reset_epoch(cls, entry: Dict[str, Any]) -> Optional[int]:
        for name in ("reset_time", "resetTime", "reset_at", "resetAt"):
            value = entry.get(name)
            if value in (None, ""):
                continue
            if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
                number = int(float(value))
                return number // 1000 if number > 10_000_000_000 else number
            if isinstance(value, str):
                try:
                    return int(datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp())
                except ValueError:
                    continue
        return None


KIMI_CODE_PROFILE = KimiCodeProfile(KIMI_CODE_SETTINGS)
