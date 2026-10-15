"""Typed provider recovery models shared by every API adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional

from backend.shared.log_redaction import redact_log_text
from backend.shared.provider_errors import ProviderRouteIdentity


class FailureKind(str, Enum):
    TRANSIENT_TRANSPORT = "transient_transport"
    TRANSIENT_SERVER = "transient_server"
    RATE_LIMIT = "rate_limit"
    TIMED_USAGE_LIMIT = "timed_usage_limit"
    CREDIT_EXHAUSTED = "credit_exhausted"
    AUTHENTICATION = "authentication"
    ENTITLEMENT = "entitlement"
    PRIVACY_POLICY = "privacy_policy"
    MODEL_UNAVAILABLE = "model_unavailable"
    ENDPOINT_UNAVAILABLE = "endpoint_unavailable"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LIMIT = "context_limit"
    OUTPUT_LIMIT = "output_limit"
    INVALID_RESPONSE = "invalid_response"
    CANCELLED = "cancelled"
    UNKNOWN_HARD = "unknown_hard"


class RecoveryAction(str, Enum):
    RETRY_ADAPTER = "retry_adapter"
    RETRY_WORKFLOW = "retry_workflow"
    ENTER_COOLDOWN = "enter_cooldown"
    ENTER_PROVIDER_PAUSE = "enter_provider_pause"
    FALLBACK = "fallback"
    OUTPUT_RECOVERY = "output_recovery"
    CONTEXT_OVERFLOW = "context_overflow"
    REPAIR_REQUIRED = "repair_required"
    CANCEL = "cancel"


class RetryLayer(str, Enum):
    ADAPTER = "adapter"
    WORKFLOW = "workflow"


class FallbackMode(str, Enum):
    CALL_ONLY = "call_only"
    UNTIL_TIME = "until_time"
    UNTIL_MANUAL_RESET = "until_manual_reset"


class QuotaScope(str, Enum):
    PROVIDER = "provider"
    CREDENTIAL = "credential"
    ACCOUNT = "account"
    PLAN = "plan"
    MODEL = "model"
    ENDPOINT = "endpoint"
    ROLE = "role"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    exponential: bool = True
    jitter_ratio: float = 0.0
    retryable_kinds: FrozenSet[FailureKind] = field(
        default_factory=lambda: frozenset(
            {
                FailureKind.TRANSIENT_TRANSPORT,
                FailureKind.TRANSIENT_SERVER,
                FailureKind.RATE_LIMIT,
                FailureKind.INVALID_RESPONSE,
            }
        )
    )

    def delay_for_attempt(
        self,
        attempt: int,
        *,
        retry_after_seconds: Optional[float] = None,
    ) -> float:
        if retry_after_seconds is not None:
            return max(0.0, min(float(retry_after_seconds), self.max_delay_seconds))
        index = max(0, int(attempt) - 1)
        multiplier = (2**index) if self.exponential else max(1, int(attempt))
        return max(0.0, min(self.base_delay_seconds * multiplier, self.max_delay_seconds))


@dataclass(frozen=True)
class CooldownKey:
    provider: str
    scope: QuotaScope = QuotaScope.PROVIDER
    scope_value: str = ""
    credential_id: str = ""

    def storage_key(self) -> str:
        return "|".join(
            (
                self.provider.strip().lower(),
                self.scope.value,
                self.scope_value.strip().lower(),
                self.credential_id.strip().lower(),
            )
        )


@dataclass(frozen=True)
class ProviderFailure(Exception):
    kind: FailureKind
    route: ProviderRouteIdentity
    safe_detail: str
    retry_after_seconds: Optional[float] = None
    reset_at: Optional[int] = None
    quota_scope: QuotaScope = QuotaScope.PROVIDER
    quota_scope_value: str = ""
    fallback_eligible: bool = True
    status_code: Optional[int] = None
    provider_code: str = ""
    cause: Optional[BaseException] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_detail", redact_log_text(self.safe_detail, 1000))
        object.__setattr__(self, "provider_code", redact_log_text(self.provider_code, 160))
        Exception.__init__(self, self.safe_detail)

    @property
    def provider(self) -> str:
        return self.route.provider

    @property
    def model(self) -> str:
        return self.route.model

    @property
    def cooldown_key(self) -> CooldownKey:
        return CooldownKey(
            provider=self.provider,
            scope=self.quota_scope,
            scope_value=self.quota_scope_value,
        )


@dataclass(frozen=True)
class ProviderRecoveryDecision:
    action: RecoveryAction
    fallback_mode: Optional[FallbackMode] = None
    wait_seconds: Optional[float] = None
    reset_at: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True)
class FallbackLease:
    role_id: str
    configured_provider: str
    configured_model: str
    effective_provider: str
    effective_model: str
    mode: FallbackMode
    reason: FailureKind
    created_at: float
    expires_at: Optional[float] = None
    reset_generation: int = 0

    def is_active(self, now: float) -> bool:
        return self.expires_at is None or now < self.expires_at
