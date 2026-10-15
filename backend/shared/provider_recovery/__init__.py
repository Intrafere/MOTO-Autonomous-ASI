"""Provider-neutral retry, cooldown, fallback, and repair contracts."""

from backend.shared.provider_recovery.models import (
    CooldownKey,
    FailureKind,
    FallbackLease,
    FallbackMode,
    ProviderFailure,
    ProviderRecoveryDecision,
    QuotaScope,
    RecoveryAction,
    RetryLayer,
    RetryPolicy,
)
from backend.shared.provider_recovery.events import canonical_recovery_event
from backend.shared.provider_recovery.policy import ProviderRecoveryPolicy
from backend.shared.provider_recovery.registry import ProviderAdapter, ProviderRegistry
from backend.shared.provider_recovery.runtime import ProviderRecoveryRuntime

__all__ = [
    "CooldownKey",
    "canonical_recovery_event",
    "FailureKind",
    "FallbackLease",
    "FallbackMode",
    "ProviderAdapter",
    "ProviderFailure",
    "ProviderRecoveryDecision",
    "ProviderRecoveryPolicy",
    "ProviderRecoveryRuntime",
    "ProviderRegistry",
    "QuotaScope",
    "RecoveryAction",
    "RetryLayer",
    "RetryPolicy",
]
