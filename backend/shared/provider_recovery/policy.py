"""Pure provider recovery policy decisions."""
from __future__ import annotations

from backend.shared.provider_recovery.models import (
    FailureKind,
    FallbackMode,
    ProviderFailure,
    ProviderRecoveryDecision,
    RecoveryAction,
)


class ProviderRecoveryPolicy:
    """Map typed failures to recovery actions without mutating workflow state."""

    def decide(
        self,
        failure: ProviderFailure,
        *,
        has_fallback: bool,
        proof_checkpoint_owned: bool = False,
    ) -> ProviderRecoveryDecision:
        if failure.kind == FailureKind.CANCELLED:
            return ProviderRecoveryDecision(RecoveryAction.CANCEL, reason=failure.kind.value)
        if failure.kind == FailureKind.CONTEXT_LIMIT:
            return ProviderRecoveryDecision(
                RecoveryAction.CONTEXT_OVERFLOW,
                reason=failure.kind.value,
            )
        if failure.kind == FailureKind.OUTPUT_LIMIT:
            return ProviderRecoveryDecision(
                RecoveryAction.OUTPUT_RECOVERY,
                reason=failure.kind.value,
            )
        if failure.kind == FailureKind.TIMED_USAGE_LIMIT:
            if has_fallback and failure.fallback_eligible:
                return ProviderRecoveryDecision(
                    RecoveryAction.FALLBACK,
                    fallback_mode=FallbackMode.UNTIL_TIME,
                    wait_seconds=failure.retry_after_seconds,
                    reset_at=failure.reset_at,
                    reason=failure.kind.value,
                )
            return ProviderRecoveryDecision(
                RecoveryAction.ENTER_COOLDOWN,
                wait_seconds=failure.retry_after_seconds,
                reset_at=failure.reset_at,
                reason=failure.kind.value,
            )
        if failure.kind == FailureKind.CREDIT_EXHAUSTED:
            if has_fallback and failure.fallback_eligible:
                return ProviderRecoveryDecision(
                    RecoveryAction.FALLBACK,
                    fallback_mode=FallbackMode.UNTIL_MANUAL_RESET,
                    reason=failure.kind.value,
                )
            if proof_checkpoint_owned:
                return ProviderRecoveryDecision(
                    RecoveryAction.ENTER_PROVIDER_PAUSE,
                    reason=failure.kind.value,
                )
            return ProviderRecoveryDecision(
                RecoveryAction.REPAIR_REQUIRED,
                reason=failure.kind.value,
            )
        if failure.kind in {
            FailureKind.TRANSIENT_TRANSPORT,
            FailureKind.TRANSIENT_SERVER,
            FailureKind.RATE_LIMIT,
            FailureKind.INVALID_RESPONSE,
        }:
            return ProviderRecoveryDecision(
                RecoveryAction.RETRY_WORKFLOW,
                wait_seconds=failure.retry_after_seconds,
                reason=failure.kind.value,
            )
        if has_fallback and failure.fallback_eligible:
            return ProviderRecoveryDecision(
                RecoveryAction.FALLBACK,
                fallback_mode=FallbackMode.UNTIL_MANUAL_RESET,
                reason=failure.kind.value,
            )
        return ProviderRecoveryDecision(
            RecoveryAction.REPAIR_REQUIRED,
            reason=failure.kind.value,
        )
