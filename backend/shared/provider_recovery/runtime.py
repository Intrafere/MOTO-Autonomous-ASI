"""Shared in-memory cooldown, fallback, retry, and stop-aware wait runtime."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable, Dict, Optional, TypeVar

from backend.shared.provider_recovery.models import (
    CooldownKey,
    FallbackLease,
    ProviderFailure,
    RetryPolicy,
)

T = TypeVar("T")
RetryObserver = Callable[[int, float, ProviderFailure], Awaitable[None]]


class ProviderRecoveryRuntime:
    def __init__(self) -> None:
        self._cooldowns: Dict[str, int] = {}
        self._fallback_leases: Dict[str, FallbackLease] = {}

    def set_cooldown(self, key: CooldownKey, reset_at: int) -> None:
        self._cooldowns[key.storage_key()] = int(reset_at)

    def get_cooldown(self, key: CooldownKey, *, now: Optional[int] = None) -> Optional[int]:
        storage_key = key.storage_key()
        reset_at = self._cooldowns.get(storage_key)
        if reset_at is None:
            return None
        if reset_at <= int(time.time() if now is None else now):
            self._cooldowns.pop(storage_key, None)
            return None
        return reset_at

    def clear_cooldown(self, key: CooldownKey) -> None:
        self._cooldowns.pop(key.storage_key(), None)

    def set_fallback_lease(self, lease: FallbackLease) -> None:
        self._fallback_leases[lease.role_id] = lease

    def get_fallback_lease(
        self,
        role_id: str,
        *,
        now: Optional[float] = None,
    ) -> Optional[FallbackLease]:
        lease = self._fallback_leases.get(role_id)
        if lease is None:
            return None
        if not lease.is_active(time.time() if now is None else now):
            self._fallback_leases.pop(role_id, None)
            return None
        return lease

    def clear_fallback_lease(self, role_id: str) -> None:
        self._fallback_leases.pop(role_id, None)

    @staticmethod
    async def sleep_stop_aware(
        seconds: float,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                return False
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return not (should_stop is not None and should_stop())

    async def run_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        policy: RetryPolicy,
        should_stop: Optional[Callable[[], bool]] = None,
        observer: Optional[RetryObserver] = None,
    ) -> T:
        last_failure: Optional[ProviderFailure] = None
        for attempt in range(1, max(1, policy.max_attempts) + 1):
            if should_stop is not None and should_stop():
                raise asyncio.CancelledError()
            try:
                return await operation()
            except ProviderFailure as failure:
                last_failure = failure
                if failure.kind not in policy.retryable_kinds or attempt >= policy.max_attempts:
                    raise
                delay = policy.delay_for_attempt(
                    attempt,
                    retry_after_seconds=failure.retry_after_seconds,
                )
                if policy.jitter_ratio:
                    delay *= 1.0 + random.uniform(
                        -policy.jitter_ratio,
                        policy.jitter_ratio,
                    )
                    delay = max(0.0, delay)
                if observer is not None:
                    await observer(attempt, delay, failure)
                if not await self.sleep_stop_aware(delay, should_stop):
                    raise asyncio.CancelledError()
        if last_failure is not None:
            raise last_failure
        raise RuntimeError("Provider retry executor ended without a result or failure")


provider_recovery_runtime = ProviderRecoveryRuntime()


async def assured_retry_wait(
    payload: dict,
    *,
    role_id: str = "",
    should_stop: Optional[Callable[[], bool]] = None,
) -> bool:
    """Route direct provider-client retries through the canonical manager seam."""
    from backend.shared.api_client_manager import api_client_manager

    return await api_client_manager.emit_and_wait_for_retry(
        payload,
        role_id=role_id,
        should_stop=should_stop,
    )
