from __future__ import annotations

import time

import pytest

from backend.shared.api_client_manager import APIClientManager
from backend.shared.provider_errors import ProviderRouteIdentity
from backend.shared.provider_recovery.events import canonical_recovery_event
from backend.shared.provider_recovery import (
    CooldownKey,
    FailureKind,
    FallbackMode,
    ProviderFailure,
    ProviderRecoveryPolicy,
    ProviderRecoveryRuntime,
    QuotaScope,
    RecoveryAction,
    RetryPolicy,
)


def _failure(kind: FailureKind, **kwargs) -> ProviderFailure:
    return ProviderFailure(
        kind=kind,
        route=ProviderRouteIdentity(provider="test", model="model"),
        safe_detail="safe failure",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("kind", "fallback", "action", "mode"),
    [
        (FailureKind.TRANSIENT_SERVER, False, RecoveryAction.RETRY_WORKFLOW, None),
        (FailureKind.TRANSIENT_SERVER, True, RecoveryAction.RETRY_WORKFLOW, None),
        (FailureKind.RATE_LIMIT, True, RecoveryAction.RETRY_WORKFLOW, None),
        (
            FailureKind.TIMED_USAGE_LIMIT,
            False,
            RecoveryAction.ENTER_COOLDOWN,
            None,
        ),
        (
            FailureKind.TIMED_USAGE_LIMIT,
            True,
            RecoveryAction.FALLBACK,
            FallbackMode.UNTIL_TIME,
        ),
        (
            FailureKind.CREDIT_EXHAUSTED,
            True,
            RecoveryAction.FALLBACK,
            FallbackMode.UNTIL_MANUAL_RESET,
        ),
        (FailureKind.CONTEXT_LIMIT, True, RecoveryAction.CONTEXT_OVERFLOW, None),
        (FailureKind.OUTPUT_LIMIT, True, RecoveryAction.OUTPUT_RECOVERY, None),
        (FailureKind.AUTHENTICATION, False, RecoveryAction.REPAIR_REQUIRED, None),
    ],
)
def test_recovery_policy_matrix(kind, fallback, action, mode):
    decision = ProviderRecoveryPolicy().decide(
        _failure(kind),
        has_fallback=fallback,
    )
    assert decision.action == action
    assert decision.fallback_mode == mode


def test_provider_failure_redacts_secret_detail():
    failure = _failure(
        FailureKind.AUTHENTICATION,
        provider_code="Bearer sk-secret-value",
    )
    assert "sk-secret-value" not in failure.provider_code


def test_scoped_cooldown_expires():
    runtime = ProviderRecoveryRuntime()
    key = CooldownKey("provider", QuotaScope.MODEL, "model-a")
    runtime.set_cooldown(key, 200)
    assert runtime.get_cooldown(key, now=100) == 200
    assert runtime.get_cooldown(key, now=200) is None


@pytest.mark.asyncio
async def test_retry_executor_uses_max_attempts_and_observer():
    runtime = ProviderRecoveryRuntime()
    attempts = 0
    observations = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _failure(FailureKind.TRANSIENT_TRANSPORT)
        return "ok"

    async def observer(attempt, delay, failure):
        observations.append((attempt, delay, failure.kind))

    result = await runtime.run_with_retry(
        operation,
        policy=RetryPolicy(max_attempts=3, base_delay_seconds=0),
        observer=observer,
    )
    assert result == "ok"
    assert attempts == 3
    assert [item[0] for item in observations] == [1, 2]


@pytest.mark.asyncio
async def test_stop_aware_wait_exits_early():
    started = time.monotonic()
    assert not await ProviderRecoveryRuntime.sleep_stop_aware(
        10,
        should_stop=lambda: True,
    )
    assert time.monotonic() - started < 1


@pytest.mark.asyncio
async def test_emit_and_wait_for_retry_broadcasts_before_canonical_wait(monkeypatch):
    manager = APIClientManager()
    order = []

    async def broadcast(payload, *, role_id=""):
        order.append(("broadcast", payload["retry_attempt"], role_id))

    async def wait(seconds, should_stop=None):
        order.append(("wait", seconds, should_stop is not None))
        return True

    monkeypatch.setattr(manager, "_broadcast_provider_retry_cooldown", broadcast)
    monkeypatch.setattr(manager, "_sleep_with_optional_stop", wait)

    completed = await manager.emit_and_wait_for_retry(
        {
            "provider": "test",
            "role_id": "agg_val",
            "retry_attempt": 2,
            "retry_after_seconds": 4,
            "reason": "transient_http_error",
        },
        role_id="agg_val",
        should_stop=lambda: False,
    )

    assert completed
    assert order == [
        ("broadcast", 2, "agg_val"),
        ("wait", 4.0, True),
    ]


def test_cooldown_chunks_share_correlation_but_have_distinct_attempt_ids():
    payload = {
        "provider": "test",
        "role_id": "agg_val",
        "model": "model",
        "reason": "usage_limit_cooldown",
        "cooldown_until": 12345,
    }
    first = canonical_recovery_event(
        "provider_retry_cooldown",
        {**payload, "retry_attempt": 1},
        recovery_action="retry",
    )
    second = canonical_recovery_event(
        "provider_retry_cooldown",
        {**payload, "retry_attempt": 2},
        recovery_action="retry",
    )

    assert first["correlation_id"] == second["correlation_id"]
    assert first["event_id"] != second["event_id"]


def test_canonical_recovery_event_preserves_explicit_correlation():
    event = canonical_recovery_event(
        "provider_retry_cooldown",
        {
            "provider": "openrouter",
            "role_id": "agg_val",
            "model": "vendor/model",
            "reason": "http_error",
            "retry_attempt": 1,
            "correlation_id": "provider-retry:call-123",
        },
        recovery_action="retry",
    )

    assert event["correlation_id"] == "provider-retry:call-123"
    assert event["event_id"] == "provider_retry_cooldown:provider-retry:call-123:1"


@pytest.mark.asyncio
async def test_provider_retry_callbacks_isolate_separate_api_call_correlations(monkeypatch):
    manager = APIClientManager()
    captured = []

    async def emit(payload, *, role_id="", should_stop=None):
        captured.append((dict(payload), role_id, should_stop))
        return True

    monkeypatch.setattr(manager, "emit_and_wait_for_retry", emit)
    first_call = manager._provider_retry_callback("agg_val")
    second_call = manager._provider_retry_callback("agg_val")

    base = {
        "provider": "openrouter",
        "model": "vendor/model",
        "reason": "http_error",
        "retry_after_seconds": 1,
    }
    await first_call({**base, "retry_attempt": 1})
    await first_call({**base, "retry_attempt": 2})
    await second_call({**base, "retry_attempt": 1})

    first_correlation = captured[0][0]["correlation_id"]
    assert captured[1][0]["correlation_id"] == first_correlation
    assert captured[2][0]["correlation_id"] != first_correlation
    assert all(callable(item[2]) for item in captured)
