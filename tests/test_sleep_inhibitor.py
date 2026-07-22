from __future__ import annotations

import threading
import time

from backend.shared.sleep_inhibitor import (
    ES_CONTINUOUS,
    ES_SYSTEM_REQUIRED,
    SleepInhibitor,
)


def test_windows_inhibitor_is_owner_idempotent_and_uses_one_worker_thread(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    calls: list[tuple[int, int]] = []

    def setter(flags: int) -> int:
        calls.append((flags, threading.get_ident()))
        return 1

    inhibitor = SleepInhibitor(
        platform="win32",
        execution_state_setter=setter,
    )

    inhibitor.acquire("aggregator")
    inhibitor.acquire("aggregator")
    inhibitor.acquire("compiler")
    inhibitor.release("aggregator")

    deadline = time.monotonic() + 2
    while len(calls) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [flags for flags, _ in calls] == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert inhibitor.owners == frozenset({"compiler"})

    inhibitor.release("compiler")

    deadline = time.monotonic() + 2
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [flags for flags, _ in calls] == [
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED,
        ES_CONTINUOUS,
    ]
    assert len({thread_id for _, thread_id in calls}) == 1
    assert inhibitor.owners == frozenset()


def test_non_windows_and_generic_mode_are_noops(monkeypatch):
    calls: list[int] = []
    setter = lambda flags: calls.append(flags) or 1

    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    non_windows = SleepInhibitor(platform="linux", execution_state_setter=setter)
    non_windows.acquire("autonomous")
    non_windows.release_all()

    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        True,
    )
    generic = SleepInhibitor(platform="win32", execution_state_setter=setter)
    generic.acquire("leanoj")
    generic.release_all()

    assert calls == []
    assert non_windows.owners == frozenset()
    assert generic.owners == frozenset()


def test_native_failure_does_not_lose_logical_owner(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )

    def failing_setter(_flags: int) -> int:
        raise OSError("simulated power API failure")

    inhibitor = SleepInhibitor(
        platform="win32",
        execution_state_setter=failing_setter,
    )

    inhibitor.acquire("autonomous")
    assert inhibitor.owners == frozenset({"autonomous"})

    inhibitor.release_all()
    assert inhibitor.owners == frozenset()


def test_public_updates_do_not_wait_for_native_setter(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_setter(_flags: int) -> int:
        entered.set()
        release.wait(timeout=2)
        return 1

    inhibitor = SleepInhibitor(
        platform="win32",
        execution_state_setter=blocked_setter,
    )
    started = time.monotonic()
    inhibitor.acquire("aggregator")
    assert time.monotonic() - started < 0.2
    assert entered.wait(timeout=1)

    started = time.monotonic()
    inhibitor.release("aggregator")
    assert time.monotonic() - started < 0.2
    release.set()


def test_return_zero_retries_until_native_acquire_succeeds(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    calls = 0

    def setter(_flags: int) -> int:
        nonlocal calls
        calls += 1
        return 0 if calls == 1 else 1

    monkeypatch.setattr("backend.shared.sleep_inhibitor.time.sleep", lambda _delay: None)
    inhibitor = SleepInhibitor(platform="win32", execution_state_setter=setter)
    inhibitor.acquire("autonomous")

    deadline = time.monotonic() + 2
    while not inhibitor.native_active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls >= 2
    assert inhibitor.native_active is True
    inhibitor.release_all()
