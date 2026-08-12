from __future__ import annotations

import threading
import time
import sys

import pytest

from backend.shared.sleep_inhibitor import (
    POWER_REQUEST_EXECUTION_REQUIRED,
    POWER_REQUEST_REASON,
    POWER_REQUEST_SYSTEM_REQUIRED,
    SleepInhibitor,
)


class FakePowerApi:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.error = 0
        self.next_handle = 100
        self.create_results: list[int | None] = []
        self.set_results: dict[int, list[bool]] = {}
        self.clear_results: dict[int, list[bool]] = {}
        self.close_results: list[bool] = []
        self.create_entered: threading.Event | None = None
        self.create_release: threading.Event | None = None

    def create(self, reason: str) -> int | None:
        self.calls.append(("create", reason))
        if self.create_entered is not None:
            self.create_entered.set()
        if self.create_release is not None:
            self.create_release.wait(timeout=2)
        if self.create_results:
            result = self.create_results.pop(0)
            if result is None:
                self.error = 5
            return result
        handle = self.next_handle
        self.next_handle += 1
        return handle

    def set(self, handle: int, request_type: int) -> bool:
        self.calls.append(("set", handle, request_type))
        results = self.set_results.get(request_type)
        result = results.pop(0) if results else True
        if not result:
            self.error = 87
        return result

    def clear(self, handle: int, request_type: int) -> bool:
        self.calls.append(("clear", handle, request_type))
        results = self.clear_results.get(request_type)
        result = results.pop(0) if results else True
        if not result:
            self.error = 6
        return result

    def close(self, handle: int) -> bool:
        self.calls.append(("close", handle))
        result = self.close_results.pop(0) if self.close_results else True
        if not result:
            self.error = 6
        return result

    def last_error(self) -> int:
        return self.error


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate()


def _enabled_inhibitor(monkeypatch, api: FakePowerApi) -> SleepInhibitor:
    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    return SleepInhibitor(platform="win32", power_api=api)


def test_windows_inhibitor_is_owner_idempotent_and_releases_after_last_owner(
    monkeypatch,
):
    api = FakePowerApi()
    inhibitor = _enabled_inhibitor(monkeypatch, api)

    inhibitor.acquire("aggregator")
    inhibitor.acquire("aggregator")
    inhibitor.acquire("compiler")
    inhibitor.release("aggregator")

    _wait_until(lambda: inhibitor.native_active)
    assert api.calls == [
        ("create", POWER_REQUEST_REASON),
        ("set", 100, POWER_REQUEST_SYSTEM_REQUIRED),
        ("set", 100, POWER_REQUEST_EXECUTION_REQUIRED),
    ]
    assert inhibitor.owners == frozenset({"compiler"})

    inhibitor.release("compiler")
    _wait_until(lambda: not inhibitor.native_active)
    assert api.calls[-3:] == [
        ("clear", 100, POWER_REQUEST_EXECUTION_REQUIRED),
        ("clear", 100, POWER_REQUEST_SYSTEM_REQUIRED),
        ("close", 100),
    ]
    assert inhibitor.owners == frozenset()


def test_non_windows_and_generic_mode_are_noops(monkeypatch):
    api = FakePowerApi()

    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    non_windows = SleepInhibitor(platform="linux", power_api=api)
    non_windows.acquire("autonomous")
    non_windows.release_all()

    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        True,
    )
    generic = SleepInhibitor(platform="win32", power_api=api)
    generic.acquire("leanoj")
    generic.release_all()

    assert api.calls == []
    assert non_windows.owners == frozenset()
    assert generic.owners == frozenset()


def test_create_failure_retries_without_losing_logical_owner(monkeypatch):
    api = FakePowerApi()
    api.create_results = [None, 100]
    monkeypatch.setattr("backend.shared.sleep_inhibitor.time.sleep", lambda _delay: None)
    inhibitor = _enabled_inhibitor(monkeypatch, api)

    inhibitor.acquire("autonomous")
    _wait_until(lambda: inhibitor.native_active)
    assert inhibitor.owners == frozenset({"autonomous"})
    assert [call[0] for call in api.calls].count("create") == 2

    inhibitor.release_all()
    _wait_until(lambda: not inhibitor.native_active)
    assert inhibitor.owners == frozenset()


def test_public_updates_do_not_wait_for_native_setter(monkeypatch):
    api = FakePowerApi()
    api.create_entered = threading.Event()
    api.create_release = threading.Event()
    inhibitor = _enabled_inhibitor(monkeypatch, api)
    started = time.monotonic()
    inhibitor.acquire("aggregator")
    assert time.monotonic() - started < 0.2
    assert api.create_entered.wait(timeout=1)

    started = time.monotonic()
    inhibitor.release("aggregator")
    assert time.monotonic() - started < 0.2
    api.create_release.set()
    _wait_until(lambda: any(call[0] == "close" for call in api.calls))
    assert inhibitor.native_active is False
    assert inhibitor.owners == frozenset()


def test_partial_setup_failure_rolls_back_then_retries(monkeypatch):
    api = FakePowerApi()
    api.set_results[POWER_REQUEST_EXECUTION_REQUIRED] = [False, True]
    monkeypatch.setattr("backend.shared.sleep_inhibitor.time.sleep", lambda _delay: None)
    inhibitor = _enabled_inhibitor(monkeypatch, api)
    inhibitor.acquire("autonomous")

    _wait_until(lambda: inhibitor.native_active)
    assert api.calls[:5] == [
        ("create", POWER_REQUEST_REASON),
        ("set", 100, POWER_REQUEST_SYSTEM_REQUIRED),
        ("set", 100, POWER_REQUEST_EXECUTION_REQUIRED),
        ("clear", 100, POWER_REQUEST_SYSTEM_REQUIRED),
        ("close", 100),
    ]
    assert ("create", POWER_REQUEST_REASON) in api.calls[5:]
    assert inhibitor.native_active is True
    inhibitor.release_all()
    _wait_until(lambda: not inhibitor.native_active)


def test_first_set_failure_closes_without_clearing(monkeypatch):
    api = FakePowerApi()
    api.set_results[POWER_REQUEST_SYSTEM_REQUIRED] = [False]
    api.create_results = [100, None]
    inhibitor = _enabled_inhibitor(monkeypatch, api)
    inhibitor.acquire("autonomous")
    _wait_until(lambda: ("close", 100) in api.calls)
    inhibitor.release_all()
    assert not any(call[0] == "clear" and call[1] == 100 for call in api.calls)


def test_release_all_closes_even_when_clear_fails_and_reacquires_fresh_handle(
    monkeypatch,
):
    api = FakePowerApi()
    api.clear_results[POWER_REQUEST_EXECUTION_REQUIRED] = [False]
    api.close_results = [False]
    inhibitor = _enabled_inhibitor(monkeypatch, api)
    inhibitor.acquire("first")
    _wait_until(lambda: inhibitor.native_active)
    inhibitor.release_all()
    _wait_until(lambda: ("close", 100) in api.calls and not inhibitor.native_active)
    assert inhibitor.last_error == 6

    inhibitor.acquire("second")
    _wait_until(lambda: inhibitor.native_active)
    assert ("set", 101, POWER_REQUEST_SYSTEM_REQUIRED) in api.calls
    inhibitor.release_all()
    _wait_until(lambda: not inhibitor.native_active)


def test_concurrent_acquires_create_once_and_concurrent_releases_close_once(monkeypatch):
    api = FakePowerApi()
    inhibitor = _enabled_inhibitor(monkeypatch, api)
    owners = [f"owner-{index}" for index in range(20)]

    acquire_threads = [
        threading.Thread(target=inhibitor.acquire, args=(owner,)) for owner in owners
    ]
    for thread in acquire_threads:
        thread.start()
    for thread in acquire_threads:
        thread.join()
    _wait_until(lambda: inhibitor.native_active)
    assert [call[0] for call in api.calls].count("create") == 1

    release_threads = [
        threading.Thread(target=inhibitor.release, args=(owner,)) for owner in owners
    ]
    for thread in release_threads:
        thread.start()
    for thread in release_threads:
        thread.join()
    _wait_until(lambda: not inhibitor.native_active)
    assert [call[0] for call in api.calls].count("close") == 1


def test_empty_owner_is_rejected(monkeypatch):
    api = FakePowerApi()
    inhibitor = _enabled_inhibitor(monkeypatch, api)
    try:
        inhibitor.acquire("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty owner should be rejected")


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows Power Requests")
def test_real_windows_power_request_activates_and_releases(monkeypatch):
    monkeypatch.setattr(
        "backend.shared.sleep_inhibitor.system_config.generic_mode",
        False,
    )
    inhibitor = SleepInhibitor(platform="win32")
    inhibitor.acquire("real-windows-power-request")
    _wait_until(lambda: inhibitor.native_active)
    assert inhibitor.system_required_active is True
    assert inhibitor.execution_required_active is True
    assert inhibitor.last_error == 0

    inhibitor.release("real-windows-power-request")
    _wait_until(lambda: not inhibitor.native_active)
    assert inhibitor.owners == frozenset()
