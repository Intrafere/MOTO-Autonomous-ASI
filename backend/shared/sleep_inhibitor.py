"""Best-effort Windows Power Requests for active MOTO work."""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from ctypes import wintypes
from typing import Hashable, Optional, Protocol

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

POWER_REQUEST_CONTEXT_VERSION = 0
POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x00000001
POWER_REQUEST_SYSTEM_REQUIRED = 1
POWER_REQUEST_EXECUTION_REQUIRED = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
POWER_REQUEST_REASON = "MOTO has active autonomous or proof work"


class _ReasonContextDetailed(ctypes.Structure):
    _fields_ = [
        ("LocalizedReasonModule", wintypes.HMODULE),
        ("LocalizedReasonId", wintypes.ULONG),
        ("ReasonStringCount", wintypes.ULONG),
        ("ReasonStrings", ctypes.POINTER(wintypes.LPWSTR)),
    ]


class _ReasonContextUnion(ctypes.Union):
    _fields_ = [
        ("Detailed", _ReasonContextDetailed),
        ("SimpleReasonString", wintypes.LPWSTR),
    ]


class REASON_CONTEXT(ctypes.Structure):
    _anonymous_ = ("Reason",)
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("Flags", wintypes.DWORD),
        ("Reason", _ReasonContextUnion),
    ]


class PowerRequestApi(Protocol):
    def create(self, reason: str) -> Optional[int]: ...

    def set(self, handle: int, request_type: int) -> bool: ...

    def clear(self, handle: int, request_type: int) -> bool: ...

    def close(self, handle: int) -> bool: ...

    def last_error(self) -> int: ...


class WindowsPowerRequestApi:
    """Typed ctypes adapter for process-scoped Windows Power Requests."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(REASON_CONTEXT)]
        kernel32.PowerCreateRequest.restype = wintypes.HANDLE
        kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        kernel32.PowerSetRequest.restype = wintypes.BOOL
        kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
        kernel32.PowerClearRequest.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._last_error = 0

    def _capture_result(self, succeeded: bool) -> bool:
        self._last_error = 0 if succeeded else int(ctypes.get_last_error())
        return succeeded

    def create(self, reason: str) -> Optional[int]:
        reason_text = ctypes.c_wchar_p(reason)
        context = REASON_CONTEXT(
            Version=POWER_REQUEST_CONTEXT_VERSION,
            Flags=POWER_REQUEST_CONTEXT_SIMPLE_STRING,
        )
        context.SimpleReasonString = reason_text
        raw_handle = self._kernel32.PowerCreateRequest(ctypes.byref(context))
        handle = ctypes.cast(raw_handle, ctypes.c_void_p).value
        if handle is None or handle == INVALID_HANDLE_VALUE:
            self._last_error = int(ctypes.get_last_error())
            return None
        self._last_error = 0
        return handle

    def set(self, handle: int, request_type: int) -> bool:
        return self._capture_result(
            bool(self._kernel32.PowerSetRequest(wintypes.HANDLE(handle), request_type))
        )

    def clear(self, handle: int, request_type: int) -> bool:
        return self._capture_result(
            bool(self._kernel32.PowerClearRequest(wintypes.HANDLE(handle), request_type))
        )

    def close(self, handle: int) -> bool:
        return self._capture_result(bool(self._kernel32.CloseHandle(wintypes.HANDLE(handle))))

    def last_error(self) -> int:
        return self._last_error


class SleepInhibitor:
    """Keep Windows awake while logical workflow owners are active.

    Public methods update desired state only. A persistent worker serializes
    create/set/clear/close operations so FastAPI callers never wait on native
    power APIs and stale activation cannot survive a last-owner release.
    """

    def __init__(
        self,
        *,
        platform: Optional[str] = None,
        power_api: Optional[PowerRequestApi] = None,
    ) -> None:
        self._platform = sys.platform if platform is None else platform
        self._power_api = power_api
        self._owners: set[Hashable] = set()
        self._lock = threading.Lock()
        self._state_changed = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._desired_active = False
        self._handle: Optional[int] = None
        self._system_required = False
        self._execution_required = False
        self._last_error = 0
        self._worker_generation = 0

    def _is_enabled(self) -> bool:
        return self._platform == "win32" and not system_config.generic_mode

    def _get_power_api(self) -> PowerRequestApi:
        if self._power_api is None:
            self._power_api = WindowsPowerRequestApi()
        return self._power_api

    @staticmethod
    def _cleanup_request(
        api: PowerRequestApi,
        handle: int,
        *,
        system_required: bool,
        execution_required: bool,
    ) -> int:
        last_error = 0
        if execution_required and not api.clear(handle, POWER_REQUEST_EXECUTION_REQUIRED):
            last_error = api.last_error()
            logger.warning(
                "Unable to clear Windows execution-required request (error=%s)",
                last_error,
            )
        if system_required and not api.clear(handle, POWER_REQUEST_SYSTEM_REQUIRED):
            last_error = api.last_error()
            logger.warning(
                "Unable to clear Windows system-required request (error=%s)",
                last_error,
            )
        if not api.close(handle):
            last_error = api.last_error()
            logger.warning("Unable to close Windows Power Request handle (error=%s)", last_error)
        return last_error

    def _create_request(self) -> tuple[Optional[int], bool, bool, int]:
        api = self._get_power_api()
        handle = api.create(POWER_REQUEST_REASON)
        if handle is None:
            return None, False, False, api.last_error()

        system_required = api.set(handle, POWER_REQUEST_SYSTEM_REQUIRED)
        if not system_required:
            error = api.last_error()
            self._cleanup_request(
                api,
                handle,
                system_required=False,
                execution_required=False,
            )
            return None, False, False, error

        execution_required = api.set(handle, POWER_REQUEST_EXECUTION_REQUIRED)
        if not execution_required:
            error = api.last_error()
            self._cleanup_request(
                api,
                handle,
                system_required=True,
                execution_required=False,
            )
            return None, False, False, error
        return handle, True, True, 0

    def _run_worker(self, generation: int) -> None:
        while True:
            self._state_changed.wait()
            self._state_changed.clear()
            with self._lock:
                if generation != self._worker_generation:
                    return
                desired_active = self._desired_active
                handle = self._handle
                system_required = self._system_required
                execution_required = self._execution_required

            if desired_active and handle is None:
                try:
                    new_handle, new_system, new_execution, error = self._create_request()
                except Exception:
                    new_handle, new_system, new_execution, error = None, False, False, 0
                    logger.exception("Unable to establish Windows Power Request")

                with self._lock:
                    still_desired = (
                        generation == self._worker_generation and self._desired_active
                    )
                    if still_desired and new_handle is not None:
                        self._handle = new_handle
                        self._system_required = new_system
                        self._execution_required = new_execution
                        self._last_error = 0
                    elif still_desired:
                        self._last_error = error
                if new_handle is not None and not still_desired:
                    self._cleanup_request(
                        self._get_power_api(),
                        new_handle,
                        system_required=new_system,
                        execution_required=new_execution,
                    )
                if still_desired and new_handle is not None:
                    logger.info("Windows idle-sleep and execution inhibition active")
                    continue
                if not still_desired:
                    self._state_changed.set()
                    continue
                logger.warning(
                    "Windows Power Request setup failed (error=%s); inhibition will be retried",
                    error,
                )
                time.sleep(1)
                self._state_changed.set()
                continue

            if not desired_active and handle is not None:
                with self._lock:
                    if generation != self._worker_generation:
                        return
                    if self._handle != handle:
                        continue
                    self._handle = None
                    self._system_required = False
                    self._execution_required = False
                try:
                    error = self._cleanup_request(
                        self._get_power_api(),
                        handle,
                        system_required=system_required,
                        execution_required=execution_required,
                    )
                except Exception:
                    error = 0
                    logger.exception("Unable to release Windows Power Request cleanly")
                with self._lock:
                    self._last_error = error
                    reactivation_needed = (
                        generation == self._worker_generation and self._desired_active
                    )
                logger.info(
                    "Windows idle-sleep and execution inhibition released"
                )
                if reactivation_needed:
                    self._state_changed.set()
                continue

    def _ensure_worker_locked(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker_generation += 1
        generation = self._worker_generation
        try:
            worker = threading.Thread(
                target=self._run_worker,
                args=(generation,),
                name="moto-sleep-inhibitor",
                daemon=True,
            )
            worker.start()
            self._worker = worker
        except Exception:
            self._worker = None
            logger.exception(
                "Unable to start Windows sleep-inhibitor worker; workflow will continue"
            )

    def acquire(self, owner: Hashable) -> None:
        """Register an owner and inhibit idle sleep when the first owner arrives."""
        if not owner:
            raise ValueError("Sleep inhibitor owner must be non-empty")
        if not self._is_enabled():
            return
        with self._lock:
            if owner in self._owners:
                return
            self._owners.add(owner)
            self._desired_active = True
            self._ensure_worker_locked()
            self._state_changed.set()

    def release(self, owner: Hashable) -> None:
        """Remove an owner and restore normal sleep behavior after the last owner."""
        if not self._is_enabled():
            return
        with self._lock:
            if owner not in self._owners:
                return
            self._owners.remove(owner)
            if self._owners:
                return
            self._desired_active = False
            self._ensure_worker_locked()
            self._state_changed.set()

    def release_all(self) -> None:
        """Clear every owner and restore normal sleep behavior."""
        if not self._is_enabled():
            return
        with self._lock:
            self._owners.clear()
            self._desired_active = False
            if self._handle is not None:
                self._ensure_worker_locked()
                self._state_changed.set()

    @property
    def owners(self) -> frozenset[Hashable]:
        with self._lock:
            return frozenset(self._owners)

    @property
    def native_active(self) -> bool:
        with self._lock:
            return (
                self._handle is not None
                and self._system_required
                and self._execution_required
            )

    @property
    def system_required_active(self) -> bool:
        with self._lock:
            return self._system_required

    @property
    def execution_required_active(self) -> bool:
        with self._lock:
            return self._execution_required

    @property
    def last_error(self) -> int:
        with self._lock:
            return self._last_error


sleep_inhibitor = SleepInhibitor()
