"""Best-effort desktop sleep inhibition for active top-level workflows."""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from typing import Callable, Hashable, Optional

from backend.shared.config import system_config

logger = logging.getLogger(__name__)

ES_SYSTEM_REQUIRED = 0x00000001
ES_CONTINUOUS = 0x80000000


class SleepInhibitor:
    """Keep Windows awake while logical workflow owners are active.

    The public methods only update in-memory desired state. One persistent
    worker owns all Windows calls, preserving SetThreadExecutionState's
    thread-affinity without blocking the FastAPI event loop.
    """

    def __init__(
        self,
        *,
        platform: Optional[str] = None,
        execution_state_setter: Optional[Callable[[int], int]] = None,
    ) -> None:
        self._platform = sys.platform if platform is None else platform
        self._execution_state_setter = execution_state_setter
        self._owners: set[Hashable] = set()
        self._lock = threading.Lock()
        self._state_changed = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._desired_active = False
        self._native_active = False
        self._worker_generation = 0

    def _is_enabled(self) -> bool:
        return self._platform == "win32" and not system_config.generic_mode

    def _set_execution_state(self, flags: int) -> int:
        setter = self._execution_state_setter
        if setter is None:
            setter = ctypes.windll.kernel32.SetThreadExecutionState
        return int(setter(flags))

    def _run_worker(self, generation: int) -> None:
        while True:
            self._state_changed.wait()
            self._state_changed.clear()
            with self._lock:
                if generation != self._worker_generation:
                    return
                desired_active = self._desired_active
                native_active = self._native_active
            if desired_active == native_active:
                continue

            flags = (
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                if desired_active
                else ES_CONTINUOUS
            )
            try:
                succeeded = self._set_execution_state(flags) != 0
            except Exception:
                succeeded = False
                logger.exception("Unable to update Windows sleep inhibition")

            with self._lock:
                if generation != self._worker_generation:
                    return
                if succeeded:
                    self._native_active = desired_active
            if succeeded:
                logger.info(
                    "Windows automatic sleep inhibition %s",
                    "active" if desired_active else "released",
                )
                continue

            logger.warning(
                "Windows SetThreadExecutionState failed; %s will be retried",
                "sleep inhibition" if desired_active else "sleep-state restoration",
            )
            time.sleep(1)
            self._state_changed.set()

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
            if self._native_active:
                self._ensure_worker_locked()
                self._state_changed.set()

    @property
    def owners(self) -> frozenset[Hashable]:
        with self._lock:
            return frozenset(self._owners)

    @property
    def native_active(self) -> bool:
        with self._lock:
            return self._native_active


sleep_inhibitor = SleepInhibitor()
