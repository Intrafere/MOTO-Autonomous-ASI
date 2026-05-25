"""Shared provider-credit pause/resume helpers for proof workflows."""
from __future__ import annotations

import asyncio
import contextlib
from typing import Callable, Optional

from backend.shared.openrouter_client import CreditExhaustionError, FreeModelExhaustedError


ShouldStopFn = Optional[Callable[[], bool]]

_provider_resume_event: Optional[asyncio.Event] = None
_active_pause_count = 0

_CREDIT_PAUSE_MARKERS = (
    "account free credits exhausted",
    "credit exhaustion",
    "credits exhausted",
    "free credits exhausted",
    "openrouter credits exhausted",
)

_HARD_CONFIG_MARKERS = (
    "no api key is set",
    "no openrouter api key is available",
    "openrouter privacy settings are blocking",
    "privacy settings are blocking",
    "data policy",
)


def _get_resume_event() -> asyncio.Event:
    global _provider_resume_event
    current_loop = None
    with contextlib.suppress(RuntimeError):
        current_loop = asyncio.get_running_loop()
    existing_loop = getattr(_provider_resume_event, "_loop", None)
    if (
        _provider_resume_event is None
        or (
            current_loop is not None
            and existing_loop is not None
            and existing_loop is not current_loop
        )
    ):
        _provider_resume_event = asyncio.Event()
        if _active_pause_count > 0:
            _provider_resume_event.clear()
        else:
            _provider_resume_event.set()
    return _provider_resume_event


def is_provider_credit_pause_error(exc: Exception) -> bool:
    """Return true when a provider failure should pause proof workflows."""
    if isinstance(exc, CreditExhaustionError):
        return True
    message = str(exc or "").lower()
    if isinstance(exc, FreeModelExhaustedError) and "account free credits exhausted" not in message:
        return False
    if any(marker in message for marker in _HARD_CONFIG_MARKERS):
        return False
    return any(marker in message for marker in _CREDIT_PAUSE_MARKERS)


def mark_provider_paused() -> int:
    """Mark at least one proof workflow as paused and require a future resume."""
    global _active_pause_count
    _active_pause_count += 1
    _get_resume_event().clear()
    return _active_pause_count


def resume_provider_pauses() -> int:
    """Wake all provider-paused proof workflows."""
    global _active_pause_count
    resumed = _active_pause_count
    _active_pause_count = 0
    _get_resume_event().set()
    return resumed


async def wait_for_provider_resume(should_stop: ShouldStopFn = None) -> None:
    """Wait until the user resets provider exhaustion or the workflow stops."""
    event = _get_resume_event()
    while not event.is_set():
        if should_stop is not None and should_stop():
            raise asyncio.CancelledError()
        try:
            await asyncio.wait_for(event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
