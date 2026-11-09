"""Mark model calls made by background services rather than workflows.

Background services such as Auto Top-p Translate run beside the active
top-level workflow and share the API client manager with it. Calls made inside
``background_service_call`` must not be credited to that workflow: session
token totals and paper/Tier 3 author tracking skip them, and the service keeps
its own usage counters. A ContextVar keeps the mark local to the calling task,
so concurrent workflow calls are never affected.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_SERVICE: ContextVar[str] = ContextVar("moto_background_service_call", default="")


@contextmanager
def background_service_call(service: str) -> Iterator[None]:
    """Mark every model call made inside this block as background-service work."""
    token = _SERVICE.set(str(service or "").strip())
    try:
        yield
    finally:
        _SERVICE.reset(token)


def current_background_service() -> str:
    """Return the background service owning the current call, or ``""``."""
    return _SERVICE.get()
