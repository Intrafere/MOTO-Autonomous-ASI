"""
Process-wide guard for mutually exclusive top-level workflow starts.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from backend.shared.sleep_inhibitor import sleep_inhibitor


@dataclass(frozen=True)
class WorkflowLease:
    owner: str
    generation: int


class WorkflowStartGuard:
    """Serialize starts and retain the committed top-level workflow owner."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_lease: WorkflowLease | None = None
        self._generation = 0

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[None]:
        async with self._lock:
            yield

    @property
    def active_owner(self) -> str | None:
        return self._active_lease.owner if self._active_lease else None

    @property
    def active_lease(self) -> WorkflowLease | None:
        return self._active_lease

    def commit(self, owner: str) -> WorkflowLease:
        """Commit a successfully started top-level workflow."""
        if not owner:
            raise ValueError("Workflow owner must be non-empty")
        if self._active_lease and self._active_lease.owner == owner:
            return self._active_lease
        if self._active_lease is not None:
            raise RuntimeError(
                f"Top-level workflow '{self._active_lease.owner}' already owns execution"
            )
        self._generation += 1
        lease = WorkflowLease(owner=owner, generation=self._generation)
        self._active_lease = lease
        try:
            sleep_inhibitor.acquire(lease)
        except Exception:
            # Power inhibition is best-effort and must never fail workflow startup.
            import logging
            logging.getLogger(__name__).exception(
                "Unable to request desktop sleep inhibition for %s", owner
            )
        return lease

    def release(self, lease: WorkflowLease | str | None) -> bool:
        """Release only the exact current lease.

        Owner-only releases are intentionally rejected: an asynchronous callback
        from an older run must never release a newer generation of the same mode.
        """
        if lease is None or self._active_lease is None:
            return False
        if isinstance(lease, str):
            return False
        if self._active_lease != lease:
            return False
        self._active_lease = None
        try:
            sleep_inhibitor.release(lease)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Unable to release desktop sleep inhibition for %s", lease.owner
            )
        return True

    def release_all(self) -> None:
        """Clear logical ownership and desktop sleep inhibition at shutdown."""
        self._generation += 1
        self._active_lease = None
        try:
            sleep_inhibitor.release_all()
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Unable to release all desktop sleep inhibition"
            )


workflow_start_guard = WorkflowStartGuard()
