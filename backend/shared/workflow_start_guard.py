"""
Process-wide guard for mutually exclusive top-level workflow starts.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class WorkflowStartGuard:
    """Serialize conflict checks and startup side effects across top-level modes."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[None]:
        async with self._lock:
            yield


workflow_start_guard = WorkflowStartGuard()
