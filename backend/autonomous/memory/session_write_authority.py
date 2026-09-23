"""Process-local serialization for writes belonging to one autonomous session."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Dict


_locks: Dict[str, asyncio.Lock] = {}
_registry_lock = asyncio.Lock()
_owners: Dict[str, asyncio.Task] = {}
_depths: Dict[str, int] = {}


def session_scope_key(path: Path) -> str:
    """Return one canonical key for papers, brainstorms, and session metadata."""
    resolved = Path(path).resolve()
    if resolved.name in {"papers", "brainstorms", "final_answer"}:
        resolved = resolved.parent
    elif resolved.name in {"auto_papers", "auto_brainstorms", "auto_final_answer"}:
        resolved = resolved.parent
    elif resolved.suffix:
        resolved = resolved.parent
    return str(resolved).casefold()


@asynccontextmanager
async def autonomous_session_write(path: Path) -> AsyncIterator[None]:
    """Acquire a re-entrant task-owned write lock for an autonomous session."""
    key = session_scope_key(path)
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("Autonomous session writes require an asyncio task.")

    if _owners.get(key) is task:
        _depths[key] += 1
        try:
            yield
        finally:
            _depths[key] -= 1
        return

    async with _registry_lock:
        lock = _locks.setdefault(key, asyncio.Lock())
    await lock.acquire()
    _owners[key] = task
    _depths[key] = 1
    try:
        yield
    finally:
        depth = _depths.get(key, 1) - 1
        if depth <= 0:
            _depths.pop(key, None)
            _owners.pop(key, None)
            lock.release()
        else:
            _depths[key] = depth
