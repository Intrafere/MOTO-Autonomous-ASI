"""Shared synchronization for manual proof active-context transitions."""
import asyncio


_manual_proof_context_lock: asyncio.Lock | None = None


def get_manual_proof_context_lock() -> asyncio.Lock:
    """Return the process-local lock guarding manual proof archive/reset races."""
    global _manual_proof_context_lock
    if _manual_proof_context_lock is None:
        _manual_proof_context_lock = asyncio.Lock()
    return _manual_proof_context_lock
