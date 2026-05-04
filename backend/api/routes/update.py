"""
Self-update routes — allows the frontend to trigger `git pull origin main`
and poll progress in real time.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

router = APIRouter(tags=["update"])
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_pull_state: Dict[str, Any] = {
    "status": "idle",
    "output_lines": [],
    "returncode": None,
}


async def _run_pull() -> None:
    """Execute git pull as an async subprocess, streaming output into _pull_state."""
    global _pull_state
    _pull_state = {"status": "running", "output_lines": [], "returncode": None}

    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "pull", "origin", "main",
            cwd=str(_REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            _pull_state["output_lines"].append(decoded)

        await proc.wait()
        _pull_state["returncode"] = proc.returncode
        _pull_state["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        logger.exception("git pull failed with exception")
        _pull_state["output_lines"].append(f"Exception: {exc}")
        _pull_state["returncode"] = -1
        _pull_state["status"] = "error"


@router.post("/api/update/pull")
async def start_pull() -> Dict[str, Any]:
    """Kick off a git pull. Returns immediately; poll /api/update/pull-status for progress."""
    if _pull_state["status"] == "running":
        return {"started": False, "reason": "A pull is already in progress."}

    asyncio.create_task(_run_pull())
    return {"started": True}


@router.get("/api/update/pull-status")
async def get_pull_status() -> Dict[str, Any]:
    """Return current pull state including streamed output lines."""
    return _pull_state
