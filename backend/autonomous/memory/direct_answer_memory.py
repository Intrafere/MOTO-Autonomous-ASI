"""Storage for the direct answer that ends a run started with ``direct_answer_enabled``.

Each autonomous session keeps its answers in ``direct_answer.json`` in its
session folder, newest last, so a question that is resumed and answered again
keeps its earlier answers. The newest answer is also copied to
``auto_direct_answer.json`` under the data root, which is what is read when no
session is active (for example after a restart, once an answered session has
been retired).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.shared.atomic_json import atomic_write_json
from backend.shared.config import system_config

logger = logging.getLogger(__name__)

DIRECT_ANSWER_SCHEMA = "moto-direct-answer-v1"
SESSION_FILE_NAME = "direct_answer.json"
LATEST_FILE_NAME = "auto_direct_answer.json"
MAX_KEPT_ANSWERS = 10


class DirectAnswerMemory:
    """Reads and writes direct answers for the current autonomous session."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached_path: Optional[Path] = None
        self._cached_answers: List[Dict[str, Any]] = []

    @staticmethod
    def _latest_path() -> Path:
        return Path(system_config.data_dir) / LATEST_FILE_NAME

    def _session_path(self) -> Optional[Path]:
        from backend.autonomous.memory.session_manager import session_manager

        if session_manager.is_session_active and session_manager.session_path is not None:
            return Path(session_manager.session_path) / SESSION_FILE_NAME
        return None

    @staticmethod
    def _read(path: Path) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            logger.warning("Could not read a saved direct answer file")
            return []
        answers = payload.get("answers") if isinstance(payload, dict) else None
        return [item for item in answers or [] if isinstance(item, dict)]

    @staticmethod
    def _write(path: Path, answers: List[Dict[str, Any]]) -> None:
        atomic_write_json(path, {"schema": DIRECT_ANSWER_SCHEMA, "answers": answers})

    async def _load(self, path: Path) -> List[Dict[str, Any]]:
        if path != self._cached_path:
            self._cached_answers = await asyncio.to_thread(self._read, path)
            self._cached_path = path
        return self._cached_answers

    async def latest(self) -> Optional[Dict[str, Any]]:
        """The active session's newest answer, or the last answer written when none is active."""
        async with self._lock:
            answers = await self._load(self._session_path() or self._latest_path())
            return dict(answers[-1]) if answers else None

    async def save(self, answer: Dict[str, Any]) -> None:
        """Add one answer to the session and publish it as the newest answer."""
        async with self._lock:
            session_path = self._session_path()
            if session_path is not None:
                answers = [*(await self._load(session_path)), dict(answer)][-MAX_KEPT_ANSWERS:]
                await asyncio.to_thread(self._write, session_path, answers)
                self._cached_path = session_path
                self._cached_answers = answers
            await asyncio.to_thread(self._write, self._latest_path(), [dict(answer)])
            if session_path is None:
                self._cached_path = self._latest_path()
                self._cached_answers = [dict(answer)]

    async def clear_latest(self) -> None:
        """Forget the newest-answer copy (session folders keep their own answers)."""
        async with self._lock:
            try:
                await asyncio.to_thread(self._latest_path().unlink)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove the saved direct answer file")
            self._cached_path = None
            self._cached_answers = []


direct_answer_memory = DirectAnswerMemory()
