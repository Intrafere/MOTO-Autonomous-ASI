"""Process-wide lifecycle registry for run-scoped solution-path managers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .engine import Reviewer, SolutionPathEngine
from .models import prompt_fingerprint


def stable_solution_path_run_id(
    workflow_mode: str,
    user_prompt: str,
    *,
    stable_run_id: str | None = None,
) -> str:
    """Build a filesystem-safe identity without using mutable phase/source text."""
    if stable_run_id:
        if Path(stable_run_id).name != stable_run_id:
            raise ValueError("stable_run_id must be a single path component")
        return stable_run_id
    mode = re.sub(r"[^a-z0-9_-]+", "-", workflow_mode.lower()).strip("-") or "workflow"
    return f"{mode}_{prompt_fingerprint(user_prompt)[:16]}"


@dataclass
class _RunLifecycle:
    """Serialization and provenance shared by every operation on one run."""

    root: Path
    run_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    generation: int = 0
    manager: SolutionPathEngine | None = None
    workflow_mode: str | None = None
    prompt_hash: str | None = None


class SolutionPathManagerRegistry:
    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], _RunLifecycle] = {}
        self._lock = asyncio.Lock()
        self._last_acquired_identity: tuple[str, str] | None = None

    @staticmethod
    def _canonical_root(root: Path | str) -> Path:
        return Path(root).expanduser().resolve(strict=False)

    @classmethod
    def _identity(cls, root: Path | str, run_id: str) -> tuple[str, str]:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a non-empty single path component")
        canonical_root = cls._canonical_root(root)
        return os.path.normcase(str(canonical_root)), run_id

    async def _lifecycle(self, root: Path | str, run_id: str) -> _RunLifecycle:
        identity = self._identity(root, run_id)
        async with self._lock:
            lifecycle = self._runs.get(identity)
            if lifecycle is None:
                lifecycle = _RunLifecycle(
                    root=self._canonical_root(root),
                    run_id=run_id,
                )
                self._runs[identity] = lifecycle
            return lifecycle

    async def _matching_lifecycles(self, run_id: str) -> list[_RunLifecycle]:
        async with self._lock:
            return [
                lifecycle
                for (_, candidate_run_id), lifecycle in self._runs.items()
                if candidate_run_id == run_id
            ]

    @staticmethod
    def _validate_provenance(
        lifecycle: _RunLifecycle,
        *,
        workflow_mode: str,
        user_prompt: str,
    ) -> None:
        prompt_hash = prompt_fingerprint(user_prompt)
        if (
            lifecycle.workflow_mode is not None
            and lifecycle.workflow_mode != workflow_mode
        ):
            raise ValueError("loaded solution path workflow_mode mismatch")
        if lifecycle.prompt_hash is not None and lifecycle.prompt_hash != prompt_hash:
            raise ValueError("loaded solution path user_prompt mismatch")

    async def acquire(
        self,
        root: Path | str,
        *,
        workflow_mode: str,
        user_prompt: str,
        reviewer: Reviewer,
        stable_run_id: str | None = None,
    ) -> SolutionPathEngine:
        user_prompt = user_prompt.strip()
        run_id = stable_solution_path_run_id(
            workflow_mode, user_prompt, stable_run_id=stable_run_id
        )
        lifecycle = await self._lifecycle(root, run_id)
        async with lifecycle.lock:
            self._validate_provenance(
                lifecycle,
                workflow_mode=workflow_mode,
                user_prompt=user_prompt,
            )
            manager = lifecycle.manager
            if manager is None:
                manager = SolutionPathEngine(
                    lifecycle.root,
                    run_id,
                    reviewer,
                    workflow_mode=workflow_mode,
                    user_prompt=user_prompt,
                )
                lifecycle.manager = manager
                lifecycle.workflow_mode = workflow_mode
                lifecycle.prompt_hash = prompt_fingerprint(user_prompt)
            else:
                await manager.set_reviewer(reviewer)
            lifecycle.generation += 1
            await manager.start()
            self._last_acquired_identity = self._identity(root, run_id)
            return manager

    def get(
        self, run_id: str, root: Path | str | None = None
    ) -> SolutionPathEngine | None:
        if root is not None:
            lifecycle = self._runs.get(self._identity(root, run_id))
            return lifecycle.manager if lifecycle is not None else None
        matches = [
            lifecycle.manager
            for (_, candidate_run_id), lifecycle in self._runs.items()
            if candidate_run_id == run_id and lifecycle.manager is not None
        ]
        if len(matches) > 1:
            raise ValueError("run_id is ambiguous across solution path roots")
        return matches[0] if matches else None

    def loaded_managers(self) -> tuple[SolutionPathEngine, ...]:
        """Return process-loaded managers for fast read-only status snapshots."""
        return tuple(
            lifecycle.manager
            for lifecycle in self._runs.values()
            if lifecycle.manager is not None
        )

    def latest_loaded_manager(self) -> SolutionPathEngine | None:
        """Return the most recently acquired manager when it is still loaded."""
        if self._last_acquired_identity is None:
            return None
        lifecycle = self._runs.get(self._last_acquired_identity)
        return lifecycle.manager if lifecycle is not None else None

    async def pause(self, run_id: str, root: Path | str | None = None) -> None:
        lifecycles = (
            [await self._lifecycle(root, run_id)]
            if root is not None
            else await self._matching_lifecycles(run_id)
        )
        if root is None and len(lifecycles) > 1:
            raise ValueError("run_id is ambiguous across solution path roots")
        for lifecycle in lifecycles:
            async with lifecycle.lock:
                if lifecycle.manager is not None:
                    lifecycle.generation += 1
                    await lifecycle.manager.stop()

    async def clear(self, run_id: str, root: Path | str | None = None) -> None:
        lifecycles = (
            [await self._lifecycle(root, run_id)]
            if root is not None
            else await self._matching_lifecycles(run_id)
        )
        if root is None and len(lifecycles) > 1:
            raise ValueError("run_id is ambiguous across solution path roots")
        for lifecycle in lifecycles:
            async with lifecycle.lock:
                lifecycle.generation += 1
                manager = lifecycle.manager
                lifecycle.manager = None
                lifecycle.workflow_mode = None
                lifecycle.prompt_hash = None
                if self._last_acquired_identity == self._identity(lifecycle.root, run_id):
                    self._last_acquired_identity = None
                if manager is not None:
                    await manager.clear()

    async def clear_run(self, root: Path | str, run_id: str) -> None:
        lifecycle = await self._lifecycle(root, run_id)
        async with lifecycle.lock:
            lifecycle.generation += 1
            manager = lifecycle.manager
            lifecycle.manager = None
            lifecycle.workflow_mode = None
            lifecycle.prompt_hash = None
            if self._last_acquired_identity == self._identity(lifecycle.root, run_id):
                self._last_acquired_identity = None
            if manager is not None:
                await manager.clear()
            directory = lifecycle.root / run_id
            if directory.exists():
                await asyncio.to_thread(shutil.rmtree, directory)

    async def clear_workflow(self, root: Path | str, workflow_mode: str) -> None:
        root_path = Path(root)
        if not root_path.exists():
            return
        run_ids = []
        for state_path in root_path.glob("*/solution_path_state.json"):
            try:
                payload = json.loads(await asyncio.to_thread(state_path.read_text, encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if payload.get("workflow_mode") == workflow_mode:
                run_ids.append(state_path.parent.name)
        for run_id in run_ids:
            await self.clear_run(root_path, run_id)

    async def clear_manager(self, manager: SolutionPathEngine | None) -> None:
        if manager is None:
            return
        matches = [
            lifecycle
            for lifecycle in await self._matching_lifecycles(manager.run_id)
            if lifecycle.manager is manager
        ]
        if len(matches) != 1:
            if not matches:
                return
            raise ValueError("manager is registered under multiple solution path roots")
        await self.clear(manager.run_id, matches[0].root)


solution_path_registry = SolutionPathManagerRegistry()
