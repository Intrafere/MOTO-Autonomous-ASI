"""Thin test-only driver for real LeanOJ routes and coordinator state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.api.routes import leanoj as leanoj_route
from backend.leanoj.core.leanoj_coordinator import LeanOJCoordinator
from backend.shared.models import LeanOJRoleConfig, LeanOJStartRequest


def minimal_leanoj_request(**overrides: Any) -> LeanOJStartRequest:
    role = LeanOJRoleConfig(
        model_id="leanoj-test-model",
        context_window=8192,
        max_output_tokens=1024,
    )
    values: dict[str, Any] = {
        "user_prompt": "Prove one equals one.",
        "lean_template": "import Mathlib\n\nexample : 1 = 1 := by\n  sorry",
        "topic_generator": role,
        "topic_validator": role,
        "brainstorm_submitters": [role],
        "brainstorm_validator": role,
        "path_decider": role,
        "final_solver": role,
    }
    values.update(overrides)
    return LeanOJStartRequest(**values)


@dataclass
class LeanOJAdapter:
    """Route actions plus bounded direct coordinator operations."""

    route: Any = leanoj_route
    coordinator: LeanOJCoordinator | None = None

    async def start(self, request: LeanOJStartRequest | None = None) -> dict[str, Any]:
        return await self.route.start_leanoj(request or minimal_leanoj_request())

    async def stop(self) -> dict[str, Any]:
        return await self.route.stop_leanoj()

    async def clear(self, *, confirm: bool = True) -> dict[str, Any]:
        return await self.route.clear_leanoj(confirm=confirm)

    async def skip_brainstorm(self) -> dict[str, Any]:
        return await self.route.skip_leanoj_brainstorm()

    async def force_brainstorm(self) -> dict[str, Any]:
        return await self.route.force_leanoj_brainstorm()

    async def write_intermediate_master_proof(self, content: str, *, summary: str) -> None:
        if self.coordinator is None:
            raise RuntimeError("A coordinator is required for direct master-proof edits.")
        await self.coordinator._write_master_proof(content, summary=summary)
