"""Thin test-only driver for the real manual Compiler route lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.api.routes import compiler as compiler_route
from backend.shared.models import CompilerStartRequest


def minimal_compiler_request(**overrides: Any) -> CompilerStartRequest:
    values: dict[str, Any] = {
        "compiler_prompt": "Write a rigorous paper.",
        "validator_model": "test-validator",
        "validator_context_size": 4096,
        "validator_max_output_tokens": 512,
        "writer_model": "test-writer",
        "writer_context_size": 4096,
        "writer_max_output_tokens": 512,
        "high_param_model": "test-rigor",
        "high_param_context_size": 4096,
        "high_param_max_output_tokens": 512,
    }
    values.update(overrides)
    return CompilerStartRequest(**values)


@dataclass
class ManualCompilerAdapter:
    """Calls production route functions while dependencies are patched by tests."""

    route: Any = compiler_route

    async def start(self, request: CompilerStartRequest | None = None) -> dict[str, Any]:
        return await self.route.start_compiler(request or minimal_compiler_request())

    async def stop(self) -> dict[str, Any]:
        return await self.route.stop_compiler()

    async def clear(self, *, confirm: bool = True) -> dict[str, Any]:
        return await self.route.clear_paper(confirm=confirm)

    async def save(self) -> dict[str, Any]:
        return await self.route.save_paper()
