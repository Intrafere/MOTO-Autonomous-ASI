"""Thin test-only driver for the real manual Aggregator route lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.api.routes import aggregator as aggregator_route
from backend.shared.models import AggregatorStartRequest, SubmitterConfig


def minimal_aggregator_request(**overrides: Any) -> AggregatorStartRequest:
    values: dict[str, Any] = {
        "user_prompt": "Build a rigorous solution.",
        "submitter_configs": [
            SubmitterConfig(
                submitter_id=1,
                provider="lm_studio",
                model_id="test-submitter",
                context_window=4096,
                max_output_tokens=512,
            )
        ],
        "validator_provider": "lm_studio",
        "validator_model": "test-validator",
        "validator_context_size": 4096,
        "validator_max_output_tokens": 512,
    }
    values.update(overrides)
    return AggregatorStartRequest(**values)


@dataclass
class ManualAggregatorAdapter:
    """Calls production route functions while dependencies are patched by tests."""

    route: Any = aggregator_route

    async def start(self, request: AggregatorStartRequest | None = None) -> dict[str, Any]:
        return await self.route.start_aggregator(request or minimal_aggregator_request())

    async def stop(self) -> dict[str, Any]:
        return await self.route.stop_aggregator()

    async def clear(self) -> dict[str, Any]:
        return await self.route.clear_all_submissions()

    async def save_results(self) -> dict[str, Any]:
        return await self.route.save_results()
