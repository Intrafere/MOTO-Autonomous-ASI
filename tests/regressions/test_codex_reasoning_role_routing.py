"""Configured Codex effort survives role routing and deliberate per-call overrides."""
from unittest.mock import AsyncMock, patch

import pytest

from backend.shared.api_client_manager import APIClientManager
from backend.shared.models import ModelConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("role_id", ["aggregator_submitter_1", "compiler_writer", "autonomous_assistant", "leanoj_final_solver"])
@pytest.mark.parametrize("effort,override,expected", [("auto", None, "auto"), ("max", None, "max"), ("high", None, "high"), ("max", "low", "low"), ("high", "none", "none")])
async def test_codex_configured_effort_and_override_reach_adapter(role_id, effort, override, expected):
    manager = APIClientManager()
    manager.configure_role(role_id, ModelConfig(
        provider="openai_codex_oauth", model_id="gpt-5.5",
        context_window=4096, max_output_tokens=512,
        openrouter_reasoning_effort=effort,
    ))
    response = {"model": "gpt-5.5", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    kwargs = {} if override is None else {"_moto_reasoning_effort_override": override}
    with patch("backend.shared.api_client_manager.openai_codex_client.generate_completion", new=AsyncMock(return_value=response)) as completion:
        await manager._generate_completion_once(
            task_id="reasoning_routing_test", role_id=role_id, model="gpt-5.5",
            messages=[{"role": "user", "content": "hello"}], **kwargs,
        )
    assert completion.await_args.kwargs["reasoning_effort"] == expected
    assert completion.await_args.kwargs["model"] == "gpt-5.5"
