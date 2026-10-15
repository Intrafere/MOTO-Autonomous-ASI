"""A reply cut off mid-thinking must never reach retries or Supercharge as the model's answer."""
from unittest.mock import AsyncMock, patch

import pytest

from backend.shared.api_client_manager import APIClientManager
from backend.shared.json_parser import (
    RETRY_CONTEXT_EMPTY_PLACEHOLDER,
    sanitize_model_output_for_retry_context,
)
from backend.shared.models import ModelConfig
from backend.shared.response_extraction import extract_message_text

UNFINISHED_THOUGHT = "PRIVATE-THOUGHT: first consider the lemma about"


def _cut_off_response():
    return {
        "model": "test-model",
        "choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": "", "reasoning": UNFINISHED_THOUGHT},
        }],
    }


def _manager(role_id, *, supercharge_enabled=False):
    manager = APIClientManager()
    manager.configure_role(role_id, ModelConfig(
        provider="openrouter", model_id="test-model",
        context_window=4096, max_output_tokens=512,
        supercharge_enabled=supercharge_enabled,
    ))
    return manager


async def _no_assistant_memory(**kwargs):
    return kwargs["messages"], ""


@pytest.mark.asyncio
async def test_cut_off_reply_reaches_callers_and_retry_context_without_thinking():
    role_id = "aggregator_submitter_1"
    manager = _manager(role_id)
    with patch.object(manager, "_generate_completion_once_routed", new=AsyncMock(return_value=_cut_off_response())):
        response = await manager._generate_completion_once(
            task_id="cut_off_test", role_id=role_id, model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    # Agents read the message directly, exactly as the aggregator submitter does.
    llm_output = extract_message_text(response["choices"][0]["message"])
    assert llm_output == ""
    assert sanitize_model_output_for_retry_context(llm_output) == RETRY_CONTEXT_EMPTY_PLACEHOLDER


@pytest.mark.asyncio
async def test_supercharge_synthesis_never_sees_unfinished_thinking():
    role_id = "aggregator_submitter_1"
    manager = _manager(role_id, supercharge_enabled=True)
    synthesis = {"model": "test-model", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}
    routed = AsyncMock(side_effect=[_cut_off_response() for _ in range(4)] + [synthesis])
    with patch.object(manager, "_maybe_add_assistant_memory_context", new=_no_assistant_memory), \
            patch.object(manager, "_generate_completion_once_routed", new=routed):
        await manager.generate_completion(
            task_id="supercharge_cut_off_test", role_id=role_id, model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    synthesis_messages = routed.await_args_list[-1].kwargs["messages"]
    assert all(UNFINISHED_THOUGHT not in str(message.get("content")) for message in synthesis_messages)
