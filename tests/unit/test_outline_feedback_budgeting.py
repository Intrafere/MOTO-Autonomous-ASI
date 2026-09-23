"""Outline creation shares one bounded recency window across both feedback streams."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.compiler.agents import writer_submitter as writer
from backend.compiler.memory import outline_memory as outline_module
from backend.compiler.memory.compiler_rejection_log import CompilerRejectionLog
from backend.compiler.memory.outline_memory import (
    OUTLINE_CREATION_FEEDBACK_FILE,
    OutlineMemory,
)
from backend.shared.provider_errors import ProviderContextLengthError, ProviderRouteIdentity


ACCEPTED_OUTLINE = "Accepted outline sentinel\nI. Introduction\nII. Complete body\nIII. Conclusion"
USER_PROMPT = "Outline budgeting regression objective"
EVIDENCE = "Isolated source evidence sentinel"


@pytest.fixture
def outline_case(monkeypatch, tmp_path):
    def setup(comment_count=5, rejection_count=5):
        comments = [f"Creation-{i}: whole comment\ncreation-tail-{i}" for i in range(comment_count)]
        rejections = [
            {"text": f"Rejection-{i}: whole rejection\nrejection-tail-{i}"}
            for i in range(rejection_count)
        ]
        feedback = list(comments)
        if feedback:
            # The accepted outline belongs to the oldest comment and must outlive it.
            feedback[0] = f"ACCEPTED {feedback[0]}\n---YOUR OUTLINE---\n{ACCEPTED_OUTLINE}"
        feedback_path = tmp_path / OUTLINE_CREATION_FEEDBACK_FILE
        feedback_path.write_text("\n\n---FEEDBACK SEPARATOR---\n\n".join(feedback), encoding="utf-8")
        monkeypatch.setattr(outline_module, "system_config", SimpleNamespace(data_dir=tmp_path))
        memory = OutlineMemory.__new__(OutlineMemory)
        import asyncio
        memory._lock = asyncio.Lock()
        log = CompilerRejectionLog()
        log.rejections = deepcopy(rejections)
        log.rejections_file = tmp_path / "rejections.txt"
        log.rejections_file.write_text(log.render_rejections(rejections), encoding="utf-8")
        before_files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        before_rejections = deepcopy(log.rejections)
        monkeypatch.setattr(writer, "outline_memory", memory)
        monkeypatch.setattr(writer, "compiler_rejection_log", log)
        retrieve = AsyncMock(return_value=SimpleNamespace(text=EVIDENCE))
        monkeypatch.setattr(writer, "compiler_rag_manager", SimpleNamespace(retrieve_for_mode=retrieve))
        error = ProviderContextLengthError(
            "Synthetic context rejection",
            route=ProviderRouteIdentity(provider="openrouter", model="test-model"),
        )
        generate = AsyncMock(side_effect=error)
        prewarm = AsyncMock()
        monkeypatch.setattr(writer, "api_client_manager", SimpleNamespace(
            generate_completion=generate, prewarm_assistant_memory_context=prewarm,
        ))
        monkeypatch.setattr(writer, "system_config", SimpleNamespace(
            compiler_writer_context_window=1000000,
            compiler_writer_max_output_tokens=100,
        ))
        monkeypatch.setattr(writer, "rag_config", SimpleNamespace(
            get_available_input_tokens=lambda context, output: context - output,
        ))
        # Count the complete real production prompt deterministically, without tokenizer variability.
        monkeypatch.setattr(writer, "count_tokens", len)
        monkeypatch.setattr("backend.shared.prompt_feedback_budget.count_tokens", len)
        agent = writer.WritingSubmitter("test-model", USER_PROMPT)

        def assert_unchanged():
            assert log.rejections == before_rejections
            assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before_files

        return SimpleNamespace(
            agent=agent, generate=generate, prewarm=prewarm, error=error,
            comments=comments, rejections=rejections, memory=memory, log=log,
            assert_unchanged=assert_unchanged,
        )
    return setup


def assert_prompt_window(prompt, case, retained):
    for entries in (case.comments, [entry["text"] for entry in case.rejections]):
        visible = entries[-retained:]
        for entry in entries:
            if entry in visible:
                assert entry in prompt  # Whole multiline entry, not a truncated preview.
            else:
                assert entry not in prompt
        assert [prompt.index(entry) for entry in visible] == sorted(prompt.index(entry) for entry in visible)
    assert ACCEPTED_OUTLINE in prompt
    assert USER_PROMPT in prompt
    assert EVIDENCE in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("comment_count,rejection_count", [(5, 5), (2, 5), (5, 2), (3, 3), (7, 7)])
async def test_provider_overflow_sheds_shared_slots_without_mutating_history(
    outline_case, comment_count, rejection_count,
):
    case = outline_case(comment_count, rejection_count)
    with pytest.raises(ProviderContextLengthError) as caught:
        await case.agent.submit_outline_create()
    assert caught.value is case.error
    windows = list(range(min(5, max(comment_count, rejection_count)), 0, -1))
    assert case.generate.await_count == len(windows)
    assert case.generate.await_count <= 5
    for call, retained in zip(case.generate.await_args_list, windows):
        assert_prompt_window(call.kwargs["messages"][0]["content"], case, retained)
        assert call.kwargs["task_id"] == "comp_writer_000"
        assert call.kwargs["max_tokens"] == 100
    assert case.agent.task_sequence == 1
    case.assert_unchanged()


@pytest.mark.asyncio
async def test_local_fitting_starts_provider_at_largest_fitting_shared_window(outline_case):
    case = outline_case()
    accepted, comments = await case.memory.get_creation_feedback_entries()
    expected = await writer.build_outline_create_prompt(
        user_prompt=USER_PROMPT, rag_evidence=EVIDENCE,
        rejection_history=case.log.render_rejections(case.rejections[-3:]),
        creation_feedback=case.memory.render_creation_feedback(accepted, comments[-3:]),
    )
    case.agent.available_input_tokens = len(expected)
    with pytest.raises(ProviderContextLengthError):
        await case.agent.submit_outline_create()
    assert case.generate.await_count == 3
    assert case.generate.await_args_list[0].kwargs["messages"][0]["content"] == expected
    for call, retained in zip(case.generate.await_args_list, [3, 2, 1]):
        prompt = call.kwargs["messages"][0]["content"]
        assert len(prompt) <= case.agent.available_input_tokens
        assert_prompt_window(prompt, case, retained)
    case.assert_unchanged()
