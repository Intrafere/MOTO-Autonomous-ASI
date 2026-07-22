import json

import pytest

from backend.shared.solution_path import (
    PlanProposal,
    RouteStep,
    SolutionRoute,
    build_review_prompt,
    compact_review_prompt,
    review_with_json_retry,
)


def _proposal() -> PlanProposal:
    return PlanProposal(
        run_id="run",
        base_revision=0,
        route=SolutionRoute(steps=[RouteStep(title="First")]),
    )


@pytest.mark.asyncio
async def test_reviewer_repairs_malformed_json_with_sanitized_retry():
    prompts = []
    responses = [
        {"text": "<|channel>analysis broken output"},
        {
            "text": json.dumps(
                {
                    "decision": "followup",
                    "reasoning": "add one step",
                    "edits": [
                        {
                            "operation": "add",
                            "step": {"title": "Second"},
                        }
                    ],
                    "more_edits": True,
                }
            )
        },
    ]

    async def call(messages):
        prompts.append(messages)
        return responses.pop(0)

    decision = await review_with_json_retry(
        prompt=build_review_prompt(
            user_prompt="Solve it",
            proposal=_proposal(),
            current_plan=None,
        ),
        call_completion=call,
        extract_text=lambda response: response["text"],
        context_window=10000,
        max_output_tokens=1000,
    )

    assert decision.decision == "followup"
    assert decision.edits[0].operation == "add"
    assert len(prompts) == 2
    assert "<|channel>analysis" not in str(prompts[1])


@pytest.mark.asyncio
async def test_reviewer_rejects_mandatory_context_overflow_before_call():
    called = False

    async def call(messages):
        nonlocal called
        called = True
        return {"text": "{}"}

    with pytest.raises(ValueError, match="mandatory context exceeds"):
        await review_with_json_retry(
            prompt="large " * 1000,
            call_completion=call,
            extract_text=lambda response: response["text"],
            context_window=100,
            max_output_tokens=50,
        )
    assert called is False


@pytest.mark.asyncio
async def test_reviewer_uses_compacted_prompt_before_terminal_overflow():
    seen = []

    async def call(messages):
        seen.append(messages[0]["content"])
        return {"text": '{"decision":"approve","reasoning":"fits"}'}

    proposal = _proposal()
    compact = compact_review_prompt(
        user_prompt="Solve it",
        proposal=proposal,
        current_plan=None,
    )
    decision = await review_with_json_retry(
        prompt=("bookkeeping " * 5000) + compact,
        compact_prompt=compact,
        call_completion=call,
        extract_text=lambda response: response["text"],
        context_window=2000,
        max_output_tokens=500,
    )

    assert decision.decision == "approve"
    assert seen == [compact]
