"""A malformed batch validator reply is operational, never a decision.

A dual/triple validator reply with the wrong number of decisions ("Expected 2
decisions, got 30"), out-of-order or malformed decisions, or unusable JSON used to
become a fabricated "reject" for every submission in the batch, and each submitter
received "Validator response format error" as rejection feedback. Now no decision
is recorded: the batch goes back to the front of the queue and those submissions
are validated one at a time.
"""
import json
from types import SimpleNamespace

import pytest

from backend.aggregator.agents import validator as validator_module
from backend.aggregator.agents.validator import BatchValidationContractError, ValidatorAgent
from backend.aggregator.core import coordinator as coordinator_module
from backend.aggregator.core.coordinator import Coordinator
from backend.aggregator.core.queue_manager import queue_manager
from backend.shared.models import Submission, ValidationResult


def _submission(index: int) -> Submission:
    return Submission(
        submission_id=f"submission-{index}",
        submitter_id=index,
        content=f"Submission {index} content.",
        reasoning="reasoning",
        chunk_size_used=512,
    )


def _decision(number: int, decision: str = "reject") -> dict:
    return {
        "submission_number": number,
        "decision": decision,
        "reasoning": f"Decision {number} reasoning.",
        "summary": f"Decision {number} summary.",
    }


def _validator(monkeypatch, reply) -> ValidatorAgent:
    """A validator whose model returns ``reply`` (a JSON-able value, raw text, or None)."""
    agent = ValidatorAgent(
        model_name="validator-model",
        user_prompt="Solve the problem.",
        user_files_content={},
    )

    async def allocate(**_kwargs):
        return {"direct": "", "rag_context": None}

    async def shared_training():
        return ""

    async def generate_completion(**_kwargs):
        if reply is None:
            return {"choices": []}
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return {"choices": [{"message": {"content": content}}]}

    async def no_repair(*_args, **_kwargs):
        return None, {}

    monkeypatch.setattr(validator_module.context_allocator, "validator_context_window", 128000)
    monkeypatch.setattr(validator_module.rag_config, "validator_context_window", 128000)
    monkeypatch.setattr(validator_module.rag_config, "validator_max_output_tokens", 8000)
    monkeypatch.setattr(validator_module.context_allocator, "allocate_validator_context", allocate)
    monkeypatch.setattr(validator_module.shared_training_memory, "get_all_content", shared_training)
    monkeypatch.setattr(validator_module.api_client_manager, "generate_completion", generate_completion)
    monkeypatch.setattr(agent, "_retry_batch_json_parse", no_repair)
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({"decisions": [_decision(number) for number in range(1, 31)]}, "Expected 2 decisions"),
        ({"decisions": [_decision(2), _decision(1)]}, "order mismatch"),
        ({"decisions": [_decision(1), _decision(2, "maybe")]}, "expected accept or reject"),
        ({"decisions": [_decision(1), "reject"]}, "not a JSON object"),
        ("not json at all", "not valid JSON"),
        (None, "no reply"),
    ],
)
async def test_malformed_batch_reply_raises_instead_of_rejecting(monkeypatch, reply, message):
    agent = _validator(monkeypatch, reply)

    with pytest.raises(BatchValidationContractError, match=message) as raised:
        await agent.validate_batch([_submission(1), _submission(2)])

    assert raised.value.batch_size == 2


@pytest.mark.asyncio
async def test_well_formed_batch_reply_still_returns_one_decision_per_submission(monkeypatch):
    agent = _validator(
        monkeypatch,
        {"decisions": [_decision(1, "accept"), _decision(2, "reject")]},
    )

    results = await agent.validate_batch([_submission(1), _submission(2)])

    assert [result.decision for result in results] == ["accept", "reject"]
    assert [result.submission_id for result in results] == ["submission-1", "submission-2"]


@pytest.mark.asyncio
async def test_coordinator_restores_malformed_batch_and_validates_it_one_at_a_time(monkeypatch):
    coordinator = Coordinator()
    coordinator.is_running = True
    batch_sizes: list[int] = []
    decided: list[tuple[str, str]] = []
    retry_payloads: list[dict] = []

    async def validate_batch(submissions):
        batch_sizes.append(len(submissions))
        if len(batch_sizes) == 1:
            raise BatchValidationContractError("Expected 2 decisions for this batch, got 30.", batch_size=2)
        return [
            ValidationResult(
                submission_id=submission.submission_id,
                decision="accept",
                reasoning="Independent single decision.",
                summary="",
            )
            for submission in submissions
        ]

    async def record_acceptance(submission, _result):
        decided.append((submission.submission_id, "accept"))
        if len(decided) == 2:
            coordinator.is_running = False

    async def record_rejection(submission, _result):
        decided.append((submission.submission_id, "reject"))

    async def no_pause():
        return None

    async def record_retry(payload, **_kwargs):
        retry_payloads.append(payload)
        return True

    coordinator.validator = SimpleNamespace(validate_batch=validate_batch)
    monkeypatch.setattr(coordinator, "_handle_acceptance", record_acceptance)
    monkeypatch.setattr(coordinator, "_handle_rejection", record_rejection)
    monkeypatch.setattr(coordinator, "_check_and_update_pause_state", no_pause)
    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", record_retry)
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await queue_manager.enqueue(_submission(2))

        await coordinator._validator_loop()
    finally:
        await queue_manager.clear()

    assert batch_sizes == [2, 1, 1]
    assert decided == [("submission-1", "accept"), ("submission-2", "accept")]
    assert coordinator._individual_validations_pending == 0
    assert len(retry_payloads) == 1
    assert retry_payloads[0]["reason"] == "validator_batch_contract"
    assert "No decision was recorded" in retry_payloads[0]["message"]
