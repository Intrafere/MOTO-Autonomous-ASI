"""An unusable Aggregator validator reply is operational, never a decision.

A dual/triple validator reply with the wrong number of decisions ("Expected 2
decisions, got 30"), out-of-order or malformed decisions, or unusable JSON used to
become a fabricated "reject" for every submission in the batch, and each submitter
received "Validator response format error" as rejection feedback. The same held
for a single-submission reply missing its decision (a KeyError) or that stayed
invalid JSON. Now the validator makes one bounded repair attempt; if the reply is
still unusable no decision is recorded: the submissions go back to the front of
the queue and are validated one at a time, with operational Live Activity.
"""
import json
from types import SimpleNamespace

import pytest

from backend.aggregator.agents import validator as validator_module
from backend.aggregator.agents.validator import (
    BatchValidationInterruptedError,
    ValidatorAgent,
    ValidatorOutputContractError,
)
from backend.aggregator.core import coordinator as coordinator_module
from backend.aggregator.core.coordinator import (
    VALIDATOR_BATCH_INTERRUPTED_EVENT,
    VALIDATOR_CONTRACT_EVENT,
    Coordinator,
)
from backend.aggregator.core.queue_manager import queue_manager
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.model_output_contract import is_model_output_contract_error
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


def _validator(monkeypatch, *replies):
    """A validator whose model returns ``replies`` in order (JSON-able, raw text, or None).

    The last reply repeats. Returns the agent and the list of message lists sent.
    """
    agent = ValidatorAgent(
        model_name="validator-model",
        user_prompt="Solve the problem.",
        user_files_content={},
    )
    calls: list[dict] = []

    async def allocate(**_kwargs):
        return {"direct": "", "rag_context": None}

    async def shared_training():
        return ""

    async def generate_completion(**kwargs):
        calls.append(kwargs)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        if reply is None:
            return {"choices": []}
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return {"choices": [{"message": {"content": content}}]}

    async def no_report(*_args, **_kwargs):
        return None

    async def no_cache(*_args, **_kwargs):
        return None

    monkeypatch.setattr(validator_module.context_allocator, "validator_context_window", 128000)
    monkeypatch.setattr(validator_module.rag_config, "validator_context_window", 128000)
    monkeypatch.setattr(validator_module.rag_config, "validator_max_output_tokens", 8000)
    monkeypatch.setattr(validator_module.context_allocator, "allocate_validator_context", allocate)
    monkeypatch.setattr(validator_module.shared_training_memory, "get_all_content", shared_training)
    monkeypatch.setattr(validator_module.api_client_manager, "generate_completion", generate_completion)
    monkeypatch.setattr(validator_module.api_client_manager, "extract_call_metadata", lambda _r: {})
    monkeypatch.setattr(validator_module, "report_model_output_normalizations", no_report)
    monkeypatch.setattr(validator_module.lm_studio_client, "cache_model_load_config", no_cache)
    return agent, calls


# --- batch (dual/triple) replies -------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "stage", "message"),
    [
        ({"decisions": [_decision(n) for n in range(1, 31)]}, "shape", "Expected 2 decisions for this batch, got 30"),
        ({"decisions": [_decision(2), _decision(1)]}, "invalid_value", "expected submission_number 1"),
        ({"decisions": [_decision(1), _decision(2, "maybe")]}, "invalid_value", "decision must be accept or reject"),
        ({"decisions": [_decision(1), "reject"]}, "shape", "not a JSON object"),
        ({"verdicts": []}, "missing_key", "Expected a 'decisions' list"),
        ("not json at all", "syntax", "not valid JSON"),
        (None, "shape", "empty reply"),
    ],
)
async def test_unusable_batch_reply_is_retried_once_then_raises_instead_of_rejecting(
    monkeypatch, reply, stage, message
):
    agent, calls = _validator(monkeypatch, reply)

    with pytest.raises(ValidatorOutputContractError, match=message) as raised:
        await agent.validate_batch([_submission(1), _submission(2)])

    assert raised.value.batch_size == 2
    assert raised.value.stage == stage
    assert len(calls) == 2  # original call plus exactly one bounded repair
    assert calls[1]["task_id"].endswith("_batch_retry")
    assert is_model_output_contract_error(raised.value)
    assert not is_transient_model_call_error(raised.value)
    assert not is_non_retryable_model_error(raised.value)


@pytest.mark.asyncio
async def test_batch_reply_names_each_bad_decision_and_field(monkeypatch):
    bad = {"submission_number": 2, "decision": "reject"}  # no reasoning
    agent, _calls = _validator(monkeypatch, {"decisions": [_decision(1), bad]})

    with pytest.raises(ValidatorOutputContractError) as raised:
        await agent.validate_batch([_submission(1), _submission(2)])

    text = str(raised.value)
    assert text.startswith("1 of 2 batch decisions was invalid")
    assert "decisions[1] '2': missing required field 'reasoning'" in text
    assert raised.value.stage == "missing_key"


@pytest.mark.asyncio
async def test_batch_contract_repair_success_returns_decisions(monkeypatch):
    agent, calls = _validator(
        monkeypatch,
        {"decisions": [_decision(1)]},
        {"decisions": [_decision(1, "accept"), _decision(2, "reject")]},
    )

    results = await agent.validate_batch([_submission(1), _submission(2)])

    assert [result.decision for result in results] == ["accept", "reject"]
    # The repair turn states what broke, not a parse error.
    assert "did not match the required JSON format" in calls[1]["messages"][-1]["content"]
    assert "Expected 2 decisions for this batch, got 1" in calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_batch_syntax_repair_prompt_keeps_existing_wording(monkeypatch):
    agent, calls = _validator(
        monkeypatch,
        "not json",
        {"decisions": [_decision(1, "accept"), _decision(2, "accept")]},
    )

    await agent.validate_batch([_submission(1), _submission(2)])

    assert calls[1]["messages"][-1]["content"].startswith(
        "Your previous response could not be parsed as valid JSON.\n\n"
        "Please provide the same validation decisions in valid JSON format:"
    )


@pytest.mark.asyncio
async def test_well_formed_batch_reply_still_returns_one_decision_per_submission(monkeypatch):
    agent, calls = _validator(
        monkeypatch,
        {"decisions": [_decision(1, "accept"), _decision(2, "reject")]},
    )

    results = await agent.validate_batch([_submission(1), _submission(2)])

    assert [result.decision for result in results] == ["accept", "reject"]
    assert [result.submission_id for result in results] == ["submission-1", "submission-2"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_batch_mislabeled_reasoning_key_is_relabeled_and_reported(monkeypatch):
    reported: list = []

    agent, _calls = _validator(
        monkeypatch,
        {
            "decisions": [
                _decision(1, "accept"),
                {
                    "submission_number": 2,
                    "decision": "reject",
                    "reasonning": "Duplicates submission 1.",
                    "summary": "Redundant.",
                },
            ]
        },
    )

    async def record(role_id, repairs, **kwargs):
        reported.append((role_id, list(repairs), kwargs))

    monkeypatch.setattr(validator_module, "report_model_output_normalizations", record)

    results = await agent.validate_batch([_submission(1), _submission(2)])

    assert results[1].reasoning == "Duplicates submission 1."
    role_id, repairs, kwargs = reported[-1]
    assert role_id == "aggregator_validator"
    assert [(r.kind, r.from_key, r.to_key, r.item_path) for r in repairs] == [
        ("key_label", "reasonning", "reasoning", "decisions[1]")
    ]
    assert kwargs["task_id"].startswith("agg_val_")


@pytest.mark.asyncio
async def test_batch_decision_field_is_never_relabeled(monkeypatch):
    item = {"submission_number": 2, "decisoin": "accept", "reasoning": "r", "summary": "s"}
    agent, _calls = _validator(monkeypatch, {"decisions": [_decision(1), item]})

    with pytest.raises(ValidatorOutputContractError) as raised:
        await agent.validate_batch([_submission(1), _submission(2)])

    assert "missing required field 'decision'" in str(raised.value)


@pytest.mark.asyncio
async def test_unexpected_batch_error_is_interrupted_not_a_contract_failure(monkeypatch):
    agent, _calls = _validator(monkeypatch, {"decisions": []})

    async def broken_allocation(**_kwargs):
        raise RuntimeError("index bookkeeping broke")

    monkeypatch.setattr(validator_module.context_allocator, "allocate_validator_context", broken_allocation)

    with pytest.raises(BatchValidationInterruptedError) as raised:
        await agent.validate_batch([_submission(1), _submission(2)])

    assert raised.value.batch_size == 2
    assert raised.value.error_type == "RuntimeError"
    # Only the exception type leaves the validator; its text stays in the log.
    assert "index bookkeeping broke" not in str(raised.value)
    assert not is_model_output_contract_error(raised.value)


# --- single-submission replies ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "stage", "field"),
    [
        ({"reasoning": "No decision key."}, "missing_key", "decision"),  # was a KeyError reject
        ({"decision": "maybe", "reasoning": "r"}, "invalid_value", "decision"),
        ({"decision": "reject"}, "missing_key", "reasoning"),
        ({"decision": "accept", "reasoning": "r", "summary": 3}, "invalid_value", "summary"),
        ("still not json", "syntax", ""),
        (None, "shape", ""),
    ],
)
async def test_unusable_single_reply_is_retried_once_then_raises_instead_of_rejecting(
    monkeypatch, reply, stage, field
):
    agent, calls = _validator(monkeypatch, reply)

    with pytest.raises(ValidatorOutputContractError) as raised:
        await agent.validate_batch([_submission(1)])

    assert raised.value.stage == stage
    assert raised.value.batch_size == 1
    if field:
        assert field in raised.value.field_summary
    assert len(calls) == 2
    assert calls[1]["task_id"].endswith("_retry")


@pytest.mark.asyncio
async def test_single_repair_prompts_name_the_failure(monkeypatch):
    agent, calls = _validator(
        monkeypatch,
        {"reasoning": "No decision key."},
        {"decision": "reject", "reasoning": "Repaired.", "summary": "Weak."},
    )

    [result] = await agent.validate_batch([_submission(1)])

    assert (result.decision, result.reasoning, result.summary) == ("reject", "Repaired.", "Weak.")
    repair = calls[1]["messages"][-1]["content"]
    assert repair.startswith("Your previous response did not match the required JSON format.\n\nCONTRACT ERROR: ")
    assert "missing required field 'decision'" in repair
    # The failed reply is replayed only as sanitized assistant context.
    assert calls[1]["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_single_syntax_repair_prompt_keeps_existing_wording_without_raw_excerpt(monkeypatch):
    agent, calls = _validator(
        monkeypatch,
        '{"decision": "accept", "reasoning": SECRET_RAW_TEXT}',
        {"decision": "accept", "reasoning": "ok", "summary": ""},
    )

    [result] = await agent.validate_batch([_submission(1)])

    assert result.decision == "accept"
    repair = calls[1]["messages"][-1]["content"]
    assert repair.startswith("Your previous response could not be parsed as valid JSON.\n\nPARSE ERROR: ")
    assert "SECRET_RAW_TEXT" not in repair


@pytest.mark.asyncio
async def test_single_semantic_reject_is_unchanged(monkeypatch):
    agent, calls = _validator(monkeypatch, {"decision": "reject", "reasoning": "Not novel.", "summary": ""})

    [result] = await agent.validate_batch([_submission(1)])

    assert (result.decision, result.summary, result.json_valid) == ("reject", "Not novel.", True)
    assert len(calls) == 1


# --- coordinator restore path ----------------------------------------------------


def _coordinator(monkeypatch, validate_batch, *, stop_after: int = 2):
    coordinator = Coordinator()
    coordinator.is_running = True
    decided: list[tuple[str, str]] = []
    broadcasts: list[tuple[str, dict]] = []
    persisted: list[tuple[str, str, dict]] = []
    retry_waits: list[dict] = []

    async def record_acceptance(submission, _result):
        decided.append((submission.submission_id, "accept"))
        if len(decided) == stop_after:
            coordinator.is_running = False

    async def record_rejection(submission, _result):
        decided.append((submission.submission_id, "reject"))

    async def no_pause():
        return None

    async def record_retry(payload, **_kwargs):
        retry_waits.append(payload)
        return True

    async def record_broadcast(event, data):
        broadcasts.append((event, data))

    async def record_persisted(event, message, metadata=None):
        persisted.append((event, message, metadata or {}))

    coordinator.validator = SimpleNamespace(validate_batch=validate_batch)
    coordinator.validator_model = "validator-model"
    coordinator.validator_provider = "lm_studio"
    monkeypatch.setattr(coordinator, "_handle_acceptance", record_acceptance)
    monkeypatch.setattr(coordinator, "_handle_rejection", record_rejection)
    monkeypatch.setattr(coordinator, "_check_and_update_pause_state", no_pause)
    monkeypatch.setattr(coordinator, "_broadcast", record_broadcast)
    monkeypatch.setattr(coordinator, "_add_persisted_event", record_persisted)
    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", record_retry)
    return coordinator, decided, broadcasts, persisted, retry_waits


def _accept_all(submissions):
    return [
        ValidationResult(
            submission_id=submission.submission_id,
            decision="accept",
            reasoning="Independent single decision.",
            summary="",
        )
        for submission in submissions
    ]


@pytest.mark.asyncio
async def test_coordinator_restores_unusable_batch_and_validates_it_one_at_a_time(monkeypatch):
    batch_sizes: list[int] = []

    async def validate_batch(submissions):
        batch_sizes.append(len(submissions))
        if len(batch_sizes) == 1:
            raise ValidatorOutputContractError("Expected 2 decisions for this batch, got 30", batch_size=2)
        return _accept_all(submissions)

    coordinator, decided, broadcasts, persisted, retry_waits = _coordinator(monkeypatch, validate_batch)
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
    # Operational activity, never a provider retry wait or a rejection.
    assert retry_waits == []
    [(event, payload)] = broadcasts
    assert event == VALIDATOR_CONTRACT_EVENT
    assert payload["activity_classification"] == "operational"
    assert payload["role_id"] == "aggregator_validator"
    assert payload["submission_ids"] == ["submission-1", "submission-2"]
    assert "No decision was recorded and no submitter is penalized" in payload["message"]
    assert "validated one at a time" in payload["message"]
    assert payload["contract_stage"] == "shape"
    assert payload["activity_category"] == "model_output"
    assert persisted[0][0] == VALIDATOR_CONTRACT_EVENT
    assert "rejected" not in VALIDATOR_CONTRACT_EVENT


@pytest.mark.asyncio
async def test_coordinator_requeues_single_contract_failure_without_submitter_feedback(monkeypatch):
    calls: list[int] = []

    async def validate_batch(submissions):
        calls.append(len(submissions))
        if len(calls) == 1:
            raise ValidatorOutputContractError(
                "The validator decision did not match the required JSON format",
                stage="missing_key",
                missing_keys=("decision",),
            )
        return _accept_all(submissions)

    coordinator, decided, broadcasts, _persisted, _waits = _coordinator(
        monkeypatch, validate_batch, stop_after=1
    )
    rejection_feedback: list = []
    coordinator.submitters = [
        SimpleNamespace(submitter_id=1, handle_rejection=lambda *a: rejection_feedback.append(a))
    ]
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await coordinator._validator_loop()
    finally:
        await queue_manager.clear()

    assert calls == [1, 1]  # the restored submission is validated again
    assert decided == [("submission-1", "accept")]
    assert rejection_feedback == []
    assert coordinator.total_rejections == 0
    [(_event, payload)] = broadcasts
    assert payload["contract_stage"] == "missing_key"
    assert payload["contract_fields"] == "decision"
    assert "will be validated again" in payload["message"]


@pytest.mark.asyncio
async def test_poisoned_submission_rotates_so_healthy_submissions_are_validated(monkeypatch):
    seen: list[list[str]] = []
    poison_failures = 0

    async def validate_batch(submissions):
        nonlocal poison_failures
        ids = [submission.submission_id for submission in submissions]
        seen.append(ids)
        if "submission-1" in ids:
            poison_failures += 1
            raise ValidatorOutputContractError(
                "The validator decision did not match the required JSON format",
                stage="missing_key",
                missing_keys=("decision",),
            )
        return _accept_all(submissions)

    coordinator, decided, broadcasts, _persisted, retry_waits = _coordinator(
        monkeypatch, validate_batch, stop_after=3
    )
    await queue_manager.clear()
    try:
        for number in (1, 2, 3, 4):
            await queue_manager.enqueue(_submission(number))
        await coordinator._validator_loop()
        remaining = list(queue_manager.queue)
    finally:
        await queue_manager.clear()

    # Every healthy submission is decided although the poisoned one never is.
    assert sorted(decided) == [
        ("submission-2", "accept"),
        ("submission-3", "accept"),
        ("submission-4", "accept"),
    ]
    # The poisoned submission is never dropped and never pinned at the front.
    assert [submission.submission_id for submission in remaining] == ["submission-1"]
    assert seen[0] == ["submission-1", "submission-2", "submission-3"]
    assert seen[1] == ["submission-1"]
    assert seen[2] == ["submission-2"]
    rotations = [payload for _event, payload in broadcasts if payload.get("requeue_position") == "back"]
    assert rotations and "moved to the back of the queue" in rotations[0]["message"]
    # The poisoned submission failing twice in a row paces the next try; the
    # healthy validations in between reset the count.
    assert [wait["retry_after_seconds"] for wait in retry_waits] == [1]
    assert retry_waits[0]["reason"] == "model_output_contract_retry"
    assert coordinator.total_rejections == 0


@pytest.mark.asyncio
async def test_single_model_workflow_restores_unusable_reply(monkeypatch):
    calls: list[int] = []

    async def validate_batch(submissions):
        calls.append(len(submissions))
        if len(calls) == 1:
            raise ValidatorOutputContractError("The validator reply was not a JSON object", batch_size=2)
        return _accept_all(submissions)

    coordinator, decided, broadcasts, _persisted, retry_waits = _coordinator(monkeypatch, validate_batch)
    coordinator.submitters = []
    coordinator.validator.role_id = "aggregator_validator"
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await queue_manager.enqueue(_submission(2))
        await coordinator._single_model_workflow()
    finally:
        await queue_manager.clear()

    assert calls == [2, 1, 1]
    assert decided == [("submission-1", "accept"), ("submission-2", "accept")]
    assert retry_waits == []
    assert [event for event, _payload in broadcasts] == [VALIDATOR_CONTRACT_EVENT]


@pytest.mark.asyncio
async def test_internal_batch_interruption_is_not_a_contract_failure(monkeypatch):
    batch_sizes: list[int] = []

    async def validate_batch(submissions):
        batch_sizes.append(len(submissions))
        if len(batch_sizes) == 1:
            raise BatchValidationInterruptedError(error_type="KeyError", batch_size=2)
        return _accept_all(submissions)

    coordinator, decided, broadcasts, persisted, retry_waits = _coordinator(monkeypatch, validate_batch)
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await queue_manager.enqueue(_submission(2))
        await coordinator._validator_loop()
    finally:
        await queue_manager.clear()

    # The batch is still restored and validated one submission at a time.
    assert batch_sizes == [2, 1, 1]
    assert decided == [("submission-1", "accept"), ("submission-2", "accept")]
    [(event, payload)] = broadcasts
    assert event == VALIDATOR_BATCH_INTERRUPTED_EVENT != VALIDATOR_CONTRACT_EVENT
    assert payload["activity_classification"] == "operational"
    assert payload["activity_category"] == "internal"
    assert payload["error_type"] == "KeyError"
    assert "contract_stage" not in payload and "error_summary" not in payload
    assert "not a model output format problem" in payload["message"]
    assert persisted[0][0] == VALIDATOR_BATCH_INTERRUPTED_EVENT
    assert coordinator.validator_contract_failures == 0
    # Paced through the shared stop-aware seam, never a contract retry.
    [wait] = retry_waits
    assert wait["retry_after_seconds"] == 2
    assert wait["reason"] == VALIDATOR_BATCH_INTERRUPTED_EVENT
    assert wait["retry_layer"] == "workflow"
    assert wait["role_id"] == "aggregator_validator"


@pytest.mark.asyncio
async def test_repeated_validator_contract_failures_back_off_without_a_cap(monkeypatch):
    failures_before_success = 9
    calls: list[int] = []
    stop_checks: list[bool] = []

    async def validate_batch(submissions):
        calls.append(len(submissions))
        if len(calls) <= failures_before_success:
            raise ValidatorOutputContractError(
                "The validator decision did not match the required JSON format",
                stage="missing_key",
                missing_keys=("decision",),
            )
        return _accept_all(submissions)

    coordinator, decided, broadcasts, _persisted, retry_waits = _coordinator(
        monkeypatch, validate_batch, stop_after=1
    )

    async def record_retry(payload, **kwargs):
        retry_waits.append(payload)
        stop_checks.append(kwargs["should_stop"]())
        return True

    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", record_retry)
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await coordinator._validator_loop()
    finally:
        await queue_manager.clear()

    # Every failure is tried again: nothing is capped, dropped, or rejected.
    assert len(calls) == failures_before_success + 1
    assert decided == [("submission-1", "accept")]
    # The first failure retries at once; later consecutive ones wait longer.
    assert [wait["retry_after_seconds"] for wait in retry_waits] == [1, 2, 4, 8, 16, 30, 30, 30]
    assert {wait["reason"] for wait in retry_waits} == {"model_output_contract_retry"}
    assert all(wait["retry_layer"] == "workflow" for wait in retry_waits)
    assert stop_checks == [False] * len(retry_waits)  # the wait is stop-aware
    assert coordinator._consecutive_validator_contract_failures == 0  # reset on success
    assert [payload["consecutive_failures"] for _event, payload in broadcasts] == list(
        range(1, failures_before_success + 1)
    )


@pytest.mark.asyncio
async def test_successful_validation_resets_the_contract_failure_backoff(monkeypatch):
    calls: list[int] = []

    async def validate_batch(submissions):
        calls.append(len(submissions))
        if len(calls) in (1, 2, 4):  # fail, fail, succeed, fail, succeed
            raise ValidatorOutputContractError("Expected 2 decisions for this batch, got 30", batch_size=1)
        return _accept_all(submissions)

    coordinator, decided, _broadcasts, _persisted, retry_waits = _coordinator(monkeypatch, validate_batch)
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await queue_manager.enqueue(_submission(2))
        await coordinator._validator_loop()
    finally:
        await queue_manager.clear()

    assert sorted(decided) == [("submission-1", "accept"), ("submission-2", "accept")]
    # Only the second consecutive failure waited; the one after a success did not.
    assert [wait["retry_after_seconds"] for wait in retry_waits] == [1]


@pytest.mark.asyncio
async def test_stopped_workflow_does_not_wait_after_a_contract_failure(monkeypatch):
    async def validate_batch(_submissions):
        coordinator.is_running = False
        raise ValidatorOutputContractError("Expected 2 decisions for this batch, got 30", batch_size=1)

    coordinator, _decided, _broadcasts, _persisted, retry_waits = _coordinator(monkeypatch, validate_batch)
    coordinator._consecutive_validator_contract_failures = 4
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await coordinator._validator_loop()
        remaining = [submission.submission_id for submission in queue_manager.queue]
    finally:
        await queue_manager.clear()

    assert retry_waits == []
    assert remaining == ["submission-1"]


@pytest.mark.asyncio
async def test_each_contract_failure_has_its_own_activity_identity_and_run_scope(monkeypatch):
    calls: list[int] = []

    async def validate_batch(submissions):
        calls.append(len(submissions))
        if len(calls) <= 2:
            raise ValidatorOutputContractError("Expected 2 decisions for this batch, got 30", batch_size=1)
        return _accept_all(submissions)

    coordinator, _decided, broadcasts, persisted, _waits = _coordinator(
        monkeypatch, validate_batch, stop_after=1
    )
    coordinator.proof_context_requesting_run_id = "run-abc"
    monkeypatch.setattr(coordinator, "_brainstorm_context_source_id", lambda: "topic-7")
    await queue_manager.clear()
    try:
        await queue_manager.enqueue(_submission(1))
        await coordinator._validator_loop()
    finally:
        await queue_manager.clear()

    first, second = (payload for _event, payload in broadcasts)
    # Identical failures on the same submission must not collapse into one row.
    assert first["event_id"].startswith(f"{VALIDATOR_CONTRACT_EVENT}:")
    assert first["event_id"] != second["event_id"]
    assert first["activity_id"] == first["event_id"]
    assert first["activity_id"] != second["activity_id"]
    for payload in (first, second):
        assert payload["run_id"] == "run-abc"
        assert payload["activity_run_id"] == "run-abc"
        assert payload["source_id"] == "topic-7"
    assert [metadata["event_id"] for _event, _message, metadata in persisted] == [
        first["event_id"],
        second["event_id"],
    ]


@pytest.mark.parametrize(
    ("file_name", "expected_source"),
    [
        ("brainstorm_topic-7.txt", "topic-7"),
        ("rag_shared_training.txt", "manual_aggregator"),
        ("exploration_candidates.txt", None),
        ("title_candidates_topic-7.txt", None),
    ],
)
def test_validator_activity_names_only_a_real_source(monkeypatch, tmp_path, file_name, expected_source):
    """Topic/title exploration children must not claim the manual Aggregator as their source."""
    coordinator = Coordinator()
    monkeypatch.setattr(
        coordinator_module.shared_training_memory, "file_path", str(tmp_path / file_name)
    )
    scope = coordinator._activity_scope_fields()
    assert scope.get("source_id") == expected_source


@pytest.mark.asyncio
async def test_single_model_workflow_drains_a_full_queue_before_generating(monkeypatch):
    generated: list[int] = []
    validated: list[str] = []

    async def validate_batch(submissions):
        validated.extend(submission.submission_id for submission in submissions)
        return _accept_all(submissions)

    coordinator, decided, _broadcasts, _persisted, _waits = _coordinator(
        monkeypatch, validate_batch, stop_after=4
    )
    # Use the real pause logic so the queue overflow threshold applies.
    monkeypatch.setattr(coordinator, "_check_and_update_pause_state", Coordinator._check_and_update_pause_state.__get__(coordinator))
    monkeypatch.setattr(coordinator_module.system_config, "queue_overflow_threshold", 3)
    coordinator.validator.role_id = "aggregator_validator"

    def _make_submitter(submitter_id):
        async def generate():
            generated.append(submitter_id)
            return None

        return SimpleNamespace(
            submitter_id=submitter_id,
            role_id=f"aggregator_submitter_{submitter_id}",
            _generate_submission=generate,
        )

    coordinator.submitters = [_make_submitter(1), _make_submitter(2)]
    await queue_manager.clear()
    try:
        for number in (1, 2, 3, 4):
            await queue_manager.enqueue(_submission(number))
        await coordinator._single_model_workflow()
    finally:
        await queue_manager.clear()

    # At the overflow threshold no new submission was generated; the restored
    # and pending work was validated first.
    assert generated == []
    assert sorted(validated) == ["submission-1", "submission-2", "submission-3", "submission-4"]
    assert len(decided) == 4


@pytest.mark.asyncio
async def test_single_model_workflow_generates_while_queue_is_below_threshold(monkeypatch):
    generated: list[int] = []

    async def validate_batch(submissions):
        return _accept_all(submissions)

    coordinator, decided, _broadcasts, _persisted, _waits = _coordinator(
        monkeypatch, validate_batch, stop_after=1
    )
    monkeypatch.setattr(coordinator, "_check_and_update_pause_state", Coordinator._check_and_update_pause_state.__get__(coordinator))
    monkeypatch.setattr(coordinator_module.system_config, "queue_overflow_threshold", 3)
    coordinator.validator.role_id = "aggregator_validator"

    async def generate():
        generated.append(1)
        return _submission(1)

    async def enqueue(submission):
        await queue_manager.enqueue(submission)

    monkeypatch.setattr(coordinator, "add_submission_to_queue", enqueue)
    coordinator.submitters = [
        SimpleNamespace(submitter_id=1, role_id="aggregator_submitter_1", _generate_submission=generate)
    ]
    await queue_manager.clear()
    try:
        await coordinator._single_model_workflow()
    finally:
        await queue_manager.clear()

    assert generated == [1]
    assert decided == [("submission-1", "accept")]


@pytest.mark.asyncio
async def test_cleanup_review_json_repair_runs(monkeypatch):
    """The cleanup-review repair helper used to fail with a NameError (count_tokens)."""
    agent, calls = _validator(monkeypatch, {"should_remove": False, "reasoning": "Keep all."})

    repaired = await agent._retry_validator_json_parse("Review prompt.", "{not json", "agg_val_007")

    assert repaired == {"should_remove": False, "reasoning": "Keep all."}
    assert calls[0]["task_id"] == "agg_val_007_json_retry"
