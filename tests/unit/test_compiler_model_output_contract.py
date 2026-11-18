"""Compiler model-output contract handling.

Covers the strict critique parser, the compiler validator's operational
contract failures, the Rigor & Proofs critique and discovery contract retries,
and the coordinator paths that keep contract failures out of rejection
feedback and provider retry waits.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.compiler.agents import high_param_submitter as hp_module
from backend.compiler.agents.high_param_submitter import (
    RIGOR_DISCOVERY_SCHEMA,
    HighParamSubmitter,
)
from backend.compiler.core import compiler_coordinator as coordinator_module
from backend.compiler.core.compiler_coordinator import (
    CompilerCoordinator,
    _classify_submitter_error,
)
from backend.compiler.validation import compiler_validator as validator_module
from backend.compiler.validation.compiler_validator import (
    CompilerValidator,
    ValidatorContractError,
    ValidatorPromptTooLargeError,
)
from backend.shared.config import system_config
from backend.shared.critique_prompts import parse_critique_response
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_transient_model_call_error,
)
from backend.shared.model_output_contract import ModelOutputContractError
from backend.shared.models import CompilerSubmission, CompilerValidationResult, Submission


def _critique(**overrides):
    data = {
        "novelty_rating": 7,
        "novelty_feedback": "Novel framing of the invariant.",
        "correctness_rating": 6,
        "correctness_feedback": "One step needs a citation.",
        "impact_rating": 5,
        "impact_feedback": "Moderate field impact.",
        "full_critique": "A careful but incremental paper.",
    }
    data.update(overrides)
    return data


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# --------------------------------------------------------------- critique parser


def test_critique_parser_accepts_valid_reply_and_keeps_extension():
    payload = _critique(solution_path_update={"kind": "note"})
    data = parse_critique_response(json.dumps(payload))
    assert data["novelty_rating"] == 7
    assert data["solution_path_update"] == {"kind": "note"}


def test_critique_parser_relabels_one_misspelled_feedback_field():
    payload = _critique()
    text = json.dumps(payload).replace('"correctness_feedback"', '"correctness_feedbak"')
    repairs: list = []

    data = parse_critique_response(text, repairs=repairs)

    assert data["correctness_feedback"] == "One step needs a citation."
    assert "correctness_feedbak" not in data
    assert [(r.kind, r.from_key, r.to_key) for r in repairs] == [
        ("key_label", "correctness_feedbak", "correctness_feedback")
    ]


def test_critique_parser_never_relabels_a_rating():
    text = json.dumps(_critique()).replace('"impact_rating"', '"impact_ratng"')
    with pytest.raises(ModelOutputContractError) as raised:
        parse_critique_response(text)
    assert raised.value.stage == "missing_key"
    assert "impact_rating" in raised.value.missing_keys
    assert "impact_ratng" in raised.value.unknown_keys


def test_critique_parser_drops_stray_key_position_character():
    text = (
        '{\n  "novelty_rating": 7,\n  "novelty_feedback": "a",\n'
        '  "correctness_rating": 6,\n  "correctness_feedback": "b",\n'
        '  "impact_rating": 5,\n  "impact_feedback": "c",\n'
        's     "full_critique": "d"\n}'
    )
    repairs: list = []
    data = parse_critique_response(text, repairs=repairs)
    assert data["full_critique"] == "d"
    assert [repair.kind for repair in repairs] == ["stray_characters"]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"novelty_rating": 0}, "novelty_rating"),
        ({"correctness_rating": 11}, "correctness_rating"),
        ({"impact_rating": True}, "impact_rating"),
        ({"impact_rating": "7"}, "impact_rating"),
        ({"novelty_feedback": "  "}, "novelty_feedback"),
        ({"full_critique": None}, "full_critique"),
    ],
)
def test_critique_parser_rejects_invalid_values_without_defaults(overrides, field):
    with pytest.raises(ModelOutputContractError) as raised:
        parse_critique_response(json.dumps(_critique(**overrides)))
    assert raised.value.stage == "invalid_value"
    assert field in raised.value.invalid_fields


def test_critique_parser_never_fabricates_ratings_from_truncated_output():
    truncated = json.dumps(_critique())[:-40]
    with pytest.raises(ModelOutputContractError) as raised:
        parse_critique_response(truncated)
    assert raised.value.stage == "syntax"
    # The message is model-safe: no raw output excerpt.
    assert "Novel framing" not in str(raised.value)


def test_critique_contract_error_is_not_a_provider_failure():
    with pytest.raises(ModelOutputContractError) as raised:
        parse_critique_response("not json at all, credits exhausted, quota timeout")
    assert not is_non_retryable_model_error(raised.value)
    assert not is_transient_model_call_error(raised.value)


# --------------------------------------------------------------- compiler validator


def _valid_validation(**overrides):
    data = {
        "decision": "accept",
        "reasoning": "Placement and rigor hold.",
        "coherence_check": True,
        "rigor_check": True,
        "placement_check": True,
    }
    data.update(overrides)
    return data


@pytest.fixture
def validator(monkeypatch):
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 8000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 1000)
    monkeypatch.setattr(
        validator_module.paper_memory, "ensure_markers_intact", AsyncMock(return_value=False)
    )
    reported: list = []

    async def fake_report(role_id, repairs, **kwargs):
        reported.append((role_id, list(repairs), kwargs))

    monkeypatch.setattr(validator_module, "report_model_output_normalizations", fake_report)
    instance = CompilerValidator(model_name="validator-model", user_prompt="Solve it.")
    instance.reported = reported
    return instance


def _construction_submission() -> CompilerSubmission:
    return CompilerSubmission(
        submission_id="sub-1",
        mode="construction",
        operation="insert_after",
        old_string="Anchor.",
        new_string="New material.",
        content="New material.",
        reasoning="Add material.",
    )


@pytest.mark.asyncio
async def test_validator_prompt_too_large_follows_context_overflow(validator, monkeypatch):
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 4000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 200)
    generate = AsyncMock()
    monkeypatch.setattr(validator_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(ValidatorPromptTooLargeError) as raised:
        await validator.validate_submission(
            _construction_submission(),
            current_paper="Anchor. " + "filler " * 8000,
            current_outline="I. Introduction\nII. Body\nIII. Conclusion",
        )

    generate.assert_not_called()
    assert raised.value.required_tokens > raised.value.available_tokens
    assert _classify_submitter_error(raised.value)[0] == "Context overflow"


@pytest.mark.asyncio
async def test_validator_empty_reply_takes_bounded_retry(validator, monkeypatch):
    replies = iter([{"choices": []}, _completion(json.dumps(_valid_validation()))])
    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        AsyncMock(side_effect=lambda **kwargs: next(replies)),
    )

    result = await validator.validate_submission(
        _construction_submission(),
        current_paper="Anchor.",
        current_outline="I. Introduction\nII. Body\nIII. Conclusion",
    )

    assert result.decision == "accept"
    assert result.json_valid is True


@pytest.mark.asyncio
async def test_validator_contract_failure_raises_naming_the_field(validator, monkeypatch):
    bad = json.dumps(_valid_validation(placement_check="yes"))
    generate = AsyncMock(return_value=_completion(bad))
    monkeypatch.setattr(validator_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(ValidatorContractError) as raised:
        await validator.validate_submission(
            _construction_submission(),
            current_paper="Anchor.",
            current_outline="I. Introduction\nII. Body\nIII. Conclusion",
        )

    assert generate.await_count == 2  # original call plus exactly one retry
    assert raised.value.stage == "invalid_value"
    assert raised.value.invalid_fields == ("placement_check",)
    assert "placement_check" in str(raised.value)


@pytest.mark.asyncio
async def test_validator_reports_applied_normalizations(validator, monkeypatch):
    text = json.dumps(_valid_validation(), indent=2)[:-2] + ",\n}"
    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        AsyncMock(return_value=_completion(text)),
    )

    result = await validator.validate_submission(
        _construction_submission(),
        current_paper="Anchor.",
        current_outline="I. Introduction\nII. Body\nIII. Conclusion",
    )

    assert result.decision == "accept"
    role_id, repairs, kwargs = validator.reported[0]
    assert role_id == "compiler_validator"
    assert [repair.kind for repair in repairs] == ["trailing_comma"]
    assert kwargs["task_id"] == "comp_val_000"


@pytest.mark.asyncio
async def test_brainstorm_operation_contract_failure_is_not_a_rejection(validator, monkeypatch):
    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        AsyncMock(return_value=_completion("no json here")),
    )
    op = SimpleNamespace(action="delete", submission_number=1, new_content="", reasoning="dup")
    monkeypatch.setattr(validator, "_build_brainstorm_validation_prompt", lambda *_: "PROMPT")

    with pytest.raises(ValidatorContractError):
        await validator.validate_brainstorm_operation(op, "brainstorm")


# --------------------------------------------------------------- coordinator validation


def _coordinator(running: bool = True) -> CompilerCoordinator:
    coordinator = CompilerCoordinator()
    coordinator.is_running = running
    coordinator.current_mode = "construction"
    coordinator.broadcasts = []

    async def capture(event, data):
        coordinator.broadcasts.append((event, data))

    coordinator.websocket_broadcaster = capture
    return coordinator


def _accept_result() -> CompilerValidationResult:
    return CompilerValidationResult(submission_id="sub-1", decision="accept", reasoning="ok")


@pytest.mark.asyncio
async def test_validator_contract_failure_revalidates_same_submission(monkeypatch):
    coordinator = _coordinator()
    add_rejection = AsyncMock()
    monkeypatch.setattr(coordinator_module.compiler_rejection_log, "add_rejection", add_rejection)
    emit = AsyncMock()
    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", emit)
    calls = []

    async def validate():
        calls.append(1)
        if len(calls) == 1:
            raise ValidatorContractError("Validation JSON contract remained invalid", stage="syntax")
        return _accept_result()

    result = await coordinator._validate_until_decided(validate, mode="construction", submission_id="sub-1")

    assert result.decision == "accept"
    assert len(calls) == 2
    add_rejection.assert_not_called()
    emit.assert_not_called()
    events = [event for event, _ in coordinator.broadcasts]
    assert events == ["compiler_warning"]
    payload = coordinator.broadcasts[0][1]
    assert payload["reason"] == "model_output_contract_failure"
    assert payload["role_id"] == "compiler_validator"
    assert payload["activity_classification"] == "operational"
    assert "not a rejection" in payload["warning"]


@pytest.mark.asyncio
async def test_validator_contract_failure_after_stop_returns_none():
    coordinator = _coordinator(running=False)
    validate = AsyncMock(side_effect=ValidatorContractError("bad", stage="syntax"))

    assert await coordinator._validate_until_decided(validate, mode="review") is None
    assert validate.await_count == 1


@pytest.mark.asyncio
async def test_validator_prompt_too_large_stops_through_context_overflow():
    coordinator = _coordinator()
    error = ValidatorPromptTooLargeError(required_tokens=9000, available_tokens=4000)
    validate = AsyncMock(side_effect=error)

    assert await coordinator._validate_until_decided(validate, mode="review") is None

    assert coordinator.is_running is False
    events = dict(coordinator.broadcasts)
    overflow = events["context_overflow_error"]
    assert overflow["role_id"] == "compiler_validator"
    assert overflow["required_tokens"] == 9000


# --------------------------------------------------------------- critique submitter


@pytest.fixture
def critique_submitter(monkeypatch):
    monkeypatch.setattr(system_config, "compiler_high_param_context_window", 10000)
    monkeypatch.setattr(system_config, "compiler_high_param_max_output_tokens", 1000)
    submitter = HighParamSubmitter(
        model_name="rigor-model",
        user_prompt="prove it",
        validator_context_window=10000,
        validator_max_tokens=1000,
    )
    submitter.critique_rejection_memory = SimpleNamespace(
        get_rejection_entries=AsyncMock(return_value=[]),
        render_rejections=lambda selected: "",
    )
    monkeypatch.setattr(hp_module, "build_critique_prompt", lambda **kwargs: "CRITIQUE PROMPT")
    submitter.reported = []

    async def fake_report(role_id, repairs, **kwargs):
        submitter.reported.append((role_id, list(repairs), kwargs))

    monkeypatch.setattr(hp_module, "report_model_output_normalizations", fake_report)
    return submitter


async def _submit(submitter):
    return await submitter.submit_critique(
        user_prompt="prove it",
        current_body="body",
        current_outline="outline",
        aggregator_db="",
    )


@pytest.mark.asyncio
async def test_submit_critique_retries_once_and_names_the_field(critique_submitter, monkeypatch):
    replies = [
        _completion(json.dumps({"critique_needed": "yes", "submission": "x", "reasoning": "r"})),
        _completion(json.dumps({"critique_needed": True, "submission": "Gap in step 3.", "reasoning": "r"})),
    ]
    generate = AsyncMock(side_effect=replies)
    monkeypatch.setattr(hp_module.api_client_manager, "generate_completion", generate)

    submission = await _submit(critique_submitter)

    assert isinstance(submission, Submission)
    assert submission.content == "Gap in step 3."
    assert submission.is_decline is False
    retry_messages = generate.await_args_list[1].kwargs["messages"]
    assert "critique_needed" in retry_messages[-1]["content"]
    assert generate.await_args_list[1].kwargs["task_id"].endswith("_retry")


@pytest.mark.asyncio
async def test_submit_critique_raises_contract_error_instead_of_none(critique_submitter, monkeypatch):
    generate = AsyncMock(return_value=_completion(json.dumps({"submission": "x"})))
    monkeypatch.setattr(hp_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(ModelOutputContractError) as raised:
        await _submit(critique_submitter)

    assert generate.await_count == 2
    assert raised.value.stage == "missing_key"
    assert set(raised.value.missing_keys) == {"critique_needed", "reasoning"}


@pytest.mark.asyncio
async def test_submit_critique_accepts_decline_without_submission(critique_submitter, monkeypatch):
    monkeypatch.setattr(
        hp_module.api_client_manager,
        "generate_completion",
        AsyncMock(return_value=_completion(json.dumps({"critique_needed": False, "reasoning": "Sound."}))),
    )
    submission = await _submit(critique_submitter)
    assert submission.is_decline is True
    assert submission.content == ""


@pytest.mark.asyncio
async def test_submit_critique_prompt_too_large_is_context_overflow(critique_submitter, monkeypatch):
    monkeypatch.setattr(hp_module, "build_critique_prompt", lambda **kwargs: "word " * 20000)
    generate = AsyncMock()
    monkeypatch.setattr(hp_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(ValueError) as raised:
        await _submit(critique_submitter)

    generate.assert_not_called()
    assert _classify_submitter_error(raised.value)[0] == "Context overflow"


# --------------------------------------------------------------- rigor discovery


def _discovery(**overrides):
    data = {
        "needs_theorem_work": True,
        "theorem_statement": "For all n, P(n).",
        "formal_sketch": "induction",
        "source_excerpt": "Section 2.",
        "theorem_origin": "existing_paper_claim",
        "placement_preference": "inline",
        "expected_novelty_tier": "mathematical_discovery",
        "prompt_relevance_rationale": "Needed for the main argument.",
        "novelty_rationale": "Not in Mathlib.",
        "why_not_standard_known_result": "Paper-specific objects.",
        "retry_existing_failure_id": "",
        "reasoning": "Best target.",
    }
    data.update(overrides)
    return data


@pytest.fixture
def rigor_submitter(critique_submitter, monkeypatch):
    monkeypatch.setattr(
        hp_module.lm_studio_client, "cache_model_load_config", AsyncMock(return_value=None)
    )
    return critique_submitter


@pytest.mark.asyncio
async def test_discovery_relabels_misspelled_rationale(rigor_submitter, monkeypatch):
    text = json.dumps(_discovery()).replace('"novelty_rationale"', '"novelty_rationally"')
    monkeypatch.setattr(
        hp_module.api_client_manager, "generate_completion", AsyncMock(return_value=_completion(text))
    )

    reply = await rigor_submitter._call_llm_and_parse(
        prompt="P", task_label="rigor_discovery", schema=RIGOR_DISCOVERY_SCHEMA
    )

    assert reply.data["novelty_rationale"] == "Not in Mathlib."
    assert [(r.from_key, r.to_key) for r in reply.repairs] == [
        ("novelty_rationally", "novelty_rationale")
    ]
    # Parsing alone never publishes; the caller publishes once its checks pass.
    assert rigor_submitter.reported == []
    await rigor_submitter._publish_rigor_repairs(reply)
    role_id, repairs, _ = rigor_submitter.reported[0]
    assert role_id == "compiler_high_param"
    assert [(r.from_key, r.to_key) for r in repairs] == [("novelty_rationally", "novelty_rationale")]


@pytest.mark.asyncio
async def test_discovery_never_relabels_protected_decision_key(rigor_submitter, monkeypatch):
    payload = _discovery()
    payload["needs_theorem_wrk"] = payload.pop("needs_theorem_work")
    monkeypatch.setattr(
        hp_module.api_client_manager,
        "generate_completion",
        AsyncMock(return_value=_completion(json.dumps(payload))),
    )

    reply = await rigor_submitter._call_llm_and_parse(
        prompt="P", task_label="rigor_discovery", schema=RIGOR_DISCOVERY_SCHEMA
    )

    assert "needs_theorem_work" not in reply.data
    assert reply.repairs == ()


def _patch_rigor_context(submitter, monkeypatch):
    for name in ("get_all_proofs_for_live_context", "get_recent_failure_hints"):
        monkeypatch.setattr(
            submitter.proof_database, name, AsyncMock(return_value=[]), raising=False
        )
    monkeypatch.setattr(hp_module.outline_memory, "get_outline", AsyncMock(return_value="I. Body"))
    monkeypatch.setattr(hp_module.paper_memory, "get_paper", AsyncMock(return_value="Paper body."))
    monkeypatch.setattr(
        hp_module.compiler_rejection_log, "get_rejection_entries", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        hp_module.compiler_rag_manager,
        "retrieve_for_mode",
        AsyncMock(return_value=SimpleNamespace(text="")),
    )
    submitter.available_input_tokens = 8000
    return submitter


def _without_decision():
    payload = _discovery()
    del payload["needs_theorem_work"]
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_reply, field",
    [
        (json.dumps(_discovery(needs_theorem_work="false")), "needs_theorem_work"),
        (json.dumps(_without_decision()), "needs_theorem_work"),
        ("42", ""),
    ],
)
async def test_discovery_decision_must_be_a_json_bool(rigor_submitter, monkeypatch, bad_reply, field):
    _patch_rigor_context(rigor_submitter, monkeypatch)
    generate = AsyncMock(return_value=_completion(bad_reply))
    monkeypatch.setattr(hp_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(ModelOutputContractError) as raised:
        await rigor_submitter._step_discovery()

    # One bounded contract retry, then the coordinator's operational path.
    assert generate.await_count == 2
    if field:
        assert field in raised.value.field_summary
    assert rigor_submitter.reported == []


@pytest.mark.asyncio
async def test_discovery_string_false_then_bool_false_is_a_decline(rigor_submitter, monkeypatch):
    _patch_rigor_context(rigor_submitter, monkeypatch)
    generate = AsyncMock(
        side_effect=[
            _completion(json.dumps(_discovery(needs_theorem_work="false"))),
            _completion(json.dumps(_discovery(needs_theorem_work=False))),
        ]
    )
    monkeypatch.setattr(hp_module.api_client_manager, "generate_completion", generate)

    assert await rigor_submitter._step_discovery() is None
    assert generate.await_count == 2


@pytest.mark.asyncio
async def test_discovery_publishes_relabel_only_when_proposal_is_accepted(rigor_submitter, monkeypatch):
    _patch_rigor_context(rigor_submitter, monkeypatch)
    accepted = json.dumps(_discovery()).replace('"novelty_rationale"', '"novelty_rationally"')
    monkeypatch.setattr(
        hp_module.api_client_manager, "generate_completion", AsyncMock(return_value=_completion(accepted))
    )
    data = await rigor_submitter._step_discovery()
    assert data["novelty_rationale"] == "Not in Mathlib."
    assert len(rigor_submitter.reported) == 1

    rigor_submitter.reported.clear()
    refused = json.dumps(_discovery(expected_novelty_tier="")).replace(
        '"novelty_rationale"', '"novelty_rationally"'
    )
    monkeypatch.setattr(
        hp_module.api_client_manager, "generate_completion", AsyncMock(return_value=_completion(refused))
    )
    assert await rigor_submitter._step_discovery() is None
    assert rigor_submitter.reported == []


@pytest.mark.asyncio
async def test_placement_publishes_normalization_only_after_anchor_checks(rigor_submitter, monkeypatch):
    _patch_rigor_context(rigor_submitter, monkeypatch)

    async def place(payload):
        text = json.dumps(payload)
        text = text[:-1] + ",}"  # a trailing comma the shared parser normalizes
        monkeypatch.setattr(
            hp_module.api_client_manager,
            "generate_completion",
            AsyncMock(return_value=_completion(text)),
        )
        return await rigor_submitter._build_placement_submission(
            proof_id="proof_1",
            theorem_statement="T",
            theorem_name="t",
            lean_code="theorem t : True := trivial",
            is_novel=True,
            placement_attempt=1,
            validator_rejection_feedback="",
        )

    refused = await place({"proceed": True, "old_string": "", "new_string": "x", "reasoning": "r"})
    assert refused is None
    assert rigor_submitter.reported == []

    submission = await place(
        {
            "proceed": True,
            "operation": "insert_after",
            "old_string": "Paper body.",
            "new_string": "x",
            "reasoning": "r",
        }
    )
    assert submission is not None
    assert [[r.kind for r in repairs] for _, repairs, _ in rigor_submitter.reported] == [
        ["trailing_comma"]
    ]


def test_discovery_schema_only_marks_rationales_repairable():
    repairable = {spec.name for spec in RIGOR_DISCOVERY_SCHEMA.keys if spec.repairable}
    assert repairable == {
        "prompt_relevance_rationale",
        "novelty_rationale",
        "why_not_standard_known_result",
    }


@pytest.mark.asyncio
async def test_rigor_llm_contract_failure_raises_after_one_retry(rigor_submitter, monkeypatch):
    generate = AsyncMock(return_value=_completion("still prose, no json"))
    monkeypatch.setattr(hp_module.api_client_manager, "generate_completion", generate)

    with pytest.raises(ModelOutputContractError):
        await rigor_submitter._call_llm_and_parse(
            prompt="P", task_label="rigor_discovery", schema=RIGOR_DISCOVERY_SCHEMA
        )
    assert generate.await_count == 2


@pytest.mark.asyncio
async def test_rigor_discovery_contract_failure_is_operational_not_decline(monkeypatch):
    coordinator = _coordinator()

    async def stop_then_fail():
        # Stop while the failed reply is reported: the regeneration loop ends.
        coordinator.is_running = False
        raise ModelOutputContractError("The response was not valid JSON", stage="syntax")

    coordinator.allow_mathematical_proofs = True
    monkeypatch.setattr(system_config, "lean4_enabled", True)
    coordinator.high_param_submitter = SimpleNamespace(
        set_source_material_context=lambda *args: None,
        submit_rigor_lean_theorem=stop_then_fail,
    )
    monkeypatch.setattr(
        coordinator, "_load_rigor_source_material_context", AsyncMock(return_value=("", ""))
    )
    add_decline = AsyncMock()
    monkeypatch.setattr(coordinator_module.compiler_rejection_log, "add_decline", add_decline)

    assert await coordinator._submit_and_validate_rigor() is False

    add_decline.assert_not_called()
    assert [event for event, _ in coordinator.broadcasts] == ["compiler_warning"]
    assert coordinator.rigor_declines == 0


@pytest.mark.asyncio
async def test_rigor_discovery_contract_failures_regenerate_with_paced_backoff(monkeypatch):
    coordinator = _coordinator()
    coordinator.allow_mathematical_proofs = True
    monkeypatch.setattr(system_config, "lean4_enabled", True)
    error = ModelOutputContractError(
        "The theorem-discovery decision must be a JSON boolean",
        stage="invalid_value",
        invalid_fields=("needs_theorem_work",),
    )
    submit = AsyncMock(side_effect=[error, error, None])
    coordinator.high_param_submitter = SimpleNamespace(
        set_source_material_context=lambda *args: None,
        submit_rigor_lean_theorem=submit,
    )
    monkeypatch.setattr(
        coordinator, "_load_rigor_source_material_context", AsyncMock(return_value=("", ""))
    )
    monkeypatch.setattr(coordinator_module.compiler_rejection_log, "add_decline", AsyncMock())
    emit = AsyncMock()
    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", emit)

    # Regenerated until the model gave a valid (declining) reply.
    assert await coordinator._submit_and_validate_rigor() is False
    assert submit.await_count == 3

    warnings = [data for event, data in coordinator.broadcasts if event == "compiler_warning"]
    assert [w["retry_attempt"] for w in warnings] == [1, 2]
    assert all(w["reason"] == "model_output_contract_failure" for w in warnings)
    assert len({w["event_id"] for w in warnings}) == 2
    assert all(w["event_id"].startswith("compiler_warning:") for w in warnings)
    # First failure retries at once; the second waits through the shared seam.
    emit.assert_awaited_once()
    wait_payload = emit.await_args.args[0]
    assert wait_payload["reason"] == "model_output_contract_retry"
    assert wait_payload["retry_layer"] == "workflow"
    assert wait_payload["retry_attempt"] == 2



# --------------------------------------------------------------- critique loop


@pytest.mark.asyncio
async def test_critique_loop_contract_failures_are_not_provider_waits(monkeypatch):
    coordinator = _coordinator()
    coordinator.in_critique_phase = True
    coordinator.paper_version = 1
    coordinator.user_prompt = "prove it"
    decline = Submission(
        submission_id="c-1",
        submitter_id=1,
        content="",
        reasoning="Body is sound.",
        chunk_size_used=512,
        is_decline=True,
    )
    submit = AsyncMock(
        side_effect=[ModelOutputContractError("The critique response was empty", stage="syntax")]
        + [decline] * 3
    )
    coordinator.critique_submitter = SimpleNamespace(
        role_id="compiler_high_param",
        submit_critique=submit,
        handle_critique_rejection=AsyncMock(),
    )
    validations = iter(
        [ModelOutputContractError("bad validator reply", role="critique_validator", stage="syntax")]
        + [
            SimpleNamespace(decision="accept", reasoning="Agreed.", summary="")
            for _ in range(3)
        ]
    )

    async def validate(_submission):
        item = next(validations)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(coordinator, "_validate_critique", validate)
    monkeypatch.setattr(coordinator, "_get_reference_papers_context_for_critique", AsyncMock(return_value=""))
    finished = AsyncMock()
    monkeypatch.setattr(coordinator, "_continue_without_self_review", finished)
    monkeypatch.setattr(coordinator_module.critique_memory, "get_critique_count", AsyncMock(return_value=0))
    monkeypatch.setattr(coordinator_module.critique_memory, "get_all_critiques", AsyncMock(return_value=""))
    monkeypatch.setattr(coordinator_module.paper_memory, "get_paper", AsyncMock(return_value="paper"))
    monkeypatch.setattr(coordinator_module.outline_memory, "get_outline", AsyncMock(return_value="outline"))
    from backend.aggregator.memory.shared_training import shared_training_memory

    monkeypatch.setattr(shared_training_memory, "get_all_content", AsyncMock(return_value=""))
    monkeypatch.setattr(coordinator_module.asyncio, "sleep", AsyncMock())
    emit = AsyncMock()
    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", emit)

    await coordinator._run_critique_aggregation()

    emit.assert_not_called()
    finished.assert_awaited_once()
    warnings = [data for event, data in coordinator.broadcasts if event == "compiler_warning"]
    assert [data["role_id"] for data in warnings] == ["compiler_high_param", "critique_validator"]
    assert all(data["reason"] == "model_output_contract_failure" for data in warnings)
    # Three counted attempts: the contract failures were not attempts.
    assert submit.await_count == 4
    coordinator.critique_submitter.handle_critique_rejection.assert_not_called()


# --------------------------------------------------------------- repeated failures and optional ops


@pytest.mark.asyncio
async def test_repeated_validator_contract_failures_back_off_without_a_cap(monkeypatch):
    coordinator = _coordinator()
    waits = []

    async def emit(payload, **kwargs):
        waits.append((payload, kwargs))
        return True

    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", emit)
    failures = 9
    calls = []

    async def validate():
        calls.append(1)
        if len(calls) <= failures:
            raise ValidatorContractError("Validation JSON contract remained invalid", stage="syntax")
        return _accept_result()

    result = await coordinator._validate_until_decided(validate, mode="review", submission_id="sub-1")

    assert result.decision == "accept"
    assert len(calls) == failures + 1  # never capped
    # The first failure retries at once; later ones wait longer, up to the cap.
    delays = [payload["retry_after_seconds"] for payload, _ in waits]
    assert delays == [1, 2, 4, 8, 16, 30, 30, 30]
    first, kwargs = waits[0]
    assert first["reason"] == "model_output_contract_retry"
    assert first["retry_layer"] == "workflow"
    assert "not a provider error" in first["message"]
    assert kwargs["role_id"] == "compiler_validator"
    # The wait is stop-aware.
    coordinator.is_running = False
    assert kwargs["should_stop"]() is True


@pytest.mark.asyncio
async def test_backoff_is_skipped_once_stopped(monkeypatch):
    coordinator = _coordinator(running=False)
    emit = AsyncMock()
    monkeypatch.setattr(coordinator_module.api_client_manager, "emit_and_wait_for_retry", emit)
    await coordinator._wait_after_repeated_contract_failure("compiler_validator", 5)
    emit.assert_not_called()


@pytest.mark.asyncio
async def test_optional_brainstorm_operation_overflow_is_skipped_not_fatal():
    coordinator = _coordinator()
    error = ValidatorPromptTooLargeError(
        required_tokens=9000, available_tokens=4000, subject="brainstorm operation"
    )
    validate = AsyncMock(side_effect=error)

    result = await coordinator._validate_until_decided(
        validate, mode="brainstorm_retroactive", optional_operation=True
    )

    assert result is None
    assert coordinator.is_running is True
    events = [event for event, _ in coordinator.broadcasts]
    assert events == ["compiler_warning"]
    payload = coordinator.broadcasts[0][1]
    assert payload["reason"] == "optional_operation_context_overflow"
    assert "paper writing continues" in payload["message"]
