import json

import pytest

from backend.compiler.validation import compiler_validator as validator_module
from backend.compiler.validation.compiler_validator import (
    CompilerValidator,
    ValidatorContractError,
)
from backend.shared.config import system_config
from backend.shared.models import CompilerSubmission


def _valid_response(**overrides):
    data = {
        "decision": "accept",
        "reasoning": "The submission satisfies the applicable criteria.",
        "coherence_check": True,
        "rigor_check": True,
        "placement_check": True,
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "payload",
    [
        {"reasoning": "Missing decision", "coherence_check": True, "rigor_check": True, "placement_check": True},
        _valid_response(decision="maybe"),
        _valid_response(reasoning=" "),
        _valid_response(coherence_check=1),
        _valid_response(rigor_check="true"),
        _valid_response(placement_check=None),
    ],
)
def test_primary_validation_contract_rejects_invalid_shapes(payload) -> None:
    with pytest.raises(ValidatorContractError):
        CompilerValidator._validate_primary_validation_contract(
            payload,
            require_checks=True,
        )


def test_optional_solution_path_extension_does_not_invalidate_primary_result() -> None:
    payload = _valid_response(solution_path_update="malformed optional extension")

    parsed = CompilerValidator._validate_primary_validation_contract(
        payload,
        require_checks=True,
    )

    assert parsed["decision"] == "accept"
    assert parsed["solution_path_update"] == "malformed optional extension"


@pytest.mark.asyncio
async def test_malformed_accept_prose_cannot_be_accepted_after_bounded_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Solve it.")
    calls = []
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 8000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 1000)

    async def fake_generate_completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {"message": {"content": "decision: accept because it looks valid"}}
            ]
        }

    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        fake_generate_completion,
    )

    with pytest.raises(ValueError):
        await validator._parse_json_with_retry(
            "decision: accept",
            "ORIGINAL PROMPT",
            "",
        )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retry_uses_sanitized_failed_output_and_succeeds_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Solve it.")
    captured = {}
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 8000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 1000)

    async def fake_generate_completion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {"message": {"content": json.dumps(_valid_response())}}
            ]
        }

    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        fake_generate_completion,
    )

    parsed = await validator._parse_json_with_retry(
        "<think>private reasoning</think> decision: accept",
        "ORIGINAL PROMPT",
        "",
    )

    assert parsed["decision"] == "accept"
    messages = captured["messages"]
    assert len(messages) == 3
    assert "<think>" not in messages[1]["content"]
    assert "private reasoning" not in messages[1]["content"]


@pytest.mark.asyncio
async def test_retry_provider_value_error_is_not_misclassified_as_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Solve it.")
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 8000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 1000)

    async def fake_generate_completion(**kwargs):
        raise ValueError("provider configuration is invalid")

    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        fake_generate_completion,
    )

    with pytest.raises(ValueError, match="provider configuration"):
        await validator._parse_json_with_retry(
            "decision: accept",
            "ORIGINAL PROMPT",
            "",
        )


@pytest.mark.asyncio
async def test_brainstorm_contract_requires_decision_and_reasoning_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Solve it.")

    parsed = await validator._parse_json_with_retry(
        json.dumps({"decision": "reject", "reasoning": "The operation is unsupported."}),
        "ORIGINAL PROMPT",
        "",
        require_checks=False,
    )

    assert parsed == {
        "decision": "reject",
        "reasoning": "The operation is unsupported.",
    }


@pytest.mark.asyncio
async def test_lean_placement_does_not_force_rigor_before_structure_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Write it.")
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 8000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 1000)

    async def fake_ensure_markers_intact():
        return False

    responses = iter(
        [
            {"choices": [{"message": {"content": json.dumps({
                "decision": "accept",
                "reasoning": "Looks placed.",
                "coherence_check": True,
                "rigor_check": "false",
                "placement_check": True,
            })}}]},
            {"choices": [{"message": {"content": "still malformed"}}]},
        ]
    )

    async def fake_generate_completion(**kwargs):
        return next(responses)

    monkeypatch.setattr(
        validator_module.paper_memory,
        "ensure_markers_intact",
        fake_ensure_markers_intact,
    )
    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        fake_generate_completion,
    )

    submission = CompilerSubmission(
        submission_id="lean-invalid-structure",
        mode="rigor",
        operation="insert_after",
        old_string="Anchor.",
        new_string="Verified theorem.",
        content="Verified theorem.",
        reasoning="Place it.",
        metadata={"rigor_mode": "lean_placement"},
    )
    result = await validator.validate_submission(
        submission,
        current_paper="Anchor.",
        current_outline="I. Introduction\nII. Body\nIII. Conclusion",
    )

    assert result.decision == "reject"
    assert result.json_valid is False
    assert result.rigor_check is False
    assert result.summary == "Invalid validator JSON contract"


def test_full_content_is_rejected_for_non_empty_paper() -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Write it.")
    submission = CompilerSubmission(
        submission_id="destructive-full-content",
        mode="construction",
        operation="full_content",
        old_string="",
        new_string="Replacement paper.",
        content="Replacement paper.",
        reasoning="Replace everything.",
    )

    result = validator._pre_validate_exact_string_match(
        submission,
        current_paper="Existing paper content.",
        current_outline="I. Introduction\nII. Body\nIII. Conclusion",
    )

    assert result is not None
    assert result.decision == "reject"
    assert "NON_EMPTY_FULL_CONTENT_ERROR" in result.reasoning
    assert result.placement_check is False


def test_full_content_remains_valid_for_empty_paper() -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Write it.")
    submission = CompilerSubmission(
        submission_id="initial-full-content",
        mode="construction",
        operation="full_content",
        old_string="",
        new_string="Initial body.",
        content="Initial body.",
        reasoning="Start the paper.",
    )

    result = validator._pre_validate_exact_string_match(
        submission,
        current_paper="",
        current_outline="I. Introduction\nII. Body\nIII. Conclusion",
    )

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_field", ["coherence_check", "rigor_check", "placement_check"])
async def test_acceptance_with_failed_check_is_converted_to_rejection(
    monkeypatch: pytest.MonkeyPatch,
    failed_field: str,
) -> None:
    validator = CompilerValidator(model_name="test-model", user_prompt="Write it.")
    monkeypatch.setattr(system_config, "compiler_validator_context_window", 8000)
    monkeypatch.setattr(system_config, "compiler_validator_max_output_tokens", 1000)

    async def fake_ensure_markers_intact():
        return False

    payload = _valid_response(**{failed_field: False})

    async def fake_generate_completion(**kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    monkeypatch.setattr(
        validator_module.paper_memory,
        "ensure_markers_intact",
        fake_ensure_markers_intact,
    )
    monkeypatch.setattr(
        validator_module.api_client_manager,
        "generate_completion",
        fake_generate_completion,
    )

    submission = CompilerSubmission(
        submission_id=f"inconsistent-{failed_field}",
        mode="construction",
        operation="insert_after",
        old_string="Anchor.",
        new_string="New material.",
        content="New material.",
        reasoning="Add material.",
    )
    result = await validator.validate_submission(
        submission,
        current_paper="Anchor.",
        current_outline="I. Introduction\nII. Body\nIII. Conclusion",
    )

    assert result.decision == "reject"
    assert failed_field in result.reasoning
    assert result.json_valid is True
