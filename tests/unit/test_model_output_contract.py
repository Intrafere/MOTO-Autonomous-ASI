"""Model-output contract errors, error-helper precedence, and normalization activity."""
import asyncio
import json

import pytest

from backend.shared.api_client_manager import api_client_manager
from backend.shared.model_error_utils import (
    is_non_retryable_model_error,
    is_provider_context_length_error,
    is_retryable_model_output_error,
    is_transient_model_call_error,
)
from backend.shared.model_output_contract import (
    MODEL_OUTPUT_NORMALIZED_EVENT,
    ItemContractProblem,
    ModelOutputContractError,
    OutputRepair,
    is_model_output_contract_error,
    report_model_output_normalizations,
)
from backend.shared.models import ModelConfig
from backend.shared.provider_errors import ProviderRouteError, ProviderRouteIdentity


def test_contract_error_is_a_value_error_with_structured_fields():
    error = ModelOutputContractError(
        "The theorem entry did not match the required format",
        role="autonomous_proof_identification_brainstorm",
        stage="missing_key",
        item_path="theorems[2]",
        item_id="lemma_b",
        missing_keys=["novelty_rationale"],
        unknown_keys=["novelty_rationally"],
    )
    assert isinstance(error, ValueError)
    assert error.stage == "missing_key" and error.retryable is True
    assert str(error) == (
        "The theorem entry did not match the required format; theorems[2] 'lemma_b': "
        "missing required field 'novelty_rationale'; unexpected key 'novelty_rationally'."
    )


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        ModelOutputContractError("x", stage="semantic")


def test_model_authored_tokens_are_stripped_quoted_and_capped():
    error = ModelOutputContractError(
        "Bad entry",
        stage="shape",
        unknown_keys=['evil" key: <|channel|> ' + "x" * 200],
    )
    message = str(error)
    token = error.unknown_keys[0]
    assert len(token) == 64
    assert set(token) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    assert f"'{token}'" in message
    assert "<|" not in message and '"' not in message


def test_from_item_problems_names_each_bad_item_and_counts_correctly():
    error = ModelOutputContractError.from_item_problems(
        role="autonomous_proof_identification_brainstorm",
        item_noun_plural="theorem entries",
        total=5,
        problems=[
            ItemContractProblem(item_path="theorems[1]", item_id="lemma_b", missing_keys=("novelty_rationale",)),
            ItemContractProblem(item_path="theorems[3]", invalid_fields=("expected_novelty_tier",)),
        ],
    )
    message = str(error)
    assert message.startswith("2 of 5 theorem entries were invalid;")
    assert "theorems[1] 'lemma_b': missing required field 'novelty_rationale'" in message
    assert "theorems[3]: invalid field 'expected_novelty_tier'" in message
    assert "no valid theorem candidates" not in message
    assert error.field_summary == "novelty_rationale, expected_novelty_tier"


def test_syntax_wrapper_never_includes_raw_output():
    raw = '{"secret_text_from_model": tru}'
    with pytest.raises(json.JSONDecodeError) as caught:
        json.loads(raw)
    error = ModelOutputContractError.syntax(caught.value, role="r")
    assert error.stage == "syntax"
    assert "secret_text_from_model" not in str(error)
    assert "line 1" in str(error)


@pytest.mark.parametrize(
    "token",
    [
        "lemma_request_too_large",  # context-length marker
        "invalid_model_permission_denied",  # non-retryable markers
        "openrouter_connection_failed_bad_gateway",  # transient markers
        "output_truncated_max_output_tokens",  # retryable-output markers
    ],
)
def test_error_helpers_check_the_contract_type_first(token):
    message_tokens = token.replace("_", " ")
    error = ModelOutputContractError(
        f"Bad entry ({message_tokens})",
        item_id=token,
        unknown_keys=[token],
    )
    assert is_non_retryable_model_error(error) is False
    assert is_transient_model_call_error(error) is False
    assert is_provider_context_length_error(error) is False
    assert is_retryable_model_output_error(error) is False


def test_error_raised_from_a_contract_error_is_classified_as_one():
    contract = ModelOutputContractError("request too large invalid model", stage="shape")
    try:
        try:
            raise contract
        except ModelOutputContractError as exc:
            raise RuntimeError("openrouter request too large invalid model") from exc
    except RuntimeError as wrapped:
        assert is_model_output_contract_error(wrapped)
        assert is_provider_context_length_error(wrapped) is False
        assert is_non_retryable_model_error(wrapped) is False


def test_provider_route_error_cause_chain_is_followed():
    contract = ModelOutputContractError("Bad entry", stage="shape")
    wrapped = ProviderRouteError(
        "request too large",
        route=ProviderRouteIdentity(provider="lm_studio", model="gemma-3-27b"),
        cause=contract,
    )
    assert is_model_output_contract_error(wrapped)
    assert is_provider_context_length_error(wrapped) is False


def test_plain_provider_errors_keep_their_classification():
    assert is_provider_context_length_error(ValueError("request too large"))
    assert not is_model_output_contract_error(ValueError("request too large"))


def test_output_repair_payload_and_description():
    repair = OutputRepair(
        kind="key_label",
        item_path="theorems[2]",
        item_id="lemma_b",
        from_key="novelty_rationally",
        to_key="novelty_rationale",
    )
    assert repair.describe("gemma-3") == (
        "Accepted gemma-3's mislabeled field 'novelty_rationally' as "
        "'novelty_rationale' (theorems[2] 'lemma_b')."
    )
    stray = OutputRepair(kind="stray_characters", item_path="theorems[0]", removed_chars=1)
    assert stray.as_payload() == {"kind": "stray_characters", "item_path": "theorems[0]", "removed_chars": 1}
    with pytest.raises(ValueError):
        OutputRepair(kind="value_fix")


def test_report_helper_broadcasts_route_identified_activity(monkeypatch):
    role = "autonomous_proof_identification_brainstorm"
    monkeypatch.setitem(
        api_client_manager._role_model_configs,
        role,
        ModelConfig(provider="lm_studio", model_id="gemma-3-27b"),
    )
    sent = []

    async def capture(event, data):
        sent.append((event, data))

    monkeypatch.setattr(api_client_manager, "_broadcast_callback", capture)
    monkeypatch.setitem(
        api_client_manager._role_activity_sources,
        role,
        {"source_type": "brainstorm", "source_id": "topic_1", "proof_run_id": "run_9"},
    )
    repairs = [
        OutputRepair(
            kind="key_label",
            item_path="theorems[2]",
            item_id="lemma_b",
            from_key="novelty_rationally",
            to_key="novelty_rationale",
        ),
        OutputRepair(kind="stray_characters", item_path="theorems[3]", removed_chars=1),
    ]
    payload = asyncio.run(
        report_model_output_normalizations(role, repairs, task_id="proof_id_007", workflow_mode="autonomous")
    )
    assert len(sent) == 1
    event, data = sent[0]
    assert event == MODEL_OUTPUT_NORMALIZED_EVENT
    assert data is payload
    assert data["role_id"] == role and data["task_id"] == "proof_id_007"
    assert data["workflow_mode"] == "autonomous"
    assert data["model"] == "gemma-3-27b" and data["provider"] == "lm_studio"
    assert data["route_identity"].startswith("[Autonomous")
    assert (data["source_type"], data["source_id"], data["proof_run_id"]) == ("brainstorm", "topic_1", "run_9")
    assert data["repairs"] == [repair.as_payload() for repair in repairs]
    assert "Accepted gemma-3-27b's mislabeled field 'novelty_rationally' as 'novelty_rationale'" in data["message"]
    assert "Ignored 1 stray character outside JSON strings" in data["message"]
    assert data["message"].startswith(data["route_identity"])
    assert data["activity_classification"] == "informational"
    assert data["activity_semantic"] is False
    assert data["activity_outcome"] == "normalized"
    assert data["activity_id"].startswith(f"{MODEL_OUTPUT_NORMALIZED_EVENT}:")

    second = asyncio.run(report_model_output_normalizations(role, repairs[:1]))
    assert second["activity_id"] != data["activity_id"]


def test_report_helper_is_silent_without_repairs_and_survives_broadcast_failure(monkeypatch):
    async def explode(event, data):
        raise RuntimeError("socket gone")

    monkeypatch.setattr(api_client_manager, "_broadcast_callback", explode)
    assert asyncio.run(report_model_output_normalizations("r", [])) is None
    assert asyncio.run(
        report_model_output_normalizations("r", [OutputRepair(kind="trailing_comma")])
    ) is None
