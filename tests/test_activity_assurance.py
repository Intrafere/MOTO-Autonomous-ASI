import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.api.routes.websocket import ConnectionManager
from backend.compiler.core.compiler_coordinator import CompilerCoordinator
from backend.shared.activity import fill_missing_activity_metadata, normalize_activity_payload


def test_activity_envelope_is_additive_bounded_and_secret_safe():
    reasoning = "Useful validator feedback " * 1000
    payload = {
        "submission_id": "sub-1",
        "reasoning": reasoning,
        "nested": {
            "authorization": "Bearer private-token",
            "message": "api_key=private-value",
        },
        "content": "preserve this legacy field",
    }

    normalized = normalize_activity_payload("submission_rejected", payload)

    assert normalized["submission_id"] == "sub-1"
    assert normalized["content"] == "preserve this legacy field"
    assert normalized["reasoning"].endswith("...[truncated]")
    assert len(normalized["reasoning"]) <= 8_013
    assert normalized["nested"]["authorization"] == "[redacted]"
    assert normalized["nested"]["message"] == "api_key=[redacted]"
    assert normalized["activity_schema_version"] == 1
    assert normalized["activity_category"] == "validator"
    assert normalized["activity_outcome"] == "rejected"
    assert normalized["activity_semantic"] is True
    assert normalized["activity_feedback"].endswith("...[truncated]")
    assert len(normalized["activity_feedback"]) <= 8_013


def test_activity_redacts_installation_identifiers_but_keeps_occurrence_ids():
    hex_id = "0123456789abcdef0123456789abcdef"
    normalized = normalize_activity_payload(
        "assistant_proof_memory_pack",
        {
            "message": f"installation moto_installation_{hex_id} root moto_root_{hex_id}",
            "search_id": f"moto:moto_proof_{hex_id}",
        },
    )

    assert hex_id not in normalized["message"]
    assert normalized["search_id"] == f"moto:moto_proof_{hex_id}"


def test_operational_validation_failure_is_not_semantic_rejection():
    normalized = normalize_activity_payload(
        "compiler_rejection",
        {
            "decision": "reject",
            "validation_stage": "provider_error",
            "reasoning": "Provider unavailable",
        },
    )

    assert normalized["activity_outcome"] == "rejected"
    assert normalized["activity_semantic"] is False
    assert normalized["activity_category"] == "provider"


def test_pre_validation_failure_is_not_semantic_rejection():
    normalized = normalize_activity_payload(
        "compiler_rejection",
        {
            "decision": "reject",
            "validation_stage": "pre-validation",
            "reasoning": "Exact string match failed",
        },
    )

    assert normalized["activity_outcome"] == "rejected"
    assert normalized["activity_semantic"] is False
    assert normalized["activity_category"] == "workflow"


@pytest.mark.asyncio
async def test_compiler_broadcast_marks_pre_validation_rejection_operational():
    coordinator = CompilerCoordinator()
    events = []

    async def capture(event_type, payload):
        events.append((event_type, normalize_activity_payload(event_type, payload)))

    coordinator.set_websocket_broadcaster(capture)
    await coordinator._broadcast(
        "compiler_rejection",
        {
            "mode": "review",
            "submission_id": "review-1",
            "reasoning": "Exact string match failed",
            "validation_stage": "pre-validation",
        },
    )

    assert events[0][1]["activity_classification"] == "operational"
    assert events[0][1]["activity_semantic"] is False
    assert events[0][1]["activity_category"] == "workflow"
    assert events[0][1]["activity_outcome"] == "error"


@pytest.mark.asyncio
async def test_compiler_critique_validator_contract_failure_is_operational(monkeypatch):
    coordinator = CompilerCoordinator()
    coordinator.user_prompt = "prompt"
    coordinator.paper_title = "paper"
    coordinator.validator_model = "validator-model"
    coordinator.validator_max_tokens = 100
    coordinator.critique_submitter = SimpleNamespace(
        next_critique_task_id=lambda _mode: "critique_val_001"
    )
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.paper_memory.get_paper",
        AsyncMock(return_value="paper"),
    )
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.outline_memory.get_outline",
        AsyncMock(return_value="outline"),
    )
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.critique_memory.get_all_critiques",
        AsyncMock(return_value=""),
    )
    monkeypatch.setattr(
        "backend.compiler.core.compiler_coordinator.api_client_manager.generate_completion",
        AsyncMock(
            side_effect=[
                {"choices": [{"message": {"content": "not json"}}]},
                {"choices": [{"message": {"content": "still not json"}}]},
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="Critique validation failed"):
        await coordinator._validate_critique(
            SimpleNamespace(submission_id="critique-1", content="critique")
        )


def test_fill_missing_activity_metadata_preserves_explicit_and_legacy_fields():
    payload = {
        "decision": "reject",
        "validator_feedback": "Item-specific feedback.",
        "activity_workflow": "explicit-owner",
        "activity_classification": "semantic",
    }

    filled = fill_missing_activity_metadata(
        payload,
        workflow="fallback-owner",
        run_id="run-1",
        classification="operational",
        feedback="fallback feedback",
    )

    assert filled["decision"] == "reject"
    assert filled["validator_feedback"] == "Item-specific feedback."
    assert filled["activity_workflow"] == "explicit-owner"
    assert filled["activity_classification"] == "semantic"
    assert filled["activity_run_id"] == "run-1"
    assert filled["activity_feedback"] == "fallback feedback"
    assert payload.get("activity_run_id") is None


def test_activity_identity_is_deterministic_and_retry_sensitive():
    base = {
        "run_id": "run-1",
        "source_id": "source-1",
        "retry_attempt": 1,
    }
    first = normalize_activity_payload("provider_retry", base)
    repeated = normalize_activity_payload("provider_retry", dict(base))
    next_attempt = normalize_activity_payload(
        "provider_retry", {**base, "retry_attempt": 2}
    )

    assert first["activity_id"] == repeated["activity_id"]
    assert first["activity_id"] != next_attempt["activity_id"]
    assert first["activity_correlation_id"] == repeated["activity_correlation_id"]


def test_activity_identity_distinguishes_proof_subjects_and_attempts():
    base = {
        "run_id": "proof-run-1",
        "source_id": "paper-1",
    }
    first = normalize_activity_payload(
        "proof_attempt_failed",
        {**base, "theorem_id": "thm-a", "attempt": 1},
    )
    second_attempt = normalize_activity_payload(
        "proof_attempt_failed",
        {**base, "theorem_id": "thm-a", "attempt": 2},
    )
    second_theorem = normalize_activity_payload(
        "proof_attempt_failed",
        {**base, "theorem_id": "thm-b", "attempt": 1},
    )

    assert first["activity_id"] != second_attempt["activity_id"]
    assert first["activity_id"] != second_theorem["activity_id"]


def test_activity_workflow_prefers_workflow_mode_over_compiler_mode():
    normalized = normalize_activity_payload(
        "provider_retry_cooldown",
        {"mode": "construction", "workflow_mode": "compiler", "role_id": "compiler_writer"},
    )

    assert normalized["activity_workflow"] == "compiler"
    assert normalized["mode"] == "construction"


def test_provider_error_activity_identity_distinguishes_models_and_hosts():
    base = {
        "workflow_mode": "autonomous",
        "run_id": "run-1",
        "role_id": "aggregator_submitter_2",
        "provider": "openrouter",
        "model": "vendor/test-free:free",
        "configured_host_provider": "",
        "effective_host_provider": "TestHost",
        "retry_attempt": 1,
    }
    first = normalize_activity_payload("provider_retry_cooldown", base)
    repeated = normalize_activity_payload("provider_retry_cooldown", dict(base))
    other_model = normalize_activity_payload(
        "provider_retry_cooldown", {**base, "model": "vendor/other-free:free"}
    )
    other_host = normalize_activity_payload(
        "provider_retry_cooldown", {**base, "effective_host_provider": "OtherHost"}
    )
    other_pinned_host = normalize_activity_payload(
        "provider_retry_cooldown", {**base, "configured_host_provider": "TestHost"}
    )

    assert first["activity_id"] == repeated["activity_id"]
    assert len({
        first["activity_id"],
        other_model["activity_id"],
        other_host["activity_id"],
        other_pinned_host["activity_id"],
    }) == 4
    generic = normalize_activity_payload("submission_rejected", base)
    assert {key for key in first if key.startswith("activity_")} == {
        key for key in generic if key.startswith("activity_")
    }


def test_non_provider_activity_identity_ignores_model_and_host():
    base = {
        "run_id": "run-1",
        "submission_id": "sub-1",
        "model": "vendor/test-free:free",
        "effective_host_provider": "TestHost",
    }
    first = normalize_activity_payload("submission_rejected", base)
    other_model = normalize_activity_payload(
        "submission_rejected",
        {**base, "model": "vendor/other-free:free", "effective_host_provider": "OtherHost"},
    )
    without_model = normalize_activity_payload(
        "submission_rejected",
        {"run_id": "run-1", "submission_id": "sub-1"},
    )

    assert first["activity_id"] == other_model["activity_id"]
    assert first["activity_id"] == without_model["activity_id"]


def test_provider_error_activity_identity_unchanged_without_route_fields():
    payload = {"run_id": "run-1", "role_id": "aggregator_submitter_2", "retry_attempt": 1}

    provider_event = normalize_activity_payload("provider_retry_cooldown", payload)
    blank_route = normalize_activity_payload(
        "provider_retry_cooldown",
        {**payload, "model": "", "provider": None, "configured_host_provider": ""},
    )

    assert provider_event["activity_id"] == blank_route["activity_id"]


def test_provider_error_activity_identity_distinguishes_calls_of_one_route():
    hung = {
        "workflow_mode": "autonomous",
        "role_id": "aggregator_submitter_2",
        "provider": "openrouter",
        "model": "vendor/test-free:free",
        "task_id": "agg_sub2_001",
        "elapsed_minutes": 25,
        "message": "The model may still be thinking.",
    }
    first_hung = normalize_activity_payload("hung_connection_alert", hung)
    redelivered_hung = normalize_activity_payload("hung_connection_alert", dict(hung))
    later_hung = normalize_activity_payload(
        "hung_connection_alert", {**hung, "task_id": "agg_sub2_009"}
    )

    leanoj = {
        "workflow_mode": "leanoj",
        "role_id": "leanoj_final_solver",
        "provider": "openrouter",
        "model": "vendor/test-free:free",
        "task_id": "leanoj_sufficiency",
        "attempt": 1,
        "message": "err A",
    }
    first_failure = normalize_activity_payload("leanoj_model_call_failed", leanoj)
    redelivered_failure = normalize_activity_payload("leanoj_model_call_failed", dict(leanoj))
    other_task = normalize_activity_payload(
        "leanoj_model_call_failed", {**leanoj, "task_id": "leanoj_path"}
    )
    other_error = normalize_activity_payload(
        "leanoj_model_call_failed", {**leanoj, "message": "err B"}
    )

    assert first_hung["activity_id"] == redelivered_hung["activity_id"]
    assert first_hung["activity_id"] != later_hung["activity_id"]
    assert first_failure["activity_id"] == redelivered_failure["activity_id"]
    assert len({
        first_failure["activity_id"],
        other_task["activity_id"],
        other_error["activity_id"],
    }) == 3
    # Per-call discriminators shape only activity_id, never the correlation digest.
    assert first_hung["activity_correlation_id"] == later_hung["activity_correlation_id"]
    assert first_failure["activity_correlation_id"] == other_error["activity_correlation_id"]
    # A producer-supplied event identity still wins.
    explicit = normalize_activity_payload(
        "hung_connection_alert", {**hung, "event_id": "hung_connection_alert:explicit"}
    )
    assert explicit["activity_id"] == "hung_connection_alert:explicit"


def test_non_provider_activity_identity_ignores_call_discriminators():
    base = {"run_id": "run-1", "submission_id": "sub-1", "task_id": "agg_sub1_001", "message": "a"}

    first = normalize_activity_payload("submission_rejected", base)
    other = normalize_activity_payload(
        "submission_rejected", {**base, "task_id": "agg_sub1_002", "message": "b"}
    )

    assert first["activity_id"] == other["activity_id"]


@pytest.mark.asyncio
async def test_websocket_boundary_preserves_outer_shape_and_normalizes_dict_payload():
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send_text(self, message):
            self.sent.append(message)

    manager = ConnectionManager()
    connection = FakeWebSocket()
    manager.active_connections.append(connection)

    await manager.broadcast(
        "submission_accepted",
        {"submission_id": "sub-1", "validator_feedback": "Looks correct."},
    )

    message = json.loads(connection.sent[0])
    assert set(message) == {"type", "data", "timestamp"}
    assert message["type"] == "submission_accepted"
    assert message["data"]["submission_id"] == "sub-1"
    assert message["data"]["activity_outcome"] == "success"
    assert message["data"]["activity_feedback"] == "Looks correct."
